#!/usr/bin/env python3
"""
DronJezdzacyAutko — Wersja ulepszona V2.1 (Poprawka wysyłania detekcji)
─────────────────────────────────────────────────────────────────
Architektura wielowątkowa (Zoptymalizowana pod RPi5):
  Wątek 1 (capture)  : Picamera2 → Bufor klatek raw @ 60 FPS
  Wątek 2 (inferencja): YOLO + QReader działające asynchronicznie (sterowane flagami)
  Wątek 3 (ROS spin) : Obsługa jazdy, chwytaka (nieblokująca) i FPV @ 30 Hz + Parowanie detekcji
"""

import cv2
import time
import threading
import queue
import numpy as _np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

from libcamera import Transform
from picamera2 import Picamera2
from ultralytics import YOLO
from qreader import QReader

from dualtech_msgs.msg import Detection
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage, NavSatFix
from std_msgs.msg import Int8, Bool

# ─── KONFIGURACJA SPRZĘTOWO-LOGICZNA ──────────────────────────────────────────
MODEL_PATH   = '/home/pi/ros2_jazzy/src/dualtech-ros-main/dualtech/dualtech/my_model4/my_model.pt'

CAM_WIDTH    = 640
CAM_HEIGHT   = 480
CAM_FPS      = 60      # Maksymalny klatkaż pobierania dla płynności

FPV_FPS      = 30      # Hz publikacji FPV na ROS
JPEG_QUALITY = 55      # Jakość obrazu dla kompresji

YOLO_IMGSZ   = 320     # Rozmiar wejściowy sieci
YOLO_FPS_MAX = 10      # Ograniczenie inferencji dla stabilności CPU
CONFIDENCE   = 0.5

QR_WIDTH     = 320
QR_HEIGHT    = 240

DETECTION_COOLDOWN = 3.0
CACHE_EXPIRY_TIME  = 2.0  # Czas (w sekundach) przez jaki pamiętamy detekcję/QR do parowania

from gpiozero import DigitalOutputDevice
from dualtech.gpiostepper import Stepper
# ─────────────────────────────────────────────────────────────────────────────


class DronJezdzacyAutko(Node):

    def __init__(self):
        super().__init__('dron_jezdzacy_autko')

        # --- Grupy wykonawcze (Callback Groups) dla bezpiecznej wielowątkowości ---
        self.callback_group_jazda = MutuallyExclusiveCallbackGroup()
        self.callback_group_chwytak = MutuallyExclusiveCallbackGroup()
        self.callback_group_default = ReentrantCallbackGroup()

        # --- Słownik mapowania etykiet YOLO na własne nazwy ---
        self.etykiety = {
            'autobus': 'autobus',
            'czerwone auto': 'maluch',
            'brazowe auto': 'polonez',
            'zielone auto': 'ferrari',
            'ciezarowka': 'tir',
            'ciezarowka wojskowa z wyrzutnia': 'pansir',
            'samochod wojskowy': 'humvee',
            'duzy czolg': 'T-90',
            'maly czolg': 'T-62',
            'czlog z antena': 'radar'
        }

        # --- Profile QoS ---
        reliable_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        image_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        # --- Publishery ---
        self.publisher_detection = self.create_publisher(Detection, 'detection', reliable_qos)
        self.image_pub = self.create_publisher(CompressedImage, 'camera/fpv_annotated/compressed', image_qos)

        # --- Subskrypcje ---
        self.create_subscription(Twist, '/cmd_vel', self.callback_jazda, 10, callback_group=self.callback_group_jazda)
        self.create_subscription(Int8, '/chwytak_sterowanie', self.callback_chwytak, 10, callback_group=self.callback_group_chwytak)
        self.create_subscription(Bool, '/scan_qr', self.callback_qr, 10, callback_group=self.callback_group_default)
        self.create_subscription(Bool, '/analiza', self.callback_yolo, 10, callback_group=self.callback_group_default)

        # ---- Flagi stanu (Modyfikowane przez ROS, czytane przez wątek inferencji) ----
        self._state_lock = threading.Lock()
        self.scan_qr = False   
        self.analiza = False   

        # --- Dane współdzielone obrazu ---
        self._raw_frame = None
        self._raw_lock = threading.Lock()
        
        self._yolo_result = None  # Przechowuje: (annotated_frame, detections_list, qr_text)
        self._yolo_lock = threading.Lock()

        # --- Nowy asynchroniczny bufor pamięci podręcznej (Cache) dla niezawodnego parowania ---
        self._cache_lock = threading.Lock()
        self._cached_qr = None
        self._cached_qr_time = 0.0
        self._cached_dets = []
        self._cached_dets_time = 0.0

        self._last_det_time = {}
        self._stop = threading.Event()

        # --- Kamera 640x480 @ 60 FPS ---
        self.get_logger().info('Inicjalizacja kamery...')
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"format": "RGB888", "size": (CAM_WIDTH, CAM_HEIGHT)},
            controls={'FrameRate': float(CAM_FPS), 'AfMode': 2, 'AfSpeed': 1, 'NoiseReductionMode': 0},
            transform=Transform(hflip=0, vflip=0),
            buffer_count=4
        )
        self.picam2.configure(config)
        self.picam2.start()

        # --- Model YOLO ---
        self.get_logger().info('Ładowanie sieci YOLO...')
        self.model = YOLO(MODEL_PATH)
        # Rozgrzanie sieci pustą macierzą
        self.model(_np.zeros((QR_HEIGHT, QR_WIDTH, 3), dtype='uint8'), imgsz=YOLO_IMGSZ, verbose=False)
        
        # --- QReader dla kodów QR ---
        self.qreader = QReader()
        self.get_logger().info('Detektory wizyjne gotowe.')

        # --- Inicjalizacja sprzętu (Silniki i Chwytak) ---
        self.setup_motors()
        self.chwytak_setup()

        # --- Uruchomienie wątków pobierania i przetwarzania ---
        self._t_capture = threading.Thread(target=self._capture_loop, daemon=True, name='capture')
        self._t_inferencja = threading.Thread(target=self._inference_loop, daemon=True, name='inference')
        self._t_capture.start()
        self._t_inferencja.start()

        # Timer głównej pętli publikacji wizji FPV (~30 FPS)
        self.create_timer(1.0 / FPV_FPS, self.process_and_publish_fpv, callback_group=self.callback_group_default)
        self.get_logger().info('Węzeł sterująco-detekcyjny uruchomiony w pełni asynchronicznie.')

    # ------------------------------------------------------------------
    # WĄTEK 1: Capture Loop (Zrzucanie klatek z sensora do pamięci RAM)
    # ------------------------------------------------------------------
    def _capture_loop(self):
        while not self._stop.is_set():
            frame = self.picam2.capture_array()
            if frame is not None:
                with self._raw_lock:
                    self._raw_frame = frame

    # ------------------------------------------------------------------
    # WĄTEK 2: Inference Loop (Ciężka analityka: YOLO + QR w tle)
    # ------------------------------------------------------------------
    def _inference_loop(self):
        interval = 1.0 / YOLO_FPS_MAX
        while not self._stop.is_set():
            t0 = time.time()

            # Pobranie flag stanu
            with self._state_lock:
                run_yolo = self.analiza
                run_qr = self.scan_qr

            # Jeśli analityka jest całkowicie wyłączona, śpimy i oszczędzamy zasoby
            if not run_yolo and not run_qr:
                with self._yolo_lock:
                    self._yolo_result = None
                time.sleep(0.1)
                continue

            with self._raw_lock:
                frame = self._raw_frame
            if frame is None:
                time.sleep(0.01)
                continue

            detections = []
            annotated = None
            qr_text = None

            # Egzekucja sieci YOLO (jeśli włączona)
            if run_yolo:
                # Wymuszenie conf=CONFIDENCE bezpośrednio w modelu filtruje podgląd OSD z błędnych ramek
                results = self.model(frame, imgsz=YOLO_IMGSZ, conf=CONFIDENCE, verbose=False)[0]
                annotated = results.plot(kpt_radius=0, kpt_line=False)
                for box in results.boxes:
                    conf = float(box.conf[0])
                    raw_cls_name = self.model.names[int(box.cls[0])]
                    # Mapowanie nazwy z YOLO na etykietę ze słownika
                    mapped_name = self.etykiety.get(raw_cls_name, raw_cls_name)
                    detections.append((mapped_name, conf, box))
                
                # Zapisz udane wykrycia do asynchronicznego cache
                if detections:
                    with self._cache_lock:
                        self._cached_dets = detections
                        self._cached_dets_time = time.time()

            # Egzekucja skanera QR (jeśli włączony)
            if run_qr:
                small = cv2.resize(frame, (QR_WIDTH, QR_HEIGHT))
                texts = self.qreader.detect_and_decode(image=small)
                valid = [t for t in texts if t is not None]
                if valid:
                    qr_text = valid[0]
                    # Zapisz poprawny odczyt kodu QR do asynchronicznego cache
                    with self._cache_lock:
                        self._cached_qr = qr_text
                        self._cached_qr_time = time.time()

            # Zapisanie bieżących wyników analitycznych dla wątku FPV
            with self._yolo_lock:
                self._yolo_result = (annotated, detections, qr_text)

            # Stabilizacja klatkażu pętli wizyjnej
            sleep_t = interval - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    # ------------------------------------------------------------------
    # WĄTEK 3 (Timer): Przetwarzanie graficzne OSD, kompresja i publikacja
    # ------------------------------------------------------------------
    def process_and_publish_fpv(self):
        with self._raw_lock:
            raw = self._raw_frame
        if raw is None:
            return

        with self._yolo_lock:
            yolo_res = self._yolo_result

        # Czytanie stanów flag do wyświetlania OSD
        with self._state_lock:
            flag_yolo_active = self.analiza
            flag_qr_active = self.scan_qr

        # Dobór bazy pod obraz wyjściowy
        if flag_yolo_active and yolo_res is not None and yolo_res[0] is not None:
            display = yolo_res[0].copy()
        else:
            display = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)

        # Ramka OSD z informacjami o statusie systemów
        cv2.rectangle(display, (5, 5), (220, 55), (0, 0, 0), -1)
        cv2.putText(display, f"YOLO: {'ON' if flag_yolo_active else 'OFF'}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if flag_yolo_active else (100, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(display, f"QR SCAN: {'ON' if flag_qr_active else 'OFF'}", (10, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255) if flag_qr_active else (100, 100, 100), 1, cv2.LINE_AA)

        # Pobieranie danych z pamięci podręcznej (Cache) do OSD i publikacji
        now = time.time()
        with self._cache_lock:
            # Sprawdzenie świeżości kodu QR w buforze
            active_qr = self._cached_qr if (now - self._cached_qr_time <= CACHE_EXPIRY_TIME) else None
            # Sprawdzenie świeżości wykrycia pojazdu w buforze
            active_dets = self._cached_dets if (now - self._cached_dets_time <= CACHE_EXPIRY_TIME) else []

        # Wypisywanie stabilnego tekstu QR na ekranie (dzięki cache nie mruga)
        if flag_qr_active and active_qr:
            cv2.rectangle(display, (5, CAM_HEIGHT - 30), (CAM_WIDTH - 5, CAM_HEIGHT - 5), (0, 0, 0), -1)
            cv2.putText(display, f'QR CODE: {active_qr}', (10, CAM_HEIGHT - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

        # Kompresja klatki OSD do formatu JPEG
        success, jpg = cv2.imencode('.jpg', display, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not success:
            return

        # Generowanie wiadomości obrazu ros
        img_msg = CompressedImage()
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = 'camera'
        img_msg.format = 'jpeg'
        img_msg.data = jpg.tobytes()
        self.image_pub.publish(img_msg)

        # Publikacja /detection — Korzysta ze stabilnych, sklejonych asynchronicznie danych z cache
        if flag_yolo_active and flag_qr_active and active_dets and active_qr:
            best_cls, best_conf, _ = max(active_dets, key=lambda x: x[1])
            key = f'{best_cls}|{active_qr}'

            if now - self._last_det_time.get(key, 0.0) >= DETECTION_COOLDOWN:
                self._last_det_time[key] = now

                det_msg = Detection()
                det_msg.object_type = best_cls
                det_msg.qr_value = str(active_qr)
                det_msg.object_image = img_msg
                det_msg.gps_location = NavSatFix()  # Pusta struktura gotowa do spięcia z GPS
                
                self.publisher_detection.publish(det_msg)
                self.get_logger().info(f'[DETEKCJA PAROWANA] Obiekt: {best_cls} ({best_conf:.2f}) | QR: "{active_qr}"')

    # ------------------------------------------------------------------
    # Sprzęt i Silniki
    # ------------------------------------------------------------------
    def setup_motors(self):
        self.m1_in1 = DigitalOutputDevice(17)
        self.m1_in2 = DigitalOutputDevice(27)
        self.m2_in1 = DigitalOutputDevice(24)
        self.m2_in2 = DigitalOutputDevice(23)
        self.speed_limit = 1.0
        self.get_logger().info('Silniki napędowe gotowe.')

    def chwytak_setup(self):
        self.step_poz = Stepper(
            motor_pins=[0, 5, 6, 26], number_of_steps=32,
            step_sequence=[[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0],
                           [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [1, 0, 0, 1]]
        )
        self.step_pion = Stepper(
            motor_pins=[12, 16, 20, 21], number_of_steps=32,
            step_sequence=[[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0],
                           [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [1, 0, 0, 1]]
        )
        self.get_logger().info('Chwytak krokowy gotowy.')

    def wylacz_cewki_chwytaka(self):
        """ Odcina zasilanie od cewek, zapobiegając przegrzewaniu się silników """
        for pin in self.step_poz.motor_pins:
            pin.off()
        for pin in self.step_pion.motor_pins:
            pin.off()

    # ------------------------------------------------------------------
    # Callbacki ROS (Obsługiwane przez pulę wątków Executora)
    # ------------------------------------------------------------------
    def callback_qr(self, msg):
        with self._state_lock:
            self.scan_qr = msg.data
        self.get_logger().info(f'Przełącznik QR: {"WŁĄCZONY" if msg.data else "WYŁĄCZONY"}')

    def callback_yolo(self, msg):
        with self._state_lock:
            self.analiza = msg.data
        self.get_logger().info(f'Przełącznik YOLO: {"WŁĄCZONY" if msg.data else "WYŁĄCZONY"}')

    def callback_jazda(self, msg):
        m1 = (msg.linear.x - msg.angular.z) * self.speed_limit
        m2 = (msg.linear.x + msg.angular.z) * self.speed_limit
        self.set_motor(self.m1_in1, self.m1_in2, m1)
        self.set_motor(self.m2_in1, self.m2_in2, m2)

    def set_motor(self, in1, in2, duty):
        duty = max(min(duty, 1.0), -1.0)
        if duty > 0:
            in1.on(); in2.off()
        elif duty < 0:
            in1.off(); in2.on()
        else:
            in1.off(); in2.off()

    def callback_chwytak(self, msg):
        kierunek = msg.data
        try:
            if kierunek == 3:
                self.step_poz.step(400)
            elif kierunek == 4:
                self.step_poz.step(-400)
            elif kierunek == 1:
                self.step_pion.step(-800)
            elif kierunek == 2:
                self.step_pion.step(800)
        finally:
            # Gwarancja natychmiastowego odcięcia prądu po wykonaniu kroków ruchu
            self.wylacz_cewki_chwytaka()

    def destroy_node(self):
        self._stop.set()
        self._t_capture.join(timeout=2.0)
        self._t_inferencja.join(timeout=2.0)
        self.picam2.stop()
        self.wylacz_cewki_chwytaka()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DronJezdzacyAutko()
    
    # Przełączamy cały węzeł na obsługę przez 4 wątki robocze
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
