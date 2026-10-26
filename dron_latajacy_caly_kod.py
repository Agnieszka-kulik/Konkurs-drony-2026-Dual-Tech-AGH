#!/usr/bin/env python3
"""
Dron Latający — Mission Node (Wersja ulepszona V2.1)
===================================================
Architektura identyczna jak DronJezdzacyAutko:
  • Wątki tła: QR worker, YOLO worker, MAVLink worker
  • Zoptymalizowany asynchroniczny bufor pamięci podręcznej (Cache) do parowania
  • Filtrowanie spamu detekcji (Cooldown)
  • Picamera2 jako źródło obrazu
  • Custom Detection msg (dualtech_msgs) ze spiętą telemetrią GPS

Topiki OUT:
  detection                          — dualtech_msgs/Detection
  camera/fpv_annotated/compressed    — sensor_msgs/CompressedImage
  /drone/gps                         — sensor_msgs/NavSatFix
  /drone/baro_altitude               — std_msgs/Float32
  /drone/gps_time                    — std_msgs/String  (ISO 8601 UTC)
  /drone/attitude                    — geometry_msgs/Vector3Stamped
  /drone/status                      — std_msgs/String  (JSON)

Topiki IN:
  /scan_qr   — std_msgs/Bool   (włącz/wyłącz QR)
  /analiza   — std_msgs/Bool   (włącz/wyłącz YOLO)
"""

# ════════════════════════════════════════════════════════════════
#  KONFIGURACJA
# ════════════════════════════════════════════════════════════════
MAVLINK_PORT  = '/dev/ttyAMA0'
MAVLINK_BAUD  = 921600
MAVLINK_HZ    = 10       # Hz żądanych strumieni z FC

CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480

YOLO_MODEL_PATH = (    "/home/pi/ros2_jazzy/src/dualtech-ros-main/dualtech/dualtech/my_model5/my_model.pt")
#YOLO_MODEL_PATH = ("/home/pi/ros2_jazzy/src/dualtech-ros-main/dualtech/dualtech/model_ania/my_model.pt")
YOLO_IMGSZ = 320
YOLO_CONF  = 0.5

TELEMETRY_HZ = 1.0     # Hz publikacji telemetrii (min. 1 — wymóg zawodów)

DETECTION_COOLDOWN = 3.0
CACHE_EXPIRY_TIME  = 2.0  # Czas (w sekundach) przez jaki pamiętamy detekcję/QR do parowania
# ════════════════════════════════════════════════════════════════

import cv2
import threading
import queue
import time
import math
import json

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from dualtech_msgs.msg import Detection
from sensor_msgs.msg import CompressedImage, NavSatFix, NavSatStatus
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float32, String, Bool
from libcamera import Transform
from picamera2 import Picamera2
from ultralytics import YOLO
from pymavlink import mavutil


class DronLatajacy(Node):

    def __init__(self):
        super().__init__('dron_latajacy')

        # ── QoS ─────────────────────────────────────────────────
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ── Publishery ───────────────────────────────────────────
        self.publisher_detection = self.create_publisher(
            Detection, 'detection', reliable_qos
        )
        self.image_pub = self.create_publisher(
            CompressedImage, 'camera/fpv_annotated/compressed', image_qos
        )
        # Telemetria
        self.pub_gps      = self.create_publisher(NavSatFix,      '/drone/gps',           reliable_qos)
        self.pub_baro     = self.create_publisher(Float32,         '/drone/baro_altitude', reliable_qos)
        self.pub_gps_time = self.create_publisher(String,          '/drone/gps_time',      reliable_qos)
        self.pub_attitude = self.create_publisher(Vector3Stamped,  '/drone/attitude',      image_qos)
        self.pub_status   = self.create_publisher(String,          '/drone/status',        image_qos)

        # ── Subskrypcje ──────────────────────────────────────────
        self.create_subscription(Bool, '/scan_qr', self.callback_qr,   10)
        self.create_subscription(Bool, '/analiza', self.callback_yolo,  10)

        # ── Flagi stanu (thread-safe) ────────────────────────────
        self._state_lock = threading.Lock()
        self.scan_qr  = False
        self.analiza  = False
        self.last_qr  = None

        # ── Bufor telemetrii ─────────────────────────────────────
        self._telem_lock = threading.Lock()
        self.telem = {
            'lat': 0.0,   'lon': 0.0,   'gps_alt': 0.0,
            'baro_alt': 0.0,
            'gps_fix': 0, 'sats': 0,
            'hdop': 9999.0, 'vdop': 9999.0,
            'gps_time_usec': 0,
            'roll': 0.0,  'pitch': 0.0, 'yaw': 0.0,
            'armed': False, 'mode': '',
            'voltage': 0.0, 'current': 0.0, 'battery_pct': -1,
        }

        # ── Kolejki klatek dla wątków ─────────────────────────────
        self._qr_frame_queue   = queue.Queue(maxsize=1)
        self._yolo_frame_queue = queue.Queue(maxsize=1)

        # Wyniki z wątków
        self._last_yolo_result = None
        self._last_yolo_boxes  = []
        self._yolo_names       = {}
        self._result_lock      = threading.Lock()

        # ── Nowy asynchroniczny bufor pamięci podręcznej (Cache) ──
        self._cache_lock = threading.Lock()
        self._cached_qr = None
        self._cached_qr_time = 0.0
        self._cached_dets = []
        self._cached_dets_time = 0.0
        self._last_det_time = {}

        # ── Kamera Picamera2 ─────────────────────────────────────
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (CAMERA_WIDTH, CAMERA_HEIGHT)},
            transform=Transform(hflip=0, vflip=0)
        )
        self.picam2.configure(config)
        self.picam2.set_controls({"AfMode": 2})
        self.picam2.start()

        # ── Model YOLO ───────────────────────────────────────────
        self.model = YOLO(YOLO_MODEL_PATH, task='detect')
        self.etykiety = {
            'autobus':               'autobus',
            'czerwone auto':         'maluch',
            'brazowe auto':          'polonez',
            'zielone auto':          'ferrari',
            'ciezarowka':            'tir',
            'ciezarowka z wyrzutnia':'pansir',
            'samochod wojskowy':     'humvee',
            'duzy czolg':            'T-90',
            'maly czolg':            'T-62',
            'czlog z antena':        'radar',
        }

        # ── Detektor QR ──────────────────────────────────────────
        self._qr_detector = cv2.QRCodeDetector()

        # ── Licznik detekcji ─────────────────────────────────────
        self._detection_count = 0

        # ── Wątki tła ────────────────────────────────────────────
        self._qr_stop   = threading.Event()
        self._yolo_stop = threading.Event()

        self._qr_thread = threading.Thread(
            target=self._qr_worker, daemon=True, name='qr-worker'
        )
        self._yolo_thread = threading.Thread(
            target=self._yolo_worker, daemon=True, name='yolo-worker'
        )
        self._mav_thread = threading.Thread(
            target=self._mavlink_thread, daemon=True, name='mavlink-worker'
        )

        self._qr_thread.start()
        self._yolo_thread.start()
        self._mav_thread.start()

        # ── Timery ROS ───────────────────────────────────────────
        self.create_timer(0.033,          self.process_frame)        # ~30 FPS
        self.create_timer(1.0 / TELEMETRY_HZ, self._publish_telemetry)  # 1 Hz

        self.get_logger().info('═══ Dron Latający uruchomiony ═══')
        self.get_logger().info(f'  MAVLink: {MAVLINK_PORT} @ {MAVLINK_BAUD}')
        self.get_logger().info('  YOLO: OFF  |  QR: OFF')

    # ════════════════════════════════════════════════════════════
    #  WĄTEK QR (Zapis do stabilnego Cache)
    # ════════════════════════════════════════════════════════════
    def _qr_worker(self):
        while not self._qr_stop.is_set():
            try:
                frame_bgr = self._qr_frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            with self._state_lock:
                active = self.scan_qr

            if not active:
                with self._state_lock:
                    self.last_qr = None
                continue

            try:
                data, points, _ = self._qr_detector.detectAndDecode(frame_bgr)
                if data and points is not None and len(points) > 0:
                    with self._state_lock:
                        self.last_qr = data
                    
                    # Zapisz poprawny odczyt kodu QR do asynchronicznego cache
                    with self._cache_lock:
                        self._cached_qr = data
                        self._cached_qr_time = time.time()
                        
                    self.get_logger().info(f'QR wykryty: {data}')
            except cv2.error:
                pass
            except Exception as e:
                self.get_logger().warn(f'QR error: {e}')

    # ════════════════════════════════════════════════════════════
    #  WĄTEK YOLO (Zapis do stabilnego Cache)
    # ════════════════════════════════════════════════════════════
    def _yolo_worker(self):
        while not self._yolo_stop.is_set():
            try:
                frame_rgb = self._yolo_frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            with self._state_lock:
                active = self.analiza

            if not active:
                with self._result_lock:
                    self._last_yolo_result = None
                    self._last_yolo_boxes  = []
                continue

            try:
                results   = self.model(frame_rgb, imgsz=YOLO_IMGSZ, conf=YOLO_CONF,
                                       verbose=False, show=False)
                annotated = results[0].plot()   # BGR
                boxes     = results[0].boxes
                names     = self.model.names
                
                with self._result_lock:
                    self._last_yolo_result = annotated
                    self._last_yolo_boxes  = boxes
                    self._yolo_names       = names

                # Mapowanie i zapis detekcji do asynchronicznego cache pojazdów
                detections = []
                for box in boxes:
                    conf = float(box.conf[0])
                    raw_cls_name = names[int(box.cls[0])]
                    mapped_name = self.etykiety.get(raw_cls_name, raw_cls_name)
                    detections.append((mapped_name, conf, box))
                
                if detections:
                    with self._cache_lock:
                        self._cached_dets = detections
                        self._cached_dets_time = time.time()
            except Exception as e:
                self.get_logger().warn(f'YOLO error: {e}')

    # ════════════════════════════════════════════════════════════
    #  WĄTEK MAVLink
    # ════════════════════════════════════════════════════════════
    def _mavlink_thread(self):
        while rclpy.ok():
            try:
                self._mavlink_connect()
                self._mavlink_loop()
            except Exception as e:
                self.get_logger().error(f'[MAVLink] {e} — ponowna próba za 5s')
                time.sleep(5)

    def _mavlink_connect(self):
        self.get_logger().info(f'[MAVLink] Łączenie {MAVLINK_PORT} @ {MAVLINK_BAUD}...')
        self._mav = mavutil.mavlink_connection(MAVLINK_PORT, baud=MAVLINK_BAUD)

        hb = self._mav.recv_match(type='HEARTBEAT', blocking=True, timeout=15)
        if not hb:
            raise RuntimeError('Brak HEARTBEAT — sprawdź kabel UART i baudrate FC')

        self._src_sys  = hb.get_srcSystem()
        self._src_comp = hb.get_srcComponent()
        self.get_logger().info(f'[MAVLink] Połączono. SysID={self._src_sys}')

        self._mav.mav.request_data_stream_send(
            self._src_sys, self._src_comp,
            mavutil.mavlink.MAV_DATA_STREAM_ALL, MAVLINK_HZ, 1
        )
        for msg_id, interval_us in [
            (mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT,          100_000),
            (mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,  100_000),
            (mavutil.mavlink.MAVLINK_MSG_ID_VFR_HUD,              100_000),
            (mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,              100_000),
            (mavutil.mavlink.MAVLINK_MSG_ID_SYSTEM_TIME,        1_000_000),
            (mavutil.mavlink.MAVLINK_MSG_ID_BATTERY_STATUS,     1_000_000),
            (mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS,         1_000_000),
        ]:
            self._mav.mav.command_long_send(
                self._src_sys, self._src_comp,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0, float(msg_id), float(interval_us),
                0, 0, 0, 0, 0
            )
        self.get_logger().info(f'[MAVLink] Strumienie @ {MAVLINK_HZ} Hz')

    def _mavlink_loop(self):
        while rclpy.ok():
            msg = self._mav.recv_match(blocking=True, timeout=2.0)
            if not msg or msg.get_type() == 'BAD_DATA':
                continue
            self._process_mavlink(msg)

    def _process_mavlink(self, msg):
        t = msg.get_type()
        with self._telem_lock:
            if t == 'HEARTBEAT':
                self.telem['armed'] = bool(
                    msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                )
                self.telem['mode'] = {
                    0: 'STABILIZE', 2: 'ALT_HOLD', 3: 'AUTO',  4: 'GUIDED',
                    5: 'LOITER',    6: 'RTL',       7: 'CIRCLE', 9: 'LAND',
                    16: 'POSHOLD',
                }.get(msg.custom_mode, f'MODE_{msg.custom_mode}')

            elif t == 'GLOBAL_POSITION_INT':
                self.telem['lat']     = msg.lat / 1e7
                self.telem['lon']     = msg.lon / 1e7
                self.telem['gps_alt'] = msg.alt / 1000.0

            elif t == 'GPS_RAW_INT':
                self.telem['gps_fix'] = msg.fix_type
                self.telem['sats']    = msg.satellites_visible
                if msg.eph != 65535:
                    self.telem['hdop'] = msg.eph / 100.0
                if msg.epv != 65535:
                    self.telem['vdop'] = msg.epv / 100.0

            elif t == 'VFR_HUD':
                self.telem['baro_alt'] = msg.alt

            elif t == 'ATTITUDE':
                self.telem['roll']  = math.degrees(msg.roll)
                self.telem['pitch'] = math.degrees(msg.pitch)
                self.telem['yaw']   = math.degrees(msg.yaw)

            elif t == 'SYSTEM_TIME':
                self.telem['gps_time_usec'] = msg.time_unix_usec

            elif t == 'BATTERY_STATUS':
                if msg.voltages[0] != 65535:
                    self.telem['voltage'] = msg.voltages[0] / 1000.0
                self.telem['current']     = msg.current_battery / 100.0
                self.telem['battery_pct'] = msg.battery_remaining

            elif t == 'STATUSTEXT':
                self.get_logger().info(f'[FC] {msg.text.strip()}')

    # ════════════════════════════════════════════════════════════
    #  TIMER TELEMETRII — 1 Hz (wymóg zawodów)
    # ════════════════════════════════════════════════════════════
    def _publish_telemetry(self):
        now = self.get_clock().now().to_msg()
        with self._telem_lock:
            d = dict(self.telem)

        # 1. GPS
        gps = NavSatFix()
        gps.header.stamp    = now
        gps.header.frame_id = 'gps'
        ns = NavSatStatus()
        ns.status  = NavSatStatus.STATUS_FIX if d['gps_fix'] >= 3 else NavSatStatus.STATUS_NO_FIX
        ns.service = NavSatStatus.SERVICE_GPS
        gps.status    = ns
        gps.latitude  = d['lat']
        gps.longitude = d['lon']
        gps.altitude  = d['gps_alt']
        hdop, vdop = d['hdop'], d['vdop']
        gps.position_covariance = [
            hdop**2, 0.0, 0.0,
            0.0, hdop**2, 0.0,
            0.0, 0.0, vdop**2,
        ]
        gps.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        self.pub_gps.publish(gps)

        # 2. Wysokość barometryczna
        baro = Float32()
        baro.data = float(d['baro_alt'])
        self.pub_baro.publish(baro)

        # 3. Czas GPS (ISO 8601 UTC)
        t_msg = String()
        usec = d['gps_time_usec']
        if usec > 0:
            t_msg.data = (
                time.strftime('%Y-%m-%dT%H:%M:%S.', time.gmtime(usec / 1e6))
                + f'{int(usec % 1_000_000):06d}Z'
            )
        else:
            t_msg.data = 'NO_GPS_TIME'
        self.pub_gps_time.publish(t_msg)

        # 4. Postawa
        att = Vector3Stamped()
        att.header.stamp    = now
        att.header.frame_id = 'base_link'
        att.vector.x = d['roll']
        att.vector.y = d['pitch']
        att.vector.z = d['yaw']
        self.pub_attitude.publish(att)

        # 5. Status JSON
        s = String()
        s.data = json.dumps({
            'armed':       d['armed'],
            'mode':        d['mode'],
            'sats':        d['sats'],
            'gps_fix':     d['gps_fix'],
            'battery_pct': d['battery_pct'],
            'voltage':     round(d['voltage'], 2),
            'baro_alt':    round(d['baro_alt'], 2),
        })
        self.pub_status.publish(s)

    # ════════════════════════════════════════════════════════════
    #  GŁÓWNA PĘTLA ~30 FPS (Parowanie danych z Cache + Cooldown)
    # ════════════════════════════════════════════════════════════
    def process_frame(self):
        start_time = time.monotonic()

        frame_rgb = self.picam2.capture_array()   # RGB888
        if frame_rgb is None:
            return

        with self._state_lock:
            scan_qr = self.scan_qr
            analiza = self.analiza

        # RGB → BGR dla cv2
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Wyślij do wątku QR
        if scan_qr:
            frame_small = cv2.resize(frame_bgr, (320, 240))
            try:
                self._qr_frame_queue.put_nowait(frame_small)
            except queue.Full:
                pass

        # Wyślij do wątku YOLO
        if analiza:
            try:
                self._yolo_frame_queue.put_nowait(frame_rgb)
            except queue.Full:
                pass

        # Pobierz ostatni graficzny wynik YOLO dla FPV
        with self._result_lock:
            yolo_frame = self._last_yolo_result

        # Wybór bazy pod obraz wyjściowy HUD
        if analiza and yolo_frame is not None:
            display = yolo_frame.copy()
        else:
            display = frame_bgr.copy()

        # Pobieranie danych z pamięci podręcznej (Cache) do OSD i publikacji
        now_time = time.time()
        with self._cache_lock:
            active_qr = self._cached_qr if (now_time - self._cached_qr_time <= CACHE_EXPIRY_TIME) else None
            active_dets = self._cached_dets if (now_time - self._cached_dets_time <= CACHE_EXPIRY_TIME) else []

        # Stabilne nakładanie tekstu QR (dzięki cache tekst nie miga)
        if scan_qr and active_qr:
            cv2.putText(display, f'QR: {active_qr}',
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2, cv2.LINE_AA)
        elif scan_qr:
            cv2.putText(display, 'QR: skanowanie...',
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 200, 255), 2, cv2.LINE_AA)

        # Status YOLO na ekranie
        if not analiza:
            cv2.putText(display, 'YOLO: OFF',
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (100, 100, 100), 2, cv2.LINE_AA)

        # Telemetria HUD na obrazie
        with self._telem_lock:
            lat      = self.telem['lat']
            lon      = self.telem['lon']
            baro_alt = self.telem['baro_alt']
            sats     = self.telem['sats']
            gps_fix  = self.telem['gps_fix']
            armed    = self.telem['armed']
            mode     = self.telem['mode']

        fix_color = (0, 255, 0) if gps_fix >= 3 else (0, 60, 255)
        arm_color = (0, 60, 255) if armed else (0, 200, 255)

        cv2.putText(display,
            f'GPS: {lat:.5f}, {lon:.5f}  Alt: {baro_alt:.1f}m  Sats: {sats}',
            (10, display.shape[0] - 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, fix_color, 1, cv2.LINE_AA)
        cv2.putText(display,
            f'{"ARMED" if armed else "DISARMED"}  {mode}',
            (10, display.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, arm_color, 2, cv2.LINE_AA)

        # Wyświetlanie FPS
        elapsed = time.monotonic() - start_time
        if elapsed > 0:
            txt = f'FPS: {1.0 / elapsed:.1f}'
            tw  = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]
            cv2.putText(display, txt,
                        (display.shape[1] - tw - 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Kompresja i publikacja obrazu do FPV
        success, compressed_img = cv2.imencode(
            '.jpg', display, [int(cv2.IMWRITE_JPEG_QUALITY), 60]
        )
        img_msg = CompressedImage()
        if success:
            img_msg.header.stamp    = self.get_clock().now().to_msg()
            img_msg.header.frame_id = 'camera'
            img_msg.format          = 'jpeg'
            img_msg.data            = compressed_img.tobytes()
            self.image_pub.publish(img_msg)

        # ── KROK PAROWANIA I PUBLIKACJI /detection ──
        if analiza and scan_qr and active_dets and active_qr:
            try:
                # Wybieramy wykryty obiekt o najwyższym współczynniku pewności (conf)
                best_cls, best_conf, _ = max(active_dets, key=lambda x: x[1])
                key = f'{best_cls}|{active_qr}'

                # Filtrowanie spamu (Cooldown)
                if now_time - self._last_det_time.get(key, 0.0) >= DETECTION_COOLDOWN:
                    self._last_det_time[key] = now_time
                    self._detection_count += 1

                    # Spięcie aktualnej telemetrii GPS drona z obiektem
                    with self._telem_lock:
                        d = dict(self.telem)
                    
                    gps_loc = NavSatFix()
                    gps_loc.header.stamp    = self.get_clock().now().to_msg()
                    gps_loc.header.frame_id = 'gps'
                    ns = NavSatStatus()
                    ns.status  = NavSatStatus.STATUS_FIX if d['gps_fix'] >= 3 else NavSatStatus.STATUS_NO_FIX
                    ns.service = NavSatStatus.SERVICE_GPS
                    gps_loc.status    = ns
                    gps_loc.latitude  = d['lat']
                    gps_loc.longitude = d['lon']
                    gps_loc.altitude  = d['gps_alt']
                    gps_loc.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN

                    # Budowanie ostatecznej struktury wiadomości
                    det_msg = Detection()
                    det_msg.object_type  = best_cls
                    det_msg.qr_value     = str(active_qr)
                    det_msg.object_image = img_msg if success else CompressedImage()
                    det_msg.gps_location = gps_loc
                    
                    self.publisher_detection.publish(det_msg)

                    self.get_logger().info(
                        f'[DETEKCJA SPAROWANA #{self._detection_count}] '
                        f'Obiekt: {best_cls} ({best_conf:.2f}) | QR: "{active_qr}" | '
                        f'GPS: ({d["lat"]:.5f}, {d["lon"]:.5f})'
                    )
            except Exception as e:
                self.get_logger().warn(f'Detection publish error: {e}')

    # ════════════════════════════════════════════════════════════
    #  CALLBACKI
    # ════════════════════════════════════════════════════════════
    def callback_qr(self, msg: Bool):
        with self._state_lock:
            self.scan_qr = msg.data
            if not msg.data:
                self.last_qr = None
        self.get_logger().info(f'Skanowanie QR: {"ON" if msg.data else "OFF"}')

    def callback_yolo(self, msg: Bool):
        with self._state_lock:
            self.analiza = msg.data
        if not msg.data:
            with self._result_lock:
                self._last_yolo_result = None
                self._last_yolo_boxes  = []
        self.get_logger().info(f'Detekcja YOLO: {"ON" if msg.data else "OFF"}')


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = DronLatajacy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._qr_stop.set()
        node._yolo_stop.set()
        node._qr_thread.join(timeout=2.0)
        node._yolo_thread.join(timeout=2.0)
        node.picam2.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
