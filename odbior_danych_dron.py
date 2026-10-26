#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import cv2
import numpy as np
from cv_bridge import CvBridge
from datetime import datetime
import json

from dualtech_msgs.msg import Detection
from sensor_msgs.msg import CompressedImage, NavSatFix
from std_msgs.msg import Float32, String
from geometry_msgs.msg import Vector3Stamped

# ─── KONFIGURACJA ────────────────────────────────────────────────────────────
DETECTION_DISPLAY_SEC = 5.0   # ile sekund trzymać baner ostatniej detekcji
# ─────────────────────────────────────────────────────────────────────────────

def draw_hud(frame: np.ndarray, tele: dict, detection_info: dict | None) -> np.ndarray:
    """Nakłada HUD telemetryczny na obraz FPV na podstawie stanu ze słownika."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # ── Pasek górny (GPS) ────────────────────────────────────────────────────
    cv2.rectangle(overlay, (0, 0), (w, 38), (0, 0, 0), -1)

    fix   = tele.get('gps_fix', 0)
    sats  = tele.get('sats', 0)
    lat   = tele.get('lat', 0.0)
    lon   = tele.get('lon', 0.0)
    t_gps = tele.get('gps_time', '--')

    fix_str = f"3D ({fix})" if fix >= 3 else f"NO_FIX ({fix})"
    fix_color = (0, 255, 0) if fix >= 3 else (0, 60, 255)
    
    cv2.putText(overlay, f'GPS [{fix_str}] {sats}sat',  (8, 16),  cv2.FONT_HERSHEY_SIMPLEX, 0.52, fix_color, 1)
    cv2.putText(overlay, f'Lat:{lat:.6f}  Lon:{lon:.6f}',    (8, 32),  cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220), 1)
    cv2.putText(overlay, t_gps[-13:] if len(t_gps) > 10 else '--', (w - 115, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # ── Pasek dolny (alt + orientacja + bateria) ───────────────────────────────
    cv2.rectangle(overlay, (0, h - 42), (w, h), (0, 0, 0), -1)

    alt_gps  = tele.get('gps_alt',  0.0)
    alt_baro = tele.get('baro_alt', 0.0)
    roll     = tele.get('roll',  0.0)
    pitch    = tele.get('pitch', 0.0)
    yaw      = tele.get('yaw',   0.0)
    vbat     = tele.get('voltage', 0.0)
    bat_pct  = tele.get('battery_pct', -1)
    armed    = tele.get('armed', False)
    mode     = tele.get('mode', '?')

    bat_color = (0, 255, 0) if bat_pct > 30 else (0, 60, 255)
    arm_color = (0, 60, 255) if armed else (0, 200, 255)

    cv2.putText(overlay, f'AltGPS:{alt_gps:.1f}m  AltBARO:{alt_baro:.1f}m',
                (8, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
    cv2.putText(overlay, f'Hdg(Yaw):{yaw:.1f} deg',
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
    cv2.putText(overlay, f'{vbat:.2f}V {bat_pct}%',
                (w - 125, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, bat_color, 1)
    cv2.putText(overlay, f'{"ARMED" if armed else "DISARMED"} {mode}',
                (w - 125, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, arm_color, 1)

    # ── Wskaźnik Roll/Pitch (sztuczny horyzont) ───────────────────────────────
    cx, cy = w // 2, h // 2
    bar_len = 40
    cv2.line(overlay, (cx - bar_len, cy), (cx + bar_len, cy), (0, 255, 0), 1)
    cv2.line(overlay, (cx, cy - bar_len), (cx, cy + bar_len), (0, 255, 0), 1)
    cv2.putText(overlay, f'R:{roll:.1f} P:{pitch:.1f}', (cx - 45, cy + bar_len + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # ── Baner ostatniej detekcji ──────────────────────────────────────────────
    if detection_info:
        obj  = detection_info.get('object_type', '?')
        qr   = detection_info.get('qr_value', '?')
        dlat = detection_info.get('lat', '?')
        dlon = detection_info.get('lon', '?')
        cv2.rectangle(overlay, (0, 38), (w, 72), (0, 80, 0), -1)
        cv2.putText(overlay, f'WYKRYTO: {obj}   QR: {qr}',
                    (8, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 80), 1)
        cv2.putText(overlay, f'Pozycja obiektu GPS: {dlat}, {dlon}',
                    (8, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1)

    return cv2.addWeighted(overlay, 0.88, frame, 0.12, 0)


class OdbiornikLaptop(Node):

    def __init__(self):
        super().__init__('odbiornik_laptop')
        self.bridge = CvBridge()

        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ── Stan słownika telemetrii zgodny z dronem ───────────────────────────
        self.tele = {
            'lat': 0.0, 'lon': 0.0, 'gps_alt': 0.0,
            'baro_alt': 0.0, 'gps_fix': 0, 'sats': 0,
            'gps_time': '--', 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'armed': False, 'mode': '?', 'voltage': 0.0, 'battery_pct': -1
        }
        self.last_detection: dict | None = None
        self.last_detection_time: float  = 0.0
        self.detection_count: int        = 0

        # ── Nowe, poprawne subskrypcje na topiki drona ────────────────────────
        self.create_subscription(NavSatFix, '/drone/gps', self._gps_cb, reliable_qos)
        self.create_subscription(Float32, '/drone/baro_altitude', self._baro_cb, reliable_qos)
        self.create_subscription(String, '/drone/gps_time', self._gps_time_cb, reliable_qos)
        self.create_subscription(Vector3Stamped, '/drone/attitude', self._attitude_cb, best_effort_qos)
        self.create_subscription(String, '/drone/status', self._status_json_cb, best_effort_qos)
        
        # Detekcja i wideo
        self.create_subscription(Detection, 'detection', self._detection_cb, reliable_qos)
        self.create_subscription(CompressedImage, 'camera/fpv_annotated/compressed', self._fpv_cb, best_effort_qos)

        # ── Okna OpenCV ───────────────────────────────────────────────────────
        cv2.namedWindow('FPV Dron', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('FPV Dron', 800, 600)
        cv2.namedWindow('Ostatnia Detekcja', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Ostatnia Detekcja', 400, 300)

        self.get_logger().info('Odbiornik uruchomiony. Słucham właściwych topików z drona...')

    # ── NOWE CALLBACKI TELEMETRII ────────────────────────────────────────────

    def _gps_cb(self, msg: NavSatFix):
        self.tele['lat'] = msg.latitude
        self.tele['lon'] = msg.longitude
        self.tele['gps_alt'] = msg.altitude
        self.tele['gps_fix'] = 3 if msg.status.status >= 0 else 0

    def _baro_cb(self, msg: Float32):
        self.tele['baro_alt'] = msg.data

    def _gps_time_cb(self, msg: String):
        self.tele['gps_time'] = msg.data

    def _attitude_cb(self, msg: Vector3Stamped):
        self.tele['roll'] = msg.vector.x
        self.tele['pitch'] = msg.vector.y
        self.tele['yaw'] = msg.vector.z

    def _status_json_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
            self.tele['armed'] = data.get('armed', False)
            self.tele['mode'] = data.get('mode', '?')
            self.tele['sats'] = data.get('sats', 0)
            self.tele['battery_pct'] = data.get('battery_pct', -1)
            self.tele['voltage'] = data.get('voltage', 0.0)
        except Exception:
            pass

    # ── POPRAWIONE CALLBACKI OBRAZU I DETEKCJI ───────────────────────────────

    def _detection_cb(self, msg: Detection):
        self.detection_count += 1
        gps = msg.gps_location

        lat_str = f'{gps.latitude:.6f}'
        lon_str = f'{gps.longitude:.6f}'
        alt_str = f'{gps.altitude:.1f}'

        self.last_detection = {
            'object_type': msg.object_type,
            'qr_value': msg.qr_value,
            'lat': lat_str, 'lon': lon_str, 'alt': alt_str,
        }
        self.last_detection_time = rclpy.clock.Clock().now().nanoseconds / 1e9

        self.get_logger().info(f'[DETEKCJA #{self.detection_count}] Trafienie: {msg.object_type}')


    def _fpv_cb(self, msg: CompressedImage):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Błąd dekodowania FPV: {e}')
            return

        # Czy baner detekcji ma być jeszcze aktywny?
        now = rclpy.clock.Clock().now().nanoseconds / 1e9
        active_detection = (
            self.last_detection 
            if (now - self.last_detection_time) < DETECTION_DISPLAY_SEC 
            else None
        )

        # Rysuj HUD korzystając ze spiętej telemetrii
        hud_frame = draw_hud(frame, self.tele, active_detection)
        cv2.imshow('FPV Dron', hud_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            raise KeyboardInterrupt
        elif key == ord('s'):
            fname = f'screenshot_{datetime.now().strftime("%H%M%S")}.jpg'
            cv2.imwrite(fname, hud_frame)


def main(args=None):
    rclpy.init(args=args)
    node = OdbiornikLaptop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()