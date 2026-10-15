print("Booting up the system…")
print("Setting up logger…")
import datetime
import logging
import sys
import traceback
from pathlib import Path
from typing import NoReturn

print("setting root path...")
# パス解決と初期設定
current = Path(__file__).resolve()
root_path = None
for parent in [current.parent, *current.parents]:
    if parent.name == "sun_find_west_v2":
        root_path = parent
        sys.path.append(str(root_path))
        break

if root_path is None:
    print(
        "[FATAL] The folder 'sun_find_west_v2' was not found in the parent directory."
    )
    sys.exit(1)

print("setting logger...")
dt = datetime.datetime.now().strftime("%Y%m%d")
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
reports_path = root_path.parent / "report" / f"{dt}"
reports_path.mkdir(parents=True, exist_ok=True)

logfile = reports_path / "logs" / f"sunfindwestV2_{ts}.log"
logfile.parent.mkdir(parents=True, exist_ok=True)

format = "%(asctime)s [%(levelname)s] %(name)s %(funcName)s: %(message)s"
formatter = logging.Formatter(format)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler(filename=logfile, mode="a", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logging.basicConfig(handlers=[file_handler, console_handler], level=logging.DEBUG)
logger = logging.getLogger(__name__)
# TONOTDO:config,schemaの更新コマンドを追加
# TODO:logで残すためにも例外処理を強化
logger.info("====== START PROCESSING ======\n\n")
logger.info(f"log={logfile}")
print("Initializing forced termination procedure…")


def cancel_process(camera=None, viz=None) -> NoReturn:
    """
    初期化中（アプリ起動前）に致命的エラーが発生した場合の強制終了処理。
    カメラおよび描画リソースを安全に解放してプログラムを終了します。

    Args:
        camera: ASIカメラのインスタンス（未接続時は None）
        viz: Visualizerのインスタンス（未起動時は None）
    """
    logger.info("===== canceling process and cleaning up =====")
    try:
        if camera is not None:
            try:
                camera.stop_video_capture()
                camera.close()
                logger.debug("Camera closed in cancel_process.")
            except Exception as e:
                logger.error(f"Failed to close camera: {e}")

        if "cv2" in sys.modules:
            sys.modules["cv2"].destroyAllWindows()

        if viz is not None:
            try:
                viz.close()
                logger.info("Visualizer closed in cancel_process.")
            except Exception as e:
                logger.error(f"Failed to close viz: {e}")
    except Exception as e:
        logger.error(f"Failed to release resources: {e}")
    logger.info("===== FINISHED =====")
    sys.exit(1)


# モジュールインポートとパラメータロード
print("Importing modules...")
try:
    import csv
    import os
    from collections import deque
    from pprint import pformat
    from time import time

    import cv2
    import numpy as np
except ImportError:
    logger.error("Failed to import standard modules.")
    logger.error(traceback.format_exc())
    cancel_process()
else:
    logger.debug("All standard modules imported successfully.")

try:
    from camera.controller import (
        apply_camera_config,
        asi,
        connect_camera,
        handle_config,
    )
    from config.config_manager import parameter
    from core.drawer import Visualizer
    from core.MIN2ver2 import MIN2_ignore_sunspots as MIN2
    from core.RANSAC import calculate_west_angle_robust as west_angle
except ImportError:
    logger.error("Failed to import custom modules.")
    logger.error(traceback.format_exc())
    cancel_process()
else:
    logger.debug("All custom modules imported successfully.")

logger.debug(f"loaded parameters:\n{pformat(parameter)}")

print("Loading parameters...")
try:
    camera_param = parameter["Camera"]
    main_param = parameter["sun_find_west_v2"]
    visualizer_constract_param = main_param["Visualizer"]
    visualizer_constract_param["acceptable"] = main_param["Analyzer"]["acceptable"]
except RuntimeError:
    logger.exception("Failed to load parameters.")
    cancel_process()
logger.info("Successfully loaded parameters.")

# カメラ接続と設定
print("Implementing parameters...")
logger.info("Attempting to connect to the camera...")
camera = None
try:
    env_filename = root_path / "camera" / "bin" / "ASICamera2.dll"
    os.environ["ZWO_ASI_LIB"] = str(env_filename)
    logger.debug(f"Successfully set ZWO_ASI_LIB environment variable:{env_filename}")

    camera = connect_camera(env_filename)
except KeyboardInterrupt:
    logger.info("Connection wait interrupted by user.")
    cancel_process()
except asi.ZWO_CaptureError as e:
    logger.critical(f"Failed to connect to the camera: {e}")
    cancel_process(camera=camera)
else:
    logger.info("Successfully connected to the camera.")

if camera is not None:
    flatten_param = {}
    for hhv in camera_param.values():
        flatten_param.update(hhv)

    apply_camera_config(camera, flatten_param)
    logger.info("Starting video capture...")
    try:
        camera.start_video_capture()
        width, height, binning, img_type = camera.get_roi_format()
        logger.debug(
            f"Capture status retrieved: width={width}, height={height}, binning={binning}, img_type={img_type}"
        )
    except asi.ZWO_Error as e:
        logger.critical(f"Failed to start video capture or retrieve ROI: {e}")
        cancel_process(camera=camera)
else:
    logger.critical("Camera connection failed.")
    cancel_process()

# 補助関数と保存先設定
# 画像フォーマットとNumPyデータ型のマッピング
_frame_format_map = {
    getattr(asi, "ASI_IMG_RAW8", 0): (np.uint8, 1),
    getattr(asi, "ASI_IMG_Y8", 3): (np.uint8, 1),
    getattr(asi, "ASI_IMG_RAW16", 2): (np.uint16, 1),
    getattr(asi, "ASI_IMG_RGB24", 1): (np.uint8, 3),
}


def frame_to_image(frame: bytes, width: int, height: int, img_type: int) -> np.ndarray:
    """
    カメラから取得したバイト列を、img_typeに応じたNumPy配列に変換します。

    Args:
        frame (bytes): カメラから取得した生の画像データ
        width (int): 画像の幅
        height (int): 画像の高さ
        img_type (int): ZWO ASI SDKで定義される画像タイプ

    Returns:
        np.ndarray: 変換された2次元または3次元のNumPy画像配列
    """
    dtype, channels = _frame_format_map.get(img_type, (np.uint8, 1))
    arr = np.frombuffer(frame, dtype=dtype)
    if channels == 1:
        return arr.reshape(height, width)
    return arr.reshape(height, width, channels)


cap_dir = reports_path / "captures"
cap_dir.mkdir(parents=True, exist_ok=True)


# SunTrackerApp クラス定義
# TODO:パラメータを可能な限り増やす(Visualyzer中心に)
class SunTrackerApp:
    """
    太陽の位置解析・角度算出・UI描画を統括するメインアプリケーションクラス。
    """

    def __init__(self, camera, viz, main_param, csv_file_path, cap_dir, img_info):
        """
        Args:
            camera: 初期化・接続済みのZWO ASIカメラインスタンス
            viz (Visualizer): UI描画用インスタンス
            main_param (dict): 設定ファイルから読み込んだメインパラメータ
            csv_file_path (Path): 解析結果を保存するCSVのパス
            cap_dir (Path): 手動キャプチャ画像を保存するディレクトリ
            img_info (tuple): (width, height, img_type) からなるカメラのROI情報
        """
        self.camera = camera
        self.viz = viz
        self.buf_lookback = int(main_param["Analyzer"]["buf_lookback"])
        self.csv_file_path = csv_file_path
        self.cap_dir = cap_dir
        self.width, self.height, self.img_type = img_info

        # 状態・軌跡バッファ管理
        self.frame_count = 0
        self.dropped_frames = 0
        self.st_time = time()
        self.buffer_c = deque[list[float]](maxlen=500)
        self.buffer_t = deque[float](maxlen=500)
        self.capture_requested = False
        self.quit_requested = False

        self._bind_ui_callbacks()

    def _bind_ui_callbacks(self):
        """UIコンポーネント（ボタン・スライダー）へのイベントバインド"""
        self.viz.add_button(
            name="reset_buffer",
            label="Reset",
            on_clicked=self.reset_buffers,
            position=[0.02, 0.05, 0.07, 0.04],
        )
        self.viz.add_button(
            name="capture_image",
            label="Capture",
            on_clicked=self.request_capture,
            position=[0.10, 0.05, 0.07, 0.04],
        )
        self.viz.add_button(
            name="quit",
            label="Quit",
            on_clicked=self.request_quit,
            position=[0.18, 0.05, 0.07, 0.04],
        )
        self.viz.add_slider(
            name="gain",
            label="Gain (dB)",
            valmin=0,
            valmax=300,
            valinit=150,
            valfmt="%1.0f",
            on_change=lambda val: handle_config(self.camera, asi.ASI_GAIN, val),
        )
        self.viz.add_slider(
            name="exposure",
            label="Exposure (µs=1e-6s)",
            valmin=1000,
            valmax=100000,
            valinit=30000,
            valfmt="%1.0f",
            on_change=lambda val: handle_config(self.camera, asi.ASI_EXPOSURE, val),
        )

    def reset_buffers(self):
        """軌跡バッファとフレームカウントを手動で初期化します。"""
        self.frame_count = 0
        self.buffer_c.clear()
        self.buffer_t.clear()
        logger.info("The trajectory buffer has been manually reset.")

    def request_capture(self):
        """現在のフレーム画像の保存をリクエストします。"""
        self.capture_requested = True

    def request_quit(self):
        """メインループの終了をリクエストします。"""
        self.quit_requested = True
        logger.info("The exit button has been pressed.")

    def cleanup(self):
        """
        リソースの解放処理。
        ループ終了後、または例外発生時に必ず実行され、カメラやウィンドウを閉じます。
        """
        logger.info("===== cleaning up and releasing resources =====")
        try:
            if self.camera is not None:
                try:
                    self.camera.stop_video_capture()
                    self.camera.close()
                    logger.debug("Camera stopped and closed successfully.")
                except (asi.ZWO_CaptureError, OSError) as e:
                    logger.error(f"Failed to close camera: {e}")

            cv2.destroyAllWindows()

            if self.viz is not None:
                try:
                    self.viz.close()
                    logger.info("Visualizer resources released.")
                except Exception as e:
                    logger.error(f"Failed to close viz: {e}")

        except Exception as e:
            logger.error(f"Failed to release resources during cleanup: {e}")
        logger.info("===== FINISHED =====")

    def run(self, csv_headers):
        """
        リアルタイム画像処理と描画のメインループ。

        Args:
            csv_headers (list): CSVファイルに出力するヘッダー行のリスト
        """
        logger.info(
            "Starting real-time visualization. Press the 'Quit' button or close the window to exit."
        )

        try:
            with open(
                self.csv_file_path, mode="w", newline="", encoding="utf-8"
            ) as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(csv_headers)

                while True:
                    if self.quit_requested:
                        logger.info("Visualization loop terminated by Quit button.")
                        break

                    try:
                        frame = self.camera.capture_video_frame(timeout=500)
                    except asi.ZWO_CaptureError as e:
                        logger.warning(f"Frame capture failed or timed out: {e}")
                        self.dropped_frames += 1
                        continue

                    img = frame_to_image(frame, self.width, self.height, self.img_type)
                    self.frame_count += 1

                    # 1. 太陽中心の解析
                    try:
                        (cx, cy), r = MIN2(img)
                    except Exception as e:
                        logger.warning(f"MIN2 processing error: {e}")
                        self.dropped_frames += 1
                        continue

                    self.buffer_c.append([cx, cy])
                    buf_c_arr = np.array(self.buffer_c)
                    recent_pts = buf_c_arr[-self.buf_lookback :]

                    capture_time_unix = time()
                    capture_time_iso = datetime.datetime.now().isoformat()
                    self.buffer_t.append(capture_time_unix)
                    buf_t_arr = np.array(self.buffer_t)
                    recent_timestamps = buf_t_arr[-self.buf_lookback :]

                    # 2. 角度の堅牢な計算 (RANSACベース)
                    if len(recent_pts) > 2:
                        try:
                            result = west_angle(recent_pts, recent_timestamps)
                            if result is None:
                                raise RuntimeError
                            robust_angle, vectorYX = result
                        except (ValueError, TypeError, RuntimeError) as e:
                            logger.warning(f"Error calculating robust west angle: {e}")
                            robust_angle = None
                    else:
                        robust_angle = None

                    # 3. カメラハードウェア状態の取得
                    try:
                        current_gain = self.camera.get_control_value(asi.ASI_GAIN)[0]
                        current_exposure = self.camera.get_control_value(
                            asi.ASI_EXPOSURE
                        )[0]
                        temp_raw = self.camera.get_control_value(asi.ASI_TEMPERATURE)[0]
                        current_temp = round(temp_raw / 10.0, 1)
                    except asi.ZWO_Error as e:
                        logger.warning(f"Failed to read camera control values: {e}")
                        current_gain, current_exposure, current_temp = None, None, None

                    # 4. 解析結果と状態のロギング
                    try:
                        csv_writer.writerow(
                            [
                                capture_time_iso,
                                capture_time_unix,
                                self.frame_count,
                                current_temp,
                                current_gain,
                                current_exposure,
                                cx,
                                cy,
                                r,
                                robust_angle,
                            ]
                        )
                        csv_file.flush()
                    except Exception as e:
                        logger.error(f"Failed to write row to CSV: {e}")

                    # 5. UIの描画更新
                    try:
                        self.viz.update(
                            img,
                            cx,
                            cy,
                            r,
                            recent_pts,
                            robust_angle,
                            vectorYX,
                            frame_idx=self.frame_count,
                            total_frames="∞",
                        )
                    except Exception as e:
                        logger.warning(f"Visualizer update failed: {e}")

                    # 6. 画像のオンデマンド保存
                    if self.capture_requested:
                        self.capture_requested = False
                        cap_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[
                            :-3
                        ]
                        raw_path = (
                            self.cap_dir / f"raw_{cap_ts}_f{self.frame_count}.png"
                        )
                        if cv2.imwrite(str(raw_path), img):
                            logger.info(f"Capture saved.:{raw_path.name}")
                        else:
                            logger.error(
                                f"Failed to save capture.: \n path = {raw_path.name}"
                            )

                    # 7. ウィンドウが外部から閉じられたかの検知
                    if not self.viz.is_alive():
                        logger.info("Visualization loop terminated (window closed).")
                        break

            # 正常終了時の統計ログ
            elapsed_time = float(time() - self.st_time)
            logger.debug(
                f"Completed successfully. Total frames: {self.frame_count}, Dropped: {self.dropped_frames}, Time: {elapsed_time:.2f}s"
            )

        except KeyboardInterrupt as e:
            elapsed_time = float(time() - self.st_time)
            logger.info(f"Keyboard interrupt: {e}")
            logger.debug(
                f"Terminated by keyboard interrupt. Total frames: {self.frame_count}, Dropped: {self.dropped_frames}, Time: {elapsed_time:.2f}s"
            )
            cancel_process(viz=self.viz, camera=self.camera)
        except RuntimeError as e:
            elapsed_time = float(time() - self.st_time)
            logger.error(f"Runtime error occurred: {e}")
            logger.debug(
                f"Terminated with error. Total frames: {self.frame_count}, Dropped:{self.dropped_frames}, Time: {elapsed_time:.2f}s"
            )
            cancel_process(viz=self.viz, camera=self.camera)
        finally:
            self.cleanup()


# メインエントリポイント

if __name__ == "__main__":
    if camera is None:
        logger.critical("Cannot proceed without initialized camera")
        sys.exit(1)

    logger.info("Initializing Visualizer instance...")

    csv_file_path = reports_path / f"sun_data_{ts}_BYsunfindwestV2.csv"
    logger.info(f"csv_file_path: {csv_file_path}")

    csv_headers = [
        "timestamp_iso",
        "timestamp_unix",
        "frame_count",
        "temperature_c",
        "gain",
        "exposure_us",
        "cx",
        "cy",
        "r",
        "robust_angle",
    ]

    viz = Visualizer(width, height, **visualizer_constract_param)

    img_info = (width, height, img_type)
    app = SunTrackerApp(
        camera=camera,
        viz=viz,
        main_param=main_param,
        csv_file_path=csv_file_path,
        cap_dir=cap_dir,
        img_info=img_info,
    )

    app.run(csv_headers)
