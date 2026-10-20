"""ZWO ASIカメラの初期化、接続、設定を管理するモジュール。"""

import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class CameraControlsProtocol(Protocol):
    """カメラ設定関数が必要とする操作を定義する。"""

    def get_camera_property(self) -> Mapping[str, Any]:
        """カメラの基本プロパティを返す。"""
        ...

    def set_roi_format(
        self,
        width: int | None = None,
        height: int | None = None,
        bins: int = 1,
        image_type: int = 0,
        start_x: int = 0,
        start_y: int = 0,
    ) -> None:
        """ROI の形式を設定する。"""
        ...

    def get_controls(self) -> Mapping[int, object]:
        """利用可能なカメラ制御項目を返す。"""
        ...

    def set_control_value(
        self, control_type: int, value: int | float, auto: bool = False
    ) -> None:
        """指定したカメラ制御項目を更新する。"""
        ...

if __name__ == "__main__":
    logger.info("--- starting as main process ---")
else:
    logger.info("--- starting as module process ---")

logger.info("standard modules imported successfully")

try:
    from camera.vid_dummy import ZWOASI_AVAILABLE, VideoDummyCamera, asi
except ImportError:
    logger.critical("no module to be asi")
    raise


def connect_camera(
    dll_path: Path, use_dummy: bool
) -> asi.Camera | VideoDummyCamera | None:
    """ASIカメラを初期化し、接続されるまで待機する。

    Parameters
    ----------
    dll_path : Path
        ASICamera2.dll のパス。
    use_dummy : bool
        True ならダミーカメラを返し、実機への接続処理は行わない。
        False なら実機ASIカメラに接続する。zwoasi が無い場合は RuntimeError を送出する。
        この値は run.py が決め、自動判定やキー入力による切替は行わない。

    Returns
    -------
    asi.Camera | VideoDummyCamera | None
        接続されたカメラ。DLL が見つからない場合は None。

    Raises
    ------
    RuntimeError
        実機モードで zwoasi が利用できない場合。
    """
    if use_dummy:
        logger.info("ダミーモードとして接続処理をskipします")
        return VideoDummyCamera()

    if not ZWOASI_AVAILABLE:
        msg = (
            "zwoasi がインストールされていません。実機カメラを使うには zwoasi を導入するか、"
            "config.yaml の Run>camera_source>dummy_camera を true にしてください。"
        )
        logger.critical(msg)
        raise RuntimeError(msg)

    # 1. DLLパスの存在確認
    if not dll_path.exists():
        logger.error(
            f"DLL not found: {dll_path}. Please place the 64-bit ASICamera2.dll at this path."
        )
        input("\nPress Enter to continue...")
        return None

    # 2. SDKの初期化
    logger.info("Initializing ASI SDK...")
    try:
        # すでに初期化されている場合の対策
        try:
            asi.get_num_cameras()
            logger.debug("ASI SDK already initialized.")
        except (asi.ZWO_Error, AttributeError):
            asi.init(str(dll_path))
        logger.info("ASI SDK initialized successfully.")
    except (asi.ZWO_Error, OSError) as e:
        logger.error("Failed to initialize ASI SDK (architecture mismatch?): %s", e)
        raise

    # 3. カメラ接続の待機ループ
    logger.info("Waiting for camera connection... (Press Ctrl+C to abort)")
    camera_index = None

    try:
        while True:
            try:
                cameras = asi.list_cameras()
            except (asi.ZWO_Error, OSError):
                # list_cameras が失敗する可能性があるため空リストにする
                cameras = []

            if cameras:
                logger.debug("Camera(s) detected (Count: %s)", len(cameras))
                logger.debug("Available cameras: %s", cameras)
                camera_index = 0
                break

            print(".", end="", flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Connection wait interrupted by user.")
        raise

    logger.info("Camera acquired successfully.")
    # 4. カメラの初期化と基本設定
    try:
        camera = asi.Camera(camera_index)
        camera_info = camera.get_camera_property()
        logger.debug("Selected camera: %s", camera_info["Name"])

        # 汎用的な基本設定
        camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, 40)
        camera.disable_dark_subtract()
        logger.info("Camera configured successfully.")
        return camera
    except asi.ZWO_Error as e:
        logger.error("ASI error occurred during camera initialization: %s", e)
        raise
    except Exception:
        # その他の予期しない例外はログに記録して終了
        logger.exception("Unexpected error initializing camera")
        raise


IMG_TYPE_MAP = {
    "RAW8": asi.ASI_IMG_RAW8,
    "RAW16": asi.ASI_IMG_RAW16,
    "RGB24": asi.ASI_IMG_RGB24,
    "Y8": asi.ASI_IMG_Y8,
}

control_map = {
    "exposure": asi.ASI_EXPOSURE,
    "gain": asi.ASI_GAIN,
    "offset": asi.ASI_OFFSET,
    "gamma": asi.ASI_GAMMA,
    "band_width": asi.ASI_BANDWIDTHOVERLOAD,
    "high_speed_mode": asi.ASI_HIGH_SPEED_MODE,
    "hardware_bin": asi.ASI_HARDWARE_BIN,
    "flip": asi.ASI_FLIP,
    "auto_max_gain": asi.ASI_AUTO_MAX_GAIN,
    "auto_max_exp": asi.ASI_AUTO_MAX_EXP,
    "auto_target_brightness": asi.ASI_AUTO_MAX_BRIGHTNESS,
    "target_temp": asi.ASI_TARGET_TEMP,
    "cooler_on": asi.ASI_COOLER_ON,
}


def apply_camera_config(cam: CameraControlsProtocol, config: dict) -> None:
    """ZWO ASIカメラの各種パラメータを一括で設定する。

    Parameters
    ----------
    cam : CameraControlsProtocol
        初期化済みのカメラ。
    config : dict
        カメラの各種パラメータを格納した辞書。
    """
    props = cam.get_camera_property()

    width = config.get("width", "max")
    if width == "max":
        width = props["MaxWidth"]
    height = config.get("height", "max")
    if height == "max":
        height = props["MaxHeight"]
    bins = config.get("bins", 1)
    img_type_str = str(config.get("img_type", "RAW8")).upper()
    img_type = IMG_TYPE_MAP.get(img_type_str, asi.ASI_IMG_RAW8)

    cam.set_roi_format(width=width, height=height, bins=bins, image_type=img_type)

    available_controls = cam.get_controls()

    for key, control_type in control_map.items():
        if key in config:
            if control_type in available_controls:
                val = config[key]
                if isinstance(val, bool):
                    val = int(val)
                try:
                    cam.set_control_value(control_type, val)
                except asi.ZWO_Error:
                    logger.exception(
                        "__ENG%s の設定に失敗しました (範囲外の値などの可能性)", key
                    )
            else:
                logger.warning("__ENGこのカメラは%sに対応していません", key)

    logger.info("__sucessful set camera config")


def handle_config(cam: CameraControlsProtocol, key: int, val: float) -> None:
    """指定された ASI コントロールを更新する。

    利用可能な主なキーは ``ASI_EXPOSURE``、``ASI_GAIN``、``ASI_OFFSET``、
    ``ASI_BRIGHTNESS``、``ASI_GAMMA``、``ASI_WB_R``、``ASI_WB_B``、
    ``ASI_BANDWIDTHOVERLOAD``、``ASI_HIGH_SPEED_MODE``、``ASI_HARDWARE_BIN``、
    ``ASI_MONO_BIN``、``ASI_FLIP``、``ASI_AUTO_MAX_GAIN``、``ASI_AUTO_MAX_EXP``、
    ``ASI_AUTO_TARGET_BRIGHTNESS``、``ASI_TARGET_TEMP``、``ASI_COOLER_ON``、
    ``ASI_FAN_ON``、``ASI_ANTI_DEW_HEATER``。

    Parameters
    ----------
    cam : CameraControlsProtocol
        設定対象のカメラ。
    key : int
        ASI コントロール ID。
    val : float
        設定する値。
    """
    try:
        cam.set_control_value(key, int(val))
        logger.debug("%s updated to: %s", key, int(val))
    except asi.ZWO_Error as e:
        logger.error("Failed to update %s: %s", key, e)


logger.info("--- finish ---")
