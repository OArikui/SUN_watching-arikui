"""実機カメラがない環境で動画を入力として使うダミーカメラを提供する。"""

import struct
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import BinaryIO

import cv2
import numpy as np

try:
    import zwoasi as asi  # pyright: ignore[reportMissingImports]

    ZWOASI_AVAILABLE = True
except ImportError:
    # zwoasiがインストールされていない環境でもダミーカメラが動くよう、定数だけのスタブを定義
    # (ダミーを使うかどうかの判定には使わない。判定は run.py が行う)
    ZWOASI_AVAILABLE = False

    class DummyASI:
        """zwoasi がない環境で使う、定数のみのスタブ。"""

        ZWO_CaptureError = Exception
        ZWO_Error = Exception
        ASI_GAIN = 1
        ASI_EXPOSURE = 2
        ASI_TEMPERATURE = 3
        ASI_OFFSET = 4
        ASI_GAMMA = 5
        ASI_BANDWIDTHOVERLOAD = 6
        ASI_HIGH_SPEED_MODE = 7
        ASI_HARDWARE_BIN = 8
        ASI_FLIP = 9
        ASI_AUTO_MAX_GAIN = 10
        ASI_AUTO_MAX_EXP = 11
        ASI_AUTO_MAX_BRIGHTNESS = 12
        ASI_TARGET_TEMP = 13
        ASI_COOLER_ON = 14
        ASI_IMG_RAW8 = 0
        ASI_IMG_RGB24 = 1
        ASI_IMG_RAW16 = 2
        ASI_IMG_Y8 = 3

        class Camera:
            """実機カメラクラスの型名を提供するスタブ。"""

    asi = DummyASI()


class SERVideoCapture:
    """SER動画ファイルを OpenCV の VideoCapture と同様の方法で読み込む。

    天体用 SER フォーマット（8bit / 16bit）に対応する独自リーダー。
    """

    def __init__(self, filepath: str | Path) -> None:
        self.filepath = str(filepath)
        self.file: BinaryIO | None = None
        self.is_opened = False
        self.width = 0
        self.height = 0
        self.pixel_depth = 0
        self.frame_count = 0
        self.bpp = 1
        self.frame_size = 0
        self.current_frame = 0
        self.is_little_endian = True

        try:
            # ファイルハンドルはフレームごとのランダムアクセスに必要なため保持する。
            self.file = Path(filepath).open("rb")  # noqa: SIM115
            header = self.file.read(178)
            # SERファイルは178バイトのヘッダを持ち、LUCAM-RECORDERから始まる
            if len(header) < 178 or not header.startswith(b"LUCAM-RECORDER"):
                self.release()
                return

            # エンディアンの取得 (0 = Big Endian, 1 = Little Endian)
            endian_flag = struct.unpack("<i", header[22:26])[0]
            self.is_little_endian = endian_flag == 1

            # 幅、高さ、ピクセル深度、フレーム数の取得
            self.width = struct.unpack("<i", header[26:30])[0]
            self.height = struct.unpack("<i", header[30:34])[0]
            self.pixel_depth = struct.unpack("<i", header[34:38])[0]
            self.frame_count = struct.unpack("<i", header[38:42])[0]

            # 8bit超の場合は1ピクセル2バイト(16bit)として扱う
            self.bpp = 2 if self.pixel_depth > 8 else 1
            self.frame_size = self.width * self.height * self.bpp
            self.current_frame = 0
            self.is_opened = True

        except (OSError, struct.error):
            self.release()

    def isOpened(self) -> bool:
        """ファイルが正常に開かれている場合に True を返す。"""
        return self.is_opened

    def get(self, propId: int) -> float:
        """指定された OpenCV 動画プロパティの値を返す。"""
        if propId == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if propId == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if propId == cv2.CAP_PROP_FPS:
            # SER形式は標準的なFPSを持たないため、仮の値として30.0を返す
            return 30.0
        if propId == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.frame_count)
        return 0.0

    def set(self, propId: int, value: float | int) -> bool:
        """指定された OpenCV 動画プロパティを設定する。"""
        if propId == cv2.CAP_PROP_POS_FRAMES:
            self.current_frame = int(value)
            return True
        return False

    def read(self) -> tuple[bool, np.ndarray | None]:
        """次のフレームを読み込み、成功フラグと画像を返す。"""
        if not self.is_opened or self.file is None or self.current_frame >= self.frame_count:
            return False, None

        self.file.seek(178 + self.current_frame * self.frame_size)
        data = self.file.read(self.frame_size)
        if len(data) != self.frame_size:
            return False, None

        if self.bpp == 1:
            frame = np.frombuffer(data, dtype=np.uint8).reshape((self.height, self.width))
        else:
            # 16bitデータの読み込み
            dtype = "<u2" if self.is_little_endian else ">u2"
            frame = np.frombuffer(data, dtype=dtype).reshape((self.height, self.width))
            if not self.is_little_endian:
                # OpenCV等の処理と互換性を持たせるためネイティブなuint16に変換
                frame = frame.astype(np.uint16)

        self.current_frame += 1
        return True, frame

    def release(self) -> None:
        """開いているファイルを閉じ、オープン状態を解除する。"""
        if self.file is not None:
            self.file.close()
            self.file = None
        self.is_opened = False


class VideoDummyCamera:
    """動画ファイルからフレームを提供し、zwoasi.Camera の動作を模倣する。

    指定された動画ファイル（8bit / 16bit）を入力ソースとして使用する。
    """

    def __init__(self, video_path: str | None = None) -> None:
        """動画ファイルを開き、カメラ互換の内部状態を初期化する。

        Parameters
        ----------
        video_path : str | None
            動画ファイルのパス。未指定の場合はファイル選択ダイアログを表示する。
        """
        if not video_path:
            # tkinterのルートウィンドウを表示しないように隠す
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)  # ダイアログを最前面に表示

            try:
                video_path = filedialog.askopenfilename(
                    title="再生する動画ファイルを選択してください",
                    filetypes=[
                        ("動画ファイル", "*.mp4 *.avi *.mov *.mkv *.wmv *.ser"),
                        ("すべてのファイル", "*.*"),
                    ],
                )
            finally:
                root.destroy()

            if not video_path:
                raise ValueError("動画ファイルが選択されませんでした。")

        self.video_path = video_path

        # 拡張子が .ser の場合は専用のリーダーを使用し、それ以外はOpenCVを使用する
        if video_path.lower().endswith(".ser"):
            self.cap = SERVideoCapture(video_path)
        else:
            self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"動画ファイルを開けませんでした: {video_path}")

        # 動画のプロパティを取得
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30.0

        # 内部状態
        self.bins = 1
        self.image_type = getattr(asi, "ASI_IMG_RAW8", 0)

        # 仮想のコントロール値 (Gain, Exposure, Temp)
        self.controls: dict[int, int | float] = {
            getattr(asi, "ASI_GAIN", 1): 150,
            getattr(asi, "ASI_EXPOSURE", 2): 30000,
            getattr(asi, "ASI_TEMPERATURE", 3): 250,  # 25.0℃
        }
        self.is_capturing = False

    def get_camera_property(self) -> dict[str, object]:
        """zwoasi.Camera.get_camera_property() と互換性のある情報を返す。

        16bit（RAW16）のサポート情報を含む。
        """
        return {
            "Name": "Dummy Video Camera",
            "CameraID": -1,
            "MaxHeight": self.height,
            "MaxWidth": self.width,
            "IsColorCam": True,
            "BayerPattern": 0,
            "SupportedBins": [1, 2, 4],
            "SupportedVideoFormat": [
                getattr(asi, "ASI_IMG_RAW8", 0),
                getattr(asi, "ASI_IMG_RAW16", 2),
            ],
            "PixelSize": 3.75,
            "MechanicalShutter": False,
            "ST4Port": False,
            "IsCoolerCam": False,
            "IsUSB3Host": True,
            "IsUSB3Camera": True,
            "ELEC_PER_ADU": 1.0,
            "BitDepth": 16,
        }

    def get_controls(self) -> dict[int, dict[str, str | int]]:
        """zwoasi.Camera.get_controls() と互換性のある制御情報を返す。

        カメラがサポートするコントロール ID と制御パラメータの辞書を返す。
        """
        # ASIカメラで利用される主要なコントロールIDを取得
        exposure_id = getattr(asi, "ASI_EXPOSURE", 2)
        gain_id = getattr(asi, "ASI_GAIN", 1)
        offset_id = getattr(asi, "ASI_OFFSET", 3)
        bandwidth_id = getattr(asi, "ASI_BANDWIDTHOVERLOAD", 4)
        target_temp_id = getattr(asi, "ASI_TARGET_TEMP", 5)
        cooler_on_id = getattr(asi, "ASI_COOLER_ON", 6)

        # apply_camera_config での 'in' 判定を通過できるように辞書を作成
        return {
            exposure_id: {
                "Name": "Exposure",
                "MinValue": 1,
                "MaxValue": 1000000,
                "DefaultValue": 10000,
            },
            gain_id: {
                "Name": "Gain",
                "MinValue": 0,
                "MaxValue": 600,
                "DefaultValue": 100,
            },
            offset_id: {
                "Name": "Offset",
                "MinValue": 0,
                "MaxValue": 100,
                "DefaultValue": 10,
            },
            bandwidth_id: {
                "Name": "BandwidthOverload",
                "MinValue": 40,
                "MaxValue": 100,
                "DefaultValue": 40,
            },
            target_temp_id: {
                "Name": "TargetTemp",
                "MinValue": -50,
                "MaxValue": 30,
                "DefaultValue": 0,
            },
            cooler_on_id: {
                "Name": "CoolerOn",
                "MinValue": 0,
                "MaxValue": 1,
                "DefaultValue": 0,
            },
        }

    def set_roi_format(
        self,
        width: int | None = None,
        height: int | None = None,
        bins: int = 1,
        image_type: int = 0,
        start_x: int = 0,
        start_y: int = 0,
    ) -> None:
        """ROI サイズ、ビニング、画像タイプなどをダミー値として保存する。

        Parameters
        ----------
        width : int | None
            設定する幅。None の場合は現在値を維持する。
        height : int | None
            設定する高さ。None の場合は現在値を維持する。
        bins : int
            ビニング値。
        image_type : int
            画像フォーマット ID。
        start_x : int
            ROI の開始 X 座標。
        start_y : int
            ROI の開始 Y 座標。
        """
        if width is not None:
            self.width = int(width)
        if height is not None:
            self.height = int(height)

        self.bins = bins
        self.image_type = image_type
        self.start_x = start_x
        self.start_y = start_y

    def start_video_capture(self) -> None:
        """動画キャプチャの開始を模倣する。"""
        self.is_capturing = True

    def stop_video_capture(self) -> None:
        """動画キャプチャの停止を模倣する。"""
        self.is_capturing = False

    def close(self) -> None:
        """動画キャプチャと関連リソースを解放する。"""
        self.stop_video_capture()
        if self.cap.isOpened():
            self.cap.release()

    def get_roi_format(self) -> tuple[int, int, int, int]:
        """現在の ROI フォーマット（width, height, binning, img_type）を返す。"""
        return (self.width, self.height, self.bins, self.image_type)

    def get_control_value(self, control_type: int) -> tuple[int | float, bool]:
        """コントロール値と自動設定フラグのタプルを返す。"""
        val = self.controls.get(control_type, 0)
        return (val, False)

    def set_control_value(
        self, control_type: int, value: int | float, auto: bool = False
    ) -> None:
        """UIからの設定変更を受け付ける。

        Parameters
        ----------
        control_type : int
            制御 ID。
        value : int | float
            設定値。
        auto : bool
            自動設定フラグ（互換性のため受け取る）。
        """
        self.controls[control_type] = value

    def capture_video_frame(self, timeout: int = 500) -> bytes:
        """動画から1フレームを読み、画像タイプに準拠したバイト列を返す。

        Parameters
        ----------
        timeout : int
            zwoasi API との互換性のためのタイムアウト値。

        Returns
        -------
        bytes
            RAW8 または RAW16 形式のフレームデータ。

        Raises
        ------
        asi.ZWO_CaptureError
            キャプチャが開始されていない、またはフレームを読み込めない場合。
        """
        if not self.is_capturing:
            raise getattr(asi, "ZWO_CaptureError", Exception)("Capture not started")

        time.sleep(1.0 / self.fps)

        ret, frame = self.cap.read()
        if not ret or frame is None:
            # 動画終端到達時は最初に戻す（ループ再生）
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise getattr(asi, "ZWO_CaptureError", Exception)(
                    "Failed to read dummy frame"
                )

        # 1. カラー(3チャンネル)の場合はグレースケールに変換
        if frame.ndim == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        raw16_type = getattr(asi, "ASI_IMG_RAW16", 2)

        # 2. 要求されているフォーマット（RAW16 or RAW8）に応じた型変換
        if self.image_type == raw16_type:
            if gray.dtype == np.uint8:
                # 8bit入力 -> 16bit出力へスケールアップ (0-65535)
                gray = gray.astype(np.uint16) * 256
            elif gray.dtype != np.uint16:
                gray = gray.astype(np.uint16)
        else:
            if gray.dtype == np.uint16:
                # 16bit入力 -> 8bit出力へスケールダウン (0-255)
                gray = (gray / 256).astype(np.uint8)
            elif gray.dtype != np.uint8:
                gray = gray.astype(np.uint8)

        return gray.tobytes()
