import struct
import time
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np

try:
    import zwoasi as asi
except ImportError:
    # zwoasiがインストールされていない環境でも動くようダミーを定義
    class DummyASI:
        dum = True
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
            pass

    asi = DummyASI()


class SERVideoCapture:
    """
    天体用 .ser フォーマット（8bit / 16bit）を cv2.VideoCapture と
    同じインターフェースで読み込むための独自リーダークラス。
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = None
        self.is_opened = False
        
        try:
            self.file = open(filepath, 'rb')
            header = self.file.read(178)
            # SERファイルは178バイトのヘッダを持ち、LUCAM-RECORDERから始まる
            if len(header) < 178 or not header.startswith(b'LUCAM-RECORDER'):
                return
                
            # エンディアンの取得 (0 = Big Endian, 1 = Little Endian)
            endian_flag = struct.unpack('<i', header[22:26])[0]
            self.is_little_endian = (endian_flag == 1)
            
            # 幅、高さ、ピクセル深度、フレーム数の取得
            self.width = struct.unpack('<i', header[26:30])[0]
            self.height = struct.unpack('<i', header[30:34])[0]
            self.pixel_depth = struct.unpack('<i', header[34:38])[0]
            self.frame_count = struct.unpack('<i', header[38:42])[0]
            
            # 8bit超の場合は1ピクセル2バイト(16bit)として扱う
            self.bpp = 2 if self.pixel_depth > 8 else 1
            self.frame_size = self.width * self.height * self.bpp
            
            self.current_frame = 0
            self.is_opened = True
            
        except Exception:
            if self.file:
                self.file.close()
            self.is_opened = False

    def isOpened(self):
        return self.is_opened

    def get(self, propId):
        if propId == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        elif propId == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        elif propId == cv2.CAP_PROP_FPS:
            return 30.0  # SER形式は標準的なFPSを持たないため、仮の値として30.0を返す
        elif propId == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.frame_count)
        return 0.0

    def set(self, propId, value):
        if propId == cv2.CAP_PROP_POS_FRAMES:
            self.current_frame = int(value)
            return True
        return False

    def read(self):
        if not self.is_opened or self.current_frame >= self.frame_count:
            return False, None
            
        self.file.seek(178 + self.current_frame * self.frame_size)
        data = self.file.read(self.frame_size)
        
        if len(data) != self.frame_size:
            return False, None
            
        if self.bpp == 1:
            frame = np.frombuffer(data, dtype=np.uint8).reshape((self.height, self.width))
        else:
            # 16bitデータの読み込み
            dtype = '<u2' if self.is_little_endian else '>u2'
            frame = np.frombuffer(data, dtype=dtype).reshape((self.height, self.width))
            if not self.is_little_endian:
                # OpenCV等の処理と互換性を持たせるためネイティブなuint16に変換
                frame = frame.astype(np.uint16)
                 
        self.current_frame += 1
        return True, frame

    def release(self):
        if self.file:
            self.file.close()
            self.is_opened = False


class VideoDummyCamera:
    """
    zwoasi.Cameraクラスの動作を模倣し、指定された動画ファイル（8bit / 16bit）から
    フレームを提供するダミークラス。
    """

    def __init__(self, video_path: str | None = None) -> None:
        """
        Args:
            video_path (str, optional): 動画ファイルのパス。
            指定がない場合はファイルダイアログを表示して選択させます。
        """
        if not video_path:
            # tkinterのルートウィンドウを表示しないように隠す
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)  # ダイアログを最前面に表示

            video_path = filedialog.askopenfilename(
                title="再生する動画ファイルを選択してください",
                filetypes=[
                    ("動画ファイル", "*.mp4 *.avi *.mov *.mkv *.wmv *.ser"),
                    ("すべてのファイル", "*.*"),
                ],
            )
            root.destroy()

            if not video_path:
                raise ValueError("動画ファイルが選択されませんでした。")

        self.video_path = video_path
        
        # 拡張子が .ser の場合は専用のリーダーを使用し、それ以外はOpenCVを使用する
        if video_path.lower().endswith('.ser'):
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
        self.controls = {
            getattr(asi, "ASI_GAIN", 1): 150,
            getattr(asi, "ASI_EXPOSURE", 2): 30000,
            getattr(asi, "ASI_TEMPERATURE", 3): 250,  # 25.0℃
        }
        self.is_capturing = False

    def get_camera_property(self) -> dict:
        """
        zwoasi.Camera.get_camera_property() の互換メソッド。
        16bit(RAW16)のサポート情報を返します。
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

    def get_controls(self):
        """
        zwoasi.Camera.get_controls() の互換メソッド。
        カメラがサポートするコントロール情報 (コントロールID -> 制御パラメータ情報の辞書) を返します。
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
        self, width=None, height=None, bins=1, image_type=0, start_x=0, start_y=0
    ):
        """
        zwoasi.Camera.set_roi_format() の互換メソッド。

        ROI（関心領域）サイズやビニング、画像タイプをダミー値として保存します。
        """
        if width is not None:
            self.width = int(width)
        if height is not None:
            self.height = int(height)

        self.bins = bins
        self.image_type = image_type
        self.start_x = start_x
        self.start_y = start_y

    def start_video_capture(self):
        """動画キャプチャの開始を模倣"""
        self.is_capturing = True

    def stop_video_capture(self):
        """動画キャプチャの停止を模倣"""
        self.is_capturing = False

    def close(self):
        """リソースの解放"""
        self.stop_video_capture()
        if self.cap.isOpened():
            self.cap.release()

    def get_roi_format(self):
        """現在のROIフォーマット(width, height, binning, img_type)を返します。"""
        return (self.width, self.height, self.bins, self.image_type)

    def get_control_value(self, control_type):
        """コントロール値と自動設定フラグ(bool)のタプルを返す"""
        val = self.controls.get(control_type, 0)
        return (val, False)

    def set_control_value(self, control_type, value, auto=False):
        """UIからの設定変更を受け付けるダミーメソッド"""
        self.controls[control_type] = value

    def capture_video_frame(self, timeout=500):
        """
        動画から1フレーム読み込み、`image_type` に準拠した画像（RAW8 / RAW16）のバイト列を返します。
        """
        if not self.is_capturing:
            raise getattr(asi, "ZWO_CaptureError", Exception)("Capture not started")

        time.sleep(1.0 / self.fps)

        ret, frame = self.cap.read()
        if not ret:
            # 動画終端到達時は最初に戻す（ループ再生）
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
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