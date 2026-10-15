import logging
import traceback

logger = logging.getLogger(__name__)

if "__main__" == __name__:
    logger.info("--- starting as main process ---")
else:
    logger.info("--- starting as module process ---")
try:
    import time
    from collections.abc import Callable

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Arc, Circle, Polygon
    from matplotlib.widgets import Button, Slider
except ImportError:
    logger.error("Failed to import standard module")
    logger.error(traceback.format_exc())
    raise


__all__ = ["OpenCircleArrow", "Visualizer"]


def convert_angle_to_west(robust_angle: float) -> float:
    """標準座標系（右0度, 上90度）の角度を西座標系（左0度, 上90度）に変換し、

    [-180, 180] の範囲に正規化します。

    Args:
        robust_angle (float): 変換前の角度（度数法 / degree）

    Returns:
        float: UI用の角度（度数法 / degree, 範囲: -180 ~ 180）
    """
    raw_ui_angle = 180.0 - robust_angle  # 座標系の変換
    ui_angle = (raw_ui_angle + 180.0) % 360.0 - 180.0  # 正規化

    return ui_angle


# 1. OpenCircleArrow クラス (描画パーツ)
class OpenCircleArrow:
    def __init__(
        self,
        ax,
        center=(0, 0),
        radius=1.0,
        angle=90,
        edgecolor="C0",
        lw=3,
        tri_size=0.12,
        tri_color="C0",
        show_triangle=True,  # 矢じりの表示/非表示を制御するフラグを追加
    ):
        """
        インタラクティブにパラメーターを更新できる矢印付き円弧オブジェクト
        """
        self.ax = ax
        self.center = center
        self.radius = radius
        self.angle = angle
        self.edgecolor = edgecolor
        self.lw = lw
        self.tri_size = tri_size
        self.tri_color = tri_color
        self.show_triangle = show_triangle
        
        # 描画したパッチを保持する変数
        self.arc_patch = None
        self.tri_patch = None

        # 初回描画
        self.draw()

    def draw(self):
        """現在保持しているパラメーターで再描画を行う内部メソッド"""
        # すでに描画されている古いパッチがあれば削除する
        if self.arc_patch is not None:
            self.arc_patch.remove()
            self.arc_patch = None
        if self.tri_patch is not None:
            self.tri_patch.remove()
            self.tri_patch = None

        cx, cy = self.center

        # 尻を右(0度)に固定し、正負で矢じりの向きを反転
        if self.angle >= 0:
            # 正の角度（反時計回り）
            arc_t1 = 180
            arc_t2 = -1 * self.angle
            tangent_angle = self.angle + 90
            tip_angle = self.angle
        else:
            # 負の角度（時計回り）
            arc_t1 = -1 * self.angle
            arc_t2 = 180
            tangent_angle = self.angle - 90
            tip_angle = self.angle

        # 円弧の生成
        self.arc_patch = Arc(
            (cx, cy),
            2 * self.radius,
            2 * self.radius,
            angle=0,
            theta1=arc_t1,
            theta2=arc_t2,
            linewidth=self.lw,
            edgecolor=self.edgecolor,
            linestyle="solid",
        )
        self.ax.add_patch(self.arc_patch)

        # 矢じりの計算と生成 (show_triangleがTrueのときのみ描画)
        if self.show_triangle:
            tip_rad = np.deg2rad(tip_angle)
            ex = cx + self.radius * np.cos(tip_rad)
            ey = cy + self.radius * np.sin(tip_rad)

            t_rad = np.deg2rad(tangent_angle)

            s = self.tri_size * self.radius
            tri = np.array([[0.0, 0.0], [-s, s / 2], [-s, -s / 2]])

            R = np.array([[np.cos(t_rad), -np.sin(t_rad)], [np.sin(t_rad), np.cos(t_rad)]])
            tri_rot = (tri @ R.T) + np.array([ex, ey])

            # 矢じり（多角形）の生成
            self.tri_patch = Polygon(tri_rot, closed=True, color=self.tri_color)
            self.ax.add_patch(self.tri_patch)

        # 画面の更新を促す
        if self.ax.figure and self.ax.figure.canvas:
            self.ax.figure.canvas.draw_idle()

    def update(self, **kwargs):
        """
        外部から変数を更新するためのメソッド。
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.draw()


# 2. Visualizer クラス (メイン描画マネージャー)
class Visualizer:
    def __init__(
        self,
        width,
        height,
        acceptable=1,
        grid_color="#FFFFFF",
        grid_ny=2,
        grid_nx=4,
        grid_r=300,
        grid_alpha=0.4,
        opencircle_triangle=False
    ):
        self.width = width
        self.height = height
        self.acceptable = acceptable
        self.sunline = (min(width, height) * 2 / 4) / 2

        # 描画の初期化
        plt.ion()
        self.fig, self.ax = plt.subplots()

        # ダミー画像で領域を確保
        dummy_img = np.zeros((height, width), dtype=np.uint8)
        self.ax_img = self.ax.imshow(dummy_img, cmap="gray", vmin=0, vmax=255)

        # 各種プロットオブジェクトの初期設定
        self.circle = Circle((0, 0), 0, fill=True, color="skyblue", linewidth=2)
        self.ax_min2 = self.ax.add_patch(self.circle)

        (self.ax_shdw_c,) = self.ax.plot(
            [],
            [],
            "o",
            color="cyan",
            markersize=3,
            alpha=0.4,
            label="circle center track",
        )

        # グリッドの描画
        self.grid_lines = []

        # 縦分割数 (grid_ny) に応じて横線を描画
        for y_val in np.linspace(0, height, grid_ny + 1)[1:-1]:
            (line,) = self.ax.plot(
                [0, width],
                [y_val, y_val],
                color=grid_color,
                linewidth=3,
                alpha=grid_alpha,
            )
            self.grid_lines.append(line)

        # 横分割数 (grid_nx) に応じて縦線を描画
        for x_val in np.linspace(0, width, grid_nx + 1)[1:-1]:
            (line,) = self.ax.plot(
                [x_val, x_val],
                [0, height],
                color=grid_color,
                linewidth=3,
                alpha=grid_alpha,
            )
            self.grid_lines.append(line)

        # 画像中心に半径 grid_r の正円を描画
        center_x, center_y = width / 2, height / 2
        self.grid_circle = Circle(
            (center_x, center_y),
            grid_r,
            fill=False,
            edgecolor=grid_color,
            linewidth=3,
            alpha=grid_alpha,
        )
        self.ax.add_patch(self.grid_circle)

        # (self.ax_sunline,) = self.ax.plot([], [], color="purple", linewidth=3)

        # 矢印を描画するための quiver オブジェクトを作成（0で1件分の領域を確保）
        self.ax_sunline = self.ax.quiver(
            0,
            0,
            0,
            0,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="purple",
            width=0.008,
            pivot="tail",
            zorder=10,
            linestyle="dashed",
        )

        # HUD（表示パネル）のカスタマイズ用設定辞書
        self.hud_style = {
            "x": 0.02,  # 画面左端からの位置 (0.0 ~ 1.0)
            "y": 0.98,  # 画面下端からの位置 (0.0 ~ 1.0)
            "ha": "left",  # 水平方向の揃え (left, center, right)
            "va": "top",  # 垂直方向の揃え (top, center, bottom)
            "fontsize": 13,  # 文字サイズ
            "color": "#00FF00",  # 文字色
            "family": "monospace",  # フォントスタイル (等幅フォントを推奨)
            "bbox": {  # 背景パネルの設定
                "facecolor": "black",
                "alpha": 0.5,
                "edgecolor": "none",
                "boxstyle": "round,pad=0.5",
            },
        }

        self.info_text = self.fig.text(
            self.hud_style["x"],
            self.hud_style["y"],
            "",
            ha=self.hud_style["ha"],
            va=self.hud_style["va"],
            fontsize=self.hud_style["fontsize"],
            color=self.hud_style["color"],
            family=self.hud_style["family"],
            bbox=self.hud_style["bbox"],
        )

        # FPS計測用の変数
        self.prev_time = time.time()

        # 同ファイル内に定義した OpenCircleArrow を直接使用
        self.arrow = OpenCircleArrow(
            self.ax,
            center=(0, 0.5),
            radius=100,
            angle=0,  # 初期角度
            edgecolor="purple",
            tri_color="purple",
            show_triangle=opencircle_triangle
        )

        self.sliders = {}
        self.buttons = {}

    def add_button(
        self,
        name: str,
        label: str,
        on_clicked: Callable,
        position: tuple[float, float, float, float] | None = None,
    ) -> Button:
        """ボタンを追加する"""
        if position is None:
            position = (0.02, 0.05, 0.07, 0.04)

        ax_button = self.fig.add_axes(position)
        button = Button(ax_button, label)

        button.on_clicked(lambda event: on_clicked())

        self.buttons[name] = button
        return button

    def add_slider(
        self,
        name: str,
        valmin: float,
        valmax: float,
        valinit: float,
        on_change: Callable | None = None,
        label: str | None = None,
        valfmt: str | None = None,  # 表示フォーマット (例: "%1.0f", "%1.2f")
    ) -> Slider:

        num_sliders = len(self.sliders)

        bottom_margin = 0.20 + (num_sliders + 1) * 0.05
        self.fig.subplots_adjust(bottom=bottom_margin)
        y_pos = 0.12 + (num_sliders * 0.05)
        ax_slider = self.fig.add_axes((0.25, y_pos, 0.6, 0.03))

        user_label = label if label is not None else name
        kwargs = {}
        if valfmt is not None:
            kwargs["valfmt"] = valfmt

        slider = Slider(
            ax_slider,
            user_label,
            valmin,
            valmax,
            valinit=valinit,
            **kwargs,  # type: ignore[arg-type]
        )

        if on_change is not None:
            slider.on_changed(on_change)

        self.sliders[name] = slider
        return slider

    def get_slider_val(self, name: str) -> float:
        """指定した名前のスライダーの現在の値を取得する"""
        if name in self.sliders:
            return self.sliders[name].val
        raise KeyError(f"__ENGSlider '{name}' は存在しません。")

    def set_slider_val(self, name: str, val: float) -> None:
        """指定した名前のスライダーの値をプログラムから更新する"""
        if name in self.sliders:
            self.sliders[name].set_val(val)
        else:
            raise KeyError(f"__ENGSlider '{name}' は存在しません。")

    def update(
        self,
        img,
        cx,
        cy,
        r,
        recent_pts,
        robust_angle,
        robust_vector,
        frame_idx=None,
        total_frames=None,
    ):
        """計算結果を受け取り、画面を更新する"""
        self.ax_img.set_data(img)
        self.ax_min2.set_center((cx, cy))  # pyright: ignore[reportAttributeAccessIssue]
        self.ax_min2.set_radius(r)  # pyright: ignore[reportAttributeAccessIssue]
        self.ax_min2.set_alpha(0.4)  # pyright: ignore[reportAttributeAccessIssue]

        self.ax_shdw_c.set_data(recent_pts[:, 0], recent_pts[:, 1])

        if robust_angle is None:
            # [FIX] データ不足などで角度が未算出の場合。
            # 矢印は直前の角度のまま灰色にして「未算出」であることを示し、
            # convert_angle_to_west(None) を呼んで落ちないようにする。
            west_angle = None
            uxc = ("dimgray", "gray")
            self.ax_sunline.set_offsets(np.c_[cx, cy])
            self.ax_sunline.set_UVC(0, 0)
            self.arrow.update(
                center=(cx, cy),
                edgecolor=uxc[1],
                tri_color=uxc[1],
            )
            self.ax_sunline.set_color(uxc[1])
        else:
            # 2つの座標系、いずれも上が正,下が負で-180~+180
            west_angle = convert_angle_to_west(robust_angle)  # 左0°の座標

            # 許容範囲に応じて色を変更
            if abs(west_angle) < self.acceptable:
                uxc = ("limegreen", "mediumseagreen")
            else:
                uxc = ("red", "purple")

            vectory,vectorx = robust_vector
            u = self.sunline * vectorx
            v = self.sunline * vectory
            
            # 矢印の始点(cx, cy)とベクトル成分(u, v)を更新
            self.ax_sunline.set_offsets(np.c_[cx, cy])
            self.ax_sunline.set_UVC(u, v)

            self.arrow.update(
                center=(cx, cy),
                angle=robust_angle,
                edgecolor=uxc[1],
                tri_color=uxc[1],
            )

            self.ax_sunline.set_color(uxc[1])

        # 実際のFPSを計算
        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now
        actual_fps = 1.0 / dt if dt > 0 else 0.0

        # [FIX] west_angle が None (未算出) の場合はN/A表示にする
        angle_text = f"{west_angle:7.2f} °" if west_angle is not None else "    N/A"

        # 表示テキストのフォーマット (桁数を揃えて視認性を向上)
        text_lines = [
            f"Angle      : {angle_text}",
            f"MIN2 Stat  : X={cx:.1f} Y={cy:.1f} R={r:.1f}",
            f"Frames     : {frame_idx} / {total_frames}",
            f"Traj Points: {len(recent_pts)}",
            f"Actual FPS : {actual_fps:7.2f}",
        ]
        self.info_text.set_text("\n".join(text_lines))
        # 描画を反映
        plt.pause(0.001)

    def is_alive(self):
        """ウィンドウが閉じられていないか判定する"""
        return plt.fignum_exists(
            self.fig.number  # pyright: ignore[reportAttributeAccessIssue]
        )

    def close(self):
        """描画リソースを安全に閉じる"""
        plt.close(self.fig)


logger.info("--- finish ---")

# テスト・デモ実行用
if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path
    from tkinter.filedialog import askopenfile

    demo_mode = "2"
    if demo_mode == "1":
        # このスクリプトを直接実行した場合、スライダー付きの矢印デモが動作します
        fig, ax = plt.subplots(figsize=(5, 6))
        plt.subplots_adjust(bottom=0.25)

        arrow = OpenCircleArrow(ax, center=(0, 0), radius=1.0, angle=45)

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")

        ax_angle = plt.axes(
            [0.2, 0.14, 0.6, 0.03]  # pyright: ignore[reportArgumentType]
        )
        ax_radius = plt.axes(
            [0.2, 0.09, 0.6, 0.03]  # pyright: ignore[reportArgumentType]
        )

        slider_angle = Slider(ax_angle, "Angle", -180, 180, valinit=45)
        slider_radius = Slider(ax_radius, "Radius", 0.1, 1.4, valinit=1.0)

        def handle_update(val):
            arrow.update(
                angle=slider_angle.val,
                radius=slider_radius.val,
            )

        slider_angle.on_changed(handle_update)
        slider_radius.on_changed(handle_update)

        plt.show()
    else:
        img_shape = (1608, 1104)
        radius = 300
        acceptable = 1
        fps = 60

        footsteps = []  # 太陽位置の時系列データ [(cx1,cy1),(cxx2,cy2),(cx3,cy3)...]
        footstep_mode = "2"  # "footsteps? existing(0)/console(1)/txt(2)/csv(3)"
        if footstep_mode == "0":
            if len(footsteps) == 0:
                print("No existing footstep")
            else:
                pass
        elif footstep_mode in ["1", "2"]:
            if footstep_mode == "1":
                print(
                    "Please enter text (press Ctrl+D [Mac/Linux] or Ctrl+Z [Windows] to finish).）:"
                )
                # すべての入力を一括で取得
                input_footsteps = sys.stdin.read().replace("^Z", "").strip()
            elif footstep_mode == "2":
                file_path = askopenfile(mode="r", filetypes=[("Text files", "*.txt")])
                if file_path is None:
                    print("No file was selected.")
                    sys.exit(1)
                input_footsteps = file_path.read().strip()
            try:
                footsteps = [
                    (
                        float(tp[1:-1].replace(" ", "").split(",")[0]),
                        float(tp[1:-1].replace(" ", "").split(",")[1]),
                    )
                    for tp in input_footsteps.split(  # pyright: ignore[reportPossiblyUnboundVariable]
                        "\n"
                    )
                ]
            except (ValueError, IndexError) as e:
                print(f"format error: {e}")
        else:
            print("your typo or yet")

        if footsteps:
            # 階層エラー対策 (find_west.py に準拠したインポート設定)
            current_dir = Path(__file__).resolve().parent
            project_root = current_dir.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            try:
                from RANSAC import calculate_west_angle_robust as west_angle
            except ImportError:
                try:
                    from RANSAC import calculate_west_angle_robust as west_angle
                except ImportError:
                    print("Error: RANSAC module not found.")
                    sys.exit(1)

            width, height = img_shape
            black_img = np.zeros((height, width), dtype=np.uint8)

            # Numpy配列化
            pts = np.array(footsteps)

            # 初めに一度だけRANSACで基準の角度を計算 
            print("The reference angle is calculated from the initial trajectory data using RANSAC...")
            west_re = west_angle(pts)
            if west_re is not None:
                base_calculate, vectorYX = west_re
            else:
                print("There is too little data.")
                sys.exit(1)

            # UI確認のため Visualizer を初期化
            # NOTE: グリッドの分割数などを変えたい場合は以下のように引数を指定します。
            # viz = Visualizer(width, height, acceptable, grid_ny=4, grid_nx=6, grid_r=400, grid_alpha=0.5)
            viz = Visualizer(width, height, acceptable)

            # 描画範囲を画像サイズに固定
            viz.ax.set_xlim(0, width)
            viz.ax.set_ylim(height, 0)

            # スライダー用の余白を画面下部に作成し、スライダーを配置
            plt.subplots_adjust(bottom=0.2)
            ax_slider = viz.fig.add_axes(
                [
                    0.2,
                    0.05,
                    0.6,
                    0.03,
                ]  # pyright: ignore[reportArgumentType, reportCallIssue]
            )
            rot_slider = Slider(ax_slider, "Rotation", -180, 180, valinit=0)

            # 回転の基準となる画像中心
            cx0, cy0 = width / 2, height / 2

            frame_idx = 0
            num_frames = len(footsteps)

            print(
                f"Starting the demo (FPS: {fps}). Close the graph window or press Ctrl+C to exit."
            )

            # アニメーションループ
            while viz.is_alive():
                start_time = time.time()

                # スライダーの値を取得
                angle_deg = rot_slider.val
                angle_rad = np.radians(angle_deg)

                # footsteps を画像中心を軸に回転
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                x = pts[:, 0] - cx0
                y = pts[:, 1] - cy0
                rx = x * cos_a - y * sin_a + cx0
                ry = x * sin_a + y * cos_a + cy0
                rotated_pts = np.column_stack((rx, ry))

                # 現在のフレームで描画する軌跡 (最大直近100件程度)
                start_idx = max(0, frame_idx - 100)
                recent_pts = rotated_pts[start_idx : frame_idx + 1]

                if len(recent_pts) > 0:
                    cx, cy = recent_pts[-1]

                    # 角度は「初期計算値 + スライダーの回転量」で決定

                    robust_angle = base_calculate + angle_deg
                    rad=np.radians(robust_angle)
                    vecx=np.cos(rad)
                    vecy=np.sin(rad)
                    vector=vecx,vecy
                    # 描画更新（frame_idx を渡すように変更）
                    viz.update(
                        black_img,
                        cx,
                        cy,
                        radius,
                        recent_pts,
                        robust_angle,
                        robust_vector=vector,
                        frame_idx=frame_idx,
                    )

                # 次のフレームへ進める
                frame_idx = (frame_idx + 1) % num_frames

                # 指定された fps に合わせた待機処理
                elapsed = time.time() - start_time
                sleep_time = max(0.001, (1.0 / fps) - elapsed)
                plt.pause(sleep_time)
