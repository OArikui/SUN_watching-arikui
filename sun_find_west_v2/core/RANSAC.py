version = 1.00

import logging
import traceback

logger = logging.getLogger(__name__)

if "__main__" == __name__:
    logger.info("--- starting as main process ---")
else:
    logger.info("--- starting as module process ---")

try:
    import math
    import sys

    import matplotlib.pyplot as plt
    import numpy as np
    from numpy.typing import NDArray
    from sklearn.linear_model import RANSACRegressor
except ImportError:
    logger.error("Failed to import standard module")
    logger.error(traceback.format_exc())
    raise

logger.info("standard modules imported successfully")


def calculate_west_angle_robust(
    p_lst: list[list[float]] | NDArray[np.float64],
    time_stomps: list[float] | NDArray[np.float32] | None = None,
) -> tuple[float, tuple[float, float]] | None:
    """時系列の座標リストから、外れ値（ノイズ）に強い移動方向（角度）と速度ベクトルを算出する。

    デカルト座標系（右: 0度、上: 90度、左: 180/-180度、下: -90度）

    Parameters:
        p_lst: [[x1, y1], [x2, y2], ...] のような2次元座標のリストまたはndarray
        time_stomps: 各点のタイムスタンプ（任意）。指定しない場合は等間隔(0, 1, 2...)として扱う

    Returns:
        Optional[Tuple[float, Tuple[float, float]]]:
            - angle_deg (float): 移動方向の角度（-180 ~ 180度）
            - (vy, vx) (Tuple[float, float]): y方向およびx方向の推定速度

    Raises:
        Exception: 与えられた点の数が2つ未満で線を近似できない場合に例外を発生させる。
    """
    points = np.asarray(p_lst, dtype=np.float64)

    # 角度計算には最低2点が必要
    if len(points) < 2:
        raise ValueError("")  # NEXT:エラーメッセージを挿入
        return False, (0.0, 0.0)

    # 時間インデックス t の作成と2次元配列化 (n_samples, 1)
    if time_stomps is None or len(time_stomps) != len(points):
        t: NDArray[np.float64] = np.arange(len(points), dtype=np.float64).reshape(-1, 1)
    else:
        t = np.asarray(time_stomps, dtype=np.float64).reshape(-1, 1)

    # x座標とy座標の分離
    x: NDArray[np.float64] = points[:, 0]
    y: NDArray[np.float64] = points[:, 1]

    # x方向の速度(傾き)をRANSACで推定
    ransac_x = RANSACRegressor(random_state=42)
    ransac_x.fit(t, x)
    # estimator_ は学習済みの LinearRegression インスタンス
    vx: float = float(ransac_x.estimator_.coef_[0])

    # y方向の速度(傾き)をRANSACで推定
    ransac_y = RANSACRegressor(random_state=42)
    ransac_y.fit(t, y)
    vy: float = float(ransac_y.estimator_.coef_[0])

    # ベクトルからラジアン・度数法へ変換
    angle_rad: float = math.atan2(vy, vx)
    angle_deg: float = math.degrees(angle_rad)

    vectorYX = vy, vx
    return -1*angle_deg, vectorYX


# --- テスト実行と描画 ---
if __name__ == "__main__":
    # 理想的な直線移動データ (右斜め上 45度方向) に、極端なノイズを混ぜたもの
    noisy_trajectory = [
        [0.0, 0.0],
        [1.1, 0.9],
        [2.0, 2.1],
        [10.0, -5.0],  # 外れ値1
        [4.2, 3.8],
        [4.9, 5.1],
        [-3.0, 12.0],  # 外れ値2
        [7.0, 7.1],
    ]

    points = np.array(noisy_trajectory)
    x_vals = points[:, 0]
    y_vals = points[:, 1]

    # 1. 単純な始点と終点の計算
    dx_simple = x_vals[-1] - x_vals[0]
    dy_simple = y_vals[-1] - y_vals[0]
    simple_angle = math.degrees(math.atan2(dy_simple, dx_simple))

    # 2. RANSACによるロバストな計算
    result = calculate_west_angle_robust(noisy_trajectory)

    if result is None:
        print("There is insufficient data.")
        sys.exit()
    else:
        robust_angle, (vy, vx) = result
        print(f"The perspective of a simple calculation: {simple_angle:.2f} degree")
        print(f"Robust Angle Estimation via RANSAC: {robust_angle:.2f} degree")

    # --- matplotlib による描画処理 ---
    plt.figure(figsize=(8, 8))

    # データの散布図と時系列の軌跡プロット
    plt.plot(
        x_vals, y_vals, color="gray", linestyle="--", alpha=0.5, label="Trajectory line"
    )
    plt.scatter(
        x_vals, y_vals, color="blue", zorder=5, label="Data points (with Noise)"
    )

    # 始点(Start)と終点(End)を強調
    plt.scatter(
        x_vals[0], y_vals[0], color="green", s=150, zorder=6, label="Start point"
    )
    plt.scatter(x_vals[-1], y_vals[-1], color="red", s=150, zorder=6, label="End point")

    # 矢印を描画するための基準点（データの中心付近）
    center_x = np.median(x_vals)
    center_y = np.median(y_vals)

    # 単純計算の方向ベクトルを矢印で描画
    # エラー回避のため linestyle ではなく、widthを細くし、alphaを薄くして表現します
    len_simple = math.hypot(dx_simple, dy_simple)
    plt.quiver(
        center_x,
        center_y,
        (dx_simple / len_simple) * 3,
        (dy_simple / len_simple) * 3,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="red",
        width=0.004,
        alpha=0.5,
        label=f"Simple Vector ({simple_angle:.1f}°)",
    )

    # RANSACで推定した方向ベクトルを矢印で描画 (青太線)
    try:
        len_robust = math.hypot(
            vx,
            vy,
        )
        plt.quiver(
            center_x,
            center_y,
            (vx / len_robust) * 3,
            (vy / len_robust) * 3,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="darkblue",
            width=0.008,
            label=f"RANSAC Vector ({robust_angle:.1f}°)",
        )
    except NameError:
        # robust_angleが定義されていない場合（データ不足など）
        pass
    # グラフの設定
    plt.title("Comparison of Direction Vector Estimation", fontsize=14)
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left")

    # 縦横比を1:1にして角度を正確に見せる
    plt.gca().set_aspect("equal", adjustable="box")

    # 表示
    plt.show()

logger.info("--- finish ---")
