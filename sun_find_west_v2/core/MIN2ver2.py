import os
import pathlib
from pprint import pformat

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.patches import Circle

"""
最小二乗法による円の検出を行う関数
main: MIN2_ignore_sunspots()

一度検出した近似円の内側にある点のうち、近似円の外側にある点だけから近似した円からlimbwigth*(3/2)の
範囲にないもんは黒点とみなします。
"""
version = "MIN2 v2.3.8"  #fix:popによるindexの変化を阻止

def cut_and_sampling(
    img_inst: str | np.ndarray, sun_threshold: float
) -> list[list[int]]:
    """画像を分割線で走査し、輝度の変化（微分値）から太陽の縁（エッジ）に相当する点の座標をサンプリングする。

    Args:
        img_inst (Union[str, np.ndarray]): 画像の指示。"GLOBAL"ならglobal変数のimgを取得する。もしくは読み込んだ画像。
        sun_threshold (Union[int, float]): 太陽像とみなす明るさのしきい値。これ以下の直線は処理をスキップする。

    Returns:
        List[List[int]]: サンプリングされた縁の座標 [x, y] のリスト。
    """
    if isinstance(img_inst, str):
        if img_inst == "GLOBAL":
            img = globals().get("img")
            if img is None:
                raise ValueError(
                    "画像が指定されていません。imgを渡すか、グローバル変数imgを設定してください。"
                )
        else:
            raise ValueError(f"Unknown instructions = {img_inst}")
    else:
        img = img_inst

    # 画像を分割して実際の縁の点を収集
    spots: list[list[int]] = []  # 実際の縁の点を格納するための配列
    for line_xy in ("x_line", "y_line"):  # x_lineは横線、y_lineは縦線
        for i in range(1, divnum):
            place = (
                height * i // divnum if line_xy == "x_line" else width * i // divnum
            )  # 分割線の位置を計算
            line = (
                img[place, :].astype(float)
                if line_xy == "x_line"
                else img[:, place].astype(float)
            )  # 分割線に沿った画素値を取得
            if np.max(line) <= sun_threshold:  # 太陽像上を通るか
                continue
            grad_t = np.diff(line)  # 一回微分
            max_idx = int(np.argmax(grad_t))  # 最大値のインデックス
            min_idx = int(np.argmin(grad_t))  # 最小値のインデックス
            if line_xy == "x_line":
                spots.append([max_idx, place])
                spots.append([min_idx, place])
            elif line_xy == "y_line":
                spots.append([place, max_idx])
                spots.append([place, min_idx])
    return spots  # 縁の点の座標を返す


def fit_circle(spots: list[list[int]] | np.ndarray, show: bool = False) -> list[float]:
    """与えられた縁の点の座標群から、最小二乗法を用いて近似円の中心座標と半径を計算する。

    Args:
        spots (Union[List[List[int]], np.ndarray]): 縁の点の座標 [x, y] を格納した二次元配列、またはNumPy配列。
        show (bool):例外発生時にshow_circleによる描画を行うか

    Raises:
        Exception: 与えられた座標が3点未満で円を確定できない場合に例外を発生させる。

    Returns:
        List[float]: 近似円の中心X座標、中心Y座標、半径を含むリスト [cx, cy, R]。
    """
    if len(spots) < 3:
        print("[ERROR]:点が3点未満のため、円を作成できません。too little spots")
        if show:
            show_circle(img_inst="GLOBAL", spots=spots, cir_stat=False)
        raise ValueError("点不足")
    x, y = np.array([s[0] for s in spots], dtype=float), np.array(
        [s[1] for s in spots], dtype=float
    )
    mat_A = np.c_[x, y, np.ones(len(x))]
    vec_B = -(x**2 + y**2)
    res, _, _, _ = np.linalg.lstsq(mat_A, vec_B, rcond=None)
    A, B, C = res
    cx = -A / 2
    cy = -B / 2
    R = np.sqrt(cx**2 + cy**2 - C)
    return [cx, cy, R]


def show_circle(
    img_inst: str | np.ndarray,
    spots: list[list[int]] | np.ndarray | None = None,
    cir_stat: tuple[float, float, float] | list[float] | bool = False,
    img_path: pathlib.Path | None = None,
    fig_info: dict[str, str] | None = None,
    iteration_count: int | str = 1,
    is_last: bool = False,
    simple: bool = False
) -> None:
    """画像上に分割線、サンプリングされた縁の点、およびフィッティングされた近似円を描画し、各エッジ点付近の明るさと微分の2軸グラフを右側に並べて表示します。

    Args:
        img_inst (Union[str, np.ndarray]): 画像の指示。"GLOBAL"ならglobal変数のimgを取得し、"PATH"ならimg_pathから画像を読み込む。もしくは読み込んだ画像。
        spots (Optional[List[List[int]]]): 描画する縁の点の座標リスト。デフォルトは None です。
        cir_stat (Union[Tuple[float, float, float], List[float], bool]): 近似円の情報 [cx, cy, R]。描画しない場合は False を指定します。デフォルトは False です。
        img_path (pathlib.path): 画像のファイルパス。デフォルトはNoneです。
        fig_info (Optional[Dict[str, str]]): 画像内にテキストとして表示するメタデータ。デフォルトは None です。
        iteration_count (Union[int, str]): 現在の反復回数。デフォルトは 1 です。
        is_last (bool): 最後の処理かどうかを示すフラグ。True の場合は反復回数の代わりに "Last" と表示します。デフォルトは False です。
        simple (bool): シンプルな表示(ポスター図用)を作成するフラグ。 デフォルトはFalse.
    Returns:
        None: 戻り値はありません（画像をウィンドウに表示します）。
    """

    if simple:
        show_circle_simple(
                    img_inst=img_inst,
                    spots=spots,
                    cir_stat=cir_stat,
                    img_path=img_path,
                    iteration_count=iteration_count,
                    is_last=is_last
                )
        return

    if isinstance(img_inst, str):
        if img_inst == "GLOBAL":
            img = globals().get("img")
            if img is None:
                raise ValueError(
                    "画像が指定されていません。imgを渡す、グローバル変数imgを設定する、または画像のパスを引数に追加してください。"
                )

        elif img_inst == "PATH":
            if img_path != None:
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError(
                        "画像が指定されていません。imgを渡す、グローバル変数imgを設定する、または画像のパスを引数に追加してください。"
                    )
            else:
                raise ValueError(
                    "画像が指定されていません。imgを渡す、グローバル変数imgを設定する、または画像のパスを引数に追加してください。"
                )

        else:
            raise ValueError(f"Unknown instructions = {img_inst}")
    else:
        img = img_inst

    if img_path is not None:
        str(img_path)
    else:
        pass
    # デフォルト引数のミュータブル回避
    if spots is None:
        spots = []

    # 点がない場合は画像のみ表示
    if len(spots) == 0:
        fig, ax = plt.subplots()
        if img is not None:
            ax.imshow(img, cmap="magma")
        if not isinstance(cir_stat, bool) and cir_stat is not None:
            cx, cy, R = cir_stat[0], cir_stat[1], cir_stat[2]
            circle = Circle((cx, cy), R, fill=False, color="orange", linewidth=2)
            ax.add_patch(circle)
        plt.show()
        return

    num_spots = len(spots)
    cols = 5  # 右側に並べる小グラフの列数
    rows = (num_spots - 1) // cols + 1

    # FigureとGridSpecの作成（左側3列分をメイン画像、右側を小グラフ群に）
    fig = plt.figure(figsize=(15, max(6, rows * 2)))
    gs = gridspec.GridSpec(rows, cols + 3, figure=fig)

    # is_last が True なら "Last"、それ以外は数値を表示
    iter_text = "Last" if is_last else str(iteration_count)

    # ウィンドウ全体の上部に大きく表示
    if img_path:
        img_name = os.path.basename(img_path)
    else:
        img_name = "Unknown"

    fig.suptitle(
        f"{img_name}  |  Iteration: {iter_text}", fontsize=16, fontweight="bold"
    )

    # メイン画像の描画
    ax_main = fig.add_subplot(gs[:, :3])

    # メイン画像の上に小さくファイルパスを表示
    ax_main.set_title(f"{img_path}", fontsize=9, color="gray", loc="left", pad=10)
    fig_text = pformat(fig_info, indent=2, width=40)
    ax_main.text(
        0.05,
        0.05,
        fig_text,
        ha="left",
        va="bottom",
        fontsize=12,
    )
    if not img is None:
        ax_main.imshow(img, cmap="magma")

    if not isinstance(cir_stat, bool) and cir_stat is not None:
        cx, cy, R = cir_stat[0], cir_stat[1], cir_stat[2]
        circle = Circle((cx, cy), R, fill=False, color="orange", linewidth=2)
        ax_main.add_patch(circle)

    x, y = zip(*spots)
    ax_main.scatter(x, y, color="red", label="Edges", s=50)

    # 座標ラベルと対応関係のための番号を表示
    for idx, (xi, yi) in enumerate(zip(x, y)):
        # グラフと対応させる番号を大きく表示
        ax_main.text(
            xi,
            yi,
            f"#{idx+1}",
            color="lime",
            fontsize=12,
            fontweight="bold",
            ha="right",
            va="bottom",
        )
        # 元の座標表示も残す
        ax_main.text(
            xi,
            yi,
            f"({xi:.0f}, {yi:.0f})",
            color="#8917fd",
            fontsize=8,
            ha="left",
            va="top",
        )

    # 画像の分割線を描画
    lines = []
    for xy in ["x", "y"]:
        lines.append([])
        for nn in range(divnum):
            ap = width / divnum if xy == "x" else height / divnum
            lines[-1].append(ap * (nn + 1))
    for li in lines[0]:
        ax_main.axvline(int(li), color="white", linestyle="--", alpha=0.3)
    for li in lines[1]:
        ax_main.axhline(int(li), color="white", linestyle="--", alpha=0.3)

    ax_main.text(
        0.05, 0.9, f"n={divnum}", color="cyan", fontsize=10, transform=ax_main.transAxes
    )
    ax_main.legend()
    ax_main.axis("equal")

    # === 各エッジ点付近の小グラフを作成 ===
    window_size = 15  # 抽出する近傍のサイズ（前後15ピクセル）

    line_data = np.linspace(0, 0, window_size * 2)
    for idx, (xi, yi) in enumerate(zip(x, y)):
        # 横線(x_line)上の点か、縦線(y_line)上の点かを判定
        is_x_line = any(yi == height * i // divnum for i in range(1, divnum))

        if is_x_line:
            if not img is None:
                line_data = img[yi, :].astype(float)
            center_idx = xi
        else:  # y_line
            if not img is None:
                line_data = img[:, xi].astype(float)
            center_idx = yi

        # 中心から前後15ピクセル分を切り出す
        start = max(0, center_idx - window_size)
        end = min(len(line_data), center_idx + window_size + 1)

        vals = line_data[start:end]
        # np.diffは要素が1つ減るため、プロット用に末尾に0を追加して長さを合わせる
        grad_t = np.append(np.diff(line_data), 0)
        grad_vals = grad_t[start:end]

        # x軸は中心のエッジ点を0とした相対座標にする
        x_coords = np.arange(start, end) - center_idx

        # 小グラフの配置場所を計算
        r_idx = idx // cols
        c_idx = idx % cols
        ax_sub = fig.add_subplot(gs[r_idx, 3 + c_idx])

        # タイトルに画像と同じ番号を表示して紐付ける
        ax_sub.set_title(f"#{idx+1}", fontsize=10, color="black", fontweight="bold")

        # 【左軸】：明るさ（オレンジ色の実線）
        color_bright = "tab:orange"
        ax_sub.plot(x_coords, vals, color=color_bright, linewidth=1.5)
        ax_sub.tick_params(axis="y", labelcolor=color_bright, labelsize=7)
        ax_sub.tick_params(axis="x", labelsize=7)
        ax_sub.grid(alpha=0.3)

        # 【右軸】：微分値（シアン色の破線）
        ax_sub_twin = ax_sub.twinx()
        color_diff = "tab:cyan"
        ax_sub_twin.plot(
            x_coords, grad_vals, color=color_diff, linewidth=1.5, linestyle="--"
        )
        ax_sub_twin.tick_params(axis="y", labelcolor=color_diff, labelsize=7)

        # 実際に検出されたエッジの点（0の位置）に赤の縦線を引く
        ax_sub.axvline(0, color="red", linestyle="-", linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.show()


def show_circle_simple(
    img_inst: str | np.ndarray,
    spots: list[list[int]] | np.ndarray | None = None,
    cir_stat: tuple[float, float, float] | list[float] | bool = False,
    img_path: pathlib.Path | None = None,
    iteration_count: int | str = 1,
    is_last: bool = False,
    markersize: int = 400,
) -> None:
    """画像上に分割線、サンプリングされた縁の点、およびフィッティングされた近似円を描画してシンプルに画面に表示します。

    Args:
        img_inst (Union[str, np.ndarray]): 画像の指示。"GLOBAL"ならglobal変数のimgを取得し、"PATH"ならimg_pathから画像を読み込む。もしくは読み込んだ画像。
        spots (list[list[int]] | None): 描画する縁の点の座標リスト。デフォルトは None です。
        cir_stat (tuple[float, float, float] | list[float] | bool): 近似円のステータス [cx, cy, R]。描画しない場合は False を指定します。デフォルトは False です。
        img_path (pathlib.path): 画像のファイルパス。デフォルトはNoneです。
        iteration_count (int | str): 現在の反復回数。デフォルトは 1 です。
        is_last (bool): 最後の処理かどうかを示すフラグ。True の場合は "Last" と表示します。デフォルトは False です。
        markersize (int): プロットする縁の点のマーカーサイズ。デフォルトは 400 です。

    Returns:
        None: 戻り値はありません（画像をウィンドウに表示します）。
    """

    if isinstance(img_inst, str):
        if img_inst == "GLOBAL":
            img = globals().get("img")
            if img is None:
                raise ValueError(
                    "画像が指定されていません。imgを渡す、グローバル変数imgを設定する、または画像のパスを引数に追加してください。"
                )

        elif img_inst == "PATH":
            if img_path != None:
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError(
                        "画像が指定されていません。imgを渡す、グローバル変数imgを設定する、または画像のパスを引数に追加してください。"
                    )
            else:
                raise ValueError(
                    "画像が指定されていません。imgを渡す、グローバル変数imgを設定する、または画像のパスを引数に追加してください。"
                )

        else:
            raise ValueError(f"Unknown instructions = {img_inst}")
    else:
        img = img_inst

    if spots is None:
        spots = []
    img_cmap = "viridis"
    circle_limbC = "white"
    spot_C = "red"

    fig, ax = plt.subplots()  # figとaxの作成
    if not img is None:
        ax.imshow(img, cmap=img_cmap)  # 画像をグレースケールで表示
    if not isinstance(cir_stat, bool):  # cir_statがFalseでないなら、円を描画
        cx, cy, R = cir_stat[0], cir_stat[1], cir_stat[2]
        circle = plt.Circle(  # pyright: ignore[reportPrivateImportUsage]
            (cx, cy), R, fill=False, color=circle_limbC, linewidth=2
        )  # 結果の円を描画
        ax.add_patch(circle)  ###
    if len(spots) > 0:
        x, y = zip(*spots)
        ax.scatter(
            x,
            y,
            color=spot_C,
            label="Edges",
            s=markersize,
            linewidths=2,
            edgecolors="white",
        )
    # ウィンドウ全体の上部に大きく表示
    if img_path:
        img_name = os.path.basename(img_path)
    else:
        img_name = "Unknown"
    # is_last が True なら "Last"、それ以外は数値を表示
    iter_text = "Last" if is_last else str(iteration_count)
    fig.suptitle(
        f"{img_name}  |  Iteration: {iter_text}", fontsize=16, fontweight="bold"
    )

    # 画像の分割線を描画
    lines = []
    for xy in ["x", "y"]:  # 各分割線のlistを作成
        lines.append([])
        for nn in range(divnum):
            ap = width / divnum if xy == "x" else height / divnum
            lines[-1].append(ap * (nn + 1))
    for li in lines[0]:  # x方向の分割線を描画
        ax.axvline(int(li), color="white", linestyle="--", alpha=0.5)
    for li in lines[1]:  # y方向の分割線を描画
        ax.axhline(int(li), color="white", linestyle="--", alpha=0.5)

    # nの値を左上に固定表示
    ax.text(0.05, 0.9, f"n={divnum}", color="cyan", fontsize=10, transform=ax.transAxes)
    ax.legend()  ###
    ax.axis("equal")  ###
    plt.show()  # windowで表示


def MIN2_ignore_sunspots(
    img_inst: np.ndarray|str="PATH",
    n: int = 10,
    light_threshold: int = 50,
    limb_wigth: int = 24,
    iter_cycles: int = 2,
    show: bool = False,
    debug: bool = False,
    img_path: pathlib.Path|str = "",
    show_simple=False,
) -> tuple[tuple[float, float], float]:
    """黒点（サンスポット）による影響を除外しながら、最小二乗法により太陽の最終的な近似円（中心と半径）を検出します。

    一度検出した近似円の外側にある点から再度円を近似し、その円の縁幅（limb_wigth*(2/3)）の範囲内にない内側の点を黒点とみなして除外します。

    Args:
        img_inst (Union[np.ndarray,str]): 読み込んだ入力画像（グレースケール画像）または読み込み指示("PATH"ならimg_pathを読み込む)。
        n (int): 画像格子の分割数。デフォルトは 10 です。
        light_threshold (int): 太陽の明るさの基準しきい値。デフォルトは 50 です。
        limb_wigth (int): 太陽の縁の幅の基準値。デフォルトは 24 です。
        iter_cycles (int): 黒点排除のイテレーション回数。複数の黒点に対応できます。0なら排除なし。デフォルトは2。
        show (bool): 最終的な検出結果の画像を表示するかどうか。デフォルトは False です。
        debug (bool): 各ステップ（1回目の円、外側の点のみの円など）の描画やログを出力するかどうか。デフォルトは False です。
        img_path (Union[pathlib.Path,str]): 処理する画像のファイルパス。デフォルトは空文字列です。
        show_simple (bool): 描画時に詳細なグラフを省いたシンプルな表示形式を使用するかどうか。デフォルトは False です。

    Returns:
        tuple[tuple[float, float], float]: 最終的に算出された円の中心X座標(cx)、中心Y座標(cy)、および半径(r)のタプル。
    """
    if isinstance(img_path,str):
        img_path=pathlib.Path(img_path)
        if not img_path.exists():
            raise ValueError("そのパスの画像は存在しません。")
    
    if isinstance(img_inst,str):
        if img_inst == "PATH":
            readed_img=cv2.imread(str(img_path),cv2.IMREAD_UNCHANGED)
            if readed_img is None:
                raise ValueError("画像の読み込みに失敗しました。")
        else:
            raise ValueError(f"Unknown instructions = {img_inst}")
    else:
        readed_img=img_inst
            
    # ===基本的な変数をglobalで宣言===
    global divnum  # 分割数、引数ではnとして受け取っている。
    divnum = n
    global img  # 読み込んだ画像
    img = readed_img
    global height, width  # 画像の高さと幅
    height, width = img.shape[0:2]

    if img.dtype == np.uint8:
        pass
    elif img.dtype == np.uint16:
        light_threshold = light_threshold * 256
    # 円の情報[cx, cy, R]
    spots = cut_and_sampling(
        img_inst="GLOBAL", sun_threshold=light_threshold
    )  # spots=[[x1,y1],[x2,y2],...]の形式で、縁の点の座標を格納したlist
    cx, cy, r = fit_circle(spots, show)  # 一回目の円情報
    if debug:
        print(f"[INFO]:trial circle (cx,cy,r)={cx,cy,r}")
        if show:
            show_circle(
                img_inst="GLOBAL",
                spots=spots,
                cir_stat=(cx, cy, r),
                img_path=img_path,
                iteration_count=1,
                fig_info={"circle": "first trial"},
                simple=show_simple
            )

    all_sunspots=[]
    safe_points=spots
    for iter in range(iter_cycles+2):
        if debug:
            print(f"iter: {iter}")
        outside_spots = []
        inside_spots = []
        for point in safe_points:
            x,y=point
            if int(((x - cx) ** 2 + (y - cy) ** 2) ** (1 / 2)) > r:
                outside_spots.append(point)
            else:
                inside_spots.append(point)

        #inside_spotsから発見された黒点をindexで管理するため
        safe_points=inside_spots+outside_spots

        cxo, cyo, ro = fit_circle(np.array(outside_spots, dtype=float), show)
        if debug:
            print(f"[INFO]:outside circle (cx,cy,r)={cxo,cyo,ro}")
            if show:
                show_circle(
                    img_inst="GLOBAL",
                    spots=outside_spots,
                    cir_stat=(cxo, cyo, ro),
                    img_path=img_path,
                    iteration_count=iter_cycles,
                    fig_info={"circle": "only points only"},
                    simple=show_simple
                )

        sunspots = []

        if debug:
            print(f"    [INFO]:外側の点の数:{len(outside_spots)},全体の点の数:{len(spots)}")

        for i,point in enumerate(inside_spots):
            x,y = point 

            if (x - cxo) ** 2 > (y - cyo) ** 2:  # 円のRLTBのうちRLなら、
                min2far = np.sqrt(ro**2 - (y - cyo) ** 2)
                if debug:
                    print(f"    {i} x,y:{x,y} min2far:{min2far},y-cyo:{np.abs(cyo-y)}")
                    
                if min2far - np.abs(cxo - x) > limb_wigth * (2 / 3):
                    sunspots.append(i)

            else:  # 円のRLTBのうちTBなら
                min2far = np.sqrt(ro**2 - (x - cxo) ** 2)
                if debug:
                    print(f"    {i} x,y:{x,y} min2far:{min2far},x-cxo:{np.abs(cxo-x)}")
                    
                if min2far - np.abs(cyo - y) > limb_wigth * (2 / 3):
                    sunspots.append(i)

        for i in sorted(sunspots, reverse=True):
            all_sunspots.append(safe_points.pop(i))

        if sunspots:
            # 黒点とみなされない点だけで円を作成
            cx, cy, r = fit_circle(
                np.array(safe_points, dtype=float), show
            )
            
        else:
            break
        
    if show:
        # 最終結果 (is_last=True)
        show_circle(
            img_inst="GLOBAL",
            spots=safe_points,
            cir_stat=(cx, cy, r),
            img_path=img_path,
            is_last=True,
            simple=show_simple
        )
    return (cx, cy), r


if __name__ == "__main__":
    from tkinter.filedialog import askdirectory, askopenfilename

    if input("[OPERATE]:onefile(0)/dir(1)?:") == "1":

        dirpath = askdirectory(title="フォルダを選択してください")
        print(f"[INFO]:dir={dirpath}")
        import glob
        import os

        patterns = ("*.jpg", "*.jpeg", "*.png", "*.tiff")
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(dirpath, p)))
        for file in files:
            img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"[ERROR]:Failed to read image: {file}")
                break
            (cx,cy),r = MIN2_ignore_sunspots(img, show=False, debug=False, limb_wigth=60)
            result=cx,cy,r
            print((float(result[0]), float(result[1]), float(result[2])))
    else:
        picpath = askopenfilename(
            title="画像を選択してください",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.tiff")],
        )
        print(f"[INFO]:image={picpath}")
        from time import time

        start = time()
        img = cv2.imread(picpath, cv2.IMREAD_UNCHANGED)
        if not img is None:
            print(
                f"[INFO]:result{MIN2_ignore_sunspots(img, show=True, debug=True, img_path=picpath,show_simple=True)}"
            )
        else:
            print(f"[ERROR]:reading img failed path={picpath}")
        print(f"[INFO]:process time :{time()-start} s")
