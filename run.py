"""SUN_watching sun_find_west_v2 の実行用エントリポイント。

実行時オプション(ダミーカメラを使うかどうか)はこのファイルだけが決める。
決定の元になるのは userSet (sun_find_west_v2/config/user_set/config.txt) の
``===Run===`` > ``dummy_camera`` で、True ならダミーカメラ、False なら実機ASIカメラ。

zwoasi が入っているかどうかや、カメラ接続待機中のキー入力でダミーに切り替わることはない。

使い方::

    python run.py
"""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT / "sun_find_west_v2"


def main() -> None:
    """userSet から実行オプションを決定し、main.py を起動する。"""
    if str(PACKAGE_DIR) not in sys.path:
        sys.path.insert(0, str(PACKAGE_DIR))

    # main.py 側と同じ import 名(config.*)で読み込み、オプションの受け渡し先を共有する
    from config.pathes import pathes
    from config.run_options import load_run_options, set_run_options

    options = load_run_options(pathes["CONFIG_TEXT_PATH"])
    set_run_options(options)
    print(f"[run.py] dummy_camera={options.dummy_camera} (userSet)")

    runpy.run_path(str(PACKAGE_DIR / "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()
