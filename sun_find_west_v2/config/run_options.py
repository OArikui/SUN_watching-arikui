"""run.py が起動時に決める実行オプションの読み込みと、main.py への受け渡し。

オプションの決定は run.py だけが行う。main.py や camera 配下は
``get_run_options()`` で受け取った値に従うだけで、自分では判断しない。
"""

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_TRUE_WORDS = {"true", "1", "on", "yes"}
_FALSE_WORDS = {"false", "0", "off", "no"}


@dataclass(frozen=True)
class RunOptions:
    """実行時オプション。

    Attributes
    ----------
    dummy_camera : bool
        True ならダミーカメラ(動画ファイル再生)、False なら実機ASIカメラを使う。
    """

    dummy_camera: bool


def _parse_bool(key: str, raw: str) -> bool:
    """userSet の値(True/False, on/off, 1/0)を bool に変換する。"""
    word = raw.strip().lower()
    if word in _TRUE_WORDS:
        return True
    if word in _FALSE_WORDS:
        return False
    msg = f"userSet の {key} は True か False で指定してください: {raw.strip()!r}"
    raise ValueError(msg)


def load_run_options(text_path: Path) -> RunOptions:
    """userSet(config.txt)の ``===Run===`` セクションから実行オプションを読む。

    Parameters
    ----------
    text_path : Path
        userSet の config.txt のパス。

    Returns
    -------
    RunOptions
        読み込んだオプション。項目が無い場合は dummy_camera=False(実機)になる。

    Raises
    ------
    FileNotFoundError
        config.txt が存在しない場合。
    ValueError
        値が True/False として解釈できない場合。
    """
    if not text_path.exists():
        msg = f"userSet が見つかりません: {text_path}"
        raise FileNotFoundError(msg)

    section = ""
    values: dict[str, str] = {}
    for line in text_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("==="):
            section = stripped.strip("=").strip()
            continue
        if stripped.startswith("---") or section != "Run" or ":" not in stripped:
            continue
        key_value = stripped.partition("/")[0]
        key, _, value = key_value.partition(":")
        values[key.strip()] = value

    if "dummy_camera" in values:
        dummy_camera = _parse_bool("dummy_camera", values["dummy_camera"])
    else:
        logger.warning("userSet に Run>dummy_camera がありません。実機カメラとして扱います。")
        dummy_camera = False
    return RunOptions(dummy_camera=dummy_camera)


_current: RunOptions | None = None


def set_run_options(options: RunOptions) -> None:
    """run.py が決めたオプションを登録する(run.py から main.py 起動前に1回だけ呼ぶ)。"""
    global _current  # noqa: PLW0603
    _current = options


def get_run_options() -> RunOptions:
    """run.py が登録したオプションを返す。

    Raises
    ------
    RuntimeError
        run.py を経由せずに起動された場合。
    """
    if _current is None:
        msg = "実行オプションが未設定です。run.py から起動してください。"
        raise RuntimeError(msg)
    return _current
