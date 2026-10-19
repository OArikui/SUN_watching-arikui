"""YAML設定ファイルの読み込みと JSON Schema による検証。

設定の読み込みは run.py だけが行い、検証済みの dict を ``main(config)`` に渡す。
main.py や camera 配下は設定ファイルに直接触れない。
"""

import json
import logging
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft7Validator

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.yaml"
SCHEMA_PATH = CONFIG_DIR / "config_schema.json"


class ConfigError(Exception):
    """設定ファイルの読み込み・検証に失敗したときの例外。"""


def _format_error(path: str, message: str) -> str:
    return f"  - {path or '(root)'}: {message}"


def load_config(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    schema_path: Path | str = SCHEMA_PATH,
) -> dict[str, Any]:
    """YAML設定を読み込み、スキーマで検証した dict を返す。

    項目の補完は行わない。欠落・未知の項目・型や範囲の違反はすべて ConfigError にする。

    Parameters
    ----------
    config_path : Path | str
        読み込む YAML ファイル。
    schema_path : Path | str
        検証に使う JSON Schema (Draft 7)。

    Returns
    -------
    dict[str, Any]
        検証済みの設定。

    Raises
    ------
    ConfigError
        ファイルが無い、YAML として不正、またはスキーマに違反している場合。
    """
    config_path = Path(config_path)
    schema_path = Path(schema_path)

    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        Draft7Validator.check_schema(schema)
    except (OSError, ValueError) as e:
        msg = f"スキーマを読み込めません: {schema_path}: {e}"
        raise ConfigError(msg) from e

    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as e:
        msg = f"設定ファイルを読み込めません: {config_path}: {e}"
        raise ConfigError(msg) from e
    except yaml.YAMLError as e:
        msg = f"設定ファイルが YAML として不正です: {config_path}\n{e}"
        raise ConfigError(msg) from e

    errors = sorted(
        Draft7Validator(schema).iter_errors(data),
        key=lambda err: [str(p) for p in err.absolute_path],
    )
    if errors:
        lines = [
            _format_error(".".join(str(p) for p in err.absolute_path), err.message)
            for err in errors
        ]
        msg = f"設定ファイルがスキーマに違反しています: {config_path}\n" + "\n".join(lines)
        raise ConfigError(msg)

    logger.info(f"Config loaded and validated: {config_path}")
    return data
