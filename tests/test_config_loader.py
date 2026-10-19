"""config.loader の検証動作のテスト。"""

import copy
from pathlib import Path

import pytest
import yaml

from sun_find_west_v2.config.loader import DEFAULT_CONFIG_PATH, ConfigError, load_config


@pytest.fixture
def base() -> dict:
    """同梱の config.yaml の中身。"""
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def test_shipped_config_is_valid() -> None:
    config = load_config()
    assert config["Camera"]["basic_parameters"]["exposure"] == 100
    assert config["Run"]["camera_source"]["dummy_camera"] is False
    assert config["sun_find_west_v2"]["Visualizer"]["grid_color"] == "#FFFFFF"


def test_valid_override(tmp_path: Path, base: dict) -> None:
    base["Run"]["camera_source"]["dummy_camera"] = True
    assert load_config(_write(tmp_path, base))["Run"]["camera_source"]["dummy_camera"] is True


@pytest.mark.parametrize(
    ("section", "group", "key", "value"),
    [
        ("Camera", "basic_parameters", "exposure", 5),  # minimum 未満
        ("Camera", "basic_parameters", "gain", True),  # bool は integer ではない
        ("Camera", "image_format", "img_type", "RGB24"),  # enum 外
        ("sun_find_west_v2", "Analyzer", "buf_lookback", "300"),  # 文字列
        ("sun_find_west_v2", "Visualizer", "grid_alpha", 1.5),  # maximum 超過
        ("sun_find_west_v2", "Visualizer", "grid_color", "red"),  # pattern 違反
        ("Run", "camera_source", "dummy_camera", "yes please"),  # 型違反
    ],
)
def test_invalid_value_is_rejected(
    tmp_path: Path, base: dict, section: str, group: str, key: str, value: object
) -> None:
    base[section][group][key] = value
    with pytest.raises(ConfigError, match=f"{section}.{group}.{key}"):
        load_config(_write(tmp_path, base))


def test_missing_key_is_rejected(tmp_path: Path, base: dict) -> None:
    del base["Camera"]["basic_parameters"]["gain"]
    with pytest.raises(ConfigError, match="'gain' is a required property"):
        load_config(_write(tmp_path, base))


def test_unknown_key_is_rejected(tmp_path: Path, base: dict) -> None:
    base["Camera"]["basic_parameters"]["gian"] = 0
    with pytest.raises(ConfigError, match="gian"):
        load_config(_write(tmp_path, base))


def test_all_errors_are_reported(tmp_path: Path, base: dict) -> None:
    broken = copy.deepcopy(base)
    broken["Camera"]["basic_parameters"]["exposure"] = 5
    broken["Camera"]["basic_parameters"]["gain"] = 999
    with pytest.raises(ConfigError) as info:
        load_config(_write(tmp_path, broken))
    assert "exposure" in str(info.value)
    assert "gain" in str(info.value)


def test_unquoted_hash_color_is_rejected(tmp_path: Path) -> None:
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").replace('"#FFFFFF"', "#FFFFFF")
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="grid_color"):
        load_config(path)


def test_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("Camera: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML"):
        load_config(path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")


def test_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)
