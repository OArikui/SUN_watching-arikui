import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def json_loader(jsonpath: Path) -> dict:
    try:
        with open(jsonpath, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.exception("JSON format error")
        raise
    except FileNotFoundError:
        logger.exception("File not found")
        raise


def json_saver(data: dict, jsonpath: Path) -> dict:
    logger.info(f"Saving JSON to {jsonpath}")
    try:
        if not jsonpath.parent.exists():
            jsonpath.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {jsonpath.parent}")

        with jsonpath.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        sha256_file(jsonpath)

        logger.info(f"JSON saved successfully: {jsonpath}")
        return data
    except Exception:
        logger.exception(f"Failed to save JSON: {jsonpath}")
        raise


def sha256_file(filepath: Path, save: bool = True) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            assert isinstance(chunk, bytes)
            h.update(chunk)
    hashV = h.hexdigest()

    if save:
        savepath = filepath.parent / f"{filepath.name}_valid"
        with open(savepath, "a", encoding="ascii") as f:
            f.write(f"{datetime.now(UTC)}\n{hashV}\n")
    return hashV


def sha256_valid(filepath: Path) -> bool:
    actual = sha256_file(filepath, False)
    expect_path = Path(str(filepath) + "_valid")
    if expect_path.exists:
        with open(expect_path, "r", encoding="ascii") as f:
            expected = f.readlines()[-1]
    else:
        expected = None
        logger.debug(f"hash not found :{Path}")
    return actual == expected
