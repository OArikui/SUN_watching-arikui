# part1 importing modules
import datetime
import logging
import sys
from pathlib import Path

update_basedon_textconfig = False
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    CONFIG_ROOT = Path(__file__).parent.resolve()
    sys.path.append(str(CONFIG_ROOT))

    logpath = Path(f"logs/config_{datetime.datetime.now().strftime('%Y-%m-%d')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s %(funcName)s: %(message)s",
        filename=logpath,
        filemode="a",
    )
    update_basedon_textconfig = True
    if not logpath.parent.exists():
        logpath.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {logpath.parent}")

    from cache_update import generate_configJ_configT
    from pathes import pathes
    from utils_json import json_loader, sha256_valid


else:
    from config.cache_update import generate_configJ_configT
    from config.pathes import pathes
    from config.utils_json import json_loader, sha256_valid
logger = logging.getLogger(__name__)
CONFIG_JSON_PATH = pathes["CONFIG_JSON_PATH"]
CONFIG_TEXT_PATH = pathes["CONFIG_TEXT_PATH"]
DEFAULT_JSON_PATH = pathes["DEFAULT_JSON_PATH"]
OUTLINE_JSON_PARH = pathes["OUTLINE_JSON_PARH"]


def file_update_check(pathes: dict, update_basedon_textconfig: bool) -> bool:
    nodiff = 0
    for filename, filepath in pathes.items():
        same = sha256_valid(filepath)
        if same:
            nodiff += 1
        else:
            logger.debug(f"file is changed: {filepath}")

        if filename == "CONFIG_JSON_PATH" and update_basedon_textconfig:
            from cache_update import generate_configJ_configT

            generate_configJ_configT(CONFIG_JSON_PATH, CONFIG_TEXT_PATH)
    return nodiff == len(pathes)


def industrial(config: dict, default_config: dict, default_outline: dict) -> dict:
    "no desc,no empty"
    buried_no_desc = {}

    for hk, hv in default_outline.items():
        try:
            parent_config = config[hk]
        except KeyError:
            logger.debug(f"Item {hk} was not found. Using the default value.")
            buried_no_desc[hk] = default_config[hk]
            continue

        data_h1 = {}
        for hhk, hhv in hv.items():
            try:
                current_config = parent_config[hhk]
            except KeyError:
                logger.debug(f"Item {hk}>{hhk} was not found. Using the default value.")
                data_h1[hhk] = default_config[hk][hhk]
                continue
            data_h2 = {}
            for k in hhv:
                getted = current_config.get(k, None)
                if getted != None:
                    data_h2[k] = getted
                else:
                    data_h2[k] = default_config[hk][hhk][k]
            data_h1[hhk] = data_h2
        buried_no_desc[hk] = data_h1
    return buried_no_desc


# textをjson形式に変換し、jsonファイルを生成
try:
    generate_configJ_configT(CONFIG_JSON_PATH, CONFIG_TEXT_PATH)
    logger.info("Successfully generated JSON from text.")
except (KeyError, ValueError):
    logger.exception("The format of the text setting is incorrect.")
    logger.info("Attempting to load parameters from the configuration JSON...")

# textからの読み込みに失敗したときは前回までのプロパティで実行する。
# 生成したjsonを読み込む
try:
    logger.info("Loading settings from JSON...")
    config = json_loader(CONFIG_JSON_PATH)
except Exception:
    config = {}
    logger.warning("Failed to load settings from JSON.")


#   for bury the empty,prepare defult json

# defaultのjsonとoutlineを空埋め用に読み込む
default_config = {}
outline = {}
try:
    default_config = json_loader(DEFAULT_JSON_PATH)
    outline = json_loader(OUTLINE_JSON_PARH)
    logger.info("Successfully loaded default_config.")
except Exception:
    logger.critical("Failed to load default_config.")

if "__main__" == __name__:
    file_update_check(pathes, True)
parameter = industrial(config, default_config, outline)
