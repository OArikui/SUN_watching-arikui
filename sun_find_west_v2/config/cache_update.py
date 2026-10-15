import logging
from pathlib import Path

if __name__ == "__main__":
    from pathes import pathes
    from utils_json import json_loader, json_saver, sha256_file
else:
    from config.pathes import pathes
    from config.utils_json import json_loader, json_saver, sha256_file

logger = logging.getLogger(__name__)


def generate_defaultJ_schemaJ(defaultpath: Path, schemapath: Path) -> dict:
    "reset to schema"
    schema = json_loader(schemapath)

    if schema.get("type") != "object" or "properties" not in schema:
        msg = "The schema format is invalid (the root must be an 'object' with 'properties')."
        logger.error(f"{msg}")
        raise ValueError(msg)

    def get_default_properties(child: dict) -> dict:
        son = {}
        for k, v in child.items():
            if v["type"] == "object":
                value = get_default_properties(v["properties"])
                son[k] = value
            else:
                son[k] = v["default"]
                son[k + "-desc"] = v["description"]

        return son

    try:
        data = get_default_properties(schema["properties"])
    except Exception as e:
        logger.error(f"An error occurred while parsing the schema.: {e}")
        raise

    json_saver(data, defaultpath)

    return data


def generate_configT_defaultJ(configpath: Path, textpath: Path) -> str:
    "reset to default json"

    data = json_loader(configpath)

    txt = ""
    for hk, hv in data.items():
        txt += f"\n==={hk}===\n"
        for hhk, hhv in hv.items():
            txt += f"\n---{hhk}---\n"
            for k, v in hhv.items():
                if k.endswith("-desc"):
                    continue
                desc_key = k + "-desc"
                d = hhv[desc_key]
                if len(k) < 10:
                    k += " " * (10 - len(k))
                v = str(v)
                if len(v) < 5:
                    v += " " * (5 - len(v))
                txt += f"{k} : {v}  /{d}\n"

    if not textpath.parent.exists():
        textpath.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {textpath.parent}")

    with open(textpath, "w", encoding="utf-8") as f:
        f.write(txt)

    return txt


def generate_outlinesJ_defaultJ(outlinepath: Path, defaultpath: Path) -> dict:
    "generate outlines based on default json"
    data = json_loader(defaultpath)

    outline = {}

    for hk, hv in data.items():
        ls_h2 = {}
        for hhk, hhv in hv.items():
            ls_h2[hhk] = [k for k in hhv if not k.endswith("-desc")]
        outline[hk] = ls_h2

    return json_saver(outline, outlinepath)


def generate_configJ_configT(configpath: Path, textpath: Path) -> dict:
    if not textpath.exists():
        logger.error(f"file not found:{textpath!s}")

    with open(textpath, "r", encoding="utf-8") as f:
        text = f.read().splitlines()

    stash = {}
    data_h1 = {}
    data_h2 = {}
    h1 = ""
    h2 = ""
    for i, line in enumerate(text):
        if ":" in line:
            kv, d = tuple(line.split("/"))
            kv = kv.strip()
            k, v = tuple(kv.split(":"))
            stash[k] = v
            stash[k + "-desc"] = d
        if line[:3] == "---":
            if stash:
                data_h2[h2] = stash
            h2 = line.split("---")[1]
            stash = {}
        if line[:3] == "===":
            if stash:
                data_h2[h2] = stash
                stash = {}
            if data_h2:
                data_h1[h1] = data_h2
            h1 = line.split("===")[1]
            data_h2 = {}
    if stash:
        data_h2[h2] = stash
    if data_h2:
        data_h1[h1] = data_h2
    return json_saver(data=data_h1, jsonpath=configpath)


def when_updated_schemaJ(pathes: dict, reset_userset: bool) -> dict:
    sha256_file(pathes["CONFIG_SCHEMA_PATH"])
    generate_defaultJ_schemaJ(pathes["DEFAULT_JSON_PATH"], pathes["CONFIG_SCHEMA_PATH"])
    generate_outlinesJ_defaultJ(
        pathes["OUTLINE_JSON_PARH"], pathes["DEFAULT_JSON_PATH"]
    )
    logger.info("Successfully updated the default outline JSON.")
    if reset_userset:
        logger.info("Resetting user settings...")
        generate_configT_defaultJ(
            pathes["DEFAULT_JSON_PATH"], pathes["CONFIG_TEXT_PATH"]
        )
        logger.info("User settings successfully reset.")
    return json_loader(pathes["CONFIG_SCHEMA_PATH"])


if __name__ == "__main__":
    import sys

    CONFIG_ROOT = Path(__file__).parent.resolve()
    sys.path.append(str(CONFIG_ROOT))
    when_updated_schemaJ(pathes, True)
