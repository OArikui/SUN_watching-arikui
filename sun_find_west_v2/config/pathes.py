from pathlib import Path

current = Path(__file__)
configDir = current.parent.resolve()
pathes = {
    "CONFIG_TEXT_PATH": configDir / "user_set/config.txt",
    "CONFIG_JSON_PATH": configDir / "cache/config.json",
    "DEFAULT_JSON_PATH": configDir / "schemas/default_config.json",
    "OUTLINE_JSON_PARH": configDir / "schemas/default_outline.json",
    "CONFIG_SCHEMA_PATH": configDir / "schemas/config_schema.json",
}
