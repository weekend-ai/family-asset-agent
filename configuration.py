"""Load local resource identifiers and owner aliases; never authentication secrets."""
import json
from pathlib import Path

import click

CONFIG_PATH = Path(__file__).resolve().parent / "private" / "config.json"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("expected an object")
        for key in ("tables", "table_names", "owner_aliases"):
            value = config.get(key, {})
            if not isinstance(value, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in value.items()):
                raise ValueError(f"{key} must map strings to strings")
        owners = config.get("owners", [])
        if not isinstance(owners, list) or any(not isinstance(owner, str) or not owner.strip() for owner in owners):
            raise ValueError("owners must be a list of nonempty names")
        if "base_token" in config and not isinstance(config["base_token"], str):
            raise ValueError("base_token must be a string")
        return config
    except (OSError, ValueError) as exc:
        raise click.ClickException("Invalid private/config.json; check JSON, resource fields and owner mappings") from exc
