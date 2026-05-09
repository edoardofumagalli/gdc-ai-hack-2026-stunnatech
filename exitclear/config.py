from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import (
    AppConfig,
    DeviceConfig,
    ImageRoi,
    OccupancyConfig,
    OutputsConfig,
    ZoneConfig,
)


def load_config(path: str | Path) -> AppConfig:
    raw = _load_yaml(Path(path))
    zones = []
    for item in raw["zones"]:
        zones.append(
            ZoneConfig(
                id=item["id"],
                label=item["label"],
                type=item["type"],
                required_clear_width_mm=int(item["required_clear_width_mm"]),
                monitored_height_mm=int(item["monitored_height_mm"]),
                monitored_depth_mm=int(item["monitored_depth_mm"]),
                persistence_threshold_s=float(item["persistence_threshold_s"]),
                transient_person_grace_s=float(item["transient_person_grace_s"]),
                image_roi=ImageRoi(**item["image_roi"]),
                occupancy=OccupancyConfig(**item["occupancy"]),
            )
        )

    return AppConfig(
        device=DeviceConfig(**raw["device"]),
        zones=zones,
        outputs=OutputsConfig(**raw["outputs"]),
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return _parse_yaml_subset(path.read_text(encoding="utf-8"))

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at top level in {path}")
    return data


def _parse_yaml_subset(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by config/zones.yaml when PyYAML is absent."""
    rows: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        rows.append((len(line) - len(line.lstrip(" ")), line.lstrip(" ")))

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(rows):
            return {}, index
        if rows[index][1].startswith("- "):
            return parse_list(index, indent)
        return parse_dict(index, indent)

    def parse_dict(index: int, indent: int) -> tuple[dict[str, Any], int]:
        result: dict[str, Any] = {}
        while index < len(rows):
            row_indent, content = rows[index]
            if row_indent < indent:
                break
            if row_indent > indent:
                raise ValueError(f"Unexpected indentation near: {content}")
            if content.startswith("- "):
                break
            key, value = split_key_value(content)
            index += 1
            if value == "":
                child_indent = rows[index][0] if index < len(rows) else indent + 2
                result[key], index = parse_block(index, child_indent)
            else:
                result[key] = parse_scalar(value)
        return result, index

    def parse_list(index: int, indent: int) -> tuple[list[Any], int]:
        result: list[Any] = []
        while index < len(rows):
            row_indent, content = rows[index]
            if row_indent < indent:
                break
            if row_indent != indent or not content.startswith("- "):
                break
            item_text = content[2:].strip()
            index += 1
            if item_text == "":
                child_indent = rows[index][0] if index < len(rows) else indent + 2
                item, index = parse_block(index, child_indent)
                result.append(item)
                continue
            if ":" not in item_text:
                result.append(parse_scalar(item_text))
                continue

            key, value = split_key_value(item_text)
            item: dict[str, Any] = {}
            if value == "":
                child_indent = rows[index][0] if index < len(rows) else indent + 2
                item[key], index = parse_block(index, child_indent)
            else:
                item[key] = parse_scalar(value)

            if index < len(rows) and rows[index][0] > indent:
                extra, index = parse_dict(index, rows[index][0])
                item.update(extra)
            result.append(item)
        return result, index

    parsed, final_index = parse_block(0, rows[0][0] if rows else 0)
    if final_index != len(rows):
        raise ValueError("Could not parse full YAML config")
    if not isinstance(parsed, dict):
        raise ValueError("Expected top-level YAML mapping")
    return parsed


def split_key_value(content: str) -> tuple[str, str]:
    if ":" not in content:
        raise ValueError(f"Expected key/value pair near: {content}")
    key, value = content.split(":", 1)
    return key.strip(), value.strip()


def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
