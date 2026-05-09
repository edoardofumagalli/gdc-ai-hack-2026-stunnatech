from __future__ import annotations


def stereo_preset(dai, name: str):
    return _enum_value(dai.node.StereoDepth.PresetMode, name, "StereoDepth preset")


def median_filter(dai, name: str):
    return _enum_value(dai.MedianFilter, name, "median filter")


def configure_stereo(stereo, dai, *, subpixel: bool, median_filter_name: str) -> None:
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(subpixel)
    try:
        stereo.initialConfig.setMedianFilter(median_filter(dai, median_filter_name))
    except AttributeError:
        pass


def _enum_value(enum_cls, name: str, label: str):
    requested = name.strip()
    if hasattr(enum_cls, requested):
        return getattr(enum_cls, requested)

    requested_lower = requested.lower()
    available = [
        item
        for item in dir(enum_cls)
        if not item.startswith("_") and item[0].isupper()
    ]
    for item in available:
        if item.lower() == requested_lower:
            return getattr(enum_cls, item)

    raise ValueError(
        f"Unknown {label} '{name}'. Available values: {', '.join(available)}"
    )
