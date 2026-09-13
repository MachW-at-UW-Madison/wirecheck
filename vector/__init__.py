from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .calibrate import FloorCalibration
    from .parse import Report
    from .rdata import RawCapture, RawCollection
    from .wirecheck import (
        Analysis,
        CalibrationConfig,
        Config,
        analyze,
        calibrate_fractional_floor,
    )

__all__ = [
    "Analysis",
    "CalibrationConfig",
    "Config",
    "FloorCalibration",
    "RawCapture",
    "RawCollection",
    "Report",
    "analyze",
    "calibrate_fractional_floor",
]

_OWNERS = {
    "FloorCalibration": "calibrate",
    "RawCapture": "rdata",
    "RawCollection": "rdata",
    "Report": "parse",
}


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(name)
    return getattr(import_module(f".{_OWNERS.get(name, 'wirecheck')}", __name__), name)
