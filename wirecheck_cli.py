from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence, TypeAlias, cast

import numpy as np
from tqdm.auto import tqdm

import wirecheck.file_adapter as file_adapter
from wirecheck import vector


def _log_wrote(path: Path) -> Path:
    print(f"wrote {path}", file=sys.stderr)
    return path


CITATION = (
    "If you use WireCheck, please cite: Xincheng Xie and Akarsh Prabhakara. "
    "2026. Demo: Locating Leakage Current via Magnetometers. In MobiCom '26. "
    "ACM. https://doi.org/10.1145/3795866.3848441"
)
DEFAULTS = vector.Config()
FLOOR_DEFAULTS = vector.CalibrationConfig()


@dataclass(frozen=True)
class CampaignArgs:
    block_seconds: float
    lever_mm: tuple[float, float, float]
    sensor_rotation: tuple[float, ...]
    max_travel_mm: float
    max_along_mm: float
    line_hz: float | None
    per_block_phase: bool
    z1_height_mm: float
    core_mm: float
    pair_spacing_mm: float


@dataclass(frozen=True)
class AnalyzeArgs(CampaignArgs):
    data: Path
    fractional_floor: float
    no_figure: bool
    show: bool
    out_dir: Path | None

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> AnalyzeArgs:
        return cls(
            block_seconds=namespace.block_seconds,
            lever_mm=tuple(namespace.lever_mm),
            sensor_rotation=tuple(namespace.sensor_rotation),
            max_travel_mm=namespace.max_travel_mm,
            max_along_mm=namespace.max_along_mm,
            line_hz=namespace.line_hz,
            per_block_phase=namespace.per_block_phase,
            z1_height_mm=namespace.z1_height_mm,
            core_mm=namespace.core_mm,
            pair_spacing_mm=namespace.pair_spacing_mm,
            data=namespace.data,
            fractional_floor=namespace.fractional_floor,
            no_figure=namespace.no_figure,
            show=namespace.show,
            out_dir=namespace.out_dir,
        )


@dataclass(frozen=True)
class FloorArgs(CampaignArgs):
    data: list[Path]
    seed_floor: float
    search_min: float
    search_max: float
    output: Path | None

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> FloorArgs:
        return cls(
            block_seconds=namespace.block_seconds,
            lever_mm=tuple(namespace.lever_mm),
            sensor_rotation=tuple(namespace.sensor_rotation),
            max_travel_mm=namespace.max_travel_mm,
            max_along_mm=namespace.max_along_mm,
            line_hz=namespace.line_hz,
            per_block_phase=namespace.per_block_phase,
            z1_height_mm=namespace.z1_height_mm,
            core_mm=namespace.core_mm,
            pair_spacing_mm=namespace.pair_spacing_mm,
            data=namespace.data,
            seed_floor=namespace.seed_floor,
            search_min=namespace.search_min,
            search_max=namespace.search_max,
            output=namespace.output,
        )


JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)


def _json_safe(value: object) -> JsonValue:
    if isinstance(value, dict):
        items = cast(dict[str, object], value)
        return {key: _json_safe(item) for key, item in items.items()}
    if isinstance(value, (list, tuple)):
        items = cast(Sequence[object], value)
        return [_json_safe(item) for item in items]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return cast(JsonValue, value)


def _print_report(report: vector.Report) -> None:
    print(f"\n{report['collection']}")
    print(f"  {report['health'].replace('_', ' ').upper()}")
    for failure in report["quality"]["failures"]:
        print(f"    ! {failure}")
    print(f"  load current ~ {report['estimated_load_current_a']:.3f} A")
    leak_ma = report["estimated_leak_current_ma"]
    coefficient = report["leak_coeff_nt_m"]
    phase_deg = report["estimated_leak_phase_deg"]
    phase_cosine = report["estimated_leak_phase_cosine_k"]
    leak_label = (
        "leak current"
        if report["health"] in {"leakage_suspected", "leakage_detected"}
        else "fitted leak current"
    )
    print(
        f"  {leak_label} ~ {leak_ma:.2f} mA "
        f"(C={coefficient:.3f} nT·m; "
        f"in-phase={report['leak_inphase_ma']:.2f} mA)"
    )
    print(
        f"  thresholds: suspect≥{report['leak_suspect_threshold_ma']:.0f} mA, "
        f"leakage≥{report['leak_detect_threshold_ma']:.0f} mA; "
        f"no cable <{report['minimum_load_current_a']:.1f} A"
    )
    print(f"  relative phase ~ {phase_deg:.1f}°; " f"cos ψ={phase_cosine:.3f}")
    print(f"  selected: {report['selected_model']}; " f"N={report['total_positions']}")


def _print_lockin(collection: vector.RawCollection) -> None:
    if collection.reference_hz is None:
        print(f"{collection.name}: lock-in phase per block")
    else:
        print(
            f"{collection.name}: one lock-in phase for the campaign at "
            f"{collection.reference_hz:.5f} Hz"
        )


def _write_figure(out_dir: Path, analysis: vector.Analysis, show: bool) -> None:
    import matplotlib.pyplot as plt

    path = out_dir / "figures" / f"{analysis.report['collection']}_summary.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = analysis.make_figure()
    figure.savefig(path, dpi=180)
    if show:
        manager = plt.figure().canvas.manager
        if manager is not None:
            manager.canvas.figure = figure
            figure.set_canvas(manager.canvas)
            plt.show()
        plt.close("all")
    _log_wrote(path)


def _rotation(text: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in text.replace(";", ",").split(","))
    if len(values) != 9:
        raise argparse.ArgumentTypeError("expected nine comma-separated numbers")
    return values


def _add_campaign_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--block-seconds",
        type=float,
        default=file_adapter.DEFAULT_BLOCK_SECONDS,
        help="lock-in block length in seconds; one block becomes one "
        "measurement with the iPad pose at its midpoint",
    )
    parser.add_argument(
        "--lever-mm",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
        help="camera-to-sensor lever arm in millimeters, in camera axes",
    )
    parser.add_argument(
        "--sensor-rotation",
        type=_rotation,
        default=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        metavar="R00,...,R22",
        help="row-major 3x3 rotation from sensor axes to camera axes",
    )
    parser.add_argument(
        "--max-travel-mm",
        type=float,
        default=file_adapter.DEFAULT_MAX_TRAVEL_M * 1000.0,
        help="discard a block when the camera moved more than this",
    )
    parser.add_argument(
        "--max-along-mm",
        type=float,
        default=file_adapter.DEFAULT_MAX_ALONG_M * 1000.0,
        help="discard a block when it is this far along the cable axis",
    )
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument(
        "--line-hz",
        type=float,
        default=None,
        help="line frequency on the magnetometer clock; every block is "
        "demodulated at it against absolute time and the campaign shares one "
        "lock-in phase; default: meta.json, else estimated from the recording",
    )
    phase.add_argument(
        "--per-block-phase",
        action="store_true",
        help="give every block its own tracked frequency and lock-in phase",
    )
    parser.add_argument(
        "--z1-height-mm",
        type=float,
        default=file_adapter.DEFAULT_Z1_HEIGHT_MM,
        help="physical lower bound for the fitted cable depth in the pose "
        "frame, millimeters",
    )
    parser.add_argument(
        "--core-mm",
        type=float,
        default=DEFAULTS.core_m * 1000.0,
        help="conductor radius in millimeters; zero leaves the "
        "twin-conductor field unsmoothed",
    )
    parser.add_argument(
        "--pair-spacing-mm",
        type=float,
        default=DEFAULTS.pair_spacing_m * 1000.0,
        help="live/neutral centre-to-centre spacing in millimeters; sets both "
        "the near-field shape of the pair term and the ampere scale",
    )


def _load(
    paths: Sequence[Path], args: CampaignArgs
) -> list[vector.RawCollection]:
    collections: list[vector.RawCollection] = []
    for collection_index, path in enumerate(paths, 1):
        resolved = path.expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"--data must be a directory: {resolved}")
        description = (
            f"Load {collection_index}/{len(paths)}: {resolved.name}"
            if len(paths) > 1
            else f"Load {resolved.name}"
        )
        bar = None

        def progress(index: int, total: int, capture: str) -> None:
            nonlocal bar
            if bar is None:
                bar = tqdm(total=total, desc=description, unit="block")
            bar.set_postfix_str(capture, refresh=False)
            bar.update(index - bar.n)

        try:
            collections.append(
                file_adapter.load_campaign(
                    resolved,
                    block_seconds=args.block_seconds,
                    lever_arm_m=tuple(value / 1000.0 for value in args.lever_mm),
                    sensor_to_camera=np.asarray(
                        args.sensor_rotation, dtype=float
                    ).reshape(3, 3),
                    max_travel_m=args.max_travel_mm / 1000.0,
                    max_along_m=args.max_along_mm / 1000.0,
                    line_hz=args.line_hz,
                    per_block_phase=args.per_block_phase,
                    progress=progress,
                )
            )
        finally:
            if bar is not None:
                bar.close()
    return collections


def _run_analyze(args: AnalyzeArgs) -> None:
    collection = _load([args.data], args)[0]
    _print_lockin(collection)
    analysis = vector.analyze(
        collection,
        vector.Config(
            block_seconds=args.block_seconds,
            core_m=args.core_mm / 1000.0,
            pair_spacing_m=args.pair_spacing_mm / 1000.0,
            vertical_lower_bound_m=-args.z1_height_mm / 1000.0,
            fractional_floor=args.fractional_floor,
        ),
    )
    _print_report(analysis.report)

    out_dir = args.out_dir or args.data.expanduser().resolve() / "vector-lockin-output"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "result.json"
    payload = {
        **analysis.report,
        "line_hz": collection.reference_hz,
        "citation": CITATION,
    }
    path.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n")
    _log_wrote(path)
    if not args.no_figure:
        _write_figure(out_dir, analysis, args.show)


def _run_calibrate_floor(args: FloorArgs) -> None:
    collections = _load(args.data, args)
    result = vector.calibrate_fractional_floor(
        collections,
        vector.CalibrationConfig(
            block_seconds=args.block_seconds,
            core_m=args.core_mm / 1000.0,
            pair_spacing_m=args.pair_spacing_mm / 1000.0,
            vertical_lower_bound_m=-args.z1_height_mm / 1000.0,
            seed_floor=args.seed_floor,
            search_min=args.search_min,
            search_max=args.search_max,
        ),
    )
    encoded = (
        json.dumps(
            {
                **asdict(result),
                "known_healthy_required": True,
                "usage_note": (
                    "Use this fixed floor only for future collections with the "
                    "same frame, positions, acquisition duration, mounting, and "
                    "electrical geometry."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(encoded, end="")
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded)
        _log_wrote(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Complex-vector WireCheck analysis and floor calibration "
        "from iPad VIO campaigns",
        epilog=CITATION,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser(
        "analyze", help="measure, fit, interpret, and plot one campaign"
    )
    analyze.add_argument(
        "--data",
        type=Path,
        required=True,
        help="campaign directory containing raw.csv and pose.csv",
    )
    _add_campaign_options(analyze)
    analyze.add_argument(
        "--fractional-floor",
        type=float,
        default=DEFAULTS.fractional_floor,
        help="fixed fractional model-discrepancy floor",
    )
    analyze.add_argument(
        "--no-figure", action="store_true", help="skip summary figure generation"
    )
    analyze.add_argument(
        "--show", action="store_true", help="display the summary figure interactively"
    )
    analyze.add_argument(
        "--out-dir", type=Path, help="directory for generated results and figures"
    )

    floor = commands.add_parser(
        "calibrate-floor",
        help="estimate a shared fractional model floor from two or more "
        "independent, known-healthy campaigns with the same scan plan",
    )
    floor.add_argument(
        "--data",
        type=Path,
        nargs="+",
        metavar="DIR",
        required=True,
        help="raw healthy campaign directories",
    )
    _add_campaign_options(floor)
    floor.add_argument(
        "--seed-floor",
        type=float,
        default=FLOOR_DEFAULTS.seed_floor,
        help="warm-start floor only; the reported estimate is " "optimized",
    )
    floor.add_argument("--search-min", type=float, default=FLOOR_DEFAULTS.search_min)
    floor.add_argument("--search-max", type=float, default=FLOOR_DEFAULTS.search_max)
    floor.add_argument("--output", type=Path, help="optional JSON output path")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> AnalyzeArgs | FloorArgs:
    namespace = build_parser().parse_args(argv)
    command = cast(str, namespace.command)
    if command == "analyze":
        return AnalyzeArgs.from_namespace(namespace)
    if command == "calibrate-floor":
        return FloorArgs.from_namespace(namespace)
    raise AssertionError(f"unrouted subcommand: {command}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if isinstance(args, AnalyzeArgs):
            _run_analyze(args)
        else:
            _run_calibrate_floor(args)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
