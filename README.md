English | [简体中文](README.zh-CN.md)

# WireCheck Complex-Vector Analysis

Estimate cable position, load current, and leakage current from iPad VIO poses and three-axis magnetic recordings, with a fixed-threshold health verdict.

One campaign is one directory holding two CSVs:

- `raw.csv`: the three-axis magnetic recording;
- `pose.csv`: iPad (ARKit) visual-inertial odometry (VIO) poses at about 60 Hz.

`WireCheck` cuts the continuous magnetic recording into fixed-length lock-in blocks, uses the VIO pose at each block midpoint as that measurement's position, rotates the block's sensor samples into one common frame, extracts the complex mains (50/60 Hz) line phasor per block, and fits one physical model:

- the exact near field of a twin-conductor pair (live + neutral antiparallel line currents);
- a monopole leakage term `C / r`, carried as Cartesian in-phase/quadrature components;
- a uniform coherent background field `U` (one complex three-vector), treated as a nuisance.

The fit yields the cable-centre position, the pair current, and the leakage magnitude and relative phase, from which the verdict follows: `no_cable`, `healthy`, `leakage_suspected`, or `leakage_detected`.

## Requirements

- Python >= 3.10
- Runtime dependencies: `numpy >= 1.24`, `scipy >= 1.11`, `matplotlib >= 3.7`, `tqdm >= 4.66`
- Optional acceleration: `numba >= 0.60`. It only accelerates the residual and Jacobian kernels; when it is absent the equivalent NumPy path is used and the results are unchanged.

## Installation

From the repository root:

```bash
# regular install
pip install .

# development (editable) install
pip install -e .

# with numba acceleration
pip install ".[speed]"
```

The install provides the `wirecheck` command:

```bash
wirecheck --help
```

### Building a wheel from source

```bash
python -m pip install build
python -m build
python -m pip install dist/magcomm_wirecheck-0.1.0-py3-none-any.whl
```

## Data formats

### 1. `raw.csv`: magnetic recording

Comma-separated, with a header row. It must contain the following columns (order is free, and the `sample` column is optional):

| Column | Meaning |
| --- | --- |
| `t_host` | sample time in seconds since the campaign started; the same clock as `pose.csv` |
| `t_mcu` | microcontroller time in seconds; must increase strictly |
| `x` | raw X-axis ADC count |
| `y` | raw Y-axis ADC count |
| `z` | raw Z-axis ADC count |

- The ADC is 18-bit with midpoint `2^17 = 131072`. Raw counts convert to nanoteslas as:

  ```text
  B[nT] = (raw - 131072) * (100000 / 16384)   # about 6.1035 nT / LSB
  ```

- Startup crop: samples with `t_mcu <= 0.1 s` are discarded.
- The sample rate is estimated as `1 / median(diff(t_mcu))` and sets the number of samples per block.

Example:

```csv
sample,t_host,t_mcu,x,y,z
0,0.200000,0.200000,131080,131071,131090
1,0.200800,0.200800,131082,131069,131088
```

### 2. `pose.csv`: iPad VIO poses

Comma-separated, with a header row, containing:

| Column | Meaning |
| --- | --- |
| `t_host` | pose time in seconds; the same clock as `raw.csv`, and must increase strictly |
| `x` | camera position X in metres |
| `y` | camera position Y in metres (ARKit world coordinates, +Y up) |
| `z` | camera position Z in metres |
| `qw` `qx` `qy` `qz` | camera orientation quaternion (ARKit convention) |

The loader removes quaternion sign flips between consecutive frames and linearly interpolates to each block midpoint by `t_host`.

Example:

```csv
t_host,x,y,z,qw,qx,qy,qz
0.200000,0.000000,0.040000,0.100000,1,0,0,0
0.216667,0.000000,0.040000,0.100000,1,0,0,0
```

### 3. Campaign directory

`--data` must be a directory that contains both `raw.csv` and `pose.csv`. One directory is one collection.

## Position and coordinate frame

Positions **no longer come from filenames**; they come from time interpolation of `pose.csv`:

- The recording is cut into non-overlapping lock-in blocks of fixed length (`--block-seconds`, default 0.5 s); each block becomes one measurement.
- Each block takes the camera pose at its **midpoint** as its position.
- The analysis plane is defined by the **first valid pose**:
  - horizontal axis: the first camera-forward vector projected onto the horizontal (world XZ) plane;
  - vertical axis: world +Y;
  - along-cable axis: their cross product.
- Each block's position is the mid-block sensor position projected onto that analysis plane (horizontal and vertical components); the along-cable component only feeds the gates and never enters the fit.
- Each block's three-axis samples are rotated into one common frame using the mid-block camera orientation, so the phasors stay in a single frame even when the operator turns the iPad.
- The lever arm and sensor orientation are given by `--lever-mm` and `--sensor-rotation`; the defaults are `0,0,0` and the identity matrix.

Two gates discard blocks that would break the "one fixed position per measurement" assumption:

- `--max-travel-mm` (default 50): discard a block when the camera moved more than this within it;
- `--max-along-mm` (default 200): discard a block when it is this far along the cable axis, keeping the sweep within one cross-section.

## Usage

The command line has two subcommands that share one campaign option group.

### analyze: analyze a single campaign

```bash
wirecheck analyze --data path/to/campaign
```

Common options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--block-seconds` | 0.5 | lock-in block length in seconds; one block = one measurement |
| `--lever-mm X Y Z` | `0 0 0` | camera-to-sensor lever arm in mm, in camera axes |
| `--sensor-rotation R00,...,R22` | identity | row-major 3×3 rotation from sensor axes to camera axes |
| `--max-travel-mm` | 50 | discard a block when the camera moved more than this |
| `--max-along-mm` | 200 | discard a block when it is this far along the cable axis |
| `--z1-height-mm` | 10000 | physical lower bound for the fitted cable depth (negated to form the bound), in mm |
| `--core-mm` | 0 | conductor radius in mm; 0 uses the unsmoothed exact twin-conductor field |
| `--pair-spacing-mm` | 5.5 | live/neutral centre-to-centre spacing in mm; sets both the near-field shape and the current scale |
| `--fractional-floor` | 0.094 | fixed model-error fraction (combined in quadrature with the statistical error) |
| `--no-figure` | off | do not generate the diagnostic figure |
| `--show` | off | display the diagnostic figure interactively |
| `--out-dir DIR` | none | output directory |

Output:

- stdout: a text report;
- `<data>/vector-lockin-output/result.json`: the full result;
- `<data>/vector-lockin-output/figures/<collection>_summary.png`: the diagnostic figure (unless `--no-figure`).

### calibrate-floor: calibrate the fractional floor

Estimate one shared, fixed `fractional_floor` from two or more independent, known-healthy campaigns that share the same scan plan and coordinate frame:

```bash
wirecheck calibrate-floor --data healthy-1 healthy-2 [healthy-3 ...]
```

Optional arguments: `--seed-floor`, `--search-min`, `--search-max`, `--output FILE`. The result is printed to stdout and can also be written to a JSON file with `--output`.

Every participating campaign must use the same coordinate frame and the same set of positions; each is fitted independently, and only `fractional_floor` is shared.

## Verdict rules

The verdict uses fixed thresholds:

| Condition | Verdict |
| --- | --- |
| fitted load current < 0.1 A | `no_cable` |
| leakage current < 12 mA | `healthy` |
| 12 mA ≤ leakage current < 18 mA | `leakage_suspected` |
| leakage current ≥ 18 mA | `leakage_detected` |

The leakage current follows from the leakage coefficient `C` (in nT·m): `I_leak[mA] = 5 × |C|`. The pair current is `I_pair[A] = 5 × a / pair_spacing[mm]`, where `a` is the fitted pair coefficient (in nT·m²). The load current is `I_pair + I_leak / 2000`.

## Running without installation

The package must be importable as `wirecheck`, and this repository directory *is* that package. So, without installing, a directory named `wirecheck` must appear on `PYTHONPATH`.

Option 1: create a symlink anywhere.

```bash
# run from the repository root
mkdir -p /tmp/wc && ln -sfn "$PWD" /tmp/wc/wirecheck
PYTHONPATH=/tmp/wc python -m wirecheck.wirecheck_cli analyze --data path/to/campaign
```

Option 2: rename the repository directory to `wirecheck`, then run from its parent:

```bash
python -m wirecheck.wirecheck_cli --help
```

Neither option installs anything or changes the source.

## Python API

```python
from wirecheck import file_adapter, vector

collection = file_adapter.load_campaign(
    "path/to/campaign",
    block_seconds=0.5,
    lever_arm_m=(0.0, 0.0, 0.0),
)
analysis = vector.analyze(collection, vector.Config(block_seconds=0.5))

print(analysis.report["health"])
print(analysis.report["estimated_leak_current_ma"])
figure = analysis.make_figure()
```

To rotate the sensor samples into the camera frame, pass a 3×3 matrix as `sensor_to_camera=` to `load_campaign`.

## Repository layout

```text
.
├── pyproject.toml
├── LICENSE                 # GPL-3.0-or-later
├── wirecheck_cli.py        # command-line entry point
├── file_adapter.py         # raw.csv / pose.csv loading, clock and frame transforms
└── vector/                 # format-independent analysis implementation
    ├── defs.py             # structural protocols
    ├── rdata.py            # raw recordings and capture contracts
    ├── preproc.py          # lock-in spectra (multi-block average and single block)
    ├── meas.py             # normalized measurements and collections
    ├── core.py             # coefficient fitting and handedness selection
    ├── _vector_kernel.py   # optional numba residual/Jacobian kernels
    ├── calibrate.py        # fractional-floor calibration
    ├── parse.py            # hard thresholds, quality gates, verdicts
    ├── plot.py             # plotting
    └── wirecheck.py        # service interface composing the modules
```

## Assumptions and limitations

- The cable runs roughly horizontally and the sweep stays in a plane approximately perpendicular to it.
- The iPad VIO provides camera poses; the sensor-to-camera rotation and lever arm must be calibrated (`--sensor-rotation`, `--lever-mm`), defaulting to identity and zero.
- `raw.csv` and `pose.csv` share the `t_host` clock.
- Each lock-in block is approximately static; blocks with more than `--max-travel-mm` of motion are discarded.
- Relative axis I/Q phase is preserved; axis delays are negligible.
- The coherent background field is uniform and phase-stable.
- `pair_spacing` sets both the near-field shape and the current scale and must match the real geometry.
- Quality gates were tuned on limited data; recalibrate `fractional_floor` when the geometry or mounting changes.

## Citation

If this tool is useful in your research, please cite our demo paper:

```text
To be added.
```

## License

This project is released under the **GNU General Public License v3.0 or later (GPL-3.0-or-later)**. See [LICENSE](LICENSE) for the full terms.
