# Naturalizing hydraulic engineering as natural systems change

Analysis code supporting the manuscript *Naturalizing hydraulic engineering as natural systems change*. The repository uses Python 3.11; package versions are listed in `requirements.txt`.

## Score interpretation

The reference-based composite score of natural-system conditions measures selected indicators relative to reference conditions. Four equally weighted domain scores form the system composite score. For river-zone comparisons, each river–reservoir zone response composite summarizes the selected indicators for that zone; it is distinct from the system composite score.

Hydraulic engineering naturalization denotes the conceptual engineering evolution process, not the metric name. Neither a system composite score nor a domain score can itself classify a naturalization stage. Definitions, formulas and statistical specifications follow the Supplementary Methods; the 2003 and 2011 comparisons are diagnostic rather than causal.

## Analysis workflow

| Script | Analysis | Main output |
|---|---|---|
| `01_global_reservoir_river_analysis.py` | Reservoir–river distance analysis using WGS84 and local azimuthal equidistant projections; empirical cumulative distributions | Fig. 1a,b |
| `02_heni_calculation.py` | System composite score: four equally weighted domain scores, Indicators of Hydrologic Alteration, Theil–Sen regression and Jensen–Shannon distance | Fig. 3b,c |
| `03_temporal_diagnosis.py` | Interrupted time-series analysis, HC3-robust inference, stage means and domain contributions | Fig. 3b–d |
| `04_longitudinal_response.py` | River–reservoir zone response composites, LOWESS residual bootstrap, post-2011 trends and paired-year bootstrap comparisons | Fig. 4a–c |
| `05_sensitivity_analysis.py` | Domain-weight sensitivity (1,771 combinations) and four data/reference alternatives | Supplementary Figs. 1,2 |
| `06_figures.py` | Rendering of analytical figure panels from generated tables | Figs. 1, 3 and 4 |

## Installation

```bash
python -m pip install -r requirements.txt
```

## Example workflow

```bash
python code/01_global_reservoir_river_analysis.py --reservoirs /data/GDW_barriers_v1_0.shp --rivers /data/FFR.gdb --river-layer FFR_river_network_v1
python code/02_heni_calculation.py --input-dir /data/authorized
python code/03_temporal_diagnosis.py
python code/04_longitudinal_response.py --input-dir /data/authorized
python code/05_sensitivity_analysis.py --input-dir /data/authorized
python code/06_figures.py --figure 3
python code/06_figures.py --figure 4
```

Replace `/data/...` with local input paths. All scripts accept `--output-dir`; generated files are written to `outputs/` by default. To render Fig. 1, run script 06 with `--figure 1` and the same GIS arguments used for script 01.

## Data and compatibility

Public GIS inputs and the required schema of restricted monitoring inputs are described in [`data/README.md`](data/README.md). Monitoring records are not distributed in this repository; access remains subject to approval by the data owner. System/domain scores, Three Gorges diagnostics and sensitivity analyses require authorized monitoring inputs.

The internal module filename `02_heni_calculation.py`, identifiers such as `system_heni`, numeric columns, output filenames and schemas are retained as legacy implementation aliases for backward compatibility. These aliases do not name the metric in public-facing prose or plots.

The plotting scripts render analytical panels only. Fig. 2, Fig. 3a, conceptual illustrations and the approved publication artwork/layout are separate; these scripts must not replace the frozen submission figures.
