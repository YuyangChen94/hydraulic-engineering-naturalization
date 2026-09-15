# Data inputs

## Public GIS data

The global analysis uses two public datasets:

- Global Dam Watch v1.0: https://doi.org/10.6084/m9.figshare.25988293
- Global free-flowing rivers v1.0: https://doi.org/10.6084/m9.figshare.7688801

Use the Global Dam Watch barrier points and the free-flowing-rivers layer `FFR_river_network_v1`. Raw and processed geometries are not redistributed in this repository.

| Source field | Code field | Selection |
|---|---|---|
| GDW `GDW_ID` | `barrier_id` | Identifier |
| GDW `CAP_MCM` | `reservoir_capacity_mcm` | ≥100 million m³ |
| FFR `BB_ID` | `river_id` | Identifier |
| FFR `BB_LEN_KM` | `river_length_km` | >1,000 km |
| FFR `INC` | `included_in_statistical_analysis` | 1 |
| FFR `CSI_FF2` | `free_flowing_status_code` | 1/2: high connectivity; 3: flow restricted |

The retained dataset contains 4,262 reservoirs, 124,227 reaches and 246 rivers. Status-code counts are 43,683 (1), 21,636 (2) and 58,908 (3).

## Restricted monitoring data

The Three Gorges analyses require authorized monitoring records that are not distributed with the repository. Data-access requests may be directed to wang_dianchang@ctg.com.cn and remain subject to approval by the data owner.

Provide four UTF-8 CSV files through `--input-dir`:

| Filename | Resolution | Unit |
|---|---|---|
| `01_flow_daily_selected.csv` | Daily discharge | m³/s |
| `02_sediment_daily_selected.csv` | Daily suspended-sediment concentration | kg/m³ |
| `03_water_quality_monthly_selected.csv` | Monthly TP, TN and DO | mg/L |
| `04_ecology_annual_selected.csv` | Annual four-major-carp fry abundance | 10^8 fry (primary); 10^8 eggs and fry (alternative) |

All files require `date`, `year`, `station`, `value`, `unit` and `data_version`. The first three also require `month`; water-quality and ecology files require `variable`. Dates use `YYYY-MM-DD`. Missing observations should be blank or `NA`, not zero.

| Input | Required stations / series |
|---|---|
| Discharge | Cuntan, Wulong, Yichang; primary `primary_flow`, alternative Yichang `alternative_yichang_flow` |
| Sediment | Cuntan, Wulong, Yichang; `primary_sediment` |
| Water quality | Cuntan, Qingxichang, Tuokou, Guandukou, Hankou; `primary_water_quality`; variables `TP`, `TN`, `DO` |
| Ecology | Primary: Jianli, `primary_ecology`; alternative: Jianli_section, `alternative_ecology` |

Inputs must be preprocessed to the English schema before running the public code. Source-data conversion is outside the scope of this repository. Ecology variable names are `four_major_carp_reproduction` (primary) and `four_major_carp_eggs_and_fry_runoff` (alternative). Station grouping uses a fixed order to preserve numerical reductions.

Optional fields are `day`, `is_missing`, `source_file`, `source_sheet`, `source_column` and `notes`. Any supplied metadata must use English. `year`, `month` and `day` are calendar integers; `value` is numeric; `station`, `variable`, `unit` and `data_version` are strings. The unit strings are `m3/s`, `kg/m3`, `mg/L`, `10^8 fry` and `10^8 eggs and fry`, respectively.

The diagnostic period is 1998–2017. Reference periods are 1960–1994 for discharge and sediment, 1998–2002 for TP, 1997–2001 for TN/DO and 1997–2002 for the primary ecology series. The historical ecology sensitivity uses the four available observations from 1960–1994; the alternative fish series leaves 2007 missing.

Coverage and aggregation rules are implemented in scripts 02 and 04. No unit conversion is applied to restricted inputs; supplied units must match the source records used for the manuscript analysis.

TP, TN and DO use mg/L. The alternative four-major-carp series uses 10^8 eggs and fry. No numerical rescaling is applied.
