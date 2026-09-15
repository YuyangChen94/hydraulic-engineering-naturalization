"""Calculate reservoir-to-reach distances using WGS84 and local AEQD projections."""

from __future__ import annotations
import math
import importlib
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import time
import geopandas as gpd
import shapely
from pyproj import CRS, Geod, Transformer
from shapely.geometry import LineString, MultiLineString, Point, box
from shapely.ops import nearest_points
from shapely.strtree import STRtree

heni = importlib.import_module("02_heni_calculation")
WGS84 = "EPSG:4326"

DISPLAY_CRS = "EPSG:8857"

GEOD = Geod(ellps="WGS84")

CAPACITY_THRESHOLD_MCM = 100.0

LONG_RIVER_THRESHOLD_KM = 1000.0

HIGH_CONNECTIVITY_CODES = (1, 2)

RESTRICTED_CODE = 3

EXPECTED_RESERVOIRS = 4262

EXPECTED_REACHES = 124227

EXPECTED_RIVERS = 246

EXPECTED_STATUS_COUNTS = {1: 43683, 2: 21636, 3: 58908}

DENSIFY_MAX_SEGMENT_KM = 2.0

SEARCH_CIRCLE_BEARINGS = 360

SEARCH_ENVELOPE_PADDING_FACTOR = 1.01

SEARCH_ENVELOPE_PADDING_KM = 1.0

MIN_SEARCH_RADIUS_KM = 10.0

MAX_INITIAL_RADIUS_KM = 1000.0

MAX_SEARCH_RADIUS_KM = 20000.0

INITIAL_RADIUS_FACTOR = 1.05

INITIAL_RADIUS_MARGIN_KM = 10.0

def _densify_linestring(line: LineString, max_segment_km: float) -> LineString:
    coords = list(line.coords)
    if len(coords) < 2:
        return line
    out = [coords[0][:2]]
    max_m = max_segment_km * 1000.0
    for a, b in zip(coords[:-1], coords[1:]):
        lon1, lat1 = a[:2]
        lon2, lat2 = b[:2]
        _, _, d_m = GEOD.inv(lon1, lat1, lon2, lat2)
        n_insert = max(0, int(math.ceil(d_m / max_m)) - 1)
        if n_insert:
            out.extend(GEOD.npts(lon1, lat1, lon2, lat2, n_insert))
        out.append((lon2, lat2))
    return LineString(out)


def densify_geometry(geom, max_segment_km: float):
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == "LineString":
        return _densify_linestring(geom, max_segment_km)
    if geom.geom_type == "MultiLineString":
        return MultiLineString(
            [_densify_linestring(part, max_segment_km) for part in geom.geoms]
        )
    raise TypeError(f"Unsupported river geometry type: {geom.geom_type}")


def densify_array(geometries, max_segment_km: float, log: logging.Logger, label: str):
    t0 = time.time()
    out = np.asarray(
        [densify_geometry(g, max_segment_km) for g in geometries], dtype=object
    )
    log.info(
        "Geodesic densification completed for %s: n=%d, max_segment=%.3f km, seconds=%.2f",
        label,
        len(out),
        max_segment_km,
        time.time() - t0,
    )
    return out


def geodesic_circle_bboxes(
    lon0: float, lat0: float, radius_km: float, n_bearings: int = SEARCH_CIRCLE_BEARINGS
):
    """Return one/two lon-lat boxes enclosing a WGS84 geodesic circle.

    Longitude is unwrapped relative to the reservoir longitude, then split at
    the antimeridian when necessary. Sampling many bearings is deliberately
    conservative for a candidate-search envelope; final distances are not
    computed from this envelope.
    """
    envelope_radius_km = (
        radius_km * SEARCH_ENVELOPE_PADDING_FACTOR + SEARCH_ENVELOPE_PADDING_KM
    )
    az = np.linspace(0.0, 360.0, n_bearings, endpoint=False)
    lons, lats, _ = GEOD.fwd(
        np.full(n_bearings, lon0),
        np.full(n_bearings, lat0),
        az,
        np.full(n_bearings, envelope_radius_km * 1000.0),
    )
    lat_min = max(-90.0, float(np.nanmin(lats)))
    lat_max = min(90.0, float(np.nanmax(lats)))
    deltas = (lons - lon0 + 180.0) % 360.0 - 180.0
    dmin = float(np.nanmin(deltas))
    dmax = float(np.nanmax(deltas))
    if dmax - dmin > 350.0 or lat_max >= 89.999999 or lat_min <= -89.999999:
        return [box(-180.0, lat_min, 180.0, lat_max)]
    lo = lon0 + dmin
    hi = lon0 + dmax
    if lo < -180.0:
        return [
            box(lo + 360.0, lat_min, 180.0, lat_max),
            box(-180.0, lat_min, hi, lat_max),
        ]
    if hi > 180.0:
        return [
            box(lo, lat_min, 180.0, lat_max),
            box(-180.0, lat_min, hi - 360.0, lat_max),
        ]
    return [box(lo, lat_min, hi, lat_max)]


def _candidate_ids(
    tree: STRtree, lon0: float, lat0: float, radius_km: float
) -> np.ndarray:
    ids = set()
    for q in geodesic_circle_bboxes(lon0, lat0, radius_km):
        ids.update((int(i) for i in tree.query(q)))
    return np.asarray(sorted(ids), dtype=int)


def _local_aeqd_transformers(lon0: float, lat0: float):
    crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0:.12f} +lon_0={lon0:.12f} +datum=WGS84 +units=m +no_defs"
    )
    fwd = Transformer.from_crs(WGS84, crs, always_xy=True, force_over=True)
    inv = Transformer.from_crs(crs, WGS84, always_xy=True, force_over=True)
    return (fwd, inv)


def nearest_distance_local_aeqd(
    lon0: float,
    lat0: float,
    tree: STRtree,
    dense_geometries: np.ndarray,
    old_distance_km: float,
):
    initial = max(
        MIN_SEARCH_RADIUS_KM,
        min(
            MAX_INITIAL_RADIUS_KM,
            float(old_distance_km) * INITIAL_RADIUS_FACTOR + INITIAL_RADIUS_MARGIN_KM,
        ),
    )
    radius = initial
    origin = Point(0.0, 0.0)
    while radius <= MAX_SEARCH_RADIUS_KM:
        ids = _candidate_ids(tree, lon0, lat0, radius)
        if len(ids):
            fwd, inv = _local_aeqd_transformers(lon0, lat0)
            projected = shapely.transform(
                dense_geometries[ids], fwd.transform, interleaved=False
            )
            distances_m = shapely.distance(origin, projected)
            k = int(np.nanargmin(distances_m))
            best_m = float(distances_m[k])
            if best_m <= radius * 1000.0:
                best_geom = projected[k]
                nearest_proj = nearest_points(origin, best_geom)[1]
                near_lon, near_lat = inv.transform(nearest_proj.x, nearest_proj.y)
                _, _, geod_m = GEOD.inv(lon0, lat0, near_lon, near_lat)
                return {
                    "distance_km": best_m / 1000.0,
                    "geod_check_km": geod_m / 1000.0,
                    "closest_reach_index": int(ids[k]),
                    "search_radius_km": float(radius),
                    "candidate_count": int(len(ids)),
                    "nearest_lon": float(near_lon),
                    "nearest_lat": float(near_lat),
                }
        radius *= 2.0
    raise RuntimeError(
        f"No nearest river candidate resolved within {MAX_SEARCH_RADIUS_KM:g} km for reservoir ({lon0:.6f}, {lat0:.6f})."
    )


def run_class_distances(
    reservoirs: gpd.GeoDataFrame,
    river_block: gpd.GeoDataFrame,
    old_distances_km: np.ndarray,
    dense_km: float,
    label: str,
    log: logging.Logger,
):
    dense = densify_array(river_block.geometry.to_numpy(), dense_km, log, label)
    tree = STRtree(dense)
    rows = []
    t0 = time.time()
    for i, geom in enumerate(reservoirs.geometry):
        res = nearest_distance_local_aeqd(
            geom.x, geom.y, tree, dense, float(old_distances_km[i])
        )
        res["reservoir_row"] = i
        rows.append(res)
        if (i + 1) % 250 == 0 or i + 1 == len(reservoirs):
            log.info("%s progress: %d/%d", label, i + 1, len(reservoirs))
    log.info("%s final distances completed in %.2f s", label, time.time() - t0)
    return (pd.DataFrame(rows), dense, tree)


def ecdf(arr):
    x = np.sort(np.asarray(arr, dtype=float))
    y = np.arange(1, len(x) + 1) / len(x) * 100.0
    return (x, y)


def load_public_layers(reservoir_path, river_path, river_layer=None):
    import pyogrio

    logging.info("Reading GDW barrier points")
    fields = pyogrio.read_info(reservoir_path)["fields"]
    selected = [
        c
        for c in fields
        if c.upper() in {"GDW_ID", "CAP_MCM"}
        or c in {"barrier_id", "reservoir_capacity_mcm"}
    ]
    reservoirs = gpd.read_file(reservoir_path, columns=selected).to_crs(WGS84)
    mapping = {"GDW_ID": "barrier_id", "CAP_MCM": "reservoir_capacity_mcm"}
    reservoirs = reservoirs.rename(
        columns={
            c: mapping[c.upper()] for c in reservoirs.columns if c.upper() in mapping
        }
    )
    heni.require_columns(reservoirs, ["barrier_id", "reservoir_capacity_mcm"], "GDW")
    logging.info("Reading the FFR statistical subset")
    fields = (
        ["BB_ID", "BB_LEN_KM", "INC", "CSI_FF2"]
        if river_layer
        else [
            "river_id",
            "river_length_km",
            "included_in_statistical_analysis",
            "free_flowing_status_code",
        ]
    )
    rivers = gpd.read_file(
        river_path,
        layer=river_layer,
        columns=fields,
        **({"where": "INC = 1 AND BB_LEN_KM > 1000"} if river_layer else {}),
    ).to_crs(WGS84)
    rivers = rivers.rename(
        columns={
            "BB_ID": "river_id",
            "BB_LEN_KM": "river_length_km",
            "INC": "included_in_statistical_analysis",
            "CSI_FF2": "free_flowing_status_code",
        }
    )
    heni.require_columns(
        rivers,
        [
            "river_id",
            "river_length_km",
            "included_in_statistical_analysis",
            "free_flowing_status_code",
        ],
        "FFR",
    )
    capacity = pd.to_numeric(reservoirs.reservoir_capacity_mcm, errors="coerce").mask(
        lambda x: x.isin([-99, -999, -9999])
    )
    reservoirs = reservoirs.loc[capacity.ge(CAPACITY_THRESHOLD_MCM)].reset_index(
        drop=True
    )
    status = pd.to_numeric(rivers.free_flowing_status_code, errors="coerce")
    rivers = rivers.loc[
        pd.to_numeric(rivers.included_in_statistical_analysis, errors="coerce").eq(1)
        & pd.to_numeric(rivers.river_length_km, errors="coerce").gt(
            LONG_RIVER_THRESHOLD_KM
        )
        & status.isin([1, 2, 3])
    ].reset_index(drop=True)
    status = pd.to_numeric(rivers.free_flowing_status_code)
    counts = status.value_counts().sort_index().astype(int).to_dict()
    if (len(reservoirs), len(rivers), rivers.river_id.nunique(), counts) != (
        EXPECTED_RESERVOIRS,
        EXPECTED_REACHES,
        EXPECTED_RIVERS,
        EXPECTED_STATUS_COUNTS,
    ):
        raise ValueError(
            "Public source release/filter counts differ from the paper; do not change the scientific filters."
        )
    if (
        reservoirs.barrier_id.duplicated().any()
        or reservoirs.geometry.is_empty.any()
        or rivers.geometry.is_empty.any()
    ):
        raise ValueError("Duplicate reservoir IDs or empty geometries")
    rivers["status_binary"] = np.where(
        status.eq(3), "flow_restricted", "high_connectivity"
    )
    logging.info(
        "Input filters verified: %d reservoirs, %d reaches",
        len(reservoirs),
        len(rivers),
    )
    return reservoirs, rivers


def main():
    parser = heni.cli(
        "WGS84/local-AEQD reservoir-to-reach distances from public GDW and FFR geometries."
    )
    parser.add_argument(
        "--reservoirs",
        type=Path,
        required=True,
        help="GDW v1.0 barrier point shapefile or mapped GeoPackage",
    )
    parser.add_argument(
        "--rivers",
        type=Path,
        required=True,
        help="Official FFR geodatabase or mapped GeoPackage",
    )
    parser.add_argument("--river-layer", help="FFR geodatabase layer name")
    args = parser.parse_args()
    out = heni.runtime(args, "01_global_reservoir_river_analysis")
    log = logging.getLogger(__name__)
    reservoirs, rivers = load_public_layers(
        args.reservoirs, args.rivers, args.river_layer
    )
    result = pd.DataFrame({"reservoir_id": reservoirs.barrier_id.astype(str)})
    ecdfs = []
    for label in ["high_connectivity", "flow_restricted"]:
        block = rivers.loc[rivers.status_binary.eq(label)].reset_index(drop=True)
        # Build the search seed in an equal-area projection; final distances use AEQD.
        seeds = (
            STRtree(block.to_crs(DISPLAY_CRS).geometry.to_numpy()).query_nearest(
                reservoirs.to_crs(DISPLAY_CRS).geometry.to_numpy(),
                return_distance=True,
                all_matches=False,
            )[1]
            / 1000
        )
        distances, _, _ = run_class_distances(
            reservoirs, block, seeds, DENSIFY_MAX_SEGMENT_KM, label, log
        )
        result["distance_to_" + label + "_km"] = distances.distance_km
        heni.save(distances, out, "01_" + label + "_distance_checks.csv")
        x, y = ecdf(distances.distance_km)
        ecdfs.append(
            pd.DataFrame(
                {"river_class": label, "distance_km": x, "cumulative_reservoirs_pct": y}
            )
        )
    heni.save(result, out, "01_reservoir_distances.csv")
    heni.save(pd.concat(ecdfs, ignore_index=True), out, "01_ecdf.csv")
    logging.info("Completed 4262 reservoirs and both reach classes")


if __name__ == "__main__":
    main()
