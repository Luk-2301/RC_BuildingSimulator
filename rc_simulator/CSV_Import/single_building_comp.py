import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
RC_HOURLY_RELATIVE_DIR = Path("Rc_Hourly_Results") / "03_medium"
RC_HOURLY_DIR = DATA_DIR / RC_HOURLY_RELATIVE_DIR
SIMSTADT_HOURLY_DIR = DATA_DIR / "SimStadt_Hourly_Heat_Demand"

# Keep plotting colors consistent with simulation_comparison.py
SIMSTADT_PLOT_COLOR = "#C65D3B"
RC_PLOT_COLOR = "#3A7CA5"
RC_SOLAR_PLOT_COLOR = "#F4D03F"
RC_COOLING_PLOT_COLOR = "#2ca02c"

RC_INDOOR_PLOT_COLOR = "#2ca02c"
RC_OUTSIDE_PLOT_COLOR = "#6E7F80"

DEFAULT_WINDOW_HOURS = 24 * 14
DEFAULT_CALENDAR_YEAR = 2013
# Seasonal window start dates for 14-day periods
DEFAULT_WINTER_START_MONTH = 1
DEFAULT_WINTER_START_DAY = 28
DEFAULT_SPRING_START_MONTH = 4
DEFAULT_SPRING_START_DAY = 7
DEFAULT_SUMMER_START_MONTH = 8
DEFAULT_SUMMER_START_DAY = 15
DEFAULT_WINDOW_START_CLOCK_HOUR = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select representative buildings based on compactness, window/heated-area "
            "ratio, usage type, and annual heating demand."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Base data directory (defaults to CSV_Import/data).",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=10,
        help="Number of representative buildings to select (default: 10).",
    )
    parser.add_argument(
        "--selection-usage",
        default="residential",
        help=(
            "Usage type to select representatives from (default: residential). "
            "Use 'all' to select from all usage types."
        ),
    )
    parser.add_argument(
        "--window-hours",
        type=int,
        default=DEFAULT_WINDOW_HOURS,
        help="Number of plotted hours (default: 336 = 14 days).",
    )
    parser.add_argument(
        "--winter-start-month",
        type=int,
        default=DEFAULT_WINTER_START_MONTH,
        help="Start month for the 14-day winter plot window (default: 12).",
    )
    parser.add_argument(
        "--winter-start-day",
        type=int,
        default=DEFAULT_WINTER_START_DAY,
        help="Start day for the 14-day winter plot window (default: 1).",
    )
    parser.add_argument(
        "--spring-start-month",
        type=int,
        default=DEFAULT_SPRING_START_MONTH,
        help="Start month for the 14-day spring plot window (default: 4).",
    )
    parser.add_argument(
        "--spring-start-day",
        type=int,
        default=DEFAULT_SPRING_START_DAY,
        help="Start day for the 14-day spring plot window (default: 1).",
    )
    parser.add_argument(
        "--summer-start-month",
        type=int,
        default=DEFAULT_SUMMER_START_MONTH,
        help="Start month for the 14-day summer plot window (default: 7).",
    )
    parser.add_argument(
        "--summer-start-day",
        type=int,
        default=DEFAULT_SUMMER_START_DAY,
        help="Start day for the 14-day summer plot window (default: 1).",
    )
    parser.add_argument(
        "--window-start-clock-hour",
        type=int,
        default=DEFAULT_WINDOW_START_CLOCK_HOUR,
        help="Start clock hour (0-23) for the 14-day plot window (default: 0).",
    )
    parser.add_argument(
        "--calendar-year",
        type=int,
        default=DEFAULT_CALENDAR_YEAR,
        help="Year used to label 24h date ticks on x-axis (default: 2013).",
    )
    return parser.parse_args()


def parse_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )


def get_numeric_series(df: pd.DataFrame, column_name: str) -> pd.Series:
    series = df.get(column_name)
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    if np.isnan(mad) or mad == 0:
        return np.zeros_like(values)
    return 0.6745 * (values - median) / mad


def load_feature_table(data_dir: Path) -> pd.DataFrame:
    annual_results_path = data_dir / "Rc_Hourly_Aggregated_Results" / "simulated_annual_results_03_medium.csv"
    input_snapshot_path = data_dir / "Rc_Hourly_Aggregated_Results" / "simulated_input_snapshot_03_medium.csv"
    building_din_path = data_dir / "ImBuchwald_DIN18599_HEATING_AND_COOLING.csv"

    annual = pd.read_csv(annual_results_path)
    snapshot = pd.read_csv(input_snapshot_path)
    din = pd.read_csv(building_din_path, sep=";", comment="#", dtype=str)

    # Remove units row.
    din = din[din["GMLId"] != "[-]"].copy()

    din["s_to_v"] = parse_numeric(din["Surface area to volume ratio"])
    din["compactness"] = np.where(din["s_to_v"] > 0, 1.0 / din["s_to_v"], np.nan)

    snapshot_columns = [
        "Building ID",
        "Window area [m2]",
        "Heated area [m2]",
        "Heated volume [m3]",
        "U walls [W/m2K]",
        "U windows [W/m2K]",
    ]
    if "Window-to-Wall ratio [-]" in snapshot.columns:
        snapshot_columns.append("Window-to-Wall ratio [-]")

    merged = annual.merge(
        snapshot[snapshot_columns],
        on="Building ID",
        how="left",
        suffixes=("", "_snapshot"),
    ).merge(
        din[["GMLId", "compactness", "BuildingType"]],
        left_on="Building ID",
        right_on="GMLId",
        how="left",
    )

    merged["usage_type"] = merged["Usage type"].astype(str).str.strip().str.lower()
    merged["building_type"] = merged["BuildingType"].astype(str).str.strip()
    merged["annual_heat_kwh"] = pd.to_numeric(
        merged["Annual Heating Demand Sim [kWh]"], errors="coerce"
    )
    merged["heated_area_m2"] = pd.to_numeric(merged["Heated area [m2]"], errors="coerce")
    merged["heated_volume_m3"] = pd.to_numeric(merged["Heated volume [m3]"], errors="coerce")
    merged["window_area_m2"] = pd.to_numeric(merged["Window area [m2]"], errors="coerce")
    merged["u_walls_w_m2k"] = pd.to_numeric(merged["U walls [W/m2K]"], errors="coerce")
    merged["u_windows_w_m2k"] = pd.to_numeric(merged["U windows [W/m2K]"], errors="coerce")
    if "Window-to-Wall ratio [-]" in merged.columns:
        merged["window_to_wall"] = pd.to_numeric(merged["Window-to-Wall ratio [-]"], errors="coerce")
    else:
        merged["window_to_wall"] = np.nan
    merged["window_to_heated"] = np.where(
        merged["heated_area_m2"] > 0,
        merged["window_area_m2"] / merged["heated_area_m2"],
        np.nan,
    )

    return merged


def load_annual_comparison_table(data_dir: Path) -> pd.DataFrame:
    comparison_path = data_dir / "comparisons_individual" / "annual_comparison_per_building.csv"
    if not comparison_path.exists():
        return pd.DataFrame()
    return pd.read_csv(comparison_path)


def load_building_comparison_heating(data_dir: Path) -> pd.DataFrame:
    """Load per-building annual RC vs SimStadt heating from comparison file."""
    comparison_path = data_dir / "comparisons_individual" / "per_building_hourly_comparison_03_medium.csv"
    if not comparison_path.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(comparison_path)
    
    # Group by building and aggregate annual heating (sum if multiple matches per building)
    annual = df.groupby("rc_building_id").agg({
        "annual_rc_kwh": "sum",
        "annual_simstadt_kwh": "sum"
    }).reset_index()
    annual = annual.rename(columns={"rc_building_id": "Building ID"})
    
    annual["annual_diff_kwh"] = annual["annual_rc_kwh"] - annual["annual_simstadt_kwh"]
    annual["annual_diff_pct"] = np.where(annual["annual_simstadt_kwh"].abs() > 0, 100.0 * annual["annual_diff_kwh"] / annual["annual_simstadt_kwh"], np.nan)
    return annual


def mark_outliers(df: pd.DataFrame) -> pd.DataFrame:
    features = ["compactness", "window_to_heated", "annual_heat_kwh"]
    df = df.copy()
    df["is_outlier"] = False

    for usage, idx in df.groupby("usage_type").groups.items():
        part = df.loc[idx, features].copy()
        z_scores = pd.DataFrame(index=part.index)
        for feature in features:
            z_scores[feature] = robust_z(part[feature].to_numpy(dtype=float))

        cond_very_extreme = (np.abs(z_scores) > 3.5).any(axis=1)
        cond_multi_extreme = (np.abs(z_scores) > 2.5).sum(axis=1) >= 2
        df.loc[idx, "is_outlier"] = (cond_very_extreme | cond_multi_extreme)

    return df


def mark_heating_mismatch_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _pick_col(patterns: list[str]) -> str | None:
        return next(
            (
                col
                for col in df.columns
                for pattern in patterns
                if re.search(pattern, col, re.I)
            ),
            None,
        )

    rc_col = "annual_rc_kwh" if "annual_rc_kwh" in df.columns else _pick_col(
        [r"\brc\b.*heat.*kwh", r"\brc\b.*kwh", r"annual.*rc", r"rc_annual"]
    )
    sim_col = "annual_simstadt_kwh" if "annual_simstadt_kwh" in df.columns else _pick_col(
        [r"simstadt.*heat.*kwh", r"simstadt.*kwh", r"annual.*simstadt", r"simstadt_annual"]
    )

    df["annual_rc_kwh"] = pd.to_numeric(df.get(rc_col, np.nan), errors="coerce")
    df["annual_simstadt_kwh"] = pd.to_numeric(df.get(sim_col, np.nan), errors="coerce")

    annual_diff = pd.to_numeric(df.get("annual_diff_kwh", pd.Series(index=df.index, dtype=float)), errors="coerce")
    annual_diff = annual_diff.fillna(df["annual_rc_kwh"] - df["annual_simstadt_kwh"])
    df["annual_diff_kwh"] = annual_diff

    annual_diff_pct = pd.to_numeric(df.get("annual_diff_pct", pd.Series(index=df.index, dtype=float)), errors="coerce")
    mask_sim = df["annual_simstadt_kwh"].abs() > 0
    annual_diff_pct.loc[mask_sim] = 100.0 * annual_diff.loc[mask_sim] / df.loc[mask_sim, "annual_simstadt_kwh"]
    df["annual_diff_pct"] = annual_diff_pct

    df["heating_diff_abs_kwh"] = annual_diff.abs()
    df["heating_diff_abs_pct"] = annual_diff_pct.abs()
    df["heating_mismatch_score"] = df["heating_diff_abs_pct"]
    df["is_heating_mismatch_outlier"] = df["heating_diff_abs_kwh"].notna()

    return df


def write_heating_mismatch_outlier_report(outliers: pd.DataFrame, output_dir: Path) -> None:
    csv_path = output_dir / "heating_mismatch_outlier_buildings.csv"
    txt_path = output_dir / "heating_mismatch_outlier_buildings.txt"

    if "heated_area_m2" in outliers.columns:
        outliers = outliers[pd.to_numeric(outliers["heated_area_m2"], errors="coerce") >= 0].copy() # disabel if you want to include all buildings regardless of size in the report.

    export_columns = [
        "Building ID",
        "building_type",
        "usage_type",
        "heated_area_m2",
        "heated_volume_m3",
        "window_to_heated",
        "window_to_wall",
        "annual_rc_kwh",
        "annual_simstadt_kwh",
        "annual_diff_kwh",
        "annual_diff_pct",
        "heating_mismatch_score",
    ]

    available_columns = [column for column in export_columns if column in outliers.columns]
    outliers_export = outliers[available_columns].copy()
    if not outliers_export.empty and "heating_diff_abs_kwh" in outliers.columns:
        outliers_export = outliers_export.assign(
            heating_diff_abs_kwh=outliers["heating_diff_abs_kwh"],
            heating_diff_abs_pct=outliers["heating_diff_abs_pct"],
        )
        # Sort by absolute percentage difference (descending) to avoid +/- sign bias
        outliers_export = outliers_export.sort_values(
            ["heating_diff_abs_pct", "heating_diff_abs_kwh"], ascending=[False, False]
        )

    outliers_export.to_csv(csv_path, index=False)

    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write("Heating Mismatch Outliers - Table Format\n")
        handle.write("(Sorted by descending absolute RC-SimStadt percentage difference)\n\n")

        if outliers_export.empty:
            handle.write("No heating mismatch outliers were detected.\n")
            return

        # Write header with units
        header = (
            "Building ID              | Building Type | Usage Type        | "
            "Heated Area [m²] | Heated Volume [m³] | Window-to-Floor | Window-to-Wall | "
            "RC [kWh]  | SimStadt [kWh] | Difference [kWh] | Difference [%]\n"
        )
        separator = (
            "-------------------------|---------------|-------------------|"
            "------------------|--------------------|-----------------|----------------|"
            "-----------|-----------------|--------------------|--------\n"
        )
        handle.write(header)
        handle.write(separator)

        # Write data rows
        for _, row in outliers_export.iterrows():
            building_id = row.get("Building ID", "")
            building_type = row.get("building_type", "")
            usage_type = row.get("usage_type", "")
            heated_area = row.get("heated_area_m2", np.nan)
            heated_volume = row.get("heated_volume_m3", np.nan)
            window_ratio = row.get("window_to_heated", np.nan)
            window_to_wall = row.get("window_to_wall", np.nan)
            rc_kwh = row.get("annual_rc_kwh", np.nan)
            simstadt_kwh = row.get("annual_simstadt_kwh", np.nan)
            diff_kwh = row.get("annual_diff_kwh", np.nan)
            diff_pct = row.get("annual_diff_pct", np.nan)

            handle.write(
                f"{building_id:<24} | {building_type:<13} | {usage_type:<17} | "
                f"{heated_area:>16.1f} | {heated_volume:>18.1f} | {window_ratio:>15.3f} | {window_to_wall:>14.3f} | "
                f"{rc_kwh:>9.1f} | {simstadt_kwh:>14.1f} | {diff_kwh:>18.1f} | {diff_pct:>8.1f}%\n"
            )

    return


def build_usage_counts(df: pd.DataFrame) -> pd.DataFrame:
    usage_counts = (
        df.groupby("usage_type", dropna=False)
        .size()
        .reset_index(name="building_count")
        .sort_values("building_count", ascending=False)
    )
    total = int(usage_counts["building_count"].sum())
    usage_counts["total_buildings"] = total
    usage_counts["share_percent"] = 100.0 * usage_counts["building_count"] / total
    usage_counts["count_of_total"] = usage_counts["building_count"].astype(str) + " from " + str(total)
    return usage_counts


def select_representatives(
    df: pd.DataFrame,
    target_count: int,
    selection_usage: str,
) -> pd.DataFrame:
    if target_count <= 0:
        raise ValueError("target_count must be > 0")

    candidates = df[~df["is_outlier"]].copy()
    if selection_usage.strip().lower() != "all":
        candidates = candidates[
            candidates["usage_type"] == selection_usage.strip().lower()
        ].copy()

    candidates = candidates.dropna(
        subset=["compactness", "window_to_heated", "annual_heat_kwh"]
    ).copy()

    if candidates.empty:
        return candidates

    for feature in ["compactness", "window_to_heated", "annual_heat_kwh"]:
        candidates[f"z_{feature}"] = robust_z(candidates[feature].to_numpy(dtype=float))

    candidates["centrality_score"] = (
        0.4 * np.abs(candidates["z_compactness"])
        + 0.3 * np.abs(candidates["z_window_to_heated"])
        + 0.3 * np.abs(candidates["z_annual_heat_kwh"])
    )

    n_bins = min(target_count, len(candidates))
    candidates["demand_bin"] = pd.qcut(
        candidates["annual_heat_kwh"],
        q=n_bins,
        labels=False,
        duplicates="drop",
    )

    selected = (
        candidates.sort_values(["demand_bin", "centrality_score"]) 
        .groupby("demand_bin", as_index=False)
        .first()
    )

    if len(selected) < target_count:
        missing = target_count - len(selected)
        selected_ids = set(selected["Building ID"].tolist())
        fill = (
            candidates[~candidates["Building ID"].isin(selected_ids)]
            .sort_values("centrality_score")
            .head(missing)
        )
        selected = pd.concat([selected, fill], ignore_index=True)

    selected = selected.sort_values("annual_heat_kwh").head(target_count)
    return selected


def normalize_building_id(value: str) -> str:
    return str(value).strip().lower()


def strip_variant_suffix(value: str) -> str:
    return re.sub(r"__\d+$", "", str(value).strip())


def strip_thermal_class_suffix(value: str) -> str:
    return re.sub(r"_0[1-5]_(very_light|light|medium|heavy|very_heavy)$", "", str(value).strip())


def parse_simstadt_prn_series(file_path: Path) -> pd.Series:
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=["HOY", "Heat Demand", "Load duration curve", "Dhw Demand"],
        engine="python",
    )
    series = pd.to_numeric(df["Heat Demand"], errors="coerce").fillna(0.0)
    return series.astype(float)


def build_simstadt_lookup(sim_files: list[Path]) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    exact: dict[str, Path] = {}
    by_base: dict[str, list[Path]] = {}

    for path in sim_files:
        building_id = path.name.replace("_hourly_demand.prn", "")
        exact[normalize_building_id(building_id)] = path

        base = normalize_building_id(strip_variant_suffix(building_id))
        by_base.setdefault(base, []).append(path)

    return exact, by_base


def build_rc_lookup(rc_files: list[Path]) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    exact: dict[str, Path] = {}
    by_base: dict[str, list[Path]] = {}

    for path in rc_files:
        building_id = path.stem.replace("rc_hourly_", "", 1)
        building_id = strip_thermal_class_suffix(building_id)
        exact[normalize_building_id(building_id)] = path

        base = normalize_building_id(strip_variant_suffix(building_id))
        by_base.setdefault(base, []).append(path)

    return exact, by_base


def resolve_rc_file(
    building_id: str,
    exact_lookup: dict[str, Path],
    base_lookup: dict[str, list[Path]],
) -> Path | None:
    normalized = normalize_building_id(building_id)
    if normalized in exact_lookup:
        return exact_lookup[normalized]

    base = normalize_building_id(strip_variant_suffix(building_id))
    candidates = base_lookup.get(base, [])
    if not candidates:
        return None

    return sorted(candidates, key=lambda p: p.name.lower())[0]


def resolve_simstadt_file(
    building_id: str,
    exact_lookup: dict[str, Path],
    base_lookup: dict[str, list[Path]],
) -> Path | None:
    normalized = normalize_building_id(building_id)
    if normalized in exact_lookup:
        return exact_lookup[normalized]

    base = normalize_building_id(strip_variant_suffix(building_id))
    candidates = base_lookup.get(base, [])
    if not candidates:
        return None

    return sorted(candidates, key=lambda p: p.name.lower())[0]


def calendar_to_hour_of_year(
    calendar_year: int,
    month: int,
    day: int,
    clock_hour: int,
) -> int:
    start_of_year = pd.Timestamp(year=calendar_year, month=1, day=1, hour=0)
    selected = pd.Timestamp(year=calendar_year, month=month, day=day, hour=clock_hour)
    delta = selected - start_of_year
    return int(delta.total_seconds() // 3600)


def create_representative_hourly_plots(
    selected_buildings: pd.DataFrame,
    data_dir: Path,
    window_hours: int,
    calendar_year: int,
    winter_start_month: int = DEFAULT_WINTER_START_MONTH,
    winter_start_day: int = DEFAULT_WINTER_START_DAY,
    spring_start_month: int = DEFAULT_SPRING_START_MONTH,
    spring_start_day: int = DEFAULT_SPRING_START_DAY,
    summer_start_month: int = DEFAULT_SUMMER_START_MONTH,
    summer_start_day: int = DEFAULT_SUMMER_START_DAY,
    window_start_clock_hour: int = DEFAULT_WINDOW_START_CLOCK_HOUR,
) -> pd.DataFrame:
    output_dir = data_dir / "comparisons_all" / "representative_14day_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    rc_dir = data_dir / RC_HOURLY_RELATIVE_DIR
    sim_dir = data_dir / SIMSTADT_HOURLY_DIR.name

    rc_files = sorted(rc_dir.glob("rc_hourly_*.csv"))
    rc_exact, rc_by_base = build_rc_lookup(rc_files)
    sim_files = sorted(sim_dir.glob("*_hourly_demand.prn"))
    sim_exact, sim_by_base = build_simstadt_lookup(sim_files)

    # Define the three seasonal windows
    seasons = [
        {
            "name": "Winter",
            "month": int(winter_start_month),
            "day": int(winter_start_day),
        },
        {
            "name": "Spring",
            "month": int(spring_start_month),
            "day": int(spring_start_day),
        },
        {
            "name": "Summer",
            "month": int(summer_start_month),
            "day": int(summer_start_day),
        },
    ]

    report_rows: list[dict[str, object]] = []

    for _, row in selected_buildings.iterrows():
        building_id = str(row["Building ID"])
        rc_path = resolve_rc_file(building_id, rc_exact, rc_by_base)

        if rc_path is None or not rc_path.exists():
            report_rows.append(
                {
                    "Building ID": building_id,
                    "status": "missing_rc_file",
                    "rc_file": str(rc_path) if rc_path is not None else "",
                }
            )
            continue

        sim_path = resolve_simstadt_file(building_id, sim_exact, sim_by_base)
        if sim_path is None:
            report_rows.append(
                {
                    "Building ID": building_id,
                    "status": "missing_simstadt_file",
                    "rc_file": str(rc_path),
                }
            )
            continue

        rc_df = pd.read_csv(rc_path)

        rc_heat = get_numeric_series(rc_df, "HeatingDemand_kWh_h").fillna(0.0)
        rc_solar = get_numeric_series(rc_df, "SolarGains").fillna(0.0) / 1000.0
        rc_cooling = get_numeric_series(rc_df, "CoolingDemand_kWh_h").fillna(0.0).abs()
        rc_indoor = get_numeric_series(rc_df, "IndoorAir")
        rc_outside = get_numeric_series(rc_df, "OutsideTemp")

        sim_heat = parse_simstadt_prn_series(sim_path)

        min_len = min(
            len(rc_heat),
            len(rc_solar),
            len(rc_cooling),
            len(sim_heat),
            len(rc_indoor),
            len(rc_outside),
        )
        if min_len <= 0:
            report_rows.append(
                {
                    "Building ID": building_id,
                    "status": "empty_timeseries",
                    "rc_file": str(rc_path),
                    "sim_file": str(sim_path),
                }
            )
            continue

        building_type = str(row.get("building_type", row.get("usage_type", "n/a")))
        usage = str(row.get("usage_type", "n/a"))
        heated_area = pd.to_numeric(pd.Series([row.get("heated_area_m2")]), errors="coerce").iloc[0]
        heated_volume = pd.to_numeric(pd.Series([row.get("heated_volume_m3")]), errors="coerce").iloc[0]
        window_to_wall_ratio = pd.to_numeric(pd.Series([row.get("window_to_wall")]), errors="coerce").iloc[0]
        mean_u_value_opaque = pd.to_numeric(pd.Series([row.get("u_walls_w_m2k")]), errors="coerce").iloc[0]
        mean_u_value_window = pd.to_numeric(pd.Series([row.get("u_windows_w_m2k")]), errors="coerce").iloc[0]

        heated_area_text = f"{heated_area:.1f} m2" if np.isfinite(heated_area) else "n/a"
        heated_volume_text = f"{heated_volume:.1f} m³" if np.isfinite(heated_volume) else "n/a"
        window_to_wall_text = f"{window_to_wall_ratio:.3f}" if np.isfinite(window_to_wall_ratio) else "n/a"
        mean_u_value_opaque_text = f"{mean_u_value_opaque:.3f}" if np.isfinite(mean_u_value_opaque) else "n/a"
        mean_u_value_window_text = f"{mean_u_value_window:.3f}" if np.isfinite(mean_u_value_window) else "n/a"

        details_text = (
            f"Building type: {building_type} | Usage: {usage} | "
            f"Heated area: {heated_area_text} | "
            f"Heated volume: {heated_volume_text} | "
            f"Window-to-wall ratio: {window_to_wall_text} | "
            f"Mean U-value (opaque): {mean_u_value_opaque_text} | "
            f"Mean U-value (window): {mean_u_value_window_text}"
        )

        # Generate plots for each season
        for season in seasons:
            season_name = season["name"]
            chosen_window_start = calendar_to_hour_of_year(
                calendar_year=calendar_year,
                month=season["month"],
                day=season["day"],
                clock_hour=int(window_start_clock_hour),
            )

            start = max(0, int(chosen_window_start))
            end = min(min_len, start + int(window_hours))
            if end <= start:
                report_rows.append(
                    {
                        "Building ID": building_id,
                        "season": season_name,
                        "status": "invalid_window",
                        "rc_file": str(rc_path),
                        "sim_file": str(sim_path),
                    }
                )
                continue

            x = np.arange(end - start)
            rc_heat_window = rc_heat.iloc[start:end].to_numpy()
            sim_heat_window = sim_heat.iloc[start:end].to_numpy()
            rc_solar_window = rc_solar.iloc[start:end].to_numpy()
            rc_cooling_window = rc_cooling.iloc[start:end].to_numpy()

            rc_indoor_window = rc_indoor.iloc[start:end].to_numpy()
            rc_outside_window = rc_outside.iloc[start:end].to_numpy()

            fig, ax_kwh = plt.subplots(figsize=(14, 6.0))

            ax_kwh.plot(x, rc_heat_window, color=RC_PLOT_COLOR, linewidth=1.6, label="RC heating demand in kWh/h")
            ax_kwh.plot(x, sim_heat_window, color=SIMSTADT_PLOT_COLOR, linewidth=1.6, label="SimStadt heating demand in kWh/h")
            ax_kwh.plot(x, rc_cooling_window, color=RC_COOLING_PLOT_COLOR, linewidth=1.0, linestyle=":", label="RC cooling demand in kWh/h")
            ax_kwh.plot(x, rc_solar_window, color=RC_SOLAR_PLOT_COLOR, linewidth=1.0, label="RC solar gains in kWh/h")

            ax_kwh.set_ylabel("Heating energy demand in kWh/h")
            ax_kwh.set_xlabel("Days")
            ax_kwh.grid(True, alpha=0.25)

            tick_positions = np.arange(0, end - start, 24)
            date_origin = pd.Timestamp(year=calendar_year, month=1, day=1)
            date_labels = [
                (date_origin + pd.Timedelta(hours=int(start + pos))).strftime("%d-%b")
                for pos in tick_positions
            ]
            ax_kwh.set_xticks(tick_positions)
            ax_kwh.set_xticklabels(date_labels, rotation=45, ha="right")

            ax_temp = ax_kwh.twinx()
            ax_temp.plot(x, rc_indoor_window, color=RC_INDOOR_PLOT_COLOR, linewidth=1.2, linestyle="--", label="RC indoor air temperature in °C")
            ax_temp.plot(x, rc_outside_window, color=RC_OUTSIDE_PLOT_COLOR, linewidth=1.2, linestyle=":", label="Outdoor air temperature in °C")
            
            ax_temp.set_ylabel("Temperature in °C")

            ax_kwh.set_title(
                f"{building_id} | {season_name} | 14-day hourly comparison"
            )

            lines_left, labels_left = ax_kwh.get_legend_handles_labels()
            lines_right, labels_right = ax_temp.get_legend_handles_labels()
            legend_handles = lines_left + lines_right
            legend_labels = labels_left + labels_right
            fig.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.055),
                ncol=3,
                fontsize=8,
                frameon=False,
            )

            fig.text(
                0.5,
                0.018,
                details_text,
                ha="center",
                va="top",
                fontsize=8,
            )

            plot_path = output_dir / f"{building_id}_{season_name.lower()}_14day_hourly_comparison.png"
            fig.tight_layout(rect=(0.0, 0.20, 1.0, 1.0))
            fig.savefig(str(plot_path), dpi=150)
            plt.close(fig)

            report_rows.append(
                {
                    "Building ID": building_id,
                    "season": season_name,
                    "status": "ok",
                    "rc_file": str(rc_path),
                    "sim_file": str(sim_path),
                    "plot_file": str(plot_path),
                    "window_start_hour": start,
                    "window_end_hour_exclusive": end,
                    "plotted_hours": end - start,
                    "window_mode": "manual-date",
                }
            )

    report_df = pd.DataFrame(report_rows)
    report_path = output_dir / "representative_14day_plot_report.csv"
    report_df.to_csv(report_path, index=False)
    return report_df


def plot_heated_area_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Create a histogram of heated area distribution in 50 m² bins with count labels."""
    heated_areas = pd.to_numeric(df["heated_area_m2"], errors="coerce").dropna()
    
    if len(heated_areas) == 0:
        return
    
    min_area = heated_areas.min()
    max_area = heated_areas.max()
    bin_width = 50
    bins = np.arange(0, max_area + bin_width, bin_width)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    counts, edges, patches = ax.hist(heated_areas, bins=bins, color="#3A7CA5", edgecolor="black", linewidth=0.5)
    
    # Add count labels on top of each bin
    for i, (count, patch) in enumerate(zip(counts, patches)):
        if count > 0:
            height = patch.get_height()
            ax.text(patch.get_x() + patch.get_width() / 2.0, height, f"{int(count)}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    ax.set_xlabel("Heated Area [m²]", fontsize=11)
    ax.set_ylabel("Number of Buildings", fontsize=11)
    ax.set_title("Distribution of Heated Areas (50 m² bins)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    output_path = output_dir / "heated_area_distribution.png"
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = data_dir / "comparisons_all"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_feature_table(data_dir)
    df = mark_outliers(df)

    usage_counts = build_usage_counts(df)
    usage_counts_path = output_dir / "buildingtype_counts.csv"
    usage_counts.to_csv(usage_counts_path, index=False)

    outliers = df[df["is_outlier"]].copy()
    outliers_export = outliers[
        [
            "Building ID",
            "usage_type",
            "annual_heat_kwh",
            "compactness",
            "window_to_heated",
        ]
    ].sort_values(["usage_type", "annual_heat_kwh"])
    outliers_export.to_csv(output_dir / "outlier_buildings.csv", index=False)

    # Load annual RC vs SimStadt heating comparison
    mismatch_df = load_building_comparison_heating(data_dir)
    if not mismatch_df.empty and len(mismatch_df) > 0:
        # Merge with feature table to add building geometry and usage info
        mismatch_df = mismatch_df.merge(
            df[[
                "Building ID",
                "building_type",
                "usage_type",
                "heated_area_m2",
                "heated_volume_m3",
                "window_to_heated",
                "window_to_wall",
            ]],
            on="Building ID",
            how="left",
        )
        mismatch_df = mismatch_df[pd.to_numeric(mismatch_df["heated_area_m2"], errors="coerce") >= 0].copy() # disabel if you want to include all buildings regardless of size in the report.
        mismatch_df = mark_heating_mismatch_outliers(mismatch_df)
        # Pass the full building stack (sorted in the report) instead of filtering
        mismatch_outliers = mismatch_df.copy()
    else:
        mismatch_outliers = pd.DataFrame()

    write_heating_mismatch_outlier_report(mismatch_outliers, output_dir)

    selected = select_representatives(
        df=df,
        target_count=args.target_count,
        selection_usage=args.selection_usage,
    )

    selected_export = selected[
        [
            "Building ID",
            "usage_type",
            "annual_heat_kwh",
            "heated_area_m2",
            "heated_volume_m3",
            "window_area_m2",
            "window_to_heated",
            "window_to_wall",
            "u_walls_w_m2k",
            "u_windows_w_m2k",
            "compactness",
            "centrality_score",
        ]
    ]
    selected_export.to_csv(output_dir / "representative_buildings.csv", index=False)

    plot_report = create_representative_hourly_plots(
        selected_buildings=selected_export,
        data_dir=data_dir,
        window_hours=args.window_hours,
        calendar_year=args.calendar_year,
        winter_start_month=args.winter_start_month,
        winter_start_day=args.winter_start_day,
        spring_start_month=args.spring_start_month,
        spring_start_day=args.spring_start_day,
        summer_start_month=args.summer_start_month,
        summer_start_day=args.summer_start_day,
        window_start_clock_hour=args.window_start_clock_hour,
    )

    df.to_csv(output_dir / "selection_feature_table_with_outliers.csv", index=False)

    # Create heated area distribution plot
    plot_heated_area_distribution(df, output_dir)

    total = len(df)
    print(f"Total buildings analyzed: {total}")
    print(f"Outliers flagged: {len(outliers_export)}")
    print(f"Heating mismatch outliers flagged: {len(mismatch_outliers)}")
    print(f"Representatives selected: {len(selected_export)}")
    print(f"Representative plots generated: {(plot_report['status'] == 'ok').sum()}")
    print(f"Note: 3 seasonal plots per building (Winter, Spring, Summer)")
    if not plot_report.empty and "season" in plot_report.columns:
        seasons = plot_report["season"].unique().tolist()
        print(f"Seasons plotted: {', '.join(sorted(seasons))}")
    print(f"Building-type counts written to: {usage_counts_path}")
    print(f"Heated area distribution plot written to: {output_dir / 'heated_area_distribution.png'}")
    print(f"All outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
