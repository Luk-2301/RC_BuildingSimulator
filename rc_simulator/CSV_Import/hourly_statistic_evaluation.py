from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
MATCHED_FILE = DATA_DIR / "comparisons_all" / "matched_buildings.csv"
SIMSTADT_HEATING_DIR = DATA_DIR / "SimStadt_Hourly_Heat_Demand"
RC_AGGREGATED_DIR = DATA_DIR / "Rc_Hourly_Aggregated_Results"
OUTPUT_DIR = DATA_DIR / "comparisons_all"
REPORT_DIR = SCRIPT_DIR / "statistic"
ENABLE_COOLING = False
# Day-of-year boundaries for summer (inclusive). Non-summer = DOY < SUMMER_START_DOY or DOY > SUMMER_END_DOY
SUMMER_START_DOY = 121
SUMMER_END_DOY = 273

THERMAL_CLASSES = [
	(1, "very light", "01_very_light"),
	(2, "light", "02_light"),
	(3, "medium", "03_medium"),
	(4, "heavy", "04_heavy"),
	(5, "very heavy", "05_very_heavy"),
]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Compare hourly summed heating and cooling demand for matched SimStadt and RC buildings."
	)
	parser.add_argument(
		"--matched-file",
		type=Path,
		default=MATCHED_FILE,
		help="Path to matched_buildings.csv.",
	)
	parser.add_argument(
		"--simstadt-heating-dir",
		type=Path,
		default=SIMSTADT_HEATING_DIR,
		help="Directory containing SimStadt per-building hourly demand .prn files.",
	)
	parser.add_argument(
		"--rc-aggregated-dir",
		type=Path,
		default=RC_AGGREGATED_DIR,
		help="Directory containing rc_hourly_accumulated_*.csv files.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=OUTPUT_DIR,
		help="Directory for hourly comparison CSV outputs.",
	)
	parser.add_argument(
		"--report-dir",
		type=Path,
		default=REPORT_DIR,
		help="Directory for the text report.",
	)
	return parser.parse_args()


def normalize_id(value: object) -> str:
	return str(value).strip().lower()


def strip_variant_suffix(value: str) -> str:
	parts = value.split("__")
	if len(parts) >= 2 and parts[-1].isdigit():
		return "__".join(parts[:-1])
	return value


def normalize_fallback_id(value: object) -> str:
	return strip_variant_suffix(normalize_id(value))


def read_numeric_prn_series(file_path: Path, column_name: str = "Heat Demand") -> pd.Series:
	df = pd.read_csv(
		file_path,
		sep=r"\s+",
		comment="#",
		header=None,
		names=["HOY", "Heat Demand", "Load duration curve", "Dhw Demand"],
		engine="python",
	)
	if column_name not in df.columns:
		raise ValueError(f"Missing '{column_name}' in {file_path.name}")
	series = pd.to_numeric(df[column_name], errors="coerce").fillna(0.0)
	return series.astype(float)


def load_matched_pairs(path: Path) -> pd.DataFrame:
	df = pd.read_csv(path)
	required = {"rc_building_id", "simstadt_building_id"}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(f"Missing columns in matched file: {sorted(missing)}")
	result = df.loc[:, ["rc_building_id", "simstadt_building_id"]].copy()
	result["rc_building_id"] = result["rc_building_id"].map(normalize_id)
	result["simstadt_building_id"] = result["simstadt_building_id"].map(normalize_id)
	result["simstadt_fallback_id"] = result["simstadt_building_id"].map(normalize_fallback_id)
	result = result[(result["rc_building_id"] != "") & (result["simstadt_building_id"] != "")]
	return result.drop_duplicates(subset=["rc_building_id", "simstadt_building_id"])


def load_simstadt_heating_map(heating_dir: Path) -> dict[str, pd.Series]:
	series_map: dict[str, pd.Series] = {}
	for file_path in sorted(heating_dir.glob("*_hourly_demand.prn")):
		building_id = normalize_id(file_path.name[: -len("_hourly_demand.prn")])
		if building_id not in series_map:
			series_map[building_id] = read_numeric_prn_series(file_path, "Heat Demand")
	return series_map


def aggregate_simstadt_hourly_heating_only(
	matched_pairs: pd.DataFrame,
	heating_map: dict[str, pd.Series],
) -> tuple[pd.Series, dict[str, int]]:
	heating_series_list: list[np.ndarray] = []
	stats = {"matched_buildings": 0, "missing_heating": 0}

	heating_lookup = dict(heating_map)
	for _, row in matched_pairs.iterrows():
		sim_id = row["simstadt_building_id"]
		fallback_id = row["simstadt_fallback_id"]

		heating_series = heating_lookup.get(sim_id)
		if heating_series is None:
			heating_series = heating_lookup.get(fallback_id)
		if heating_series is None:
			stats["missing_heating"] += 1
			continue

		heating_values = heating_series.to_numpy(dtype=float)
		if heating_values.size < 8760:
			heating_values = np.pad(heating_values, (0, 8760 - heating_values.size), constant_values=0.0)
		elif heating_values.size > 8760:
			heating_values = heating_values[:8760]
		heating_series_list.append(heating_values)
		stats["matched_buildings"] += 1

	if heating_series_list:
		heating_agg = pd.Series(np.sum(heating_series_list, axis=0), dtype=float)
	else:
		heating_agg = pd.Series(np.zeros(8760, dtype=float))

	return heating_agg, stats


def load_rc_aggregated_series(path: Path) -> tuple[pd.Series, pd.Series]:
	df = pd.read_csv(path)
	required = {"HeatingDemand_kWh_h_sum", "CoolingDemand_kWh_h_sum"}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(f"Missing columns in RC aggregated file {path.name}: {sorted(missing)}")
	heating = pd.to_numeric(df["HeatingDemand_kWh_h_sum"], errors="coerce").fillna(0.0).astype(float)
	cooling = pd.to_numeric(df["CoolingDemand_kWh_h_sum"], errors="coerce").fillna(0.0).astype(float)
	return heating.reset_index(drop=True), cooling.reset_index(drop=True)


def align_series(rc_series: pd.Series, sim_series: pd.Series) -> tuple[pd.Series, pd.Series]:
	min_len = min(len(rc_series), len(sim_series))
	return rc_series.iloc[:min_len].reset_index(drop=True), sim_series.iloc[:min_len].reset_index(drop=True)


def compute_metrics(rc_series: pd.Series, sim_series: pd.Series) -> dict[str, float]:
	rc_aligned, sim_aligned = align_series(rc_series, sim_series)
	if len(rc_aligned) == 0:
		return {"hours_compared": 0, "mae": np.nan, "rmse": np.nan, "corr": np.nan}
	diff = rc_aligned - sim_aligned
	corr = float(np.corrcoef(rc_aligned, sim_aligned)[0, 1]) if len(rc_aligned) > 1 else np.nan
	return {
		"hours_compared": int(len(rc_aligned)),
		"mae": float(np.mean(np.abs(diff))),
		"rmse": float(np.sqrt(np.mean(np.square(diff)))),
		"corr": corr,
	}


def compute_metrics_mask(rc_series: pd.Series, sim_series: pd.Series, mask: np.ndarray) -> dict[str, float]:
	"""Compute metrics for elements where mask is True. Mask is applied after aligning series."""
	rc_aligned, sim_aligned = align_series(rc_series, sim_series)
	if len(rc_aligned) == 0:
		return {"hours_compared": 0, "mae": np.nan, "rmse": np.nan, "corr": np.nan}
	# Ensure mask length matches aligned length
	mask_arr = np.asarray(mask, dtype=bool)
	if mask_arr.size != len(rc_aligned):
		# Truncate or pad mask to match
		if mask_arr.size > len(rc_aligned):
			mask_arr = mask_arr[: len(rc_aligned)]
		else:
			pad = np.zeros(len(rc_aligned) - mask_arr.size, dtype=bool)
			mask_arr = np.concatenate([mask_arr, pad])

	rc_sel = rc_aligned.iloc[np.nonzero(mask_arr)[0]]
	sim_sel = sim_aligned.iloc[np.nonzero(mask_arr)[0]]
	if len(rc_sel) == 0:
		return {"hours_compared": 0, "mae": np.nan, "rmse": np.nan, "corr": np.nan}
	diff = rc_sel - sim_sel
	corr = float(np.corrcoef(rc_sel, sim_sel)[0, 1]) if len(rc_sel) > 1 else np.nan
	return {
		"hours_compared": int(len(rc_sel)),
		"mae": float(np.mean(np.abs(diff))),
		"rmse": float(np.sqrt(np.mean(np.square(diff)))),
		"corr": corr,
	}


def write_simstadt_hourly_output(output_dir: Path, heating: pd.Series) -> Path:
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "simstadt_hourly_accumulated_matched.csv"
	df = pd.DataFrame(
		{
			"hour": np.arange(1, len(heating) + 1),
			"simstadt_heating_demand_kwh_h": heating.astype(float).values,
		}
	)
	df.to_csv(output_path, index=False)
	return output_path


def write_comparison_csv(output_dir: Path, class_suffix: str, rc_heating: pd.Series, sim_heating: pd.Series) -> Path:
	output_dir.mkdir(parents=True, exist_ok=True)
	comparison_path = output_dir / f"hourly_accumulated_comparison_{class_suffix}.csv"
	rc_heating_aligned, sim_heating_aligned = align_series(rc_heating, sim_heating)
	length = len(rc_heating_aligned)
	df = pd.DataFrame(
		{
			"hour": np.arange(1, length + 1),
			"rc_heating_demand_kwh_h": rc_heating_aligned.iloc[:length].values,
			"simstadt_heating_demand_kwh_h": sim_heating_aligned.iloc[:length].values,
			"diff_heating_kwh_h": (rc_heating_aligned.iloc[:length] - sim_heating_aligned.iloc[:length]).values,
		}
	)
	df.to_csv(comparison_path, index=False)
	return comparison_path


def format_value(value: Any) -> str:
	try:
		if value is None:
			return "n/a"
		return f"{float(value):.2f}"
	except (TypeError, ValueError):
		return "n/a"


def write_report(report_dir: Path, rows: list[dict[str, object]], matched_buildings: int, simstadt_path: Path, notes: list[str]) -> Path:
	report_dir.mkdir(parents=True, exist_ok=True)
	report_path = report_dir / "hourly_statistic_evaluation_results.txt"
	label_width = max(len("Comparison"), max(len(str(row["comparison"])) for row in rows)) if rows else len("Comparison")
	lines = [
		"Hourly Statistical Evaluation Report",
		"",
		f"Matched buildings used: {matched_buildings}",
		f"SimStadt matched hourly output: {simstadt_path.name}",
		"",
		f"{'Comparison'.ljust(label_width)}  {'Heat MAE'.rjust(10)}  {'Heat RMSE'.rjust(10)}  {'Heat Corr'.rjust(10)}",
		f"{'-' * label_width}  {'-' * 10}  {'-' * 10}  {'-' * 10}",
	]
	for row in rows:
		lines.append(
			f"{str(row['comparison']).ljust(label_width)}  {format_value(row['heat_mae']).rjust(10)}  {format_value(row['heat_rmse']).rjust(10)}  {format_value(row['heat_corr']).rjust(10)}"
		)
	lines.extend(["", "Notes:"])
	lines.extend(notes)
	report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
	return report_path


def main() -> int:
	args = parse_args()

	matched_pairs = load_matched_pairs(args.matched_file)
	heating_map = load_simstadt_heating_map(args.simstadt_heating_dir)
	simstadt_heating, stats = aggregate_simstadt_hourly_heating_only(matched_pairs, heating_map)
	simstadt_path = write_simstadt_hourly_output(args.output_dir, simstadt_heating)

	rows: list[dict[str, object]] = []
	notes = [
		"MAE, RMSE, and correlation are computed over the 8760 aligned hourly points for each variant.",
		"Additional metrics labeled '(just_heating_period)' are computed only for hours with day-of-year outside 121..273 (inclusive).",
		"SimStadt heating is aggregated from the matched per-building hourly PRN files.",
		"Cooling logic has been removed for now and can be reintroduced when hourly SimStadt cooling data is available.",
	]
	for _, label, class_suffix in THERMAL_CLASSES:
		rc_file = args.rc_aggregated_dir / f"rc_hourly_accumulated_{class_suffix}.csv"
		if not rc_file.exists():
			raise FileNotFoundError(f"Missing RC aggregated file: {rc_file}")
		rc_heating, _ = load_rc_aggregated_series(rc_file)
		# Align series and compute DOY mask for just_heating_period hours
		rc_aligned, sim_aligned = align_series(rc_heating, simstadt_heating)
		length = len(rc_aligned)
		hours = np.arange(1, length + 1)
		doy = ((hours - 1) // 24) + 1
		just_heating_period_mask = (doy < SUMMER_START_DOY) | (doy > SUMMER_END_DOY)

		# Compute metrics for all hours and for just_heating_period hours
		heating_metrics_all = compute_metrics(rc_aligned, sim_aligned)
		heating_metrics_just_heating_period = compute_metrics_mask(rc_aligned, sim_aligned, just_heating_period_mask)

		# Write the unfiltered comparison CSV (full year)
		write_comparison_csv(args.output_dir, class_suffix, rc_heating, simstadt_heating)

		# Append both full-year and just_heating_period rows to the report
		rows.append(
			{
				"comparison": f"SimStadt - {label}",
				"heat_mae": heating_metrics_all["mae"],
				"heat_rmse": heating_metrics_all["rmse"],
				"heat_corr": heating_metrics_all["corr"],
			}
		)
		rows.append(
			{
				"comparison": f"SimStadt - {label} (just_heating_period)",
				"heat_mae": heating_metrics_just_heating_period["mae"],
				"heat_rmse": heating_metrics_just_heating_period["rmse"],
				"heat_corr": heating_metrics_just_heating_period["corr"],
			}
		)

	report_path = write_report(
		args.report_dir,
		rows,
		stats["matched_buildings"],
		simstadt_path,
		notes,
	)

	print(f"Wrote SimStadt hourly aggregate to {simstadt_path}")
	print(f"Wrote {len(rows)} hourly comparison summaries to {args.output_dir}")
	print(f"Wrote report to {report_path}")
	print(f"Matched buildings used: {stats['matched_buildings']}")
	print(f"Missing heating series: {stats['missing_heating']}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
