from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_MATCHED_FILE = DATA_DIR / "comparisons_all" / "matched_buildings.csv"
DEFAULT_SIMSTADT_FILE = DATA_DIR / "ImBuchwald_DIN18599_HEATING_AND_COOLING.csv"
DEFAULT_RC_RESULTS_DIR = DATA_DIR / "Rc_Hourly_Aggregated_Results"
DEFAULT_OUTPUT_FILE = SCRIPT_DIR / "statistic" / "statistic_evaluation_results.txt"

THERMAL_CLASSES = [
    (1, "very light", "01_very_light"),
    (2, "light", "02_light"),
    (3, "medium", "03_medium"),
    (4, "heavy", "04_heavy"),
    (5, "very heavy", "05_very_heavy"),
]


@dataclass(frozen=True)
class DemandPair:
	heating: float
	cooling: float


@dataclass(frozen=True)
class ComparisonResult:
	comparison: str
	mape: float
	r2: float
	matched_count: int


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Compare SimStadt annual heating and cooling demand with RC thermal classes."
	)
	parser.add_argument(
		"--matched-file",
		type=Path,
		default=DEFAULT_MATCHED_FILE,
		help="Path to matched_buildings.csv.",
	)
	parser.add_argument(
		"--simstadt-file",
		type=Path,
		default=DEFAULT_SIMSTADT_FILE,
		help="Path to ImBuchwald_DIN18599_HEATING_AND_COOLING.csv.",
	)
	parser.add_argument(
		"--rc-results-dir",
		type=Path,
		default=DEFAULT_RC_RESULTS_DIR,
		help="Directory containing simulated_annual_results_0*.csv files.",
	)
	parser.add_argument(
		"--output-file",
		type=Path,
		default=DEFAULT_OUTPUT_FILE,
		help="Path to the generated text report.",
	)
	return parser.parse_args()


def to_numeric_series(series: pd.Series) -> pd.Series:
	return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def normalize_id_series(series: pd.Series) -> pd.Series:
	return series.astype(str).str.strip().str.lower()


def normalize_fallback_id_series(series: pd.Series) -> pd.Series:
	return normalize_id_series(series).str.replace(r"__\d+$", "", regex=True)


def normalize_fallback_id_value(value: object) -> str:
	return re.sub(r"__\d+$", "", normalize_id_series(pd.Series([value])).iloc[0])


def build_row_lookup(df: pd.DataFrame, id_columns: list[str]) -> dict[str, dict[str, object]]:
	lookup: dict[str, dict[str, object]] = {}
	for _, row in df.iterrows():
		row_data: dict[str, object] = {str(key): value for key, value in row.to_dict().items()}
		for column in id_columns:
			value = str(row_data.get(column, "")).strip()
			if value and value not in lookup:
				lookup[value] = row_data
	return lookup


def load_matched_pairs(path: Path) -> pd.DataFrame:
	df = pd.read_csv(path)
	required = {"rc_building_id", "simstadt_building_id"}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(f"Missing columns in matched file: {sorted(missing)}")
	df = df.loc[:, ["rc_building_id", "simstadt_building_id"]].copy()
	df["rc_building_id"] = normalize_id_series(df["rc_building_id"])
	df["simstadt_building_id"] = normalize_id_series(df["simstadt_building_id"])
	df["simstadt_fallback_id"] = normalize_fallback_id_series(df["simstadt_building_id"])
	df = df[(df["rc_building_id"] != "") & (df["simstadt_building_id"] != "")]
	df = df.drop_duplicates(subset=["rc_building_id", "simstadt_building_id"])
	return df


def load_simstadt_demands(path: Path) -> pd.DataFrame:
	df = pd.read_csv(path, sep=";", decimal=",", comment="#", skip_blank_lines=True)
	required = {"GMLId", "Yearly Heating demand", "Yearly Cooling demand"}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(f"Missing columns in SimStadt file: {sorted(missing)}")
	df = df.loc[:, ["GMLId", "Yearly Heating demand", "Yearly Cooling demand"]].copy()
	df["GMLId"] = normalize_id_series(df["GMLId"])
	df["simstadt_fallback_id"] = normalize_fallback_id_series(df["GMLId"])
	df["Yearly Heating demand"] = to_numeric_series(df["Yearly Heating demand"])
	df["Yearly Cooling demand"] = to_numeric_series(df["Yearly Cooling demand"])
	df = df.dropna(subset=["GMLId", "Yearly Heating demand", "Yearly Cooling demand"])
	df = df[df["GMLId"] != ""]
	df = df.drop_duplicates(subset=["GMLId"], keep="first")
	return df


def load_rc_demands(path: Path, class_label: str) -> pd.DataFrame:
	df = pd.read_csv(path)
	required = {
		"Building ID",
		"Building ID [filename]",
		"Annual Heating Demand Sim [kWh]",
		"Annual Cooling Demand Sim [kWh]",
	}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(f"Missing columns in RC file {path.name}: {sorted(missing)}")
	df = df.loc[:, ["Building ID", "Building ID [filename]", "Annual Heating Demand Sim [kWh]", "Annual Cooling Demand Sim [kWh]"]].copy()
	df["Building ID"] = normalize_id_series(df["Building ID"])
	df["Building ID [filename]"] = normalize_id_series(df["Building ID [filename]"])
	df["Annual Heating Demand Sim [kWh]"] = to_numeric_series(df["Annual Heating Demand Sim [kWh]"])
	df["Annual Cooling Demand Sim [kWh]"] = to_numeric_series(df["Annual Cooling Demand Sim [kWh]"])
	df = df.dropna(subset=["Building ID", "Annual Heating Demand Sim [kWh]", "Annual Cooling Demand Sim [kWh]"])
	df = df[df["Building ID"] != ""]
	df = df.rename(
		columns={
			"Building ID": "rc_building_id",
			"Building ID [filename]": "rc_building_filename_id",
			"Annual Heating Demand Sim [kWh]": f"rc_heating_{class_label}",
			"Annual Cooling Demand Sim [kWh]": f"rc_cooling_{class_label}",
		}
	)
	df["rc_fallback_id"] = normalize_fallback_id_series(df["rc_building_id"])
	df["rc_filename_fallback_id"] = normalize_fallback_id_series(df["rc_building_filename_id"])
	return df.loc[
		:,
		[
			"rc_building_id",
			"rc_building_filename_id",
			"rc_fallback_id",
			"rc_filename_fallback_id",
			f"rc_heating_{class_label}",
			f"rc_cooling_{class_label}",
		],
	]


def build_aligned_frame(
	matched_pairs: pd.DataFrame,
	simstadt: pd.DataFrame,
	rc_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
	simstadt_lookup = build_row_lookup(simstadt, ["GMLId", "simstadt_fallback_id"])
	rc_lookups = {
		class_label: build_row_lookup(
			rc_frame,
			["rc_building_id", "rc_building_filename_id", "rc_fallback_id", "rc_filename_fallback_id"],
		)
		for class_label, rc_frame in rc_frames.items()
	}

	rows: list[dict[str, object]] = []
	for _, pair in matched_pairs.iterrows():
		rc_id = str(pair["rc_building_id"]).strip()
		sim_id = str(pair["simstadt_building_id"]).strip()
		sim_fallback_id = str(pair["simstadt_fallback_id"]).strip()

		sim_row = simstadt_lookup.get(sim_id)
		sim_match_source = "exact"
		if sim_row is None:
			sim_row = simstadt_lookup.get(sim_fallback_id)
			sim_match_source = "fallback"
		if sim_row is None:
			continue

		row_data: dict[str, object] = {
			"rc_building_id": rc_id,
			"simstadt_building_id": sim_id,
			"simstadt_fallback_id": sim_fallback_id,
			"simstadt_match_source": sim_match_source,
		}
		row_data.update(sim_row)

		missing_rc_values = False
		for class_label, rc_lookup in rc_lookups.items():
			rc_row = rc_lookup.get(rc_id)
			rc_match_source = "exact"
			if rc_row is None:
				rc_row = rc_lookup.get(normalize_fallback_id_value(rc_id))
				rc_match_source = "fallback"
			if rc_row is None:
				missing_rc_values = True
				break
			row_data[f"rc_match_source_{class_label}"] = rc_match_source
			row_data[f"rc_heating_{class_label}"] = rc_row[f"rc_heating_{class_label}"]
			row_data[f"rc_cooling_{class_label}"] = rc_row[f"rc_cooling_{class_label}"]

		if not missing_rc_values:
			rows.append(row_data)

	return pd.DataFrame(rows)


def calculate_mape(actual: Iterable[float], predicted: Iterable[float]) -> float:
	actual_arr = np.asarray(list(actual), dtype=float)
	pred_arr = np.asarray(list(predicted), dtype=float)
	mask = np.isfinite(actual_arr) & np.isfinite(pred_arr) & (actual_arr != 0.0)
	if not np.any(mask):
		return float("nan")
	return float(np.mean(np.abs((actual_arr[mask] - pred_arr[mask]) / actual_arr[mask])) * 100.0)


def calculate_r2(actual: Iterable[float], predicted: Iterable[float]) -> float:
	actual_arr = np.asarray(list(actual), dtype=float)
	pred_arr = np.asarray(list(predicted), dtype=float)
	mask = np.isfinite(actual_arr) & np.isfinite(pred_arr)
	actual_arr = actual_arr[mask]
	pred_arr = pred_arr[mask]
	if actual_arr.size < 2:
		return float("nan")
	ss_tot = float(np.sum((actual_arr - np.mean(actual_arr)) ** 2))
	ss_res = float(np.sum((actual_arr - pred_arr) ** 2))
	if ss_tot == 0.0:
		return 1.0 if ss_res == 0.0 else float("nan")
	return float(1.0 - (ss_res / ss_tot))


def compute_results(aligned: pd.DataFrame) -> list[ComparisonResult]:
	results: list[ComparisonResult] = []
	for thermal_index, thermal_label, class_suffix in THERMAL_CLASSES:
		heating_actual = aligned["Yearly Heating demand"]
		heating_predicted = aligned[f"rc_heating_{class_suffix}"]
		cooling_actual = aligned["Yearly Cooling demand"]
		cooling_predicted = aligned[f"rc_cooling_{class_suffix}"]
		heating_mask = np.isfinite(np.asarray(heating_actual, dtype=float)) & np.isfinite(np.asarray(heating_predicted, dtype=float))
		cooling_mask = np.isfinite(np.asarray(cooling_actual, dtype=float)) & np.isfinite(np.asarray(cooling_predicted, dtype=float))

		results.append(
			ComparisonResult(
				comparison=f"SimStadt - {thermal_label} heating",
				mape=calculate_mape(heating_actual, heating_predicted),
				r2=calculate_r2(heating_actual, heating_predicted),
				matched_count=int(heating_mask.sum()),
			)
		)
		results.append(
			ComparisonResult(
				comparison=f"SimStadt - {thermal_label} cooling",
				mape=calculate_mape(cooling_actual, cooling_predicted),
				r2=calculate_r2(cooling_actual, cooling_predicted),
				matched_count=int(cooling_mask.sum()),
			)
		)
	return results


def format_value(value: float) -> str:
	if pd.isna(value):
		return "n/a"
	return f"{value:.2f}"


def write_report(output_file: Path, results: list[ComparisonResult], aligned: pd.DataFrame) -> None:
	output_file.parent.mkdir(parents=True, exist_ok=True)
	comparison_width = max(len("Comparison"), max(len(result.comparison) for result in results))
	simstadt_exact = int((aligned["simstadt_match_source"] == "exact").sum()) if "simstadt_match_source" in aligned.columns else 0
	simstadt_fallback = int((aligned["simstadt_match_source"] == "fallback").sum()) if "simstadt_match_source" in aligned.columns else 0
	lines = [
		"Statistical Evaluation Report",
		"",
		f"Matched buildings used: {len(aligned)}",
		f"SimStadt exact matches: {simstadt_exact}",
		f"SimStadt fallback matches: {simstadt_fallback}",
		f"SimStadt source: {DEFAULT_SIMSTADT_FILE.name}",
		f"RC results directory: {DEFAULT_RC_RESULTS_DIR.name}",
		"",
		f"{'Comparison'.ljust(comparison_width)}  {'MAPE'.rjust(10)}  {'R²'.rjust(10)}  {'n'.rjust(6)}",
		f"{'-' * comparison_width}  {'-' * 10}  {'-' * 10}  {'-' * 6}",
	]
	for result in results:
		lines.append(
			f"{result.comparison.ljust(comparison_width)}  {format_value(result.mape).rjust(10)}  {format_value(result.r2).rjust(10)}  {str(result.matched_count).rjust(6)}"
		)
	lines.extend([
		"",
		"Notes:",
		"MAPE is calculated against annual SimStadt demand in percent.",
		"R² is calculated with SimStadt as the reference series.",
		"Cooling and heating rows are aligned by exact ID first, then fallback ID when needed.",
	])
	output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
	args = parse_args()

	matched_pairs = load_matched_pairs(args.matched_file)
	simstadt = load_simstadt_demands(args.simstadt_file)

	rc_frames: dict[str, pd.DataFrame] = {}
	for _, _, class_suffix in THERMAL_CLASSES:
		rc_file = args.rc_results_dir / f"simulated_annual_results_{class_suffix}.csv"
		if not rc_file.exists():
			raise FileNotFoundError(f"Missing RC results file: {rc_file}")
		rc_frames[class_suffix] = load_rc_demands(rc_file, class_suffix)

	aligned = build_aligned_frame(matched_pairs, simstadt, rc_frames)
	results = compute_results(aligned)
	write_report(args.output_file, results, aligned)

	print(f"Wrote {len(results)} comparison rows to {args.output_file}")
	print(f"Matched buildings used: {len(aligned)}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
