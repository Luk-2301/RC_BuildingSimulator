import argparse
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

DEFAULT_RC_CLASS_FOLDER = "03_medium"  
DEFAULT_RC_CLASS_SUFFIX = "_03_medium"  

RC_HOURLY_DIR = DATA_DIR / "Rc_Hourly_Results" / DEFAULT_RC_CLASS_FOLDER
SIMSTADT_HOURLY_DIR = DATA_DIR / "SimStadt_Hourly_Heat_Demand"
RC_AGGREGATED_DIR = DATA_DIR / "Rc_Hourly_Aggregated_Results"
SIMSTADT_AGGREGATED_DIR = DATA_DIR / "SimStadt_Hourly_Aggregated_Results"

COMPARISON_ALL_DIR = DATA_DIR / "comparisons_all"
COMPARISON_INDIVIDUAL_DIR = DATA_DIR / "comparisons_individual"
COMPARISON_PLOTS_DIR = DATA_DIR / "comparison_plots"

RC_HOURLY_PREFIX = "rc_hourly_"
RC_HOURLY_COLUMN = "HeatingDemand_kWh_h"
RC_SOLAR_COLUMN = "SolarGains"
RC_COOLING_COLUMN = "CoolingDemand_kWh_h"

SIMSTADT_PRN_SUFFIX = "_hourly_demand.prn"
SIMSTADT_HOURLY_COLUMN = "Heat Demand"

RC_AGGREGATED_FILE = f"rc_hourly_accumulated{DEFAULT_RC_CLASS_SUFFIX}.csv"
RC_AGGREGATED_COLUMN = "HeatingDemand_kWh_h_sum"
SIMSTADT_DIN_MONTHLY_FILE = "ImBuchwald_DIN18599_HEATING_AND_COOLING.csv"
SIMSTADT_AGGREGATED_CANDIDATES = (
	"ImBuchwald_hourly_demand.csv",
	"ImBuchenwald_hourly_demand.csv",
)
SIMSTADT_AGGREGATED_COLUMN = "Hourly demand"
SIMSTADT_DIN_MONTHLY_COOLING_COLUMNS = (
	"January Cooling demand",
	"February Cooling demand",
	"March Cooling demand",
	"April Cooling demand",
	"May Cooling demand",
	"June Cooling demand",
	"July Cooling demand",
	"August Cooling demand",
	"September Cooling demand",
	"October Cooling demand",
	"November Cooling demand",
	"December Cooling demand",
)

SIMSTADT_PLOT_COLOR = "#C65D3B"	#Teracota
RC_PLOT_COLOR = "#3A7CA5"	#Blue
RC_COOLING_PLOT_COLOR = "#2ca02c"  # Green for cooling need
RC_SOLAR_PLOT_COLOR = "#F4D03F"	#Yellow
SIMSTADT_COOLING_PLOT_COLOR = "#0B3D0B"
RC_INDOOR_PLOT_COLOR = "#2ca02c"
RC_OUTSIDE_PLOT_COLOR = "#6E7F80"
MONTH_NAMES_DE = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
MONTH_HOURS = np.array([744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744], dtype=int)
DEFAULT_WINDOW_HOURS = 24 * 14
DEFAULT_WINDOW_CALENDAR_YEAR = 2013
DEFAULT_WINDOW_START_MONTH = 1
DEFAULT_WINDOW_START_DAY = 28
DEFAULT_WINDOW_START_CLOCK_HOUR = 0
DEFAULT_SPRING2_START_MONTH = 4
DEFAULT_SPRING2_START_DAY = 7
DEFAULT_SUMMER2_START_MONTH = 8
DEFAULT_SUMMER2_START_DAY = 15
RC_INPUT_SNAPSHOT_FILE = f"simulated_input_snapshot{DEFAULT_RC_CLASS_SUFFIX}.csv"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Compare RC and SimStadt hourly heating demand results."
	)
	parser.add_argument(
		"--data-dir",
		type=Path,
		default=DATA_DIR,
		help="Base data directory (defaults to CSV_Import/data).",
	)
	parser.add_argument(
		"--disable-plots",
		action="store_true",
		help="Disable plot output generation.",
	)
	parser.add_argument(
		"--window-start-month",
		type=int,
		default=DEFAULT_WINDOW_START_MONTH,
		help="Start month for the 14-day plot window (default: 1).",
	)
	parser.add_argument(
		"--window-start-day",
		type=int,
		default=DEFAULT_WINDOW_START_DAY,
		help="Start day for the 14-day plot window (default: 28).",
	)
	parser.add_argument(
		"--window-start-clock-hour",
		type=int,
		default=DEFAULT_WINDOW_START_CLOCK_HOUR,
		help="Start clock hour (0-23) for the 14-day plot window (default: 0).",
	)
	parser.add_argument(
		"--spring2-start-month",
		type=int,
		default=DEFAULT_SPRING2_START_MONTH,
		help="Start month for the 14-day second spring plot window (default: 5).",
	)
	parser.add_argument(
		"--spring2-start-day",
		type=int,
		default=DEFAULT_SPRING2_START_DAY,
		help="Start day for the 14-day second spring plot window (default: 15).",
	)
	parser.add_argument(
		"--summer2-start-month",
		type=int,
		default=DEFAULT_SUMMER2_START_MONTH,
		help="Start month for the 14-day second summer plot window (default: 8).",
	)
	parser.add_argument(
		"--summer2-start-day",
		type=int,
		default=DEFAULT_SUMMER2_START_DAY,
		help="Start day for the 14-day second summer plot window (default: 15).",
	)
	return parser.parse_args()


def ensure_output_dirs() -> None:
	COMPARISON_ALL_DIR.mkdir(parents=True, exist_ok=True)
	COMPARISON_INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)
	COMPARISON_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def slugify_label(value: str) -> str:
	text = str(value).strip().lower()
	text = re.sub(r"[^a-z0-9]+", "_", text)
	text = re.sub(r"_+", "_", text).strip("_")
	return text or "unknown"


def with_file_suffix(file_name: str, suffix: str) -> str:
	if not suffix:
		return file_name
	path_obj = Path(file_name)
	return f"{path_obj.stem}{suffix}{path_obj.suffix}"


def append_title_suffix(title: str, title_suffix: str) -> str:
	return f"{title} ({title_suffix})" if title_suffix else title


def resolve_thermal_class_context(snapshot_path: Path) -> dict[str, str]:
	default_context = {
		"file_suffix": "",
		"title_suffix": "",
		"plot_subdir": "general",
	}

	if not snapshot_path.exists():
		return default_context

	try:
		df = pd.read_csv(snapshot_path)
	except Exception:
		return default_context

	if "Thermal class" not in df.columns and "Thermal class index" not in df.columns:
		return default_context

	classes = (
		df["Thermal class"].astype(str).str.strip()
		if "Thermal class" in df.columns
		else pd.Series(dtype=str)
	)
	classes = classes[(classes != "") & (classes.str.lower() != "nan")]
	class_values = sorted(classes.unique().tolist())

	indices: list[int] = []
	if "Thermal class index" in df.columns:
		index_values = pd.to_numeric(df["Thermal class index"], errors="coerce").dropna().astype(int)
		indices = sorted(index_values.unique().tolist())

	if len(class_values) == 1 and len(indices) <= 1:
		class_label = class_values[0]
		class_slug = slugify_label(class_label)
		if indices:
			class_index = indices[0]
			class_id = f"{class_index:02d}_{class_slug}"
			title_suffix = f"Thermische Klasse: {class_label}"
		else:
			class_id = class_slug
			title_suffix = f"Thermal class: {class_label}"
		return {
			"file_suffix": f"_{class_id}",
			"title_suffix": title_suffix,
			"plot_subdir": class_id,
		}

	if len(class_values) > 1:
		return {
			"file_suffix": "_mixed_classes",
			"title_suffix": "Thermal class: mixed",
			"plot_subdir": "mixed_classes",
		}

	return default_context


def normalize_id(value: str) -> str:
	return value.strip().lower()


def strip_variant_suffix(value: str) -> str:
	parts = value.split("__")
	if len(parts) >= 2 and parts[-1].isdigit():
		return "__".join(parts[:-1])
	return value


def get_numeric_variant_suffix(value: str) -> int | None:
	parts = value.split("__")
	if len(parts) >= 2 and parts[-1].isdigit():
		return int(parts[-1])
	return None


def extract_rc_id(file_path: Path) -> str:
	name = file_path.stem
	if name.startswith(RC_HOURLY_PREFIX):
		name = name[len(RC_HOURLY_PREFIX) :]
	# Strip thermal-class file suffixes like _03_medium to recover building ID.
	name = re.sub(r"_0[1-5]_(very_light|light|medium|heavy|very_heavy)$", "", name)
	return name


def extract_simstadt_id(file_path: Path) -> str:
	name = file_path.name
	if name.endswith(SIMSTADT_PRN_SUFFIX):
		return name[: -len(SIMSTADT_PRN_SUFFIX)]
	return file_path.stem


def find_simstadt_aggregated_file(simstadt_aggregated_dir: Path) -> Path:
	for candidate in SIMSTADT_AGGREGATED_CANDIDATES:
		candidate_path = simstadt_aggregated_dir / candidate
		if candidate_path.exists():
			return candidate_path
	available = sorted(path.name for path in simstadt_aggregated_dir.glob("*.csv"))
	raise FileNotFoundError(
		"Could not find SimStadt aggregated file. Expected one of "
		f"{SIMSTADT_AGGREGATED_CANDIDATES}. Available: {available}"
	)


def load_rc_hourly_series(file_path: Path) -> pd.Series:
	df = pd.read_csv(file_path)
	if RC_HOURLY_COLUMN not in df.columns:
		raise ValueError(
			f"Missing '{RC_HOURLY_COLUMN}' in RC hourly file: {file_path.name}"
		)
	series = pd.to_numeric(df[RC_HOURLY_COLUMN], errors="coerce").fillna(0.0)
	return series.astype(float)


def load_rc_solar_series(file_path: Path) -> pd.Series:
	df = pd.read_csv(file_path)
	if RC_SOLAR_COLUMN not in df.columns:
		raise ValueError(
			f"Missing '{RC_SOLAR_COLUMN}' in RC hourly file: {file_path.name}"
		)
	# SolarGains is exported in Wh per hour; convert to kW for consistent monthly kWh sums.
	series = pd.to_numeric(df[RC_SOLAR_COLUMN], errors="coerce").fillna(0.0) / 1000.0
	return series.astype(float)


def load_rc_cooling_series(file_path: Path) -> pd.Series:
	df = pd.read_csv(file_path)
	if RC_COOLING_COLUMN not in df.columns:
		raise ValueError(
			f"Missing '{RC_COOLING_COLUMN}' in RC hourly file: {file_path.name}"
		)
	# Cooling need should be accumulated as positive values.
	series = pd.to_numeric(df[RC_COOLING_COLUMN], errors="coerce").fillna(0.0).abs()
	return series.astype(float)


def load_simstadt_prn_series(file_path: Path) -> pd.Series:
	df = pd.read_csv(
		file_path,
		sep=r"\s+",
		comment="#",
		header=None,
		names=["HOY", "Heat Demand", "Load duration curve", "Dhw Demand"],
		engine="python",
	)
	if SIMSTADT_HOURLY_COLUMN not in df.columns:
		raise ValueError(
			f"Missing '{SIMSTADT_HOURLY_COLUMN}' in SimStadt file: {file_path.name}"
		)
	series = pd.to_numeric(df[SIMSTADT_HOURLY_COLUMN], errors="coerce").fillna(0.0)
	return series.astype(float)


def load_rc_aggregated_series(file_path: Path) -> pd.Series:
	df = pd.read_csv(file_path)
	if RC_AGGREGATED_COLUMN not in df.columns:
		raise ValueError(
			f"Missing '{RC_AGGREGATED_COLUMN}' in RC aggregated file: {file_path.name}"
		)
	series = pd.to_numeric(df[RC_AGGREGATED_COLUMN], errors="coerce").fillna(0.0)
	return series.astype(float)


def load_simstadt_aggregated_series(file_path: Path) -> pd.Series:
	df = pd.read_csv(file_path, sep=";", decimal=",", skiprows=[1])
	if SIMSTADT_AGGREGATED_COLUMN not in df.columns:
		raise ValueError(
			f"Missing '{SIMSTADT_AGGREGATED_COLUMN}' in SimStadt aggregated file: "
			f"{file_path.name}"
		)
	series = pd.to_numeric(df[SIMSTADT_AGGREGATED_COLUMN], errors="coerce").fillna(0.0)
	return series.astype(float)


def build_matched_aggregated_series(
	matched_rows: list[dict[str, str]],
	rc_series_by_id: dict[str, pd.Series],
	sim_series_by_id: dict[str, pd.Series],
) -> tuple[pd.Series, pd.Series]:
	"""Build aggregated RC and SimStadt series from matched buildings only."""
	if not matched_rows:
		raise ValueError("No matched buildings found for aggregation.")

	rc_values_list = []
	sim_values_list = []

	for row in matched_rows:
		rc_id = row["rc_building_id"]
		sim_id = row["simstadt_building_id"]

		rc_series = rc_series_by_id.get(rc_id)
		sim_series = sim_series_by_id.get(sim_id)

		if rc_series is None or sim_series is None:
			continue

		rc_values_list.append(rc_series.to_numpy(dtype=float))
		sim_values_list.append(sim_series.to_numpy(dtype=float))

	if not rc_values_list or not sim_values_list:
		raise ValueError("No valid hourly series found for matched buildings.")

	rc_agg = pd.Series(np.sum(rc_values_list, axis=0), dtype=float)
	sim_agg = pd.Series(np.sum(sim_values_list, axis=0), dtype=float)

	return rc_agg, sim_agg


def load_simstadt_monthly_cooling_sums(file_path: Path) -> np.ndarray:
	if not file_path.exists():
		return np.zeros(12, dtype=float)

	header_index = None
	with file_path.open("r", encoding="utf-8-sig") as handle:
		for index, line in enumerate(handle):
			if line.startswith("GMLId;"):
				header_index = index
				break

	if header_index is None:
		raise ValueError(f"Could not locate DIN header row in {file_path.name}.")

	df = pd.read_csv(
		file_path,
		sep=";",
		decimal=",",
		dtype=str,
		skiprows=header_index,
		engine="python",
		encoding="utf-8-sig",
	)
	if df.empty:
		return np.zeros(12, dtype=float)

	# Remove the units row that starts with "[-]" in the GMLId column.
	if "GMLId" in df.columns:
		df = df[df["GMLId"].astype(str).str.strip() != "[-]"]

	missing = [
		column
		for column in SIMSTADT_DIN_MONTHLY_COOLING_COLUMNS
		if column not in df.columns
	]
	if missing:
		raise ValueError(
			"Missing DIN monthly cooling columns in "
			f"{file_path.name}: {missing}"
		)

	monthly_sums = []
	for column in SIMSTADT_DIN_MONTHLY_COOLING_COLUMNS:
		values = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
		monthly_sums.append(float(values.sum()))

	return np.array(monthly_sums, dtype=float)


def load_simstadt_annual_cooling_by_id(file_path: Path) -> dict[str, float]:
	"""Return a mapping of GMLId -> annual cooling sum (kWh) from the DIN monthly table.

	If the file or expected header is missing, an empty dict is returned.
	"""
	if not file_path.exists():
		return {}

	header_index = None
	with file_path.open("r", encoding="utf-8-sig") as handle:
		for index, line in enumerate(handle):
			if line.startswith("GMLId;"):
				header_index = index
				break

	if header_index is None:
		return {}

	df = pd.read_csv(
		file_path,
		sep=";",
		decimal=",",
		dtype=str,
		skiprows=header_index,
		engine="python",
		encoding="utf-8-sig",
	)
	if df.empty or "GMLId" not in df.columns:
		return {}

	# Remove unit/placeholder rows that start with "[-]"
	if "GMLId" in df.columns:
		df = df[df["GMLId"].astype(str).str.strip() != "[-]"]

	result: dict[str, float] = {}
	for _, row in df.iterrows():
		gid = str(row.get("GMLId", "")).strip()
		if not gid or gid.lower() == "nan":
			continue
		vals = []
		for col in SIMSTADT_DIN_MONTHLY_COOLING_COLUMNS:
			val = pd.to_numeric(row.get(col, 0), errors="coerce")
			vals.append(float(val) if pd.notna(val) else 0.0)
		total = float(sum(vals))
		result[gid] = total

	return result


def load_heated_area_by_building_id(file_path: Path) -> dict[str, float]:
	if not file_path.exists():
		return {}

	df = pd.read_csv(file_path)
	required_columns = {"Heated area [m2]"}
	if not required_columns.issubset(df.columns):
		return {}

	areas = pd.to_numeric(df["Heated area [m2]"], errors="coerce")
	if "Building ID [filename]" in df.columns:
		building_ids = df["Building ID [filename]"].astype(str)
	elif "Building ID" in df.columns:
		building_ids = df["Building ID"].astype(str)
	else:
		return {}
	result: dict[str, float] = {}
	for building_id, area in zip(building_ids, areas):
		if pd.notna(area) and float(area) > 0:
			result[building_id] = float(area)
	return result


def align_series(rc_series: pd.Series, sim_series: pd.Series) -> tuple[pd.Series, pd.Series]:
	min_len = min(len(rc_series), len(sim_series))
	if min_len == 0:
		return rc_series.iloc[0:0], sim_series.iloc[0:0]
	return (
		rc_series.iloc[:min_len].reset_index(drop=True),
		sim_series.iloc[:min_len].reset_index(drop=True),
	)


def compute_hourly_metrics(rc_series: pd.Series, sim_series: pd.Series) -> dict[str, float]:
	rc_aligned, sim_aligned = align_series(rc_series, sim_series)
	if len(rc_aligned) == 0:
		return {
			"hours_compared": 0,
			"mae_kwh_h": np.nan,
			"rmse_kwh_h": np.nan,
			"bias_kwh_h": np.nan,
			"corr": np.nan,
			"annual_rc_kwh": 0.0,
			"annual_simstadt_kwh": 0.0,
			"annual_diff_kwh": 0.0,
			"annual_diff_pct": np.nan,
		}

	diff = rc_aligned - sim_aligned
	mae = float(np.mean(np.abs(diff)))
	rmse = float(np.sqrt(np.mean(np.square(diff))))
	bias = float(np.mean(diff))
	corr = float(np.corrcoef(rc_aligned, sim_aligned)[0, 1]) if len(rc_aligned) > 1 else np.nan

	annual_rc = float(rc_aligned.sum())
	annual_sim = float(sim_aligned.sum())
	total_metrics = compute_total_metrics(annual_rc, annual_sim, "heating")

	return {
		"hours_compared": int(len(rc_aligned)),
		"mae_kwh_h": mae,
		"rmse_kwh_h": rmse,
		"bias_kwh_h": bias,
		"corr": corr,
		"annual_rc_kwh": total_metrics["annual_rc_heating_kwh"],
		"annual_simstadt_kwh": total_metrics["annual_simstadt_heating_kwh"],
		"annual_diff_kwh": total_metrics["annual_diff_heating_kwh"],
		"annual_diff_pct": total_metrics["annual_diff_heating_pct"],
	}


def compute_total_metrics(rc_value: float, sim_value: float, value_name: str) -> dict[str, float]:
	rc_total = float(rc_value)
	sim_total = float(sim_value)
	diff_total = rc_total - sim_total
	diff_pct = (diff_total / sim_total * 100.0) if sim_total != 0 else np.nan
	return {
		f"annual_rc_{value_name}_kwh": rc_total,
		f"annual_simstadt_{value_name}_kwh": sim_total,
		f"annual_diff_{value_name}_kwh": diff_total,
		f"annual_diff_{value_name}_pct": diff_pct,
	}


def build_suffix_index(simstadt_ids: list[str]) -> dict[str, list[str]]:
	suffix_map: dict[str, list[str]] = {}
	for sim_id in simstadt_ids:
		normalized = normalize_id(strip_variant_suffix(sim_id))
		suffix_map.setdefault(normalized, []).append(sim_id)
	return suffix_map


def pick_suffix_match(rc_id: str, simstadt_ids: list[str]) -> str | None:
	rc_norm = normalize_id(strip_variant_suffix(rc_id))
	matches: list[tuple[int, str]] = []
	for sim_id in simstadt_ids:
		sim_norm = normalize_id(strip_variant_suffix(sim_id))
		if rc_norm.endswith(sim_norm) or sim_norm.endswith(rc_norm):
			score = min(len(rc_norm), len(sim_norm))
			matches.append((score, sim_id))

	if not matches:
		return None

	matches.sort(key=lambda item: item[0], reverse=True)
	top_score = matches[0][0]
	top_matches = [match_id for score, match_id in matches if score == top_score]
	if len(top_matches) == 1:
		return top_matches[0]

	def tie_break_key(sim_id: str) -> tuple[int, int, str]:
		normalized_sim = normalize_id(sim_id)
		variant = get_numeric_variant_suffix(sim_id)
		is_exact_base_without_variant = variant is None and normalized_sim == rc_norm
		return (
			0 if is_exact_base_without_variant else 1,
			variant if variant is not None else 10**9,
			normalized_sim,
		)

	return min(top_matches, key=tie_break_key)


def match_building_ids(
	rc_ids: list[str], simstadt_ids: list[str]
) -> tuple[list[dict[str, str]], list[str], list[str]]:
	rc_norm_map = {normalize_id(rc_id): rc_id for rc_id in rc_ids}
	sim_norm_map = {normalize_id(sim_id): sim_id for sim_id in simstadt_ids}

	matched_rows: list[dict[str, str]] = []
	matched_rc: set[str] = set()
	matched_sim: set[str] = set()

	# Pass 1: exact full-ID match.
	for rc_norm, rc_id in rc_norm_map.items():
		sim_id = sim_norm_map.get(rc_norm)
		if sim_id is not None:
			matched_rows.append(
				{
					"rc_building_id": rc_id,
					"simstadt_building_id": sim_id,
					"match_type": "exact",
				}
			)
			matched_rc.add(rc_id)
			matched_sim.add(sim_id)

	remaining_rc = [rc_id for rc_id in rc_ids if rc_id not in matched_rc]
	remaining_sim = [sim_id for sim_id in simstadt_ids if sim_id not in matched_sim]

	# Pass 2: suffix fallback match.
	for rc_id in remaining_rc:
		sim_id = pick_suffix_match(rc_id, remaining_sim)
		if sim_id is None:
			continue
		matched_rows.append(
			{
				"rc_building_id": rc_id,
				"simstadt_building_id": sim_id,
				"match_type": "suffix_fallback",
			}
		)
		matched_rc.add(rc_id)
		matched_sim.add(sim_id)
		remaining_sim = [candidate for candidate in remaining_sim if candidate != sim_id]

	unmatched_rc = [rc_id for rc_id in rc_ids if rc_id not in matched_rc]
	unmatched_sim = [sim_id for sim_id in simstadt_ids if sim_id not in matched_sim]
	return matched_rows, unmatched_rc, unmatched_sim


def write_matching_reports(
	matched_rows: list[dict[str, str]],
	unmatched_rc: list[str],
	unmatched_sim: list[str],
) -> None:
	matched_df = pd.DataFrame(matched_rows)
	unmatched_rc_df = pd.DataFrame({"rc_building_id": unmatched_rc})
	unmatched_sim_df = pd.DataFrame({"simstadt_building_id": unmatched_sim})

	matched_df.to_csv(COMPARISON_ALL_DIR / "matched_buildings.csv", index=False)
	unmatched_rc_df.to_csv(COMPARISON_ALL_DIR / "unmatched_rc_buildings.csv", index=False)
	unmatched_sim_df.to_csv(COMPARISON_ALL_DIR / "ignored_unmatched_simstadt_buildings.csv", index=False)


def create_accumulated_hourly_plot(
	rc_series: pd.Series,
	sim_series: pd.Series,
	output_path: Path,
	title_suffix: str = "",
) -> None:
	rc_monthly = compute_monthly_sums(rc_series) / 1_000_000.0
	sim_monthly = compute_monthly_sums(sim_series) / 1_000_000.0
	x_positions = np.arange(1, 13)
	bar_width = 0.2

	fig, ax = plt.subplots(figsize=(12, 5))
	ax.bar(
		x_positions - bar_width / 2,
		sim_monthly,
		width=bar_width,
		color=SIMSTADT_PLOT_COLOR,
		alpha=0.85,
		label="SimStadt cumulative monthly demand",
	)
	ax.bar(
		x_positions + bar_width / 2,
		rc_monthly,
		width=bar_width,
		color=RC_PLOT_COLOR,
		label="RC cumulative monthly demand",
	)
	ax.set_title(append_title_suffix("Monthly cumulative heating demand of the portfolio: RC vs SimStadt", title_suffix))
	ax.set_xlabel("Month")
	ax.set_ylabel("Heating demand in GWh/month")
	ax.set_xticks(x_positions)
	ax.set_xticklabels(MONTH_NAMES_DE)
	ax.ticklabel_format(axis="y", style="plain", useOffset=False)
	ax.grid(alpha=0.25)
	ax.legend(loc="upper center")
	fig.tight_layout()
	fig.savefig(output_path, dpi=180)
	plt.close(fig)


def create_annual_hourly_line_plot(
	rc_series: pd.Series,
	sim_series: pd.Series,
	rc_cooling_series: pd.Series,
	output_path: Path,
	title_suffix: str = "",
) -> None:
	rc_aligned, sim_aligned = align_series(rc_series, sim_series)
	if rc_aligned.empty or sim_aligned.empty:
		return

	rc_hourly = rc_aligned.to_numpy(dtype=float)
	sim_hourly = sim_aligned.to_numpy(dtype=float)
	x = np.arange(len(rc_hourly), dtype=int)

	fig, ax = plt.subplots(figsize=(14, 6))
	ax.plot(
		x,
		rc_hourly,
		color=RC_PLOT_COLOR,
		linewidth=1.2,
		zorder=2,
		label="RC hourly heating demand",
	)

	if not rc_cooling_series.empty:
		rc_cooling_aligned = pd.to_numeric(rc_cooling_series, errors="coerce").fillna(0.0)
		rc_cooling_aligned = rc_cooling_aligned.iloc[: len(x)].reset_index(drop=True)
		rc_cooling_hourly = rc_cooling_aligned.to_numpy(dtype=float)
		ax.plot(
			x[: len(rc_cooling_hourly)],
			rc_cooling_hourly,
			color=RC_COOLING_PLOT_COLOR,
			linewidth=1.2,
			linestyle="--",
			zorder=3,
			label="RC hourly cooling demand",
		)

	ax.plot(
		x,
		sim_hourly,
		color=SIMSTADT_PLOT_COLOR,
		linewidth=1.2,
		zorder=4,
		label="SimStadt hourly heating demand",
	)

	month_bounds = np.concatenate(([0], np.cumsum(MONTH_HOURS)))
	valid_month_count = int(np.sum(month_bounds[1:] <= len(x)))
	if valid_month_count > 0:
		month_centers = (month_bounds[:valid_month_count] + month_bounds[1 : valid_month_count + 1]) / 2.0
		ax.set_xticks(month_centers)
		ax.set_xticklabels(MONTH_NAMES_DE[:valid_month_count])
		# Draw month separator lines between months (at boundaries), not at label centers.
		boundary_positions = month_bounds[1:valid_month_count]
		ax.set_xticks(boundary_positions, minor=True)

	ax.set_title(append_title_suffix("Annual hourly energy demand profile: RC vs SimStadt + RC cooling demand", title_suffix))
	ax.set_xlabel("Month")
	ax.set_ylabel("Hourly energy demand in kW")
	ax.grid(axis="y", alpha=0.25)
	ax.grid(axis="x", which="minor", alpha=0.25)
	ax.legend(loc="upper center")

	fig.tight_layout()
	fig.savefig(output_path, dpi=180)
	plt.close(fig)


def create_accumulated_hourly_cooling_plot(
	rc_heating_series: pd.Series,
	sim_heating_series: pd.Series,
	rc_cooling_series: pd.Series,
	output_path: Path,
	title_suffix: str = "",
) -> None:
	if rc_cooling_series.empty:
		return
	rc_heating_monthly = compute_monthly_sums(rc_heating_series) / 1_000_000.0
	sim_heating_monthly = compute_monthly_sums(sim_heating_series) / 1_000_000.0
	rc_cooling_monthly = compute_monthly_sums(rc_cooling_series) / 1_000_000.0
	x_positions = np.arange(1, 13)
	bar_width = 0.2

	fig, ax = plt.subplots(figsize=(12, 5))
	ax.bar(
		x_positions - bar_width,
		sim_heating_monthly,
		width=bar_width,
		color=SIMSTADT_PLOT_COLOR,
		alpha=0.85,
		label="SimStadt heating demand",
	)
	ax.bar(
		x_positions,
		rc_heating_monthly,
		width=bar_width,
		color=RC_PLOT_COLOR,
		label="RC heating demand",
	)
	ax.bar(
		x_positions + bar_width,
		rc_cooling_monthly,
		width=bar_width,
		color=RC_COOLING_PLOT_COLOR,
		label="RC cooling demand",
	)
	ax.set_title(append_title_suffix("Monthly heating and cooling demand of the portfolio", title_suffix))
	ax.set_xlabel("Month")
	ax.set_ylabel("Heating and cooling demand in GWh/month")
	ax.set_xticks(x_positions)
	ax.set_xticklabels(MONTH_NAMES_DE)
	ax.ticklabel_format(axis="y", style="plain", useOffset=False)
	ax.grid(alpha=0.25)
	ax.legend(loc="upper center")
	fig.tight_layout()
	fig.savefig(output_path, dpi=180)
	plt.close(fig)


def create_annual_scatter_plot(
	summary_df: pd.DataFrame,
	output_path: Path,
	title_suffix: str = "",
) -> None:
	if summary_df.empty:
		return

	fig, ax = plt.subplots(figsize=(7, 7))
	x = summary_df["annual_simstadt_kwh"]
	y = summary_df["annual_rc_kwh"]
	ax.scatter(x, y, alpha=0.65, color=SIMSTADT_PLOT_COLOR)

	max_val = float(max(x.max(), y.max())) if len(summary_df) else 1.0
	ax.plot([0, max_val], [0, max_val], linestyle="--", linewidth=1.0, color="black")

	ax.set_title(append_title_suffix("Jährlicher Heizbedarf pro Gebäude", title_suffix))
	ax.set_xlabel("SimStadt jährlicher Heizbedarf in kWh/a")
	ax.set_ylabel("RC jährlicher Heizbedarf in kWh/a")
	ax.grid(alpha=0.25)
	fig.tight_layout()
	fig.savefig(output_path, dpi=180)
	plt.close(fig)


def create_annual_cooling_scatter_plot(
	summary_df: pd.DataFrame,
	output_path: Path,
	title_suffix: str = "",
) -> None:
	if summary_df.empty:
		return

	fig, ax = plt.subplots(figsize=(7, 7))
	x = summary_df["annual_simstadt_cooling_kwh"]
	y = summary_df["annual_rc_cooling_kwh"]
	ax.scatter(x, y, alpha=0.65, color=RC_PLOT_COLOR)

	max_val = float(max(x.max(), y.max())) if len(summary_df) else 1.0
	ax.plot([0, max_val], [0, max_val], linestyle="--", linewidth=1.0, color="black")

	ax.set_title(append_title_suffix("Jährlicher Kühlbedarf pro Gebäude", title_suffix))
	ax.set_xlabel("SimStadt jährlicher Kühlbedarf in kWh/a")
	ax.set_ylabel("RC jährlicher Kühlbedarf in kWh/a")
	ax.grid(alpha=0.25)
	fig.tight_layout()
	fig.savefig(output_path, dpi=180)
	plt.close(fig)


def create_specific_heating_demand_bins_plot(
	per_building_df: pd.DataFrame,
	heated_area_by_id: dict[str, float],
	output_path: Path,
	bin_width: int = 50,
	title_suffix: str = "",
) -> None:
	if per_building_df.empty or not heated_area_by_id:
		return

	plot_df = per_building_df.copy()
	plot_df["heated_area_m2"] = plot_df["rc_building_id"].map(heated_area_by_id)
	plot_df = plot_df.dropna(
		subset=["annual_rc_kwh", "annual_simstadt_kwh", "heated_area_m2"]
	)
	plot_df = plot_df[plot_df["heated_area_m2"] > 0]
	if plot_df.empty:
		return

	plot_df["specific_rc_kwh_m2a"] = plot_df["annual_rc_kwh"] / plot_df["heated_area_m2"]
	plot_df["specific_sim_kwh_m2a"] = (
		plot_df["annual_simstadt_kwh"] / plot_df["heated_area_m2"]
	)

	all_values = np.concatenate(
		[
			plot_df["specific_sim_kwh_m2a"].to_numpy(dtype=float),
			plot_df["specific_rc_kwh_m2a"].to_numpy(dtype=float),
		]
	)
	max_value = float(np.nanmax(all_values)) if len(all_values) else 0.0
	if not np.isfinite(max_value) or max_value <= 0:
		return

	upper_edge = int(np.ceil(max_value / bin_width) * bin_width)
	bins = np.arange(0, upper_edge + bin_width, bin_width, dtype=float)
	if len(bins) < 2:
		bins = np.array([0.0, float(bin_width)])

	sim_counts, _ = np.histogram(plot_df["specific_sim_kwh_m2a"], bins=bins)
	rc_counts, _ = np.histogram(plot_df["specific_rc_kwh_m2a"], bins=bins)

	x_positions = np.arange(len(sim_counts))
	bar_width = 0.38
	labels = [
		f"{int(bins[i])}-{int(bins[i + 1])}" for i in range(len(bins) - 1)
	]

	fig, ax = plt.subplots(figsize=(14, 6))
	ax.bar(
		x_positions - bar_width / 2,
		sim_counts,
		width=bar_width,
		color=SIMSTADT_PLOT_COLOR,
		alpha=0.85,
		label="SimStadt (specific heating demand in kWh/m²a)",
	)
	ax.bar(
		x_positions + bar_width / 2,
		rc_counts,
		width=bar_width,
		color=RC_PLOT_COLOR,
		label="RC (specific heating demand in kWh/m²a)",
	)

	ax.set_title(append_title_suffix("Specific heating demand in 50 kWh/m²a bins (matched building IDs only)", title_suffix))
	ax.set_xlabel("Specific heating demand bins in kWh/m²a")
	ax.set_ylabel("Number of buildings")
	ax.set_xticks(x_positions)
	ax.set_xticklabels(labels, rotation=45, ha="right")
	ax.grid(axis="y", alpha=0.25)
	ax.legend(loc="upper right")

	for bar in ax.patches:
		height = bar.get_height()
		if height > 0:
			ax.annotate(
				f"{int(height)}",
				xy=(bar.get_x() + bar.get_width() / 2, height),
				xytext=(0, 3),
				textcoords="offset points",
				ha="center",
				va="bottom",
				fontsize=7,
			)

	fig.tight_layout()
	fig.savefig(output_path, dpi=180)
	plt.close(fig)


def create_specific_cooling_demand_bins_plot(
	per_building_df: pd.DataFrame,
	heated_area_by_id: dict[str, float],
	output_path: Path,
	bin_width: int = 50,
	title_suffix: str = "",
) -> None:
	if per_building_df.empty or not heated_area_by_id:
		return

	plot_df = per_building_df.copy()
	plot_df["heated_area_m2"] = plot_df["rc_building_id"].map(heated_area_by_id)
	plot_df = plot_df.dropna(
		subset=["annual_rc_cooling_kwh", "annual_simstadt_cooling_kwh", "heated_area_m2"]
	)
	plot_df = plot_df[plot_df["heated_area_m2"] > 0]
	if plot_df.empty:
		return

	plot_df["specific_rc_cooling_kwh_m2a"] = (
		plot_df["annual_rc_cooling_kwh"] / plot_df["heated_area_m2"]
	)
	plot_df["specific_sim_cooling_kwh_m2a"] = (
		plot_df["annual_simstadt_cooling_kwh"] / plot_df["heated_area_m2"]
	)

	all_values = np.concatenate(
		[
			plot_df["specific_sim_cooling_kwh_m2a"].to_numpy(dtype=float),
			plot_df["specific_rc_cooling_kwh_m2a"].to_numpy(dtype=float),
		]
	)
	max_value = float(np.nanmax(all_values)) if len(all_values) else 0.0
	if not np.isfinite(max_value) or max_value <= 0:
		return

	upper_edge = int(np.ceil(max_value / bin_width) * bin_width)
	bins = np.arange(0, upper_edge + bin_width, bin_width, dtype=float)
	if len(bins) < 2:
		bins = np.array([0.0, float(bin_width)])

	sim_counts, _ = np.histogram(plot_df["specific_sim_cooling_kwh_m2a"], bins=bins)
	rc_counts, _ = np.histogram(plot_df["specific_rc_cooling_kwh_m2a"], bins=bins)

	x_positions = np.arange(len(sim_counts))
	bar_width = 0.38
	labels = [
		f"{int(bins[i])}-{int(bins[i + 1])}" for i in range(len(bins) - 1)
	]

	fig, ax = plt.subplots(figsize=(14, 6))
	sim_bars = ax.bar(
		x_positions - bar_width / 2,
		sim_counts,
		width=bar_width,
		color=SIMSTADT_PLOT_COLOR,
		alpha=0.85,
		label="SimStadt (specific cooling demand in kWh/m²a)",
	)
	rc_bars = ax.bar(
		x_positions + bar_width / 2,
		rc_counts,
		width=bar_width,
		color=RC_COOLING_PLOT_COLOR,
		label="RC (specific cooling demand in kWh/m²a)",
	)

	for bar in list(sim_bars) + list(rc_bars):
		height = bar.get_height()
		if height > 0:
			ax.annotate(
				f"{int(height)}",
				xy=(bar.get_x() + bar.get_width() / 2, height),
				xytext=(0, 3),
				textcoords="offset points",
				ha="center",
				va="bottom",
				fontsize=7,
			)

	ax.set_title(append_title_suffix("Specific cooling demand in 50 kWh/m²a bins (matched building IDs only)", title_suffix))
	ax.set_xlabel("Specific cooling demand bins in kWh/m²a")
	ax.set_ylabel("Number of buildings")
	ax.set_xticks(x_positions)
	ax.set_xticklabels(labels, rotation=45, ha="right")
	ax.grid(axis="y", alpha=0.25)
	ax.legend(loc="upper right")

	fig.tight_layout()
	fig.savefig(output_path, dpi=180)
	plt.close(fig)


def create_specific_heating_demand_two_row_scatter_plot(
	per_building_df: pd.DataFrame,
	heated_area_by_id: dict[str, float],
	output_path: Path,
	title_suffix: str = "",
) -> None:
	if per_building_df.empty or not heated_area_by_id:
		return

	plot_df = per_building_df.copy()
	plot_df["heated_area_m2"] = plot_df["rc_building_id"].map(heated_area_by_id)
	plot_df = plot_df.dropna(
		subset=["annual_rc_kwh", "annual_simstadt_kwh", "heated_area_m2"]
	)
	plot_df = plot_df[plot_df["heated_area_m2"] > 0]
	if plot_df.empty:
		return

	rc_specific = (
		pd.to_numeric(plot_df["annual_rc_kwh"], errors="coerce")
		/ plot_df["heated_area_m2"]
	)
	sim_specific = (
		pd.to_numeric(plot_df["annual_simstadt_kwh"], errors="coerce")
		/ plot_df["heated_area_m2"]
	)

	rc_specific = rc_specific.replace([np.inf, -np.inf], np.nan).dropna()
	sim_specific = sim_specific.replace([np.inf, -np.inf], np.nan).dropna()
	rc_specific = rc_specific[(rc_specific >= 1.0) & (rc_specific <= 600.0)]
	sim_specific = sim_specific[(sim_specific >= 1.0) & (sim_specific <= 600.0)]
	if rc_specific.empty and sim_specific.empty:
		return

	rng = np.random.default_rng(42)
	y_sim = np.zeros(len(sim_specific)) + rng.normal(0.0, 0.03, len(sim_specific))
	y_rc = np.ones(len(rc_specific)) + rng.normal(0.0, 0.03, len(rc_specific))

	fig, ax = plt.subplots(figsize=(14, 6))
	ax.scatter(
		sim_specific.to_numpy(dtype=float),
		y_sim,
		s=24,
		alpha=0.75,
		color=SIMSTADT_PLOT_COLOR,
		label="SimStadt",
	)
	ax.scatter(
		rc_specific.to_numpy(dtype=float),
		y_rc,
		s=24,
		alpha=0.75,
		color=RC_PLOT_COLOR,
		label="RC",
	)

	ax.set_xlim(1, 600)
	ax.set_ylim(-0.35, 1.35)
	ax.set_yticks([0, 1])
	ax.set_yticklabels(["SimStadt", "RC"])
	ax.set_xlabel("Specific heating demand in kWh/m²a")
	ax.set_ylabel("Model")
	ax.set_title(append_title_suffix("Specific heating demand as scatter plot (matched building IDs only)", title_suffix))
	ax.grid(axis="x", alpha=0.25)
	ax.legend(loc="upper right")

	fig.tight_layout()
	fig.savefig(output_path, dpi=180)
	plt.close(fig)


def create_specific_cooling_demand_two_row_scatter_plot(
	per_building_df: pd.DataFrame,
	heated_area_by_id: dict[str, float],
	output_path: Path,
	title_suffix: str = "",
) -> None:
	if per_building_df.empty or not heated_area_by_id:
		return

	plot_df = per_building_df.copy()
	plot_df["heated_area_m2"] = plot_df["rc_building_id"].map(heated_area_by_id)
	plot_df = plot_df.dropna(
		subset=["annual_rc_cooling_kwh", "annual_simstadt_cooling_kwh", "heated_area_m2"]
	)
	plot_df = plot_df[plot_df["heated_area_m2"] > 0]
	if plot_df.empty:
		return

	rc_specific = (
		pd.to_numeric(plot_df["annual_rc_cooling_kwh"], errors="coerce")
		/ plot_df["heated_area_m2"]
	)
	sim_specific = (
		pd.to_numeric(plot_df["annual_simstadt_cooling_kwh"], errors="coerce")
		/ plot_df["heated_area_m2"]
	)

	rc_specific = rc_specific.replace([np.inf, -np.inf], np.nan).dropna()
	sim_specific = sim_specific.replace([np.inf, -np.inf], np.nan).dropna()
	rc_specific = rc_specific[(rc_specific >= 1.0) & (rc_specific <= 600.0)]
	sim_specific = sim_specific[(sim_specific >= 1.0) & (sim_specific <= 600.0)]
	if rc_specific.empty and sim_specific.empty:
		return

	rng = np.random.default_rng(42)
	y_sim = np.zeros(len(sim_specific)) + rng.normal(0.0, 0.03, len(sim_specific))
	y_rc = np.ones(len(rc_specific)) + rng.normal(0.0, 0.03, len(rc_specific))

	fig, ax = plt.subplots(figsize=(14, 6))
	ax.scatter(
		sim_specific.to_numpy(dtype=float),
		y_sim,
		s=24,
		alpha=0.75,
		color=SIMSTADT_COOLING_PLOT_COLOR,
		label="SimStadt",
	)
	ax.scatter(
		rc_specific.to_numpy(dtype=float),
		y_rc,
		s=24,
		alpha=0.75,
		color=RC_COOLING_PLOT_COLOR,
		label="RC",
	)

	ax.set_xlim(1, 600)
	ax.set_ylim(-0.35, 1.35)
	ax.set_yticks([0, 1])
	ax.set_yticklabels(["SimStadt", "RC"])
	ax.set_xlabel("Specific cooling demand in kWh/m²a")
	ax.set_ylabel("Model")
	ax.set_title(append_title_suffix("Specific cooling demand as scatter plot (matched building IDs only)", title_suffix))
	ax.grid(axis="x", alpha=0.25)
	ax.legend(loc="upper right")

	fig.tight_layout()
	fig.savefig(output_path, dpi=180)
	plt.close(fig)




def compute_monthly_sums(series: pd.Series) -> np.ndarray:
	values = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
	required_len = int(MONTH_HOURS.sum())
	if len(values) < required_len:
		raise ValueError(
			f"Series has only {len(values)} values, but {required_len} are required for monthly sums."
		)
	values = values[:required_len]
	bounds = np.concatenate(([0], np.cumsum(MONTH_HOURS)))
	return np.array([values[bounds[i] : bounds[i + 1]].sum() for i in range(12)], dtype=float)


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


def create_monthly_columns_plot(
	matched_rows: list[dict[str, str]],
	rc_series_by_id: dict[str, pd.Series],
	rc_solar_by_id: dict[str, pd.Series],
	sim_series_by_id: dict[str, pd.Series],
	output_path: Path,
	title_suffix: str = "",
) -> None:
	if not matched_rows:
		return

	sim_monthly_accumulated = np.zeros(12, dtype=float)
	rc_monthly_accumulated = np.zeros(12, dtype=float)
	rc_monthly_solar_accumulated = np.zeros(12, dtype=float)

	for row in matched_rows:
		rc_id = row["rc_building_id"]
		sim_id = row["simstadt_building_id"]

		rc_series = rc_series_by_id.get(rc_id)
		rc_solar_series = rc_solar_by_id.get(rc_id)
		sim_series = sim_series_by_id.get(sim_id)
		if rc_series is None or rc_solar_series is None or sim_series is None:
			continue

		rc_monthly = compute_monthly_sums(rc_series)
		rc_monthly_solar = compute_monthly_sums(rc_solar_series)
		sim_monthly = compute_monthly_sums(sim_series)

		rc_monthly_accumulated += rc_monthly
		rc_monthly_solar_accumulated += rc_monthly_solar
		sim_monthly_accumulated += sim_monthly

	if (
		np.all(sim_monthly_accumulated == 0.0)
		and np.all(rc_monthly_accumulated == 0.0)
		and np.all(rc_monthly_solar_accumulated == 0.0)
	):
		return

	x_positions = np.arange(1, 13)
	bar_width = 0.38

	# Convert from kWh/month to MWh/month before plotting.
	sim_monthly_accumulated = sim_monthly_accumulated / 1_000.0
	rc_monthly_accumulated = rc_monthly_accumulated / 1_000.0
	rc_monthly_solar_accumulated = rc_monthly_solar_accumulated / 1_000.0

	fig, ax = plt.subplots(figsize=(14, 7))
	bar1 = ax.bar(
		x_positions - bar_width / 2,
		sim_monthly_accumulated,
		width=bar_width,
		color=SIMSTADT_PLOT_COLOR,
		alpha=0.85,
		label="SimStadt Heizbedarf",
	)
	bar2 = ax.bar(
		x_positions + bar_width / 2,
		rc_monthly_accumulated,
		width=bar_width,
		color=RC_PLOT_COLOR,
		label="RC Heizbedarf",
	)
	bar3 = ax.bar(
		x_positions + bar_width / 2,
		rc_monthly_solar_accumulated,
		width=bar_width,
		bottom=rc_monthly_accumulated,
		color=RC_SOLAR_PLOT_COLOR,
		label="RC solarer Wärmeintrag",
	)

	# Add value labels on top of SimStadt bars
	for bar in bar1:
		height = bar.get_height()
		if height > 0:
			ax.annotate(
				f"{height:.1f}",
				xy=(bar.get_x() + bar.get_width() / 2, height),
				xytext=(0, 3),
				textcoords="offset points",
				ha="center",
				va="bottom",
				fontsize=7,
				rotation=90,
			)

	# Add value labels on stacked RC bars (demand and solar separately)
	for i, pos in enumerate(x_positions):
		# Label for demand portion
		if rc_monthly_accumulated[i] > 0:
			ax.annotate(
				f"{rc_monthly_accumulated[i]:.1f}",
				xy=(pos + bar_width / 2, rc_monthly_accumulated[i] / 2),
				xytext=(0, 0),
				textcoords="offset points",
				ha="center",
				va="center",
				fontsize=7,
				rotation=90,
			)
		# Label for solar portion
		if rc_monthly_solar_accumulated[i] > 0:
			ax.annotate(
				f"{rc_monthly_solar_accumulated[i]:.1f}",
				xy=(pos + bar_width / 2, rc_monthly_accumulated[i] + rc_monthly_solar_accumulated[i] / 2),
				xytext=(0, 0),
				textcoords="offset points",
				ha="center",
				va="center",
				fontsize=7,
				rotation=90,
			)

	ax.set_title(append_title_suffix("Monatlicher Heizbedarf mit solarem Wärmeintrag für RC", title_suffix))
	ax.set_xlabel("Monat")
	ax.set_ylabel("Heizbedarf in MWh/Monat")
	ax.set_xticks(x_positions)
	ax.set_xticklabels(MONTH_NAMES_DE)
	ax.ticklabel_format(axis="y", style="plain", useOffset=False)
	ax.grid(axis="y", alpha=0.25)
	ax.legend(loc="upper center")

	fig.tight_layout()
	fig.savefig(output_path, dpi=180)
	plt.close(fig)


def create_monthly_columns_plot_with_cooling(
	matched_rows: list[dict[str, str]],
	rc_series_by_id: dict[str, pd.Series],
	rc_solar_by_id: dict[str, pd.Series],
	rc_cooling_by_id: dict[str, pd.Series],
	sim_series_by_id: dict[str, pd.Series],
	simstadt_monthly_cooling_sums: np.ndarray,
	output_path: Path,
	title_suffix: str = "",
) -> None:
	if not matched_rows:
		return

	sim_monthly_accumulated = np.zeros(12, dtype=float)
	rc_monthly_accumulated = np.zeros(12, dtype=float)
	rc_monthly_solar_accumulated = np.zeros(12, dtype=float)
	rc_monthly_cooling_accumulated = np.zeros(12, dtype=float)

	for row in matched_rows:
		rc_id = row["rc_building_id"]
		sim_id = row["simstadt_building_id"]

		rc_series = rc_series_by_id.get(rc_id)
		rc_solar_series = rc_solar_by_id.get(rc_id)
		rc_cooling_series = rc_cooling_by_id.get(rc_id)
		sim_series = sim_series_by_id.get(sim_id)
		if (
			rc_series is None
			or rc_solar_series is None
			or rc_cooling_series is None
			or sim_series is None
		):
			continue

		rc_monthly = compute_monthly_sums(rc_series)
		rc_monthly_solar = compute_monthly_sums(rc_solar_series)
		rc_monthly_cooling = compute_monthly_sums(rc_cooling_series)
		sim_monthly = compute_monthly_sums(sim_series)

		rc_monthly_accumulated += rc_monthly
		rc_monthly_solar_accumulated += rc_monthly_solar
		rc_monthly_cooling_accumulated += rc_monthly_cooling
		sim_monthly_accumulated += sim_monthly

	if (
		np.all(sim_monthly_accumulated == 0.0)
		and np.all(rc_monthly_accumulated == 0.0)
		and np.all(rc_monthly_solar_accumulated == 0.0)
		and np.all(rc_monthly_cooling_accumulated == 0.0)
	):
		return

	x_positions = np.arange(1, 13)
	bar_width = 0.2

	# Convert from kWh/month to MWh/month before plotting.
	sim_monthly_accumulated = sim_monthly_accumulated / 1_000.0
	rc_monthly_accumulated = rc_monthly_accumulated / 1_000.0
	rc_monthly_solar_accumulated = rc_monthly_solar_accumulated / 1_000.0
	rc_monthly_cooling_accumulated = rc_monthly_cooling_accumulated / 1_000.0
	simstadt_monthly_cooling_accumulated = (
		pd.to_numeric(pd.Series(simstadt_monthly_cooling_sums), errors="coerce")
		.fillna(0.0)
		.to_numpy(dtype=float)
	)
	if len(simstadt_monthly_cooling_accumulated) != 12:
		raise ValueError(
			"SimStadt monthly cooling sums must contain exactly 12 values."
		)
	simstadt_monthly_cooling_accumulated = simstadt_monthly_cooling_accumulated / 1_000.0

	fig, ax = plt.subplots(figsize=(14, 7))
	bar1 = ax.bar(
		x_positions - 1.5 * bar_width,
		sim_monthly_accumulated,
		width=bar_width,
		color=SIMSTADT_PLOT_COLOR,
		alpha=0.85,
		label="SimStadt Heizbedarf",
	)
	bar2 = ax.bar(
		x_positions - 0.5 * bar_width,
		rc_monthly_accumulated,
		width=bar_width,
		color=RC_PLOT_COLOR,
		label="RC Heizbedarf",
	)
	bar3 = ax.bar(
		x_positions - 0.5 * bar_width,
		rc_monthly_solar_accumulated,
		width=bar_width,
		bottom=rc_monthly_accumulated,
		color=RC_SOLAR_PLOT_COLOR,
		label="RC solarer Wärmeintrag",
	)
	bar4 = ax.bar(
		x_positions + 0.5 * bar_width,
		rc_monthly_cooling_accumulated,
		width=bar_width,
		color=RC_COOLING_PLOT_COLOR,
		label="RC Kühlbedarf",
	)
	bar5 = ax.bar(
		x_positions + 1.5 * bar_width,
		simstadt_monthly_cooling_accumulated,
		width=bar_width,
		color=SIMSTADT_COOLING_PLOT_COLOR,
		label="SimStadt Kühlbedarf",
	)

	# Add value labels on top of each bar group
	for bar in bar1:
		height = bar.get_height()
		if height > 0:
			ax.annotate(
				f"{height:.1f}",
				xy=(bar.get_x() + bar.get_width() / 2, height),
				xytext=(0, 3),
				textcoords="offset points",
				ha="center",
				va="bottom",
				fontsize=7,
				rotation=90,
			)

	# Add value labels on stacked RC bars (demand and solar separately)
	for i, pos in enumerate(x_positions):
		# Label for demand portion
		if rc_monthly_accumulated[i] > 0:
			ax.annotate(
				f"{rc_monthly_accumulated[i]:.1f}",
				xy=(pos - 0.5 * bar_width, rc_monthly_accumulated[i] / 2),
				xytext=(0, 0),
				textcoords="offset points",
				ha="center",
				va="center",
				fontsize=7,
				rotation=90,
			)
		# Label for solar portion
		if rc_monthly_solar_accumulated[i] > 0:
			ax.annotate(
				f"{rc_monthly_solar_accumulated[i]:.1f}",
				xy=(pos - 0.5 * bar_width, rc_monthly_accumulated[i] + rc_monthly_solar_accumulated[i] / 2),
				xytext=(0, 0),
				textcoords="offset points",
				ha="center",
				va="center",
				fontsize=7,
				rotation=90,
			)

	# Add value labels on top of RC cooling bars
	for bar in bar4:
		height = bar.get_height()
		if height > 0:
			ax.annotate(
				f"{height:.1f}",
				xy=(bar.get_x() + bar.get_width() / 2, height),
				xytext=(0, 3),
				textcoords="offset points",
				ha="center",
				va="bottom",
				fontsize=7,
				rotation=90,
			)

	# Add value labels on top of SimStadt cooling bars
	for bar in bar5:
		height = bar.get_height()
		if height > 0:
			ax.annotate(
				f"{height:.1f}",
				xy=(bar.get_x() + bar.get_width() / 2, height),
				xytext=(0, 3),
				textcoords="offset points",
				ha="center",
				va="bottom",
				fontsize=7,
				rotation=90,
			)

	ax.set_title(append_title_suffix("Monatlicher Heiz- und Kühlbedarf mit solarem Wärmeintrag für RC", title_suffix))
	ax.set_xlabel("Monat")
	ax.set_ylabel("Heiz und Kühlbedarf in MWh/Monat")
	ax.set_xticks(x_positions)
	ax.set_xticklabels(MONTH_NAMES_DE)
	ax.ticklabel_format(axis="y", style="plain", useOffset=False)
	ax.grid(axis="y", alpha=0.25)
	ax.legend(loc="upper center", bbox_to_anchor=(0.3, 1),)


	fig.tight_layout()
	fig.savefig(output_path, dpi=180)
	plt.close(fig)


def create_hourly_heating_heatmap_comparison_plot(
	rc_series: pd.Series,
	sim_series: pd.Series,
	output_path: Path,
	title_suffix: str = "",
) -> None:
	rc_aligned, sim_aligned = align_series(rc_series, sim_series)
	if rc_aligned.empty or sim_aligned.empty:
		return

	hours_per_day = 24
	day_count = len(rc_aligned) // hours_per_day
	if day_count <= 0:
		return

	usable_hours = day_count * hours_per_day
	rc_values = rc_aligned.iloc[:usable_hours].to_numpy(dtype=float)
	sim_values = sim_aligned.iloc[:usable_hours].to_numpy(dtype=float)

	rc_matrix = rc_values.reshape(day_count, hours_per_day).T
	sim_matrix = sim_values.reshape(day_count, hours_per_day).T
	diff_matrix = rc_matrix - sim_matrix

	max_common = float(max(np.nanmax(rc_matrix), np.nanmax(sim_matrix)))
	if not np.isfinite(max_common) or max_common <= 0:
		max_common = 1.0

	max_abs_diff = float(np.nanmax(np.abs(diff_matrix)))
	if not np.isfinite(max_abs_diff) or max_abs_diff <= 0:
		max_abs_diff = 1.0

	fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, constrained_layout=True)

	im0 = axes[0].imshow(
		rc_matrix,
		aspect="auto",
		cmap="viridis",
		origin="upper",
		vmin=0.0,
		vmax=max_common,
		interpolation="nearest",
	)
	axes[0].set_title("RC")
	axes[0].set_xlabel("DOY")
	axes[0].set_ylabel("HOD")

	im1 = axes[1].imshow(
		sim_matrix,
		aspect="auto",
		cmap="viridis",
		origin="upper",
		vmin=0.0,
		vmax=max_common,
		interpolation="nearest",

	)
	axes[1].set_title("SimStadt")
	axes[1].set_xlabel("DOY")

	im2 = axes[2].imshow(
		diff_matrix,
		aspect="auto",
		cmap="RdBu_r",
		origin="upper",
		vmin=-max_abs_diff,
		vmax=max_abs_diff,
		interpolation="nearest",
	)
	axes[2].set_title("RC - SimStadt")
	axes[2].set_xlabel("DOY")

	if day_count > 1:
		x_tick_count = min(12, day_count)
		x_tick_positions = np.linspace(0, day_count - 1, num=x_tick_count, dtype=int)
		x_tick_labels = [str(int(position + 1)) for position in x_tick_positions]
		for axis in axes:
			axis.set_xticks(x_tick_positions)
			axis.set_xticklabels(x_tick_labels, rotation=90)

	y_tick_positions = np.arange(0, 24, 1)
	for axis in axes:
		axis.set_yticks(y_tick_positions - 0.5) 
		#axis.set_yticklabels([str(int(value)) for value in y_tick_positions])
		axis.set_yticklabels([str(int(v)) if i % 2 == 0 else "" for i, v in enumerate(y_tick_positions)])

	fig.suptitle(
		append_title_suffix("Heatmap des stündlichen Gesamtheizbedarfs", title_suffix),
		fontsize=14,
	)

	colorbar0 = fig.colorbar(im0, ax=axes[:2], location="left", fraction=0.046, pad=0.04)
	colorbar0.set_label("Heizbedarf in kW", rotation=90, labelpad=14)
	colorbar0.ax.yaxis.set_ticks_position("left")
	colorbar0.ax.yaxis.set_label_position("left")
	colorbar0.ax.tick_params(labelleft=True, labelright=False, left=True, right=False, pad=4)

	colorbar1 = fig.colorbar(im2, ax=axes[:3], location="right", fraction=0.046, pad=0.04)
	colorbar1.set_label("Unterschied in kW", labelpad=14)
	colorbar1.ax.yaxis.set_ticks_position("right")
	colorbar1.ax.yaxis.set_label_position("right")
	colorbar1.ax.tick_params(labelleft=False, labelright=True, left=False, right=True, pad=4)

	fig.savefig(output_path, dpi=180)
	plt.close(fig)


def create_aggregated_14day_heat_plot(
	rc_series: pd.Series,
	sim_series: pd.Series,
	rc_cooling_series: pd.Series,
	rc_solar_series: pd.Series,
	rc_indoor_series: pd.Series,
	ambient_temp_series: pd.Series,
	matched_building_count: int,
	start_hour: int,
	window_hours: int,
	calendar_year: int,
	output_path: Path,
	title_suffix: str = "",
) -> None:
	rc_aligned, sim_aligned = align_series(rc_series, sim_series)
	if rc_aligned.empty or sim_aligned.empty:
		return

	min_len = min(
		len(rc_aligned),
		len(rc_solar_series),
		len(rc_cooling_series),
		len(rc_indoor_series),
		len(ambient_temp_series),
	)
	if min_len == 0:
		return

	rc_aligned = rc_aligned.iloc[:min_len].reset_index(drop=True)
	sim_aligned = sim_aligned.iloc[:min_len].reset_index(drop=True)
	rc_solar_series = rc_solar_series.iloc[:min_len].reset_index(drop=True)
	rc_cooling_series = rc_cooling_series.iloc[:min_len].reset_index(drop=True)
	rc_indoor_series = rc_indoor_series.iloc[:min_len].reset_index(drop=True)
	ambient_temp_series = ambient_temp_series.iloc[:min_len].reset_index(drop=True)

	start = max(0, int(start_hour))
	end = min(len(rc_aligned), start + int(window_hours))
	if end <= start:
		return

	x = np.arange(end - start)
	rc_window = rc_aligned.iloc[start:end].to_numpy(dtype=float)
	sim_window = sim_aligned.iloc[start:end].to_numpy(dtype=float)
	rc_solar_window = rc_solar_series.iloc[start:end].to_numpy(dtype=float)
	rc_cooling_window = rc_cooling_series.iloc[start:end].to_numpy(dtype=float)
	rc_indoor_window = rc_indoor_series.iloc[start:end].to_numpy(dtype=float)
	ambient_temp_window = ambient_temp_series.iloc[start:end].to_numpy(dtype=float)

	fig, ax = plt.subplots(figsize=(14, 6))
	ax.plot(x, rc_window, color=RC_PLOT_COLOR, linewidth=1.8, label="RC Heizbedarf in kW")
	ax.plot(x, sim_window, color=SIMSTADT_PLOT_COLOR, linewidth=1.8, label="SimStadt Heizbedarf in kW")
	ax.plot(x, rc_cooling_window, color=RC_COOLING_PLOT_COLOR, linewidth=1, linestyle="--", label="RC Kühlbedarf in kW")
	ax.plot(x, rc_solar_window, color=RC_SOLAR_PLOT_COLOR, linewidth=1, label="RC solarer Wärmeeintrag in kW")

	ax.set_title(append_title_suffix("Kumulierter Energiebedarf: RC vs SimStadt", title_suffix))
	ax.set_xlabel("Tage")
	ax.set_ylabel("Energiebedarf in kW")
	ax.grid(alpha=0.25)

	tick_positions = np.arange(0, end - start, 24)
	date_origin = pd.Timestamp(year=calendar_year, month=1, day=1)
	date_labels = [
		(date_origin + pd.Timedelta(hours=int(start + pos))).strftime("%d-%b")
		for pos in tick_positions
	]
	ax.set_xticks(tick_positions)
	ax.set_xticklabels(date_labels, rotation=45, ha="right")

	ax_temp = ax.twinx()
	ax_temp.plot(
		x,
		rc_indoor_window,
		color=RC_INDOOR_PLOT_COLOR,
		linewidth=1.2,
		linestyle=":",
		label="RC Raumlufttemperatur in °C",
	)
	ax_temp.plot(
		x,
		ambient_temp_window,
		color=RC_OUTSIDE_PLOT_COLOR,
		linewidth=1.2,
		linestyle=":",
		label="Außenlufttemperatur in °C",
	)
	ax_temp.set_ylabel("Temperatur in °C")

	lines_left, labels_left = ax.get_legend_handles_labels()
	lines_right, labels_right = ax_temp.get_legend_handles_labels()
	fig.legend(
		lines_left + lines_right,
		labels_left + labels_right,
		loc="lower center",
		bbox_to_anchor=(0.5, 0.05),
		ncol=3,
		fontsize=9,
		frameon=False,
	)

	fig.text(
		0.5,
		0.015,
		f"Anzahl der gezeigten Gebäude: {matched_building_count}",
		ha="center",
		va="bottom",
		fontsize=9,
	)
	fig.tight_layout(rect=(0.0, 0.16, 1.0, 1.0))
	fig.savefig(output_path, dpi=180)
	plt.close(fig)


def run_comparison(
	data_dir: Path,
	disable_plots: bool,
	window_start_month: int,
	window_start_day: int,
	window_start_clock_hour: int,
	spring2_start_month: int,
	spring2_start_day: int,
	summer2_start_month: int,
	summer2_start_day: int,
) -> None:
	global DATA_DIR
	global RC_HOURLY_DIR
	global SIMSTADT_HOURLY_DIR
	global RC_AGGREGATED_DIR
	global SIMSTADT_AGGREGATED_DIR
	global COMPARISON_ALL_DIR
	global COMPARISON_INDIVIDUAL_DIR
	global COMPARISON_PLOTS_DIR

	DATA_DIR = data_dir.resolve()
	RC_HOURLY_DIR = DATA_DIR / "Rc_Hourly_Results" / DEFAULT_RC_CLASS_FOLDER
	SIMSTADT_HOURLY_DIR = DATA_DIR / "SimStadt_Hourly_Heat_Demand"
	RC_AGGREGATED_DIR = DATA_DIR / "Rc_Hourly_Aggregated_Results"
	SIMSTADT_AGGREGATED_DIR = DATA_DIR / "SimStadt_Hourly_Aggregated_Results"
	COMPARISON_ALL_DIR = DATA_DIR / "comparisons_all"
	COMPARISON_INDIVIDUAL_DIR = DATA_DIR / "comparisons_individual"
	COMPARISON_PLOTS_DIR = DATA_DIR / "comparison_plots"

	ensure_output_dirs()
	thermal_context = resolve_thermal_class_context(RC_AGGREGATED_DIR / RC_INPUT_SNAPSHOT_FILE)
	file_suffix = thermal_context["file_suffix"]
	title_suffix = thermal_context["title_suffix"]
	plot_output_dir = COMPARISON_PLOTS_DIR / thermal_context["plot_subdir"]
	plot_output_dir.mkdir(parents=True, exist_ok=True)

	rc_files = sorted(RC_HOURLY_DIR.glob(f"{RC_HOURLY_PREFIX}*.csv"))
	simstadt_files = sorted(SIMSTADT_HOURLY_DIR.glob(f"*{SIMSTADT_PRN_SUFFIX}"))

	if not rc_files:
		raise FileNotFoundError(f"No RC hourly files found in: {RC_HOURLY_DIR}")
	if not simstadt_files:
		raise FileNotFoundError(
			f"No SimStadt hourly files found in: {SIMSTADT_HOURLY_DIR}"
		)

	rc_series_by_id = {extract_rc_id(path): load_rc_hourly_series(path) for path in rc_files}
	rc_solar_by_id = {extract_rc_id(path): load_rc_solar_series(path) for path in rc_files}
	rc_cooling_by_id = {extract_rc_id(path): load_rc_cooling_series(path) for path in rc_files}
	sim_series_by_id = {
		extract_simstadt_id(path): load_simstadt_prn_series(path) for path in simstadt_files
	}

	rc_ids = sorted(rc_series_by_id.keys())
	sim_ids = sorted(sim_series_by_id.keys())

	matched_rows, unmatched_rc, unmatched_sim = match_building_ids(rc_ids, sim_ids)
	write_matching_reports(matched_rows, unmatched_rc, unmatched_sim)
	if file_suffix:
		matched_df = pd.DataFrame(matched_rows)
		unmatched_rc_df = pd.DataFrame({"rc_building_id": unmatched_rc})
		unmatched_sim_df = pd.DataFrame({"simstadt_building_id": unmatched_sim})
		matched_df.to_csv(COMPARISON_ALL_DIR / with_file_suffix("matched_buildings.csv", file_suffix), index=False)
		unmatched_rc_df.to_csv(COMPARISON_ALL_DIR / with_file_suffix("unmatched_rc_buildings.csv", file_suffix), index=False)
		unmatched_sim_df.to_csv(COMPARISON_ALL_DIR / with_file_suffix("ignored_unmatched_simstadt_buildings.csv", file_suffix), index=False)

	# Build matched-only aggregated series from hourly data (for portfolio plots)
	rc_agg_series, sim_agg_series = build_matched_aggregated_series(
		matched_rows, rc_series_by_id, sim_series_by_id
	)

	# Build matched-only cooling aggregation (for portfolio plots)
	rc_cooling_values_list = [
		rc_cooling_by_id[row["rc_building_id"]].to_numpy(dtype=float)
		for row in matched_rows
		if row["rc_building_id"] in rc_cooling_by_id
	]
	if rc_cooling_values_list:
		rc_cooling_accumulated_series = pd.Series(
			np.sum(rc_cooling_values_list, axis=0), dtype=float
		)
	else:
		rc_cooling_accumulated_series = pd.Series(dtype=float)

	# Load SimStadt cooling data early for per-building comparison
	simstadt_monthly_cooling_sums = load_simstadt_monthly_cooling_sums(
		DATA_DIR / SIMSTADT_DIN_MONTHLY_FILE
	)
	simstadt_annual_cooling = float(np.asarray(simstadt_monthly_cooling_sums, dtype=float).sum())

	# Load per-building SimStadt cooling sums (GMLId -> annual cooling kWh)
	simstadt_cooling_by_id = load_simstadt_annual_cooling_by_id(DATA_DIR / SIMSTADT_DIN_MONTHLY_FILE)
	# Also build a normalized-key map to improve lookup robustness (strip/lower)
	simstadt_cooling_by_id_norm = {normalize_id(k): v for k, v in simstadt_cooling_by_id.items()}

	comparison_rows: list[dict[str, float | str | int]] = []
	
	for row in matched_rows:
		rc_id = row["rc_building_id"]
		sim_id = row["simstadt_building_id"]

		metrics = compute_hourly_metrics(rc_series_by_id[rc_id], sim_series_by_id[sim_id])
		
		# Calculate annual RC cooling for this building
		rc_cooling_series = rc_cooling_by_id.get(rc_id, pd.Series(dtype=float))
		annual_rc_cooling = float(np.abs(rc_cooling_series.sum())) if not rc_cooling_series.empty else 0.0

		# Use per-building SimStadt cooling (GMLId) for comparison; fall back to 0.0 when missing
		# Try exact match, then stripped-variant, then normalized match.
		sim_annual_for_building = 0.0
		if sim_id in simstadt_cooling_by_id:
			sim_annual_for_building = float(simstadt_cooling_by_id[sim_id])
		else:
			# try stripping numeric variant suffix like __01
			stripped = strip_variant_suffix(sim_id)
			if stripped in simstadt_cooling_by_id:
				sim_annual_for_building = float(simstadt_cooling_by_id[stripped])
			else:
				# try normalized (lower/strip) match against normalized map
				norm_key = normalize_id(sim_id)
				if norm_key in simstadt_cooling_by_id_norm:
					sim_annual_for_building = float(simstadt_cooling_by_id_norm[norm_key])
				else:
					stripped_norm = normalize_id(stripped)
					if stripped_norm in simstadt_cooling_by_id_norm:
						sim_annual_for_building = float(simstadt_cooling_by_id_norm[stripped_norm])
					else:
						logging.getLogger(__name__).warning(
							"SimStadt DIN cooling entry not found for building GMLId '%s' (tried stripped '%s') - using 0.0",
							sim_id,
							stripped,
						)
		cooling_metrics = compute_total_metrics(annual_rc_cooling, sim_annual_for_building, "cooling")
		
		comparison_rows.append(
			{
				"rc_building_id": rc_id,
				"simstadt_building_id": sim_id,
				"match_type": row["match_type"],
				**cooling_metrics,
				**metrics,
			}
		)

	per_building_df = pd.DataFrame(comparison_rows)
	per_building_df.to_csv(
		COMPARISON_INDIVIDUAL_DIR / with_file_suffix("per_building_hourly_comparison.csv", file_suffix),
		index=False,
	)

	if not per_building_df.empty:
		summary = {
			"matched_buildings": int(len(per_building_df)),
			"mean_mae_kwh_h": float(per_building_df["mae_kwh_h"].mean()),
			"mean_rmse_kwh_h": float(per_building_df["rmse_kwh_h"].mean()),
			"mean_bias_kwh_h": float(per_building_df["bias_kwh_h"].mean()),
			"mean_corr": float(per_building_df["corr"].mean()),
			"sum_annual_rc_kwh": float(per_building_df["annual_rc_kwh"].sum()),
			"sum_annual_simstadt_kwh": float(per_building_df["annual_simstadt_kwh"].sum()),
			"sum_annual_diff_kwh": float(per_building_df["annual_diff_kwh"].sum()),
			"sum_annual_rc_cooling_kwh": float(per_building_df["annual_rc_cooling_kwh"].sum()),
			"sum_annual_simstadt_cooling_kwh": float(per_building_df["annual_simstadt_cooling_kwh"].sum()),
			"sum_annual_diff_cooling_kwh": float(per_building_df["annual_diff_cooling_kwh"].sum()),
		}
	else:
		summary = {
			"matched_buildings": 0,
			"mean_mae_kwh_h": np.nan,
			"mean_rmse_kwh_h": np.nan,
			"mean_bias_kwh_h": np.nan,
			"mean_corr": np.nan,
			"sum_annual_rc_kwh": 0.0,
			"sum_annual_simstadt_kwh": 0.0,
			"sum_annual_diff_kwh": 0.0,
			"sum_annual_rc_cooling_kwh": 0.0,
			"sum_annual_simstadt_cooling_kwh": 0.0,
			"sum_annual_diff_cooling_kwh": 0.0,
		}

	pd.DataFrame([summary]).to_csv(
		COMPARISON_ALL_DIR / with_file_suffix("per_building_comparison_summary.csv", file_suffix),
		index=False,
	)

	# Build matched-only solar, indoor, and ambient series for portfolio plots
	rc_solar_values_list = [
		rc_solar_by_id[row["rc_building_id"]].to_numpy(dtype=float)
		for row in matched_rows
		if row["rc_building_id"] in rc_solar_by_id
	]
	if rc_solar_values_list:
		rc_solar_agg_series = pd.Series(
			np.sum(rc_solar_values_list, axis=0), dtype=float
		)
	else:
		rc_solar_agg_series = pd.Series(dtype=float)

	# Get aggregated file to extract indoor temp (taking average from matched buildings)
	rc_aggregated_path = RC_AGGREGATED_DIR / RC_AGGREGATED_FILE
	sim_aggregated_path = find_simstadt_aggregated_file(SIMSTADT_AGGREGATED_DIR)

	rc_agg_df = pd.read_csv(rc_aggregated_path)
	rc_indoor_values_list = [
		pd.to_numeric(rc_agg_df.get("IndoorAir_mean", pd.Series(dtype=float)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
		for row in matched_rows
		if row["rc_building_id"] in rc_series_by_id
	]
	if rc_indoor_values_list:
		rc_indoor_agg_series = pd.Series(
			np.mean(rc_indoor_values_list, axis=0), dtype=float
		)
	else:
		rc_indoor_agg_series = pd.Series(dtype=float)

	sim_agg_df = pd.read_csv(sim_aggregated_path,sep=";", decimal=",", skiprows=[1])
	ambient_temp_agg_series = pd.to_numeric(
		sim_agg_df.get("Ambient temperature", pd.Series(dtype=float)), errors="coerce"
	).fillna(0.0)

	heated_area_by_id = load_heated_area_by_building_id(
		RC_AGGREGATED_DIR / RC_INPUT_SNAPSHOT_FILE
	)

	rc_agg_aligned, sim_agg_aligned = align_series(rc_agg_series, sim_agg_series)
	sim_agg_export = pd.DataFrame(
		{
			"hour": np.arange(1, len(sim_agg_aligned) + 1),
			"simstadt_heating_demand_kwh_h": sim_agg_aligned.values,
		}
	)
	sim_agg_export.to_csv(
		COMPARISON_ALL_DIR / with_file_suffix("simstadt_hourly_accumulated.csv", file_suffix),
		index=False,
	)

	agg_df = pd.DataFrame(
		{
			"hour": np.arange(1, len(rc_agg_aligned) + 1),
			"rc_heating_demand_kwh_h": rc_agg_aligned.values,
			"simstadt_heating_demand_kwh_h": sim_agg_aligned.values,
		}
	)
	agg_df["diff_kwh_h"] = agg_df["rc_heating_demand_kwh_h"] - agg_df[
		"simstadt_heating_demand_kwh_h"
	]
	agg_df.to_csv(
		COMPARISON_ALL_DIR / with_file_suffix("hourly_accumulated_comparison.csv", file_suffix),
		index=False,
	)

	agg_metrics = compute_hourly_metrics(rc_agg_series, sim_agg_series)
	rc_cooling_kwh = float(np.abs(rc_cooling_accumulated_series.sum()))
	simstadt_cooling_kwh = float(np.asarray(simstadt_monthly_cooling_sums, dtype=float).sum())
	annual_cooling_diff_kwh = rc_cooling_kwh - simstadt_cooling_kwh
	annual_cooling_diff_pct = (
		annual_cooling_diff_kwh / simstadt_cooling_kwh * 100.0
		if simstadt_cooling_kwh != 0
		else np.nan
	)
	agg_metrics.update(
		{
			"annual_rc_cooling_kwh": rc_cooling_kwh,
			"annual_simstadt_cooling_kwh": simstadt_cooling_kwh,
			"annual_diff_cooling_kwh": annual_cooling_diff_kwh,
			"annual_diff_cooling_pct": annual_cooling_diff_pct,
			"annual_rc_cooling_mwh": rc_cooling_kwh / 1000.0,
			"annual_simstadt_cooling_mwh": simstadt_cooling_kwh / 1000.0,
			"annual_diff_cooling_mwh": annual_cooling_diff_kwh / 1000.0,
		}
	)
	pd.DataFrame([agg_metrics]).to_csv(
		COMPARISON_ALL_DIR / with_file_suffix("hourly_accumulated_comparison_summary.csv", file_suffix),
		index=False,
	)

	annual_df = per_building_df[
		[
			"rc_building_id",
			"simstadt_building_id",
			"annual_rc_kwh",
			"annual_simstadt_kwh",
			"annual_diff_kwh",
			"annual_diff_pct",
		]
	]
	annual_df.to_csv(
		COMPARISON_INDIVIDUAL_DIR / with_file_suffix("annual_comparison_per_building.csv", file_suffix),
		index=False,
	)

	if not disable_plots:
		window_start_hour = calendar_to_hour_of_year(
			calendar_year=DEFAULT_WINDOW_CALENDAR_YEAR,
			month=window_start_month,
			day=window_start_day,
			clock_hour=window_start_clock_hour,
		)
		create_accumulated_hourly_plot(
			rc_agg_series,
			sim_agg_series,
			plot_output_dir / with_file_suffix("hourly_accumulated_rc_vs_simstadt.png", file_suffix),
			title_suffix=title_suffix,
		)
		create_annual_hourly_line_plot(
			rc_agg_series,
			sim_agg_series,
			rc_cooling_accumulated_series,
			plot_output_dir / with_file_suffix("annual_hourly_rc_vs_simstadt_line.png", file_suffix),
			title_suffix=title_suffix,
		)
		create_annual_scatter_plot(
			per_building_df,
			plot_output_dir / with_file_suffix("annual_building_scatter_rc_vs_simstadt.png", file_suffix),
			title_suffix=title_suffix,
		)
		create_annual_cooling_scatter_plot(
			per_building_df,
			plot_output_dir / with_file_suffix("annual_cooling_scatter_rc_vs_simstadt.png", file_suffix),
			title_suffix=title_suffix,
		)
		create_specific_heating_demand_bins_plot(
			per_building_df,
			heated_area_by_id,
			plot_output_dir / with_file_suffix("specific_heating_demand_50kwh_bins_rc_vs_simstadt.png", file_suffix),
			title_suffix=title_suffix,
		)
		create_specific_cooling_demand_bins_plot(
			per_building_df,
			heated_area_by_id,
			plot_output_dir / with_file_suffix("specific_cooling_demand_50kwh_bins_rc_vs_simstadt.png", file_suffix),
			title_suffix=title_suffix,
		)
		create_specific_heating_demand_two_row_scatter_plot(
			per_building_df,
			heated_area_by_id,
			plot_output_dir / with_file_suffix("specific_heating_demand_scatter_rc_vs_simstadt.png", file_suffix),
			title_suffix=title_suffix,
		)
		create_specific_cooling_demand_two_row_scatter_plot(
			per_building_df,
			heated_area_by_id,
			plot_output_dir / with_file_suffix("specific_cooling_demand_scatter_rc_vs_simstadt.png", file_suffix),
			title_suffix=title_suffix,
		)
		create_monthly_columns_plot(
			matched_rows,
			rc_series_by_id,
			rc_solar_by_id,
			sim_series_by_id,
			plot_output_dir / with_file_suffix("monthly_accumulated_simstadt_vs_rc_columns.png", file_suffix),
			title_suffix=title_suffix,
		)
		create_monthly_columns_plot_with_cooling(
			matched_rows,
			rc_series_by_id,
			rc_solar_by_id,
			rc_cooling_by_id,
			sim_series_by_id,
			simstadt_monthly_cooling_sums,
			plot_output_dir / with_file_suffix("monthly_accumulated_simstadt_vs_rc_columns_with_cooling.png", file_suffix),
			title_suffix=title_suffix,
		)
		create_hourly_heating_heatmap_comparison_plot(
			rc_agg_series,
			sim_agg_series,
			plot_output_dir / with_file_suffix("hourly_heating_heatmap_rc_vs_simstadt.png", file_suffix),
			title_suffix=title_suffix,
		)

		# Additional 14-day windows: Spring2 and Summer2, plus the primary window
		spring2_start_hour = calendar_to_hour_of_year(
			calendar_year=DEFAULT_WINDOW_CALENDAR_YEAR,
			month=spring2_start_month,
			day=spring2_start_day,
			clock_hour=window_start_clock_hour,
		)
		summer2_start_hour = calendar_to_hour_of_year(
			calendar_year=DEFAULT_WINDOW_CALENDAR_YEAR,
			month=summer2_start_month,
			day=summer2_start_day,
			clock_hour=window_start_clock_hour,
		)

		windows = [
			("primary", window_start_hour, "Winter"),
			("spring2", spring2_start_hour, "Frühling"),
			("summer2", summer2_start_hour, "Sommer"),
		]

		for name, start_hour, season_label in windows:
			create_aggregated_14day_heat_plot(
				rc_agg_series,
				sim_agg_series,
				rc_cooling_accumulated_series,
				rc_solar_agg_series,
				rc_indoor_agg_series,
				ambient_temp_agg_series,
				matched_building_count=len(matched_rows),
				start_hour=start_hour,
				window_hours=DEFAULT_WINDOW_HOURS,
				calendar_year=DEFAULT_WINDOW_CALENDAR_YEAR,
				output_path=plot_output_dir / with_file_suffix(f"aggregated_14day_rc_vs_simstadt_{name}.png", file_suffix),
				title_suffix=append_title_suffix(title_suffix, season_label),
			)

	print(f"Matched buildings: {len(matched_rows)}")
	print(f"Unmatched RC buildings: {len(unmatched_rc)}")
	print(f"Ignored unmatched SimStadt buildings: {len(unmatched_sim)}")
	print(f"Outputs written to: {DATA_DIR}")


def main() -> None:
	args = parse_args()
	run_comparison(
		args.data_dir,
		disable_plots=args.disable_plots,
		window_start_month=args.window_start_month,
		window_start_day=args.window_start_day,
		window_start_clock_hour=args.window_start_clock_hour,
		spring2_start_month=args.spring2_start_month,
		spring2_start_day=args.spring2_start_day,
		summer2_start_month=args.summer2_start_month,
		summer2_start_day=args.summer2_start_day,
	)


if __name__ == "__main__":
	main()


