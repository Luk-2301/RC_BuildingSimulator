from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
RC_AGGREGATED_DIR = DATA_DIR / "Rc_Hourly_Aggregated_Results"
SIMSTADT_DIN_PATH = DATA_DIR / "ImBuchwald_DIN18599_HEATING_AND_COOLING.csv"
PLOT_DIR = DATA_DIR / "comparison_plots" / "general"
REPORT_DIR = DATA_DIR / "comparisons_all"

RC_ANNUAL_PATTERN = "simulated_annual_results_0[1-5]_*.csv"
RC_HEATING_COL = "Annual Heating Demand Sim [kWh]"
RC_COOLING_COL = "Annual Cooling Demand Sim [kWh]"
SIMSTADT_HEATING_COL = "Yearly Heating demand"
SIMSTADT_COOLING_COL = "Yearly Cooling demand"
SIMSTADT_GMLID_COL = "GMLId"
SIMSTADT_HEATED_AREA_COL = "Heated area"
RC_BUILDING_ID_COL = "Building ID"
RC_HEATED_AREA_COL = "Heated area [m2]"

THERMAL_CLASS_LABELS = {
	"01": "seht leicht",
	"02": "leicht",
	"03": "mittel",
	"04": "schwer",
	"05": "sehr schwer",
}

RC_HEATING_COLOR = "#3A7CA5"
RC_COOLING_COLOR = "#2ca02c"


def _to_numeric_series(series: pd.Series) -> pd.Series:
	if series.dtype.kind in "biufc":
		return pd.to_numeric(series, errors="coerce").fillna(0.0)

	return pd.to_numeric(
		series.astype(str).str.replace(",", ".", regex=False), errors="coerce"
	).fillna(0.0)


def _class_index_from_filename(path: Path) -> str:
	name = path.stem
	parts = name.split("_")
	if len(parts) < 4:
		raise ValueError(f"Unexpected annual RC filename: {path.name}")
	return parts[3]


def _normalize_building_id(value: str) -> str:
	return str(value).strip().lower()


def _strip_variant_suffix(value: str) -> str:
	parts = str(value).split("__")
	if len(parts) >= 2 and parts[-1].isdigit():
		return "__".join(parts[:-1])
	return str(value)


def _get_numeric_variant_suffix(value: str) -> int | None:
	parts = str(value).split("__")
	if len(parts) >= 2 and parts[-1].isdigit():
		return int(parts[-1])
	return None


def _pick_suffix_match(rc_id: str, simstadt_ids: list[str]) -> str | None:
	rc_norm = _normalize_building_id(_strip_variant_suffix(rc_id))
	matches: list[tuple[int, str]] = []
	for sim_id in simstadt_ids:
		sim_norm = _normalize_building_id(_strip_variant_suffix(sim_id))
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
		normalized_sim = _normalize_building_id(sim_id)
		variant = _get_numeric_variant_suffix(sim_id)
		is_exact_base_without_variant = variant is None and normalized_sim == rc_norm
		return (
			0 if is_exact_base_without_variant else 1,
			variant if variant is not None else 10**9,
			normalized_sim,
		)

	return min(top_matches, key=tie_break_key)


def _match_building_ids(
	rc_ids: list[str], simstadt_ids: list[str]
) -> tuple[list[dict[str, str]], list[str], list[str]]:
	rc_norm_map = {_normalize_building_id(rc_id): rc_id for rc_id in rc_ids}
	sim_norm_map = {_normalize_building_id(sim_id): sim_id for sim_id in simstadt_ids}

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
		sim_id = _pick_suffix_match(rc_id, remaining_sim)
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


def build_matched_comparison_dataset() -> dict[str, Any]:
	class_files = sorted(
		RC_AGGREGATED_DIR.glob(RC_ANNUAL_PATTERN), key=lambda p: _class_index_from_filename(p)
	)
	if len(class_files) != 5:
		raise FileNotFoundError(
			f"Expected 5 class annual files with pattern {RC_ANNUAL_PATTERN!r}, found {len(class_files)}"
		)

	sim_df = pd.read_csv(SIMSTADT_DIN_PATH,sep=";", comment="#", dtype=str)
	if (
		SIMSTADT_GMLID_COL not in sim_df.columns
		or SIMSTADT_HEATING_COL not in sim_df.columns
		or SIMSTADT_COOLING_COL not in sim_df.columns
		or SIMSTADT_HEATED_AREA_COL not in sim_df.columns
	):
		raise ValueError(
			f"Missing SimStadt columns in {SIMSTADT_DIN_PATH.name}. "
			f"Expected '{SIMSTADT_GMLID_COL}', '{SIMSTADT_HEATING_COL}', "
			f"'{SIMSTADT_COOLING_COL}' and '{SIMSTADT_HEATED_AREA_COL}'."
		)

	# Remove units row.
	sim_df = sim_df[sim_df[SIMSTADT_GMLID_COL].astype(str).str.strip() != "[-]"].copy()
	sim_df["sim_id_raw"] = sim_df[SIMSTADT_GMLID_COL].astype(str).str.strip()
	sim_df["sim_id_norm"] = sim_df["sim_id_raw"].apply(_normalize_building_id)
	sim_df["sim_heating_kwh"] = _to_numeric_series(sim_df[SIMSTADT_HEATING_COL])
	sim_df["sim_cooling_kwh"] = _to_numeric_series(sim_df[SIMSTADT_COOLING_COL])
	sim_df["sim_heated_area_m2"] = _to_numeric_series(sim_df[SIMSTADT_HEATED_AREA_COL])
	sim_df["sim_specific_heating_kwh_m2a"] = np.where(
		sim_df["sim_heated_area_m2"] > 0,
		sim_df["sim_heating_kwh"] / sim_df["sim_heated_area_m2"],
		np.nan,
	)

	sim_ids = sim_df["sim_id_raw"].tolist()
	sim_row_by_raw: dict[str, dict[str, float | str]] = {}
	for _, row in sim_df.iterrows():
		raw_id = str(row["sim_id_raw"])
		sim_row_by_raw.setdefault(
			raw_id,
			{
				"sim_id_norm": str(row["sim_id_norm"]),
				"heating_kwh": float(row["sim_heating_kwh"]),
				"cooling_kwh": float(row["sim_cooling_kwh"]),
				"specific_heating_kwh_m2a": float(row["sim_specific_heating_kwh_m2a"])
				if pd.notna(row["sim_specific_heating_kwh_m2a"])
				else np.nan,
			},
		)

	class_entries: list[tuple[str, dict[str, dict[str, float]]]] = []
	common_sim_norm_ids: set[str] | None = None
	matched_pairs_per_class: list[dict[str, str]] = []

	for file_path in class_files:
		df = pd.read_csv(file_path)
		if (
			RC_BUILDING_ID_COL not in df.columns
			or RC_HEATING_COL not in df.columns
			or RC_COOLING_COL not in df.columns
			or RC_HEATED_AREA_COL not in df.columns
		):
			raise ValueError(
				f"Missing RC columns in {file_path.name}. "
				f"Expected '{RC_BUILDING_ID_COL}', '{RC_HEATING_COL}', '{RC_COOLING_COL}' and '{RC_HEATED_AREA_COL}'."
			)

		df["rc_id_raw"] = df[RC_BUILDING_ID_COL].astype(str).str.strip()
		df["rc_heating_kwh"] = _to_numeric_series(df[RC_HEATING_COL])
		df["rc_cooling_kwh"] = _to_numeric_series(df[RC_COOLING_COL])
		df["rc_heated_area_m2"] = _to_numeric_series(df[RC_HEATED_AREA_COL])
		df["rc_specific_heating_kwh_m2a"] = np.where(
			df["rc_heated_area_m2"] > 0,
			df["rc_heating_kwh"] / df["rc_heated_area_m2"],
			np.nan,
		)

		rc_ids = df["rc_id_raw"].tolist()
		matched_rows, _, _ = _match_building_ids(rc_ids, sim_ids)

		rc_row_by_raw: dict[str, dict[str, float]] = {}
		for _, row in df.iterrows():
			rc_row_by_raw.setdefault(
				str(row["rc_id_raw"]),
				{
					"heating_kwh": float(row["rc_heating_kwh"]),
					"cooling_kwh": float(row["rc_cooling_kwh"]),
					"specific_heating_kwh_m2a": float(row["rc_specific_heating_kwh_m2a"])
					if pd.notna(row["rc_specific_heating_kwh_m2a"])
					else np.nan,
				},
			)

		by_sim_norm: dict[str, dict[str, float]] = {}
		class_index = _class_index_from_filename(file_path)
		class_label = THERMAL_CLASS_LABELS.get(class_index, class_index)

		for match_row in matched_rows:
			rc_id = match_row["rc_building_id"]
			sim_id = match_row["simstadt_building_id"]
			rc_row = rc_row_by_raw.get(rc_id)
			sim_row = sim_row_by_raw.get(sim_id)
			if rc_row is None or sim_row is None:
				continue
			sim_norm = str(sim_row["sim_id_norm"])
			by_sim_norm[sim_norm] = {
				"rc_heating_kwh": float(rc_row["heating_kwh"]),
				"rc_cooling_kwh": float(rc_row["cooling_kwh"]),
				"rc_specific_heating_kwh_m2a": float(rc_row["specific_heating_kwh_m2a"]),
			}
			matched_pairs_per_class.append(
				{
					"thermal_class_index": class_index,
					"thermal_class": class_label,
					"rc_building_id": rc_id,
					"simstadt_building_id": sim_id,
					"simstadt_building_id_normalized": sim_norm,
					"match_type": match_row["match_type"],
				}
			)

		matched_sim_norm_ids = set(by_sim_norm.keys())
		if common_sim_norm_ids is None:
			common_sim_norm_ids = matched_sim_norm_ids
		else:
			common_sim_norm_ids &= matched_sim_norm_ids

		class_entries.append((class_label, by_sim_norm))

	if not common_sim_norm_ids:
		raise ValueError(
			"No common matched buildings found across all RC classes and SimStadt after 2-pass ID matching."
		)

	common_ids_sorted = sorted(common_sim_norm_ids)

	sim_by_norm: dict[str, dict[str, float]] = {}
	for row in sim_row_by_raw.values():
		sim_norm = str(row["sim_id_norm"])
		sim_by_norm.setdefault(
			sim_norm,
			{
				"heating_kwh": float(row["heating_kwh"]),
				"cooling_kwh": float(row["cooling_kwh"]),
				"specific_heating_kwh_m2a": float(row["specific_heating_kwh_m2a"]),
			},
		)

	labels: list[str] = []
	rc_heat_mwh: list[float] = []
	rc_cool_mwh: list[float] = []
	rc_specific_rows: list[tuple[str, np.ndarray]] = []

	for class_label, by_sim_norm in class_entries:

		heat_sum_kwh = sum(
			float(by_sim_norm[sim_id]["rc_heating_kwh"])
			for sim_id in common_ids_sorted
			if sim_id in by_sim_norm
		)
		cool_sum_kwh = sum(
			float(by_sim_norm[sim_id]["rc_cooling_kwh"])
			for sim_id in common_ids_sorted
			if sim_id in by_sim_norm
		)
		specific_values = np.array(
			[
				float(by_sim_norm[sim_id]["rc_specific_heating_kwh_m2a"])
				for sim_id in common_ids_sorted
				if sim_id in by_sim_norm
			],
			dtype=float,
		)

		labels.append(class_label)
		rc_heat_mwh.append(heat_sum_kwh / 1000.0)
		rc_cool_mwh.append(cool_sum_kwh / 1000.0)
		rc_specific_rows.append((f"RC {class_label}", specific_values))

	sim_heat_mwh = sum(
		float(sim_by_norm[sim_id]["heating_kwh"])
		for sim_id in common_ids_sorted
		if sim_id in sim_by_norm
	) / 1000.0
	sim_cool_mwh = sum(
		float(sim_by_norm[sim_id]["cooling_kwh"])
		for sim_id in common_ids_sorted
		if sim_id in sim_by_norm
	) / 1000.0
	sim_specific_values = np.array(
		[
			float(sim_by_norm[sim_id]["specific_heating_kwh_m2a"])
			for sim_id in common_ids_sorted
			if sim_id in sim_by_norm
		],
		dtype=float,
	)

	return {
		"common_count": len(common_ids_sorted),
		"common_sim_norm_ids": common_ids_sorted,
		"matched_pairs_per_class": matched_pairs_per_class,
		"labels": labels,
		"rc_heat_mwh": rc_heat_mwh,
		"rc_cool_mwh": rc_cool_mwh,
		"sim_heat_mwh": sim_heat_mwh,
		"sim_cool_mwh": sim_cool_mwh,
		"sim_specific_values": sim_specific_values,
		"rc_specific_rows": rc_specific_rows,
	}


def export_matching_reports(matched_dataset: dict[str, Any]) -> tuple[Path, Path]:
	REPORT_DIR.mkdir(parents=True, exist_ok=True)

	pairs_rows = list(matched_dataset.get("matched_pairs_per_class", []))
	common_ids = set(matched_dataset.get("common_sim_norm_ids", []))

	for row in pairs_rows:
		row["in_common_cohort"] = row.get("simstadt_building_id_normalized") in common_ids

	pairs_path = REPORT_DIR / "matched_id_pairs_per_class.csv"
	pd.DataFrame(pairs_rows).to_csv(pairs_path, index=False)

	common_path = REPORT_DIR / "common_matched_cohort_ids.csv"
	pd.DataFrame(
		{
			"simstadt_building_id_normalized": sorted(common_ids),
		}
	).to_csv(common_path, index=False)

	return pairs_path, common_path


def add_value_labels(ax: plt.Axes, bars) -> None:
	for bar in bars:
		height = bar.get_height()
		ax.annotate(
			f"{height:.0f}",
			xy=(bar.get_x() + bar.get_width() / 2, height),
			xytext=(0, 4),
			textcoords="offset points",
			ha="center",
			va="bottom",
			fontsize=8,
		)


def plot_thermal_class_comparison(matched_dataset: dict[str, Any]) -> Path:
	labels = list(matched_dataset["labels"])
	rc_heat_mwh = list(matched_dataset["rc_heat_mwh"])
	rc_cool_mwh = list(matched_dataset["rc_cool_mwh"])
	sim_heat_mwh = float(matched_dataset["sim_heat_mwh"])
	sim_cool_mwh = float(matched_dataset["sim_cool_mwh"])

	labels = labels + ["SimStadt"]
	heat_values = rc_heat_mwh + [sim_heat_mwh]
	cool_values = rc_cool_mwh + [sim_cool_mwh]

	x = np.arange(len(labels))
	width = 0.38

	fig, ax = plt.subplots(figsize=(11, 6))
	bars_heat = ax.bar(
		x - width / 2,
		heat_values,
		width,
		label="Heizbedarf",
		color=RC_HEATING_COLOR,
	)
	bars_cool = ax.bar(
		x + width / 2,
		cool_values,
		width,
		label="Kühlbedarf",
		color=RC_COOLING_COLOR,
	)

	add_value_labels(ax, bars_heat)
	add_value_labels(ax, bars_cool)
	ax.set_ylim(0, 7300)
	ax.set_title("Jährlicher Heiz- und Kühlbedarf nach thermischer Klasse")
	ax.set_xlabel("Thermische klasse")
	ax.set_ylabel("Energiebedarf in MWh/a")
	ax.set_xticks(x)
	ax.set_xticklabels(labels)
	ax.grid(axis="y", linestyle="--", alpha=0.4)
	ax.legend()

	plt.tight_layout()
	PLOT_DIR.mkdir(parents=True, exist_ok=True)
	output_path = PLOT_DIR / "annual_heating_cooling_thermal_classes_vs_simstadt.png"
	fig.savefig(str(output_path), dpi=200)
	plt.close(fig)
	return output_path


def plot_specific_heating_all_rc_variants(matched_dataset: dict[str, Any]) -> Path:
	sim_values = np.asarray(matched_dataset["sim_specific_values"], dtype=float)
	rc_values_rows = list(matched_dataset["rc_specific_rows"])
	if len(sim_values) == 0:
		raise ValueError("No matched SimStadt specific heating values are available.")
	if not rc_values_rows:
		raise ValueError("No matched RC class-specific heating values are available.")

	row_entries: list[tuple[str, np.ndarray, str]] = [("SimStadt", sim_values, "#C65D3B")]
	rc_colors = ["#9ecae1", "#6baed6", "#3182bd", "#2171b5", "#08519c"]
	for idx, (label, values) in enumerate(rc_values_rows):
		row_entries.append((label, values, rc_colors[idx % len(rc_colors)]))

	filtered_rows: list[np.ndarray] = []
	for _, values, _ in row_entries:
		valid = values[np.isfinite(values)]
		valid = valid[(valid >= 1.0) & (valid <= 600.0)]
		filtered_rows.append(valid)

	if not any(len(values) > 0 for values in filtered_rows):
		raise ValueError("No values remained for the scatter plot after filtering to 1..600 kWh/m2a.")

	rng = np.random.default_rng(42)
	fig, ax = plt.subplots(figsize=(14, 7))
	for row_idx, ((label, _values, color), values) in enumerate(zip(row_entries, filtered_rows)):
		if len(values) == 0:
			continue
		y = np.full(len(values), row_idx, dtype=float) + rng.normal(0.0, 0.03, len(values))
		ax.scatter(values, y, s=24, alpha=0.75, color=color, label=label)

	ax.set_xlim(1, 600)
	ax.set_ylim(-0.35, len(row_entries) - 1 + 0.35)
	ax.set_yticks(np.arange(len(row_entries), dtype=float))
	ax.set_yticklabels([label for label, _, _ in row_entries])
	ax.set_xlabel("Specific heating demand [kWh/m2a]")
	ax.set_ylabel("Model variant")
	ax.set_title("Specific heating demand with all RC thermal mass variants")
	ax.grid(axis="x", alpha=0.25)
	ax.legend(loc="upper right")

	plt.tight_layout()
	PLOT_DIR.mkdir(parents=True, exist_ok=True)
	output_path = PLOT_DIR / "specific_heating_demand_scatter_all_rc_variants.png"
	fig.savefig(str(output_path), dpi=200)
	plt.close(fig)
	return output_path


def main() -> None:
	matched_dataset = build_matched_comparison_dataset()
	pairs_report_path, common_report_path = export_matching_reports(matched_dataset)
	bar_plot_path = plot_thermal_class_comparison(matched_dataset)
	variants_plot_path = plot_specific_heating_all_rc_variants(matched_dataset)
	print(f"Matched buildings used in both plots: {matched_dataset['common_count']}")
	print(f"Saved report: {pairs_report_path}")
	print(f"Saved report: {common_report_path}")
	print(f"Saved plot: {bar_plot_path}")
	print(f"Saved plot: {variants_plot_path}")


if __name__ == "__main__":
	main()
