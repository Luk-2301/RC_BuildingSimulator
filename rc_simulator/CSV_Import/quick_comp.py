from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

RC_HOURLY_DIR = DATA_DIR / "Rc_Hourly_Results" / "03_medium"
SIMSTADT_HOURLY_DIR = DATA_DIR / "SimStadt_Hourly_Heat_Demand"
PLOT_DIR = DATA_DIR / "comparison_plots"

RC_HOURLY_PREFIX = "rc_hourly_"
SIMSTADT_SUFFIX = "_hourly_demand.prn"

RC_COLUMN = "HeatingDemand_kWh_h"
RC_COOLING_COLUMN = "CoolingDemand_kWh_h"
SIM_COLUMN = "Heat Demand"


def normalize_id(value: str) -> str:
	return value.strip().lower()


def strip_variant_suffix(value: str) -> str:
	parts = value.split("__")
	if len(parts) >= 2 and parts[-1].isdigit():
		return "__".join(parts[:-1])
	return value


def strip_thermal_class_suffix(value: str) -> str:
	# Handles exporter suffixes like: _01_very_light ... _05_very_heavy
	return re.sub(r"_0[1-5]_(very_light|light|medium|heavy|very_heavy)$", "", str(value).strip())


def get_numeric_variant_suffix(value: str) -> int | None:
	parts = value.split("__")
	if len(parts) >= 2 and parts[-1].isdigit():
		return int(parts[-1])
	return None


def pick_suffix_match(rc_id: str, sim_ids: list[str]) -> str | None:
	rc_norm = normalize_id(strip_variant_suffix(rc_id))
	matches: list[tuple[int, str]] = []
	for sim_id in sim_ids:
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


def extract_rc_id(path: Path) -> str:
	name = path.stem
	if name.startswith(RC_HOURLY_PREFIX):
		name = name[len(RC_HOURLY_PREFIX) :]
	return strip_thermal_class_suffix(name)


def extract_sim_id(path: Path) -> str:
	name = path.name
	if name.endswith(SIMSTADT_SUFFIX):
		return name[: -len(SIMSTADT_SUFFIX)]
	return path.stem


def load_rc_series(path: Path) -> pd.Series:
	df = pd.read_csv(path)
	if RC_COLUMN not in df.columns:
		raise ValueError(f"Missing '{RC_COLUMN}' in {path.name}")
	return pd.to_numeric(df[RC_COLUMN], errors="coerce").fillna(0.0).astype(float)


def total_annual_rc_cooling_mwh() -> float:
	rc_files = sorted(RC_HOURLY_DIR.glob(f"{RC_HOURLY_PREFIX}*.csv"))
	if not rc_files:
		raise FileNotFoundError(f"No RC hourly files found in: {RC_HOURLY_DIR}")

	total_kwh = 0.0
	for rc_path in rc_files:
		df = pd.read_csv(rc_path)
		if RC_COOLING_COLUMN not in df.columns:
			continue
		cooling = pd.to_numeric(df[RC_COOLING_COLUMN], errors="coerce").fillna(0.0).astype(float)
		total_kwh += cooling.abs().sum()

	return total_kwh / 1000.0


def load_sim_series(path: Path) -> pd.Series:
	df = pd.read_csv(
		path,
		sep=r"\s+",
		comment="#",
		header=None,
		names=["HOY", "Heat Demand", "Load duration curve", "Dhw Demand"],
		engine="python",
	)
	if SIM_COLUMN not in df.columns:
		raise ValueError(f"Missing '{SIM_COLUMN}' in {path.name}")
	return pd.to_numeric(df[SIM_COLUMN], errors="coerce").fillna(0.0).astype(float)


def pick_first_10_matches() -> list[tuple[str, Path, str, Path]]:
	rc_files = sorted(RC_HOURLY_DIR.glob(f"{RC_HOURLY_PREFIX}*.csv"))
	sim_files = sorted(SIMSTADT_HOURLY_DIR.glob(f"*{SIMSTADT_SUFFIX}"))

	if not rc_files:
		raise FileNotFoundError(f"No RC hourly files found in: {RC_HOURLY_DIR}")
	if not sim_files:
		raise FileNotFoundError(f"No SimStadt hourly files found in: {SIMSTADT_HOURLY_DIR}")

	sim_by_exact: dict[str, Path] = {}
	sim_remaining_ids: list[str] = []
	sim_path_by_id: dict[str, Path] = {}
	for sim_path in sim_files:
		sim_id = extract_sim_id(sim_path)
		sim_by_exact[normalize_id(sim_id)] = sim_path
		sim_remaining_ids.append(sim_id)
		sim_path_by_id[sim_id] = sim_path

	matches: list[tuple[str, Path, str, Path]] = []
	used_sim_paths: set[Path] = set()

	for rc_path in rc_files:
		rc_id = extract_rc_id(rc_path)
		rc_norm = normalize_id(rc_id)

		sim_path = sim_by_exact.get(rc_norm)
		sim_id = extract_sim_id(sim_path) if sim_path is not None else ""

		if sim_path is not None and sim_path in used_sim_paths:
			sim_path = None
			sim_id = ""

		if sim_path is None:
			candidate_id = pick_suffix_match(rc_id, sim_remaining_ids)
			if candidate_id is not None:
				sim_id = candidate_id
				sim_path = sim_path_by_id.get(candidate_id)

		if sim_path is None:
			continue

		if sim_path in used_sim_paths:
			continue

		matches.append((rc_id, rc_path, sim_id, sim_path))
		used_sim_paths.add(sim_path)
		sim_remaining_ids = [candidate for candidate in sim_remaining_ids if candidate != sim_id]

		if len(matches) == 10:
			break

	if len(matches) < 10:
		raise RuntimeError(
			f"Found only {len(matches)} RC-to-SimStadt matches, expected at least 10."
		)

	return matches


def make_plot() -> Path:
	PLOT_DIR.mkdir(parents=True, exist_ok=True)
	matches = pick_first_10_matches()

	fig, axes = plt.subplots(5, 2, figsize=(18, 20), sharex=False, sharey=False)
	flat_axes = axes.flatten()
	colors = plt.cm.tab10(np.linspace(0, 1, 10))

	for idx, (rc_id, rc_path, sim_id, sim_path) in enumerate(matches):
		rc_series = load_rc_series(rc_path)
		sim_series = load_sim_series(sim_path)
		min_len = min(len(rc_series), len(sim_series))

		rc_aligned = rc_series.iloc[:min_len].reset_index(drop=True)
		sim_aligned = sim_series.iloc[:min_len].reset_index(drop=True)
		hours = np.arange(1, min_len + 1)

		ax = flat_axes[idx]
		color = colors[idx]
		ax.plot(hours, sim_aligned.values, color=color, linewidth=1.1, label="SimStadt [kWh/h]")
		ax.plot(
			hours,
			rc_aligned.values,
			color=color,
			linestyle="--",
			linewidth=1.1,
			alpha=0.85,
			label="RC [kWh/h]",
		)
		ax.set_title(f"{idx + 1}. {rc_id}")
		ax.set_xlabel("Hour of year")
		ax.set_ylabel("Heating demand [kWh/h]")
		ax.grid(alpha=0.25)
		ax.legend(loc="upper right", fontsize=8)

	fig.suptitle("First 10 RC-matched buildings: Hourly heating demand", fontsize=16)
	fig.tight_layout(rect=[0, 0, 1, 0.985])

	output_path = PLOT_DIR / "first_10_buildings_rc_vs_simstadt.png"
	fig.savefig(output_path, dpi=180)
	plt.close(fig)
	return output_path


def main() -> None:
	output_path = make_plot()
	cooling_mwh = total_annual_rc_cooling_mwh()
	print(f"Total annual RC cooling demand: {cooling_mwh:.3f} MWh/a")
	print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
	main()
