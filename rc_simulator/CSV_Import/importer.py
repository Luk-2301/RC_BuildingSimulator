import os
import logging
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Set root folder one level up, just for this example
mainPath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, mainPath)

import numpy as np
import pandas as pd
from building_physics import Zone #Importing Zone Class
# import supply_system
import emission_system
from radiation import Location
from radiation import Window 


# run all thermal mass classes with post-processing: python rc_simulator/CSV_Import/importer.py --all-thermal-classes
# run all thermal mass classes without post-processing: python rc_simulator/CSV_Import/importer.py --all-thermal-classes --skip-postprocess
# --refurbishment-mode

# Centralized data directory (data is stored in the local "data" folder next to this file)
DATA_DIR = Path(__file__).resolve().parent / "data"

#BUILDING_DATA_PATH = DATA_DIR / "buildingData.csv"
BUILDING_DATA_PATH = DATA_DIR / "Physics Preprocessor Surface export Base SimStadt export.csv" # U-values for surfaces. (addet Physics Preprocessor to filename for clarification and to avoid files with the same name.)
#BUILDING_GEOMETRY_PATH = DATA_DIR / "buildingGeometry.csv"
BUILDING_GEOMETRY_PATH = DATA_DIR / "Geometric Preprocessor Surface export Base SimStadt export.csv" # Surface orientation data. (addet Geometric Preprocessor to filename for clarification and to avoid files with the same name.)
BUILDING_DIN_PATH = DATA_DIR / "ImBuchwald_DIN18599_HEATING_AND_COOLING.csv"
WEATHER_DATA_PATH = DATA_DIR / "weatherData.prn"
VENTILATION_EXPORT_PATH = DATA_DIR / "Buchwald_Usage_export.csv"
REFURBISHMENT_SURFACE_EXPORT_PATH = DATA_DIR / "Refurbishment Surface export Base SimStadt export.csv" # U-values for opaque surfaces in refurbishment mode, matched by Surface ID. (addet Refurbishment to filename for clarification and to avoid files with the same name.)

logger = logging.getLogger(__name__)

# file readers (use centralized paths)
building_data = pd.read_csv(BUILDING_DATA_PATH, sep=';')  # needed for defining envelope area in m² and U-values
building_geometry = pd.read_csv(BUILDING_GEOMETRY_PATH, sep=';')  # needed for defining orientation of surfaces
building_din = pd.read_csv(BUILDING_DIN_PATH, sep=';', comment='#')  # needed for building type, occupancy, heated area, volume...
_ventilation_df = pd.read_csv(VENTILATION_EXPORT_PATH, sep=';')


weather_data = pd.read_csv(WEATHER_DATA_PATH, sep=r'\s+', header=None, names=['total_radiation', 'diffuse_radiation', 'ambient_temperature'])  # weather file from SimStadt
weather_data['total_radiation'] = pd.to_numeric(weather_data['total_radiation'], errors='coerce')
weather_data['diffuse_radiation'] = pd.to_numeric(weather_data['diffuse_radiation'], errors='coerce')
weather_data['ambient_temperature'] = pd.to_numeric(weather_data['ambient_temperature'], errors='coerce')
weather_data['direct_radiation'] = np.maximum(weather_data['total_radiation'] - weather_data['diffuse_radiation'], 0.0)
#weather_data = pd.read_csv('data/weatherData.csv')     
location = Location(weather_file_path=str(WEATHER_DATA_PATH))

RC_HOURLY_DIR = DATA_DIR / "Rc_Hourly_Results"
RC_HOURLY_AGGREGATED_DIR = DATA_DIR / "RC_Hourly_Aggregated_Results"
RC_HOURLY_ACCUMULATED_PATH = RC_HOURLY_AGGREGATED_DIR / "rc_hourly_accumulated.csv"
SIM_ANNUAL_SUMMARY_PATH = RC_HOURLY_AGGREGATED_DIR / "simulated_annual_results.csv"
SIM_INPUT_SNAPSHOT_PATH = RC_HOURLY_AGGREGATED_DIR / "simulated_input_snapshot.csv"
THERMAL_CLASS_MANIFEST_PATH = RC_HOURLY_AGGREGATED_DIR / "thermal_class_manifest.csv"

THERMAL_CLASS_DEFINITIONS = [
    {'key': 'very_light', 'label': 'very light', 'index': 1, 'capacitance': 80000},
    {'key': 'light', 'label': 'light', 'index': 2, 'capacitance': 110000},
    {'key': 'medium', 'label': 'medium', 'index': 3, 'capacitance': 165000},
    {'key': 'heavy', 'label': 'heavy', 'index': 4, 'capacitance': 260000},
    {'key': 'very_heavy', 'label': 'very heavy', 'index': 5, 'capacitance': 370000},
]
THERMAL_CLASS_BY_KEY = {entry['key']: entry for entry in THERMAL_CLASS_DEFINITIONS}
THERMAL_CLASS_ALIASES = {
    'very light': 'very_light',
    'very-light': 'very_light',
    'very_light': 'very_light',
    'light': 'light',
    'medium': 'medium',
    'heavy': 'heavy',
    'very heavy': 'very_heavy',
    'very-heavy': 'very_heavy',
    'very_heavy': 'very_heavy',
}


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the RC simulation and optionally trigger post-processing."
    )
    thermal_group = parser.add_mutually_exclusive_group()
    thermal_group.add_argument(
        "--thermal-class",
        dest="thermal_class",
        help="Run one thermal class only (very_light, light, medium, heavy, very_heavy).",
    )
    thermal_group.add_argument(
        "--all-thermal-classes",
        action="store_true",
        help="Run the simulation once for each thermal class and write separate outputs.",
    )
    parser.add_argument(
        "--skip-postprocess",
        action="store_true",
        help="Do not run simulation_comparison.py and single_building_comp.py after the simulation finishes.",
    )
    parser.add_argument(
        "--refurbishment-mode",
        action="store_true",
        help="Load opaque-surface U-values from the SimStadt surface export and use a fixed window U-value of 0.91.",
    )
    return parser.parse_args()



# Manual internal gains configuration [W/m²] by usage type based on the usage libary from SimStadt.
# Add aditional building types and their corresponding internal gains here as needed. 
INTERNAL_GAINS_W_M2_BY_USAGE = {
    #'community hall': '2.6',
    #'company building': '9.9',
    #'energy supply building': '9.9',
    'residential': '4.2',
   # 'restaurant': '13.0',
   # 'retail': '6.5',
}

DEFAULT_INTERNAL_GAINS_W_M2 = 4.2   # in W/m², fallback value if usage type is unknown.

def _normalize_usage_type(value):
    return str(value).strip().lower()

def _parse_internal_gain_value(value, usage_type):
    value_str = str(value).strip()
    if value_str.upper() == 'XX' or value_str == '':
        return DEFAULT_INTERNAL_GAINS_W_M2
    try:
        return float(value_str.replace(',', '.'))
    except ValueError:
        return DEFAULT_INTERNAL_GAINS_W_M2


# Build a cleaned lookup dict: {Building Parent ID: ach_value}
VENTILATION_RATES_BY_BUILDING = {}
if not _ventilation_df.empty and 'Building Parent ID' in _ventilation_df.columns and 'Air change rate [vol/h]' in _ventilation_df.columns:
    df_v = _ventilation_df.copy()
    df_v['Building Parent ID'] = df_v['Building Parent ID'].astype(str).str.strip()
    df_v['Air change rate [vol/h]'] = pd.to_numeric(
        df_v['Air change rate [vol/h]'].astype(str).str.replace(',', '.', regex=False),
        errors='coerce',
    )
    df_v = df_v.loc[
        df_v['Building Parent ID'].ne('')
        & df_v['Building Parent ID'].str.lower().ne('nan')
        & df_v['Air change rate [vol/h]'].notna()
    ]
    # Keep first occurrence if duplicates exist
    VENTILATION_RATES_BY_BUILDING = (
        df_v.drop_duplicates(subset=['Building Parent ID'], keep='first')
        .set_index('Building Parent ID')['Air change rate [vol/h]']
        .astype(float)
        .to_dict()
    )


def get_ventilation_rate_ach(building_id):
    return VENTILATION_RATES_BY_BUILDING.get(str(building_id).strip(), 0.3)


def build_unique_export_building_ids(building_ids):
    export_ids = {}
    grouped_ids = {}

    for building_id in building_ids:
        building_id_str = str(building_id).strip()
        normalized_key = building_id_str.lower()
        grouped_ids.setdefault(normalized_key, []).append(building_id_str)

    for normalized_key in grouped_ids:
        grouped_ids[normalized_key].sort(
            key=lambda value: (
                value.lower(),
                sum(1 for character in value if character.isupper()),
                value,
            )
        )

    for normalized_key, ordered_ids in grouped_ids.items():
        for occurrence_count, building_id_str in enumerate(ordered_ids):
            if occurrence_count == 0:
                export_id = building_id_str
            else:
                export_id = f"{building_id_str}__{occurrence_count:02d}"
            export_ids[building_id_str] = export_id

    return export_ids


def _normalize_thermal_class_key(value):
    value_str = str(value).strip().lower().replace('-', '_').replace(' ', '_')
    return THERMAL_CLASS_ALIASES.get(value_str, value_str)


def _normalize_surface_id(value):
    return str(value).strip()


def _format_id_sample(values, max_items=10):
    ordered_values = list(values)
    sample = ordered_values[:max_items]
    suffix = ''
    if len(ordered_values) > max_items:
        suffix = f" ... (+{len(ordered_values) - max_items} more)"
    return ', '.join(sample) + suffix if sample else 'none'


def _load_refurbishment_surface_uvalues(model_surface_ids):
    if not REFURBISHMENT_SURFACE_EXPORT_PATH.exists():
        raise FileNotFoundError(
            f"Refurbishment mode requested, but surface export file not found: {REFURBISHMENT_SURFACE_EXPORT_PATH}"
        )

    refurbishment_df = pd.read_csv(REFURBISHMENT_SURFACE_EXPORT_PATH, sep=';')
    required_columns = {'Surface ID', 'U-Value opaque element'}
    missing_columns = required_columns.difference(refurbishment_df.columns)
    if missing_columns:
        raise ValueError(
            f"Refurbishment surface export is missing required column(s): {', '.join(sorted(missing_columns))}"
        )

    refurbishment_df = refurbishment_df.copy()
    refurbishment_df['Surface ID'] = refurbishment_df['Surface ID'].astype(str).str.strip()
    refurbishment_df['U-Value opaque element'] = pd.to_numeric(
        refurbishment_df['U-Value opaque element'].astype(str).str.replace(',', '.', regex=False),
        errors='coerce',
    )

    invalid_rows = refurbishment_df[
        refurbishment_df['Surface ID'].eq('')
        | refurbishment_df['Surface ID'].str.lower().eq('nan')
        | refurbishment_df['U-Value opaque element'].isna()
    ]
    if not invalid_rows.empty:
        logger.warning(
            "Refurbishment mode: ignoring %d invalid row(s) in %s with missing Surface ID or U-Value: %s",
            len(invalid_rows),
            REFURBISHMENT_SURFACE_EXPORT_PATH.name,
            _format_id_sample(invalid_rows['Surface ID'].astype(str).tolist()),
        )
        refurbishment_df = refurbishment_df.drop(index=invalid_rows.index)

    duplicate_mask = refurbishment_df['Surface ID'].duplicated(keep=False)
    duplicate_ids = sorted(refurbishment_df.loc[duplicate_mask, 'Surface ID'].dropna().unique().tolist())
    if duplicate_ids:
        logger.warning(
            "Refurbishment mode: duplicate Surface ID(s) found in %s. Keeping the first occurrence for: %s",
            REFURBISHMENT_SURFACE_EXPORT_PATH.name,
            _format_id_sample(duplicate_ids),
        )
        refurbishment_df = refurbishment_df.drop_duplicates(subset=['Surface ID'], keep='first')

    model_surface_id_set = {
        _normalize_surface_id(surface_id)
        for surface_id in model_surface_ids
        if _normalize_surface_id(surface_id) not in {'', 'nan'}
    }
    csv_surface_id_set = set(refurbishment_df['Surface ID'].tolist())
    missing_matches = sorted(csv_surface_id_set.difference(model_surface_id_set))
    if missing_matches:
        logger.warning(
            "Refurbishment mode: %d Surface ID(s) from %s do not exist in the model data and will be ignored: %s",
            len(missing_matches),
            REFURBISHMENT_SURFACE_EXPORT_PATH.name,
            _format_id_sample(missing_matches),
        )

    refurbishment_lookup = refurbishment_df.set_index('Surface ID')['U-Value opaque element'].astype(float).to_dict()
    return {
        'lookup': refurbishment_lookup,
        'csv_surface_ids': csv_surface_id_set,
    }


def _log_unmatched_opaque_surfaces(building_id, surface_type, missing_surface_ids):
    if not missing_surface_ids:
        return
    logger.warning(
        "Refurbishment mode: building %s has %d unmatched %s surface(s) without a CSV U-value: %s",
        building_id,
        len(missing_surface_ids),
        surface_type.lower(),
        _format_id_sample(sorted(missing_surface_ids)),
    )


def _apply_refurbishment_uvalues(surface_rows, surface_type, area_column, uvalue_column, refurbishment_lookup, default_uvalue=None):
    surface_rows = surface_rows.copy()
    if surface_rows.empty:
        return surface_rows

    matched_uvalues = surface_rows['Surface ID'].map(refurbishment_lookup)
    missing_mask = matched_uvalues.isna()
    _log_unmatched_opaque_surfaces(
        surface_rows['Building ID'].iloc[0],
        surface_type,
        surface_rows.loc[missing_mask, 'Surface ID'].astype(str).tolist(),
    )

    if default_uvalue is None:
        surface_rows[uvalue_column] = matched_uvalues.fillna(surface_rows[uvalue_column])
    else:
        surface_rows[uvalue_column] = matched_uvalues.fillna(default_uvalue)

    surface_rows['Area_x_UValue'] = surface_rows[area_column] * surface_rows[uvalue_column]
    return surface_rows


def get_thermal_class_info(thermal_class_key):
    normalized_key = _normalize_thermal_class_key(thermal_class_key)
    if normalized_key not in THERMAL_CLASS_BY_KEY:
        valid_classes = ', '.join(entry['key'] for entry in THERMAL_CLASS_DEFINITIONS)
        raise ValueError(f"Unknown thermal class '{thermal_class_key}'. Valid values: {valid_classes}")
    return THERMAL_CLASS_BY_KEY[normalized_key]


def get_selected_thermal_class_keys(args):
    if args.all_thermal_classes:
        return [entry['key'] for entry in THERMAL_CLASS_DEFINITIONS]

    selected_class = args.thermal_class if args.thermal_class else 'medium'
    return [get_thermal_class_info(selected_class)['key']]


def build_class_output_suffix(thermal_class_info, include_class_suffix):
    # Keep class suffixes always (also for single-class runs) so file naming
    # stays consistent across medium/light/heavy workflows.
    return f"_{thermal_class_info['index']:02d}_{thermal_class_info['key']}"


def build_class_specific_path(base_path, output_suffix):
    if not output_suffix:
        return base_path
    return base_path.with_name(f"{base_path.stem}{output_suffix}{base_path.suffix}")


def build_class_output_directory(thermal_class_info, include_class_suffix):
    if not include_class_suffix:
        # Keep single-class runs in a class-specific folder (e.g. 03_medium)
        # to avoid overwriting/colliding root-level RC hourly exports.
        return RC_HOURLY_DIR / f"{thermal_class_info['index']:02d}_{thermal_class_info['key']}"
    return RC_HOURLY_DIR / f"{thermal_class_info['index']:02d}_{thermal_class_info['key']}"

def get_internal_gain_w_m2(building_din_for_id):
    if building_din_for_id.empty or 'PrimaryUsageZoneType' not in building_din_for_id.columns:
        return DEFAULT_INTERNAL_GAINS_W_M2

    usage_raw = building_din_for_id['PrimaryUsageZoneType'].iloc[0]
    usage_type = _normalize_usage_type(usage_raw)

    if usage_type not in INTERNAL_GAINS_W_M2_BY_USAGE:
        return DEFAULT_INTERNAL_GAINS_W_M2

    configured_value = INTERNAL_GAINS_W_M2_BY_USAGE[usage_type]
    return _parse_internal_gain_value(configured_value, usage_type)



def get_envelope_uvalue_area_products(building_data_frame, building_id, refurbishment_mode=False, refurbishment_context=None):
    building_rows = building_data_frame[building_data_frame['Building ID'] == building_id].copy()

    def to_numeric(series):
        return pd.to_numeric(series.astype(str).str.replace(',', '.', regex=False), errors='coerce')

    wall_rows = building_rows[building_rows['Surface Type'] == 'WALL'].copy()
    roof_rows = building_rows[building_rows['Surface Type'] == 'ROOF'].copy()
    building_rows['Window area'] = to_numeric(building_rows['Window area'])
    window_rows = building_rows[building_rows['Window area'] > 0].copy()

    wall_rows['Wall area (without window)'] = to_numeric(wall_rows['Wall area (without window)'])
    wall_rows['Wall U-Value (without window)'] = to_numeric(wall_rows['Wall U-Value (without window)'])
    wall_rows['Area_x_UValue'] = wall_rows['Wall area (without window)'] * wall_rows['Wall U-Value (without window)']

    roof_rows['Wall area (without window)'] = to_numeric(roof_rows['Wall area (without window)'])
    roof_rows['Wall U-Value (without window)'] = to_numeric(roof_rows['Wall U-Value (without window)'])
    roof_rows['Area_x_UValue'] = roof_rows['Wall area (without window)'] * roof_rows['Wall U-Value (without window)']

    window_rows['Window area'] = to_numeric(window_rows['Window area'])
    window_rows['Window U-Value'] = to_numeric(window_rows['Window U-Value'])
    window_rows['Area_x_UValue'] = window_rows['Window area'] * window_rows['Window U-Value']

    if refurbishment_mode:
        refurbishment_lookup = refurbishment_context['lookup'] if refurbishment_context else {}
        wall_rows = _apply_refurbishment_uvalues(
            wall_rows,
            'WALL',
            'Wall area (without window)',
            'Wall U-Value (without window)',
            refurbishment_lookup,
        )
        roof_rows = _apply_refurbishment_uvalues(
            roof_rows,
            'ROOF',
            'Wall area (without window)',
            'Wall U-Value (without window)',
            refurbishment_lookup,
        )
        window_rows = window_rows.copy()
        window_rows['Window U-Value'] = 0.91    # based on the physics libary from SimStadt for Buildings renovated with Low-E triple-glazed window. Needs to be automated for renovation.
        window_rows['Area_x_UValue'] = window_rows['Window area'] * window_rows['Window U-Value']

    return {
        'walls': wall_rows[['Surface ID', 'Building ID', 'Wall area (without window)', 'Wall U-Value (without window)', 'Area_x_UValue']],
        'roofs': roof_rows[['Surface ID', 'Building ID', 'Wall area (without window)', 'Wall U-Value (without window)', 'Area_x_UValue']],
        'windows': window_rows[['Surface ID', 'Building ID', 'Window area', 'Window U-Value', 'Area_x_UValue']],
    }

#change the result file path to Rc_Hourly_aggregated_Results to avoid confusion with the per building hourly files.
def build_portfolio_hourly_accumulated(annual_summary_df, hourly_dir, output_suffix=''):
    hourly_pattern = f"rc_hourly_*{output_suffix}.csv" if output_suffix else "rc_hourly_*.csv"
    hourly_files = sorted(hourly_dir.glob(hourly_pattern))
    if not hourly_files:
        print(f"No per-building hourly files found in: {hourly_dir}", flush=True)
        return

    hourly_frames = []
    for hourly_file in hourly_files:
        df = pd.read_csv(hourly_file)
        if 'HOY' not in df.columns:
            continue
        hourly_frames.append(df)

    if not hourly_frames:
        print("No valid hourly files available to build accumulated portfolio file.", flush=True)
        return

    all_hourly = pd.concat(hourly_frames, ignore_index=True)
    all_hourly['HOY'] = pd.to_numeric(all_hourly['HOY'], errors='coerce')
    all_hourly = all_hourly.dropna(subset=['HOY'])
    all_hourly['HOY'] = all_hourly['HOY'].astype(int)

    numeric_cols = [
        'HeatingDemand', 'HeatingEnergy', 'CoolingDemand', 'CoolingEnergy',
        'IndoorAir', 'OutsideTemp', 'SolarGains', 'COP',
    ]
    for col in numeric_cols:
        if col in all_hourly.columns:
            all_hourly[col] = pd.to_numeric(all_hourly[col], errors='coerce')

    aggregated = all_hourly.groupby('HOY', as_index=False).agg({
        'HeatingDemand': 'sum',
        'HeatingEnergy': 'sum',
        'CoolingDemand': 'sum',
        'CoolingEnergy': 'sum',
        'IndoorAir': 'mean',
        'OutsideTemp': 'mean',
        'SolarGains': 'sum',
        'COP': 'mean',
    })

    aggregated['HeatingDemand_kWh_h_sum'] = pd.to_numeric(aggregated['HeatingDemand'], errors='coerce') / 1000.0
    aggregated['CoolingDemand_kWh_h_sum'] = pd.to_numeric(aggregated['CoolingDemand'], errors='coerce') / 1000.0

    total_heated_area = pd.to_numeric(annual_summary_df['Heated area [m2]'], errors='coerce').sum()
    if pd.notna(total_heated_area) and total_heated_area > 0:
        aggregated['HeatingDemand_Wh_m2h_sum'] = np.maximum(
            pd.to_numeric(aggregated['HeatingDemand'], errors='coerce'), 0.0
        ) / total_heated_area
        aggregated['CoolingDemand_Wh_m2h_sum'] = np.maximum(
            -pd.to_numeric(aggregated['CoolingDemand'], errors='coerce'), 0.0
        ) / total_heated_area
    else:
        aggregated['HeatingDemand_Wh_m2h_sum'] = np.nan
        aggregated['CoolingDemand_Wh_m2h_sum'] = np.nan

    aggregated = aggregated.rename(columns={
        'HeatingDemand': 'HeatingDemand_sum',
        'HeatingEnergy': 'HeatingEnergy_sum',
        'CoolingDemand': 'CoolingDemand_sum',
        'CoolingEnergy': 'CoolingEnergy_sum',
        'IndoorAir': 'IndoorAir_mean',
        'OutsideTemp': 'OutsideTemp_mean',
        'SolarGains': 'SolarGains_sum',
        'COP': 'COP_mean',
    })

    aggregated['Portfolio ID'] = 'ALL_BUILDINGS'
    aggregated = aggregated[[
        'HOY',
        'HeatingDemand_sum',
        'HeatingEnergy_sum',
        'CoolingDemand_sum',
        'CoolingEnergy_sum',
        'IndoorAir_mean',
        'OutsideTemp_mean',
        'SolarGains_sum',
        'COP_mean',
        'HeatingDemand_kWh_h_sum',
        'CoolingDemand_kWh_h_sum',
        'HeatingDemand_Wh_m2h_sum',
        'CoolingDemand_Wh_m2h_sum',
        'Portfolio ID',
    ]]

    accumulated_path = build_class_specific_path(RC_HOURLY_ACCUMULATED_PATH, output_suffix)
    aggregated.to_csv(accumulated_path, index=False)
    print(f"Saved accumulated portfolio hourly file to: {accumulated_path}", flush=True)


def simulate_single_building(task):
    building_index, total_buildings, building_id, building_export_id, thermal_class_info, output_suffix, refurbishment_mode, refurbishment_context = task
    building_start = time.perf_counter()

    # Matching rows from the data files for the current building.
    # Use building_geometry_for_id and building_din_for_id later in the loop.
    building = building_data[building_data['Building ID'] == building_id]
    building_geometry_for_id = building_geometry[building_geometry['Building Parent ID'] == building_id]
    building_din_for_id = building_din[building_din['GMLId'] == building_id]
    envelope_data = get_envelope_uvalue_area_products(
        building_data,
        building_id,
        refurbishment_mode=refurbishment_mode,
        refurbishment_context=refurbishment_context,
    )
    wall_data = envelope_data['walls']
    roof_data = envelope_data['roofs']
    window_data = envelope_data['windows']

    #Data out of the Din18599 file form SimStadt.
    latitude_deg = 48.77
    longitude_deg = 9.21
    usage_type = ''

    if not building_din_for_id.empty:
        if 'PrimaryUsageZoneType' in building_din_for_id.columns and not building_din_for_id['PrimaryUsageZoneType'].empty:
            usage_type = str(building_din_for_id['PrimaryUsageZoneType'].iloc[0]).strip()
        lat_raw = pd.to_numeric(
            building_din_for_id['Latitude'].astype(str).str.replace(',', '.', regex=False),
            errors='coerce'
        )
        lon_raw = pd.to_numeric(
            building_din_for_id['Longitude'].astype(str).str.replace(',', '.', regex=False),
            errors='coerce'
        )
        if not lat_raw.empty and pd.notna(lat_raw.iloc[0]):
            latitude_deg = float(lat_raw.iloc[0])
        if not lon_raw.empty and pd.notna(lon_raw.iloc[0]):
            longitude_deg = float(lon_raw.iloc[0])

    # Empty Lists for Storing Data to Plot
    ElectricityOut = []
    HeatingDemand = []  # Energy required by the zone
    HeatingEnergy = []  # Energy required by the supply system to provide HeatingDemand
    CoolingDemand = []  # Energy surplus of the zone
    CoolingEnergy = []  # Energy required by the supply system to get rid of CoolingDemand
    IndoorAir = []
    OutsideTemp = []
    SolarGains = []
    COP = []    #do we need this?
    sun_altitude_series = []
    sun_azimuth_series = []
    irradiation_series = []

    floors = building[building['Surface Type'] == 'GROUND']
    heated_area = pd.to_numeric(building_din_for_id['Heated area'].astype(str).str.replace(',', '.', regex=False), errors='coerce')
    heated_volume = pd.to_numeric(building_din_for_id['Heated volume'].astype(str).str.replace(',', '.', regex=False), errors='coerce')
    total_shared_wall_area = pd.to_numeric(building_din_for_id['Total shared wall area'].astype(str).str.replace(',', '.', regex=False), errors='coerce')
    floor_area_fallback = float(pd.to_numeric(floors['Wall area (without window)'], errors='coerce').sum())
    floor_area_raw = float(heated_area.iloc[0]) if not heated_area.empty and pd.notna(heated_area.iloc[0]) else floor_area_fallback
    floor_area_value = floor_area_raw if floor_area_raw > 0 else floor_area_fallback
    room_volume_raw = float(heated_volume.iloc[0]) if not heated_volume.empty and pd.notna(heated_volume.iloc[0]) else (floor_area_value * 2.5)
    room_volume_value = room_volume_raw if room_volume_raw > 0 else (floor_area_value * 2.5)
    external_wall_area_value = wall_data['Wall area (without window)'].sum()
    roof_area_value = roof_data['Wall area (without window)'].sum()
    wall_roof_area_sum = external_wall_area_value + roof_area_value
    window_area_sum = window_data['Window area'].sum()
    window_to_wall_ratio = window_area_sum / (external_wall_area_value + window_area_sum) if external_wall_area_value > 0 else np.nan
    ach_vent_value = get_ventilation_rate_ach(building_id)
    thermal_class_label = thermal_class_info['label']
    thermal_class_index = thermal_class_info['index']
    thermal_capacitance = thermal_class_info['capacitance']
    # total_internal_area includes shared wall area + floor area + external opaque area (walls + roof) + window area.
    total_shared_wall_area_value = (
        float(total_shared_wall_area.iloc[0])
        if not total_shared_wall_area.empty and pd.notna(total_shared_wall_area.iloc[0])
        else 0.0
    )
    total_internal_area_value = (
        total_shared_wall_area_value
        + floor_area_value
        + external_wall_area_value
        + wall_roof_area_sum
        + window_area_sum
    )
    wall_roof_uvalue_sum = wall_data['Area_x_UValue'].sum() + roof_data['Area_x_UValue'].sum()
    u_walls_value = wall_roof_uvalue_sum / wall_roof_area_sum if wall_roof_area_sum else 0.8
    window_uvalue_sum = window_data['Area_x_UValue'].sum()
    u_windows_value = window_uvalue_sum / window_area_sum if window_area_sum else 1.1

    av_storey_height = pd.to_numeric(building_din_for_id['Average Storey Height'].astype(str).str.replace(',', '.', regex=False), errors='coerce').iloc[0] if not building_din_for_id['Average Storey Height'].empty else 2.5

    # Initialise an instance of the Zone. Empty spaces take on the default
    # parameters. See ZonePhysics.py to see the default values
    rc_building = Zone(window_area = window_area_sum,
                       walls_area = wall_roof_area_sum,
                       floor_area = floor_area_value,
                       room_vol = room_volume_value,
                       total_internal_area = total_internal_area_value,
                       average_storey_height = av_storey_height,
                       u_walls = u_walls_value,
                       u_windows = u_windows_value,
                       ach_vent = ach_vent_value, # imported from Buchwald_Usage_export.csv by Building Parent ID
                       ach_infl = 0, # kept at 0 because the imported ACH value is used as the full ventilation input
                       ventilation_efficiency = 0, #not needet since no vent in SimStadt. (efficency = 0 = off)
                       thermal_capacitance_per_floor_area = thermal_capacitance, #in J/(K*m²) ISO_52016-1:2018-4 p.138, very light = 80'000, light = 110'000, medium = 165'000, heavy = 260'000, very heavy = 370'000.
                       #in SimStadt the thermal capacitcanc is defined by DIN 18599-2:2025-10 p. 90-91 as Cwirk in Wh/(m²*K): light = 50, medium = 90, heavy = 130.
                       t_set_heating_day = 20.0,
                       t_set_heating_night = 16.0,   #is there an output in simstadt that we can use to define this more dynamically?
                       t_set_cooling = 25.0,
                       summer_start_doy = 121,
                       summer_end_doy = 273,
                       max_cooling_energy_per_floor_area =- np.inf,
                       max_heating_energy_per_floor_area = np.inf)

    # Define one window object per surface with Window area > 0.
    # Orientation is read from building_geometry via matching Surface ID.
    window_entries = []
    for _, window_surface in window_data.iterrows():
        window_area = window_surface['Window area']
        if pd.isna(window_area) or window_area <= 0:
            continue

        geometry_match = building_geometry_for_id[building_geometry_for_id['Surface ID'] == window_surface['Surface ID']]
        if geometry_match.empty:
            altitude_tilt = 90.0
            azimut_tilt = 0.0
        else:
            inclination_values = pd.to_numeric(
                geometry_match['Inclination'].astype(str).str.replace(',', '.', regex=False),
                errors='coerce'
            )
            azimut_column = 'Azimut' if 'Azimut' in geometry_match.columns else 'Azimuth'
            azimut_values = pd.to_numeric(
                geometry_match[azimut_column].astype(str).str.replace(',', '.', regex=False),
                errors='coerce'
            )

            altitude_tilt = float(inclination_values.iloc[0]) if not inclination_values.empty and pd.notna(inclination_values.iloc[0]) else 90.0
            azimut_tilt = float(azimut_values.iloc[0]) if not azimut_values.empty and pd.notna(azimut_values.iloc[0]) else 0.0

        window_entries.append({
            'surface_id': window_surface['Surface ID'],
            'window': Window(
                azimuth_tilt=azimut_tilt,
                alititude_tilt=altitude_tilt,
                #glass_solar_transmittance=0.76, #based on the physics libary from SimStadt for Buildings built at 1980 like the buildings ImBuchenwald. Needs to be automated for renovation.
                glass_solar_transmittance=0.5, #based on the physics libary from SimStadt for Buildings renovated with Low-E triple-glazed window. Needs to be automated for renovation.
                area=float(window_area * 0.7),    # *0.7 since the Frame ration defined in SimStadt is 0,3 for Windows.
            ),
        })

    # A catch statement to prevent future coding bugs when modifying window area
    #if SouthWindow.area != rc_building.window_area:
        #raise ValueError('Window area defined in radiation file doesnt match area defined in zone')

    internal_gains_w_m2 = get_internal_gain_w_m2(building_din_for_id)
    t_m_prev = 20   #start temperature of the building. Could be automated in the future by using the first time step of the simulation to set it dynamically based on the initial conditions of the building in SimStadt.

    # Loop through all 8760 hours of the year.
    for hour in range(8760):
        #Print progress every 1000 hours
        internal_gains = internal_gains_w_m2 * floor_area_value
        t_out = weather_data.loc[hour, 'ambient_temperature']

        altitude, azimuth = location.calc_sun_position(
            latitude_deg=latitude_deg, longitude_deg=longitude_deg, year=2015, hoy=hour)

        if building_index == 1:
            sun_altitude_series.append(altitude)
            sun_azimuth_series.append(azimuth)
            irradiation_series.append(
                weather_data.loc[hour, 'total_radiation']
            )

        total_solar_gains = 0.0
        for window_entry in window_entries:
            current_window = window_entry['window']
            current_window.calc_solar_gains(
                sun_altitude=altitude,
                sun_azimuth=azimuth,
                normal_direct_radiation=weather_data.loc[hour, 'direct_radiation'],
                horizontal_diffuse_radiation=weather_data.loc[hour, 'diffuse_radiation']
            )
            total_solar_gains += current_window.solar_gains

        # SouthWindow.calc_illuminance(sun_altitude=Altitude, sun_azimuth=Azimuth,
        #                             normal_direct_illuminance=weather_data.loc[hour, 'direct_radiation'],   #we have no illuminace data in the weater file fro mSimStadt, so we use the radiation data as a proxy, assuming a constant luminous efficacy. This is a simplification and should be improved in the future.
        #                             horizontal_diffuse_illuminance=weather_data.loc[hour, 'diffuse_radiation'])

        rc_building.solve_energy(internal_gains=internal_gains,
                                 solar_gains=total_solar_gains,
                                 t_out=t_out,
                                 t_m_prev=t_m_prev,
                                 hour=hour)

        # rc_building.solve_lighting(
        #     illuminance=SouthWindow.transmitted_illuminance, occupancy=occupancy)

        # Set previous temperature for the next time step.
        t_m_prev = rc_building.t_m_next

        HeatingDemand.append(rc_building.heating_demand)
        HeatingEnergy.append(rc_building.heating_energy)
        CoolingDemand.append(rc_building.cooling_demand)
        CoolingEnergy.append(rc_building.cooling_energy)
        ElectricityOut.append(rc_building.electricity_out)
        IndoorAir.append(rc_building.t_air)
        OutsideTemp.append(t_out)
        SolarGains.append(total_solar_gains)
        COP.append(rc_building.cop)

    annual_results = pd.DataFrame({
        'HeatingDemand': HeatingDemand,
        'HeatingEnergy': HeatingEnergy,
        'CoolingDemand': CoolingDemand,
        'CoolingEnergy': CoolingEnergy,
        'IndoorAir': IndoorAir,
        'OutsideTemp': OutsideTemp,
        'SolarGains': SolarGains,
        'COP': COP
    })

    annual_heating_kwh = annual_results['HeatingDemand'].sum() / 1000.0
    annual_cooling_kwh = abs(annual_results['CoolingDemand'].sum()) / 1000.0
    annual_solar_kwh = annual_results['SolarGains'].sum() / 1000.0
    heated_area_m2 = floor_area_value
    annual_heating_kwh_m2a = annual_heating_kwh / heated_area_m2 if heated_area_m2 > 0 else np.nan
    annual_cooling_kwh_m2a = annual_cooling_kwh / heated_area_m2 if heated_area_m2 > 0 else np.nan
    solar_kwh_m2a = annual_solar_kwh / heated_area_m2 if heated_area_m2 > 0 else np.nan

    # Persist hourly RC results per building for later RC-vs-PRN comparison.
    hourly_out = annual_results.copy()
    hourly_out.insert(0, 'HOY', np.arange(1, len(hourly_out) + 1))
    hourly_out['HeatingDemand_kWh_h'] = pd.to_numeric(hourly_out['HeatingDemand'], errors='coerce') / 1000.0
    hourly_out['CoolingDemand_kWh_h'] = pd.to_numeric(hourly_out['CoolingDemand'], errors='coerce') / 1000.0
    if heated_area_m2 > 0:
        hourly_out['HeatingDemand_Wh_m2h'] = np.maximum(pd.to_numeric(hourly_out['HeatingDemand'], errors='coerce'), 0.0) / heated_area_m2
        hourly_out['CoolingDemand_Wh_m2h'] = np.maximum(-pd.to_numeric(hourly_out['CoolingDemand'], errors='coerce'), 0.0) / heated_area_m2
    else:
        hourly_out['HeatingDemand_Wh_m2h'] = np.nan
        hourly_out['CoolingDemand_Wh_m2h'] = np.nan
    hourly_out['Building ID'] = building_id
    hourly_out['Building ID [filename]'] = building_export_id
    hourly_out['Thermal class'] = thermal_class_label
    hourly_out['Thermal class index'] = thermal_class_index
    hourly_out['Thermal capacitance [J/m2K]'] = thermal_capacitance
    hourly_output_dir = build_class_output_directory(thermal_class_info, bool(output_suffix))
    hourly_output_dir.mkdir(parents=True, exist_ok=True)
    hourly_out_path = hourly_output_dir / f"rc_hourly_{building_export_id}{output_suffix}.csv"
    hourly_out.to_csv(hourly_out_path, index=False)

    building_elapsed = time.perf_counter() - building_start

    return {
        'building_index': building_index,
        'building_id': building_id,
        'total_buildings': total_buildings,
        'annual_summary_row': {
            'Building ID': building_id,
            'Building ID [filename]': building_export_id,
            'Thermal class': thermal_class_label,
            'Thermal class index': thermal_class_index,
            'Thermal capacitance [J/m2K]': thermal_capacitance,
            'Annual Solar Gains [kWh]': annual_solar_kwh,
            'Annual Heating Demand Sim [kWh]': annual_heating_kwh,
            'Annual Cooling Demand Sim [kWh]': annual_cooling_kwh,
            'Heated area [m2]': heated_area_m2,
            'Internal gains [W/m2]': internal_gains_w_m2,
            'Usage type': usage_type,
        },
        'annual_heating_kwh_m2a': annual_heating_kwh_m2a,
        'annual_cooling_kwh_m2a': annual_cooling_kwh_m2a,
        'solar_kwh_m2a': solar_kwh_m2a,
        'heated_area_m2': heated_area_m2,
        'window_area_sum': window_area_sum,
        'window_to_wall_ratio': window_to_wall_ratio,
        'u_walls_w_m2k': u_walls_value,
        'u_windows_w_m2k': u_windows_value,
        'ach_vent_value': ach_vent_value,
        'building_elapsed': building_elapsed,
        'hourly_out_path': str(hourly_out_path),
        'hourly_output_dir': str(hourly_output_dir),
        'input_snapshot_row': {
            'Building ID': building_id,
            'Building ID [filename]': building_export_id,
            'Thermal class': thermal_class_label,
            'Thermal class index': thermal_class_index,
            'Usage type': usage_type,
            'Latitude [deg]': latitude_deg,
            'Longitude [deg]': longitude_deg,
            'Heated area [m2]': heated_area_m2,
            'Heated volume [m3]': room_volume_value,
            'Window area [m2]': window_area_sum,
            'Window-to-Wall ratio [-]': window_to_wall_ratio,
            'U walls [W/m2K]': u_walls_value,
            'U windows [W/m2K]': u_windows_value,
            'Internal gains [W/m2]': internal_gains_w_m2,
            'ACH ventilation [1/h]': ach_vent_value,
            'ACH infiltration [1/h]': 0,
            'Ventilation efficiency [-]': 0,
            'Thermal capacitance [J/m2K]': thermal_capacitance,
            'Heating setpoint day [C]': 20.0,
            'Heating setpoint night [C]': 16.0,
            'Cooling setpoint [C]': 25.0,
            'Summer start day-of-year': 121,
            'Summer end day-of-year': 273,
            'Max heating energy per floor area [W/m2]': np.inf,
            'Max cooling energy per floor area [W/m2]': -np.inf,
        },
    }
    import subprocess
    #automatically initiate the graphical comparison and output.
    subprocess.run([sys.executable, str(Path(mainPath) / "CSV_Import" / "simulation_comparison.py")], check=True)
    subprocess.run([sys.executable, str(Path(mainPath) / "CSV_Import" / "single_building_comp.py")], check=True)

def run_thermal_class_simulation(thermal_class_key, skip_postprocess=False, include_class_suffix=False, refurbishment_mode=False, refurbishment_context=None):
    # Determine thermal class meta and output suffix early
    thermal_class_info = get_thermal_class_info(thermal_class_key)
    output_suffix = build_class_output_suffix(thermal_class_info, include_class_suffix)
    hourly_output_dir = build_class_output_directory(thermal_class_info, include_class_suffix)

    #filter for bukding types that should be includet in the heating demand clacuations. For now its based on ImBuchenwald. May need furter refinment.
    allowed_usage_types = set(INTERNAL_GAINS_W_M2_BY_USAGE.keys())
    allowed_building_ids = set(
        building_din[
            building_din['PrimaryUsageZoneType'].astype(str).str.strip().str.lower().isin(allowed_usage_types)
        ]['GMLId'].astype(str)
    )

    # Build export IDs mapping from all model building IDs so filename suffixes are reserved
    all_model_building_ids = [building_id for building_id, _ in building_data.groupby('Building ID')]
    building_export_ids = build_unique_export_building_ids(all_model_building_ids)

    # Filter building IDs for simulation by allowed usage types, heated-area threshold, and secondary usage zone type
    building_ids = [
        building_id
        for building_id, _ in building_data.groupby('Building ID')
        if building_id in allowed_building_ids
        and pd.to_numeric(
            building_din.loc[building_din['GMLId'] == building_id, 'Heated area']
            .astype(str)
            .str.replace(',', '.', regex=False),
            errors='coerce',
        ).max()
        >= 50   # Only include buildings with at least 50 m2 of heated area
        and building_din.loc[building_din['GMLId'] == building_id, 'SecondaryUsageZoneType']
            .astype(str).str.strip().str.lower().isin(['none']).any()  # Only include buildings with 'none' as secondary usage zone type
    ]
    total_buildings = len(building_ids)
    annual_summary_rows = []
    input_snapshot_rows = []
    run_start = time.perf_counter()
    print(f"Starting simulation for {total_buildings} buildings.", flush=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    hourly_output_dir.mkdir(parents=True, exist_ok=True)

    max_default_workers = max(1, min((os.cpu_count() or 1) - 1, 8))
    parallel_workers = min(max_default_workers, total_buildings) if total_buildings > 0 else 1
    print(
        f"Parallel mode for thermal class '{thermal_class_info['label']}' ({thermal_class_info['index']}): {parallel_workers} worker process(es)",
        flush=True,
    )

    tasks = [
        (
            idx,
            total_buildings,
            building_id,
            building_export_ids[building_id],
            thermal_class_info,
            output_suffix,
            refurbishment_mode,
            refurbishment_context,
        )
        for idx, building_id in enumerate(building_ids, start=1)
    ]

    results = []
    with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        future_to_task = {executor.submit(simulate_single_building, task): task for task in tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            result = future.result()
            results.append(result)
            print(
                f"[{result['building_index']}/{result['total_buildings']}] Building {result['building_id']}: "
                f"Heating demand={result['annual_heating_kwh_m2a']:.2f} kWh/m2a, "
                f"Cooling demand={result['annual_cooling_kwh_m2a']:.2f} kWh/m2a, "
                f"Heated area={result['heated_area_m2']:.2f} m2, "
                f"total window area={result['window_area_sum']:.2f} m2, "
                f"window-to-wall ratio={result['window_to_wall_ratio']:.3f}, "
                f"ACH ventilation={result['ach_vent_value']:.2f} 1/h, "
                f"Solar gains={result['solar_kwh_m2a']:.2f} kWh/m2a, "
                f"U walls={result['u_walls_w_m2k']:.2f} W/m2K, "
                f"U windows={result['u_windows_w_m2k']:.2f} W/m2K",
                flush=True,
            )
            print(
                f"[{result['building_index']}/{result['total_buildings']}] Building {result['building_id']}: "
                f"done in {result['building_elapsed']:.1f}s",
                flush=True,
            )

    results.sort(key=lambda item: item['building_index'])

    for result in results:
        annual_summary_rows.append(result['annual_summary_row'])
        input_snapshot_rows.append(result['input_snapshot_row'])

    run_elapsed = time.perf_counter() - run_start
    print(f"Simulation finished in {run_elapsed:.1f}s", flush=True)

    annual_summary_df = pd.DataFrame(annual_summary_rows)
    annual_summary_path = build_class_specific_path(SIM_ANNUAL_SUMMARY_PATH, output_suffix)
    annual_summary_df.to_csv(annual_summary_path, index=False)
    print(f"Saved annual summary to: {annual_summary_path}", flush=True)

    build_portfolio_hourly_accumulated(annual_summary_df, hourly_output_dir, output_suffix=output_suffix)

    input_snapshot_df = pd.DataFrame(input_snapshot_rows)
    input_snapshot_path = build_class_specific_path(SIM_INPUT_SNAPSHOT_PATH, output_suffix)
    input_snapshot_df.to_csv(input_snapshot_path, index=False)
    print(f"Saved input snapshot to: {input_snapshot_path}", flush=True)

    manifest_row = {
        'Thermal class index': thermal_class_info['index'],
        'Thermal class': thermal_class_info['label'],
        'Thermal capacitance [J/m2K]': thermal_class_info['capacitance'],
        'Output suffix': output_suffix,
        'Annual summary path': str(annual_summary_path),
        'Input snapshot path': str(input_snapshot_path),
        'Accumulated hourly path': str(build_class_specific_path(RC_HOURLY_ACCUMULATED_PATH, output_suffix)),
        'Hourly results directory': str(hourly_output_dir),
    }

    manifest_df = pd.DataFrame([manifest_row])
    manifest_df.to_csv(THERMAL_CLASS_MANIFEST_PATH, index=False)

    # Run post-processing only for medium, which is the default standard class
    # for general comparisons.
    if not skip_postprocess and thermal_class_info['key'] == 'medium':
        subprocess.run(
            [sys.executable, str(Path(mainPath) / "CSV_Import" / "simulation_comparison.py")],
            check=True,
        )
        subprocess.run(
            [sys.executable, str(Path(mainPath) / "CSV_Import" / "single_building_comp.py")],
            check=True,
        )
    elif not skip_postprocess:
        print(
            f"Skipping automatic post-processing for thermal class '{thermal_class_info['key']}' because general comparisons default to medium.",
            flush=True,
        )

    return manifest_row


def run_simulation(skip_postprocess=False, thermal_class_keys=None, refurbishment_mode=False):
    thermal_class_keys = thermal_class_keys if thermal_class_keys else ['medium']
    include_class_suffix = True
    manifest_rows = []
    refurbishment_context = None

    if refurbishment_mode:
        model_surface_ids = building_data['Surface ID'].dropna().astype(str).str.strip().tolist()
        refurbishment_context = _load_refurbishment_surface_uvalues(model_surface_ids)

    for thermal_class_key in thermal_class_keys:
        manifest_rows.append(
            run_thermal_class_simulation(
                thermal_class_key,
                skip_postprocess=skip_postprocess,
                include_class_suffix=include_class_suffix,
                refurbishment_mode=refurbishment_mode,
                refurbishment_context=refurbishment_context,
            )
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(THERMAL_CLASS_MANIFEST_PATH, index=False)
    print(f"Saved thermal class manifest to: {THERMAL_CLASS_MANIFEST_PATH}", flush=True)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = parse_args()
    run_simulation(
        skip_postprocess=args.skip_postprocess,
        thermal_class_keys=get_selected_thermal_class_keys(args),
        refurbishment_mode=args.refurbishment_mode,
    )

