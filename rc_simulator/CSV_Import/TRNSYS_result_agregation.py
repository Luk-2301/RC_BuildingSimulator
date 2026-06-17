"""
TRNSYS Result Aggregation and Converter

Parses .pr1/.pr2 TRNSYS result files, extracts heating/cooling/solar data,
performs unit conversion (W/m² → kWh), and produces aggregated annual/monthly summaries.

Author: Automated Code Generation
Date: May 2026
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np


class TRNSYSFileParser:
    """Parses TRNSYS .pr1/.pr2 result files."""

    # Month hour counts for aggregation (non-leap year)
    MONTH_HOURS = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
    MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    def __init__(self, delimiter: str = '\t', area_m2: float = 242.0, skip_line_3: bool = True):
        """
        Initialize parser.

        Args:
            delimiter: Column delimiter in files (default: tab)
            area_m2: Heated area in m² for W/m² conversion (default: 242)
            skip_line_3: If True, ignore line 3 (header info line) and start from line 4
        """
        self.delimiter = delimiter
        self.area_m2 = area_m2
        self.skip_line_3 = skip_line_3
        self.errors = []
        self.matched_pairs = []
        self.unmatched_columns = []

    def find_pr_file_pairs(self, trnsys_folder: Path) -> Dict[str, Tuple[Path, Path]]:
        """
        Find matching .pr1 and .pr2 file pairs by base name.

        Args:
            trnsys_folder: Path to folder containing .pr1/.pr2 files

        Returns:
            Dict mapping building base name to (pr1_path, pr2_path) tuple
        """
        trnsys_folder = Path(trnsys_folder)
        if not trnsys_folder.exists():
            raise FileNotFoundError(f"TRNSYS folder not found: {trnsys_folder}")

        pr_files = {
            'pr1': {},
            'pr2': {},
        }

        for ext in ['pr1', 'pr2']:
            for file_path in trnsys_folder.glob(f'*.{ext}'):
                base_name = file_path.stem  # filename without extension
                pr_files[ext][base_name] = file_path

        pairs = {}
        for base_name in pr_files['pr1'].keys():
            if base_name in pr_files['pr2']:
                pairs[base_name] = (pr_files['pr1'][base_name], pr_files['pr2'][base_name])
            else:
                self.unmatched_columns.append({
                    'building_base_name': base_name,
                    'reason': 'Missing .pr2 file',
                    'pr1_file': str(pr_files['pr1'][base_name]),
                    'pr2_file': None,
                })

        for base_name in pr_files['pr2'].keys():
            if base_name not in pr_files['pr1']:
                self.unmatched_columns.append({
                    'building_base_name': base_name,
                    'reason': 'Missing .pr1 file',
                    'pr1_file': None,
                    'pr2_file': str(pr_files['pr2'][base_name]),
                })

        return pairs

    def read_pr_file(self, file_path: Path) -> Tuple[List[str], List[str], pd.DataFrame]:
        """
        Read a .pr1 or .pr2 file.

        Args:
            file_path: Path to the .pr file

        Returns:
            Tuple of (column_names, units, hourly_data_df) where data has 8760 rows
        """
        file_path = Path(file_path)
        try:
            with open(file_path, 'r', encoding='latin1') as f:
                lines = f.readlines()

            if len(lines) < 4:
                raise ValueError(f"File has fewer than 4 lines: {len(lines)}")

            # Parse line 1: column names
            names_line = lines[0].strip()
            column_names = names_line.split(self.delimiter)
            column_names = [c.strip() for c in column_names]

            # Parse line 2: units
            units_line = lines[1].strip()
            units = units_line.split(self.delimiter)
            units = [u.strip() for u in units]

            # Line 3 is skipped if skip_line_3 is True (index 2)
            # Lines 4+ (index 3+) contain hourly data

            data_start_line = 3 if self.skip_line_3 else 2
            data_lines = lines[data_start_line:]

            # Parse hourly data
            hourly_data = []
            for i, line in enumerate(data_lines):
                if not line.strip():
                    continue
                parts = line.strip().split(self.delimiter)
                parts = [p.strip() for p in parts]
                try:
                    values = [float(p) for p in parts]
                    hourly_data.append(values)
                except ValueError as e:
                    self.errors.append({
                        'file': str(file_path),
                        'line_number': data_start_line + i + 1,
                        'error': f"Could not parse values: {line[:100]}",
                    })
                    continue

            if len(hourly_data) < 8760:
                raise ValueError(f"Expected 8760 data rows, got {len(hourly_data)}")

            # Truncate to 8760 rows (discard extra)
            hourly_data = hourly_data[:8760]

            df = pd.DataFrame(hourly_data, columns=column_names)
            return column_names, units, df

        except Exception as e:
            self.errors.append({
                'file': str(file_path),
                'error': str(e),
            })
            raise

    def extract_columns_case_insensitive(
        self,
        df: pd.DataFrame,
        target_columns: List[str],
    ) -> Dict[str, pd.Series]:
        """
        Extract columns from dataframe, matching case-insensitively.

        Args:
            df: Input dataframe
            target_columns: List of column names to extract (case-insensitive)

        Returns:
            Dict mapping requested column name to extracted Series
        """
        result = {}
        df_cols_lower = {col.lower(): col for col in df.columns}

        for target in target_columns:
            target_lower = target.lower()
            if target_lower in df_cols_lower:
                actual_col = df_cols_lower[target_lower]
                result[target] = df[actual_col]
            else:
                result[target] = None

        return result

    def convert_pr_files_to_combined(
        self,
        pr1_path: Path,
        pr2_path: Path,
        output_folder: Path,
        building_name: str,
    ) -> Optional[Dict]:
        """
        Convert a .pr1/.pr2 file pair into a single combined CSV.

        Args:
            pr1_path: Path to .pr1 file
            pr2_path: Path to .pr2 file
            output_folder: Output directory for combined CSV
            building_name: Base name of building (for output filename and logs)

        Returns:
            Dict with conversion metadata, or None if conversion failed
        """
        try:
            # Read both files
            pr1_names, pr1_units, pr1_df = self.read_pr_file(pr1_path)
            pr2_names, pr2_units, pr2_df = self.read_pr_file(pr2_path)

            # Extract required columns
            pr1_data = self.extract_columns_case_insensitive(pr1_df, ['Tzone_luft'])
            pr2_data = self.extract_columns_case_insensitive(pr2_df, ['Q_heiz', 'Q_kuehl', 'Q_soltr'])

            # Check for missing columns
            missing_cols = []
            if pr1_data['Tzone_luft'] is None:
                missing_cols.append('Tzone_luft (in .pr1)')
            for col in ['Q_heiz', 'Q_kuehl', 'Q_soltr']:
                if pr2_data[col] is None:
                    missing_cols.append(f'{col} (in .pr2)')

            if missing_cols:
                self.unmatched_columns.append({
                    'building_name': building_name,
                    'pr1_file': str(pr1_path),
                    'pr2_file': str(pr2_path),
                    'missing_columns': ', '.join(missing_cols),
                })
                self.errors.append({
                    'file_pair': f"{building_name}.pr1/.pr2",
                    'error': f"Missing required columns: {', '.join(missing_cols)}",
                })
                return None

            # Create combined dataframe
            combined_df = pd.DataFrame()
            combined_df['Tzone_luft'] = pr1_data['Tzone_luft']
            
            # Convert Q_* from W/m² to kWh (multiply by area and divide by 1000 per hour)
            # Note: hourly value in W/m² → kWh = value * area_m2 / 1000.0
            combined_df['Q_heiz'] = pr2_data['Q_heiz'] * self.area_m2 / 1000.0
            combined_df['Q_kuehl'] = pr2_data['Q_kuehl'] * self.area_m2 / 1000.0
            combined_df['Q_soltr'] = pr2_data['Q_soltr'] * self.area_m2 / 1000.0

            output_name = building_name.removesuffix('_woNT')

            # Create output CSV
            output_folder = Path(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)
            output_file = output_folder / f"{output_name}_TRNSYS_converted.csv"

            # Write header lines: line 1 = names, line 2 = units
            with open(output_file, 'w', encoding='utf-8') as f:
                # Line 1: column names
                f.write(','.join(['Tzone_luft', 'Q_heiz', 'Q_kuehl', 'Q_soltr']) + '\n')
                # Line 2: units
                f.write(','.join(['°C', 'kWh', 'kWh', 'kWh']) + '\n')

            # Append hourly data (lines 3+)
            combined_df.to_csv(output_file, mode='a', header=False, index=False)

            self.matched_pairs.append({
                'building_name': output_name,
                'pr1_file': str(pr1_path),
                'pr2_file': str(pr2_path),
                'output_file': str(output_file),
                'status': 'Success',
                'hours_processed': 8760,
            })

            return {
                'building_name': output_name,
                'output_file': output_file,
                'combined_df': combined_df,
                'pr1_units': pr1_units,
                'pr2_units': pr2_units,
            }

        except Exception as e:
            self.errors.append({
                'file_pair': f"{building_name}.pr1/.pr2",
                'error': str(e),
            })
            return None

    def process_all_pairs(
        self,
        trnsys_folder: Path,
        output_folder: Path,
    ) -> Dict[str, Dict]:
        """
        Process all .pr1/.pr2 pairs in the TRNSYS folder.

        Args:
            trnsys_folder: Path to folder containing .pr1/.pr2 files
            output_folder: Output directory for converted CSVs

        Returns:
            Dict mapping building names to conversion results
        """
        pairs = self.find_pr_file_pairs(trnsys_folder)
        results = {}

        for building_name, (pr1_path, pr2_path) in pairs.items():
            result = self.convert_pr_files_to_combined(
                pr1_path,
                pr2_path,
                output_folder,
                building_name,
            )
            if result:
                results[result['building_name']] = result

        return results


class TRNSYSAggregator:
    """Aggregates converted TRNSYS data into annual and monthly summaries."""

    MONTH_HOURS = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
    MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    def __init__(self):
        self.errors = []

    def compute_monthly_sums(self, series: pd.Series) -> List[float]:
        """
        Compute monthly aggregated sums from hourly series (8760 hours).

        Args:
            series: Hourly series with 8760 values

        Returns:
            List of 12 monthly sums
        """
        monthly_sums = []
        hour_index = 0
        for month_hours in self.MONTH_HOURS:
            month_sum = series.iloc[hour_index : hour_index + month_hours].sum()
            monthly_sums.append(month_sum)
            hour_index += month_hours
        return monthly_sums

    def aggregate_converted_files(self, converted_files: Dict[str, Dict]) -> pd.DataFrame:
        """
        Aggregate converted TRNSYS files into annual/monthly summary.

        Args:
            converted_files: Dict mapping building names to conversion results
                            (from TRNSYSFileParser.process_all_pairs)

        Returns:
            DataFrame with one row per building, columns for annual and monthly values
        """
        summary_rows = []

        for building_name, result in converted_files.items():
            try:
                df = result['combined_df']

                # Compute annual sums
                annual_heiz = df['Q_heiz'].sum()
                annual_kuehl = df['Q_kuehl'].sum()
                annual_soltr = df['Q_soltr'].sum()

                # Compute monthly sums
                monthly_heiz = self.compute_monthly_sums(df['Q_heiz'])
                monthly_kuehl = self.compute_monthly_sums(df['Q_kuehl'])
                monthly_soltr = self.compute_monthly_sums(df['Q_soltr'])

                # Build row
                row = {'BuildingID': building_name}

                # Annual and monthly Q_heiz
                row['Annual_Q_heiz_kWh'] = annual_heiz
                for i, month_name in enumerate(self.MONTH_NAMES):
                    row[f'{month_name}_Q_heiz_kWh'] = monthly_heiz[i]

                # Annual and monthly Q_kuehl
                row['Annual_Q_kuehl_kWh'] = annual_kuehl
                for i, month_name in enumerate(self.MONTH_NAMES):
                    row[f'{month_name}_Q_kuehl_kWh'] = monthly_kuehl[i]

                # Annual and monthly Q_soltr
                row['Annual_Q_soltr_kWh'] = annual_soltr
                for i, month_name in enumerate(self.MONTH_NAMES):
                    row[f'{month_name}_Q_soltr_kWh'] = monthly_soltr[i]

                summary_rows.append(row)

            except Exception as e:
                self.errors.append({
                    'building_name': building_name,
                    'error': str(e),
                })

        summary_df = pd.DataFrame(summary_rows)
        return summary_df


def process_trnsys_results(
    trnsys_folder: str,
    output_folder: str,
    delimiter: str = '\t',
    area_m2: float = 242.0,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Main entry point: process all TRNSYS .pr1/.pr2 files and produce summaries.

    Args:
        trnsys_folder: Path to folder containing .pr1/.pr2 files
        output_folder: Output directory for converted CSVs and summaries
        delimiter: Column delimiter in TRNSYS files (default: tab)
        area_m2: Heated area in m² for unit conversion (default: 242)
        verbose: If True, print progress messages

    Returns:
        Tuple of (summary_df, metadata_dict)
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[TRNSYS Parser] Starting processing...")
        print(f"  Input folder: {trnsys_folder}")
        print(f"  Output folder: {output_folder}")
        print(f"  Delimiter: {repr(delimiter)}")
        print(f"  Area for conversion: {area_m2} m²")

    # Parse and convert files
    parser = TRNSYSFileParser(delimiter=delimiter, area_m2=area_m2)
    converted_files = parser.process_all_pairs(trnsys_folder, output_folder)

    if verbose:
        print(f"[TRNSYS Parser] Processed {len(converted_files)} building pairs")
        if parser.errors:
            print(f"[TRNSYS Parser] Encountered {len(parser.errors)} errors during parsing")
        if parser.unmatched_columns:
            print(f"[TRNSYS Parser] Found {len(parser.unmatched_columns)} unmatched files/columns")

    # Save parser logs
    parser_df = pd.DataFrame(parser.matched_pairs)
    if not parser_df.empty:
        parser_df.to_csv(output_folder / 'matched_pairs.csv', index=False)
        if verbose:
            print(f"  ✓ Saved matched_pairs.csv")

    unmatched_df = pd.DataFrame(parser.unmatched_columns)
    if not unmatched_df.empty:
        unmatched_df.to_csv(output_folder / 'unmatched_columns.csv', index=False)
        if verbose:
            print(f"  ✓ Saved unmatched_columns.csv")

    errors_df = pd.DataFrame(parser.errors)
    if not errors_df.empty:
        errors_df.to_csv(output_folder / 'parsing_errors.csv', index=False)
        if verbose:
            print(f"  ✓ Saved parsing_errors.csv")

    # Aggregate annual/monthly summaries
    aggregator = TRNSYSAggregator()
    summary_df = aggregator.aggregate_converted_files(converted_files)

    if verbose:
        print(f"[TRNSYS Aggregator] Produced {len(summary_df)} building summaries")

    # Save summary
    summary_file = output_folder / 'TRNSYS_annual_monthly_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    if verbose:
        print(f"  ✓ Saved TRNSYS_annual_monthly_summary.csv")

    metadata = {
        'total_buildings': len(converted_files),
        'matched_pairs': len(parser.matched_pairs),
        'unmatched': len(parser.unmatched_columns),
        'errors': len(parser.errors),
        'output_folder': str(output_folder),
        'summary_file': str(summary_file),
    }

    if verbose:
        print(f"\n[TRNSYS Complete]")
        print(f"  Buildings processed: {metadata['total_buildings']}")
        print(f"  Unmatched file pairs: {metadata['unmatched']}")
        print(f"  Parsing errors: {metadata['errors']}")

    return summary_df, metadata


if __name__ == '__main__':
    # Example usage
    trnsys_folder = Path(__file__).parent / 'data' / 'TRNSYS'
    output_folder = Path(__file__).parent / 'data' / 'TRNSYS' / 'processed'

    if trnsys_folder.exists():
        summary_df, metadata = process_trnsys_results(
            str(trnsys_folder),
            str(output_folder),
            delimiter='\t',
            area_m2=242.0,
            verbose=True,
        )
        print("\nSummary DataFrame:")
        print(summary_df)
    else:
        print(f"TRNSYS folder not found: {trnsys_folder}")
