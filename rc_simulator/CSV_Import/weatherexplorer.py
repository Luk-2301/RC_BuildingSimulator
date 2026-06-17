# Basic Weather evaluation graphs.

from pathlib import Path
from datetime import datetime, timedelta
import calendar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Original file name: hourly_GHI_DHI_Ta_pvgis_SARAH3_2005_2023_N48_8__E9_2.prn

DATA_DIR = Path(__file__).resolve().parent / "data"
WEATHER_PATH = DATA_DIR / "weatherData.prn"
TEMP_PLOT_PATH = DATA_DIR / "weather_air_temperature_year.png"
SOLAR_MONTHLY_PLOT_PATH = DATA_DIR / "weather_monthly_accumulated_solar_irradiation.png"
SUMMARY_PATH = DATA_DIR / "weather_temperature_summary.txt"

# Synthetic calendar year used to map HOY to dates and month labels.
BASE_YEAR = 2015
HEATING_BASE_T_C = 15.0
COOLING_BASE_T_C = 25.0


def load_weather(path: Path) -> pd.DataFrame:
    # weatherData.prn format: total_radiation diffuse_radiation ambient_temperature
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["total_radiation", "diffuse_radiation", "ambient_temperature"],
    )
    df["total_radiation"] = pd.to_numeric(df["total_radiation"], errors="coerce")
    df["diffuse_radiation"] = pd.to_numeric(df["diffuse_radiation"], errors="coerce")
    df["ambient_temperature"] = pd.to_numeric(df["ambient_temperature"], errors="coerce")
    df["direct_radiation"] = np.maximum(df["total_radiation"] - df["diffuse_radiation"], 0.0)
    df = df.dropna(subset=["ambient_temperature"]).reset_index(drop=True)
    return df


def month_ticks(total_hours: int):
    start_of_year = datetime(BASE_YEAR, 1, 1, 0, 0, 0)
    ticks = []
    labels = []

    for month in range(1, 13):
        dt = datetime(BASE_YEAR, month, 1, 0, 0, 0)
        hour = int((dt - start_of_year).total_seconds() / 3600) + 1  # HOY starts at 1
        if hour <= total_hours:
            ticks.append(hour)
            labels.append(calendar.month_abbr[month])

    return ticks, labels


def compute_degree_days(temps_c: pd.Series):
    # HDD/CDD based on daily mean temperature and fixed base temperatures.
    n = len(temps_c)
    timestamps = pd.date_range(start=datetime(BASE_YEAR, 1, 1), periods=n, freq="h")

    daily_mean = (
        pd.DataFrame({"timestamp": timestamps, "temperature": pd.to_numeric(temps_c, errors="coerce")})
        .assign(day=timestamps.normalize())
        .groupby("day", as_index=False)["temperature"]
        .mean()["temperature"]
    )

    heizgradtage = (HEATING_BASE_T_C - daily_mean).clip(lower=0).sum()
    kuehlgradtage = (daily_mean - COOLING_BASE_T_C).clip(lower=0).sum()
    return heizgradtage, kuehlgradtage


def calculate_monthly_radiation_kwh(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    timestamps = pd.date_range(start=datetime(BASE_YEAR, 1, 1), periods=n, freq="h")

    direct = pd.to_numeric(df["direct_radiation"], errors="coerce").fillna(0.0)
    diffuse = pd.to_numeric(df["diffuse_radiation"], errors="coerce").fillna(0.0)

    monthly = (
        pd.DataFrame(
            {
                "month": timestamps.month,
                "direct_wh_m2h": direct.to_numpy(),
                "diffuse_wh_m2h": diffuse.to_numpy(),
            }
        )
        .groupby("month", as_index=True)
        .sum()
        .reindex(range(1, 13), fill_value=0.0)
    )

    # Assumes hourly values are Wh/m2 per hour; sum/1000 gives kWh/m2 per month.
    monthly["direct_kwh_month"] = monthly["direct_wh_m2h"] / 1000.0
    monthly["diffuse_kwh_month"] = monthly["diffuse_wh_m2h"] / 1000.0
    monthly["total_kwh_month"] = monthly["direct_kwh_month"] + monthly["diffuse_kwh_month"]

    monthly["direct_share_pct"] = np.where(
        monthly["total_kwh_month"] > 0,
        100.0 * monthly["direct_kwh_month"] / monthly["total_kwh_month"],
        0.0,
    )
    monthly["diffuse_share_pct"] = np.where(
        monthly["total_kwh_month"] > 0,
        100.0 * monthly["diffuse_kwh_month"] / monthly["total_kwh_month"],
        0.0,
    )

    return monthly


def create_temperature_plot(df: pd.DataFrame):
    n = len(df)
    temps = df["ambient_temperature"].to_numpy()
    timestamps = pd.date_range(start=datetime(BASE_YEAR, 1, 1), periods=n, freq="h")

    daily_stats = (
        pd.DataFrame({"timestamp": timestamps, "temperature": temps})
        .assign(day=timestamps.normalize())
        .groupby("day", as_index=False)
        .agg({"temperature": ["min", "mean", "max"]})
    )
    daily_stats.columns = ["day", "temp_min", "temp_mean", "temp_max"]
    daily_stats["day_of_year"] = daily_stats["day"].dt.dayofyear

    month_tick_days = []
    month_labels = []
    last_day = int(daily_stats["day_of_year"].max())
    for month in range(1, 13):
        month_start_day = datetime(BASE_YEAR, month, 1).timetuple().tm_yday
        if month_start_day <= last_day:
            month_tick_days.append(month_start_day)
            month_labels.append(calendar.month_abbr[month])

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(
        daily_stats["day_of_year"],
        daily_stats["temp_max"],
        color="red",
        linewidth=1.0,
        label="Tägliche Maximaltemperatur",
    )
    ax.plot(
        daily_stats["day_of_year"],
        daily_stats["temp_mean"],
        color="gray",
        linewidth=1.0,
        label="Tägliche Durchscnhittstemperatur",
    )
    ax.plot(
        daily_stats["day_of_year"],
        daily_stats["temp_min"],
        color="blue",
        linewidth=1.0,
        label="Tägliche Minimaltemperatur",
    )

    ax.set_xticks(month_tick_days)
    ax.set_xticklabels(month_labels)

    ax.set_xlim(1, last_day)
    ax.set_ylim(-15, 35)
    ax.set_yticks(np.arange(-15, 36, 5))

    ax.set_xlabel("Tag des Jahres")
    ax.set_ylabel("Luft temperatur [°C]")
    ax.set_title("Tägliche Lufttemperatur über das Jahr (Min, Mittel, Max)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(TEMP_PLOT_PATH, dpi=150)
    plt.close(fig)


def create_monthly_solar_plot(monthly: pd.DataFrame):
    x = np.arange(1, 13)
    month_labels = [calendar.month_abbr[m] for m in range(1, 13)]

    diffuse_vals = monthly["diffuse_kwh_month"].to_numpy()
    direct_vals = monthly["direct_kwh_month"].to_numpy()
    total_vals = monthly["total_kwh_month"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_diffuse = ax.bar(
        x,
        diffuse_vals,
        color="tab:orange",
        edgecolor="black",
        linewidth=0.6,
        label="Diffuse",
    )
    ax.bar(
        x,
        direct_vals,
        bottom=diffuse_vals,
        color="#f4d03f",
        edgecolor="black",
        linewidth=0.6,
        label="Direct",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(month_labels)
    ax.set_xlabel("Monat")
    ax.set_ylabel("Solare einstrahlung [kWh/Monat]")
    ax.set_title("Monatliche Solare einstrahlung (Direkt + Diffus)")
    ax.grid(axis="y", alpha=0.3)

    max_total = float(np.max(total_vals)) if len(total_vals) > 0 else 0.0
    label_offset = max(2.0, 0.02 * max_total)

    for idx, diffuse_bar in enumerate(bars_diffuse):
        x_center = diffuse_bar.get_x() + diffuse_bar.get_width() / 2.0
        total = total_vals[idx]
        direct_pct = monthly["direct_share_pct"].iloc[idx]
        diffuse_pct = monthly["diffuse_share_pct"].iloc[idx]

        ax.text(
            x_center,
            total + label_offset,
            f"{total:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
        if diffuse_vals[idx] > 0:
            ax.text(
                x_center,
                diffuse_vals[idx] / 2.0,
                f"{diffuse_pct:.0f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

        if direct_vals[idx] > 0:
            ax.text(
                x_center,
                diffuse_vals[idx] + direct_vals[idx] / 2.0,
                f"{direct_pct:.0f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(SOLAR_MONTHLY_PLOT_PATH, dpi=150)
    plt.close(fig)


def write_summary(df: pd.DataFrame, monthly: pd.DataFrame):
    temps = df["ambient_temperature"]
    n = len(df)

    start = datetime(BASE_YEAR, 1, 1, 0, 0, 0)
    time_index = pd.Series([start + timedelta(hours=i) for i in range(n)])
    timestamps = pd.date_range(start=datetime(BASE_YEAR, 1, 1), periods=n, freq="h")

    idx_min = int(temps.idxmin())
    idx_max = int(temps.idxmax())

    t_min = float(temps.iloc[idx_min])
    t_mean = float(temps.mean())
    t_max = float(temps.iloc[idx_max])

    hoy_min = idx_min + 1
    hoy_max = idx_max + 1

    dt_min = time_index.iloc[idx_min].strftime("%Y-%m-%d %H:%M")
    dt_max = time_index.iloc[idx_max].strftime("%Y-%m-%d %H:%M")

    heizgradtage, kuehlgradtage = compute_degree_days(temps)

    yearly_direct_kwh = float(monthly["direct_kwh_month"].sum())
    yearly_diffuse_kwh = float(monthly["diffuse_kwh_month"].sum())
    yearly_total_kwh = yearly_direct_kwh + yearly_diffuse_kwh

    lines = [
        "Weather temperature summary",
        f"Source file: {WEATHER_PATH}",
        "",
        f"Min temperature [°C]: {t_min:.2f}",
        f"Min occurs at HOY: {hoy_min}",
        f"Min occurs at date-time: {dt_min}",
        "",
        f"Mean temperature [°C]: {t_mean:.2f}",
        "",
        f"Max temperature [°C]: {t_max:.2f}",
        f"Max occurs at HOY: {hoy_max}",
        f"Max occurs at date-time: {dt_max}",
        "",
        "Degree-day values [K*d]:",
        f"Heizgradtage (Basis {HEATING_BASE_T_C:.1f} °C): {heizgradtage:.2f}",
        f"Kuehlgradtage (Basis {COOLING_BASE_T_C:.1f} °C): {kuehlgradtage:.2f}",
        "",
        "Monthly radiation summary [kWh/Monat]",
        "Month | Direct | Diffuse | Total | Direct[%] | Diffuse[%]",
    ]

    for month in range(1, 13):
        row = monthly.loc[month]
        lines.append(
            f"{calendar.month_abbr[month]:>3} | "
            f"{row['direct_kwh_month']:>7.1f} | "
            f"{row['diffuse_kwh_month']:>7.1f} | "
            f"{row['total_kwh_month']:>7.1f} | "
            f"{row['direct_share_pct']:>8.1f} | "
            f"{row['diffuse_share_pct']:>9.1f}"
        )

    lines.extend(
        [
            "",
            "Yearly radiation sums [kWh/Jahr]",
            f"Direct yearly sum: {yearly_direct_kwh:.1f}",
            f"Diffuse yearly sum: {yearly_diffuse_kwh:.1f}",
            f"Total yearly sum: {yearly_total_kwh:.1f}",
        ]
    )

     # Compute monthly temperature statistics
    monthly_temps = (
        pd.DataFrame({"timestamp": timestamps, "temperature": temps.to_numpy()})
        .assign(month=timestamps.month)
        .groupby("month", as_index=True)["temperature"]
        .agg(["min", "mean", "max"])
        .reindex(range(1, 13), fill_value=0.0)
    )

    lines.extend(
        [
            "",
            "Monthly temperature summary [°C]",
            "Month | Min | Mean | Max",
        ]
    )

    for month in range(1, 13):
        row = monthly_temps.loc[month]
        lines.append(
            f"{calendar.month_abbr[month]:>3} | "
            f"{row['min']:>6.1f} | "
            f"{row['mean']:>6.1f} | "
            f"{row['max']:>6.1f}"
        )


    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    if not WEATHER_PATH.exists():
        raise FileNotFoundError(f"Weather file not found: {WEATHER_PATH}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    weather = load_weather(WEATHER_PATH)
    monthly = calculate_monthly_radiation_kwh(weather)
    create_temperature_plot(weather)
    create_monthly_solar_plot(monthly)
    write_summary(weather, monthly)

    print(f"Saved weather temperature plot to: {TEMP_PLOT_PATH}")
    print(f"Saved monthly solar plot to: {SOLAR_MONTHLY_PLOT_PATH}")
    print(f"Saved weather summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()