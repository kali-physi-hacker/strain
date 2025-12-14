#!/usr/bin/env python3
def float_with_comma(value):
    """
    Type function for argparse to handle float values with commas.
    """
    return float(value.replace(",", ""))
import os
import re
import argparse
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# ==========================
#  COLORFUL TERMINAL OUTPUT
# ==========================
class Console:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"

    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"

    @staticmethod
    def info(msg):
        print(f"{Console.CYAN}[INFO]{Console.RESET} {msg}")

    @staticmethod
    def success(msg):
        print(f"{Console.GREEN}[SUCCESS]{Console.RESET} {msg}")

    @staticmethod
    def warn(msg):
        print(f"{Console.YELLOW}[WARN]{Console.RESET} {msg}")

    @staticmethod
    def error(msg):
        print(f"{Console.RED}[ERROR]{Console.RESET} {msg}")

    @staticmethod
    def header(msg):
        line = "═" * (len(msg) + 8)
        print(f"\n{Console.MAGENTA}{Console.BOLD}╔{line}╗")
        print(f"║    {msg}    ║")
        print(f"╚{line}╝{Console.RESET}")


def banner():
    print(f"""
{Console.CYAN}{Console.BOLD}
   ╔══════════════════════════════════════════════════════╗
   ║                                                      ║
   ║         ███████╗████████╗██████╗  █████╗ ██╗███╗   ██╗║
   ║         ██╔════╝╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║║
   ║         ███████╗   ██║   ██████╔╝███████║██║██╔██╗ ██║║
   ║         ╚════██║   ██║   ██╔══██╗██╔══██║██║██║╚██╗██║║
   ║         ███████║   ██║   ██║  ██║██║  ██║██║██║ ╚████║║
   ║         ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝║
   ║                                                      ║
   ║        Scientific Stress–Strain Curve Plotter        ║
   ║              Version 1.0 · Engineering Lab Tool      ║
   ║                by Desmond Brown                      ║  
   ║                                                      ║
   ╚══════════════════════════════════════════════════════╝
{Console.RESET}
""")


# ==========================
#  ENGINEERING PLOT STYLE
# ==========================
def apply_engineering_style():
    mpl.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,

        "lines.linewidth": 2.2,

        "axes.grid": True,
        "grid.color": "#cccccc",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,

        "figure.figsize": (12, 8),
        "figure.dpi": 150,

        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.facecolor": "white",
        "legend.edgecolor": "#cccccc",

        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,
    })


# ==========================
#  SMALL UTILITIES
# ==========================
def extract_label_from_filename(name: str) -> str:
    """
    Extract a meaningful label from filename.
    Tries to find percentage in name, e.g. '15pct.xlsx' -> '15%'.
    Falls back to base filename.
    """
    base = os.path.splitext(name)[0]
    match = re.search(r"(\d+)\s*%?", base)
    if match:
        return match.group(1) + "%"
    return base


def detect_column(df: pd.DataFrame, keywords) -> str | None:
    """
    Return the FIRST column containing ANY of the keywords.
    """
    for col in df.columns:
        for key in keywords:
            if key.lower() in col.lower():
                return col
    return None


def progress_bar(current, total, prefix="Processing"):
    """
    Simple text progress bar for multiple-file mode.
    """
    bar_length = 30
    fraction = current / total if total > 0 else 1
    filled = int(bar_length * fraction)
    bar = "█" * filled + "-" * (bar_length - filled)
    percent = int(fraction * 100)
    print(f"\r{Console.BLUE}{prefix}{Console.RESET} |{bar}| {percent:3d}%", end="")
    if current == total:
        print()  # newline at end


# ==========================
#  MECHANICAL PROPERTIES
# ==========================
def analyze_material_properties(strain, stress):
    """
    Compute mechanical properties:
    - Modulus (E) via linear fit in initial elastic region
    - UTS / maximum stress
    - Failure strain

    All units assumed MPa for stress, strain as mm/mm.
    """
    eps = np.asarray(strain, dtype=float)
    sig = np.asarray(stress, dtype=float)

    if eps.size == 0:
        return None, None, None

    # Elastic region: up to min(0.25% strain, 10% of max strain)
    strain_limit = min(0.0025, float(eps.max()) * 0.10) if eps.max() > 0 else 0
    elastic_mask = eps <= strain_limit
    eps_elastic = eps[elastic_mask]
    sig_elastic = sig[elastic_mask]

    youngs_modulus = None
    if eps_elastic.size >= 5:
        coeffs = np.polyfit(eps_elastic, sig_elastic, 1)
        youngs_modulus = coeffs[0]

    uts = float(sig.max())
    failure_strain = float(eps[-1])

    return youngs_modulus, uts, failure_strain


# ==========================
#  CORE PLOTTING LOGIC
# ==========================
def plot_strain_stress_curve(
    path: str | None,
    gauge_length: float | None,
    area: float | None,
    multiple: bool = False,
    data_directory: str | None = None,
    save: bool = True,
    save_path: str = ".",
    no_calculation: bool = False,
    compression: bool = False,
):
    Console.header("CONFIGURATION SUMMARY")
    Console.info(f"Mode: {'MULTIPLE (comparison)' if multiple else 'SINGLE'}")
    Console.info(f"Using precomputed Stress/Strain (--no-calculation): {no_calculation}")
    Console.info(f"Compression mode: {compression}")
    if not no_calculation:
        Console.info(f"Gauge length (mm): {gauge_length}")
        Console.info(f"Area (mm^2): {area}")

    # Resolve files
    if multiple:
        if not data_directory:
            Console.error("Missing --data-directory for multiple-curves mode.")
            raise SystemExit(1)

        if not os.path.isdir(data_directory):
            Console.error(f"Data directory not found: {data_directory}")
            raise SystemExit(1)

        excel_files = [
            os.path.join(data_directory, f)
            for f in os.listdir(data_directory)
            if f.lower().endswith(".xlsx")
        ]
        excel_files.sort()

        if not excel_files:
            Console.error("No .xlsx files found in data directory.")
            raise SystemExit(1)

        Console.header("DISCOVERED INPUT FILES")
        for f in excel_files:
            Console.info(os.path.basename(f))

    else:
        if not path:
            Console.error("Single mode requires -p / --path.")
            raise SystemExit(1)

        if not os.path.isfile(path):
            Console.error(f"Input file not found: {path}")
            raise SystemExit(1)

        excel_files = [path]
        Console.header("INPUT FILE")
        Console.info(os.path.basename(path))

    # Color palette
    COLORS = [
        "#1f77b4", "#ff7f0e", "#2ca02c",
        "#d62728", "#9467bd", "#8c564b",
    ]

    plt.figure()
    start_time = time.time()

    Console.header("LOADING & PROCESSING CURVES")

    total_files = len(excel_files)
    for idx, file_path in enumerate(excel_files, start=1):
        if multiple:
            progress_bar(idx - 1, total_files, prefix="Loading curves")

        fname = os.path.basename(file_path)
        Console.info(f"Reading: {fname}")

        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            Console.error(f"Failed to read {fname}: {e}")
            continue

        # Detect columns
        strain_col = detect_column(df, ["strain"])
        stress_col = detect_column(df, ["stress", "mstress", "σ", "sigma"])

        # ----- PRECOMPUTED STRESS/STRAIN MODE -----
        if no_calculation:
            if strain_col is None or stress_col is None:
                Console.warn(f"Missing Strain/Stress columns in {fname}, skipping.")
                continue

            strain = df[strain_col]
            stress = df[stress_col]

            # Unit handling
            if "gpa" in stress_col.lower():
                Console.info(f"{fname}: Stress in GPa detected → converting to MPa.")
                stress = stress * 1000.0

        # ----- RAW MACHINE DATA MODE (LOAD & ELONGATION) -----
        else:
            load_col = detect_column(df, ["load"])
            elong_col = detect_column(df, ["elong", "extension"])

            if not load_col or not elong_col:
                Console.error(f"{fname}: Missing Load/Elongation for calculation, skipping.")
                continue

            load = df[load_col] * 1000.0  # kN → N
            deltaL = df[elong_col]        # mm displacement (positive for movement)

            if compression:
                Console.info(f"{fname}: Compression mode ON → using |ΔL|/L0 and |load|/A.")
                stress = load.abs() / area
                strain = deltaL.abs() / gauge_length
            else:
                Console.info(f"{fname}: Tension mode → using ΔL/L0 and load/A.")
                stress = load / area
                strain = deltaL / gauge_length

        # Clean data: remove negative/NaN
        mask = (strain >= 0) & (stress >= 0)
        strain = strain[mask]
        stress = stress[mask]

        # For precomputed tests, trim initial noisy vertical jump
        if no_calculation:
            trim_mask = strain > 0.0005
            strain = strain[trim_mask]
            stress = stress[trim_mask]

        if len(strain) == 0 or len(stress) == 0:
            Console.warn(f"{fname}: No valid data points after filtering, skipping.")
            continue

        label = extract_label_from_filename(fname)
        color = COLORS[(idx - 1) % len(COLORS)]

        # === Mechanical properties ===
        E, UTS, failure_strain = analyze_material_properties(strain, stress)

        if compression:
            Console.header(f"COMPRESSION PROPERTIES: {label}")
        else:
            Console.header(f"TENSILE PROPERTIES: {label}")

        if E is not None:
            if compression:
                Console.info(f"Compressive Modulus (E): {E:,.2f} MPa")
            else:
                Console.info(f"Young’s Modulus (E): {E:,.2f} MPa")
        else:
            Console.warn("Modulus could not be determined (insufficient elastic data).")

        if UTS is not None:
            if compression:
                Console.info(f"Compressive Strength (σ_max): {UTS:,.2f} MPa")
            else:
                Console.info(f"Ultimate Tensile Strength (UTS): {UTS:,.2f} MPa")
        else:
            Console.warn("Maximum stress could not be determined.")

        if failure_strain is not None:
            Console.info(f"Failure Strain: {failure_strain:.4f} mm/mm")
        else:
            Console.warn("Failure strain could not be determined.")

        # === Plot curve ===
        plt.plot(strain, stress, label=label, color=color)
        Console.success(f"Curve successfully processed: {label}")

    if multiple:
        progress_bar(total_files, total_files, prefix="Loading curves")

    Console.header("GENERATING ENGINEERING PLOT")

    plt.xlabel("Engineering Strain (mm/mm)")
    plt.ylabel("Engineering Stress (MPa)")
    if multiple:
        plt.title("Stress–Strain Curves (Comparison of Samples)")
    else:
        plt.title("Stress–Strain Curve")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)

    if save:
        if multiple:
            png_path = os.path.join(save_path, "stress_strain_comparison.png")
            svg_path = os.path.join(save_path, "stress_strain_comparison.svg")
        else:
            input_base = os.path.splitext(os.path.basename(path))[0]
            png_path = os.path.join(save_path, f"{input_base}.png")
            svg_path = os.path.join(save_path, f"{input_base}.svg")

        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.savefig(svg_path, dpi=300, bbox_inches="tight")

        Console.header("OUTPUT")
        Console.success(f"Saved PNG: {png_path}")
        Console.success(f"Saved SVG: {svg_path}")
    else:
        Console.info("Displaying plot window (no files saved).")
        plt.show()

    elapsed = time.time() - start_time
    Console.header("SESSION SUMMARY")
    Console.info(f"Total runtime: {elapsed:.3f} s")
    Console.success("StrainCurvePlotter run completed.")


# ==========================
#  CUSTOM HELP FORMATTER
# ==========================
class ColorHelp(argparse.HelpFormatter):
    def start_section(self, heading):
        heading = f"{Console.MAGENTA}{Console.BOLD}{heading}{Console.RESET}"
        super().start_section(heading)


# ==========================
#  MAIN
# ==========================
if __name__ == "__main__":
    apply_engineering_style()
    banner()

    parser = argparse.ArgumentParser(
        prog="StrainCurvePlotter",
        description=f"""{Console.CYAN}{Console.BOLD}
Scientific Stress–Strain Curve Plotting Tool
{Console.RESET}
Processes tensile / flexural / compression test data from Excel exports and
generates engineering-grade stress–strain plots and mechanical summaries.
""",
        epilog=f"""{Console.GREEN}{Console.BOLD}
EXAMPLE USAGE
{Console.RESET}

{Console.YELLOW}1.{Console.RESET} Single precomputed curve (e.g. flexural with MStress/Strain):
  {Console.CYAN}python strain_curve_plotter.py -p control.xlsx --no-calculation --save{Console.RESET}

{Console.YELLOW}2.{Console.RESET} Single raw TENSILE test (needs gauge length and area):
  {Console.CYAN}python strain_curve_plotter.py -p tensile.xlsx -L 50 -A 78.5 --save{Console.RESET}

{Console.YELLOW}3.{Console.RESET} Single raw COMPRESSION test on cylinder:
  {Console.CYAN}python strain_curve_plotter.py -p compression.xlsx -L 30 -A 490.87 --compression --save{Console.RESET}

{Console.YELLOW}4.{Console.RESET} Compare all samples in a folder (e.g. control, 5%, 10%, ...):
  {Console.CYAN}python strain_curve_plotter.py --multiple-curves --data-directory ./samples --no-calculation --save{Console.RESET}
""",
        formatter_class=ColorHelp,
    )

    parser.add_argument(
        "-p", "--path",
        help="Path to a single Excel file (single-curve mode)."
    )
    parser.add_argument(
        "-L", "--gauge-length",
        type=float,
        help="Gauge length Lo in mm (required if NOT using --no-calculation)."
    )
    parser.add_argument(
        "-A", "--area",
        type=float_with_comma,
        help="Cross-sectional area in mm^2 (required if NOT using --no-calculation)."
    )
    parser.add_argument(
        "--multiple-curves",
        action="store_true",
        help="Enable comparison mode: plot all .xlsx files in --data-directory."
    )
    parser.add_argument(
        "--data-directory",
        help="Directory containing multiple Excel files for comparison mode."
    )
    parser.add_argument(
        "--no-calculation",
        action="store_true",
        help="Use existing Stress/Strain columns from file instead of computing from Load/Elongation."
    )
    parser.add_argument(
        "--compression",
        action="store_true",
        help="Treat test as COMPRESSION (|load|/A, |ΔL|/L0, and compressive terminology)."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the generated plot as PNG and SVG in --save-path."
    )
    parser.add_argument(
        "--save-path",
        default=".",
        help="Output directory for saved plots (default: current directory)."
    )

    args = parser.parse_args()

    if not args.no_calculation:
        if args.gauge_length is None or args.area is None:
            Console.error("When NOT using --no-calculation, you must provide -L/--gauge-length and -A/--area.")
            raise SystemExit(1)

    plot_strain_stress_curve(
        path=args.path,
        gauge_length=args.gauge_length,
        area=args.area,
        multiple=args.multiple_curves,
        data_directory=args.data_directory,
        save=args.save,
        save_path=args.save_path,
        no_calculation=args.no_calculation,
        compression=args.compression,
    )