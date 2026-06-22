import argparse
import os
import sys
from typing import Optional

import numpy as np

from weather_calibrator import DataLoader, Calibrator


def run_pipeline(
    input_csv: str,
    output_json: str,
    num_lat_points: Optional[int] = None,
    num_lon_points: Optional[int] = None,
    target_lat_step: Optional[float] = None,
    target_lon_step: Optional[float] = None,
) -> str:
    print(f"Loading data from {input_csv}...")
    loader = DataLoader(input_csv)
    dataset = loader.process()

    print(
        f"Loaded {len(dataset.records)} records from "
        f"{len(dataset.unique_stations)} stations across "
        f"{len(dataset.timestamps)} timestamps"
    )

    temp_nan = int(np.isnan(dataset.temperature_grid).sum())
    press_nan = int(np.isnan(dataset.pressure_grid).sum())
    print(f"Missing values: temperature={temp_nan}, pressure={press_nan}")

    print("Running calibration and bicubic interpolation...")
    calibrator = Calibrator(dataset)
    result = calibrator.calibrate(
        num_lat_points=num_lat_points,
        num_lon_points=num_lon_points,
        target_lat_step=target_lat_step,
        target_lon_step=target_lon_step,
    )

    print(
        f"Interpolated grid shape (time x lat x lon): "
        f"{result.bicubic_pressure.shape}"
    )

    output_path = calibrator.to_json(output_json)
    print(f"Results written to: {output_path}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Weather station data calibration and bicubic interpolation tool"
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Input CSV file with weather station data"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output JSON file path for calibrated results"
    )
    parser.add_argument(
        "--num-lat", type=int, default=None, help="Number of latitude grid points"
    )
    parser.add_argument(
        "--num-lon", type=int, default=None, help="Number of longitude grid points"
    )
    parser.add_argument(
        "--lat-step", type=float, default=None, help="Latitude step size in degrees"
    )
    parser.add_argument(
        "--lon-step", type=float, default=None, help="Longitude step size in degrees"
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    run_pipeline(
        input_csv=args.input,
        output_json=args.output,
        num_lat_points=args.num_lat,
        num_lon_points=args.num_lon,
        target_lat_step=args.lat_step,
        target_lon_step=args.lon_step,
    )
