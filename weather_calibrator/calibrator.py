import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy.interpolate import griddata

from .data_loader import CleanedDataset


@dataclass
class CalibrationResult:
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    latitudes: np.ndarray = field(default_factory=lambda: np.array([]))
    longitudes: np.ndarray = field(default_factory=lambda: np.array([]))
    temperature_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    pressure_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    calibrated_pressure_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    filled_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    target_latitudes: np.ndarray = field(default_factory=lambda: np.array([]))
    target_longitudes: np.ndarray = field(default_factory=lambda: np.array([]))
    bicubic_temperature: np.ndarray = field(default_factory=lambda: np.array([]))
    bicubic_pressure: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dict(self) -> Dict:
        def _tolist(arr):
            if isinstance(arr, np.ndarray):
                return arr.tolist()
            return arr

        return {
            "timestamps": _tolist(self.timestamps),
            "latitudes": _tolist(self.latitudes),
            "longitudes": _tolist(self.longitudes),
            "temperature_grid": _tolist(self.temperature_grid),
            "pressure_grid": _tolist(self.pressure_grid),
            "calibrated_pressure_grid": _tolist(self.calibrated_pressure_grid),
            "filled_mask": _tolist(self.filled_mask.astype(bool) if self.filled_mask.size else []),
            "target_latitudes": _tolist(self.target_latitudes),
            "target_longitudes": _tolist(self.target_longitudes),
            "bicubic_temperature": _tolist(self.bicubic_temperature),
            "bicubic_pressure": _tolist(self.bicubic_pressure),
        }


class Calibrator:
    SEA_LEVEL_PRESSURE_HPA = 1013.25
    TEMPERATURE_LAPSE_RATE = 0.0065
    GAS_CONSTANT_DRY_AIR = 287.058
    GRAVITY = 9.80665

    def __init__(
        self,
        dataset: Optional[CleanedDataset] = None,
        elevation_m: Optional[np.ndarray] = None,
    ) -> None:
        self.dataset = dataset
        self.elevation_m = elevation_m
        self._result: Optional[CalibrationResult] = None

    def _fill_missing(self, grid: np.ndarray) -> np.ndarray:
        if grid.size == 0:
            return grid

        filled = grid.copy()
        mask = np.isnan(filled)

        if not np.any(mask):
            return filled

        if np.all(mask):
            return np.zeros_like(filled, dtype=np.float64)

        valid_coords = np.argwhere(~mask)
        valid_values = filled[~mask]
        missing_coords = np.argwhere(mask)

        n_dims = grid.ndim

        if n_dims == 1:
            x = np.arange(len(filled))
            filled[mask] = np.interp(
                x[mask], x[~mask], valid_values, left=valid_values[0], right=valid_values[-1]
            )
            return filled

        for mc in missing_coords:
            distances = np.sqrt(
                np.sum((valid_coords - mc) ** 2, axis=1)
            )
            if len(valid_values) >= 4:
                sorted_idx = np.argsort(distances)
                nearest_idx = sorted_idx[:4]
                weights = 1.0 / (distances[nearest_idx] + 1e-10)
                weights /= weights.sum()
                filled[tuple(mc)] = np.sum(valid_values[nearest_idx] * weights)
            elif len(valid_values) > 0:
                filled[tuple(mc)] = np.mean(valid_values)
            else:
                filled[tuple(mc)] = 0.0

        return filled

    def _apply_elevation_correction(
        self,
        pressure_grid: np.ndarray,
        temperature_grid: np.ndarray,
        latitudes: np.ndarray,
    ) -> np.ndarray:
        n_times = pressure_grid.shape[0]
        n_stations = pressure_grid.shape[1]
        corrected = pressure_grid.copy()

        elevations = self.elevation_m
        if elevations is None:
            elevations = self._estimate_elevation(latitudes)

        for s_idx in range(n_stations):
            h = elevations[s_idx]
            temps_k = temperature_grid[:, s_idx] + 273.15
            valid = ~(np.isnan(pressure_grid[:, s_idx]) | np.isnan(temperature_grid[:, s_idx]))
            if not np.any(valid):
                continue
            t_kelvin = temps_k[valid]
            exponent = (self.GRAVITY * h) / (
                self.GAS_CONSTANT_DRY_AIR * (t_kelvin + self.TEMPERATURE_LAPSE_RATE * h / 2.0)
            )
            corrected[valid, s_idx] = pressure_grid[valid, s_idx] * np.exp(exponent)

        return corrected

    def _estimate_elevation(self, latitudes: np.ndarray) -> np.ndarray:
        n = latitudes.shape[0]
        elev = np.zeros(n, dtype=np.float64)
        lat_mean = np.nanmean(latitudes) if n > 0 else 45.0
        for i in range(n):
            elev[i] = 50.0 + 20.0 * np.sin(np.radians(latitudes[i] - lat_mean))
        return np.maximum(elev, 0.0)

    def _interpolate_scatter_to_grid(
        self,
        scatter_values: np.ndarray,
        src_lat: np.ndarray,
        src_lon: np.ndarray,
        tgt_lat: np.ndarray,
        tgt_lon: np.ndarray,
    ) -> np.ndarray:
        n_times = scatter_values.shape[0]
        n_tgt_lat = tgt_lat.size
        n_tgt_lon = tgt_lon.size
        result = np.full((n_times, n_tgt_lat, n_tgt_lon), np.nan, dtype=np.float64)

        tgt_lon_mesh, tgt_lat_mesh = np.meshgrid(tgt_lon, tgt_lat)
        tgt_points = np.column_stack([tgt_lat_mesh.ravel(), tgt_lon_mesh.ravel()])

        src_points = np.column_stack([src_lat, src_lon])

        for t in range(n_times):
            values = scatter_values[t, :]
            valid = ~np.isnan(values)
            n_valid = valid.sum()

            if n_valid == 0:
                result[t, :, :] = 0.0
                continue

            if n_valid < 3:
                fill_val = np.nanmean(values)
                result[t, :, :] = fill_val
                continue

            interp_cubic = griddata(
                src_points[valid],
                values[valid],
                tgt_points,
                method="cubic",
                fill_value=np.nan,
            )

            nan_mask = np.isnan(interp_cubic)
            if np.any(nan_mask):
                interp_linear = griddata(
                    src_points[valid],
                    values[valid],
                    tgt_points[nan_mask],
                    method="linear",
                    fill_value=np.nan,
                )
                linear_ok = ~np.isnan(interp_linear)
                if np.any(linear_ok):
                    interp_cubic[nan_mask] = np.where(
                        linear_ok, interp_linear, np.nan
                    )
                still_nan = np.isnan(interp_cubic)
                if np.any(still_nan):
                    nearest = griddata(
                        src_points[valid],
                        values[valid],
                        tgt_points[still_nan],
                        method="nearest",
                    )
                    interp_cubic[still_nan] = nearest

            result[t, :, :] = interp_cubic.reshape(n_tgt_lat, n_tgt_lon)

        return result

    def calibrate(
        self,
        dataset: Optional[CleanedDataset] = None,
        target_lat_step: Optional[float] = None,
        target_lon_step: Optional[float] = None,
        num_lat_points: Optional[int] = None,
        num_lon_points: Optional[int] = None,
    ) -> CalibrationResult:
        data = dataset or self.dataset
        if data is None:
            raise ValueError("No dataset provided")
        if data.pressure_grid.size == 0:
            raise ValueError("Dataset has no pressure grid")

        temp_grid = self._fill_missing(data.temperature_grid)
        press_grid = self._fill_missing(data.pressure_grid)
        filled_mask = np.isnan(data.temperature_grid) | np.isnan(data.pressure_grid)

        calibrated_pressure = self._apply_elevation_correction(
            press_grid, temp_grid, data.latitudes
        )

        sort_idx = np.lexsort((data.longitudes, data.latitudes))
        latitudes = data.latitudes[sort_idx]
        longitudes = data.longitudes[sort_idx]
        temp_grid = temp_grid[:, sort_idx]
        press_grid = press_grid[:, sort_idx]
        calibrated_pressure = calibrated_pressure[:, sort_idx]
        filled_mask = filled_mask[:, sort_idx]

        lat_min, lat_max = latitudes.min(), latitudes.max()
        lon_min, lon_max = longitudes.min(), longitudes.max()

        if lat_min == lat_max:
            lat_min -= 0.5
            lat_max += 0.5
        if lon_min == lon_max:
            lon_min -= 0.5
            lon_max += 0.5

        if target_lat_step is not None:
            tgt_lat = np.arange(lat_min, lat_max + target_lat_step / 2.0, target_lat_step)
        elif num_lat_points is not None:
            tgt_lat = np.linspace(lat_min, lat_max, num_lat_points)
        else:
            n = max(len(latitudes) * 2, 10)
            tgt_lat = np.linspace(lat_min, lat_max, n)

        if target_lon_step is not None:
            tgt_lon = np.arange(lon_min, lon_max + target_lon_step / 2.0, target_lon_step)
        elif num_lon_points is not None:
            tgt_lon = np.linspace(lon_min, lon_max, num_lon_points)
        else:
            n = max(len(longitudes) * 2, 10)
            tgt_lon = np.linspace(lon_min, lon_max, n)

        bicubic_temp = self._interpolate_scatter_to_grid(
            temp_grid, latitudes, longitudes, tgt_lat, tgt_lon
        )
        bicubic_press = self._interpolate_scatter_to_grid(
            calibrated_pressure, latitudes, longitudes, tgt_lat, tgt_lon
        )

        self._result = CalibrationResult(
            timestamps=data.timestamps,
            latitudes=latitudes,
            longitudes=longitudes,
            temperature_grid=temp_grid,
            pressure_grid=press_grid,
            calibrated_pressure_grid=calibrated_pressure,
            filled_mask=filled_mask,
            target_latitudes=tgt_lat,
            target_longitudes=tgt_lon,
            bicubic_temperature=bicubic_temp,
            bicubic_pressure=bicubic_press,
        )

        return self._result

    def to_json(self, output_path: str, result: Optional[CalibrationResult] = None) -> str:
        res = result or self._result
        if res is None:
            raise RuntimeError("No calibration result available. Call calibrate() first.")

        directory = os.path.dirname(output_path)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(res.to_dict(), f, indent=2, ensure_ascii=False, allow_nan=False)

        return os.path.abspath(output_path)
