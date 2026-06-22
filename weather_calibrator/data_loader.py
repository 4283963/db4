import csv
import os
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict

import numpy as np


@dataclass
class StationRecord:
    timestamp: float
    station_id: str
    latitude: float
    longitude: float
    temperature: Optional[float]
    pressure: Optional[float]


@dataclass
class CleanedDataset:
    records: List[StationRecord] = field(default_factory=list)
    unique_stations: List[str] = field(default_factory=list)
    latitudes: np.ndarray = field(default_factory=lambda: np.array([]))
    longitudes: np.ndarray = field(default_factory=lambda: np.array([]))
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    temperature_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    pressure_grid: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dict(self) -> Dict:
        return {
            "records": [asdict(r) for r in self.records],
            "unique_stations": self.unique_stations,
            "latitudes": self.latitudes.tolist(),
            "longitudes": self.longitudes.tolist(),
            "timestamps": self.timestamps.tolist(),
            "temperature_grid": self.temperature_grid.tolist(),
            "pressure_grid": self.pressure_grid.tolist(),
        }


class DataLoader:
    REQUIRED_COLUMNS = {
        "timestamp",
        "station_id",
        "latitude",
        "longitude",
        "temperature",
        "pressure",
    }

    TEMPERATURE_RANGE = (-60.0, 60.0)
    PRESSURE_RANGE = (800.0, 1100.0)

    def __init__(
        self,
        csv_path: Optional[str] = None,
        delimiter: str = ",",
        encoding: str = "utf-8",
    ) -> None:
        self.csv_path = csv_path
        self.delimiter = delimiter
        self.encoding = encoding
        self._raw_records: List[Dict] = []

    def load_csv(self, csv_path: Optional[str] = None) -> List[Dict]:
        path = csv_path or self.csv_path
        if path is None:
            raise ValueError("CSV path must be provided either in constructor or load_csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"CSV file not found: {path}")

        self._raw_records = []
        with open(path, "r", encoding=self.encoding) as f:
            reader = csv.DictReader(f, delimiter=self.delimiter)
            if reader.fieldnames is None:
                raise ValueError("CSV file has no header row")

            missing = self.REQUIRED_COLUMNS - set(reader.fieldnames)
            if missing:
                raise ValueError(f"CSV missing required columns: {sorted(missing)}")

            for row in reader:
                self._raw_records.append(dict(row))

        return self._raw_records

    def _parse_float(self, value: str, field_name: str) -> Optional[float]:
        if value is None or str(value).strip() in ("", "NA", "N/A", "null", "None", "nan", "NaN"):
            return None
        try:
            return float(value.strip())
        except (ValueError, AttributeError):
            return None

    def _is_valid_temperature(self, value: Optional[float]) -> bool:
        if value is None:
            return True
        return self.TEMPERATURE_RANGE[0] <= value <= self.TEMPERATURE_RANGE[1]

    def _is_valid_pressure(self, value: Optional[float]) -> bool:
        if value is None:
            return True
        return self.PRESSURE_RANGE[0] <= value <= self.PRESSURE_RANGE[1]

    def clean(self) -> List[StationRecord]:
        if not self._raw_records:
            raise RuntimeError("No data loaded. Call load_csv() first.")

        cleaned: List[StationRecord] = []
        for row in self._raw_records:
            timestamp = self._parse_float(row.get("timestamp", ""), "timestamp")
            if timestamp is None:
                continue

            station_id = str(row.get("station_id", "")).strip()
            if not station_id:
                continue

            latitude = self._parse_float(row.get("latitude", ""), "latitude")
            longitude = self._parse_float(row.get("longitude", ""), "longitude")
            if latitude is None or longitude is None:
                continue

            temperature = self._parse_float(row.get("temperature", ""), "temperature")
            pressure = self._parse_float(row.get("pressure", ""), "pressure")

            if not self._is_valid_temperature(temperature):
                temperature = None
            if not self._is_valid_pressure(pressure):
                pressure = None

            cleaned.append(
                StationRecord(
                    timestamp=timestamp,
                    station_id=station_id,
                    latitude=latitude,
                    longitude=longitude,
                    temperature=temperature,
                    pressure=pressure,
                )
            )

        return cleaned

    def build_grid(self, records: List[StationRecord]) -> CleanedDataset:
        if not records:
            raise ValueError("No cleaned records available to build grid")

        unique_stations = sorted({r.station_id for r in records})
        station_map = {sid: idx for idx, sid in enumerate(unique_stations)}

        unique_timestamps = sorted({r.timestamp for r in records})
        time_map = {t: idx for idx, t in enumerate(unique_timestamps)}

        n_stations = len(unique_stations)
        n_times = len(unique_timestamps)

        latitudes = np.full(n_stations, np.nan, dtype=np.float64)
        longitudes = np.full(n_stations, np.nan, dtype=np.float64)
        temperature_grid = np.full((n_times, n_stations), np.nan, dtype=np.float64)
        pressure_grid = np.full((n_times, n_stations), np.nan, dtype=np.float64)

        for r in records:
            s_idx = station_map[r.station_id]
            t_idx = time_map[r.timestamp]

            if np.isnan(latitudes[s_idx]):
                latitudes[s_idx] = r.latitude
                longitudes[s_idx] = r.longitude

            if r.temperature is not None:
                temperature_grid[t_idx, s_idx] = r.temperature

            if r.pressure is not None:
                pressure_grid[t_idx, s_idx] = r.pressure

        return CleanedDataset(
            records=records,
            unique_stations=unique_stations,
            latitudes=latitudes,
            longitudes=longitudes,
            timestamps=np.array(unique_timestamps, dtype=np.float64),
            temperature_grid=temperature_grid,
            pressure_grid=pressure_grid,
        )

    def process(self, csv_path: Optional[str] = None) -> CleanedDataset:
        self.load_csv(csv_path)
        cleaned = self.clean()
        return self.build_grid(cleaned)
