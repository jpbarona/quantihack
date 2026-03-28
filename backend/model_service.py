from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def _as_numpy(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.padding = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        front = x[:, :, 0:1].repeat(1, 1, self.padding)
        end = x[:, :, -1:].repeat(1, 1, self.padding)
        x = torch.cat([front, x, end], dim=2)
        x = self.avg(x)
        return x.transpose(1, 2)


class DLinearBaseline(nn.Module):
    def __init__(
        self,
        input_window: int,
        forecast_horizon: int,
        input_feature_count: int,
        moving_avg_kernel: int = 25,
    ):
        super().__init__()
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon
        self.input_feature_count = input_feature_count
        self.decomposition = MovingAverage(kernel_size=moving_avg_kernel)
        self.linear = nn.Linear(input_window * input_feature_count * 2, forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.decomposition(x)
        seasonal = x - trend
        combined = torch.cat([seasonal, trend], dim=2)
        flattened = combined.reshape(x.size(0), self.input_window * self.input_feature_count * 2)
        output = self.linear(flattened)
        return output.unsqueeze(-1)


@dataclass(frozen=True)
class ForecastMeta:
    input_window: int
    forecast_horizon: int
    feature_columns: list[str]
    target_column: str
    target_index: int
    available_start_min: int
    available_start_max: int
    available_start_count: int
    forecast_min_index: int
    forecast_max_index: int
    forecast_index_count: int
    has_timestamp: bool
    timestamp_column: str | None
    available_timestamps: list[str] | None


class ForecastModelService:
    def __init__(self):
        root = Path(__file__).resolve().parents[1]
        checkpoint_path = root / "src/jp/grid_forecast_checkpoint.pth"
        self.data_path = Path(os.getenv("DATA_PATH", str(root / "data/CleanGridData/data_1h.parquet"))).resolve()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.feature_columns = list(checkpoint["feature_columns"])
        self.target_index = int(checkpoint["target_index"])
        self.target_column = self.feature_columns[self.target_index]
        self.input_window = int(checkpoint["input_window"])
        self.forecast_horizon = int(checkpoint["forecast_horizon"])
        self.scaler_mean = _as_numpy(checkpoint["scaler_mean"])
        self.scaler_scale = _as_numpy(checkpoint["scaler_scale"])
        self.moving_avg_kernel = int(checkpoint["dlinear_moving_avg_kernel"])
        self.model = DLinearBaseline(
            input_window=self.input_window,
            forecast_horizon=self.forecast_horizon,
            input_feature_count=len(self.feature_columns),
            moving_avg_kernel=self.moving_avg_kernel,
        )
        self.model.load_state_dict(checkpoint["dlinear_state_dict"])
        self.model.eval()

        self.df = pd.read_parquet(self.data_path)
        self.timestamp_column = self._detect_timestamp_column(self.df)
        self._ensure_residual_demand(self.df)
        self._validate_columns(self.df)
        self.values = self.df[self.feature_columns].to_numpy(dtype=np.float32)
        self.scaled_values = (self.values - self.scaler_mean) / self.scaler_scale

        if len(self.scaled_values) < self.input_window:
            raise ValueError("Dataset is shorter than input_window.")
        self.available_start_min = 0
        self.available_start_max = len(self.scaled_values) - self.input_window
        self.available_start_count = self.available_start_max - self.available_start_min + 1
        self.forecast_min_index = self.input_window
        self.forecast_max_index = len(self.scaled_values) - 1
        self.forecast_index_count = self.forecast_max_index - self.forecast_min_index + 1
        self.available_timestamps = self._build_available_timestamps()

    def _detect_timestamp_column(self, df: pd.DataFrame) -> str | None:
        datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if datetime_cols:
            return datetime_cols[0]
        for name in ("timestamp", "datetime", "time", "date"):
            for col in df.columns:
                if col.lower() == name:
                    return col
        return None

    def _build_available_timestamps(self) -> list[str] | None:
        if not self.timestamp_column:
            return None
        ts = pd.to_datetime(self.df[self.timestamp_column], errors="coerce")
        valid = ts.iloc[self.input_window :].dropna()
        if valid.empty:
            return None
        return [x.isoformat() for x in valid]

    def _ensure_residual_demand(self, df: pd.DataFrame) -> None:
        if "residual_demand" in df.columns:
            return
        required = ["demand", "wind", "solar", "hydro", "biomass"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Cannot build residual_demand; missing columns: {missing}")
        df["residual_demand"] = df["demand"] - (df["wind"] + df["solar"] + df["hydro"] + df["biomass"])

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing checkpoint feature columns: {missing}")

    def get_meta(self) -> ForecastMeta:
        return ForecastMeta(
            input_window=self.input_window,
            forecast_horizon=self.forecast_horizon,
            feature_columns=self.feature_columns,
            target_column=self.target_column,
            target_index=self.target_index,
            available_start_min=self.available_start_min,
            available_start_max=self.available_start_max,
            available_start_count=self.available_start_count,
            forecast_min_index=self.forecast_min_index,
            forecast_max_index=self.forecast_max_index,
            forecast_index_count=self.forecast_index_count,
            has_timestamp=self.timestamp_column is not None,
            timestamp_column=self.timestamp_column,
            available_timestamps=self.available_timestamps,
        )

    def _to_model_input(self, start_index: int) -> np.ndarray:
        end_index = start_index + self.input_window
        x_levels = self.scaled_values[start_index:end_index]
        x_deltas = np.diff(x_levels, axis=0, prepend=x_levels[0:1])
        x = np.concatenate([x_levels, x_deltas], axis=1)
        return x.astype(np.float32)

    def _validate_start_index(self, start_index: int) -> None:
        if start_index < self.available_start_min or start_index > self.available_start_max:
            raise ValueError(
                f"start_index must be in [{self.available_start_min}, {self.available_start_max}], got {start_index}"
            )

    def forecast(self, start_index: int) -> dict[str, object]:
        self._validate_start_index(start_index)
        x_np = self._to_model_input(start_index)
        x_t = torch.from_numpy(x_np).unsqueeze(0)
        with torch.no_grad():
            pred_deltas_scaled = self.model(x_t).squeeze(0).squeeze(-1).cpu().numpy()

        last_scaled_level = self.scaled_values[start_index + self.input_window - 1, self.target_index]
        pred_levels_scaled = last_scaled_level + np.cumsum(pred_deltas_scaled)
        pred_levels = (
            pred_levels_scaled * self.scaler_scale[self.target_index] + self.scaler_mean[self.target_index]
        ).astype(np.float32)

        forecast_start_index = start_index + self.input_window
        absolute_indices = np.arange(forecast_start_index, forecast_start_index + self.forecast_horizon, dtype=int)
        if self.timestamp_column is not None:
            timestamps = pd.to_datetime(self.df[self.timestamp_column], errors="coerce")
            labels = []
            ts_out = []
            for idx in absolute_indices:
                if idx < len(timestamps) and not pd.isna(timestamps.iloc[idx]):
                    value = timestamps.iloc[idx].isoformat()
                    labels.append(value)
                    ts_out.append(value)
                else:
                    labels.append(f"t+{idx - forecast_start_index + 1}")
                    ts_out.append(None)
        else:
            labels = [f"t+{i + 1}" for i in range(self.forecast_horizon)]
            ts_out = [None] * self.forecast_horizon

        return {
            "start_index": start_index,
            "forecast_start_index": forecast_start_index,
            "indices": absolute_indices.tolist(),
            "labels": labels,
            "timestamps": ts_out,
            "values": pred_levels.tolist(),
        }
