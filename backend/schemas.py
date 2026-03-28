from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class IndexRange(BaseModel):
    min: int
    max: int
    count: int


class MetaResponse(BaseModel):
    data_path: str
    input_window: int
    forecast_horizon: int
    feature_columns: list[str]
    target_column: str
    available_start_indices: IndexRange
    forecast_index_range: IndexRange
    has_timestamp: bool
    timestamp_column: str | None = None
    available_timestamps: list[str] | None = None


class ForecastRequest(BaseModel):
    start_index: int = Field(ge=0)


class ForecastPoint(BaseModel):
    horizon_step: int
    label: str
    forecast_index: int
    timestamp: str | None = None
    value: float


class ForecastResponse(BaseModel):
    start_index: int
    horizon: int
    points: list[ForecastPoint]


class RecommendRequest(BaseModel):
    current_index: int = Field(ge=0)
    duration_hours: int = Field(ge=1)
    latest_finish_index: int | None = Field(default=None, ge=0)
    gpu_preset: str | None = None


class WindowScore(BaseModel):
    start_index: int
    end_index: int
    duration_steps: int
    mean_residual_demand: float
    label: Literal["Great", "Okay", "Avoid"]


class TimelineSegment(BaseModel):
    start_index: int
    end_index: int
    label: Literal["Great", "Okay", "Avoid"]


class RecommendResponse(BaseModel):
    current_index: int
    forecast_start_index: int
    duration_hours: int
    duration_steps: int
    latest_finish_index: int
    recommendation: WindowScore
    run_now: WindowScore
    timeline: list[TimelineSegment]
    points: list[ForecastPoint]
    details: dict[str, float | int | str | None]
