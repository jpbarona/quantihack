from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.model_service import ForecastModelService
from backend.scheduler_service import SchedulerService
from backend.schemas import (
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    IndexRange,
    MetaResponse,
    RecommendRequest,
    RecommendResponse,
)


class AppState:
    model_service: ForecastModelService
    scheduler_service: SchedulerService


state = AppState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    state.model_service = ForecastModelService()
    state.scheduler_service = SchedulerService(state.model_service)
    yield


app = FastAPI(title="ML Scheduler Demo API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/meta", response_model=MetaResponse)
def get_meta() -> MetaResponse:
    meta = state.model_service.get_meta()
    return MetaResponse(
        data_path=str(state.model_service.data_path),
        input_window=meta.input_window,
        forecast_horizon=meta.forecast_horizon,
        feature_columns=meta.feature_columns,
        target_column=meta.target_column,
        available_start_indices=IndexRange(
            min=meta.available_start_min,
            max=meta.available_start_max,
            count=meta.available_start_count,
        ),
        forecast_index_range=IndexRange(
            min=meta.forecast_min_index,
            max=meta.forecast_max_index,
            count=meta.forecast_index_count,
        ),
        has_timestamp=meta.has_timestamp,
        timestamp_column=meta.timestamp_column,
        available_timestamps=meta.available_timestamps,
    )


@app.post("/api/forecast", response_model=ForecastResponse)
def post_forecast(payload: ForecastRequest) -> ForecastResponse:
    try:
        forecast = state.model_service.forecast(payload.start_index)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    points = [
        ForecastPoint(
            horizon_step=i + 1,
            label=forecast["labels"][i],
            forecast_index=forecast["indices"][i],
            timestamp=forecast["timestamps"][i],
            value=forecast["values"][i],
        )
        for i in range(len(forecast["values"]))
    ]
    return ForecastResponse(start_index=payload.start_index, horizon=len(points), points=points)


@app.post("/api/recommend", response_model=RecommendResponse)
def post_recommend(payload: RecommendRequest) -> RecommendResponse:
    try:
        response = state.scheduler_service.recommend(
            current_index=payload.current_index,
            duration_hours=payload.duration_hours,
            latest_finish_index=payload.latest_finish_index,
            gpu_preset=payload.gpu_preset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RecommendResponse(**response)
