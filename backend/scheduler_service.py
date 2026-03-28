from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from backend.model_service import ForecastModelService


@dataclass(frozen=True)
class WindowResult:
    start_index: int
    end_index: int
    duration_steps: int
    mean_residual_demand: float
    label: str


class SchedulerService:
    def __init__(self, model_service: ForecastModelService):
        self.model_service = model_service

    def _tertile_thresholds(self, values: np.ndarray) -> tuple[float, float]:
        q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
        return float(q1), float(q2)

    def _label_from_value(self, value: float, q1: float, q2: float) -> str:
        if value <= q1:
            return "Great"
        if value <= q2:
            return "Okay"
        return "Avoid"

    def _score_window(self, values: np.ndarray, start_offset: int, duration_steps: int) -> float:
        return float(values[start_offset : start_offset + duration_steps].mean())

    def recommend(
        self,
        current_index: int,
        duration_hours: int,
        latest_finish_index: int | None = None,
        gpu_preset: str | None = None,
    ) -> dict[str, object]:
        forecast = self.model_service.forecast(current_index)
        values = np.asarray(forecast["values"], dtype=np.float32)
        horizon = int(values.shape[0])
        duration_steps = max(1, int(math.ceil(duration_hours)))
        if duration_steps > horizon:
            raise ValueError(f"duration_hours={duration_hours} exceeds forecast horizon={horizon}")

        indices = np.asarray(forecast["indices"], dtype=int)
        forecast_start = int(indices[0])
        forecast_end = int(indices[-1])
        latest_finish = forecast_end if latest_finish_index is None else min(latest_finish_index, forecast_end)
        if latest_finish < forecast_start:
            raise ValueError(
                f"latest_finish_index={latest_finish_index} is before first forecast index={forecast_start}"
            )

        q1, q2 = self._tertile_thresholds(values)
        max_start_index = latest_finish - duration_steps + 1
        candidate_offsets = [
            i for i, abs_idx in enumerate(indices) if abs_idx <= max_start_index and i + duration_steps <= horizon
        ]
        if not candidate_offsets:
            raise ValueError("No candidate windows satisfy duration and latest_finish_index.")

        scores: list[tuple[int, float]] = []
        for offset in candidate_offsets:
            scores.append((offset, self._score_window(values, offset, duration_steps)))
        best_offset, best_mean = min(scores, key=lambda x: x[1])

        run_now_offset = 0
        run_now_mean = self._score_window(values, run_now_offset, duration_steps)

        def build_window(offset: int, mean_value: float) -> WindowResult:
            start = int(indices[offset])
            end = int(indices[offset + duration_steps - 1])
            return WindowResult(
                start_index=start,
                end_index=end,
                duration_steps=duration_steps,
                mean_residual_demand=float(mean_value),
                label=self._label_from_value(float(mean_value), q1, q2),
            )

        recommended = build_window(best_offset, best_mean)
        run_now = build_window(run_now_offset, run_now_mean)

        point_labels = [self._label_from_value(float(v), q1, q2) for v in values]
        segments: list[dict[str, object]] = []
        seg_start = 0
        for idx in range(1, len(point_labels) + 1):
            is_boundary = idx == len(point_labels) or point_labels[idx] != point_labels[seg_start]
            if is_boundary:
                segments.append(
                    {
                        "start_index": int(indices[seg_start]),
                        "end_index": int(indices[idx - 1]),
                        "label": point_labels[seg_start],
                    }
                )
                seg_start = idx

        points = []
        for i, (abs_idx, value) in enumerate(zip(indices.tolist(), values.tolist(), strict=True)):
            points.append(
                {
                    "horizon_step": i + 1,
                    "label": forecast["labels"][i],
                    "forecast_index": abs_idx,
                    "timestamp": forecast["timestamps"][i],
                    "value": float(value),
                }
            )

        return {
            "current_index": current_index,
            "forecast_start_index": forecast_start,
            "duration_hours": duration_hours,
            "duration_steps": duration_steps,
            "latest_finish_index": int(latest_finish),
            "recommendation": recommended.__dict__,
            "run_now": run_now.__dict__,
            "timeline": segments,
            "points": points,
            "details": {
                "gpu_preset": gpu_preset,
                "candidate_count": len(candidate_offsets),
                "best_offset": int(best_offset),
                "q1": q1,
                "q2": q2,
            },
        }
