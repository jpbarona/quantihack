import { useMemo } from "react";
import { labelAtForecastStart, labelAtRow } from "../scheduleLabels";
import TimelineBand from "./TimelineBand";
import type { MetaResponse, RecommendResponse } from "../types";

type SchedulerState = {
  currentIndex: number;
  durationHours: number;
  latestFinishIndex: number;
  gpuPreset: string;
};

type SchedulerTabProps = {
  meta: MetaResponse;
  state: SchedulerState;
  recommendation: RecommendResponse | null;
  loading: boolean;
  error: string | null;
  lastRecommendationAt: string | null;
  onStateChange: (patch: Partial<SchedulerState>) => void;
  onRequestRecommendation: () => void;
};

const GPU_PRESETS = ["A100 x1", "A100 x4", "H100 x8"];

function meanResidualImprovementPct(runNow: number, recommended: number): string | null {
  if (!Number.isFinite(runNow) || runNow === 0) return null;
  const pct = ((runNow - recommended) / runNow) * 100;
  const rounded = Math.round(pct * 10) / 10;
  const prefix = rounded > 0 ? "+" : "";
  return `${prefix}${rounded.toFixed(1)}%`;
}

function buildCurrentIndexOptions(min: number, max: number, current: number): number[] {
  const total = max - min + 1;
  const maxOptions = 120;
  if (total <= maxOptions) {
    return Array.from({ length: total }, (_, i) => min + i);
  }
  const step = Math.max(1, Math.floor(total / (maxOptions - 1)));
  const options = new Set<number>();
  for (let value = min; value <= max; value += step) {
    options.add(value);
  }
  options.add(max);
  options.add(current);
  return Array.from(options).sort((a, b) => a - b);
}

export default function SchedulerTab({
  meta,
  state,
  recommendation,
  loading,
  error,
  lastRecommendationAt,
  onStateChange,
  onRequestRecommendation
}: SchedulerTabProps) {
  const currentIndexOptions = useMemo(
    () => buildCurrentIndexOptions(meta.available_start_indices.min, meta.available_start_indices.max, state.currentIndex),
    [meta.available_start_indices.max, meta.available_start_indices.min, state.currentIndex]
  );

  const latestFinishOptions = useMemo(() => {
    const start = state.currentIndex + meta.input_window;
    const end = start + meta.forecast_horizon - 1;
    const options = [];
    for (let idx = start; idx <= end; idx += 1) {
      options.push(idx);
    }
    return options;
  }, [meta.forecast_horizon, meta.input_window, state.currentIndex]);

  return (
    <div className="tab-layout">
      <div className="card">
        <h3>Scheduler Inputs</h3>
        <div className="form-grid">
          <label>
            {meta.has_timestamp && meta.available_timestamps?.length
              ? "Current position (forecast start)"
              : "Current position (step index)"}
            <select
              value={state.currentIndex}
              onChange={(event) => {
                const next = Number(event.target.value);
                const forecastStart = next + meta.input_window;
                const end = forecastStart + meta.forecast_horizon - 1;
                let latestFinishIndex = state.latestFinishIndex;
                if (latestFinishIndex < forecastStart || latestFinishIndex > end) {
                  latestFinishIndex = end;
                }
                onStateChange({ currentIndex: next, latestFinishIndex });
              }}
            >
              {currentIndexOptions.map((idx) => (
                <option key={idx} value={idx}>
                  {labelAtForecastStart(meta, idx)}
                </option>
              ))}
            </select>
          </label>
          <label>
            Duration (hours)
            <input
              type="number"
              min={1}
              max={meta.forecast_horizon}
              value={state.durationHours}
              onChange={(event) => onStateChange({ durationHours: Number(event.target.value) })}
            />
          </label>
          <label>
            {meta.has_timestamp && meta.available_timestamps?.length
              ? "Latest finish (deadline step)"
              : "Latest finish (step index)"}
            <select
              value={state.latestFinishIndex}
              onChange={(event) => onStateChange({ latestFinishIndex: Number(event.target.value) })}
            >
              {latestFinishOptions.map((idx) => (
                <option key={idx} value={idx}>
                  {labelAtRow(meta, idx)}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="preset-row">
          {GPU_PRESETS.map((preset) => (
            <button
              type="button"
              key={preset}
              className={`preset-pill ${state.gpuPreset === preset ? "active" : ""}`}
              onClick={() => onStateChange({ gpuPreset: preset })}
            >
              {preset}
            </button>
          ))}
        </div>
        <button className="primary-btn" type="button" onClick={onRequestRecommendation} disabled={loading}>
          {loading ? "Computing..." : "Get Recommendation"}
        </button>
        {lastRecommendationAt && <p className="muted">Last updated at {lastRecommendationAt}</p>}
        {error && <p className="error-text">{error}</p>}
      </div>

      {recommendation && (
        <>
          <div className="card hero-card">
            <h3>Recommended Window</h3>
            <div className="hero-metrics">
              <span>
                Start{" "}
                <strong title={`Step ${recommendation.recommendation.start_index}`}>
                  {labelAtRow(meta, recommendation.recommendation.start_index)}
                </strong>
              </span>
              <span>
                End{" "}
                <strong title={`Step ${recommendation.recommendation.end_index}`}>
                  {labelAtRow(meta, recommendation.recommendation.end_index)}
                </strong>
              </span>
              <span>
                Mean <strong>{recommendation.recommendation.mean_residual_demand.toFixed(2)}</strong>
              </span>
              <span className={`status ${recommendation.recommendation.label.toLowerCase()}`}>
                {recommendation.recommendation.label}
              </span>
            </div>
          </div>

          <div className="card">
            <h3>Timeline Band</h3>
            <TimelineBand
              minIndex={recommendation.points[0].forecast_index}
              maxIndex={recommendation.points[recommendation.points.length - 1].forecast_index}
              segments={recommendation.timeline}
              recommended={recommendation.recommendation}
            />
          </div>

          <div className="card split compare">
            <div>
              <h4>Run Now</h4>
              <p className="metric">{recommendation.run_now.mean_residual_demand.toFixed(2)}</p>
              <span className={`status ${recommendation.run_now.label.toLowerCase()}`}>{recommendation.run_now.label}</span>
            </div>
            <div className="split-delta">
              <span className="muted">Improvement</span>
              <p className="metric metric-delta">
                {meanResidualImprovementPct(
                  recommendation.run_now.mean_residual_demand,
                  recommendation.recommendation.mean_residual_demand
                ) ?? "—"}
              </p>
            </div>
            <div>
              <h4>Recommended</h4>
              <p className="metric">{recommendation.recommendation.mean_residual_demand.toFixed(2)}</p>
              <span className={`status ${recommendation.recommendation.label.toLowerCase()}`}>
                {recommendation.recommendation.label}
              </span>
            </div>
          </div>

          <div className="card">
            <details>
              <summary>Details</summary>
              <div className="details-grid">
                <span>Forecast Start Index</span>
                <span>{labelAtRow(meta, recommendation.forecast_start_index)}</span>
                <span>Duration Steps</span>
                <span>{recommendation.duration_steps}</span>
                <span>Latest Finish</span>
                <span>{labelAtRow(meta, recommendation.latest_finish_index)}</span>
                <span>GPU Preset</span>
                <span>{String(recommendation.details.gpu_preset ?? "-")}</span>
                <span>Candidate Windows</span>
                <span>{String(recommendation.details.candidate_count ?? "-")}</span>
              </div>
            </details>
          </div>
        </>
      )}
    </div>
  );
}
