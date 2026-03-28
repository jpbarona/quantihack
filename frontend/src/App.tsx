import { useEffect, useState } from "react";
import { getMeta, getRecommendation } from "./api";
import ForecastsTab from "./components/ForecastsTab";
import SchedulerTab from "./components/SchedulerTab";
import type { ForecastPoint, MetaResponse, RecommendResponse } from "./types";

type TabKey = "scheduler" | "forecasts";

type SchedulerState = {
  currentIndex: number;
  durationHours: number;
  latestFinishIndex: number;
  gpuPreset: string;
};

function defaultSchedulerState(meta: MetaResponse): SchedulerState {
  const currentIndex = Math.max(
    meta.available_start_indices.min,
    meta.available_start_indices.max - meta.forecast_horizon
  );
  const forecastStart = currentIndex + meta.input_window;
  const latestFinishIndex = forecastStart + meta.forecast_horizon - 1;
  return {
    currentIndex,
    durationHours: Math.min(8, meta.forecast_horizon),
    latestFinishIndex,
    gpuPreset: "A100 x1"
  };
}

export default function App() {
  const [activeTab, setActiveTab] = useState<TabKey>("scheduler");
  const [meta, setMeta] = useState<MetaResponse | null>(null);
  const [schedulerState, setSchedulerState] = useState<SchedulerState | null>(null);
  const [recommendation, setRecommendation] = useState<RecommendResponse | null>(null);
  const [forecastPoints, setForecastPoints] = useState<ForecastPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastRecommendationAt, setLastRecommendationAt] = useState<string | null>(null);

  async function runRecommendation(state: SchedulerState) {
    setLoading(true);
    setError(null);
    try {
      const rec = await getRecommendation({
        current_index: state.currentIndex,
        duration_hours: state.durationHours,
        latest_finish_index: state.latestFinishIndex,
        gpu_preset: state.gpuPreset
      });
      setRecommendation(rec);
      setForecastPoints(rec.points);
      setLastRecommendationAt(new Date().toLocaleTimeString());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Recommendation failed.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    async function bootstrap() {
      setLoading(true);
      setError(null);
      try {
        const metaResponse = await getMeta();
        const initialState = defaultSchedulerState(metaResponse);
        setMeta(metaResponse);
        setSchedulerState(initialState);
        await runRecommendation(initialState);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load API metadata.");
      } finally {
        setLoading(false);
      }
    }
    void bootstrap();
  }, []);

  if (!meta || !schedulerState) {
    return (
      <div className="app-shell">
        <header>
          <h1>ML Scheduler Demo</h1>
          <p className="muted">Loading model metadata...</p>
          {error && <p className="error-text">{error}</p>}
        </header>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <header className="header">
        <h1>ML Scheduler Demo</h1>
        <p className="muted">Residual demand-aware scheduling with DLinear checkpoint inference.</p>
      </header>

      <div className="tabs">
        <button
          type="button"
          className={`tab-btn ${activeTab === "scheduler" ? "active" : ""}`}
          onClick={() => setActiveTab("scheduler")}
        >
          Scheduler
        </button>
        <button
          type="button"
          className={`tab-btn ${activeTab === "forecasts" ? "active" : ""}`}
          onClick={() => setActiveTab("forecasts")}
        >
          Forecasts
        </button>
      </div>

      {activeTab === "scheduler" ? (
        <SchedulerTab
          meta={meta}
          state={schedulerState}
          recommendation={recommendation}
          loading={loading}
          error={error}
          lastRecommendationAt={lastRecommendationAt}
          onStateChange={(patch) =>
            setSchedulerState((prev) => {
              if (!prev) {
                return prev;
              }
              return { ...prev, ...patch };
            })
          }
          onRequestRecommendation={() => {
            if (schedulerState) {
              void runRecommendation(schedulerState);
            }
          }}
        />
      ) : (
        <ForecastsTab points={forecastPoints} />
      )}
    </div>
  );
}
