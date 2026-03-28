export type IndexRange = {
  min: number;
  max: number;
  count: number;
};

export type MetaResponse = {
  data_path: string;
  input_window: number;
  forecast_horizon: number;
  feature_columns: string[];
  target_column: string;
  available_start_indices: IndexRange;
  forecast_index_range: IndexRange;
  has_timestamp: boolean;
  timestamp_column: string | null;
  available_timestamps: string[] | null;
};

export type ForecastPoint = {
  horizon_step: number;
  label: string;
  forecast_index: number;
  timestamp: string | null;
  value: number;
};

export type ForecastResponse = {
  start_index: number;
  horizon: number;
  points: ForecastPoint[];
};

export type WindowScore = {
  start_index: number;
  end_index: number;
  duration_steps: number;
  mean_residual_demand: number;
  label: "Great" | "Okay" | "Avoid";
};

export type TimelineSegment = {
  start_index: number;
  end_index: number;
  label: "Great" | "Okay" | "Avoid";
};

export type RecommendResponse = {
  current_index: number;
  forecast_start_index: number;
  duration_hours: number;
  duration_steps: number;
  latest_finish_index: number;
  recommendation: WindowScore;
  run_now: WindowScore;
  timeline: TimelineSegment[];
  points: ForecastPoint[];
  details: Record<string, string | number | null>;
};

export type RecommendRequest = {
  current_index: number;
  duration_hours: number;
  latest_finish_index: number;
  gpu_preset: string;
};
