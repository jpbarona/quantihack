import type {
  ForecastResponse,
  MetaResponse,
  RecommendRequest,
  RecommendResponse
} from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json"
    },
    ...options
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

export function getMeta(): Promise<MetaResponse> {
  return request<MetaResponse>("/api/meta");
}

export function getForecast(startIndex: number): Promise<ForecastResponse> {
  return request<ForecastResponse>("/api/forecast", {
    method: "POST",
    body: JSON.stringify({ start_index: startIndex })
  });
}

export function getRecommendation(payload: RecommendRequest): Promise<RecommendResponse> {
  return request<RecommendResponse>("/api/recommend", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}
