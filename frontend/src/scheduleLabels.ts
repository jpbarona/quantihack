import type { MetaResponse } from "./types";

export function formatScheduleTimestamp(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) {
    return iso;
  }
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export function labelAtForecastStart(meta: MetaResponse, startIndex: number): string {
  const list = meta.available_timestamps;
  if (!meta.has_timestamp || !list || startIndex < 0 || startIndex >= list.length) {
    return `Step ${startIndex}`;
  }
  return `${formatScheduleTimestamp(list[startIndex])} · step ${startIndex}`;
}

export function labelAtRow(meta: MetaResponse, rowIndex: number): string {
  const list = meta.available_timestamps;
  const off = rowIndex - meta.input_window;
  if (!meta.has_timestamp || !list || off < 0 || off >= list.length) {
    return `Step ${rowIndex}`;
  }
  return `${formatScheduleTimestamp(list[off])} · step ${rowIndex}`;
}
