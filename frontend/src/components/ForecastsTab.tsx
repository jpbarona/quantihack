import { useMemo, useState } from "react";
import type { ForecastPoint } from "../types";

type ForecastsTabProps = {
  points: ForecastPoint[];
};

const WIDTH = 860;
const HEIGHT = 280;
const PADDING = 28;

export default function ForecastsTab({ points }: ForecastsTabProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  const { path, circles, minValue, maxValue } = useMemo(() => {
    if (points.length === 0) {
      return { path: "", circles: [], minValue: 0, maxValue: 0 };
    }
    const values = points.map((point) => point.value);
    const minValueLocal = Math.min(...values);
    const maxValueLocal = Math.max(...values);
    const range = Math.max(1e-6, maxValueLocal - minValueLocal);
    const innerW = WIDTH - PADDING * 2;
    const innerH = HEIGHT - PADDING * 2;

    const coords = points.map((point, i) => {
      const x = PADDING + (i / Math.max(1, points.length - 1)) * innerW;
      const y = PADDING + (1 - (point.value - minValueLocal) / range) * innerH;
      return { x, y };
    });

    const pathStr = coords.map((c, i) => `${i === 0 ? "M" : "L"} ${c.x} ${c.y}`).join(" ");
    return { path: pathStr, circles: coords, minValue: minValueLocal, maxValue: maxValueLocal };
  }, [points]);

  const activePoint = hoveredIndex == null ? null : points[hoveredIndex];
  const activeCircle = hoveredIndex == null ? null : circles[hoveredIndex];

  return (
    <div className="card">
      <h3>Residual Demand Forecast</h3>
      <p className="muted">Hover over points to inspect level values.</p>
      {points.length === 0 ? (
        <div className="empty-state">No forecast data loaded.</div>
      ) : (
        <div className="chart-shell">
          <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="chart-svg" role="img" aria-label="Forecast chart">
            <rect className="forecast-chart-bg" x={0} y={0} width={WIDTH} height={HEIGHT} />
            <path className="forecast-chart-line" d={path} />
            {circles.map((circle, i) => (
              <circle
                key={`pt-${i}`}
                className={`forecast-chart-point${hoveredIndex === i ? " is-active" : ""}`}
                cx={circle.x}
                cy={circle.y}
                r={hoveredIndex === i ? 5 : 3}
                onMouseEnter={() => setHoveredIndex(i)}
                onMouseLeave={() => setHoveredIndex(null)}
              />
            ))}
            {activeCircle && (
              <g>
                <line
                  className="forecast-chart-guide"
                  x1={activeCircle.x}
                  x2={activeCircle.x}
                  y1={PADDING}
                  y2={HEIGHT - PADDING}
                  strokeDasharray="4 4"
                />
              </g>
            )}
          </svg>
          <div className="chart-meta">
            <span>Min: {minValue.toFixed(2)}</span>
            <span>Max: {maxValue.toFixed(2)}</span>
            {activePoint ? (
              <span>
                {activePoint.label}: {activePoint.value.toFixed(2)}
              </span>
            ) : (
              <span>Hover a point</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
