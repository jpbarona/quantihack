import type { TimelineSegment, WindowScore } from "../types";

const COLORS: Record<string, string> = {
  Great: "#2f9e44",
  Okay: "#f08c00",
  Avoid: "#c92a2a"
};

type TimelineBandProps = {
  minIndex: number;
  maxIndex: number;
  segments: TimelineSegment[];
  recommended: WindowScore;
};

export default function TimelineBand({
  minIndex,
  maxIndex,
  segments,
  recommended
}: TimelineBandProps) {
  const span = Math.max(1, maxIndex - minIndex + 1);
  const recLeft = ((recommended.start_index - minIndex) / span) * 100;
  const recWidth = (recommended.duration_steps / span) * 100;

  return (
    <div className="timeline-wrap">
      <div className="timeline-band">
        {segments.map((segment) => {
          const left = ((segment.start_index - minIndex) / span) * 100;
          const width = ((segment.end_index - segment.start_index + 1) / span) * 100;
          return (
            <div
              className="timeline-segment"
              key={`${segment.start_index}-${segment.end_index}`}
              style={{
                left: `${left}%`,
                width: `${width}%`,
                background: COLORS[segment.label]
              }}
              title={`${segment.label}: ${segment.start_index}-${segment.end_index}`}
            />
          );
        })}
        <div
          className="timeline-highlight"
          style={{ left: `${recLeft}%`, width: `${recWidth}%` }}
          title={`Recommended: ${recommended.start_index}-${recommended.end_index}`}
        />
      </div>
      <div className="timeline-axis">
        <span>{minIndex}</span>
        <span>{maxIndex}</span>
      </div>
    </div>
  );
}
