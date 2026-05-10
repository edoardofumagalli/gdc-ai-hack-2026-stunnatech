function heatColor(value) {
  const t = Math.max(0, Math.min(1, value / 100));
  if (t <= 0.02) return "rgba(247, 246, 242, 0.045)";

  const r = Math.round(85 + 170 * t);
  const g = Math.round(108 - 56 * t);
  const b = Math.round(255 - 185 * t);
  return `rgba(${r}, ${g}, ${b}, ${0.2 + 0.78 * t})`;
}

function formatUpdatedAt(value) {
  if (!value) return "Live";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Live";
  return date.toLocaleTimeString("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export default function EmergencyHeatmap({ heatmap }) {
  const history = heatmap?.history || [];
  const maxTotal = Math.max(1, ...history.map((frame) => frame.total || 0));
  const exit = heatmap?.exit;
  const camera = heatmap?.camera;

  return (
    <section className="emergency-heatmap" aria-label="Crowd heatmap">
      <div className="heatmap-header">
        <span>
          <strong>Crowd Heatmap</strong>
          <small>{heatmap ? formatUpdatedAt(heatmap.updatedAt) : "Waiting for camera data"}</small>
        </span>
        <span className="heatmap-legend" aria-hidden="true">
          <span className="heatmap-gradient" />
          <span className="heatmap-legend-labels">
            <span>Low</span>
            <span>Medium</span>
            <span>High</span>
          </span>
        </span>
      </div>

      {heatmap ? (
        <>
          <div className="heatmap-stage">
            <div
              className="heatmap-grid"
              style={{ gridTemplateColumns: `repeat(${heatmap.width}, minmax(0, 1fr))` }}
            >
              {heatmap.values.map((value, index) => (
              <span
                className="heatmap-cell"
                key={`${heatmap.updatedAt || "heatmap"}-${index}`}
                style={{ backgroundColor: heatColor(value) }}
                title={`${Math.round(value)}%`}
              />
              ))}
            </div>
            {exit && (
              <span
                className={`heatmap-marker exit edge-${exit.edge || "top"}`}
                style={{
                  left: `${exit.x * 100}%`,
                  top: `${exit.y * 100}%`,
                }}
                title={exit.label}
              >
                <span>{exit.label}</span>
              </span>
            )}
            {camera && (
              <span
                className={`heatmap-marker camera edge-${camera.edge || "bottom"}`}
                style={{
                  left: `${camera.x * 100}%`,
                  top: `${camera.y * 100}%`,
                }}
                title={camera.label}
              >
                <span>{camera.label}</span>
              </span>
            )}
          </div>

          <div className="heatmap-evolution" aria-label="Heatmap evolution">
            <small>Evolution</small>
            <div className="heatmap-bars">
              {history.length > 0 ? (
                history.map((frame, index) => (
                  <span
                    className="heatmap-bar"
                    key={`${frame.updatedAt || "history"}-${index}`}
                    style={{ height: `${Math.max(8, (frame.total / maxTotal) * 100)}%` }}
                    title={`${formatUpdatedAt(frame.updatedAt)} - ${frame.total.toFixed(1)}`}
                  />
                ))
              ) : (
                <span className="heatmap-empty">Collecting evolution</span>
              )}
            </div>
          </div>
        </>
      ) : (
        <div className="heatmap-placeholder">People density map unavailable</div>
      )}
    </section>
  );
}
