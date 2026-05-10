const exitStatus = {
  clear: { className: "exit-ok", label: "CLEAR" },
  occupied_pending: { className: "exit-caution", label: "OCCUPIED" },
  triggered: { className: "exit-danger", label: "TRIGGERED" },
};

export default function ExitList({ exits }) {
  if (exits.length === 0) {
    return <div className="empty-state">No exits received from API</div>;
  }

  return (
    <div className="exit-list">
      {exits.map((exit) => {
        const status = exitStatus[exit.status] || exitStatus.clear;

        return (
          <article className={`exit-row ${status.className}`} key={exit.id}>
            <span className="exit-dot" />
            <span className="exit-info">
              <strong>{exit.name}</strong>
              <small>
                {exit.type} - Threshold {exit.occupancyThreshold}%
              </small>
            </span>
            <strong className="exit-value">{exit.occupancy}%</strong>
            <span className="exit-badge">{status.label}</span>
          </article>
        );
      })}
    </div>
  );
}
