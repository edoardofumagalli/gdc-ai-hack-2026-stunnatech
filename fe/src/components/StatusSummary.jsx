export default function StatusSummary({ status }) {
  return (
    <section className="status-summary" aria-live="polite">
      <h1>{status.word}</h1>
      <p>{status.description}</p>

      {status.chip && (
        <div className={`alert-chip ${status.chip.className}`}>
          <span className="alert-chip-icon">{status.chip.icon}</span>
          <span>
            <strong>{status.chip.title}</strong>
            <small>{status.chip.description}</small>
          </span>
        </div>
      )}
    </section>
  );
}
