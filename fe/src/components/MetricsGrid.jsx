export default function MetricsGrid({ metrics }) {
  return (
    <section className="metrics" aria-label="Room metrics">
      {metrics.map((metric) => (
        <article className={`metric ${metric.className}`} key={metric.id}>
          <strong>{metric.value}</strong>
          <small>{metric.label}</small>
        </article>
      ))}
    </section>
  );
}
