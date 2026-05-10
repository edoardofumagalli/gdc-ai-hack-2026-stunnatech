import { useEffect, useMemo, useRef, useState } from "react";
import { fetchStatus, getRuntimeConfig } from "./api/statusClient.js";
import { normalizeDashboard } from "./domain/dashboard.js";
import EmergencyHeatmap from "./components/EmergencyHeatmap.jsx";
import ExitList from "./components/ExitList.jsx";
import Header from "./components/Header.jsx";
import MetricsGrid from "./components/MetricsGrid.jsx";
import StatusSummary from "./components/StatusSummary.jsx";

function formatClock(date) {
  return date.toLocaleTimeString("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function App() {
  const config = useMemo(() => getRuntimeConfig(), []);
  const [dashboard, setDashboard] = useState(null);
  const [connection, setConnection] = useState("connecting");
  const [lastError, setLastError] = useState("");
  const [now, setNow] = useState(() => new Date());
  const requestRef = useRef(null);
  const lastAttemptedAudioKeyRef = useRef("");
  const lastPlayedAudioKeyRef = useRef("");
  const activeAudioRef = useRef(null);
  const dashboardRef = useRef(null);
  const pendingAudioRef = useRef(null);

  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    dashboardRef.current = dashboard;
  }, [dashboard]);

  useEffect(() => {
    function armAudio() {
      primeAudioPlayback().finally(() => {
        const audio = pendingAudioRef.current || getEmergencyAudio(dashboardRef.current);
        if (audio && audio.key !== lastPlayedAudioKeyRef.current) {
          attemptAudioPlayback(audio, {
            activeAudioRef,
            pendingAudioRef,
            lastPlayedAudioKeyRef,
            setLastError,
          });
        }
      });
    }

    window.addEventListener("pointerdown", armAudio, true);
    window.addEventListener("keydown", armAudio, true);

    return () => {
      window.removeEventListener("pointerdown", armAudio, true);
      window.removeEventListener("keydown", armAudio, true);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    let timer = null;

    async function load() {
      requestRef.current?.abort();
      const controller = new AbortController();
      requestRef.current = controller;

      try {
        const payload = await fetchStatus({
          statusUrl: config.statusUrl,
          timeoutMs: config.timeoutMs,
          signal: controller.signal,
        });

        if (!cancelled) {
          setDashboard(normalizeDashboard(payload));
          setConnection("api");
          setLastError("");
        }
      } catch (error) {
        if (!cancelled && error.name !== "AbortError") {
          setConnection("offline");
          setLastError(error.message || "Status API unavailable");
        }
      } finally {
        if (!cancelled) {
          timer = window.setTimeout(load, config.pollMs);
        }
      }
    }

    load();

    return () => {
      cancelled = true;
      requestRef.current?.abort();
      window.clearTimeout(timer);
    };
  }, [config]);

  useEffect(() => {
    const audio = getEmergencyAudio(dashboard);
    if (!audio) return;
    if (
      audio.key === lastPlayedAudioKeyRef.current ||
      audio.key === lastAttemptedAudioKeyRef.current
    ) {
      return;
    }

    lastAttemptedAudioKeyRef.current = audio.key;
    attemptAudioPlayback(audio, {
      activeAudioRef,
      pendingAudioRef,
      lastPlayedAudioKeyRef,
      setLastError,
    });
  }, [dashboard]);

  const clock = formatClock(now);

  if (!dashboard) {
    return (
      <main className="panel loading">
        <div className="panel-content">
          <Header room={{ name: "Waiting for backend" }} time={clock} />
          <section className="status-summary">
            <h1>WAITING</h1>
            <p>Waiting for a valid status payload from {config.statusUrl}.</p>
          </section>
          <footer className="footer">
            <div className={`connection ${connection}`}>
              <span className="connection-dot" />
              <span>
                {connection === "offline" && "API Offline"}
                {connection === "connecting" && "Connecting API"}
              </span>
            </div>
          </footer>
          {lastError && (
            <div className="api-toast" role="status">
              {lastError}
            </div>
          )}
        </div>
      </main>
    );
  }

  const isEmergency =
    dashboard.mode === "emergency" || dashboard.status.className === "emergency";

  return (
    <>
      <main className={`panel ${dashboard.status.className}`}>
        <div className="panel-content">
          <Header room={dashboard.room} time={clock} />
          <StatusSummary status={dashboard.status} />
          <MetricsGrid metrics={dashboard.metrics} />
          {isEmergency && <EmergencyHeatmap heatmap={dashboard.heatmap} />}
          {!isEmergency && (
            <section className="exits-section" aria-label="Exit status">
              <div className="section-label">Exit Status</div>
              <ExitList exits={dashboard.exits} />
            </section>
          )}
          <footer className="footer">
            <div className={`connection ${connection}`}>
              <span className="connection-dot" />
              <span>
                {connection === "api" && `Live API - ${dashboard.room.deviceId}`}
                {connection === "connecting" && `Connecting API - ${dashboard.room.deviceId}`}
                {connection === "offline" && `API Offline - ${dashboard.room.deviceId}`}
              </span>
            </div>
          </footer>
        </div>
      </main>

      {lastError && (
        <div className="api-toast" role="status">
          {lastError}
        </div>
      )}
    </>
  );
}

function attemptAudioPlayback(audio, refs) {
  const {
    activeAudioRef,
    pendingAudioRef,
    lastPlayedAudioKeyRef,
    setLastError,
  } = refs;

  activeAudioRef.current?.pause();
  return playAudioSequence(audio.sequence, audio.pauseMs, activeAudioRef)
    .then(() => {
      lastPlayedAudioKeyRef.current = audio.key;
      pendingAudioRef.current = null;
      setLastError("");
    })
    .catch((error) => {
      console.warn("Emergency audio playback blocked or failed", error);
      pendingAudioRef.current = audio;
    });
}

let audioContext = null;

function primeAudioPlayback() {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) return Promise.resolve();

  audioContext ||= new AudioContextClass();
  if (audioContext.state === "running") return Promise.resolve();
  return audioContext.resume();
}

function getEmergencyAudio(dashboard) {
  const evacuation = dashboard?.evacuation;
  const alert = dashboard?.alerts?.[0];
  const sequence =
    evacuation?.audioSequence?.length > 0
      ? evacuation.audioSequence
      : alert?.audioSequence?.length > 0
        ? alert.audioSequence
        : [evacuation?.audioUrl || alert?.audioUrl].filter(Boolean);
  const pauseMs = evacuation?.audioPauseMs || alert?.audioPauseMs || 0;
  if (sequence.length === 0) return null;

  return {
    sequence,
    pauseMs,
    key: `${sequence.join("|")}::${pauseMs}`,
  };
}

async function playAudioSequence(sequence, pauseMs, activeAudioRef) {
  for (const url of sequence) {
    const audio = new Audio(url);
    activeAudioRef.current = audio;
    await playAudio(audio);
    if (pauseMs > 0) {
      await wait(pauseMs);
    }
  }
}

function playAudio(audio) {
  return new Promise((resolve, reject) => {
    audio.addEventListener("ended", resolve, { once: true });
    audio.addEventListener("error", () => reject(new Error("Audio playback failed")), { once: true });
    audio.play().catch(reject);
  });
}

function wait(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}
