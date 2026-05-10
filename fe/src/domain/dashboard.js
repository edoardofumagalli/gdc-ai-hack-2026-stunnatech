const modeCopy = {
  safe: {
    className: "safe",
    word: "SAFE",
    description: "All exits are clear. The room is operating normally.",
  },
  caution: {
    className: "caution",
    word: "ALERT",
    description: "Objects should not be left within the exit doors.",
  },
  danger: {
    className: "danger",
    word: "HAZARD",
    description: "Exit door is being blocked by objects, please unblock it for safety reasons.",
  },
  alarm: {
    className: "danger",
    word: "ALARM",
    description: "Room occupancy is above capacity. Start crowd reduction immediately.",
  },
  emergency: {
    className: "emergency",
    word: "EVACUATE",
    description: "Leave the room immediately.",
  },
};

function asNumber(value, fallback = 0) {
  if (value === null || value === undefined || value === "") return fallback;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function asPercent(value, fallback = 0) {
  const parsed = asNumber(value, fallback);
  return Math.min(100, Math.max(0, parsed));
}

function normalizeExitStatus(exit) {
  const raw = String(exit.status || "")
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, "_");

  if (["triggered", "blocked", "danger", "closed", "occluded"].includes(raw)) return "triggered";
  if (["occupied_pending", "pending", "occupied", "caution", "warning", "reduced", "partial"].includes(raw)) {
    return "occupied_pending";
  }
  if (["clear", "safe", "open", "ok"].includes(raw)) return "clear";

  const occupancy = asPercent(exit.occupancy ?? exit.occupancyPercent ?? exit.occupancy_percentage, 0);
  const occupancyThreshold = asPercent(exit.occupancyThreshold ?? exit.occupancy_threshold, 100);
  return occupancy >= occupancyThreshold ? "occupied_pending" : "clear";
}

function normalizeExits(rawExits = []) {
  return rawExits.map((exit, index) => {
    return {
      id: String(exit.id ?? exit.code ?? index + 1),
      name: String(exit.name ?? `Exit ${index + 1}`),
      type: String(exit.type ?? "Exit"),
      status: normalizeExitStatus(exit),
      occupancy: asPercent(exit.occupancy ?? exit.occupancyPercent ?? exit.occupancy_percentage, 0),
      occupancyThreshold: asPercent(exit.occupancyThreshold ?? exit.occupancy_threshold, 100),
    };
  });
}

function normalizeMode(raw) {
  const explicitStatus = typeof raw.status === "string" ? raw.status : "";
  const explicit = String(raw.state ?? raw.mode ?? raw.panelState ?? explicitStatus).toLowerCase();
  const emergencyActive = raw.emergency === true || raw.emergency?.active === true || raw.evacuation?.active === true;

  if (["emergency", "emerg", "evacuate", "evacuation"].includes(explicit) || emergencyActive) return "emergency";
  if (["alarm"].includes(explicit)) return "alarm";
  if (["danger", "blocked", "triggered"].includes(explicit)) return "danger";
  if (["caution", "warning", "reduced", "occupied_pending"].includes(explicit)) return "caution";
  if (["safe", "clear", "ok"].includes(explicit)) return "safe";

  throw new Error("Status API payload must include a valid state: safe, caution, alarm, danger, or emergency");
}

function modeWithPeopleThreshold(baseMode, people) {
  if (baseMode === "emergency") return "emergency";
  const peopleState = peopleCapacityState(people);
  if (peopleState === "danger") return "danger";
  if (peopleState === "caution" && baseMode === "safe") return "caution";
  return baseMode;
}

function peopleCapacityState(people) {
  const capacity = asNumber(people?.capacity, 0);
  if (capacity <= 0) return "safe";

  const current = asNumber(people?.current, 0);
  if (current > capacity) return "danger";
  if (current > capacity * 0.9) return "caution";
  return "safe";
}

function normalizeRoom(raw) {
  const room = raw.room || {};

  return {
    name: String(room.name ?? raw.roomName ?? "Aula 4"),
    deviceId: String(room.deviceId ?? room.device_id ?? raw.deviceId ?? "OAK-4D"),
    capacity: asNumber(room.capacity, 100),
  };
}

function normalizePeople(raw, room) {
  const people = raw.people;

  if (typeof people === "number") {
    return { current: people, capacity: room.capacity };
  }

  return {
    current: asNumber(people?.current ?? people?.count ?? people?.inRoom ?? raw.peopleCount, 0),
    capacity: room.capacity,
  };
}

function buildDerivedAlert(exits, mode) {
  if (mode === "safe") return null;

  const triggered = exits.find((exit) => exit.status === "triggered");
  if (triggered) {
    return {
      severity: "danger",
      title: `${triggered.name} triggered`,
      description: "Exit door is being blocked by objects, please unblock it for safety reasons.",
    };
  }

  const pending = exits.find((exit) => exit.status === "occupied_pending");
  if (pending) {
    return {
      severity: "caution",
      title: `${pending.name} occupied pending`,
      description: "Objects should not be left within the exit doors.",
    };
  }

  return null;
}

function normalizeAlerts(raw, exits, mode) {
  const alerts = Array.isArray(raw.alerts) ? raw.alerts : raw.alert ? [raw.alert] : [];

  if (alerts.length > 0) {
    return alerts.map((alert) => ({
      severity: String(alert.severity ?? alert.level ?? mode).toLowerCase(),
      title: String(alert.title ?? alert.name ?? "Exit alert"),
      description: String(alert.description ?? alert.message ?? ""),
      audioUrl: alert.audioUrl ?? alert.audio_url ?? null,
      audioSequence: normalizeAudioSequence(alert.audioSequence ?? alert.audio_sequence),
      audioPauseMs: asNumber(alert.audioPauseMs ?? alert.audio_pause_ms, 0),
    }));
  }

  const derived = buildDerivedAlert(exits, mode);
  return derived ? [derived] : [];
}

function directionArrow(value, fallback = "←") {
  const raw = String(value || "").toLowerCase();
  if (["right", "r", "east", "e"].includes(raw)) return "→";
  if (["up", "north", "n"].includes(raw)) return "↑";
  if (["down", "south", "s"].includes(raw)) return "↓";
  if (["left", "l", "west", "w"].includes(raw)) return "←";
  return fallback;
}

function normalizeEvacuation(raw, exits) {
  const evacuation = raw.evacuation || {};
  const preferredId = String(evacuation.primaryExitId ?? evacuation.primary_exit_id ?? raw.primaryExitId ?? "");
  const primaryExit =
    exits.find((exit) => exit.id === preferredId) ||
    exits.find((exit) => exit.status === "clear") ||
    exits[0] ||
    { id: "", name: "Nearest Exit" };
  const route = String(evacuation.route ?? evacuation.path ?? "Corridor -> Scala C");

  return {
    primaryExitId: primaryExit.id,
    primaryExitName: primaryExit.name,
    route,
    arrow: evacuation.arrow || directionArrow(evacuation.direction, primaryExit.id === "R" ? "→" : "←"),
    startedAt: evacuation.startedAt ?? evacuation.started_at ?? raw.emergencyStartedAt ?? null,
    label: String(evacuation.primaryLabel ?? evacuation.label ?? `${primaryExit.name} -> ${route}`),
    audioUrl: evacuation.audioUrl ?? evacuation.audio_url ?? raw.audioUrl ?? null,
    audioSequence: normalizeAudioSequence(evacuation.audioSequence ?? evacuation.audio_sequence ?? raw.audioSequence),
    audioPauseMs: asNumber(evacuation.audioPauseMs ?? evacuation.audio_pause_ms ?? raw.audioPauseMs, 0),
  };
}

function normalizeExitTime(raw, people, exitsFree) {
  const evacuation = raw.evacuation || {};
  const explicit = asNumber(
    raw.averageExitTimeS ??
      raw.averageExitTimeSeconds ??
      raw.avgExitTimeSeconds ??
      raw.exitTime?.averageSeconds ??
      raw.exit_time?.average_seconds ??
      evacuation.averageExitTimeS ??
      evacuation.averageExitTimeSeconds,
    NaN,
  );
  if (Number.isFinite(explicit)) return Math.max(0, explicit);
  return estimateAverageExitTimeSeconds(people, exitsFree);
}

function estimateAverageExitTimeSeconds(people, exitsFree) {
  if (people.current <= 0) return 0;
  const clearExits = Math.max(1, exitsFree);
  const peoplePerSecondPerExit = 1.3;
  const averageQueueSeconds = people.current / (2 * clearExits * peoplePerSecondPerExit);
  const nominalTravelSeconds = 8;
  return Math.round(nominalTravelSeconds + averageQueueSeconds);
}

function formatExitTime(seconds) {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  return remainder === 0 ? `${minutes}m` : `${minutes}m ${remainder}s`;
}

function normalizeAudioSequence(value) {
  if (!Array.isArray(value)) return [];
  return value.map((item) => String(item || "").trim()).filter(Boolean);
}

function normalizeHeatmapFrame(frame) {
  if (!frame || typeof frame !== "object") return null;
  const width = Math.max(0, Math.round(asNumber(frame.width, 0)));
  const height = Math.max(0, Math.round(asNumber(frame.height, 0)));
  const values = Array.isArray(frame.values)
    ? frame.values.map((value) => asPercent(value, 0))
    : [];

  if (width <= 0 || height <= 0 || values.length !== width * height) {
    return null;
  }

  return {
    active: frame.active !== false,
    width,
    height,
    values,
    exit: normalizeHeatmapMarker(frame.exit ?? frame.exitLocation ?? frame.exit_location, "Exit"),
    camera: normalizeHeatmapMarker(frame.camera ?? frame.cameraLocation ?? frame.camera_location, "You are here"),
    peak: asNumber(frame.peak, 0),
    total: asNumber(frame.total, 0),
    updatedAt: frame.updatedAt ?? frame.updated_at ?? null,
  };
}

function normalizeHeatmapMarker(marker, fallbackLabel) {
  if (!marker || typeof marker !== "object") return null;
  const edge = String(marker.edge ?? "").toLowerCase();
  return {
    label: String(marker.label ?? marker.name ?? fallbackLabel),
    x: Math.max(0, Math.min(1, asNumber(marker.x, 0))),
    y: Math.max(0, Math.min(1, asNumber(marker.y, 0))),
    edge: ["top", "right", "bottom", "left"].includes(edge) ? edge : "",
  };
}

function normalizeHeatmap(raw) {
  const current = normalizeHeatmapFrame(raw.heatmap);
  if (!current) return null;

  const historySource = Array.isArray(raw.heatmapHistory)
    ? raw.heatmapHistory
    : Array.isArray(raw.heatmap?.history)
      ? raw.heatmap.history
      : [];

  return {
    ...current,
    history: historySource.map(normalizeHeatmapFrame).filter(Boolean),
  };
}

function normalizeStatus(raw, mode, alerts, people, capacityDriven = false) {
  const status = typeof raw.status === "object" ? raw.status : {};
  const copy = modeCopy[mode] || modeCopy.safe;
  const firstAlert = alerts[0];
  const isDanger = ["danger", "blocked", "triggered", "critical", "emergency"].includes(
    String(firstAlert?.severity || mode).toLowerCase(),
  );
  const capacityState = peopleCapacityState(people);
  const defaultDescription = capacityDriven ? capacityDescription(capacityState) : copy.description;

  return {
    className: copy.className,
    word: String(status.word ?? raw.word ?? raw.statusWord ?? copy.word),
    description: String(status.description ?? raw.description ?? raw.statusDescription ?? defaultDescription ?? firstAlert?.description),
    chip: mode !== "emergency" && firstAlert
      ? {
          className: isDanger ? "chip-danger" : "chip-caution",
          icon: isDanger ? "!" : "!",
          title: firstAlert.title,
          description: firstAlert.description,
        }
      : null,
  };
}

function capacityDescription(capacityState) {
  if (capacityState === "danger") {
    return "Room capacity exceeded. Reduce occupancy immediately.";
  }
  if (capacityState === "caution") {
    return "Room occupancy is near capacity. Monitor crowding.";
  }
  return "";
}

function peopleMetricClass(people) {
  const capacityState = peopleCapacityState(people);
  if (capacityState === "danger") return "bad";
  if (capacityState === "caution") return "warn";
  return "ok";
}

function exitsMetricClass(exits) {
  if (exits.length === 0) return "bad";
  if (exits.some((exit) => exit.status === "triggered")) return "bad";
  if (exits.some((exit) => exit.status === "occupied_pending")) return "warn";
  return "ok";
}

function exitTimeClass(seconds) {
  if (seconds >= 60) return "bad";
  if (seconds >= 40) return "warn";
  return "ok";
}

function buildMetrics(dashboard) {
  const totalExits = dashboard.exits.length;

  return [
    {
      id: "people",
      label: "People",
      value: `${dashboard.people.current} / ${dashboard.people.capacity}`,
      className: peopleMetricClass(dashboard.people),
    },
    {
      id: "exits",
      label: "Exits Clear",
      value: `${dashboard.exitsFree} / ${totalExits}`,
      className: exitsMetricClass(dashboard.exits),
    },
    {
      id: "avg-exit-time",
      label: "Avg Exit Time",
      value: formatExitTime(dashboard.averageExitTimeSeconds),
      className: exitTimeClass(dashboard.averageExitTimeSeconds),
    },
  ];
}

export function normalizeDashboard(payload) {
  const raw = Array.isArray(payload) ? { exits: payload } : payload || {};
  const exits = normalizeExits(raw.exits ?? raw.exitPayload ?? raw.doors ?? []);
  const baseMode = normalizeMode(raw);
  const room = normalizeRoom(raw);
  const people = normalizePeople(raw, room);
  const mode = modeWithPeopleThreshold(baseMode, people);
  const capacityState = peopleCapacityState(people);
  const capacityDriven =
    baseMode !== "emergency" &&
    (capacityState === "danger" || (capacityState === "caution" && baseMode === "safe"));
  const alerts = normalizeAlerts(raw, exits, mode);
  const status = normalizeStatus(raw, mode, alerts, people, capacityDriven);
  const evacuation = normalizeEvacuation(raw, exits);
  const exitsFree = exits.filter((exit) => exit.status === "clear").length;
  const alertCount = asNumber(raw.alertCount ?? raw.alertsCount, alerts.length);
  const averageExitTimeSeconds = normalizeExitTime(raw, people, exitsFree);
  const heatmap = mode === "emergency" ? normalizeHeatmap(raw) : null;

  const dashboard = {
    mode,
    room,
    people,
    alerts,
    exits,
    exitsFree,
    alertCount,
    averageExitTimeSeconds,
    status,
    evacuation,
    heatmap,
    updatedAt: raw.updatedAt ?? raw.timestamp ?? new Date().toISOString(),
  };

  return {
    ...dashboard,
    metrics: buildMetrics(dashboard),
  };
}
