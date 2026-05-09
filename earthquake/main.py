#!/usr/bin/env python3
"""
Rilevamento terremoti tramite IMU integrato OAK4.
Si ferma e stampa un alert appena viene rilevata una vibrazione
sostenuta sopra soglia.
"""

import math
import time
import depthai as dai

IMU_SAMPLE_RATE      = 400         # Hz
BATCH_THRESHOLD      = 20          # campioni per batch (~50ms a 400Hz)
EARTHQUAKE_THRESHOLD = 0.3 * 9.81  # m/s²  (~0.5g)
EARTHQUAKE_DURATION  = 0.1         # secondi di vibrazione sostenuta prima di allarmare


class EarthquakeDetector:
    def __init__(self, threshold: float = EARTHQUAKE_THRESHOLD,
                 min_duration_s: float = EARTHQUAKE_DURATION):
        self.threshold    = threshold
        self.min_duration = min_duration_s
        self._above_since = None
        self._alerted     = False

    def update(self, ax: float, ay: float, az: float) -> bool:
        magnitude = math.sqrt(ax**2 + ay**2 + az**2)
        vibration = abs(magnitude - 9.81)

        now = time.monotonic()
        if vibration >= self.threshold:
            if self._above_since is None:
                self._above_since = now
            elif not self._alerted and (now - self._above_since) >= self.min_duration:
                self._alerted = True
                return True
        else:
            self._above_since = None
            self._alerted     = False

        return False


def run():
    eq = EarthquakeDetector()

    with dai.Pipeline() as pipeline:
        imu = pipeline.create(dai.node.IMU)
        imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, IMU_SAMPLE_RATE)
        imu.setBatchReportThreshold(BATCH_THRESHOLD)
        imu.setMaxBatchReports(10)

        qImu = imu.out.createOutputQueue()

        pipeline.start()
        print("In ascolto... (Ctrl+C per uscire)")

        while pipeline.isRunning():
            imuData = qImu.get()
            for packet in imuData.packets:
                acc       = packet.acceleroMeter
                magnitude = math.sqrt(acc.x**2 + acc.y**2 + acc.z**2)
                vibration = abs(magnitude - 9.81)

                print(f"vibrazione: {vibration:.3f} m/s²", end="\r")

                if eq.update(acc.x, acc.y, acc.z):
                    print(f"\n⚠️  TERREMOTO RILEVATO! vibrazione={vibration:.2f} m/s²")
                    pipeline.stop()
                    return


if __name__ == "__main__":
    run()