#!/usr/bin/env python3

"""
Crowd counting tramite DM-Count su OAK4.

Pipeline:
    RGB Camera -> DM-Count model
    Output:
        - density map
        - conteggio persone

NOTE:
    - Nessuna bounding box
    - Conteggio ottenuto dalla density map
    - Compatibile con DepthAI V3
"""

import time
from collections import deque

import cv2
import depthai as dai
import numpy as np


# =========================================================
# CONFIG
# =========================================================

FPS = 30

SMOOTH_WINDOW = 5

MODEL_NAME = "luxonis/dm-count:shb-426x240"

#
# Fattore empirico di calibrazione.
#
# 1 persona ≈ 242 raw units
#
RAW_SCALE = 242.0


# =========================================================
# UTILS
# =========================================================

def normalize_density_map(density):

    density = density.astype(np.float32)

    density -= density.min()

    if density.max() > 0:
        density /= density.max()

    density *= 255.0

    return density.astype(np.uint8)


# =========================================================
# MAIN
# =========================================================

def run():

    count_history = deque(maxlen=SMOOTH_WINDOW)

    with dai.Pipeline() as pipeline:

        # =================================================
        # CAMERA
        # =================================================

        camera = pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_A,
            sensorFps=FPS
        )

        # =================================================
        # MODELLO
        # =================================================

        modelDescription = dai.NNModelDescription()

        modelDescription.model = MODEL_NAME
        modelDescription.platform = "RVC4"

        archivePath = dai.getModelFromZoo(
            modelDescription,
            useCached=True
        )

        print("Model path:", archivePath)

        nnArchive = dai.NNArchive(archivePath)

        # =================================================
        # NEURAL NETWORK
        # =================================================

        nn = pipeline.create(dai.node.NeuralNetwork)

        nn.build(camera, nnArchive)

        # =================================================
        # QUEUES
        # =================================================

        qRgb = nn.passthrough.createOutputQueue()
        qNN  = nn.out.createOutputQueue()

        # =================================================
        # START
        # =================================================

        pipeline.start()

        startTime = time.monotonic()

        counter = 0

        while pipeline.isRunning():

            inRgb = qRgb.get()

            inNN = qNN.get()

            frame = inRgb.getCvFrame()

            # =============================================
            # DEBUG LAYERS
            # =============================================

            layerNames = inNN.getAllLayerNames()

            if counter == 0:
                print("Layer names:", layerNames)

            # =============================================
            # READ OUTPUT
            # =============================================

            tensor = inNN.getTensor("density_map")

            try:
                nnData = np.array(tensor.data)

            except Exception:
                nnData = np.array(tensor)

            # =============================================
            # DEBUG
            # =============================================

            if counter == 0:

                print("Tensor shape:", nnData.shape)

                flat_debug = nnData.flatten()

                print("Tensor length:", len(flat_debug))

            # =============================================
            # RESHAPE
            # =============================================

            #
            # Se già 2D
            #

            if len(nnData.shape) == 2:

                density_map = nnData

            #
            # Se 3D/4D
            #

            elif len(nnData.shape) >= 3:

                density_map = nnData.squeeze()

            #
            # Fallback
            #

            else:

                flat = nnData.flatten()

                expected_h = 30
                expected_w = 53

                if len(flat) != expected_h * expected_w:

                    print(
                        f"\nUnexpected tensor size: {len(flat)}"
                    )

                    continue

                density_map = flat.reshape(
                    (expected_h, expected_w)
                )

            # =============================================
            # RAW VALUES
            # =============================================

            raw_sum = float(density_map.sum())

            #
            # Conteggio calibrato
            #

            people_count = raw_sum / RAW_SCALE

            count_history.append(people_count)

            smooth_count = (
                sum(count_history) / len(count_history)
                if count_history else 0
            )

            # =============================================
            # DEBUG STATS
            # =============================================

            if counter % 60 == 0:

                print(
                    "\n"
                    f"min={density_map.min():.3f} "
                    f"max={density_map.max():.3f} "
                    f"mean={density_map.mean():.3f} "
                    f"sum={raw_sum:.3f}"
                )

            # =============================================
            # VISUALIZATION
            # =============================================

            density_vis = normalize_density_map(
                density_map
            )

            density_vis = cv2.applyColorMap(
                density_vis,
                cv2.COLORMAP_JET
            )

            density_vis = cv2.resize(
                density_vis,
                (frame.shape[1], frame.shape[0])
            )

            overlay = cv2.addWeighted(
                frame,
                0.6,
                density_vis,
                0.4,
                0
            )

            # =============================================
            # TEXT
            # =============================================

            cv2.putText(
                overlay,
                f"Persone: {smooth_count:.1f}",
                (20, 50),
                cv2.FONT_HERSHEY_TRIPLEX,
                1.5,
                (0, 0, 255),
                3
            )

            counter += 1

            fps = counter / (
                time.monotonic() - startTime
            )

            cv2.putText(
                overlay,
                f"FPS: {fps:.1f}",
                (20, overlay.shape[0] - 20),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.7,
                (255, 255, 255),
                1
            )

            cv2.imshow(
                "Crowd Counting",
                overlay
            )

            print(
                f"Persone stimate: {smooth_count:.1f}",
                end="\r"
            )

            if cv2.waitKey(1) == ord("q"):
                break

        pipeline.stop()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()