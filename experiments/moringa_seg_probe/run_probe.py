"""Prueba exploratoria del modelo de segmentacion de moringa sobre `moringa_sin_ai.mp4`.

NO forma parte del producto ni de ninguna etapa del plan: es un experimento aislado
(entorno propio `.venv`, fuera de `uv.lock`) para decidir si el modelo sirve de
proveedor de observaciones en E6 / el MVP. No modifica datos ni anotaciones de E0.

Que hace:
1. Verifica el SHA-256 del video y de los pesos.
2. Infiere en los 972 cuadros (GPU si hay CUDA, si no CPU; imgsz 640, conf >= 0.25)
   y guarda por deteccion:
   caja, confianza, area y centroide de la mascara (pixeles del cuadro original).
3. Contrasta con la tabla manual de E0: para cada punto `stem_axis_proxy` anotado,
   si cae dentro de alguna mascara del mismo cuadro; para cada planta anotada como
   oculta, si hay alguna deteccion en el cuadro en su linea (izquierda/derecha).
   Es un contraste indicativo, no una evaluacion formal de precision/recall.
4. Dibuja las mascaras en las 24 muestras de E0 y genera un video anotado.

    experiments/moringa_seg_probe/.venv/bin/python experiments/moringa_seg_probe/run_probe.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
VIDEO = ROOT / "resources/moringa_sin_ai.mp4"
WEIGHTS = ROOT / ".context/Moringa_Train_Annotations/Train/weights/best.pt"
OBSERVATIONS = ROOT / "experiments/phase0/moringa_sin_ai/stem_observations.csv"
SAMPLES = ROOT / "experiments/phase0/moringa_sin_ai/samples.csv"
VIDEO_SHA256 = "63f43e111c3a8031aafcdc738eef32ed47f2e12b877e87fcce0817a226e9e083"
CONF = 0.25
IMGSZ = 640
MID_U = 960  # separa linea izquierda/derecha en el cuadro 1920 px de ancho
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(4)  # no acaparar la CPU mientras se decodifica el video


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    video_hash, weights_hash = sha256(VIDEO), sha256(WEIGHTS)
    if video_hash != VIDEO_SHA256:
        raise SystemExit(f"video distinto del caracterizado en E0: {video_hash}")
    torch.manual_seed(0)
    model = YOLO(str(WEIGHTS))

    capture = cv2.VideoCapture(str(VIDEO))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(OUT / "moringa_sin_ai_detections.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width // 2, height // 2),
    )
    sample_frames = {int(r["frame_index"]) for r in csv.DictReader(SAMPLES.open())}
    (OUT / "samples").mkdir(exist_ok=True)

    rows: list[dict] = []
    # Poligonos de mascara por cuadro (pixeles originales); sin rasterizar a 1920x1080.
    masks_by_frame: dict[int, list[np.ndarray]] = {}
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result = model.predict(frame, imgsz=IMGSZ, conf=CONF, device=DEVICE, verbose=False)[0]
        frame_masks = []
        overlay = frame.copy()
        fill = frame.copy()
        if result.masks is not None:
            polygons = result.masks.xy  # coordenadas en el cuadro original
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for det, (box, conf, polygon) in enumerate(zip(boxes, confs, polygons, strict=True)):
                contour = polygon.astype(np.float32).reshape(-1, 1, 2)
                moments = cv2.moments(contour) if len(polygon) >= 3 else {"m00": 0.0}
                area = float(moments["m00"])
                rows.append(
                    {
                        "frame_index": index,
                        "detection": det,
                        "confidence": round(float(conf), 4),
                        "x1": round(float(box[0]), 1),
                        "y1": round(float(box[1]), 1),
                        "x2": round(float(box[2]), 1),
                        "y2": round(float(box[3]), 1),
                        "mask_area_px": round(area, 1),
                        "centroid_u": round(moments["m10"] / area, 1) if area > 0 else "",
                        "centroid_v": round(moments["m01"] / area, 1) if area > 0 else "",
                    }
                )
                if len(polygon) >= 3:
                    frame_masks.append(contour)
                    cv2.fillPoly(fill, [polygon.astype(np.int32)], (255, 80, 0))
                cv2.rectangle(
                    overlay, tuple(box[:2].astype(int)), tuple(box[2:].astype(int)), (255, 80, 0), 3
                )
                cv2.putText(
                    overlay, f"{conf:.2f}", (int(box[0]), int(box[1]) - 8), 0, 1.2, (255, 255, 0), 3
                )
        overlay = cv2.addWeighted(fill, 0.45, overlay, 0.55, 0.0)
        masks_by_frame[index] = frame_masks
        cv2.putText(
            overlay,
            f"frame {index}  detecciones {len(frame_masks)}  (modelo exploratorio, conf>={CONF})",
            (20, 50), 0, 1.3, (0, 255, 255), 3,
        )
        writer.write(cv2.resize(overlay, (width // 2, height // 2)))
        if index in sample_frames:
            sample_path = OUT / "samples" / f"frame_{index:06d}.jpg"
            cv2.imwrite(str(sample_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
        index += 1
    writer.release()
    frames = index

    with (OUT / "detections.csv").open("w", newline="") as handle:
        fields = ["frame_index", "detection", "confidence", "x1", "y1", "x2", "y2",
                  "mask_area_px", "centroid_u", "centroid_v"]
        writer_csv = csv.DictWriter(handle, fieldnames=fields)
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    # --- contraste con la tabla manual de E0 --------------------------------------
    manual = []
    for obs in csv.DictReader(OBSERVATIONS.open()):
        frame = int(obs["frame_index"])
        masks = masks_by_frame.get(frame, [])
        entry = {"event_id": obs["event_id"], "plant_id": obs["plant_id"], "frame": frame,
                 "target": obs["observed_target"],
                 "line": obs["notes"].split()[-1] if "linea" in obs["notes"] else ""}
        if obs["observed_target"] != "none":
            u, v = float(obs["u_px"]), float(obs["v_px"])
            r = float(obs["uncertainty_px"])
            # Distancia con signo al contorno: > 0 dentro, < 0 fuera (pixeles).
            distances = [cv2.pointPolygonTest(m, (u, v), True) for m in masks]
            inside = any(d >= 0 for d in distances)
            # tolerancia de lectura: mascara a menos del radio de incertidumbre del punto
            near = any(d >= -r for d in distances)
            entry.update({"u_px": u, "v_px": v, "point_inside_mask": bool(inside),
                          "mask_within_uncertainty": near})
        else:
            notes = obs["notes"]
            side = "left" if "izquierda" in notes else ("right" if "derecha" in notes else None)
            hits = [
                r_ for r_ in rows
                if r_["frame_index"] == frame and r_["centroid_u"] != ""
                and ((r_["centroid_u"] < MID_U) == (side == "left"))
            ] if side else []
            entry.update({"side": side, "detections_on_that_line": len(hits)})
        manual.append(entry)

    counts = np.bincount([r["frame_index"] for r in rows], minlength=frames)
    confidences = np.array([r["confidence"] for r in rows]) if rows else np.array([0.0])
    located = [m for m in manual if m["target"] != "none"]
    hidden = [m for m in manual if m["target"] == "none"]
    summary = {
        "scope": "experimento exploratorio; no es evaluacion formal ni parte del producto",
        "video_sha256": video_hash,
        "weights_sha256": weights_hash,
        "weights_path": str(WEIGHTS.relative_to(ROOT)),
        "inference": {"imgsz": IMGSZ, "conf": CONF, "device": DEVICE},
        "versions": {
            "ultralytics": ultralytics.__version__, "torch": torch.__version__,
            "opencv": cv2.__version__, "python": platform.python_version(),
        },
        "frames": frames,
        "detections_total": len(rows),
        "frames_with_detection": int((counts > 0).sum()),
        "detections_per_frame": {
            "mean": float(counts.mean()), "median": float(np.median(counts)),
            "max": int(counts.max()), "p95": float(np.percentile(counts, 95)),
        },
        "confidence": {
            "median": float(np.median(confidences)), "p10": float(np.percentile(confidences, 10)),
        },
        "manual_contrast": {
            "located_proxy_points": len(located),
            "point_inside_mask": sum(m["point_inside_mask"] for m in located),
            "mask_within_uncertainty": sum(m["mask_within_uncertainty"] for m in located),
            "hidden_base_rows": len(hidden),
            "hidden_rows_with_detection_on_line": sum(
                m["detections_on_that_line"] > 0 for m in hidden
            ),
            "detail": manual,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "manual_contrast"}, indent=2))
    contrast = {k: v for k, v in summary["manual_contrast"].items() if k != "detail"}
    print(json.dumps(contrast, indent=2))


if __name__ == "__main__":
    main()
