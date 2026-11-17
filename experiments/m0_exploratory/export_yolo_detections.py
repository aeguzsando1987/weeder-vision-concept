"""Proveedor aprendido de M0-P: exporta mascaras YOLO de un video (entorno separado).

Se ejecuta con el entorno aislado del experimento de segmentacion (ultralytics/torch
fuera de `uv.lock` principal), en GPU si existe:

    experiments/moringa_seg_probe/.venv/bin/python \\
        experiments/m0_exploratory/export_yolo_detections.py \\
        --video resources/moringa_sin_ai.mp4 \\
        --weights .context/Moringa_Train_Annotations/Train/weights/best.pt \\
        --output experiments/m0_exploratory/cache/moringa_sin_ai_yolo.json

Escribe la exportacion versionada (esquema 1 de `m0p.observations`) con mascaras
poligonales en `image_px`, objetivo `canopy_proxy` (los pesos segmentan copa, no
contacto tallo-suelo), y hashes de video, pesos y configuracion. El pipeline la
consume sin volver a inferir: el replay no depende de que la inferencia se repita
bit a bit en otra GPU.

`content_hash` replica la regla de `precision_weeder.core.identity` (JSON canonico,
claves ordenadas, sin NaN) para no instalar el paquete en este entorno; el
adaptador del pipeline lo verifica y rechaza cualquier discrepancia.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import cv2
import torch
import ultralytics
from ultralytics import YOLO

SCHEMA_VERSION = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: dict) -> str:
    text = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = YOLO(str(args.weights))
    capture = cv2.VideoCapture(str(args.video))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        result = model.predict(
            image, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False
        )[0]
        detections = []
        if result.masks is not None:
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            for polygon, conf, cls in zip(result.masks.xy, confs, classes, strict=True):
                points = [[round(float(u), 2), round(float(v), 2)] for u, v in polygon]
                if any(not math.isfinite(c) for p in points for c in p):
                    continue
                detections.append(
                    {
                        "class": model.names[int(cls)],
                        "confidence": round(float(conf), 4),
                        "polygon": points,
                    }
                )
        frames.append({"frame_index": index, "detections": detections})
        index += 1
    capture.release()
    config = {"imgsz": args.imgsz, "conf": args.conf, "device_kind": args.device.split(":")[0]}
    body = {
        "schema_version": SCHEMA_VERSION,
        "coordinate_domain": "image_px",
        "target": "canopy_proxy",
        "provider": {
            "kind": "learned",
            "name": "YOLOv8n-seg moringa (pesos existentes, sin reentrenar)",
            "weights_path": str(args.weights),
            "weights_sha256": sha256(args.weights),
            "config": config,
            "config_hash": canonical_hash(config),
            "versions": {
                "ultralytics": ultralytics.__version__,
                "torch": torch.__version__,
                "opencv": cv2.__version__,
                "python": platform.python_version(),
            },
            "license_risk": "AGPL-3.0 (ultralytics) y CC BY 4.0 (dataset): riesgo pendiente",
        },
        "video": {
            "sha256": sha256(args.video),
            "frame_count": index,
            "width": width,
            "height": height,
        },
        "frames": frames,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {**body, "content_hash": canonical_hash(body)}
    args.output.write_text(json.dumps(payload), encoding="utf-8")
    total = sum(len(f["detections"]) for f in frames)
    print(
        json.dumps(
            {
                "frames": index,
                "detections": total,
                "output": str(args.output),
                "content_hash": payload["content_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
