#!/usr/bin/env python3
"""
Entrenamiento de YOLOv11m sobre el dataset Fishnet balanceado.
Pensado para GPU NVIDIA T4 (16 GB) en AWS (g4dn.xlarge/2xlarge).

Preparación del servidor:
    pip install ultralytics
    # dataset en el servidor: images/ + labels (ver README del dataset)
    # si el data.yaml trae la ruta de otra máquina, este script la corrige solo.

Uso:
    python3 train_yolo.py                          # config recomendada T4
    python3 train_yolo.py --imgsz 960 --batch 8    # mejor para objetos pequeños, mas lento
    python3 train_yolo.py --resume                 # continuar un run interrumpido

Resultados en runs/detect/<name>/: pesos (best.pt segun fitness = 0.9*mAP50-95
+ 0.1*mAP50, y last.pt), results.csv/png, matrices de confusion y curvas.
"""

import argparse
import os

import yaml
from ultralytics import YOLO


def batch_size(value):
    """Entero para tamaños fijos (y -1 = auto), float solo para fracciones de VRAM."""
    f = float(value)
    return int(f) if f.is_integer() else f


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="dataset_yolo_balanced/data.yaml")
    p.add_argument("--model", default="yolo11m.pt",
                   help="Checkpoint inicial (preentrenado COCO) o last.pt para reanudar")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=640,
                   help="640 = baseline T4; 960 mejora clases pequeñas (SKJ) ~2x mas lento")
    p.add_argument("--batch", type=batch_size, default=-1,
                   help="-1 = auto-ajuste al 60%% de la VRAM (solo CUDA), "
                        "entero = tamaño fijo, fraccion 0-1 = %% de VRAM")
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8,
                   help="Bajar a 4 en g4dn.xlarge (4 vCPU)")
    p.add_argument("--name", default="foid_yolo11m")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def resolve_data_yaml(path):
    """Si el data.yaml apunta a una ruta de otra máquina, la corrige a la
    carpeta donde vive el propio yaml y guarda una copia resuelta."""
    path = os.path.abspath(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    root = cfg.get("path", "")
    if not os.path.isdir(root):
        cfg["path"] = os.path.dirname(path)
        resolved = os.path.join(os.path.dirname(path), "data.resolved.yaml")
        with open(resolved, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"'path' del data.yaml no existe aqui; usando {resolved} "
              f"con path={cfg['path']}")
        return resolved
    return path


def main():
    args = parse_args()
    data = resolve_data_yaml(args.data)
    model = YOLO(args.model)

    results = model.train(
        data=data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        name=args.name,
        resume=args.resume,

        # convergencia
        patience=15,          # early stopping si val no mejora en 15 epocas
        cos_lr=True,          # decaimiento coseno del lr
        close_mosaic=10,      # apaga mosaic las ultimas 10 epocas (afina cajas)

        # augmentation ajustada a monitoreo electronico en cubierta:
        # los peces aparecen en cualquier orientacion -> rotacion y flip vertical;
        # iluminacion muy variable entre camaras/clima -> mas jitter de brillo.
        degrees=180.0,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.5,
        translate=0.1,
        scale=0.5,
        mosaic=1.0,

        plots=True,
        val=True,
    )

    best = os.path.join(str(results.save_dir), "weights", "best.pt")
    print(f"\nEntrenamiento terminado. Mejor checkpoint: {best}")
    print(f"Evaluar en test: python3 evaluate_yolo.py --weights {best} --split test")


if __name__ == "__main__":
    main()
