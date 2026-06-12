#!/usr/bin/env python3
"""
Genera un dataset limpio en formato YOLO (Ultralytics, compatible con YOLOv11)
a partir de las imágenes de ./images y las etiquetas de
./foid_labels_v100/foid_labels_v100.csv (Fishnet Open Images v1.0.0).

- Usa las clases gruesas `label_l2` (excluyendo NoF, WATER y OIL).
- Respeta el split train/val/test que trae el CSV.
- Crea symlinks a las imágenes originales (no duplica espacio en disco).
- Convierte las cajas de píxeles (x_min, x_max, y_min, y_max) al formato YOLO
  normalizado (class x_center y_center width height), con clipping a los
  bordes de la imagen.

Uso:
    python3 make_yolo_dataset.py [--out dataset_yolo] [--copy]
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

# Clases que no representan objetos detectables: sus cajas se descartan y las
# imágenes que solo las contienen quedan como "background" (label vacío).
EXCLUDED_LABELS = {"NoF", "WATER", "OIL"}

CLASSES = [
    "ALB", "BET", "BILL", "DOL", "HUMAN", "LAG",
    "OTH", "PLS", "SHARK", "SKJ", "TUNA", "YFT",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}


def parse_args():
    base = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", default=os.path.join(base, "images"),
                   help="Carpeta con las imágenes originales")
    p.add_argument("--labels-csv",
                   default=os.path.join(base, "foid_labels_v100", "foid_labels_v100.csv"),
                   help="CSV de etiquetas de Fishnet")
    p.add_argument("--out", default=os.path.join(base, "dataset_yolo"),
                   help="Carpeta de salida del dataset limpio")
    p.add_argument("--copy", action="store_true",
                   help="Copiar las imágenes en lugar de crear symlinks")
    p.add_argument("--workers", type=int, default=8,
                   help="Hilos para leer dimensiones de imágenes")
    return p.parse_args()


def load_annotations(csv_path):
    """Agrupa las cajas por imagen y determina el split de cada una."""
    boxes = defaultdict(list)   # img_id -> [(label, x_min, x_max, y_min, y_max)]
    splits = {}                 # img_id -> 'train' | 'val' | 'test'
    skipped_rows = 0
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            img_id = row["img_id"]
            if img_id not in splits:
                if row["train"] == "True":
                    splits[img_id] = "train"
                elif row["val"] == "True":
                    splits[img_id] = "val"
                elif row["test"] == "True":
                    splits[img_id] = "test"
                else:
                    skipped_rows += 1
                    continue
            label = row["label_l2"]
            if label in EXCLUDED_LABELS:
                continue
            try:
                box = (label, float(row["x_min"]), float(row["x_max"]),
                       float(row["y_min"]), float(row["y_max"]))
            except ValueError:
                skipped_rows += 1
                continue
            boxes[img_id].append(box)
    return boxes, splits, skipped_rows


def yolo_lines(img_boxes, width, height):
    """Convierte cajas en píxeles a líneas YOLO normalizadas y con clipping."""
    lines = []
    for label, x_min, x_max, y_min, y_max in img_boxes:
        x_min, x_max = max(0.0, min(x_min, x_max)), min(float(width), max(x_min, x_max))
        y_min, y_max = max(0.0, min(y_min, y_max)), min(float(height), max(y_min, y_max))
        bw, bh = x_max - x_min, y_max - y_min
        if bw <= 1 or bh <= 1:  # caja degenerada tras el clipping
            continue
        xc = (x_min + x_max) / 2 / width
        yc = (y_min + y_max) / 2 / height
        lines.append(f"{CLASS_TO_ID[label]} {xc:.6f} {yc:.6f} {bw / width:.6f} {bh / height:.6f}")
    return lines


def process_image(task):
    """Crea el symlink/copia de la imagen y escribe su archivo de label."""
    img_id, split, img_boxes, args = task
    src = os.path.join(args.images, img_id + ".jpg")
    if not os.path.exists(src):
        return ("missing", img_id)
    try:
        with Image.open(src) as im:
            width, height = im.size
    except Exception:
        return ("corrupt", img_id)

    lines = yolo_lines(img_boxes, width, height)

    dst_img = os.path.join(args.out, "images", split, img_id + ".jpg")
    if not os.path.exists(dst_img):
        if args.copy:
            import shutil
            shutil.copy2(src, dst_img)
        else:
            os.symlink(src, dst_img)

    label_path = os.path.join(args.out, "labels", split, img_id + ".txt")
    with open(label_path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

    return ("background" if not lines else "ok", img_id)


def write_data_yaml(out_dir):
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        f.write(
            f"path: {out_dir}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            f"\nnames:\n{names}\n"
        )


def main():
    args = parse_args()

    print(f"Leyendo etiquetas de {args.labels_csv} ...")
    boxes, splits, skipped_rows = load_annotations(args.labels_csv)
    print(f"  {len(splits)} imágenes en el CSV, {skipped_rows} filas descartadas")

    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(args.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", split), exist_ok=True)

    tasks = [(img_id, split, boxes.get(img_id, []), args)
             for img_id, split in splits.items()]

    stats = defaultdict(int)
    per_split = defaultdict(int)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for (status, img_id), (_, split, _, _) in zip(pool.map(process_image, tasks), tasks):
            stats[status] += 1
            if status in ("ok", "background"):
                per_split[split] += 1
            done += 1
            if done % 10000 == 0:
                print(f"  {done}/{len(tasks)} procesadas ...")

    write_data_yaml(args.out)

    print("\nListo. Resumen:")
    print(f"  Imágenes con cajas:        {stats['ok']}")
    print(f"  Imágenes background:       {stats['background']} (label vacío)")
    print(f"  Imágenes faltantes:        {stats['missing']}")
    print(f"  Imágenes corruptas:        {stats['corrupt']}")
    for split in ("train", "val", "test"):
        print(f"  {split}: {per_split[split]} imágenes")
    print(f"\nDataset en: {args.out}")
    print(f"data.yaml en: {os.path.join(args.out, 'data.yaml')}")
    print("\nPara entrenar YOLOv11 medium:")
    print(f"  yolo detect train data={os.path.join(args.out, 'data.yaml')} "
          "model=yolo11m.pt epochs=100 imgsz=640")


if __name__ == "__main__":
    main()
