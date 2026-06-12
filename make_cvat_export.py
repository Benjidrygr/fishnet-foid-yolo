#!/usr/bin/env python3
"""
Genera zips importables en CVAT (formato "Ultralytics YOLO Detection 1.0")
a partir del dataset creado por make_yolo_dataset.py.

Crea un zip por split (train/val/test) con esta estructura:

    cvat_<split>.zip
    ├── data.yaml          # path: ./  +  <split>: <split>.txt  +  names
    ├── <split>.txt        # lista de rutas images/<split>/xxx.jpg
    ├── labels/<split>/*.txt
    └── images/<split>/*.jpg   (solo con --include-images)

Por defecto NO incluye las imágenes (zip ligero, para "Upload annotations"
sobre un task que ya tiene las imágenes). Con --include-images las empaqueta
resolviendo los symlinks, para crear el task completo desde el zip.

Uso:
    python3 make_cvat_export.py [--splits train,val,test] [--include-images]
"""

import argparse
import os
import re
import zipfile


def parse_args():
    base = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=os.path.join(base, "dataset_yolo"),
                   help="Carpeta del dataset YOLO generado")
    p.add_argument("--out", default=os.path.join(base, "cvat_export"),
                   help="Carpeta donde dejar los zips")
    p.add_argument("--splits", default="train,val,test",
                   help="Splits a exportar, separados por coma")
    p.add_argument("--include-images", action="store_true",
                   help="Incluir las imágenes en el zip (resuelve symlinks)")
    return p.parse_args()


def load_class_names(dataset_dir):
    """Lee el bloque `names:` del data.yaml del dataset."""
    names = {}
    in_names = False
    with open(os.path.join(dataset_dir, "data.yaml")) as f:
        for line in f:
            if line.strip() == "names:":
                in_names = True
                continue
            if in_names:
                m = re.match(r"\s+(\d+):\s*(\S+)", line)
                if not m:
                    break
                names[int(m.group(1))] = m.group(2)
    if not names:
        raise SystemExit(f"No pude leer las clases de {dataset_dir}/data.yaml")
    return names


def export_split(dataset_dir, split, out_dir, names, include_images):
    labels_dir = os.path.join(dataset_dir, "labels", split)
    images_dir = os.path.join(dataset_dir, "images", split)
    if not os.path.isdir(labels_dir):
        print(f"  [{split}] no existe {labels_dir}, lo salto")
        return

    stems = sorted(os.path.splitext(f)[0]
                   for f in os.listdir(labels_dir) if f.endswith(".txt"))

    names_block = "\n".join(f"  {i}: {names[i]}" for i in sorted(names))
    data_yaml = (
        "path: ./\n"
        f"{split}: {split}.txt\n"
        f"\nnames:\n{names_block}\n"
    )
    file_list = "\n".join(f"images/{split}/{s}.jpg" for s in stems) + "\n"

    zip_path = os.path.join(out_dir, f"cvat_{split}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data.yaml", data_yaml)
        zf.writestr(f"{split}.txt", file_list)
        for s in stems:
            zf.write(os.path.join(labels_dir, s + ".txt"),
                     f"labels/{split}/{s}.txt")
        if include_images:
            for n, s in enumerate(stems, 1):
                src = os.path.realpath(os.path.join(images_dir, s + ".jpg"))
                zf.write(src, f"images/{split}/{s}.jpg")
                if n % 10000 == 0:
                    print(f"  [{split}] {n}/{len(stems)} imágenes empaquetadas ...")

    size_mb = os.path.getsize(zip_path) / 1024 / 1024
    print(f"  [{split}] {len(stems)} imágenes -> {zip_path} ({size_mb:.1f} MB)")


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    names = load_class_names(args.dataset)
    print(f"Clases: {', '.join(names[i] for i in sorted(names))}")
    print(f"Exportando ({'con' if args.include_images else 'sin'} imágenes):")
    for split in args.splits.split(","):
        export_split(args.dataset, split.strip(), args.out, names,
                     args.include_images)
    print("\nEn CVAT: Task/Project -> Upload annotations -> "
          "'Ultralytics YOLO Detection 1.0' y selecciona el zip del split "
          "que corresponda al subset del task.")


if __name__ == "__main__":
    main()
