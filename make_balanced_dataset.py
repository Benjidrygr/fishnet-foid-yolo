#!/usr/bin/env python3
"""
Genera una variante balanceada del dataset YOLO (dataset_yolo) para entrenar:

1. Fusión de clases: PLS -> OTH (PLS tiene 17 cajas en train, inentrenable).
   El dataset queda en 11 clases y los ids se remapean en train/val/test.
2. Oversampling SOLO en train: las imágenes que contienen clases raras se
   duplican N veces (symlinks <stem>_osK.jpg + copia del label). Si una imagen
   contiene varias clases raras, se usa el factor mayor. La augmentation
   online de Ultralytics (mosaic, HSV, escala) hace que cada repetición se
   vea distinta en cada época.

val y test solo se remapean, nunca se sobremuestrean.

Uso:
    python3 make_balanced_dataset.py [--oversample "TUNA=10,SHARK=10,LAG=8,DOL=4"]
"""

import argparse
import os
from collections import Counter, defaultdict

SRC_CLASSES = ["ALB", "BET", "BILL", "DOL", "HUMAN", "LAG",
               "OTH", "PLS", "SHARK", "SKJ", "TUNA", "YFT"]
MERGE = {"PLS": "OTH"}
DEFAULT_OVERSAMPLE = "TUNA=10,SHARK=10,LAG=8,DOL=4"


def parse_args():
    base = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default=os.path.join(base, "dataset_yolo"))
    p.add_argument("--out", default=os.path.join(base, "dataset_yolo_balanced"))
    p.add_argument("--oversample", default=DEFAULT_OVERSAMPLE,
                   help='Factores por clase, ej. "TUNA=10,SHARK=10" '
                        "(una imagen con la clase aparece N veces en train)")
    return p.parse_args()


def build_class_maps():
    """Devuelve (clases nuevas, mapa id_viejo -> id_nuevo)."""
    new_classes = [c for c in SRC_CLASSES if c not in MERGE]
    new_index = {c: i for i, c in enumerate(new_classes)}
    id_map = {}
    for old_id, name in enumerate(SRC_CLASSES):
        id_map[old_id] = new_index[MERGE.get(name, name)]
    return new_classes, id_map


def remap_label_file(src_path, id_map):
    """Devuelve (líneas remapeadas, ids nuevos presentes)."""
    lines, present = [], set()
    with open(src_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue
            new_id = id_map[int(parts[0])]
            present.add(new_id)
            lines.append(" ".join([str(new_id)] + parts[1:]))
    return lines, present


def main():
    args = parse_args()
    new_classes, id_map = build_class_maps()
    class_id = {c: i for i, c in enumerate(new_classes)}

    factors = {}
    if args.oversample:
        for part in args.oversample.split(","):
            name, _, n = part.strip().partition("=")
            if name not in class_id:
                raise SystemExit(f"Clase desconocida en --oversample: {name}")
            factors[class_id[name]] = int(n)

    print(f"Clases ({len(new_classes)}): {', '.join(new_classes)}  [{' ,'.join(f'{k}->{v}' for k, v in MERGE.items())}]")
    print("Oversampling train:",
          ", ".join(f"{new_classes[i]} x{n}" for i, n in sorted(factors.items())) or "ninguno")

    box_counts = defaultdict(Counter)   # split -> id -> cajas (tras oversampling)
    img_counts = Counter()

    for split in ("train", "val", "test"):
        src_lbl = os.path.join(args.src, "labels", split)
        src_img = os.path.join(args.src, "images", split)
        out_lbl = os.path.join(args.out, "labels", split)
        out_img = os.path.join(args.out, "images", split)
        os.makedirs(out_lbl, exist_ok=True)
        os.makedirs(out_img, exist_ok=True)

        for fname in os.listdir(src_lbl):
            stem = os.path.splitext(fname)[0]
            lines, present = remap_label_file(os.path.join(src_lbl, fname), id_map)
            content = "\n".join(lines) + ("\n" if lines else "")
            img_target = os.path.realpath(os.path.join(src_img, stem + ".jpg"))

            copies = 1
            if split == "train":
                copies = max([factors.get(i, 1) for i in present] or [1])

            for k in range(copies):
                name = stem if k == 0 else f"{stem}_os{k}"
                with open(os.path.join(out_lbl, name + ".txt"), "w") as f:
                    f.write(content)
                dst = os.path.join(out_img, name + ".jpg")
                if not os.path.lexists(dst):
                    os.symlink(img_target, dst)
                img_counts[split] += 1
                for i in present:
                    box_counts[split][i] += sum(1 for ln in lines
                                                if ln.split()[0] == str(i))

    names_block = "\n".join(f"  {i}: {c}" for i, c in enumerate(new_classes))
    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        f.write(f"path: {args.out}\ntrain: images/train\nval: images/val\n"
                f"test: images/test\n\nnames:\n{names_block}\n")

    print("\nImágenes por split (train incluye duplicados de oversampling):")
    for split in ("train", "val", "test"):
        print(f"  {split}: {img_counts[split]}")
    print("\nCajas por clase en train (efectivas tras oversampling):")
    for i, c in enumerate(new_classes):
        mark = f"  <- x{factors[i]}" if i in factors else ""
        print(f"  {c:6s} {box_counts['train'][i]:7d}{mark}")
    print(f"\nDataset en: {args.out}")
    print(f"Entrenar: yolo detect train data={os.path.join(args.out, 'data.yaml')} "
          "model=yolo11m.pt epochs=100 imgsz=960")


if __name__ == "__main__":
    main()
