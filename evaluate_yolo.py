#!/usr/bin/env python3
"""
Evaluación de métricas de un checkpoint YOLO sobre val o test del dataset
Fishnet. Imprime la tabla por clase, guarda un CSV y genera los gráficos
(matriz de confusión, curvas PR/F1/P/R).

Uso:
    python3 evaluate_yolo.py --weights runs/detect/foid_yolo11m/weights/best.pt
    python3 evaluate_yolo.py --weights best.pt --split test --imgsz 960

Protocolo estándar: conf=0.001 e IoU NMS=0.7 (las curvas barren todos los
umbrales de confianza; el mAP no depende de un umbral fijo).
"""

import argparse
import csv
import os

from ultralytics import YOLO

from train_yolo import resolve_data_yaml


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default="dataset_yolo_balanced/data.yaml")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--imgsz", type=int, default=640,
                   help="Usar el MISMO imgsz con el que se entreno")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--name", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    data = resolve_data_yaml(args.data)
    model = YOLO(args.weights)

    metrics = model.val(
        data=data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=True,
        name=args.name or f"eval_{args.split}",
    )

    names = metrics.names
    print(f"\n{'='*72}")
    print(f"RESULTADOS GLOBALES ({args.split}, {len(names)} clases)")
    print(f"{'='*72}")
    print(f"  Precision media : {metrics.box.mp:.4f}")
    print(f"  Recall medio    : {metrics.box.mr:.4f}")
    print(f"  mAP@50          : {metrics.box.map50:.4f}")
    print(f"  mAP@50-95       : {metrics.box.map:.4f}")

    # class_result() indexa por posicion entre las clases CON instancias en el
    # set evaluado; ap_class_index da el id de clase real de cada posicion.
    by_class = {int(cid): metrics.box.class_result(pos)
                for pos, cid in enumerate(metrics.box.ap_class_index)}

    rows = []
    print(f"\n{'clase':8s} {'P':>7s} {'R':>7s} {'mAP50':>7s} {'mAP50-95':>9s}")
    for i in sorted(names):
        if i in by_class:
            p, r, ap50, ap = by_class[i]
            rows.append({"clase": names[i], "precision": round(p, 4),
                         "recall": round(r, 4), "mAP50": round(ap50, 4),
                         "mAP50_95": round(ap, 4)})
            print(f"{names[i]:8s} {p:7.3f} {r:7.3f} {ap50:7.3f} {ap:9.3f}")
        else:
            rows.append({"clase": names[i], "precision": "", "recall": "",
                         "mAP50": "", "mAP50_95": ""})
            print(f"{names[i]:8s} {'sin instancias en este split':>33s}")

    out_csv = os.path.join(str(metrics.save_dir), "metrics_per_class.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"\nCSV por clase : {out_csv}")
    print(f"Graficos      : {metrics.save_dir}/ (confusion_matrix*.png, "
          f"*_curve.png)")
    sp = metrics.speed
    print(f"Velocidad     : {sp['preprocess']:.1f} + {sp['inference']:.1f} + "
          f"{sp['postprocess']:.1f} ms/imagen (pre+inferencia+post)")


if __name__ == "__main__":
    main()
