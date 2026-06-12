# Fishnet FOID → YOLOv11 — pipeline de detección para monitoreo electrónico pesquero

Pipeline completo para entrenar un detector YOLOv11m sobre el dataset
[Fishnet Open Images v1.0.0](https://www.fishnet.ai/) (143,818 imágenes de
cámaras de monitoreo electrónico en palangreros atuneros, 549k bounding boxes):
limpieza y conversión a formato YOLO, balanceo de clases, anotación en CVAT,
entrenamiento y evaluación de métricas.

> Las imágenes (35 GB) y el CSV de etiquetas **no** están en este repo.
> Descarga `foid_images_v100.zip` y `foid_labels_v100.zip` desde Fishnet y
> descomprímelos en la raíz (carpetas `images/` y `foid_labels_v100/`).

## Scripts

| Script | Qué hace |
|---|---|
| `make_yolo_dataset.py` | CSV de Fishnet → dataset YOLO (`dataset_yolo/`, 12 clases `label_l2`, sin NoF/WATER/OIL; split train/val/test original por cámara; symlinks, no duplica disco). |
| `make_balanced_dataset.py` | Variante de entrenamiento (`dataset_yolo_balanced/`, 11 clases): fusión PLS→OTH + oversampling de clases raras solo en train (TUNA/SHARK ×10, LAG ×8, DOL ×4). |
| `train_yolo.py` | Entrenamiento YOLOv11m (pensado para NVIDIA T4 16 GB): augmentation ajustada a cubierta de barco (rotación 180°, flip vertical, jitter fuerte de brillo), early stopping, cos LR. |
| `evaluate_yolo.py` | Métricas sobre val o test: P/R/mAP50/mAP50-95 global y por clase, CSV, matriz de confusión y curvas PR/F1. |
| `make_cvat_export.py` | Zips de anotaciones por split en formato "Ultralytics YOLO Detection 1.0" importable en CVAT. |
| `upload_to_cvat.py` | Sube el dataset completo a un CVAT local (proyecto + tasks por split leyendo de un file share + anotaciones), vía cvat-sdk. |

## Uso

```bash
# 1. dataset limpio y variante balanceada
python3 make_yolo_dataset.py
python3 make_balanced_dataset.py

# 2. entrenamiento (GPU) y evaluación
pip install ultralytics
python3 train_yolo.py                  # defaults T4: 640px, 80 epochs, batch auto
python3 evaluate_yolo.py --weights runs/detect/foid_yolo11m/weights/best.pt --split test
```

Los detalles de clases y conteos están en `dataset_yolo/README.md` y
`dataset_yolo_balanced/README.md` (los labels/imágenes se regeneran con los
scripts; al moverse de máquina los symlinks deben regenerarse, no copiarse).

## CVAT local (opcional, para revisar/extender anotaciones)

```bash
git clone https://github.com/cvat-ai/cvat && cd cvat
# montar esta carpeta como file share (ver docker-compose.override.yml en docs de CVAT)
docker compose up -d
python3 upload_to_cvat.py   # requiere: pip install cvat-sdk
```
