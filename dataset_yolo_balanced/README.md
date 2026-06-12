# Dataset Fishnet FOID balanceado — formato YOLO (11 clases)

Generado por `make_balanced_dataset.py` a partir de `dataset_yolo/`. Cambios:

1. **Fusión de clases:** PLS → OTH (PLS tenía 17 cajas en train, inentrenable). Quedan 11 clases.
2. **Oversampling solo en train:** TUNA ×10, SHARK ×10, LAG ×8, DOL ×4. Las imágenes que contienen estas clases se duplican (symlinks `_osN`); la augmentation online de Ultralytics hace que cada copia se vea distinta por época.
3. **val y test intactos** (solo remapeo de ids): la evaluación no está inflada.

- Imágenes train: **125,559** efectivas (101,718 únicas + 23,841 duplicados)
- Imágenes val: 21,465 | test: 20,635

## Clases

En train: «efectivas» = lo que ve el modelo por época (incluye duplicados); «únicas» = imágenes físicas distintas.

| id | Clase | Oversample | Cajas train efect. | Imgs train efect. | Imgs train únicas | Cajas val | Imgs val | Cajas test | Imgs test |
|---|---|---|---|---|---|---|---|---|---|
| 0 | ALB | — | 70,569 | 36,952 | 31,775 | 12,541 | 7,217 | 10,347 | 7,101 |
| 1 | BET | — | 4,565 | 4,043 | 3,844 | 2,190 | 1,861 | 3,144 | 2,154 |
| 2 | BILL | — | 4,658 | 3,885 | 3,735 | 826 | 755 | 883 | 773 |
| 3 | DOL | ×4 | 13,128 | 10,684 | 2,605 | 370 | 301 | 839 | 718 |
| 4 | HUMAN | — | 268,009 | 75,223 | 53,813 | 46,601 | 14,022 | 47,313 | 14,598 |
| 5 | LAG | ×8 | 10,220 | 9,620 | 1,202 | 257 | 238 | 147 | 138 |
| 6 | OTH | — | 7,171 | 5,056 | 4,744 | 1,905 | 1,008 | 871 | 626 |
| 7 | SHARK | ×10 | 5,610 | 5,180 | 518 | 138 | 132 | 110 | 101 |
| 8 | SKJ | — | 21,996 | 6,744 | 5,950 | 1,826 | 748 | 1,714 | 532 |
| 9 | TUNA | ×10 | 4,130 | 3,440 | 344 | 32 | 26 | 390 | 373 |
| 10 | YFT | — | 106,965 | 49,532 | 47,036 | 11,962 | 6,588 | 14,142 | 8,489 |

> Nota: el oversampling también eleva las cajas efectivas de las clases que co-ocurren en esas imágenes (especialmente HUMAN). Es esperable e inofensivo.

## Entrenamiento

```bash
yolo detect train data=/Users/azariel/Downloads/foid/dataset_yolo_balanced/data.yaml model=yolo11m.pt epochs=100 imgsz=960
```

Para regenerar con otros factores (borrar antes la carpeta para no dejar duplicados huérfanos):

```bash
rm -rf dataset_yolo_balanced && python3 make_balanced_dataset.py --oversample "TUNA=15,SHARK=15,LAG=10,DOL=5"
```