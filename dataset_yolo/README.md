# Dataset Fishnet FOID — formato YOLO (12 clases)

Generado por `make_yolo_dataset.py` desde Fishnet Open Images v1.0.0 (`foid_labels_v100.csv`). Etiquetas `label_l2`; se excluyeron NoF, WATER y OIL (sus imágenes quedan como background con label vacío). Split train/val/test original del CSV (diseñado por cámara para evitar fuga de secuencias). Las imágenes son symlinks a `../images/`.

- Imágenes: **143,818** (train 101,718 / val 21,465 / test 20,635)
- Cajas: **547,166**
- Background (0 cajas): 1,964

## Clases

Cajas = anotaciones; Imágenes = imágenes que contienen la clase.

| id | Clase | Cajas train | Cajas val | Cajas test | Cajas total | Imgs train | Imgs val | Imgs test | Imgs total |
|---|---|---|---|---|---|---|---|---|---|
| 0 | ALB | 58,583 | 12,541 | 10,347 | 81,471 | 31,775 | 7,217 | 7,101 | 46,093 |
| 1 | BET | 4,366 | 2,190 | 3,144 | 9,700 | 3,844 | 1,861 | 2,154 | 7,859 |
| 2 | BILL | 4,502 | 826 | 883 | 6,211 | 3,735 | 755 | 773 | 5,263 |
| 3 | DOL | 3,216 | 370 | 839 | 4,425 | 2,605 | 301 | 718 | 3,624 |
| 4 | HUMAN | 187,126 | 46,601 | 47,313 | 281,040 | 53,813 | 14,022 | 14,598 | 82,433 |
| 5 | LAG | 1,277 | 257 | 147 | 1,681 | 1,202 | 238 | 138 | 1,578 |
| 6 | OTH | 6,752 | 1,902 | 840 | 9,494 | 4,727 | 1,005 | 595 | 6,327 |
| 7 | PLS | 17 | 3 | 31 | 51 | 17 | 3 | 31 | 51 |
| 8 | SHARK | 561 | 138 | 110 | 809 | 518 | 132 | 101 | 751 |
| 9 | SKJ | 18,716 | 1,826 | 1,714 | 22,256 | 5,950 | 748 | 532 | 7,230 |
| 10 | TUNA | 413 | 32 | 390 | 835 | 344 | 26 | 373 | 743 |
| 11 | YFT | 103,089 | 11,962 | 14,142 | 129,193 | 47,036 | 6,588 | 8,489 | 62,113 |

Ver `dataset_yolo_balanced/` para la variante de entrenamiento (fusión PLS→OTH + oversampling de clases raras).
