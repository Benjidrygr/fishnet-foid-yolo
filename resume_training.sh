#!/usr/bin/env bash
# Reanuda un entrenamiento interrumpido de YOLOv11m desde un run recuperado.
# Continúa exactamente donde se cortó (época, optimizador, EMA y scheduler de LR
# del checkpoint), NO reentrena desde cero.
#
# Qué hace:
#   - localiza la carpeta del run recuperado (con weights/last.pt + args.yaml)
#   - la coloca en runs/detect/<name>/ donde Ultralytics la espera
#   - parchea la ruta del dataset dentro de last.pt (el run venía de otra máquina,
#     y en 'resume' Ultralytics ignora --data y usa la ruta guardada en el .pt)
#   - reanuda el entrenamiento hasta completar las epochs originales
# Idempotente: si el run ya está en runs/detect/ y parcheado, solo reanuda.
#
# Uso:
#   bash resume_training.sh                       # busca ./foid_yolo11m o runs/detect/foid_yolo11m
#   bash resume_training.sh ruta/al/run           # carpeta recuperada en otra ubicación
#   bash resume_training.sh --no-venv [ruta]      # usar el Python del entorno activo (ej. conda)

set -euo pipefail
cd "$(dirname "$0")"

VENV=.venv-train
NAME=foid_yolo11m
DATA=dataset_yolo_balanced/data.yaml

USE_ACTIVE_ENV=0
[ "${1:-}" = "--no-venv" ] && { USE_ACTIVE_ENV=1; shift; }
SRC="${1:-}"   # carpeta del run recuperado (opcional)

ok()   { printf '\033[32m[OK]\033[0m %s\n' "$1"; }
info() { printf '\033[36m[..]\033[0m %s\n' "$1"; }
warn() { printf '\033[33m[!!]\033[0m %s\n' "$1"; }
fail() { printf '\033[31m[XX]\033[0m %s\n' "$1"; exit 1; }

# ---------- 1. Python a usar ----------
if [ "$USE_ACTIVE_ENV" = "1" ]; then
    PY=python3
    ok "usando el entorno activo: $(command -v python3)"
elif [ -x "$VENV/bin/python" ]; then
    PY="$VENV/bin/python"
    ok "usando venv $VENV"
else
    PY=python3
    warn "no existe $VENV; usando python3 del sistema. Corre setup_server.sh si falta ultralytics."
fi
"$PY" -c "import ultralytics" 2>/dev/null || fail "ultralytics no está instalado en este Python — corre 'bash setup_server.sh' primero."

# ---------- 2. dataset presente ----------
[ -f "$DATA" ] || fail "no existe $DATA — construye los datasets con 'bash setup_server.sh' antes de reanudar."

# ---------- 3. ubicar el run recuperado en runs/detect/<name> ----------
DST="runs/detect/$NAME"
if [ -f "$DST/weights/last.pt" ]; then
    ok "run ya está en $DST"
else
    [ -z "$SRC" ] && { [ -d "$NAME" ] && SRC="$NAME"; }
    [ -z "$SRC" ] && fail "no encuentro el run recuperado. Pásalo como argumento: bash resume_training.sh ruta/al/run"
    [ -f "$SRC/weights/last.pt" ] || fail "'$SRC' no parece un run de Ultralytics (falta weights/last.pt)"
    info "moviendo $SRC -> $DST"
    mkdir -p runs/detect
    mv "$SRC" "$DST"
    ok "run colocado en $DST"
fi
rm -f "$DST"/.results.csv.swp   # swap de vim que quedó del servidor viejo

# ---------- 4. parchear rutas absolutas de la máquina vieja dentro de last.pt ----------
# El checkpoint guardó 'data' y 'save_dir' con rutas de /home/ubuntu/...; en resume
# Ultralytics las reutiliza tal cual e intenta leer/escribir allí. Las reapuntamos
# a esta máquina (si no, falla con FileNotFound/PermissionError en /home/ubuntu).
info "parcheando rutas (data y save_dir) en last.pt"
"$PY" - "$DST/weights/last.pt" "$DATA" "$DST" "$NAME" <<'PY'
import os, sys, torch
ckpt, data, dst, name = sys.argv[1:5]
data, dst = os.path.abspath(data), os.path.abspath(dst)
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
a = ck["train_args"]
a["data"] = data
a["save_dir"] = dst
a["project"] = os.path.dirname(dst)
a["name"] = name
torch.save(ck, ckpt)
print(f"    data     -> {data}")
print(f"    save_dir -> {dst} (época guardada: {ck.get('epoch', '?')})")
PY

# ---------- 5. reanudar ----------
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
    ok "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
else
    warn "nvidia-smi no disponible: reanudará en CPU (lentísimo)."
fi

info "reanudando entrenamiento (Ctrl+C para abortar). Ejecútalo dentro de tmux para sobrevivir a desconexiones SSH."
"$PY" - "$DST/weights/last.pt" <<'PY'
import sys
from ultralytics import YOLO
YOLO(sys.argv[1]).train(resume=True)
PY

echo
ok "Entrenamiento reanudado/terminado. Mejor checkpoint: $DST/weights/best.pt"
cat <<EOF

    # evaluar en test:
    $PY evaluate_yolo.py --weights $DST/weights/best.pt --split test
EOF
