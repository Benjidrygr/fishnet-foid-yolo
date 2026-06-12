#!/usr/bin/env bash
# Setup del servidor de entrenamiento (Ubuntu/Debian con GPU NVIDIA, ej. AWS g4dn).
# Verifica e instala todo lo necesario para entrenar y evaluar:
#   - paquetes de sistema (tmux, unzip, python3-venv, libs de OpenCV)
#   - venv de Python con ultralytics (incluye torch con CUDA)
#   - descomprime los zips de Fishnet si están en la raíz del repo
#   - construye dataset_yolo y dataset_yolo_balanced (symlinks locales)
# Idempotente: se puede correr varias veces; salta lo que ya está hecho.
#
# Uso:
#   bash setup_server.sh
#
# Los zips se descargan antes, en la raíz del repo, desde https://www.fishnet.ai/:
#   wget <url-de-fishnet>/foid_images_v100.zip
#   wget <url-de-fishnet>/foid_labels_v100.zip

set -euo pipefail
cd "$(dirname "$0")"

VENV=.venv-train
# --no-venv: usar el Python del entorno activo (ej. conda) en vez de crear venv
USE_ACTIVE_ENV=0
[ "${1:-}" = "--no-venv" ] && USE_ACTIVE_ENV=1
ok()   { printf '\033[32m[OK]\033[0m %s\n' "$1"; }
info() { printf '\033[36m[..]\033[0m %s\n' "$1"; }
warn() { printf '\033[33m[!!]\033[0m %s\n' "$1"; }
fail() { printf '\033[31m[XX]\033[0m %s\n' "$1"; exit 1; }

SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

# ---------- 1. paquetes de sistema ----------
if command -v apt-get >/dev/null; then
    PKGS=""
    for p in tmux unzip python3-venv python3-pip libgl1 libglib2.0-0; do
        dpkg -s "$p" >/dev/null 2>&1 || PKGS="$PKGS $p"
    done
    if [ -n "$PKGS" ]; then
        info "instalando paquetes:$PKGS"
        $SUDO apt-get update -qq
        $SUDO apt-get install -y -qq $PKGS
    fi
    ok "paquetes de sistema (tmux, unzip, python3-venv, libs OpenCV)"
else
    warn "no es un sistema apt; instala manualmente: tmux unzip python3-venv libgl1"
fi

# ---------- 2. GPU ----------
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
    ok "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
else
    warn "nvidia-smi no disponible: sin driver NVIDIA esta máquina entrenará en CPU."
    warn "En AWS usa una Deep Learning AMI (drivers preinstalados)."
fi

# ---------- 3. entorno de Python ----------
if [ "$USE_ACTIVE_ENV" = "1" ]; then
    # PY apunta al python activo; el resto del script lo usa via $VENV/bin/python,
    # asi que creamos un alias minimo con la misma estructura de rutas.
    VENV=$(mktemp -d)
    mkdir -p "$VENV/bin"
    PYBIN=$(command -v python3)
    ln -s "$PYBIN" "$VENV/bin/python"
    printf '#!/bin/sh\nexec "%s" -m pip "$@"\n' "$PYBIN" > "$VENV/bin/pip"
    chmod +x "$VENV/bin/pip"
    ok "usando el entorno activo: $PYBIN"
elif [ ! -x "$VENV/bin/python" ]; then
    info "creando venv $VENV"
    python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/python" -c "import ultralytics" 2>/dev/null || {
    info "instalando ultralytics (descarga torch, puede tardar varios minutos)"
    "$VENV/bin/pip" install --quiet ultralytics
}
ok "ultralytics $("$VENV/bin/python" -c 'import ultralytics; print(ultralytics.__version__)')"

if "$VENV/bin/python" -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    ok "torch ve la GPU: $("$VENV/bin/python" -c 'import torch; print(torch.cuda.get_device_name(0))')"
else
    warn "torch NO ve ninguna GPU (torch.cuda.is_available() = False)"
fi

# ---------- 4. dataset: descomprimir zips ----------
if [ -d images ] && [ "$(ls images | head -1)" ]; then
    ok "images/ presente ($(ls images | wc -l | tr -d ' ') archivos)"
elif [ -f foid_images_v100.zip ]; then
    info "descomprimiendo foid_images_v100.zip (35GB, tarda)..."
    unzip -q foid_images_v100.zip      # el zip ya contiene la carpeta images/
    ok "images/ extraída ($(ls images | wc -l | tr -d ' ') archivos)"
else
    fail "falta images/ y no está foid_images_v100.zip — descárgalo con wget desde fishnet.ai"
fi

if [ -f foid_labels_v100/foid_labels_v100.csv ]; then
    ok "labels CSV presente"
elif [ -f foid_labels_v100.zip ]; then
    info "descomprimiendo foid_labels_v100.zip..."
    unzip -q foid_labels_v100.zip -x "__MACOSX/*" -d foid_labels_v100   # el zip trae archivos sueltos
    ok "foid_labels_v100/ extraída"
else
    fail "falta el CSV y no está foid_labels_v100.zip — descárgalo con wget desde fishnet.ai"
fi

# ---------- 5. construir datasets (symlinks locales de ESTA máquina) ----------
if [ -d dataset_yolo/labels/train ]; then
    ok "dataset_yolo ya existe"
else
    info "construyendo dataset_yolo (lee dimensiones de 143k imágenes, ~10 min)..."
    "$VENV/bin/python" make_yolo_dataset.py
fi

if [ -d dataset_yolo_balanced/labels/train ]; then
    ok "dataset_yolo_balanced ya existe"
else
    info "construyendo dataset_yolo_balanced (fusión PLS->OTH + oversampling)..."
    "$VENV/bin/python" make_balanced_dataset.py
fi

BROKEN=$(find dataset_yolo_balanced/images -type l ! -exec test -e {} \; -print 2>/dev/null | head -1)
[ -z "$BROKEN" ] && ok "symlinks del dataset verificados (ninguno roto)" \
                 || fail "hay symlinks rotos (ej: $BROKEN) — regenera los datasets"

# ---------- 6. resumen ----------
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu)
PYCMD="$VENV/bin/python"
[ "$USE_ACTIVE_ENV" = "1" ] && PYCMD="python3   # (entorno activo: activa el mismo env conda dentro de tmux)"
echo
ok "TODO LISTO. Para entrenar dentro de tmux (sobrevive a desconexiones SSH):"
cat <<EOF

    tmux new -s train
    $PYCMD train_yolo.py --workers $(( NPROC < 8 ? NPROC : 8 ))
    # despegarse: Ctrl+B y luego D   |   volver: tmux attach -t train

    # al terminar:
    ${PYCMD%%   *} evaluate_yolo.py \\
        --weights runs/detect/foid_yolo11m/weights/best.pt --split test
EOF
