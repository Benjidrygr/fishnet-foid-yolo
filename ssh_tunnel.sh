#!/usr/bin/env bash
# Túnel SSH inverso (reverse tunnel) para llegar al servidor de entrenamiento
# cuando está detrás de NAT/firewall y NO tiene IP pública.
#
# Idea: el SERVIDOR abre una conexión SALIENTE hacia un VPS con IP pública (que sí
# es alcanzable). Esa conexión publica el puerto SSH del servidor en el VPS, así
# desde cualquier lado entras al servidor "pasando por" el VPS. No hace falta
# abrir puertos en el router de la red del servidor.
#
#     [tu laptop] --ssh--> [VPS público] <--reverse tunnel-- [servidor detrás de NAT]
#
# Por defecto el puerto reenviado queda escuchando SOLO en localhost del VPS (más
# seguro: no se expone a internet). Para conectarte usas ProxyJump a través del VPS
# (el comando exacto lo imprime este script). Si prefieres exponerlo público en el
# VPS, pon BIND_ADDRESS=0.0.0.0 y habilita 'GatewayPorts yes' en el sshd del VPS.
#
# Usa autossh si está instalado (reconecta solo si se cae la red); si no, hace un
# loop de reconexión con keepalives.
#
# Uso:
#   bash ssh_tunnel.sh setup-key  # crea la llave SSH y la registra en el VPS (corre esto 1ª vez)
#   bash ssh_tunnel.sh            # abre el túnel en primer plano (Ctrl+C para cerrar)
#   bash ssh_tunnel.sh run        # igual que arriba (lo que invoca el servicio systemd)
#   bash ssh_tunnel.sh install    # instala servicio systemd (sobrevive reboots/desconexiones)
#   bash ssh_tunnel.sh status     # estado del servicio systemd
#   bash ssh_tunnel.sh stop       # detiene el servicio systemd
#   bash ssh_tunnel.sh uninstall  # quita el servicio systemd
#
# Configuración: edita tunnel.conf (se crea al primer uso, está en .gitignore) o
# exporta las variables antes de correr el script.

set -euo pipefail
cd "$(dirname "$0")"

ok()   { printf '\033[32m[OK]\033[0m %s\n' "$1"; }
info() { printf '\033[36m[..]\033[0m %s\n' "$1"; }
warn() { printf '\033[33m[!!]\033[0m %s\n' "$1"; }
fail() { printf '\033[31m[XX]\033[0m %s\n' "$1"; exit 1; }

CONF="tunnel.conf"
SERVICE_NAME="foid-tunnel"

# ---------- valores por defecto (sobreescribibles por tunnel.conf o env) ----------
VPS_HOST="${VPS_HOST:-}"            # IP o dominio del VPS público (OBLIGATORIO)
VPS_USER="${VPS_USER:-ubuntu}"      # usuario en el VPS
VPS_SSH_PORT="${VPS_SSH_PORT:-22}"  # puerto SSH del VPS
REMOTE_PORT="${REMOTE_PORT:-2222}"  # puerto en el VPS que reenvía al :22 del servidor
LOCAL_SSH_PORT="${LOCAL_SSH_PORT:-22}"  # puerto SSH local del servidor
BIND_ADDRESS="${BIND_ADDRESS:-localhost}"  # localhost = privado; 0.0.0.0 = público (req. GatewayPorts)
SSH_KEY="${SSH_KEY:-}"              # ruta a la llave privada (opcional; si vacío usa la default de ssh)

# ---------- crear tunnel.conf de ejemplo si no existe ----------
if [ ! -f "$CONF" ]; then
    info "creando $CONF de ejemplo (edítalo con los datos de tu VPS)"
    cat > "$CONF" <<'EOF'
# Configuración del túnel SSH inverso. Este archivo está en .gitignore.
# Rellena al menos VPS_HOST.

VPS_HOST=""              # IP o dominio del VPS público, ej. 203.0.113.10
VPS_USER="ubuntu"        # usuario SSH en el VPS
VPS_SSH_PORT="22"        # puerto SSH del VPS
REMOTE_PORT="2222"       # puerto en el VPS que apuntará al SSH del servidor
LOCAL_SSH_PORT="22"      # puerto SSH local de este servidor
BIND_ADDRESS="localhost" # localhost = solo accesible vía ProxyJump (recomendado)
                         # 0.0.0.0   = expuesto público en el VPS (requiere GatewayPorts yes)
SSH_KEY=""               # ruta a llave privada para conectar al VPS, ej. ~/.ssh/id_ed25519
EOF
    warn "edita $CONF y vuelve a ejecutar. Como mínimo necesitas VPS_HOST."
    exit 1
fi

# shellcheck disable=SC1090
source "$CONF"

[ -n "$VPS_HOST" ] || fail "VPS_HOST vacío. Edita $CONF con la IP/dominio de tu VPS."

# ---------- construir comando ssh ----------
SSH_BASE=(ssh -p "$VPS_SSH_PORT")
[ -n "$SSH_KEY" ] && SSH_BASE+=(-i "$SSH_KEY")
SSH_OPTS=(
    -N                                   # no ejecutar comando remoto, solo el forward
    -o ExitOnForwardFailure=yes          # si el -R no se puede crear, falla (no túnel zombie)
    -o ServerAliveInterval=30            # keepalive cada 30s
    -o ServerAliveCountMax=3             # corta tras 90s sin respuesta
    -o StrictHostKeyChecking=accept-new  # confía en el VPS la primera vez sin prompt
    -R "${BIND_ADDRESS}:${REMOTE_PORT}:localhost:${LOCAL_SSH_PORT}"
)
TARGET="${VPS_USER}@${VPS_HOST}"

connect_hint() {
    echo
    ok "Túnel activo. Para entrar al servidor desde fuera de su red:"
    if [ "$BIND_ADDRESS" = "localhost" ] || [ "$BIND_ADDRESS" = "127.0.0.1" ]; then
        local jump="$TARGET"
        [ "$VPS_SSH_PORT" != "22" ] && jump="$jump (puerto $VPS_SSH_PORT)"
        cat <<EOF

    # ProxyJump a través del VPS (el puerto $REMOTE_PORT vive en localhost del VPS):
    ssh -J ${VPS_USER}@${VPS_HOST}:${VPS_SSH_PORT} -p ${REMOTE_PORT} <usuario-del-servidor>@localhost

    # o en dos saltos: primero entras al VPS y desde ahí:
    ssh -p ${REMOTE_PORT} <usuario-del-servidor>@localhost
EOF
    else
        cat <<EOF

    # El puerto está expuesto en el VPS (requiere 'GatewayPorts yes' en su sshd):
    ssh -p ${REMOTE_PORT} <usuario-del-servidor>@${VPS_HOST}
EOF
    fi
    echo
}

# ---------- modo run: abre el túnel (autossh si existe, si no loop) ----------
do_run() {
    info "túnel: ${BIND_ADDRESS}:${REMOTE_PORT} (VPS ${VPS_HOST}) -> localhost:${LOCAL_SSH_PORT} (servidor)"
    connect_hint
    if command -v autossh >/dev/null 2>&1; then
        ok "usando autossh (reconexión automática)"
        AUTOSSH_GATETIME=0 exec autossh -M 0 "${SSH_BASE[@]:1}" "${SSH_OPTS[@]}" "$TARGET"
        # nota: SSH_BASE[0] es 'ssh'; autossh lo reemplaza, por eso :1
    fi
    warn "autossh no está instalado; usando loop de reconexión (instala 'autossh' para algo más robusto)."
    while true; do
        "${SSH_BASE[@]}" "${SSH_OPTS[@]}" "$TARGET" || warn "túnel caído (rc=$?), reconectando en 5s..."
        sleep 5
    done
}

# ---------- modo install: servicio systemd ----------
do_install() {
    command -v systemctl >/dev/null 2>&1 || fail "systemd no disponible; usa 'bash ssh_tunnel.sh run' dentro de tmux."
    local script_path unit
    script_path="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    unit="/etc/systemd/system/${SERVICE_NAME}.service"
    local SUDO=""
    [ "$(id -u)" -ne 0 ] && SUDO="sudo"
    info "instalando servicio systemd en $unit"
    $SUDO tee "$unit" >/dev/null <<EOF
[Unit]
Description=Reverse SSH tunnel del servidor de entrenamiento foid hacia ${VPS_HOST}
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$(id -un)
WorkingDirectory=$(pwd)
ExecStart=/usr/bin/env bash ${script_path} run
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    $SUDO systemctl daemon-reload
    $SUDO systemctl enable --now "$SERVICE_NAME"
    ok "servicio '$SERVICE_NAME' habilitado y arrancado (sobrevive reboots)."
    info "logs en vivo:  journalctl -u $SERVICE_NAME -f"
    connect_hint
}

# ---------- modo setup-key: genera la llave y la registra en el VPS ----------
do_setup_key() {
    local key="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
    if [ -f "$key" ]; then
        ok "ya existe la llave $key (no la regenero)"
    else
        info "generando llave SSH ed25519 sin passphrase en $key"
        mkdir -p "$(dirname "$key")"
        ssh-keygen -t ed25519 -N "" -f "$key" -C "foid-tunnel@$(hostname)"
        ok "llave creada: $key (privada) y $key.pub (pública)"
    fi
    info "registrando la llave pública en ${VPS_USER}@${VPS_HOST} (te pedirá la password del VPS una vez)"
    ssh-copy-id -i "$key.pub" -p "$VPS_SSH_PORT" "${VPS_USER}@${VPS_HOST}"
    info "probando conexión sin password..."
    if ssh -p "$VPS_SSH_PORT" -i "$key" -o BatchMode=yes -o ConnectTimeout=10 \
           "${VPS_USER}@${VPS_HOST}" true 2>/dev/null; then
        ok "listo: el servidor entra al VPS sin password. Ya puedes 'bash ssh_tunnel.sh install'."
    else
        warn "la conexión sin password aún falla; revisa VPS_HOST/VPS_USER en $CONF y que el VPS permita PubkeyAuthentication."
    fi
}

do_status()    { systemctl status "$SERVICE_NAME" --no-pager; }
do_stop()      { local S=""; [ "$(id -u)" -ne 0 ] && S=sudo; $S systemctl stop "$SERVICE_NAME"; ok "servicio detenido."; }
do_uninstall() {
    local S=""; [ "$(id -u)" -ne 0 ] && S=sudo
    $S systemctl disable --now "$SERVICE_NAME" 2>/dev/null || true
    $S rm -f "/etc/systemd/system/${SERVICE_NAME}.service"
    $S systemctl daemon-reload
    ok "servicio '$SERVICE_NAME' eliminado."
}

# ---------- dispatch ----------
case "${1:-run}" in
    run)       do_run ;;
    setup-key) do_setup_key ;;
    install)   do_install ;;
    status)    do_status ;;
    stop)      do_stop ;;
    uninstall) do_uninstall ;;
    *) fail "comando desconocido: $1 (usa: setup-key | run | install | status | stop | uninstall)" ;;
esac
