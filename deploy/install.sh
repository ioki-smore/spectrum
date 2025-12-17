#!/bin/bash
# Spectrum 服务安装脚本 (Linux)
# 使用方法: sudo bash install.sh

set -e

# 配置变量
SERVICE_NAME="spectrum"
INSTALL_DIR="/opt/spectrum"
SERVICE_USER="spectrum"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Spectrum 服务安装脚本 ==="
echo "源目录: $REPO_DIR"
echo "安装目录: $INSTALL_DIR"
echo ""

# 检查是否以 root 运行
if [ "$EUID" -ne 0 ]; then
    echo "错误: 请使用 sudo 运行此脚本"
    exit 1
fi

# 创建服务用户
echo "[1/6] 创建服务用户..."
if ! id "$SERVICE_USER" &>/dev/null; then
    useradd --system --no-create-home --shell /bin/false "$SERVICE_USER"
    echo "  已创建用户: $SERVICE_USER"
else
    echo "  用户已存在: $SERVICE_USER"
fi

# 创建安装目录
echo "[2/6] 创建安装目录..."
mkdir -p "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR/logs"
mkdir -p "$INSTALL_DIR/data/source"
mkdir -p "$INSTALL_DIR/saved_models"
mkdir -p "$INSTALL_DIR/results"

# 复制代码
echo "[3/6] 复制代码..."
rsync -av --exclude='.git' \
          --exclude='__pycache__' \
          --exclude='*.pyc' \
          --exclude='.venv' \
          --exclude='venv' \
          --exclude='logs' \
          --exclude='ignored' \
          --exclude='*.log' \
          "$REPO_DIR/" "$INSTALL_DIR/"

# 创建虚拟环境并安装依赖
echo "[4/6] 创建虚拟环境并安装依赖..."
python3 -m venv "$INSTALL_DIR/venv"
"$INSTALL_DIR/venv/bin/pip" install --upgrade pip
"$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt"

# 设置权限
echo "[5/6] 设置权限..."
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
chmod 755 "$INSTALL_DIR"

# 安装 systemd 服务
echo "[6/6] 安装 systemd 服务..."
cp "$INSTALL_DIR/deploy/spectrum.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"

echo ""
echo "=== 安装完成 ==="
echo ""
echo "管理命令:"
echo "  启动服务:   sudo systemctl start $SERVICE_NAME"
echo "  停止服务:   sudo systemctl stop $SERVICE_NAME"
echo "  重启服务:   sudo systemctl restart $SERVICE_NAME"
echo "  查看状态:   sudo systemctl status $SERVICE_NAME"
echo "  查看日志:   sudo journalctl -u $SERVICE_NAME -f"
echo ""
echo "日志文件:"
echo "  stdout:     $INSTALL_DIR/logs/stdout.log"
echo "  stderr:     $INSTALL_DIR/logs/stderr.log"
echo "  app log:    $INSTALL_DIR/spectrum.log"
echo ""
echo "配置文件:     $INSTALL_DIR/config/config.yaml"
echo ""
