#!/bin/bash
# Spectrum 服务卸载脚本 (Linux)
# 使用方法: sudo bash uninstall.sh

set -e

SERVICE_NAME="spectrum"
INSTALL_DIR="/opt/spectrum"
SERVICE_USER="spectrum"

echo "=== Spectrum 服务卸载脚本 ==="
echo ""

# 检查是否以 root 运行
if [ "$EUID" -ne 0 ]; then
    echo "错误: 请使用 sudo 运行此脚本"
    exit 1
fi

# 停止并禁用服务
echo "[1/4] 停止服务..."
if systemctl is-active --quiet "$SERVICE_NAME"; then
    systemctl stop "$SERVICE_NAME"
    echo "  服务已停止"
else
    echo "  服务未运行"
fi

echo "[2/4] 禁用服务..."
if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
    systemctl disable "$SERVICE_NAME"
    echo "  服务已禁用"
else
    echo "  服务未启用"
fi

# 删除 systemd 服务文件
echo "[3/4] 删除服务文件..."
if [ -f "/etc/systemd/system/$SERVICE_NAME.service" ]; then
    rm "/etc/systemd/system/$SERVICE_NAME.service"
    systemctl daemon-reload
    echo "  服务文件已删除"
else
    echo "  服务文件不存在"
fi

# 询问是否删除安装目录
echo ""
read -p "是否删除安装目录 $INSTALL_DIR? [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "[4/4] 删除安装目录..."
    rm -rf "$INSTALL_DIR"
    echo "  安装目录已删除"
else
    echo "[4/4] 保留安装目录"
fi

# 询问是否删除服务用户
read -p "是否删除服务用户 $SERVICE_USER? [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if id "$SERVICE_USER" &>/dev/null; then
        userdel "$SERVICE_USER"
        echo "  用户已删除"
    else
        echo "  用户不存在"
    fi
fi

echo ""
echo "=== 卸载完成 ==="
