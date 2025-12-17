# Spectrum 服务部署指南

## macOS (launchd)

### 1. 配置 Python 路径

编辑 `com.spectrum.service.plist`，确保 `ProgramArguments` 中的 Python 路径正确：

```bash
# 查看当前 Python 路径
which python3

# 如果使用虚拟环境，使用虚拟环境中的 Python
# 例如: /Users/smore/Workspace/spectrum/.venv/bin/python
```

### 2. 安装服务

```bash
# 复制 plist 到 LaunchAgents 目录（用户级别，登录后自动启动）
cp deploy/com.spectrum.service.plist ~/Library/LaunchAgents/

# 或者复制到 LaunchDaemons（系统级别，开机自动启动，需要 sudo）
# sudo cp deploy/com.spectrum.service.plist /Library/LaunchDaemons/
```

### 3. 加载并启动服务

```bash
# 加载服务
launchctl load ~/Library/LaunchAgents/com.spectrum.service.plist

# 启动服务（如果 RunAtLoad 为 true，加载时会自动启动）
launchctl start com.spectrum.service
```

### 4. 管理服务

```bash
# 查看服务状态
launchctl list | grep spectrum

# 停止服务
launchctl stop com.spectrum.service

# 卸载服务（停止并移除）
launchctl unload ~/Library/LaunchAgents/com.spectrum.service.plist

# 重新加载（更新配置后）
launchctl unload ~/Library/LaunchAgents/com.spectrum.service.plist
launchctl load ~/Library/LaunchAgents/com.spectrum.service.plist
```

### 5. 查看日志

```bash
# 查看 launchd 输出日志
tail -f logs/launchd_stdout.log
tail -f logs/launchd_stderr.log

# 查看应用日志
tail -f spectrum.log
```

---

## Linux (systemd)

如果需要部署到 Linux 服务器，请使用以下 systemd 配置：

### 1. 创建 service 文件

```bash
sudo nano /etc/systemd/system/spectrum.service
```

内容如下：

```ini
[Unit]
Description=Spectrum Anomaly Detection Service
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/spectrum
ExecStart=/usr/bin/python3 main.py run
Restart=on-failure
RestartSec=10
StandardOutput=append:/path/to/spectrum/logs/systemd_stdout.log
StandardError=append:/path/to/spectrum/logs/systemd_stderr.log
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

### 2. 管理服务

```bash
# 重新加载 systemd 配置
sudo systemctl daemon-reload

# 启动服务
sudo systemctl start spectrum

# 设置开机自启
sudo systemctl enable spectrum

# 查看状态
sudo systemctl status spectrum

# 查看日志
sudo journalctl -u spectrum -f

# 停止服务
sudo systemctl stop spectrum

# 重启服务
sudo systemctl restart spectrum
```

---

## 故障排除

### 服务无法启动

1. 检查 Python 路径是否正确
2. 检查工作目录是否存在
3. 检查日志文件权限

```bash
# macOS: 查看 launchd 错误
launchctl error <error_code>

# Linux: 查看详细日志
sudo journalctl -u spectrum -n 50 --no-pager
```

### 依赖问题

确保所有依赖已安装：

```bash
pip install -r requirements.txt
```

### 权限问题

确保日志目录可写：

```bash
mkdir -p logs
chmod 755 logs
```
