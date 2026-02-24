#!/bin/bash

# FRC YOLO Vision Auto-Boot Setup Script
# This script configures the FRC vision system to start automatically on boot

# Define variables
PROJECT_DIR="/home/pcpirates/Desktop/object-detection"
SERVICE_NAME="frc-vision"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
PYTHON_SCRIPT="run_inference.py"  # Change this if using a different entry point

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

if [ -x "${PROJECT_DIR}/.venv314/bin/python3" ]; then
    PYTHON_BIN="${PROJECT_DIR}/.venv314/bin/python3"
elif [ -x "${PROJECT_DIR}/.venv/bin/python3" ]; then
    PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python3"
else
    echo -e "${RED}✗ Python interpreter not found in .venv314 or .venv${NC}"
    exit 1
fi

echo -e "${YELLOW}FRC YOLO Vision Auto-Boot Setup${NC}"
echo "=================================="

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}This script must be run as root${NC}"
   echo "Try: sudo ./setup_autoboot.sh"
   exit 1
fi

# Create the service file
echo -e "${YELLOW}Creating systemd service file...${NC}"
cat > "$SERVICE_FILE" << EOF
[Unit]
Description=FRC YOLO Vision
After=network.target

[Service]
ExecStart=${PYTHON_BIN} ${PROJECT_DIR}/${PYTHON_SCRIPT}
WorkingDirectory=${PROJECT_DIR}
Environment=PYTORCH_DISABLE_NNPACK=1
Environment=VISION_HEADLESS=1
Environment=CAMERA_INDEX=0
Environment=CAMERA_INDEX_CANDIDATES=0,1,2
StandardOutput=journal
StandardError=journal
Restart=always
RestartSec=10
User=pcpirates

[Install]
WantedBy=multi-user.target
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Service file created successfully${NC}"
else
    echo -e "${RED}✗ Failed to create service file${NC}"
    exit 1
fi

# Reload systemd daemon
echo -e "${YELLOW}Reloading systemd daemon...${NC}"
systemctl daemon-reload
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Systemd daemon reloaded${NC}"
else
    echo -e "${RED}✗ Failed to reload systemd daemon${NC}"
    exit 1
fi

# Enable the service to start on boot
echo -e "${YELLOW}Enabling service to start on boot...${NC}"
systemctl enable "${SERVICE_NAME}.service"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Service enabled${NC}"
else
    echo -e "${RED}✗ Failed to enable service${NC}"
    exit 1
fi

# Start the service
echo -e "${YELLOW}Starting the service...${NC}"
systemctl start "${SERVICE_NAME}.service"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Service started${NC}"
else
    echo -e "${RED}✗ Failed to start service${NC}"
    exit 1
fi

# Verify service status
echo -e "${YELLOW}Verifying service status...${NC}"
systemctl status "${SERVICE_NAME}.service"

echo ""
echo -e "${GREEN}=================================="
echo "Setup Complete!"
echo "==================================${NC}"
echo ""
echo "Useful commands:"
echo "  View logs:     sudo journalctl -u ${SERVICE_NAME} -f"
echo "  Check status:  sudo systemctl status ${SERVICE_NAME}"
echo "  Stop service:  sudo systemctl stop ${SERVICE_NAME}"
echo "  Restart:       sudo systemctl restart ${SERVICE_NAME}"
echo "  Disable:       sudo systemctl disable ${SERVICE_NAME}"
