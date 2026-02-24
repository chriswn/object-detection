# FRC YOLO Vision Auto-Boot Setup

This folder contains scripts to configure your FRC vision system to start automatically on boot on your Linux system (Kangaroo).

## Files Included

1. **setup_autoboot.sh** - Automated setup script (recommended)
2. **frc-vision.service** - Service file template

## Quick Setup (Recommended Method)

### On Your Linux System:

1. **Transfer the script to your Linux system:**
   ```bash
   # From Windows, use SCP or copy via USB
   # Or download from your repository
   ```

2. **Run the setup script:**
   ```bash
   sudo chmod +x setup_autoboot.sh
   sudo ./setup_autoboot.sh
   ```

   The script will automatically:
   - Create the systemd service file
   - Enable the service to start on boot
   - Start the service immediately
   - Verify everything is working

## Manual Setup (If Preferred)

If you prefer to set it up manually:

1. **Create the service file:**
   ```bash
   sudo nano /etc/systemd/system/frc-vision.service
   ```

2. **Paste the service file content** from `frc-vision.service`

3. **Save and exit:**
   - Press `Ctrl+O`, then `Enter`
   - Press `Ctrl+X`

4. **Enable and start the service:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable frc-vision.service
   sudo systemctl start frc-vision.service
   ```

## Useful Commands After Setup

```bash
# View service logs in real-time
sudo journalctl -u frc-vision -f

# Check current status
sudo systemctl status frc-vision

# Stop the service
sudo systemctl stop frc-vision

# Restart the service
sudo systemctl restart frc-vision

# Disable auto-start on boot
sudo systemctl disable frc-vision

# Remove the service completely
sudo systemctl disable frc-vision
sudo rm /etc/systemd/system/frc-vision.service
sudo systemctl daemon-reload
```

## Important Notes

- **Update the paths** in the service file if your installation directory differs from `/home/pcpirates/Desktop/object-detection`
- **Change the Python script** if using a different entry point (currently set to `run_inference.py`)
- **Change the username** if running under a different user account (currently `pcpirates`)
- The service will automatically restart if it crashes (with a 10-second delay)
- Your system logs will be stored in the systemd journal

## Troubleshooting

**Service won't start?**
```bash
# Check for errors
sudo systemctl status frc-vision -l
sudo journalctl -u frc-vision -n 50

# Verify the Python script exists and is executable
ls -la /path/to/run_inference.py

# Test running the script manually
/home/pcpirates/Desktop/object-detection/.venv/bin/python3 /home/pcpirates/Desktop/object-detection/run_inference.py
```

**Service keeps crashing?**
```bash
# Check the logs for error messages
sudo journalctl -u frc-vision -f
```
