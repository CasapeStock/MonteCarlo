# Monte Carlo Stock Simulation App - Installation Guide

## Prerequisites
- Python 3.8+
- Windows 10/11
- Visual Studio Build Tools
- Git

## Detailed Installation Steps

### 1. Install Visual Studio Build Tools
1. Download Visual Studio Build Tools from Microsoft's website
2. During installation, select:
   - "Desktop development with C++"
   - Windows 10/11 SDK
   - MSVC v143 C++ x64/x86 build tools

### 2. Install Python Dependencies
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install Kivy dependencies first
pip install kivy_deps.sdl2==0.8.0
pip install kivy_deps.gstreamer==0.3.4

# Install project dependencies
pip install flask yfinance numpy pandas matplotlib buildozer
pip install kivy==2.1.0
```

### 3. Troubleshooting
- If you encounter SDL2 or Cython issues, manually download and install from:
  - SDL2: https://www.libsdl.org/download-2.0.php
  - Cython: `pip install cython==0.29.28`

### 4. Running the Application
```bash
# Web Server
python app/server.py

# Mobile App
python app/main.py
```

## Android Packaging (Optional)
Requires additional setup with Buildozer and Android SDK.
