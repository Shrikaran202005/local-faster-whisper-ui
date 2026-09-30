# 🎙️ Faster-Whisper ASR Studio

A lightweight, high-performance, local Automatic Speech Recognition (ASR) web application built using **Python**, **Faster-Whisper** (CTranslate2 execution engine), and **Gradio**.

This application allows users to transcribe audio files or live microphone recordings locally without relying on external cloud APIs. It includes features such as Voice Activity Detection (VAD) filtering, model size switching, multi-language support, timestamped segment breakdowns, and export to `.TXT` and `.SRT` subtitle formats.

---

## 🚀 Features

- **Local & Private**: All transcription is performed locally on your machine.
- **Ultra-Fast Performance**: Utilizes `faster-whisper` based on CTranslate2, running up to **4x faster** than standard OpenAI Whisper with lower memory footprint.
- **CPU & GPU Ready**: Configured to run out-of-the-box on CPU using 8-bit quantization (`int8`), with easy toggle to NVIDIA GPU (`cuda` with `float16`).
- **Voice Activity Detection (VAD)**: Integrated Silero VAD to crop leading/trailing silences and eliminate phantom transcriptions during quiet sections.
- **Export Options**: Download generated transcriptions as plain text (`.txt`) or formatted subtitle files (`.srt`).
- **Flexible Inputs**: Supports file uploads (`.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`) and live microphone recording directly in the browser.

---

## 🛠️ Tools & Technologies Used

| Technology | Purpose & Rationale |
| :--- | :--- |
| **Python (3.9+)** | Primary programming language offering rich ecosystem for ML/AI applications. |
| **Faster-Whisper** | Re-implementation of OpenAI's Whisper model using **CTranslate2**, a custom inference engine for Transformer models that delivers superior speed and reduced RAM/VRAM usage. |
| **Gradio** | Modern Python framework for rapidly constructing interactive web interfaces for machine learning models. |
| **FFmpeg** | Critical multimedia framework required by Whisper to decode, slice, and resample audio files into 16kHz mono audio. |
| **PyTorch** | Deep learning framework underlying tensor operations and device capabilities. |

---

## 📋 Prerequisites

Before installing the project, ensure you have the following installed:

1. **Python 3.9 or higher**: Download from [python.org](https://www.python.org/downloads/).
2. **Git**: Installed on your system for version control.
3. **FFmpeg**: Required for audio decoding (see installation instructions below).

---

## ⚙️ System Dependencies: Installing FFmpeg

FFmpeg must be installed and added to your system's PATH.

### 🪟 Windows
Option 1: Using **winget** (Recommended):
```cmd
winget install FFmpeg
```

Option 2: Using **Chocolatey**:
```cmd
choco install ffmpeg
```

Option 3: Manual Installation:
1. Download full build zip from [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/).
2. Extract to `C:\ffmpeg`.
3. Add `C:\ffmpeg\bin` to your System Environment Variables under `Path`.

### 🍎 macOS
Using **Homebrew**:
```bash
brew install ffmpeg
```

### 🐧 Linux (Ubuntu/Debian)
Using **APT**:
```bash
sudo apt update && sudo apt install -y ffmpeg
```

*Verify FFmpeg installation in your terminal:*
```bash
ffmpeg -version
```

---

## 📥 Installation & Setup

1. **Clone the Repository** (or navigate to the project directory):
   ```bash
   git clone https://github.com/YOUR_USERNAME/faster-whisper-asr-studio.git
   cd faster-whisper-asr-studio
   ```

2. **Create a Virtual Environment**:
   - **Windows**:
     ```cmd
     
     
     ```
   - **macOS/Linux**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 💻 Running the Application

Start the web interface with:
```bash
python app.py
```

Once launched, open your web browser and navigate to:
```
http://127.0.0.1:7860
```

---

## ⚡ Enabling GPU Acceleration (NVIDIA CUDA)

By default, the application runs on CPU with `int8` quantization. If you have an NVIDIA GPU:

1. Install PyTorch with CUDA support matching your system drivers:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
2. Install cuDNN libraries required by CTranslate2:
   ```bash
   pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
   ```
3. In the Web UI under **Model & Performance Settings**, change **Compute Device** from `cpu` to `cuda`.

---

## 📁 Project Structure

```
.
├── app.py              # Main Gradio application script
├── requirements.txt    # Python dependencies list
├── README.md           # Comprehensive project documentation
└── .gitignore          # Version control ignore list
```

