<p align="center">
  <img src="assets/logo.png" alt="QualityShoot Logo" width="120"/>
</p>

# QualityShoot Backend

Welcome to the **Backend** of **QualityShoot** — the AI-powered image & video upscaler that gives blurry pixels a second chance.  
Built in **Python** (with a pinch of magic and a healthy dose of trial-and-error) by [FOX](https://github.com/Tiger-Foxx).

---

## 🌍 Project Context & Motivation

### Why QualityShoot?

Ever wanted to make your old or low-res images/videos look like they were shot on the latest hardware?  
Tired of sending your personal files to mysterious cloud services, or fighting with command lines and random AI scripts from the internet?

**QualityShoot** was born to solve all that:
- **A cross-platform, open source, local upscaler** that respects your privacy and your time.
- **Blazing fast AI transformations** with a slick GUI (see the frontend repo!).
- **No internet, no sketchy servers, no lost data — you’re in control.**

#### The Backend’s Role

This repo is the **core engine** of QualityShoot:
- It runs the AI models for upscaling images and videos.
- It exposes a local API/process that the Electron frontend can call to process files, check progress, and manage jobs.
- It handles the heavy-lifting — so users get results, not headaches.

---

## 🧠 Features

- **Multiple AI upscaling engines**: (ESRGAN, Real-ESRGAN, custom models, etc.)
- **Image and video support**: JPEG, PNG, MP4, WebM, etc.
- **Batch processing**: Handle multiple files at once.
- **Progress reporting**: Communicates status to the frontend in real time.
- **CLI and API compatibility**: Use directly or integrated with the desktop app.
- **Platform independence**: Windows, macOS, Linux (with some quirks, see below!)
- **Automatic model download/management**: No more hunting for weight files.
- **Crash resistance**: (well, mostly — see the "peripeties" below!)

---

## ⚡ Architecture Overview

- **Python 3.x** backend, structured as a command-line tool and API process.
- **Main entry**: `main.py` (or compiled to `main.exe` via PyInstaller for production).
- **Serves requests**: via HTTP (Flask/FastAPI) or by command-line invocations from Electron.
- **Image/video processing**: Uses OpenCV, FFmpeg, NumPy, Pillow, and AI model libraries.
- **Job management**: Handles queues, progress, errors, and reporting back to the frontend.
- **Configurable**: Models, output quality, format, temp folder, logging, etc.

---

## 💾 Installation & Usage

### Prerequisites

- **Python 3.9+** (recommended: 3.10+ for best compatibility)
- **pip** (Python package manager)
- **Git** (to clone this repo)
- **FFmpeg** (must be in your PATH)
- **CUDA** (optional, for GPU acceleration)

### Set Up Locally

```bash
git clone https://github.com/Tiger-Foxx/QualityShoot---backend.git
cd QualityShoot---backend
pip install -r requirements.txt
```

#### Running in Dev Mode

```bash
python main.py
```
> This will start the local API server or CLI, ready to process jobs.

#### Building for Production

QualityShoot frontend expects a single binary (Windows: `main.exe`).  
Build using [PyInstaller](https://pyinstaller.org/):

```bash
pyinstaller --onefile main.py
```
> The resulting `dist/main.exe` can be bundled by the Electron frontend.

---

## 🗂️ Directory Structure

```
.
├── assets/             # Logo, test images, sample videos, model files
├── models/             # AI upscaling models (downloaded/managed automatically)
├── src/                # Core backend source code
│   ├── api/            # REST or CLI API handlers
│   ├── jobs/           # Job queue and progress logic
│   ├── processing/     # Image/video upscaling logic
│   └── utils/          # Helper functions, logging, etc.
├── tests/              # Unit and integration tests
├── requirements.txt    # Python dependencies
├── main.py             # Entry point
├── README.md           # You are here!
```

---

## 🤪 Dev Peripeties & Anecdotes

- **Virtualenv vortex**: Accidentally upscaled my virtual environment. Don’t ask.
- **CUDA roulette**: Works on one GPU, not on another. If your PC sounds like a jet, it’s working!
- **PyInstaller pain**: Bundling a Python project into a single EXE is black magic (with a side of DLL hell).
- **FFmpeg fumbles**: If the backend crashes, it’s probably FFmpeg’s fault (or yours, but mostly FFmpeg).
- **Model downloads**: Network down? No weights for you!
- **Progress reporting**: Sometimes the progress bar goes backwards. It’s not a bug, it’s a feature (for time travelers).
- **API vs CLI**: Supported both, so users can automate or click buttons to their heart’s content.
- **File locks**: Windows loves to keep files open. If you can’t delete something, reboot or rage.

---

## 🧩 Frontend Integration

- **Electron frontend** launches this backend as a process (either via CLI or HTTP API).
- **IPC**: Communicates via command line arguments, pipes, or HTTP endpoints.
- **Status & progress**: The backend reports status, errors, and completion — the frontend turns that into pretty bars and emojis.
- **Output**: All results are written to a user-specified output folder; no temp files left behind (unless you crash it).

---

## 👨‍💻 Contribution & Community

Pull requests, issues, and hilarious bug reports are welcome.

- Fork, clone, and branch away!
- Write clean Python, annotate types if you can.
- Document new models or processing flows.
- If you find a way to make FFmpeg behave, you win eternal respect.

---

## 📃 License

This project is **open source** — see the [LICENSE](./LICENSE) file for details.  
Feel free to fork, tinker, and break (then fix) things!

---

## 🦊 About the Author

Made by [FOX](https://github.com/Tiger-Foxx)  
More projects, tools, and pixels: [site.the-fox.tech](https://site.the-fox.tech)

---

<p align="center">
  <b>With great resolution comes great responsibility.</b>
</p>