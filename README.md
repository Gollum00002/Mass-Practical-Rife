# Mass-Practical-Rife (MPR)

This project started out of frustration with low-framerate video being everywhere online — 24fps, 30fps, sometimes as low as 5fps. MPR fixes that. It uses **Practical-RIFE 4.26** and **FFmpeg** to interpolate entire directories of videos to a target framerate in one go, with zero user interaction required between files.

This is the first program I ever developed, originally built back in 2024. Over the years it has been steadily improved to the point where I'm confident putting it in front of everyone.

> ⚠️ This is a hobbyist project maintained in my spare time. Updates will be infrequent — don't expect regular releases.

> Built on [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) by hzwer — without their work, this project would not exist.

---

## ✨ Features

### 🧠 AI-Powered Frame Interpolation
- Uses the **RIFE HDv3** (Real-Time Intermediate Flow Estimation) model to synthesize new frames between existing ones — not frame duplication
- Produces smooth, temporally accurate motion
- Configurable target frame rate (default: 60 FPS)
- Precise timestamp-based frame mapping ensures interpolated frames land at the correct point in time

### ⚡ GPU Acceleration
- Full **CUDA** support for fast inference on NVIDIA GPUs
- **FP16 (half-precision)** mode reduces VRAM usage and increases throughput on supported hardware
- Automatic **NVENC hardware encoding** used for final output when available, with software (`libx264`) fallback
- Gracefully falls back to **CPU** if no CUDA device is detected

### 📂 Batch Directory Processing
- Recursively scans an input directory and processes all supported video files
- Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`, `.flv`, `.webm`, `.m4v`
- Preserves folder structure in the output directory — your file tree stays intact
- **Smart skip**: automatically detects already-processed output files and skips them, so interrupted runs can be resumed without redoing work

### 🎞️ Resolution & Aspect Ratio Handling
- **Auto-scale** downscales videos larger than 1280×720 during processing, then upscales the final output back to the original resolution — a 4K input will remain 4K
- Correctly handles **non-square pixels** (SAR/DAR) by reading `display_aspect_ratio` and `sample_aspect_ratio` from stream metadata and preserving them in the output
- Uses **Lanczos resampling** for high-quality frame upscaling

### 🔊 Audio Preservation
- Automatically transfers the original audio track to the interpolated output
- Audio muxing is done via FFmpeg stream copy — no re-encoding, no quality loss

### 🛡️ Robust Frame Validation
- **Black and empty frames** are automatically detected and replaced with the nearest valid frame, preventing corrupted output from bad source material
- Frames containing NaN or infinite values (from GPU precision errors) are caught and substituted with a clean fallback frame
- Corrupt or unreadable frames never crash the run — they are handled silently and processing continues

### 🚦 Interruption Safety
- Press `/` at any time to **skip the current video** and continue with the next one
- Press `Shift+/` (`?`) to **exit the program** cleanly — active FFmpeg processes are terminated and temp files are cleaned up
- Incomplete output files from interrupted runs are automatically deleted so you never end up with a corrupt file

### 📊 Progress Reporting
- Per-file progress bar showing interpolation status frame-by-frame
- Overall progress bar showing how many files have been processed, skipped, or failed across the whole batch run
- Final summary with counts for processed, skipped (already done / matching FPS), and failed files

---

## 🚀 Usage

### Configuration

Open `launch.sh` and set the following parameters before running:

| Parameter | Description |
|---|---|
| `DISTROBOX_NAME` | Name of your Distrobox container |
| `SCALE` | Processing scale factor. Keep at `1` — this temporarily lowers resolution during interpolation for speed but does **not** affect the final output resolution |
| `TARGET_FPS` | The frame rate to interpolate to (e.g. `60`, `120`). Higher values mean longer processing time |
| `OUTPUT_DIR` | Path/name of the output folder. If no path is given, the folder is created in the same directory as `launch.sh` |
| `INPUT_DIR_ARG` | Path to your input directory. If left empty, the script scans all folders and subdirectories in its own directory (excluding `OUTPUT_DIR`) |

### Running

Cd into MPR Folder

```bash
./launch.sh
```
For fish use 'bash Launch.sh'
---

## 📋 Requirements

- Python 3.x
- FFmpeg (must be in `PATH`)
- NVIDIA GPU with CUDA (recommended) or CPU fallback
- 4 GB+ available RAM

---

## 🙏 Credits

This project is built on **[Practical-RIFE](https://github.com/hzwer/Practical-RIFE)** by [hzwer](https://github.com/hzwer), licensed under the MIT License. The included model weights are also from this project and covered under the same license.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
