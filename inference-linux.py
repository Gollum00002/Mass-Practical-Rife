#!/usr/bin/env python3
import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import skvideo.io
from queue import Queue, Empty
import glob
import threading
import time
import shutil
import tempfile
import math
import re
import subprocess
import psutil
import signal
import sys
from pathlib import Path

warnings.filterwarnings("ignore")

# --- FIX: Global variable to track the current FFmpeg subprocess ---
g_current_subprocess = None

# Global variables for process management
current_output_file = None
processing_interrupted = False
exit_program = False
temp_dir = None
MAX_BUFFER_SIZE = 20  # Reduced from 500 to limit memory usage
CACHE_LIMIT = 100     # Maximum number of frames to keep in cache



def get_project_temp_dir():
    """
    Creates and returns the path to a temporary directory for processing.
    This version creates the temp folder in the current directory to avoid tmpfs space issues.
    """
    global temp_dir
    if temp_dir is None:
        current_dir = os.getcwd()
        temp_dir = os.path.join(current_dir, "enhanced_inference_temp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, mode=0o755, exist_ok=True)
    return temp_dir

def ensure_temp_dir_cleaned():
    global temp_dir
    if temp_dir is not None and os.path.isdir(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            temp_dir = None
        except Exception as e:
            tqdm.write(f"Warning: Could not clean temp directory: {e}", file=sys.stdout)

def check_ffmpeg_installed():
    """Check if FFmpeg is installed and accessible"""
    try:
        result = subprocess.run(['ffmpeg', '-version'],
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def check_system_requirements():
    """Check if all system requirements are met"""
    issues = []

    # Check FFmpeg
    if not check_ffmpeg_installed():
        issues.append("FFmpeg is not installed or not in PATH")

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"CUDA detected: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU (will be much slower)")

    # Check available memory
    mem = psutil.virtual_memory()
    if mem.available < 4 * 1024**3:  # Less than 4GB
        issues.append(f"Low available memory: {mem.available / 1024**3:.1f}GB (recommend at least 4GB)")

    if issues:
        print("System Requirements Issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    return True

def find_ffmpeg_path():
    """Find FFmpeg executable path"""
    common_paths = [
        '/usr/bin/ffmpeg',
        '/usr/local/bin/ffmpeg',
        '/opt/homebrew/bin/ffmpeg',  # macOS with Homebrew
        shutil.which('ffmpeg')
    ]

    for path in common_paths:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return 'ffmpeg'  # Fallback to system PATH

def transcode_to_mp4(src_path):
    ffmpeg_path = find_ffmpeg_path()
    base = os.path.splitext(os.path.basename(src_path))[0]
    tmp = os.path.join(get_project_temp_dir(), f"{base}_TMP.mp4")

    cmd = [
        ffmpeg_path, '-y', '-i', src_path,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-c:a', 'aac', '-b:a', '128k',
        tmp
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return tmp
    except subprocess.CalledProcessError as e:
        print(f"Transcoding failed: {e}")
        raise

def get_video_info(video_path):
    ffmpeg_path = find_ffmpeg_path()
    ffprobe_path = ffmpeg_path.replace('ffmpeg', 'ffprobe')

    info = {
        'width': 0,
        'height': 0,
        'display_aspect_ratio': None,
        'sample_aspect_ratio': '1:1',
        'fps': 0,
        'total_frames': 0,
        'duration': 0
    }

    try:
        cmd = [
            ffprobe_path, '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,display_aspect_ratio,sample_aspect_ratio,avg_frame_rate,nb_frames,duration',
            '-of', 'default=noprint_wrappers=1:nokey=0',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout.strip()

        for line in output.split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                if key == 'width':
                    info['width'] = int(value)
                elif key == 'height':
                    info['height'] = int(value)
                elif key == 'display_aspect_ratio':
                    info['display_aspect_ratio'] = value if value != 'N/A' else None
                elif key == 'sample_aspect_ratio':
                    info['sample_aspect_ratio'] = value if value != 'N/A' else '1:1'
                elif key == 'avg_frame_rate':
                    if '/' in value:
                        num, den = map(int, value.split('/'))
                        info['fps'] = num / den if den else 0
                    else:
                        info['fps'] = float(value)
                elif key == 'nb_frames':
                    info['total_frames'] = int(value) if value.isdigit() else 0
                elif key == 'duration':
                    info['duration'] = float(value) if value and value != 'N/A' else 0

        # Fallback to OpenCV if FFprobe data is incomplete
        if info['fps'] > 1000 or info['fps'] == 0:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                info['fps'] = cap.get(cv2.CAP_PROP_FPS)
                if info['width'] == 0:
                    info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                if info['height'] == 0:
                    info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if info['total_frames'] == 0:
                    info['total_frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

        # Calculate missing values
        if info['total_frames'] == 0 and info['duration'] > 0 and info['fps'] > 0:
            info['total_frames'] = int(info['duration'] * info['fps'])
        if info['duration'] == 0 and info['total_frames'] > 0 and info['fps'] > 0:
            info['duration'] = info['total_frames'] / info['fps']
        if not info['display_aspect_ratio'] and info['width'] and info['height']:
            gcd = math.gcd(info['width'], info['height'])
            info['display_aspect_ratio'] = f"{info['width']//gcd}:{info['height']//gcd}"

    except Exception as e:
        print(f"Error getting video info: {e}")
        # Fallback to OpenCV
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                info['fps'] = cap.get(cv2.CAP_PROP_FPS)
                info['total_frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if info['fps'] > 0:
                    info['duration'] = info['total_frames'] / info['fps']
                cap.release()
        except:
            pass

    return info

def parse_aspect_ratio(ratio_str):
    if not ratio_str or ':' not in ratio_str:
        return None
    try:
        w, h = map(float, ratio_str.split(':'))
        return w / h if h else None
    except:
        return None

def calculate_target_dimensions(original_w, original_h, display_aspect_ratio=None, auto_scale=True):
    if not auto_scale:
        return original_w, original_h
    target_width, target_height = 1280, 720
    dar = parse_aspect_ratio(display_aspect_ratio) if display_aspect_ratio else None

    if dar and dar > 0:
        current_ar = original_w / original_h
        if abs(current_ar - dar) > 0.01:
            if dar > current_ar:
                new_h = original_h
                new_w = int(new_h * dar)
            else:
                new_w = original_w
                new_h = int(new_w / dar)
            original_w, original_h = new_w, new_h

    ar = original_w / original_h
    if original_w > target_width or original_h > target_height:
        if ar > (target_width / target_height):
            new_w = target_width
            new_h = int(new_w / ar)
        else:
            new_h = target_height
            new_w = int(new_h * ar)
    else:
        new_w = original_w
        new_h = original_h

    new_w = (new_w // 2) * 2
    new_h = (new_h // 2) * 2
    return new_w, new_h

def transferAudio(sourceVideo, targetVideo):
    global g_current_subprocess
    ffmpeg_path = find_ffmpeg_path()
    process = None
    try:
        tmp_dir = get_project_temp_dir()
        audio_dir = os.path.join(tmp_dir, f"audio_{os.getpid()}_{int(time.time())}")
        os.makedirs(audio_dir, exist_ok=True)
        a_mkv = os.path.join(audio_dir, "audio.mkv")

        ffprobe_path = ffmpeg_path.replace('ffmpeg', 'ffprobe')
        audio_check = subprocess.run([
            ffprobe_path, '-i', sourceVideo, '-show_streams', '-select_streams', 'a'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if audio_check.returncode == 0:
            tqdm.write("\nTransferring audio...", file=sys.stdout)
            no_audio_video = os.path.join(audio_dir, "no_audio_video.mp4")
            shutil.copy2(targetVideo, no_audio_video)

            cmd = [
                ffmpeg_path, '-y', '-i', no_audio_video, '-i', sourceVideo,
                '-c', 'copy', '-map', '0:v:0', '-map', '1:a:0?',
                '-shortest', targetVideo
            ]

            process = subprocess.Popen(cmd, stderr=subprocess.PIPE, universal_newlines=True)
            g_current_subprocess = process
            
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)

    except Exception as e:
        if not exit_program:
            tqdm.write(f"Audio transfer failed: {e}", file=sys.stdout)
    finally:
        g_current_subprocess = None
        if 'audio_dir' in locals():
            shutil.rmtree(audio_dir, ignore_errors=True)

def finalize_video_with_ffmpeg(input_video, output_video, target_fps, original_w, original_h,
                               original_duration, display_aspect_ratio, sample_aspect_ratio):
    global g_current_subprocess
    ffmpeg_path = find_ffmpeg_path()
    process = None
    try:
        tmp_dir = get_project_temp_dir()
        work_dir = os.path.join(tmp_dir, f"ffmpeg_{os.getpid()}_{int(time.time())}")
        os.makedirs(work_dir, exist_ok=True)
        temp_output = os.path.join(work_dir, os.path.basename(output_video))

        try:
            ffprobe_path = ffmpeg_path.replace('ffmpeg', 'ffprobe')
            dur_str = subprocess.check_output([
                ffprobe_path, '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', input_video
            ], text=True).strip()
            dur = float(dur_str) if dur_str else original_duration
        except Exception:
            dur = original_duration

        pts_factor = original_duration / dur if dur and original_duration else 1.0

        enc_args = ['-c:v', 'libx264', '-preset', 'medium', '-profile:v', 'high', '-pix_fmt', 'yuv420p']
        try:
            enc_list = subprocess.run([ffmpeg_path, '-hide_banner', '-encoders'],
                                    capture_output=True, text=True, timeout=10)
            if 'h264_nvenc' in enc_list.stdout and original_w <= 4096 and original_h <= 4096:
                tqdm.write("\nUsing h264_nvenc hardware encoder.", file=sys.stdout)
                enc_args = ['-c:v', 'h264_nvenc', '-preset', 'p7', '-tune', 'hq',
                            '-b:v', '50M', '-maxrate', '100M', '-bufsize', '200M']
        except Exception:
            pass

        filters = []
        if display_aspect_ratio and parse_aspect_ratio(display_aspect_ratio):
            filters.append(f'setdar={display_aspect_ratio.replace(":", "/")}')

        filters.extend([
            f'setpts=PTS*{pts_factor}',
            f'scale={original_w}:{original_h}:flags=lanczos',
            'format=yuv420p'
        ])
        filter_str = ','.join(filters)

        cmd = [ffmpeg_path, '-y', '-fflags', '+genpts', '-i', input_video] + enc_args + [
            '-vf', filter_str,
            '-movflags', '+faststart',
            '-r', str(target_fps),
            temp_output
        ]

        tqdm.write("\nFinalizing video with FFmpeg (this may take a while)...", file=sys.stdout)

        process = subprocess.Popen(cmd, stderr=subprocess.PIPE, universal_newlines=True)
        g_current_subprocess = process
        
        for line in process.stderr:
            if exit_program: break
            tqdm.write(f"  FFmpeg: {line.strip()}", file=sys.stdout)
        process.wait()

        if process.returncode != 0:
            if not exit_program:
                raise subprocess.CalledProcessError(process.returncode, cmd)

        if os.path.exists(temp_output) and os.path.getsize(temp_output) > 0:
            shutil.copy2(temp_output, output_video)
            os.remove(input_video)
        elif not exit_program:
            raise Exception("FFmpeg output file is empty or missing")

    except Exception as e:
        if not exit_program:
            tqdm.write(f"Video finalization failed: {e}", file=sys.stdout)
            raise
    finally:
        g_current_subprocess = None
        if 'work_dir' in locals():
            shutil.rmtree(work_dir, ignore_errors=True)

def check_for_keypress():
    """Linux-compatible keypress detection using select"""
    global processing_interrupted, exit_program, g_current_subprocess
    import select
    import termios
    import tty

    try:
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())

        while not exit_program:
            if select.select([sys.stdin], [], [], 0.1) == ([sys.stdin], [], []):
                char = sys.stdin.read(1)
                if char == '/':
                    print("\n'/' pressed. Skipping current video...", flush=True)
                    processing_interrupted = True
                elif char == '?':
                    print("\n'Shift+/' pressed. Exiting...", flush=True)
                    exit_program = True
                    processing_interrupted = True
                    # Terminate any active FFmpeg subprocess
                    if g_current_subprocess and g_current_subprocess.poll() is None:
                        try:
                            g_current_subprocess.terminate()
                            g_current_subprocess.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            g_current_subprocess.kill()
                        except Exception:
                            g_current_subprocess.kill()
                    break
            time.sleep(0.1)
    except Exception:
        # Fallback for environments where tty manipulation is not possible
        while not exit_program:
            time.sleep(1)
    finally:
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except Exception:
            pass

@torch.inference_mode()
def process_frame(model, I0, I1, ratio, scale, h, w, original_h, original_w, device, fp16):
    try:
        # Validate input tensors
        if torch.isnan(I0).any() or torch.isnan(I1).any():
            tqdm.write("Warning: NaN detected in input frames, using fallback", file=sys.stdout)
            mid = I0 if ratio < 0.5 else I1
        elif torch.isinf(I0).any() or torch.isinf(I1).any():
            tqdm.write("Warning: Inf detected in input frames, using fallback", file=sys.stdout)
            mid = I0 if ratio < 0.5 else I1
        else:
            if hasattr(model, 'version') and model.version >= 3.9:
                mid = model.inference(I0, I1, ratio, scale)
            else:
                mid = model.inference(I0, I1, scale)

        if torch.isnan(mid[0]).any() or torch.isinf(mid[0]).any():
            tqdm.write("Warning: Invalid interpolation result, using fallback", file=sys.stdout)
            fallback = I0 if ratio < 0.5 else I1
            mid_frame = ((fallback[0] * 255).byte().cpu().numpy().transpose(1, 2, 0))
        else:
            mid_frame = ((mid[0] * 255).byte().cpu().numpy().transpose(1, 2, 0))

        mid_frame = mid_frame[:h, :w]

        if mid_frame.size == 0 or mid_frame.shape[0] == 0 or mid_frame.shape[1] == 0:
            tqdm.write("Warning: Empty frame detected, skipping", file=sys.stdout)
            return None

        mid_frame_bgr = cv2.cvtColor(mid_frame, cv2.COLOR_RGB2BGR)
        upscaled = cv2.resize(mid_frame_bgr, (original_w, original_h), interpolation=cv2.INTER_LANCZOS4)

        if upscaled.size == 0:
            tqdm.write("Warning: Upscaling failed, returning black frame", file=sys.stdout)
            return np.zeros((original_h, original_w, 3), dtype=np.uint8)

        blurred = cv2.GaussianBlur(upscaled, (5, 5), 0)
        result = cv2.addWeighted(upscaled, 1.5, blurred, -0.5, 0)

        if result.size == 0 or np.all(result == 0):
            tqdm.write("Warning: Result frame is empty or all black", file=sys.stdout)
            return upscaled

        return result

    except Exception as e:
        if not exit_program:
            tqdm.write(f"Error in process_frame: {str(e)}, using fallback", file=sys.stdout)
        try:
            fallback = I0 if ratio < 0.5 else I1
            fallback_frame = ((fallback[0] * 255).byte().cpu().numpy().transpose(1, 2, 0))
            fallback_frame = fallback_frame[:h, :w]
            fallback_frame_bgr = cv2.cvtColor(fallback_frame, cv2.COLOR_RGB2BGR)
            return cv2.resize(fallback_frame_bgr, (original_w, original_h), interpolation=cv2.INTER_LANCZOS4)
        except:
            return np.zeros((original_h, original_w, 3), dtype=np.uint8)

def check_memory_safety():
    mem = psutil.virtual_memory()
    return mem.available > 2 * 1024**3

def validate_frame(frame, expected_shape=None):
    if frame is None or frame.size == 0 or len(frame.shape) != 3: return False
    if expected_shape and frame.shape != expected_shape: return False
    if np.any(np.isnan(frame)) or np.any(np.isinf(frame)): return False
    if frame.dtype != np.uint8: return False
    return True

def get_actual_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return 0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count > 0:
        cap.release()
        return frame_count

    actual_count = 0
    while True:
        ret, _ = cap.read()
        if not ret: break
        actual_count += 1
    cap.release()
    return actual_count

def process_video(video_path, output_dir, target_fps, modelDir='train_log', fp16=False, scale=1.0, ext='mp4', png=False, auto_scale=True):
    global current_output_file, processing_interrupted, exit_program, args

    base_input_dir = os.path.abspath(args.input_dir if args.input_dir else os.getcwd())
    abs_video_path = os.path.abspath(video_path)
    rel_dir = os.path.dirname(abs_video_path)
    rel_path = os.path.relpath(rel_dir, base_input_dir)
    if rel_path == '.':
        rel_path = ''
    specific_output_dir = os.path.join(output_dir, rel_path)
    os.makedirs(specific_output_dir, exist_ok=True)
    video_filename = os.path.basename(os.path.splitext(video_path)[0])
    output_filepath = os.path.join(specific_output_dir, f"{video_filename}.{ext}")

    if os.path.exists(output_filepath) and os.path.getsize(output_filepath) > 1024:
        # Don't log every skip to avoid spam, just return
        return None

    start_time = time.time()
    inp = video_path
    is_tmp = False

    if os.path.splitext(video_path)[1].lower() not in ['.mp4', '.webm']:
        try:
            inp = transcode_to_mp4(video_path)
            is_tmp = True
        except Exception:
            inp = video_path

    try:
        video_info = get_video_info(inp)
        original_fps = video_info['fps']
        tot_frame = video_info['total_frames']
        original_w = video_info['width']
        original_h = video_info['height']
        duration = video_info['duration']
        display_aspect_ratio = video_info['display_aspect_ratio']
        sample_aspect_ratio = video_info['sample_aspect_ratio']

        if original_fps <= 0.1:
            tqdm.write(f"Invalid FPS {original_fps}, skipping", file=sys.stdout)
            return False

        if original_fps >= target_fps - 0.5:
            tqdm.write(f"\nSkipping (matching FPS): {video_path}", file=sys.stdout)
            tqdm.write(f"  -> Copying to output folder: {output_filepath}", file=sys.stdout)
            try:
                shutil.copy2(video_path, output_filepath)
                tqdm.write(f"  -> Copied successfully.", file=sys.stdout)
            except Exception as copy_err:
                tqdm.write(f"  -> Copy failed: {copy_err}", file=sys.stdout)
            return None

        actual_frame_count = get_actual_frame_count(inp)
        if actual_frame_count == 0:
            tqdm.write(f"No readable frames found in video: {video_path}", file=sys.stdout)
            return False

        tot_frame = min(tot_frame, actual_frame_count) if tot_frame > 0 else actual_frame_count
        if tot_frame > 0 and original_fps > 0:
            duration = tot_frame / original_fps

        new_w, new_h = calculate_target_dimensions(
            original_w, original_h, display_aspect_ratio=display_aspect_ratio, auto_scale=auto_scale)
        interpolation_method = cv2.INTER_AREA if (new_w < original_w or new_h < original_h) else cv2.INTER_LINEAR

        tqdm.write(f"\nProcessing video: {video_path}", file=sys.stdout)
        tqdm.write("  [Press '/' to skip this video | Press 'Shift+/' (?) to exit]", file=sys.stdout)
        tqdm.write(f"  Original: {original_w}x{original_h} @ {original_fps:.2f} FPS ({int(tot_frame)} frames, {duration:.2f}s)", file=sys.stdout)
        tqdm.write(f"  Processing at: {new_w}x{new_h}", file=sys.stdout)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # inference_mode decorator on process_frame replaces set_grad_enabled
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if fp16:
                torch.set_default_tensor_type(torch.cuda.HalfTensor)

        try:
            sys.path.insert(0, modelDir)
            from RIFE_HDv3 import Model
            model = Model()
            model.load_model(modelDir, -1)
            model.eval()
            model.device()
        except Exception as e:
            tqdm.write(f"Failed to load model: {e}", file=sys.stdout)
            return False

        source_frame_time = 1.0 / original_fps
        target_frame_count = math.ceil(duration * target_fps)
        source_timestamps = [i * source_frame_time for i in range(int(tot_frame))]
        target_timestamps = np.linspace(0, duration, num=target_frame_count, endpoint=False).tolist()

        frame_mapping = []
        current_src_idx = 0
        for target_time in target_timestamps:
            if current_src_idx >= int(tot_frame) - 1:
                frame_mapping.append((int(tot_frame)-1, int(tot_frame)-1, 1.0))
            else:
                while current_src_idx < int(tot_frame)-1 and source_timestamps[current_src_idx+1] <= target_time + 1e-6:
                    current_src_idx += 1
                next_src_idx = min(current_src_idx+1, int(tot_frame)-1)
                time_diff = source_timestamps[next_src_idx] - source_timestamps[current_src_idx]
                ratio = (target_time - source_timestamps[current_src_idx]) / time_diff if time_diff else 0
                ratio = max(0.0, min(1.0, ratio))
                frame_mapping.append((current_src_idx, next_src_idx, ratio))

        cap = cv2.VideoCapture(inp)
        if not cap.isOpened():
            tqdm.write(f"Failed to open video: {inp}", file=sys.stdout)
            return False

        ret, lastframe = cap.read()
        if not ret or not validate_frame(lastframe):
            tqdm.write(f"Failed to read first frame from video: {inp}", file=sys.stdout)
            cap.release()
            return False

        lastframe = cv2.resize(lastframe, (new_w, new_h), interpolation=interpolation_method)
        h, w, _ = lastframe.shape

        vid_out = None
        if not png:
            current_output_file = os.path.join(get_project_temp_dir(), f"temp_{video_filename}.{ext}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vid_out = cv2.VideoWriter(current_output_file, fourcc, target_fps, (original_w, original_h))

        write_buffer = Queue(maxsize=MAX_BUFFER_SIZE)
        frame_cache = {0: lastframe}

        # RIFE HD requires 32-pixel alignment; 64 gives a safe margin and
        # is much smaller than the original 256, reducing tensor size.
        padding_tmp = max(64, int(64/scale))
        ph = ((h-1)//padding_tmp+1)*padding_tmp
        pw = ((w-1)//padding_tmp+1)*padding_tmp
        padding = (0, pw-w, 0, ph-h)
        tqdm.write(f'  [pad] Processing tensor: {pw}x{ph} (frame: {w}x{h})', file=sys.stdout)

        def process_batch_sync(batch_mappings, frame_cache):
            results = []
            for idx, mapping in enumerate(batch_mappings):
                if exit_program: break
                i1, i2, r = mapping
                if i1 == i2 or r == 0:
                    if i1 in frame_cache:
                        frame_output = cv2.resize(frame_cache[i1], (original_w, original_h), interpolation=cv2.INTER_LANCZOS4)
                        results.append((idx, frame_output))
                    continue

                if i1 in frame_cache and i2 in frame_cache:
                    try:
                        frame1_rgb = cv2.cvtColor(frame_cache[i1], cv2.COLOR_BGR2RGB)
                        frame2_rgb = cv2.cvtColor(frame_cache[i2], cv2.COLOR_BGR2RGB)

                        I0 = torch.from_numpy(frame1_rgb.transpose(2,0,1)).to(device).unsqueeze(0).float()/255.
                        I1 = torch.from_numpy(frame2_rgb.transpose(2,0,1)).to(device).unsqueeze(0).float()/255.
                        I0 = F.pad(I0, padding)
                        I1 = F.pad(I1, padding)
                        if fp16: I0, I1 = I0.half(), I1.half()

                        frm = process_frame(model, I0, I1, r, scale, h, w, original_h, original_w, device, fp16)
                        if frm is not None:
                            results.append((idx, frm))
                        else:
                            fallback_frame = frame_cache[i1] if r < 0.5 else frame_cache[i2]
                            fallback_resized = cv2.resize(fallback_frame, (original_w, original_h), interpolation=cv2.INTER_LANCZOS4)
                            results.append((idx, fallback_resized))
                    except Exception as e:
                        if not exit_program:
                           tqdm.write(f"Error processing frame pair ({i1}, {i2}): {str(e)}", file=sys.stdout)
                        fallback_frame = frame_cache[i1] if r < 0.5 else frame_cache[i2]
                        fallback_resized = cv2.resize(fallback_frame, (original_w, original_h), interpolation=cv2.INTER_LANCZOS4)
                        results.append((idx, fallback_resized))
            return sorted(results, key=lambda x: x[0])

        def clear_write_buffer(write_buffer, png_dir=None):
            cnt = 0
            while not exit_program:
                try:
                    item = write_buffer.get(timeout=0.5)
                    if item is None: break
                    if png:
                        cv2.imwrite(os.path.join(png_dir, f'{cnt:07d}.png'), cv2.cvtColor(item, cv2.COLOR_BGR2RGB))
                        cnt += 1
                    else:
                        vid_out.write(item)
                except Empty:
                    continue

        write_thread = threading.Thread(target=clear_write_buffer, args=(write_buffer, os.path.join(specific_output_dir, video_filename) if png else None))
        write_thread.daemon = True
        write_thread.start()

        pbar_desc = os.path.basename(video_path)
        gen_pbar = tqdm(total=len(frame_mapping), desc=f"  -> Interp: {pbar_desc}", position=1, leave=False)

        frame_cursor = 1
        video_ended = False

        for batch_start in range(0, len(frame_mapping), 8):
            if exit_program or processing_interrupted: break

            batch_end = min(batch_start + 8, len(frame_mapping))
            batch = frame_mapping[batch_start:batch_end]

            highest_needed = max(max(i1, i2) for i1, i2, _ in batch)
            while frame_cursor <= highest_needed and not video_ended:
                if exit_program or processing_interrupted: break
                ret, frame = cap.read()
                if not ret:
                    video_ended = True
                    break

                if validate_frame(frame, (original_h, original_w, 3)):
                    frame_cache[frame_cursor] = cv2.resize(frame, (new_w, new_h), interpolation=interpolation_method)
                else:
                    frame_cache[frame_cursor] = frame_cache[frame_cursor-1].copy()
                frame_cursor += 1

            res = process_batch_sync(batch, frame_cache)
            for _, frm in res:
                if frm is not None:
                    while not exit_program and not processing_interrupted:
                        try:
                            write_buffer.put(frm, timeout=0.2)
                            break
                        except Exception:
                            continue

            gen_pbar.update(len(res))

            needed_keys = set(i for m in frame_mapping[batch_end:batch_end+CACHE_LIMIT] for i in m[:2])
            keys_to_del = [k for k in frame_cache if k not in needed_keys and k < frame_cursor - (CACHE_LIMIT // 2)]
            for k in keys_to_del: del frame_cache[k]

        gen_pbar.close()
        cap.release()

        # Drain the queue first if we're exiting, so the sentinel can always get in
        if exit_program or processing_interrupted:
            while not write_buffer.empty():
                try:
                    write_buffer.get_nowait()
                except Exception:
                    break
        write_buffer.put(None)

        while write_thread.is_alive():
            write_thread.join(timeout=0.5)
        
        if vid_out: vid_out.release()

        if exit_program or processing_interrupted:
            if os.path.exists(current_output_file): os.remove(current_output_file)
            print(f"\nProcessing of {video_path} was interrupted.", flush=True)
            # Reset interruption flag for the next file unless exiting program
            if not exit_program:
                processing_interrupted = False
            return False

        if not png:
            finalize_video_with_ffmpeg(current_output_file, output_filepath, target_fps,
                                      original_w, original_h, duration, display_aspect_ratio, sample_aspect_ratio)
            if not exit_program:
                transferAudio(video_path, output_filepath)
        
        tqdm.write(f"  -> Finished in {time.time()-start_time:.2f} seconds.", file=sys.stdout)
        return True

    except Exception as e:
        if not exit_program:
            tqdm.write(f"Error processing {video_path}: {e}", file=sys.stdout)
        return False
    finally:
        if 'cap' in locals() and cap.isOpened(): cap.release()
        if is_tmp and os.path.exists(inp):
            try: os.remove(inp)
            except PermissionError: pass
        ensure_temp_dir_cleaned()

def find_all_videos(root_dir, output_dir):
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    video_files = []
    abs_out = os.path.abspath(output_dir)
    for d, _, files in os.walk(root_dir):
        if os.path.abspath(d).startswith(abs_out): continue
        for f in files:
            if os.path.splitext(f)[1].lower() in video_extensions:
                video_files.append(os.path.join(d, f))
    # Sort to ensure consistent order across runs
    return sorted(video_files)

def process_videos_sequential(video_files, args):
    global processing_interrupted, exit_program
    
    # First pass: quickly count what needs to be done
    print(f"\nQuick scan of {len(video_files)} videos...")
    to_process = []
    already_done = 0
    
    for vp in video_files:
        base_input_dir = os.path.abspath(args.input_dir if args.input_dir else os.getcwd())
        abs_video_path = os.path.abspath(vp)
        rel_dir = os.path.dirname(abs_video_path)
        rel_path = os.path.relpath(rel_dir, base_input_dir)
        if rel_path == '.':
            rel_path = ''
        specific_output_dir = os.path.join(args.output, rel_path)
        video_filename = os.path.basename(os.path.splitext(vp)[0])
        output_filepath = os.path.join(specific_output_dir, f"{video_filename}.{args.ext}")
        
        if os.path.exists(output_filepath) and os.path.getsize(output_filepath) > 1024:
            already_done += 1
        else:
            to_process.append(vp)
    
    print(f"  ✓ Already processed: {already_done} files")
    print(f"  → Need to process: {len(to_process)} files")
    print()
    
    if not to_process:
        print("All files already processed!")
        return
    
    overall_pbar = tqdm(total=len(to_process), desc="Overall Progress", position=0, leave=True)
    summary = {'Processed': 0, 'Skipped': 0, 'Failed': 0}

    for i, vp in enumerate(to_process):
        if exit_program: break

        overall_pbar.set_description(f"Processing: {os.path.basename(vp)}")
        result = process_video(
            video_path=vp, output_dir=args.output, target_fps=args.target_fps,
            modelDir=args.modelDir, fp16=args.fp16, scale=args.scale, ext=args.ext,
            png=args.png, auto_scale=not args.disable_auto_scale)

        if result is True: 
            summary['Processed'] += 1
        elif result is None: 
            summary['Skipped'] += 1
        else: 
            summary['Failed'] += 1

        overall_pbar.update(1)
        overall_pbar.set_postfix(summary)
        ensure_temp_dir_cleaned()

    overall_pbar.close()
    if not exit_program:
        tqdm.write("\n" + "="*60, file=sys.stdout)
        tqdm.write("PROCESSING COMPLETE!", file=sys.stdout)
        tqdm.write(f"  Total videos found: {len(video_files)}", file=sys.stdout)
        tqdm.write(f"  Already done (skipped): {already_done}", file=sys.stdout)
        tqdm.write(f"  Successfully processed this run: {summary['Processed']}", file=sys.stdout)
        tqdm.write(f"  Skipped (matching FPS): {summary['Skipped']}", file=sys.stdout)
        tqdm.write(f"  Failed: {summary['Failed']}", file=sys.stdout)
        tqdm.write("="*60, file=sys.stdout)

def main():
    global exit_program, args

    if not check_system_requirements():
        print("\nPlease install missing requirements and try again.")
        return 1

    parser = argparse.ArgumentParser(description="RIFE Video Frame Interpolation (Linux Optimized)")
    parser.add_argument('--output', type=str, default='fpsConv', help='Output directory')
    parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='RIFE model directory')
    parser.add_argument('--fp16', action='store_true', help='Use half precision')
    parser.add_argument('--scale', type=float, default=1.0, help='Processing scale factor')
    parser.add_argument('--png', action='store_true', help='Output as PNG sequence')
    parser.add_argument('--ext', type=str, default='mp4', help='Output video extension')
    parser.add_argument('--target-fps', type=int, default=60, help='Target frame rate')
    parser.add_argument('--input-dir', type=str, default='', help='Input directory (default: current)')
    parser.add_argument('--disable-auto-scale', action='store_true', help='Disable auto resolution scaling')
    args = parser.parse_args()

    model_path = Path(args.modelDir)
    if not model_path.exists() or not any(model_path.glob("RIFE_HD*.py")):
        print(f"Error: RIFE model files not found in '{args.modelDir}'")
        return 1

    input_dir = args.input_dir if args.input_dir else os.getcwd()
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist")
        return 1

    os.makedirs(args.output, exist_ok=True)

    print(f"RIFE Video Processing (Linux)")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {args.output}")
    print(f"Target FPS: {args.target_fps}")
    print(f"Using {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"FP16 mode: {'Enabled' if args.fp16 else 'Disabled'}")


    keyboard_thread = threading.Thread(target=check_for_keypress, daemon=True)
    keyboard_thread.start()
        
    try:
        video_files = find_all_videos(input_dir, args.output)
        if not video_files:
            print("No video files found in input directory")
            return 0

        print(f"\nFound {len(video_files)} video files to process")
        process_videos_sequential(video_files, args)

    except Exception as e:
        print(f"An unexpected error occurred in main: {e}")
    finally:
        exit_program = True
        print("\nFinal cleanup...")
        ensure_temp_dir_cleaned()

    return 0

if __name__ == "__main__":
    sys.exit(main())