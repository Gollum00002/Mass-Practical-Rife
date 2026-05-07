#!/bin/bash
# Launch script for RIFE video interpolation
# Uses a Python virtual environment — no container needed.

# This script automatically determines the correct paths.
# It finds the directory where this script itself is located.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PYTHON_SCRIPT_NAME="inference-linux.py"
SCRIPT_PATH="$SCRIPT_DIR/$PYTHON_SCRIPT_NAME"
VENV_DIR="$SCRIPT_DIR/venv"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"

# Check if the main Python script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: The main Python script was not found at:"
    echo "$SCRIPT_PATH"
    echo "Please ensure this launch script is in the same directory as the Python script."
    read -p "Press Enter to Exit... "
    exit 1
fi

# --- Virtual Environment Setup ---
if [ ! -d "$VENV_DIR" ]; then
    echo "No virtual environment found. Creating one at: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        echo "Make sure python3-venv is installed: sudo dnf install python3-venv"
        read -p "Press Enter to Exit... "
        exit 1
    fi

    echo "Installing requirements from $REQUIREMENTS_FILE ..."
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install -r "$REQUIREMENTS_FILE"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install requirements. Check the output above for details."
        read -p "Press Enter to Exit... "
        exit 1
    fi
    echo "Virtual environment ready."
    echo ""
fi

# Activate the virtual environment (must happen before anything else runs)
source "$VENV_DIR/bin/activate"
PYTHON_BIN="$VENV_DIR/bin/python"

# --- Default Parameters ---
SCALE="1"
TARGET_FPS="120"
FP16="--fp16"
INPUT_DIR_ARG=""
OUTPUT_DIR="fpsConv"
MODEL_DIR_ARG=""

# Parse command line arguments for user overrides
while [[ $# -gt 0 ]]; do
    case $1 in
        --scale)
            SCALE="$2"
            shift 2
            ;;
        --target-fps)
            TARGET_FPS="$2"
            shift 2
            ;;
        --no-fp16)
            FP16=""
            shift
            ;;
        --input-dir)
            INPUT_DIR_ARG="--input-dir '$(realpath "$2")'"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model)
            MODEL_DIR_ARG="--model '$(realpath "$2")'"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--scale N] [--target-fps N] [--no-fp16] [--input-dir PATH] [--output PATH] [--model PATH]"
            exit 1
            ;;
    esac
done

echo "Running RIFE with the following settings:"
echo "  Project Directory: $SCRIPT_DIR"
echo "  Scale:             $SCALE"
echo "  Target FPS:        $TARGET_FPS"
echo "  FP16:              $([ -n "$FP16" ] && echo "Enabled" || echo "Disabled")"
echo "  Output Subfolder:  $OUTPUT_DIR"
if [ -n "$MODEL_DIR_ARG" ]; then
    echo "  Model:             Custom path provided by user"
else
    echo "  Model:             Default ('train_log')"
fi
echo ""

# Change to the script's directory and run
cd "$SCRIPT_DIR"
"$PYTHON_BIN" "$SCRIPT_PATH" --scale $SCALE --target-fps $TARGET_FPS $FP16 $INPUT_DIR_ARG --output "$OUTPUT_DIR" $MODEL_DIR_ARG

echo ""
stty sane 2>/dev/null
read -p "Press Enter to close this terminal... " </dev/tty

# Walk up the process tree to find and close the Konsole window cleanly
PID=$$
while [ "$PID" -gt 1 ]; do
    PID=$(awk '/^PPid:/{print $2}' /proc/$PID/status 2>/dev/null)
    COMM=$(cat /proc/$PID/comm 2>/dev/null)
    if [ "$COMM" = "konsole" ]; then
        kill "$PID"
        break
    fi
done