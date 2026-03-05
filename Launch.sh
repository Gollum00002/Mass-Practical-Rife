#!/bin/bash
# save as: run_rife_linux.sh

# --- Configuration ---
# The name of your distrobox container.
DISTROBOX_NAME="FedForWork"
# --- End of Configuration ---

# This script automatically determines the correct paths.
# It finds the directory where this script itself is located.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PYTHON_SCRIPT_NAME="enhanced-inference-video-linux.py"
SCRIPT_PATH="$SCRIPT_DIR/$PYTHON_SCRIPT_NAME"

# Check if the main Python script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: The main Python script was not found at:"
    echo "$SCRIPT_PATH"
    echo "Please ensure this launch script is in the same directory as the Python script."
    read -p "Press Enter to Exit... "
    exit 1
fi

# Default parameters
SCALE="1"
TARGET_FPS="120"
FP16="--fp16"
# The Python script's --input-dir will default to its current working directory
INPUT_DIR_ARG=""
OUTPUT_DIR="fpsConv"
MODEL_DIR_ARG="" # Let the Python script use its default 'train_log'

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
            # Resolve to an absolute path for the container
            INPUT_DIR_ARG="--input-dir '$(realpath "$2")'"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model)
            # Resolve to an absolute path for the container
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

# We build a single command string that first changes to the script's directory,
# and then executes the python script. This is the universal way to set the working directory.
PYTHON_PART="python3 '$SCRIPT_PATH' --scale $SCALE --target-fps $TARGET_FPS $FP16 $INPUT_DIR_ARG --output '$OUTPUT_DIR' $MODEL_DIR_ARG"
COMMAND="cd '$SCRIPT_DIR' && $PYTHON_PART"

echo "Running RIFE with the following settings:"
echo "  Project Directory: $SCRIPT_DIR"
echo "  Scale: $SCALE"
echo "  Target FPS: $TARGET_FPS"
echo "  FP16: $([ -n "$FP16" ] && echo "Enabled" || echo "Disabled")"
echo "  Output Subfolder: $OUTPUT_DIR"
if [ -n "$MODEL_DIR_ARG" ]; then
    echo "  Model: Custom path provided by user"
else
    echo "  Model: Default ('train_log')"
fi
echo ""

# Execute the combined 'cd && python' command inside the container's shell
distrobox enter "$DISTROBOX_NAME" -- bash -c "$COMMAND"

echo ""
read -p "Press Enter to Exit... "