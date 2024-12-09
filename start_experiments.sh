#!/bin/bash

# Function to run experiment
exp_name="mul4_p5_token_number_4_categories"


run_experiment() {
    local cuda_device=$1
    nohup bash -c "CUDA_VISIBLE_DEVICES=$cuda_device python -u finetune.py --exp_name $exp_name --weights_preset_file /shared/share_mala/hc3295/finetune_mixture/weight_presets/weights_${cuda_device}.npy" > outs/device_${cuda_device}.out &
}

# Parse command line arguments
devices=()


while [[ $# -gt 0 ]]; do
    case $1 in
        --devices)
            shift
            while [[ $# -gt 0 && ! $1 == --* ]]; do
                devices+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if required arguments are provided
if [ ${#devices[@]} -eq 0 ]; then
    echo "Usage: $0 --devices <device1> <device2> ..."
    exit 1
fi

# Create the 'outs' directory if it doesn't exist
mkdir -p outs

# Launch experiments
for i in "${!devices[@]}"; do
    device="${devices[$i]}"
    echo "Launching experiment on CUDA device $device"
    run_experiment "$device"
done

echo "All experiments have been launched. Exiting..."
exit 0
