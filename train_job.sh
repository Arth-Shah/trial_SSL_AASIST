#!/bin/bash
#SBATCH --job-name=aasist-train
#SBATCH --account=f200791001
#SBATCH --partition=rtx6000-fullgpu
#SBATCH --gres=gpu:rtx_pro:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err
# Tokens are charged on (CPUs/GPUs/memory x --time) you REQUEST, so don't ask for more than needed.
# If 24 h is not enough for 15 epochs, just submit this script again: it resumes automatically.

cd $SLURM_SUBMIT_DIR

module load anaconda3-2025-12-2
module load cuda-13.0
source /apps/compilers/anaconda3-2025-12-2/etc/profile.d/conda.sh
conda activate aasist

export HF_HUB_OFFLINE=1        # use the XLS-R weights cached by download_model.py
export PYTHONUNBUFFERED=1      # print() lines show up in the log immediately

DATA=$HOME/data/processed_data_1     # <-- change if your data is elsewhere

# Auto-resume if a previous run left a "last" checkpoint.
# (Delete the checkpoints/ folder if you want to start training from scratch.)
RESUME=""
if [ -f checkpoints/last_model_XLSR.pth ]; then
    RESUME="--resume checkpoints/last_model_XLSR.pth"
fi

echo "Running on $(hostname)"; nvidia-smi
python train.py --data-root "$DATA" --num-workers 4 $RESUME
