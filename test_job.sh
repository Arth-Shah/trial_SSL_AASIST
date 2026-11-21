#!/bin/bash
#SBATCH --job-name=aasist-test
#SBATCH --account=f200791001
#SBATCH --partition=rtx6000-fullgpu
#SBATCH --gres=gpu:rtx_pro:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/test-%j.out
#SBATCH --error=logs/test-%j.err

cd $SLURM_SUBMIT_DIR

module load anaconda3-2025-12-2
module load cuda-13.0
source /apps/compilers/anaconda3-2025-12-2/etc/profile.d/conda.sh
conda activate aasist

export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

DATA=$HOME/data/processed_data_1     # <-- change if your data is elsewhere

echo "Running on $(hostname)"; nvidia-smi
python test.py --data-root "$DATA" --num-workers 4 \
    --checkpoint checkpoints/best_model_XLSR.pth \
    --results-json results_test.json

# Optional: params + FLOPs (notebook cells 50 & 52)
python flops.py
