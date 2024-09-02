#!/bin/bash -l

# Job name:
#SBATCH -J train_vit

# Dawn SLURM account:
#SBATCH -A BRAINMRI-DAWN-GPU

# Do not requeue the job if it fails:
#SBATCH --no-requeue

# Partition to submit to:
#SBATCH --partition=pvc

# Number of nodes and tasks:
#SBATCH --nodes=1           # Number of nodes
#SBATCH --ntasks=1          # Number of tasks (usually the number of MPI ranks)
#SBATCH --cpus-per-task=24  # Number of CPU cores per task (adjusted for multithreading)

# GPU configuration:
#SBATCH --gres=gpu:1        # Number of GPUs per node (1 GPU requested)

# Output and error files:
#SBATCH --output=/home/fp427/rds/hpc-work/vit-pytorch/logs/train_vit_%j.out    # Standard output log
#SBATCH --error=/home/fp427/rds/hpc-work/vit-pytorch/logs/train_vit_%j.err     # Standard error log

#SBATCH --time=12:00:00     # Set a time limit of 12 hours

# Purge all loaded modules and load the base environment for Dawn:
module purge
module load default-dawn

# Load Intel Python and activate PyTorch GPU environment:
module load intelpython-conda
conda activate pytorch-gpu

# Print Python version to verify environment setup:
python --version

# Navigate to the working directory:
cd /home/fp427/rds/hpc-work/vit-pytorch
pwd


srun python main.py \
    --image_size 224 \
    --patch_size 16 \
    --num_classes 101 \
    --dim 768 \
    --depth 12 \
    --heads 12 \
    --mlp_dim_ratio 4 \
    --batch_size 64 \
    --num_epochs 100 \
    --learning_rate 3e-4 \
    --weight_decay 0.1 \
    --dropout 0.1