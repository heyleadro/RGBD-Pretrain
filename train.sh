#!/bin/bash

#SBATCH --partition=sgpu_medium

#SBATCH --time=23:00:00

#SBATCH --gpus-per-node=4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1 

#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=s94falmu@uni-bonn.de

# prep
source ~/.bashrc
conda init
module load CUDA/12.6.0
conda activate mt
pip freeze
echo "NUM PROC:"
nproc
nvidia-smi

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901
export OMP_NUM_THREADS=64


# TODO: COPY TO TMP??

# run
MODEL=DFormerv2_S
srun torchrun \
    --nproc_per_node 4 --nnodes $SLURM_JOB_NUM_NODES \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
    /lustre/scratch/data/s94falmu_hpc-thesis/imagenet \
    /lustre/scratch/data/s94falmu_hpc-thesis/depth \
    --model $MODEL \
    -b 256 \
    --epochs 300 \
    --opt adamw \
    -j 16 \
    --input-size 4 224 224 \
    --warmup-lr 1e-6 \
    --warmup-epochs 5 \
    --weight-decay 0.05 \
    --grad-accum-steps 1 \
    --amp \
    --clip-grad 1.0 \
    --drop-path 0.45 \
    --lr 0.001 \
    --aa rand-m9-mstd0.5-inc1 \
    --remode pixel \
    --reprob 0.25 \
    --cutmix 1.0 \
    --mixup 0.8 \
    # --resume outputs/XXXXX/last.pth.tar

echo "FINISHED"


