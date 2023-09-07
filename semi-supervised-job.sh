#!/bin/bash
#SBATCH --account=rrg-rmansbac
#SBATCH --time=0-02:10:00
#SBATCH --nodes=1
#SBATCH --mem=4g
#SBATCH --gpus-per-node=1

module load python/3.10 scipy-stack
source $PROJECTSRRG/project-property-manifold/venv/bin/activate

python train.py \
    --batch_size 256 \
    --epochs 200 \
    --n_latent 32 \
    --n_model 32 \
    --n_embd 32 \
    --drop_percent_of_labels $1 \
    --save_dir "./experiments/semi-supervised/drop-{$1}percent/"