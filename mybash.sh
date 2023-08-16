#!/bin/bash
#SBATCH --account=rrg-rmansbac
#SBATCH --time=0-00:30:00
#SBATCH --job-name="test-rnn-vae"
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --gpus-per-node=1

module load python/3.10 scipy-stack
source $PROJECTSRRG/project-property-manifold/venv/bin/activate

pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

python --version
python train.py --batch_size 128 --epochs 10 --n_latent 32 --n_model 64 