#!/bin/bash
#SBATCH --account=def-rmansbac
#SBATCH --time=0-00:05:00
#SBATCH --job-name="test-run"
#SBATCH --nodes=1

module load python/3.10 scipy-stack
source $PROJECTSRRG/project-property-manifold/venv/bin/activate

pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

python --version
python train.py