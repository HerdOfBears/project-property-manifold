#!/bin/bash
# This script submits multiple jobs to cluster

for i in {0..90..15}
do
    echo "Submitting job with drop_percent_of_labels = $i"
    sbatch --job-name "semi-sup-{$i}" semi-supervised-job.sh $i
done