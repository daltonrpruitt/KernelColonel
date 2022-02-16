#!/bin/bash

#SBATCH --partition= % GPU partition %
#SBATCH --job-name=gpu_microbenchmarks
#SBATCH --output=output/test_job.o%j
#SBATCH --mail-user=%email%@msstate.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1

./build/debug/ctx_driver | tee output/($SLURM_JOB_ID)_output.txt
