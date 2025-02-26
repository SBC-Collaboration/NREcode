#!/bin/bash
#SBATCH --time=05-00:00
#SBATCH --account=def-kenclark-ab

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=12000M

#SBATCH --mail-user=ddurnfor@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

#SBATCH --output=XeNR_period302_MC.out

#SBATCH --array=1-250

python3.10 XeNR_Wrapper_i2A_v5_July2024_MC.py  $SLURM_ARRAY_TASK_ID 1
