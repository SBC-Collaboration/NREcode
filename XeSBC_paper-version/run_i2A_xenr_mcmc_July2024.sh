#!/bin/bash
#SBATCH --time=02-00:00
#SBATCH --account=def-kenclark-ab

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=12000M

#SBATCH --mail-user=ddurnfor@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

#SBATCH --output=XeNR_period302.out

python3.10 XeNR_Wrapper_i2A_v5_July2024.py
