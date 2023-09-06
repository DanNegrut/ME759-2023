#!/usr/bin/env zsh
#SBATCH --job-name=HelloScript
#SBATCH --partition=wacc
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --time=0-00:00:10
#SBATCH --output=hello_output-%j.txt

cd $SLURM_SUBMIT_DIR

name_str=World
echo "Hello, $name_str!"
