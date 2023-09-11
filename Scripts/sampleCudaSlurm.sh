#!/usr/bin/env zsh
#SBATCH --job-name=CudaHello
#SBATCH --partition=instruction
#SBATCH --time=00-00:03:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=cuda_hello-%j.out

module load nvidia/cuda/11.8.0
nvcc cudaHello.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o cudaHello
./cudaHello

