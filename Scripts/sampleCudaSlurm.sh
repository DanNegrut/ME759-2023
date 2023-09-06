#!/usr/bin/env bash
#SBATCH --job-name=CudaHello
#SBATCH --partition=wacc
#SBATCH --time=00-00:03:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:gtx1080:1
#SBATCH --output=cuda_hello-%j.out

module load nvidia/cuda
nvcc cudaHello.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o cudaHello
./cudaHello

