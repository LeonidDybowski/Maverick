#!/bin/bash
#SBATCH --partition=any
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=180G
#SBATCH --time=1-00:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load openmpi

CONTAINER="../horovod_image_3/horovod_mpi.sif"

export UCX_TLS=tcp,cuda_copy,cuda_ipc
export UCX_POSIX_USE_PROC_LINK=n

apptainer exec -B "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" "$CONTAINER" bash -c 'python3 ${SLURM_SUBMIT_DIR}/maverick_horovod_all_evaluate_all_models_cpu.py'
