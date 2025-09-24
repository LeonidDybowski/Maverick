#!/bin/bash
#SBATCH --partition=gpu4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCh --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --gres=gpu:4
#SBATCH --time=2-00:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load openmpi

CONTAINER="../horovod_image_3/horovod_mpi.sif"

export UCX_TLS=tcp,cuda_copy,cuda_ipc
export UCX_POSIX_USE_PROC_LINK=n
export OMP_NUM_THREADS=16

mpirun -np 4 --map-by ppr:4:node --bind-to none \
  apptainer exec --nv -B "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" "$CONTAINER" \
  bash -c 'python3 ${SLURM_SUBMIT_DIR}/maverick_horovod_all_train_arch1.py'
