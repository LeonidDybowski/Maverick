#!/bin/bash
#SBATCH --partition=hpc1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=360G
#SBATCH --time=3-00:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# load module
module load gcc

# export path
export PATH=$PWD/mmseqs/bin:$PATH

QUERY_DB="queryDB"
TARGET_DB="targetDB"
RESULT_DB="resultDB"
MSA_DB="result_msa"
TMP_DIR="tmp"

# export OMP_NUM_THREADS=1
THREADS=64
MAX_SEQS=2000
NUM_ITER=2

mkdir -p "$TMP_DIR"

# create databases
mmseqs search $QUERY_DB $TARGET_DB $RESULT_DB $TMP_DIR --threads "$THREADS" --max-seqs "$MAX_SEQS" --num-iterations "$NUM_ITER"

