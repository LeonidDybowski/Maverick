#!/bin/bash
#SBATCH --partition=hpc1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G
#SBATCH --time=3-00:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# === Configuration ===
WORKDIR=$PWD
QUERY_FASTA="gencode.v47.pc_translactions.fa"
TARGET_FASTA="uniclust90_2018_08_consensus.fasta"
MMSEQS_IMG="mmseqs2.sif"
HHSUITE_IMG="hhsuite.sif"

QUERY_DB="queryDB"
TARGET_DB="targetDB"
RESULT_DB="resultDB"
MSA_DB="result_msa"
TMP_DIR="tmp"
A3M_DIR="a3m_out"
HHM_DIR="hhm_out"

# === MMseqs2 Steps ===
export OMP_NUM_THREADS=1
THREADS=16
MAX_SEQS=2000
NUM_ITER=2

mkdir -p "$TMP_DIR" "$A3M_DIR" "$HHM_DIR"

# === MMseqs2 Steps ===
apptainer exec --bind "$WORKDIR:/data" "$MMSEQS_IMG" \
  mmseqs createdb /data/$QUERY_FASTA /data/$QUERY_DB

apptainer exec --bind "$WORKDIR:/data" "$MMSEQS_IMG" \
  mmseqs createdb /data/$TARGET_FASTA /data/$TARGET_DB

apptainer exec --bind "$WORKDIR:/data" "$MMSEQS_IMG" \
  mmseqs search /data/$QUERY_DB /data/$TARGET_DB /data/$RESULT_DB /data/$TMP_DIR \
  --threads 1 --max-seqs $MAX_SEQS --num-iterations $NUM_ITER --db-load-mode 2

apptainer exec --bind "$WORKDIR:/data" "$MMSEQS_IMG" \
  mmseqs result2msa /data/$QUERY_DB /data/$TARGET_DB /data/$RESULT_DB /data/$MSA_DB \
  --msa-format-mode 5 --threads 1

apptainer exec --bind "$WORKDIR:/data" "$MMSEQS_IMG" \
  mmseqs unpackdb /data/$MSA_DB /data/$A3M_DIR

# === Parallel hhmake ===
find "$A3M_DIR" -name '*.a3m' | xargs -n 1 -P "$THREADS" -I{} \
apptainer exec --bind "$WORKDIR:/data" "$HHSUITE_IMG" \
hhmake -M first -i /data/{} -o /data/$HHM_DIR/$(basename {} .a3m).hhm

