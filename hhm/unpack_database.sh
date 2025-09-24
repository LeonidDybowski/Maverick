#!/bin/bash
#SBATCH --partition=hpc1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G
#SBATCH --time=0-12:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# === Configuration ===
WORKDIR=$PWD
MMSEQS_IMG="mmseqs2.sif"

QUERY_DB="queryDB"
TARGET_DB="targetDB"
RESULT_DB="resultDB"
MSA_DB="result_msa"
A3M_DIR="a3m_out"

# === MMseqs2 Steps ===
THREADS=16

mkdir -p "$A3M_DIR"

apptainer exec --bind "$WORKDIR:/data" "$MMSEQS_IMG" \
  mmseqs unpackdb /data/$MSA_DB /data/$A3M_DIR --unpack-suffix .a3m
