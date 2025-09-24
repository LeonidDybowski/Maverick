#!/bin/bash
#SBATCH --partition=hpc1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --time=3-00:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# === Configuration ===
WORKDIR=$PWD
HHSUITE_IMG="hhsuite.sif"

A3M_DIR_TRANSCRIPTS="a3m_renamed"
HHM_DIR="hhm_out_first"

# === MMseqs2 Steps ===
export OMP_NUM_THREADS=1
THREADS=32

mkdir -p "$HHM_DIR"

# === Parallel hhmake ===
find "$A3M_DIR_TRANSCRIPTS" -name '*.a3m' | xargs -P "$THREADS" -I{} \
bash -c 'file="{}"; base=$(basename "$file" .a3m); apptainer exec '"$HHSUITE_IMG"' hhmake -M first -i "$file" -o '"$HHM_DIR"'/"$base".hhm -maxres 40000'
