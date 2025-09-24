#!/bin/bash
#SBATCH --partition=any
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=0-8:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# load module
module load gcc

# export path
export PATH=$PWD/mmseqs/bin:$PATH

# create databases
mmseqs createdb gencode.v47.pc_translations.fa queryDB
mmseqs createdb uniclust90_2018_08_consensus.fasta targetDB
