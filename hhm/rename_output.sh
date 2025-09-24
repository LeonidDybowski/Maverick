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
LOOKUP="queryDB.lookup"
IN_DIR="a3m_out"
OUT_DIR="a3m_renamed"
LOGFILE="rename_missing.log"

mkdir -p "$OUT_DIR"
> "$LOGFILE"

awk -F'\t' -v IN="$IN_DIR" -v OUT="$OUT_DIR" -v LOG="$LOGFILE" '
{
  id = $1
  split($2, ids, "|")
  enst = ids[2]
  infile = IN "/" id ".a3m"
  outfile = OUT "/" enst ".a3m"
  cmd_check = "[ -f \"" infile "\" ]"
  cmd_copy = "cp \"" infile "\" \"" outfile "\""

  if (system(cmd_check) == 0) {
    system(cmd_copy)
  } else {
    print "Missing file for ID " id " → " infile >> LOG
  }
}' "$LOOKUP"
