#!/bin/bash

fileName=$1

BASE=$(sed 's/\.vcf//' <<< ${fileName})

# process variants with annovar
echo "Starting Step 1: Get coding changes with Annovar"
dos2unix ${fileName}
# remove chr prefix if present
sed -i 's/^chr//' ${fileName}
resources/annovar/convert2annovar.pl -format vcf4 --keepindelref ${BASE}.vcf > ${BASE}.avinput
resources/annovar/annotate_variation.pl -dbtype wgEncodeGencodeBasicV33lift37 -buildver hg19 --exonicsplicing ${BASE}.avinput annovar/humandb/
# if there are no scorable variants, end early
SCORABLEVARIANTS=$(cat ${BASE}.avinput.exonic_variant_function | wc -l || true)
if [[ ${SCORABLEVARIANTS} -eq 0 ]]; then echo "No scorable variants found"; exit 0; fi
resources/annovar/coding_change.pl ${BASE}.avinput.exonic_variant_function resources/annovar/humandb/hg19_wgEncodeGencodeBasicV33lift37.txt resources/annovar/humandb/hg19_wgEncodeGencodeBasicV33lift37Mrna.fa --includesnp --onlyAltering --alltranscript > ${BASE}.coding_changes.txt

# select transcript
echo "Starting Step 2: Select transcript"
python Maverick2025/InferenceScripts/groomAnnovarOutput.py --inputBase=${BASE}
# if there are no scorable variants, end early
SCORABLEVARIANTS=$(cat ${BASE}.groomed.txt | wc -l || true)
if [[ ${SCORABLEVARIANTS} -lt 2 ]]; then echo "No scorable variants found"; exit 0; fi

# add on annotations
echo "Starting Step 3: Merge on annotations"
python Maverick2025/InferenceScripts/annotateVariants.py --inputBase=${BASE}

# run variants through each of the models
echo "Starting Step 4: Score variants with the 8 models"
python Maverick2025/InferenceScripts/runModels.py --inputBase=${BASE}

python Maverick2025/InferenceScripts/rankVariants.py --inputBase=${BASE}

echo "Done"
