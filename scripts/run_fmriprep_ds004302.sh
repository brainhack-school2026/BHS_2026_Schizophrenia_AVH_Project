#!/bin/bash

#SBATCH --job-name=fmriprep_ds004302_batch
#SBATCH --output=/home/l/lcl_uotmsc1127/lcl_uotmsc1127s1934/logs/%x_%j.out
#SBATCH --error=/home/l/lcl_uotmsc1127/lcl_uotmsc1127s1934/logs/%x_%j.err
#SBATCH --account=teach
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

set -euo pipefail

export PROJECT_DIR=${HOME}/Brainhack_project
export BIDS_DIR=${PROJECT_DIR}/ds004302

export FMRIPREP_HOME=${HOME}/templates
export SING_CONTAINER=${HOME}/links/common/fmriprep-25.2.4.simg
export FS_LICENSE=${HOME}/links/common/fs_license.txt

export OUTPUT_DIR=${PROJECT_DIR}/derivatives/ds004302/fmriprep/25.2.4
export LOGS_DIR=${HOME}/logs

mkdir -vp "${OUTPUT_DIR}" "${LOGS_DIR}" "${FMRIPREP_HOME}"

export WORK_DIR=${SLURM_TMPDIR}/fmriprep_work
mkdir -vp "${WORK_DIR}"

module load apptainer/1.3.5

export APPTAINERENV_TEMPLATEFLOW_HOME=/home/fmriprep/.cache/templateflow
export APPTAINERENV_TEMPLATEFLOW_OFFLINE=1

#changing --participant-label "${SUBJECT}" to only preprocess a subset of subjects to save time for testing
apptainer run --cleanenv \
    -B "${FMRIPREP_HOME}:/home/fmriprep" --home /home/fmriprep \
    -B "${BIDS_DIR}:/bids:ro" \
    -B "${OUTPUT_DIR}:/derived" \
    -B "${WORK_DIR}:/work" \
    -B "${FS_LICENSE}:/li" \
    "${SING_CONTAINER}" \
    /bids /derived participant \
    --participant-label 32 33 34 36 38 \
    -w /work \
    --skip-bids-validation \
    --nthreads 8 \
    --omp-nthreads 4 \
    --mem-mb 30000 \
    --output-spaces MNI152NLin2009cAsym:res-2 \
    --fs-license-file /li \
    --ignore slicetiming \
    --skull-strip-t1w skip \
    --notrack

echo "Done at $(date). Outputs: ${OUTPUT_DIR}"
