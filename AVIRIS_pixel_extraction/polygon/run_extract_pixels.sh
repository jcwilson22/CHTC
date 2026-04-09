#!/bin/bash
set -euo pipefail

IMAGE="$1"

CONTAINER=/staging/jcwilson22/gdal_python.sif
SCRIPT=extract_pixels_aviris_nc.py
SHAPEFILE=capetrait_patches_34s2.shp

IMAGE_BASENAME=$(basename "${IMAGE%.nc}")
OUTCSV="extracted_pixels_${IMAGE_BASENAME}.csv.gz"

singularity exec \
    --bind /staging/groups:/staging/groups \
    "${CONTAINER}" \
    python3 ${SCRIPT} \
        --image ${IMAGE} \
        --shapefile ${SHAPEFILE} \
        --polygon_id_field PlotID \
        --out_csv "${OUTCSV}"