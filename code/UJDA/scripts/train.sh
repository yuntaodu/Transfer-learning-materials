#!/bin/bash

export CUDA_VISIBLE_DEVICES=1MDD


PROJ_ROOT="/home/app/MDD"
PROJ_NAME="D2A"
LOG_FILE="${PROJ_ROOT}/log/${PROJ_NAME}-`date +'%Y-%m-%d-%H-%M-%S'`.log"

echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
python ${PROJ_ROOT}/trainer/train.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset Office-31 \
    --src_address /home/app/MDD/data/dslr_list.txt \
    --tgt_address /home/app/MDD/data/amazon_list.txt \
    >> ${LOG_FILE}  2>&1
