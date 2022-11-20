#!/usr/bin/env bash
CONFIG=$1
CHECKPOINT="$(basename $CONFIG)"
iters=(30000 35000 40000 45000 50000)

# echo "${CHECKPOINT%.*}"

for iter in ${iters[@]}
do
python tools/custom_test.py \
--config ${CONFIG} \
 --checkpoint work_dirs/${CHECKPOINT%.*}/iter_${iter}.pth
done

# 