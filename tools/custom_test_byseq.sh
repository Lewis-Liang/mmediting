#!/usr/bin/env bash
CONFIG=$1
CHECKPOINT="$(basename $CONFIG)"
# iters=(46000 47000 48000 49000)
iters=(30000 35000 40000 45000 50000)


# echo "${CHECKPOINT%.*}"

for iter in ${iters[@]}
do
python tools/test_dehaze_video_inference.py \
--seq 9 \
--config ${CONFIG} \
 --checkpoint work_dirs/${CHECKPOINT%.*}/iter_${iter}.pth
done

