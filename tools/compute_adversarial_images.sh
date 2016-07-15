#!/usr/bin/env sh
# Compute adversarial images given the normal images
# as input and the network definition (with its weights)
# which will be used during the computation.

echo "Start script for computing adversarial examples"
DATA=/datasets2/
NETWORK=models/bvlc_reference_caffenet/deploy.prototxt
WEIGHTS=models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel

TOOLS=build/tools

echo "Computing adversarial images (script)."

$TOOLS/compute_adversarial \
    --images=$DATA \
    --model=$NETWORK \
    --weights=$WEIGHTS \
    2>&1 adversarial_log.log

echo "Adversarial images computed (script)."
