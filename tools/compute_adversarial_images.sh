#!/usr/bin/env sh
# Compute adversarial images given the normal images
# as input and the network definition (with its weights)
# which will be used during the computation.

echo "Start script for computing adversarial examples"
DATA=~/image_folder/
NETWORK=~/DeepFool/resources/deploy_caffenet.prototxt
WEIGHTS=~/DeepFool/bvlc_reference_caffenet.caffemodel
LABELS=~/caffe/data/ilsvrc12/synset_words.txt
MEAN_DATA=~/caffe/data/ilsvrc12/imagenet_mean.binaryproto
TOOLS=build/tools

echo "Computing adversarial images (script)."

$TOOLS/compute_adversarial \
    --images=$DATA \
    --model=$NETWORK \
    --weights=$WEIGHTS \
    --labels_file=$LABELS \
    --mean_file=$MEAN_DATA \
    2>&1 adversarial_log.log

echo "Adversarial images computed (script)."
