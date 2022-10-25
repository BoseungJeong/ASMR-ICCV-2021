#!/bin/bash
#Basic range in for llop
SET=$(seq 1 10)
ids=0
pre_epoch=4

echo Training is start. 
lambda=6

name=Market_pretrained_${pre_epochs}_s12_m02_lambda_${lamb}
python train.py --MA_loss --batchsize 128 --seed 1 --decay 5 --gpu_ids $ids --name $name --num_epoch 10 --s 12 --m 0.2 --lr 0.001 --lamd $lamb --Pretrained --pre_epoch $pre_epochs --train_all
for epochs in $SET
do
    echo Evaluation at $epochs
    python test.py --batchsize 200 --which_epoch $epochs --gpu_ids $ids --name $name
done
echo All done


