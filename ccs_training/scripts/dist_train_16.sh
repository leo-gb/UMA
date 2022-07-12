#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=35001 train.py \
--distributed \
--task_name 'e17_r18_preFalse_step_OurX6_v1' \
--mode 'train' \
--gpus '0' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr16/e17_r18_preFalse_step_OurX6_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/train_attrbute_16.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/valid_attrbute_16.txt' \
--valid > e17_r18_preFalse_step_OurX6_v1_16log_00.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=35101 train.py \
--distributed \
--task_name 'e17_r18_preFalse_step_OurX6_v1' \
--mode 'train' \
--gpus '1' \
--train_ratio 2 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr16/e17_r18_preFalse_step_OurX6_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/train_attrbute_16.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/valid_attrbute_16.txt' \
--valid > e17_r18_preFalse_step_OurX6_v1_16log_01.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=35201 train.py \
--distributed \
--task_name 'e17_r18_preFalse_step_OurX6_v1' \
--mode 'train' \
--gpus '2' \
--train_ratio 4 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr16/e17_r18_preFalse_step_OurX6_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/train_attrbute_16.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/valid_attrbute_16.txt' \
--valid > e17_r18_preFalse_step_OurX6_v1_16log_02.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=35301 train.py \
--distributed \
--task_name 'e17_r18_preFalse_step_OurX6_v1' \
--mode 'train' \
--gpus '3' \
--train_ratio 8 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr16/e17_r18_preFalse_step_OurX6_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/train_attrbute_16.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/valid_attrbute_16.txt' \
--valid > e17_r18_preFalse_step_OurX6_v1_16log_03.txt &


# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=35401 train.py \
# --distributed \
# --task_name 'e17_r18_preFalse_step_OurX6_v1' \
# --mode 'train' \
# --gpus '4' \
# --train_ratio 16 \
# --delimiter '||$$||' \
# --round '1' \
# --log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr16/e17_r18_preFalse_step_OurX6_v1/' \
# --train_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/train_attrbute_16.txt' \
# --valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/valid_attrbute_16.txt' \
# --valid > e17_r18_preFalse_step_OurX6_v1_16log_04.txt &


# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=35501 train.py \
# --distributed \
# --task_name 'e17_r18_preFalse_step_OurX6_v1' \
# --mode 'train' \
# --gpus '5' \
# --train_ratio 32 \
# --delimiter '||$$||' \
# --round '1' \
# --log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr16/e17_r18_preFalse_step_OurX6_v1/' \
# --train_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/train_attrbute_16.txt' \
# --valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/valid_attrbute_16.txt' \
# --valid > e17_r18_preFalse_step_OurX6_v1_16log_05.txt &


# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=35601 train.py \
# --distributed \
# --task_name 'e17_r18_preFalse_step_OurX6_v1' \
# --mode 'train' \
# --gpus '6' \
# --train_ratio 64 \
# --delimiter '||$$||' \
# --round '1' \
# --log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr16/e17_r18_preFalse_step_OurX6_v1/' \
# --train_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/train_attrbute_16.txt' \
# --valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/valid_attrbute_16.txt' \
# --valid > e17_r18_preFalse_step_OurX6_v1_16log_06.txt &


# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=35701 train.py \
# --distributed \
# --task_name 'e17_r18_preFalse_step_OurX6_v1' \
# --mode 'train' \
# --gpus '7' \
# --train_ratio 128 \
# --delimiter '||$$||' \
# --round '1' \
# --log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr16/e17_r18_preFalse_step_OurX6_v1/' \
# --train_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/train_attrbute_16.txt' \
# --valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/valid_attrbute_16.txt' \
# --valid > e17_r18_preFalse_step_OurX6_v1_16log_07.txt &

