#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}


# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=39001 train.py \
# --distributed \
# --task_name 'e18_r18_preFalse_step_OurX10_v1' \
# --mode 'train' \
# --gpus '0' \
# --train_ratio 1 \
# --delimiter '||$$||' \
# --round '1' \
# --log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/e18_r18_preFalse_step_OurX10_v1/' \
# --train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/train_attrbute_CA5.txt' \
# --valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/valid_attrbute_CA5.txt' \
# --valid > e18_r18_preFalse_step_OurX10_v1_CA5log_00.txt &


# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=39101 train.py \
# --distributed \
# --task_name 'e18_r18_preFalse_step_OurX10_v1' \
# --mode 'train' \
# --gpus '1' \
# --train_ratio 2 \
# --delimiter '||$$||' \
# --round '1' \
# --log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/e18_r18_preFalse_step_OurX10_v1/' \
# --train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/train_attrbute_CA5.txt' \
# --valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/valid_attrbute_CA5.txt' \
# --valid > e18_r18_preFalse_step_OurX10_v1_CA5log_01.txt &


# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=39201 train.py \
# --distributed \
# --task_name 'e18_r18_preFalse_step_OurX10_v1' \
# --mode 'train' \
# --gpus '2' \
# --train_ratio 4 \
# --delimiter '||$$||' \
# --round '1' \
# --log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/e18_r18_preFalse_step_OurX10_v1/' \
# --train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/train_attrbute_CA5.txt' \
# --valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/valid_attrbute_CA5.txt' \
# --valid > e18_r18_preFalse_step_OurX10_v1_CA5log_02.txt &


# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=39301 train.py \
# --distributed \
# --task_name 'e18_r18_preFalse_step_OurX10_v1' \
# --mode 'train' \
# --gpus '3' \
# --train_ratio 8 \
# --delimiter '||$$||' \
# --round '1' \
# --log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/e18_r18_preFalse_step_OurX10_v1/' \
# --train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/train_attrbute_CA5.txt' \
# --valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/valid_attrbute_CA5.txt' \
# --valid > e18_r18_preFalse_step_OurX10_v1_CA5log_03.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=39401 train.py \
--distributed \
--task_name 'e18_r18_preFalse_step_OurX10_v1' \
--mode 'train' \
--gpus '4' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/e18_r18_preFalse_step_OurX10_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/train_attrbute_CA5.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/valid_attrbute_CA5.txt' \
--valid > e18_r18_preFalse_step_OurX10_v1_CA5log_04.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=39501 train.py \
--distributed \
--task_name 'e18_r18_preFalse_step_OurX10_v1' \
--mode 'train' \
--gpus '5' \
--train_ratio 2 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/e18_r18_preFalse_step_OurX10_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/train_attrbute_CA5.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/valid_attrbute_CA5.txt' \
--valid > e18_r18_preFalse_step_OurX10_v1_CA5log_05.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=39601 train.py \
--distributed \
--task_name 'e18_r18_preFalse_step_OurX10_v1' \
--mode 'train' \
--gpus '6' \
--train_ratio 4 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/e18_r18_preFalse_step_OurX10_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/train_attrbute_CA5.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/valid_attrbute_CA5.txt' \
--valid > e18_r18_preFalse_step_OurX10_v1_CA5log_06.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=39701 train.py \
--distributed \
--task_name 'e18_r18_preFalse_step_OurX10_v1' \
--mode 'train' \
--gpus '7' \
--train_ratio 8 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/e18_r18_preFalse_step_OurX10_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/train_attrbute_CA5.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/valid_attrbute_CA5.txt' \
--valid > e18_r18_preFalse_step_OurX10_v1_CA5log_07.txt &

