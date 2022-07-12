#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=34001 train.py \
--distributed \
--task_name 'e06_r18_preFalse_step_v1_M1PNM2PN' \
--mode 'train' \
--gpus '0' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr21/e06_r18_preFalse_step_v1_M1PNM2PN/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr21/train_attrbute_21.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr21/valid_attrbute_21.txt' \
--valid > e06_r18_preFalse_step_v1_M1PNM2PN_21log_00.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=34101 train.py \
--distributed \
--task_name 'e06_r18_preFalse_step_v1_M1PNM2PN' \
--mode 'train' \
--gpus '1' \
--train_ratio 2 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr21/e06_r18_preFalse_step_v1_M1PNM2PN/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr21/train_attrbute_21.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr21/valid_attrbute_21.txt' \
--valid > e06_r18_preFalse_step_v1_M1PNM2PN_21log_01.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=34201 train.py \
--distributed \
--task_name 'e06_r18_preFalse_step_v1_M1PNM2PN' \
--mode 'train' \
--gpus '2' \
--train_ratio 4 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr21/e06_r18_preFalse_step_v1_M1PNM2PN/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr21/train_attrbute_21.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr21/valid_attrbute_21.txt' \
--valid > e06_r18_preFalse_step_v1_M1PNM2PN_21log_02.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=34301 train.py \
--distributed \
--task_name 'e06_r18_preFalse_step_v1_M1PNM2PN' \
--mode 'train' \
--gpus '3' \
--train_ratio 8 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr21/e06_r18_preFalse_step_v1_M1PNM2PN/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr21/train_attrbute_21.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr21/valid_attrbute_21.txt' \
--valid > e06_r18_preFalse_step_v1_M1PNM2PN_21log_03.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=34401 train.py \
--distributed \
--task_name 'e06_r18_preFalse_step_v1_M1PNM2PN' \
--mode 'train' \
--gpus '4' \
--train_ratio 16 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr21/e06_r18_preFalse_step_v1_M1PNM2PN/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr21/train_attrbute_21.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr21/valid_attrbute_21.txt' \
--valid > e06_r18_preFalse_step_v1_M1PNM2PN_21log_04.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=34501 train.py \
--distributed \
--task_name 'e06_r18_preFalse_step_v1_M1PNM2PN' \
--mode 'train' \
--gpus '5' \
--train_ratio 32 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr21/e06_r18_preFalse_step_v1_M1PNM2PN/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr21/train_attrbute_21.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr21/valid_attrbute_21.txt' \
--valid > e06_r18_preFalse_step_v1_M1PNM2PN_21log_05.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=34601 train.py \
--distributed \
--task_name 'e06_r18_preFalse_step_v1_M1PNM2PN' \
--mode 'train' \
--gpus '6' \
--train_ratio 64 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr21/e06_r18_preFalse_step_v1_M1PNM2PN/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr21/train_attrbute_21.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr21/valid_attrbute_21.txt' \
--valid > e06_r18_preFalse_step_v1_M1PNM2PN_21log_06.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=34701 train.py \
--distributed \
--task_name 'e06_r18_preFalse_step_v1_M1PNM2PN' \
--mode 'train' \
--gpus '7' \
--train_ratio 128 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr21/e06_r18_preFalse_step_v1_M1PNM2PN/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr21/train_attrbute_21.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr21/valid_attrbute_21.txt' \
--valid > e06_r18_preFalse_step_v1_M1PNM2PN_21log_07.txt &

