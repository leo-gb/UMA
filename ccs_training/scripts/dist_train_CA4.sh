#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36001 train.py \
--distributed \
--task_name 'e05_r18_preFalse_step_v1_M2PN' \
--mode 'train' \
--gpus '0' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr4-32-10/e05_r18_preFalse_step_v1_M2PN/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr4-32-10/train_attrbute_CA4.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr4-32-10/valid_attrbute_CA4.txt' \
--valid > e05_r18_preFalse_step_v1_M2PN_CA4log_00.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36101 train.py \
--distributed \
--task_name 'e05_r18_preFalse_step_v1_M2PN' \
--mode 'train' \
--gpus '1' \
--train_ratio 2 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr4-32-10/e05_r18_preFalse_step_v1_M2PN/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr4-32-10/train_attrbute_CA4.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr4-32-10/valid_attrbute_CA4.txt' \
--valid > e05_r18_preFalse_step_v1_M2PN_CA4log_01.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36201 train.py \
--distributed \
--task_name 'e05_r18_preFalse_step_v1_M2PN' \
--mode 'train' \
--gpus '2' \
--train_ratio 4 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr4-32-10/e05_r18_preFalse_step_v1_M2PN/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr4-32-10/train_attrbute_CA4.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr4-32-10/valid_attrbute_CA4.txt' \
--valid > e05_r18_preFalse_step_v1_M2PN_CA4log_02.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36301 train.py \
--distributed \
--task_name 'e05_r18_preFalse_step_v1_M2PN' \
--mode 'train' \
--gpus '3' \
--train_ratio 8 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr4-32-10/e05_r18_preFalse_step_v1_M2PN/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr4-32-10/train_attrbute_CA4.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr4-32-10/valid_attrbute_CA4.txt' \
--valid > e05_r18_preFalse_step_v1_M2PN_CA4log_03.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36401 train.py \
--distributed \
--task_name 'e05_r18_preFalse_step_v1_M2PN' \
--mode 'train' \
--gpus '4' \
--train_ratio 16 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr4-32-10/e05_r18_preFalse_step_v1_M2PN/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr4-32-10/train_attrbute_CA4.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr4-32-10/valid_attrbute_CA4.txt' \
--valid > e05_r18_preFalse_step_v1_M2PN_CA4log_04.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36501 train.py \
--distributed \
--task_name 'e05_r18_preFalse_step_v1_M2PN' \
--mode 'train' \
--gpus '5' \
--train_ratio 32 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr4-32-10/e05_r18_preFalse_step_v1_M2PN/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr4-32-10/train_attrbute_CA4.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr4-32-10/valid_attrbute_CA4.txt' \
--valid > e05_r18_preFalse_step_v1_M2PN_CA4log_05.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36601 train.py \
--distributed \
--task_name 'e05_r18_preFalse_step_v1_M2PN' \
--mode 'train' \
--gpus '6' \
--train_ratio 64 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr4-32-10/e05_r18_preFalse_step_v1_M2PN/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr4-32-10/train_attrbute_CA4.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr4-32-10/valid_attrbute_CA4.txt' \
--valid > e05_r18_preFalse_step_v1_M2PN_CA4log_06.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36701 train.py \
--distributed \
--task_name 'e05_r18_preFalse_step_v1_M2PN' \
--mode 'train' \
--gpus '7' \
--train_ratio 128 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr4-32-10/e05_r18_preFalse_step_v1_M2PN/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr4-32-10/train_attrbute_CA4.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr4-32-10/valid_attrbute_CA4.txt' \
--valid > e05_r18_preFalse_step_v1_M2PN_CA4log_07.txt &

