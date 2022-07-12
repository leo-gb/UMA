#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}

# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=32002 train.py \
# --distributed \
# --task_name 'e09_r18_preFalse_step_Mixup_v1' \
# --mode 'train' \
# --gpus '0' \
# --train_ratio 5 \
# --batch_size 2 \
# --delimiter '||$$||' \
# --round '1' \
# --log_dir    'leogb/causal_logs/DAMapDataset/e09_r18_preFalse_step_Mixup_v1' \
# --train_file 'leogb/causal_logs/DAMapDataset/train_attrbute.txt' \
# --valid_file 'leogb/causal_logs/DAMapDataset/valid_attrbute.txt' \
# --OverwriteDAConfig \
# --Mixup \
# --valid > e09_r18_preFalse_step_Mixup_v1-DAMaplog_08.txt &

# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=32003 train.py \
# --distributed \
# --task_name 'e10_r18_preFalse_step_Cutmix_v1' \
# --mode 'train' \
# --gpus '1' \
# --train_ratio 5 \
# --batch_size 2 \
# --delimiter '||$$||' \
# --round '1' \
# --log_dir    'leogb/causal_logs/DAMapDataset/e10_r18_preFalse_step_Cutmix_v1' \
# --train_file 'leogb/causal_logs/DAMapDataset/train_attrbute.txt' \
# --valid_file 'leogb/causal_logs/DAMapDataset/valid_attrbute.txt' \
# --OverwriteDAConfig \
# --Cutmix \
# --valid > e10_r18_preFalse_step_Cutmix_v1-DAMaplog_08.txt &

# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=32009 train.py \
# --distributed \
# --task_name 'e16_r18_preFalse_step_StyleMix_v1' \
# --mode 'train' \
# --gpus '3' \
# --train_ratio 5 \
# --batch_size 2 \
# --delimiter '||$$||' \
# --round '1' \
# --log_dir    'leogb/causal_logs/DAMapDataset/e16_r18_preFalse_step_StyleMix_v1' \
# --train_file 'leogb/causal_logs/DAMapDataset/train_attrbute.txt' \
# --valid_file 'leogb/causal_logs/DAMapDataset/valid_attrbute.txt' \
# --OverwriteDAConfig \
# --StyleMix \
# --StyleMix_method 'StyleMix' \
# --valid > e16_r18_preFalse_step_StyleMix_v1-DAMaplog_08.txt &

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=32010 train.py \
--distributed \
--task_name 'e15_r18_preFalse_step_StyleCutMix_v1' \
--mode 'train' \
--gpus '4' \
--train_ratio 5 \
--batch_size 2 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/DAMapDataset/e15_r18_preFalse_step_StyleCutMix_v1' \
--train_file 'leogb/causal_logs/DAMapDataset/train_attrbute.txt' \
--valid_file 'leogb/causal_logs/DAMapDataset/valid_attrbute.txt' \
--OverwriteDAConfig \
--StyleMix \
--StyleMix_method 'StyleCutMix' \
--valid > e15_r18_preFalse_step_StyleCutMix_v1-DAMaplog_08.txt &
