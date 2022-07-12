#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}

# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=37000 train.py \
# --distributed \
# --task_name 'e19_r18_preFalse_step_PseudoLabel_v1' \
# --mode 'train' \
# --gpus '0' \
# --train_ratio 8 \
# --delimiter '||$$||' \
# --round '1' \
# --log_dir    'leogb/causal_logs/stanford_cars_v2_Sedan/e19_r18_preFalse_step_PseudoLabel_v1/' \
# --train_file 'leogb/causal_logs/stanford_cars_v2_Sedan/train_attrbute_Sedan.txt' \
# --valid_file 'leogb/causal_logs/stanford_cars_v2_Sedan/valid_attrbute_Sedan.txt' \
# --OverwriteDAConfig \
# --UMAx10_PLv1 \
# --valid > e19_r18_preFalse_step_PseudoLabel_v1-Sedanlog_01.txt &

# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=37001 train.py \
# --distributed \
# --task_name 'e19_r18_preFalse_step_PseudoLabel_v2' \
# --mode 'train' \
# --gpus '1' \
# --train_ratio 8 \
# --delimiter '||$$||' \
# --round '1' \
# --log_dir    'leogb/causal_logs/stanford_cars_v2_Sedan/e19_r18_preFalse_step_PseudoLabel_v2/' \
# --train_file 'leogb/causal_logs/stanford_cars_v2_Sedan/train_attrbute_Sedan.txt' \
# --valid_file 'leogb/causal_logs/stanford_cars_v2_Sedan/valid_attrbute_Sedan.txt' \
# --OverwriteDAConfig \
# --UMAx10_PLv2 \
# --valid > e19_r18_preFalse_step_PseudoLabel_v2-Sedanlog_02.txt &

# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=37002 train.py \
# --distributed \
# --task_name 'e19_r18_preFalse_step_PseudoLabel_v3' \
# --mode 'train' \
# --gpus '2' \
# --train_ratio 8 \
# --delimiter '||$$||' \
# --round '1' \
# --log_dir    'leogb/causal_logs/stanford_cars_v2_Sedan/e19_r18_preFalse_step_PseudoLabel_v3/' \
# --train_file 'leogb/causal_logs/stanford_cars_v2_Sedan/train_attrbute_Sedan.txt' \
# --valid_file 'leogb/causal_logs/stanford_cars_v2_Sedan/valid_attrbute_Sedan.txt' \
# --OverwriteDAConfig \
# --UMAx10_PLv3 \
# --valid > e19_r18_preFalse_step_PseudoLabel_v3-Sedanlog_03.txt &

# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=37003 train.py \
# --distributed \
# --task_name 'e19_r18_preFalse_step_PseudoLabel_v4' \
# --mode 'train' \
# --gpus '3' \
# --train_ratio 8 \
# --delimiter '||$$||' \
# --round '1' \
# --log_dir    'leogb/causal_logs/stanford_cars_v2_Sedan/e19_r18_preFalse_step_PseudoLabel_v4/' \
# --train_file 'leogb/causal_logs/stanford_cars_v2_Sedan/train_attrbute_Sedan.txt' \
# --valid_file 'leogb/causal_logs/stanford_cars_v2_Sedan/valid_attrbute_Sedan.txt' \
# --OverwriteDAConfig \
# --UMAx10_PLv4 \
# --valid > e19_r18_preFalse_step_PseudoLabel_v4-Sedanlog_04.txt &

# ----------

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=37012 train.py \
--distributed \
--task_name 'e19_r18_preFalse_step_PseudoLabel_v1' \
--mode 'train' \
--gpus '4' \
--train_ratio 8 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/stanford_cars_v2_SUV/e19_r18_preFalse_step_PseudoLabel_v1/' \
--train_file 'leogb/causal_logs/stanford_cars_v2_SUV/train_attrbute_SUV.txt' \
--valid_file 'leogb/causal_logs/stanford_cars_v2_SUV/valid_attrbute_SUV.txt' \
--OverwriteDAConfig \
--UMAx10_PLv1 \
--valid > e19_r18_preFalse_step_PseudoLabel_v1-SUVlog_01.txt &

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=37013 train.py \
--distributed \
--task_name 'e19_r18_preFalse_step_PseudoLabel_v2' \
--mode 'train' \
--gpus '5' \
--train_ratio 8 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/stanford_cars_v2_SUV/e19_r18_preFalse_step_PseudoLabel_v2/' \
--train_file 'leogb/causal_logs/stanford_cars_v2_SUV/train_attrbute_SUV.txt' \
--valid_file 'leogb/causal_logs/stanford_cars_v2_SUV/valid_attrbute_SUV.txt' \
--OverwriteDAConfig \
--UMAx10_PLv2 \
--valid > e19_r18_preFalse_step_PseudoLabel_v2-SUVlog_02.txt &

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=37014 train.py \
--distributed \
--task_name 'e19_r18_preFalse_step_PseudoLabel_v3' \
--mode 'train' \
--gpus '6' \
--train_ratio 8 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/stanford_cars_v2_SUV/e19_r18_preFalse_step_PseudoLabel_v3/' \
--train_file 'leogb/causal_logs/stanford_cars_v2_SUV/train_attrbute_SUV.txt' \
--valid_file 'leogb/causal_logs/stanford_cars_v2_SUV/valid_attrbute_SUV.txt' \
--OverwriteDAConfig \
--UMAx10_PLv3 \
--valid > e19_r18_preFalse_step_PseudoLabel_v3-SUVlog_03.txt &

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=37015 train.py \
--distributed \
--task_name 'e19_r18_preFalse_step_PseudoLabel_v4' \
--mode 'train' \
--gpus '7' \
--train_ratio 8 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/stanford_cars_v2_SUV/e19_r18_preFalse_step_PseudoLabel_v4/' \
--train_file 'leogb/causal_logs/stanford_cars_v2_SUV/train_attrbute_SUV.txt' \
--valid_file 'leogb/causal_logs/stanford_cars_v2_SUV/valid_attrbute_SUV.txt' \
--OverwriteDAConfig \
--UMAx10_PLv4 \
--valid > e19_r18_preFalse_step_PseudoLabel_v4-SUVlog_04.txt &

# ----------

# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=37004 train.py \
# --distributed \
# --task_name 'e19_r18_preFalse_step_PseudoLabel_v1' \
# --mode 'train' \
# --gpus '4' \
# --train_ratio 1 \
# --delimiter '||$$||' \
# --round '1' \
# --log_dir    'leogb/causal_logs/horse_v2_brown/e19_r18_preFalse_step_PseudoLabel_v1/' \
# --train_file 'leogb/causal_logs/horse_v2_brown/train_attrbute_brown.txt' \
# --valid_file 'leogb/causal_logs/horse_v2_brown/valid_attrbute_brown.txt' \
# --OverwriteDAConfig \
# --UMAx10_PLv1 \
# --valid > e19_r18_preFalse_step_PseudoLabel_v1-brownhorselog_01.txt &

# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=37005 train.py \
# --distributed \
# --task_name 'e19_r18_preFalse_step_PseudoLabel_v2' \
# --mode 'train' \
# --gpus '5' \
# --train_ratio 1 \
# --delimiter '||$$||' \
# --round '1' \
# --log_dir    'leogb/causal_logs/horse_v2_brown/e19_r18_preFalse_step_PseudoLabel_v2/' \
# --train_file 'leogb/causal_logs/horse_v2_brown/train_attrbute_brown.txt' \
# --valid_file 'leogb/causal_logs/horse_v2_brown/valid_attrbute_brown.txt' \
# --OverwriteDAConfig \
# --UMAx10_PLv2 \
# --valid > e19_r18_preFalse_step_PseudoLabel_v2-brownhorselog_02.txt &

# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=37006 train.py \
# --distributed \
# --task_name 'e19_r18_preFalse_step_PseudoLabel_v3' \
# --mode 'train' \
# --gpus '6' \
# --train_ratio 1 \
# --delimiter '||$$||' \
# --round '1' \
# --log_dir    'leogb/causal_logs/horse_v2_brown/e19_r18_preFalse_step_PseudoLabel_v3/' \
# --train_file 'leogb/causal_logs/horse_v2_brown/train_attrbute_brown.txt' \
# --valid_file 'leogb/causal_logs/horse_v2_brown/valid_attrbute_brown.txt' \
# --OverwriteDAConfig \
# --UMAx10_PLv3 \
# --valid > e19_r18_preFalse_step_PseudoLabel_v3-brownhorselog_03.txt &

# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=37007 train.py \
# --distributed \
# --task_name 'e19_r18_preFalse_step_PseudoLabel_v4' \
# --mode 'train' \
# --gpus '7' \
# --train_ratio 1 \
# --delimiter '||$$||' \
# --round '1' \
# --log_dir    'leogb/causal_logs/horse_v2_brown/e19_r18_preFalse_step_PseudoLabel_v4/' \
# --train_file 'leogb/causal_logs/horse_v2_brown/train_attrbute_brown.txt' \
# --valid_file 'leogb/causal_logs/horse_v2_brown/valid_attrbute_brown.txt' \
# --OverwriteDAConfig \
# --UMAx10_PLv4 \
# --valid > e19_r18_preFalse_step_PseudoLabel_v4-brownhorselog_04.txt &

# ----------

# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=37008 train.py \
# --distributed \
# --task_name 'e19_r18_preFalse_step_PseudoLabel_v1' \
# --mode 'train' \
# --gpus '0' \
# --train_ratio 1 \
# --delimiter '||$$||' \
# --round '1' \
# --log_dir    'leogb/causal_logs/horse_v2_white/e19_r18_preFalse_step_PseudoLabel_v1/' \
# --train_file 'leogb/causal_logs/horse_v2_white/train_attrbute_white.txt' \
# --valid_file 'leogb/causal_logs/horse_v2_white/valid_attrbute_white.txt' \
# --OverwriteDAConfig \
# --UMAx10_PLv1 \
# --valid > e19_r18_preFalse_step_PseudoLabel_v1-whitehorse_01.txt &

# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=37009 train.py \
# --distributed \
# --task_name 'e19_r18_preFalse_step_PseudoLabel_v2' \
# --mode 'train' \
# --gpus '1' \
# --train_ratio 1 \
# --delimiter '||$$||' \
# --round '1' \
# --log_dir    'leogb/causal_logs/horse_v2_white/e19_r18_preFalse_step_PseudoLabel_v2/' \
# --train_file 'leogb/causal_logs/horse_v2_white/train_attrbute_white.txt' \
# --valid_file 'leogb/causal_logs/horse_v2_white/valid_attrbute_white.txt' \
# --OverwriteDAConfig \
# --UMAx10_PLv2 \
# --valid > e19_r18_preFalse_step_PseudoLabel_v2-whitehorse_02.txt &

# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=37010 train.py \
# --distributed \
# --task_name 'e19_r18_preFalse_step_PseudoLabel_v3' \
# --mode 'train' \
# --gpus '2' \
# --train_ratio 1 \
# --delimiter '||$$||' \
# --round '1' \
# --log_dir    'leogb/causal_logs/horse_v2_white/e19_r18_preFalse_step_PseudoLabel_v3/' \
# --train_file 'leogb/causal_logs/horse_v2_white/train_attrbute_white.txt' \
# --valid_file 'leogb/causal_logs/horse_v2_white/valid_attrbute_white.txt' \
# --OverwriteDAConfig \
# --UMAx10_PLv3 \
# --valid > e19_r18_preFalse_step_PseudoLabel_v3-whitehorse_03.txt &

# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=37011 train.py \
# --distributed \
# --task_name 'e19_r18_preFalse_step_PseudoLabel_v4' \
# --mode 'train' \
# --gpus '3' \
# --train_ratio 1 \
# --delimiter '||$$||' \
# --round '1' \
# --log_dir    'leogb/causal_logs/horse_v2_white/e19_r18_preFalse_step_PseudoLabel_v4/' \
# --train_file 'leogb/causal_logs/horse_v2_white/train_attrbute_white.txt' \
# --valid_file 'leogb/causal_logs/horse_v2_white/valid_attrbute_white.txt' \
# --OverwriteDAConfig \
# --UMAx10_PLv4 \
# --valid > e19_r18_preFalse_step_PseudoLabel_v4-whitehorse_04.txt &
