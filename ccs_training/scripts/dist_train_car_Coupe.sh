#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}

# e01_baseline_r18_preFalse_step_v3
# e09_r18_preFalse_step_Mixup_v1
# e10_r18_preFalse_step_Cutmix_v1
# e07_r18_preFalse_step_AutoAugment_v1
# e13_r18_preFalse_step_ManiMixup_v1
# e08_r18_preFalse_step_RandomErasing_v1
# e12_r18_preFalse_step_RandAug_v1
# e11_r18_preFalse_step_MoEx_v1
# e16_r18_preFalse_step_StyleMix_v1
# e15_r18_preFalse_step_StyleCutMix_v1
# e14_r18_preFalse_step_StyleCutMixAG_v1
# e17_r18_preFalse_step_OurX6_v1
# e18_r18_preFalse_step_OurX10_v1

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=33001 train.py \
--distributed \
--task_name 'e01_baseline_r18_preFalse_step_v3' \
--mode 'train' \
--gpus '0' \
--train_ratio 8 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/stanford_cars_Coupe/e01_baseline_r18_preFalse_step_v3' \
--train_file 'leogb/causal_logs/stanford_cars_Coupe/train_attrbute_Coupe.txt' \
--valid_file 'leogb/causal_logs/stanford_cars_Coupe/valid_attrbute_Coupe.txt' \
--valid > e01_baseline_r18_preFalse_step_v3-Coupelog_00.txt &

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=33002 train.py \
--distributed \
--task_name 'e09_r18_preFalse_step_Mixup_v1' \
--mode 'train' \
--gpus '1' \
--train_ratio 8 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/stanford_cars_Coupe/e09_r18_preFalse_step_Mixup_v1' \
--train_file 'leogb/causal_logs/stanford_cars_Coupe/train_attrbute_Coupe.txt' \
--valid_file 'leogb/causal_logs/stanford_cars_Coupe/valid_attrbute_Coupe.txt' \
--OverwriteDAConfig \
--Mixup \
--valid > e09_r18_preFalse_step_Mixup_v1-Coupelog_00.txt &

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=33003 train.py \
--distributed \
--task_name 'e10_r18_preFalse_step_Cutmix_v1' \
--mode 'train' \
--gpus '2' \
--train_ratio 8 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/stanford_cars_Coupe/e10_r18_preFalse_step_Cutmix_v1' \
--train_file 'leogb/causal_logs/stanford_cars_Coupe/train_attrbute_Coupe.txt' \
--valid_file 'leogb/causal_logs/stanford_cars_Coupe/valid_attrbute_Coupe.txt' \
--OverwriteDAConfig \
--Cutmix \
--valid > e10_r18_preFalse_step_Cutmix_v1-Coupelog_00.txt &

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=33004 train.py \
--distributed \
--task_name 'e07_r18_preFalse_step_AutoAugment_v1' \
--mode 'train' \
--gpus '3' \
--train_ratio 8 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/stanford_cars_Coupe/e07_r18_preFalse_step_AutoAugment_v1' \
--train_file 'leogb/causal_logs/stanford_cars_Coupe/train_attrbute_Coupe.txt' \
--valid_file 'leogb/causal_logs/stanford_cars_Coupe/valid_attrbute_Coupe.txt' \
--OverwriteDAConfig \
--AutoAugment \
--valid > e07_r18_preFalse_step_AutoAugment_v1-Coupelog_00.txt &

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=33005 train.py \
--distributed \
--task_name 'e13_r18_preFalse_step_ManiMixup_v1' \
--mode 'train' \
--gpus '4' \
--train_ratio 8 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/stanford_cars_Coupe/e13_r18_preFalse_step_ManiMixup_v1' \
--train_file 'leogb/causal_logs/stanford_cars_Coupe/train_attrbute_Coupe.txt' \
--valid_file 'leogb/causal_logs/stanford_cars_Coupe/valid_attrbute_Coupe.txt' \
--OverwriteDAConfig \
--Manifold_Mixup \
--valid > e13_r18_preFalse_step_ManiMixup_v1-Coupelog_00.txt &

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=33006 train.py \
--distributed \
--task_name 'e08_r18_preFalse_step_RandomErasing_v1' \
--mode 'train' \
--gpus '5' \
--train_ratio 8 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/stanford_cars_Coupe/e08_r18_preFalse_step_RandomErasing_v1' \
--train_file 'leogb/causal_logs/stanford_cars_Coupe/train_attrbute_Coupe.txt' \
--valid_file 'leogb/causal_logs/stanford_cars_Coupe/valid_attrbute_Coupe.txt' \
--OverwriteDAConfig \
--random_erasing \
--valid > e08_r18_preFalse_step_RandomErasing_v1-Coupelog_00.txt &

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=33007 train.py \
--distributed \
--task_name 'e12_r18_preFalse_step_RandAug_v1' \
--mode 'train' \
--gpus '6' \
--train_ratio 8 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/stanford_cars_Coupe/e12_r18_preFalse_step_RandAug_v1' \
--train_file 'leogb/causal_logs/stanford_cars_Coupe/train_attrbute_Coupe.txt' \
--valid_file 'leogb/causal_logs/stanford_cars_Coupe/valid_attrbute_Coupe.txt' \
--OverwriteDAConfig \
--RandAugment \
--valid > e12_r18_preFalse_step_RandAug_v1-Coupelog_00.txt &

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=33008 train.py \
--distributed \
--task_name 'e11_r18_preFalse_step_MoEx_v1' \
--mode 'train' \
--gpus '7' \
--train_ratio 8 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/stanford_cars_Coupe/e11_r18_preFalse_step_MoEx_v1' \
--train_file 'leogb/causal_logs/stanford_cars_Coupe/train_attrbute_Coupe.txt' \
--valid_file 'leogb/causal_logs/stanford_cars_Coupe/valid_attrbute_Coupe.txt' \
--OverwriteDAConfig \
--MoEx \
--valid > e11_r18_preFalse_step_MoEx_v1-Coupelog_00.txt &

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=33009 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '4' \
--train_ratio 8 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/stanford_cars_Coupe/e16_r18_preFalse_step_StyleMix_v1' \
--train_file 'leogb/causal_logs/stanford_cars_Coupe/train_attrbute_Coupe.txt' \
--valid_file 'leogb/causal_logs/stanford_cars_Coupe/valid_attrbute_Coupe.txt' \
--OverwriteDAConfig \
--StyleMix \
--StyleMix_method 'StyleMix' \
--valid > e16_r18_preFalse_step_StyleMix_v1-Coupelog_00.txt &

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=33010 train.py \
--distributed \
--task_name 'e15_r18_preFalse_step_StyleCutMix_v1' \
--mode 'train' \
--gpus '5' \
--train_ratio 8 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/stanford_cars_Coupe/e15_r18_preFalse_step_StyleCutMix_v1' \
--train_file 'leogb/causal_logs/stanford_cars_Coupe/train_attrbute_Coupe.txt' \
--valid_file 'leogb/causal_logs/stanford_cars_Coupe/valid_attrbute_Coupe.txt' \
--OverwriteDAConfig \
--StyleMix \
--StyleMix_method 'StyleCutMix' \
--valid > e15_r18_preFalse_step_StyleCutMix_v1-Coupelog_00.txt &

# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=33011 train.py \
# --distributed \
# --task_name 'e14_r18_preFalse_step_StyleCutMixAG_v1' \
# --mode 'train' \
# --gpus '2' \
# --train_ratio 8 \
# --delimiter '||$$||' \
# --round '1' \
# --log_dir    'leogb/causal_logs/stanford_cars_Coupe/e14_r18_preFalse_step_StyleCutMixAG_v1' \
# --train_file 'leogb/causal_logs/stanford_cars_Coupe/train_attrbute_Coupe.txt' \
# --valid_file 'leogb/causal_logs/stanford_cars_Coupe/valid_attrbute_Coupe.txt' \
# --OverwriteDAConfig \
# --StyleMix \
# --StyleMix_method 'StyleCutMix_Auto_Gamma' \
# --valid > e14_r18_preFalse_step_StyleCutMixAG_v1-Coupelog_00.txt &

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=33012 train.py \
--distributed \
--task_name 'e17_r18_preFalse_step_OurX6_v1' \
--mode 'train' \
--gpus '6' \
--train_ratio 8 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/stanford_cars_Coupe/e17_r18_preFalse_step_OurX6_v1' \
--train_file 'leogb/causal_logs/stanford_cars_Coupe/train_attrbute_Coupe.txt' \
--valid_file 'leogb/causal_logs/stanford_cars_Coupe/valid_attrbute_Coupe.txt' \
--OverwriteDAConfig \
--UMAx6 \
--valid > e17_r18_preFalse_step_OurX6_v1-Coupelog_00.txt &

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=33013 train.py \
--distributed \
--task_name 'e18_r18_preFalse_step_OurX10_v1' \
--mode 'train' \
--gpus '7' \
--train_ratio 8 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/stanford_cars_Coupe/e18_r18_preFalse_step_OurX10_v1' \
--train_file 'leogb/causal_logs/stanford_cars_Coupe/train_attrbute_Coupe.txt' \
--valid_file 'leogb/causal_logs/stanford_cars_Coupe/valid_attrbute_Coupe.txt' \
--OverwriteDAConfig \
--UMAx10 \
--valid > e18_r18_preFalse_step_OurX10_v1-Coupelog_00.txt &

