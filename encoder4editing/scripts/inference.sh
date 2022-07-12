CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/inference_editing_da.py \
--anno_file='leogb/causal_logs/CelebAMask-HQ-Attr9/train_attrbute_9.txt' \
--save_dir='./testdata_output/CelebAMask-HQ-Attr9/' \
'./ckpt/e4e_ffhq_encode.pt' > log_09.log &

CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/inference_editing_da.py \
--anno_file='leogb/causal_logs/CelebAMask-HQ-Attr10/train_attrbute_10.txt' \
--save_dir='./testdata_output/CelebAMask-HQ-Attr10/' \
'./ckpt/e4e_ffhq_encode.pt' > log_10.log &

CUDA_VISIBLE_DEVICES=2 nohup python -u scripts/inference_editing_da.py \
--anno_file='leogb/causal_logs/CelebAMask-HQ-Attr12/train_attrbute_12.txt' \
--save_dir='./testdata_output/CelebAMask-HQ-Attr12/' \
'./ckpt/e4e_ffhq_encode.pt' > log_12.log &

CUDA_VISIBLE_DEVICES=3 nohup python -u scripts/inference_editing_da.py \
--anno_file='leogb/causal_logs/CelebAMask-HQ-Attr18/train_attrbute_18.txt' \
--save_dir='./testdata_output/CelebAMask-HQ-Attr18/' \
'./ckpt/e4e_ffhq_encode.pt' > log_18.log &


CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/inference_editing_da.py \
--anno_file='leogb/causal_logs/CelebAMask-HQ-Attr5/train_attrbute_5.txt' \
--save_dir='./testdata_output/CelebAMask-HQ-Attr5/' \
'./ckpt/e4e_ffhq_encode.pt' > log_05.log &

CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/inference_editing_da.py \
--anno_file='leogb/causal_logs/CelebAMask-HQ-Attr6/train_attrbute_6.txt' \
--save_dir='./testdata_output/CelebAMask-HQ-Attr6/' \
'./ckpt/e4e_ffhq_encode.pt' > log_06.log &

CUDA_VISIBLE_DEVICES=2 nohup python -u scripts/inference_editing_da.py \
--anno_file='leogb/causal_logs/CelebAMask-HQ-Attr33/train_attrbute_33.txt' \
--save_dir='./testdata_output/CelebAMask-HQ-Attr33/' \
'./ckpt/e4e_ffhq_encode.pt' > log_33.log &

CUDA_VISIBLE_DEVICES=3 nohup python -u scripts/inference_editing_da.py \
--anno_file='leogb/causal_logs/CelebAMask-HQ-Attr34/train_attrbute_34.txt' \
--save_dir='./testdata_output/CelebAMask-HQ-Attr34/' \
'./ckpt/e4e_ffhq_encode.pt' > log_34.log &


CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/inference_editing_da.py \
--anno_file='leogb/causal_logs/CelebAMask-HQ-Attr19/train_attrbute_19.txt' \
--save_dir='./testdata_output/CelebAMask-HQ-Attr19/' \
'./ckpt/e4e_ffhq_encode.pt' > log_19.log &

CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/inference_editing_da.py \
--anno_file='leogb/causal_logs/CelebAMask-HQ-Attr32/train_attrbute_32.txt' \
--save_dir='./testdata_output/CelebAMask-HQ-Attr32/' \
'./ckpt/e4e_ffhq_encode.pt' > log_32.log &

CUDA_VISIBLE_DEVICES=2 nohup python -u scripts/inference_editing_da.py \
--anno_file='leogb/causal_logs/CelebAMask-HQ-Attr16/train_attrbute_16.txt' \
--save_dir='./testdata_output/CelebAMask-HQ-Attr16/' \
'./ckpt/e4e_ffhq_encode.pt' > log_16.log &

CUDA_VISIBLE_DEVICES=3 nohup python -u scripts/inference_editing_da.py \
--anno_file='leogb/causal_logs/CelebAMask-HQ-Attr21/train_attrbute_21.txt' \
--save_dir='./testdata_output/CelebAMask-HQ-Attr21/' \
'./ckpt/e4e_ffhq_encode.pt' > log_21.log &


CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/inference_editing_da.py \
--anno_file='leogb/causal_logs/CelebAMask-HQ-ComAttr1-19-6/train_attrbute_CA1.txt' \
--save_dir='./testdata_output/CelebAMask-HQ-ComAttr1-19-6/' \
'./ckpt/e4e_ffhq_encode.pt' > log_ComAttr1.log &

CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/inference_editing_da.py \
--anno_file='leogb/causal_logs/CelebAMask-HQ-ComAttr2-19-33/train_attrbute_CA2.txt' \
--save_dir='./testdata_output/CelebAMask-HQ-ComAttr2-19-33/' \
'./ckpt/e4e_ffhq_encode.pt' > log_ComAttr2.log &

CUDA_VISIBLE_DEVICES=2 nohup python -u scripts/inference_editing_da.py \
--anno_file='leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/train_attrbute_CA3.txt' \
--save_dir='./testdata_output/CelebAMask-HQ-ComAttr3-19-32/' \
'./ckpt/e4e_ffhq_encode.pt' > log_ComAttr3.log &

CUDA_VISIBLE_DEVICES=3 nohup python -u scripts/inference_editing_da.py \
--anno_file='leogb/causal_logs/CelebAMask-HQ-ComAttr4-32-10/train_attrbute_CA4.txt' \
--save_dir='./testdata_output/CelebAMask-HQ-ComAttr4-32-10/' \
'./ckpt/e4e_ffhq_encode.pt' > log_ComAttr4.log &

CUDA_VISIBLE_DEVICES=4 nohup python -u scripts/inference_editing_da.py \
--anno_file='leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/train_attrbute_CA5.txt' \
--save_dir='./testdata_output/CelebAMask-HQ-ComAttr5-21-9/' \
'./ckpt/e4e_ffhq_encode.pt' > log_ComAttr5.log &

