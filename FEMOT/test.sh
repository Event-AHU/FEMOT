source /usr/local/cuda-11.8/env.sh
source /gcc/11.2/env.sh
source activate MeMOTR

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.run \
--nproc_per_node=1 --master_port 40007 main.py  --mode submit \
--outputs-dir ./outputs/FEMOT/ \
--config-path ./outputs/FEMOT/train/config.yaml \
--submit-dir ./outputs/FEMOT/ --submit-model checkpoint_12.pth \
--use-distributed --data-root /rydata/mot/Datasets
