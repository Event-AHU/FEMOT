source /usr/local/cuda-11.8/env.sh
source /gcc/11.2/env.sh
source activate MeMOTR

NAME="FEMOT" 
if [ ! -d "log" ]; then   
    mkdir "log"  
fi  

echo "Starting process with PID: $$" > log/${NAME}.log
echo "log/${NAME}.log is starting!" 
# echo "log/${NAME}.log is starting! "

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.run --master_port 36371 --nproc_per_node=1 main.py \
--use-distributed --config-path ./configs/femot.yaml \
--outputs-dir ./outputs/FEMOT/ \
--batch-size 1 --data-root /wangx/_dataset/ \
# >>  log/${NAME}.log 2>&1 &
