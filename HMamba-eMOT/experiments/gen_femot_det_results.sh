# FEMOT detection: RGB images live under {split}/{seq}/{seq}_aps/
python tools/gen_det_results.py \
    --dataset_name femot \
    --data_root /rydata/wangx_dataset/_dataset/FEMOT \
    --split val \
    --img_subdir aps \
    --exp_file yolox_exps/custom/yolox_femot.py \
    --model_path weights/yolox_femot_rgb_best.pth.tar \
    --img_size 416 416 \
    --generate_meta_data
