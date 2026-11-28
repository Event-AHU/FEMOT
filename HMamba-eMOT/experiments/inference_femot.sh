cd ssm_tracker

python track.py \
    --det_path ../det_results/femot/val \
    --motion_model_path saved_ckpts/mambatrack_femot/epoch25.pth \
    --config_file cfgs/MambaTrack_femot.yaml \
    --data_root /rydata/wangx_dataset/_dataset/FEMOT/{split}/{seq}/{seq}_aps/{frame_id:05d}.jpg

cd ..
