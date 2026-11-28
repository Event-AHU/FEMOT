cd ssm_tracker

if false; then
    python track.py \
        --det_path ../det_results/mot17/train \
        --motion_model_path saved_ckpts/mambatrack_mot17/epoch10.pth \
        --config_file cfgs/MambaTrack.yaml \
        --data_root /data/wujiapeng/datasets/MOT17/images/{split}/{seq}/{frame_id:06d}.jpg
fi

if true; then
    python track.py \
        --det_path ../det_results/dancetrack/val \
        --motion_model_path saved_ckpts/mambatrack_dancetrack/epoch25.pth \
        --config_file cfgs/MambaTrack.yaml \
        --data_root /data/wujiapeng/datasets/DanceTrack/images/{split}/{seq}/{frame_id:08d}.jpg --vis
fi

if false; then
    python track.py \
        --det_path ../det_results/visdrone/test \
        --motion_model_path saved_ckpts/mambatrack_visdrone/epoch10.pth \
        --config_file cfgs/MambaTrack.yaml \
        --data_root /data/wujiapeng/datasets/VisDrone2019/VisDrone2019/images/{split}/{seq}/{frame_id:06d}.jpg
fi

cd ../
# --det_path ../det_results/dancetrack/train --motion_model_path saved_ckpts/trackssm_dancetrack_sep_scale/epoch20.pth --config_file cfgs/TrackSSM.yaml --data_root /data/wujiapeng/datasets/DanceTrack/images/{split}/{seq}/{frame_id:08d}.jpg
# --det_path ../det_results/dancetrack/train --motion_model_path saved_ckpts/mambatrack_dancetrack/epoch25.pth --config_file cfgs/MambaTrack.yaml --data_root /data/wujiapeng/datasets/DanceTrack/images/{split}/{seq}/{frame_id:08d}.jpg
# --det_path ../det_results/mot17/train --motion_model_path saved_ckpts/mambatrack_dancetrack/epoch25.pth --config_file cfgs/MambaTrack.yaml --data_root /data/wujiapeng/datasets/MOT17/images/{split}/{seq}/{frame_id:06d}.jpg
# --data_root /data/wujiapeng/datasets/MOT17/images/{split}/{seq}/{frame_id:06d}.jpg --vis 