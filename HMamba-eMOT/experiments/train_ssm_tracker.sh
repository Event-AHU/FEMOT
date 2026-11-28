cd ssm_tracker

if false; then 
    python train.py --exp_name mambatrack_dancetrack2 --config_file cfgs/MambaTrack.yaml
fi

if true; then 
    python train.py --exp_name trackssm_dancetrack_sep_scale_one_dec_layer --config_file cfgs/TrackSSM.yaml
fi

cd ..