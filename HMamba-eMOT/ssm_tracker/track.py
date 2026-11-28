"""
Track from saved det results
"""

import numpy as np 
import os 
import argparse
from loguru import logger
import cv2 
import yaml
from copy import deepcopy

from tqdm import tqdm

from track_utils.tracker import MambaTracker

from train_utils.envs import select_device

def get_args():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--det_path', required=True, type=str)
    parser.add_argument('--motion_model_path', required=False, type=str, default=None)

    parser.add_argument('--config_file', required=True, type=str, default='cfgs/MambaTrack.yaml')

    parser.add_argument('--device', type=str, default='4')

    parser.add_argument('--data_root', type=str, default='/data/wujiapeng/datasets/MOT17/images/{split}/{seq}/{frame_id:06d}.jpg')
    parser.add_argument('--vis', action='store_true')

    parser.add_argument('--save_dir', type=str, default='track_results/{dataset_name}/{split}')
    parser.add_argument(
        '--run_name',
        type=str,
        default=None,
        help='name used for the prediction folder (defaults to the config filename)',
    )

    parser.add_argument('--debug', action='store_true', help='debug mode, the bbox is the predicted bbox by motion model')
    parser.add_argument(
        '--zero_motion',
        action='store_true',
        help='disable the motion model and predict zero displacement (hold last bbox)',
    )

    return parser.parse_args()

def save_results(folder_name, seq_name, results, data_type='default'):
    """
    write results to txt file

    results: list  row format: frame id, target id, box coordinate, class(optional)
    to_file: file path(optional)
    data_type: 'default' | 'mot_challenge', write data format, default or MOT submission
    """
    assert len(results)

    if not os.path.exists(f'./track_results/{folder_name}'):

        os.makedirs(f'./track_results/{folder_name}')

    with open(os.path.join('./track_results', folder_name, seq_name + '.txt'), 'w') as f:
        for frame_id, target_ids, tlwhs, clses, scores in results:
            for id, tlwh, score in zip(target_ids, tlwhs, scores):
                    f.write(f'{frame_id},{id},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{score:.2f},-1,-1,-1\n')

    f.close()

    return folder_name

def plot_img(img, frame_id, results, save_dir):
    """
    img: np.ndarray: (H, W, C)
    frame_id: int
    results: [tlwhs, ids, clses]
    save_dir: sr

    plot images with bboxes of a seq
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if len(img.shape) > 3:
        img = img.squeeze(0)

    img_ = np.ascontiguousarray(np.copy(img))

    tlwhs, ids, clses = results[0], results[1], results[2]
    for tlwh, id, cls in zip(tlwhs, ids, clses):

        # convert tlwh to tlbr
        tlbr = tuple([int(tlwh[0]), int(tlwh[1]), int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])])
        # draw a rect
        cv2.rectangle(img_, tlbr[:2], tlbr[2:], get_color(id), thickness=3, )
        # note the id and cls
        text = f'{cls}_{id}'
        cv2.putText(img_, text, (tlbr[0], tlbr[1]), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, 
                        color=(255, 164, 0), thickness=2)

    cv2.imwrite(filename=os.path.join(save_dir, f'{frame_id:05d}.jpg'), img=img_)

def get_color(idx):
    """
    aux func for plot_seq
    get a unique color for each id
    """
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

def read_meta_data(file_path):
    ret = dict()  # key: str, value: list
    with open(file_path, 'r') as f:

        while True:
            row = f.readline().strip()
            if not row: break

            items = row.split(',')
            ret[items[0]] = [float(items[i]) for i in range(1, len(items))]
    f.close()

    return ret

def main(args):

    config_file = args.config_file
    with open(config_file, 'r') as f:
        cfgs = yaml.safe_load(f)

    dataset_cfgs = cfgs['dataset']
    scale_factor = dataset_cfgs['scale_factor']
    scale_factor_diff = dataset_cfgs['scale_factor_diff'] if 'scale_factor_diff' in dataset_cfgs else None
    manner = dataset_cfgs['manner']
    train_cfgs = cfgs['train']
    inference_cfgs = cfgs['inference']
    # merge configs 
    cfgs = deepcopy(train_cfgs)
    cfgs.update(inference_cfgs)
    cfgs.update({'motion_model_path': args.motion_model_path})
    cfgs.update({'zero_motion': bool(args.zero_motion)})
    cfgs.update({'scale_factor': scale_factor})
    cfgs.update({'scale_factor_diff': scale_factor_diff})
    cfgs.update({'manner': manner})
    for key in (
        'hyper_spatial_radius',
        'hyper_rgb_threshold',
        'hyper_evt_threshold',
        'hyper_topk',
        'hyper_pairwise',
        'ablate_drop_relations',
        'ablate_keep_relations',
    ):
        if key in dataset_cfgs:
            cfgs[key] = dataset_cfgs[key]

    # set device
    device = select_device(args.device)

    exp_infos = args.det_path.split('/')

    
    dataset_name, split = exp_infos[-2], exp_infos[-1]
    
    seq_dets = sorted(os.listdir(args.det_path))

    save_dir = args.save_dir.format(dataset_name=dataset_name, split=split)

    # read meta data (including the h and w of image)
    meta_data = read_meta_data(os.path.join(args.det_path, 'meta_data.txt'))


    for seq in seq_dets:

        if not '.txt' in seq or 'meta_data' in seq:  # exclude meta data file
            continue
        
        logger.info(f'tracking seq {seq}')

        file_name = os.path.join(args.det_path, seq)        
        file_content = np.loadtxt(file_name, dtype=float, delimiter=',')

        # get max frames 
        max_frame_id = file_content[:, 0].max()
        max_frame_id = int(max_frame_id)

        # init tracker
        trakcer = MambaTracker(cfgs, device=device)

        feat_file = os.path.join(args.det_path, seq[:-4] + '_feat.npz')
        seq_feats = np.load(feat_file) if os.path.exists(feat_file) else None
        rgb_feat_file = os.path.join(args.det_path, seq[:-4] + '_rgb_feat.npz')
        evt_feat_file = os.path.join(args.det_path, seq[:-4] + '_evt_feat.npz')
        seq_rgb_feats = np.load(rgb_feat_file) if os.path.exists(rgb_feat_file) else None
        seq_evt_feats = np.load(evt_feat_file) if os.path.exists(evt_feat_file) else None
        if seq_feats is not None:
            logger.info(f'loaded appearance feats from {feat_file}')

        results = []

        process_bar = tqdm([i for i in range(1, max_frame_id + 1)], ncols=150)

        # update sequence meta data that tracker need
        seq_meta_data = meta_data[seq[:-4]]

        for frame_id in range(1, max_frame_id + 1):

            process_bar.update()

            current_det = file_content[file_content[:, 0] == float(frame_id)]
            det_app_feats = None
            det_rgb_feats = None
            det_evt_feats = None
            if seq_feats is not None:
                feat_key = f'f{frame_id}'
                if feat_key in seq_feats.files:
                    frame_feats = seq_feats[feat_key]
                    if len(frame_feats) == len(current_det):
                        det_app_feats = frame_feats
            if seq_rgb_feats is not None and f'f{frame_id}' in seq_rgb_feats.files:
                frame_feats = seq_rgb_feats[f'f{frame_id}']
                if len(frame_feats) == len(current_det):
                    det_rgb_feats = frame_feats
            if seq_evt_feats is not None and f'f{frame_id}' in seq_evt_feats.files:
                frame_feats = seq_evt_feats[f'f{frame_id}']
                if len(frame_feats) == len(current_det):
                    det_evt_feats = frame_feats

            current_det = current_det[:, 2: 8]

            output_tracklets = trakcer.update(
                current_det,
                meta_data=seq_meta_data,
                det_app_feats=det_app_feats,
                det_rgb_feats=det_rgb_feats,
                det_evt_feats=det_evt_feats,
            )
            
            # save results
            cur_tlwh, cur_id, cur_cls, cur_score = [], [], [], []
            for trk in output_tracklets:

                if args.debug:
                    if trk.predicted_last_bbox is None:
                        bbox = trk.get_bbox()
                    else:
                        bbox = trk.predicted_last_bbox 
                        bbox[:2] -= 0.5 * bbox[2:]
                else:
                    bbox = trk.get_bbox()
                    
                id = trk.track_id
                cls = trk.category
                score = trk.score

                # TODO filter box
                cur_tlwh.append(bbox)
                cur_id.append(id)
                cur_cls.append(cls)
                cur_score.append(score)

            results.append((frame_id, cur_id, cur_tlwh, cur_cls, cur_score))

            if args.vis:
                cur_frame_path = os.path.join(args.data_root.format(split=split, seq=seq[:-4], frame_id=frame_id))
               
                cur_frame = cv2.imread(cur_frame_path)
                assert cur_frame is not None, cur_frame_path
                plot_img(img=cur_frame, frame_id=frame_id, results=[cur_tlwh, cur_id, cur_cls], 
                         save_dir=os.path.join(save_dir, 'vis_results'))
                
        logger.info(f'saving results of {seq}')

        model_name = args.run_name or args.config_file.split('/')[-1].split('.')[0]
        save_results(folder_name=os.path.join(dataset_name + '_' + model_name, split), 
                     seq_name=seq[:-4], 
                     results=results)


if __name__ == '__main__':
    args = get_args()
    main(args)
