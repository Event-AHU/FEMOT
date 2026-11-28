"""
Track from saved det results
"""

import numpy as np 
import os 
import argparse
from loguru import logger
import cv2 

from tqdm import tqdm

from naive_tracker import NaiveTracker

def get_args():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--det_path', required=True, type=str)

    parser.add_argument('--motion', required=True, type=str, default='byte')

    parser.add_argument('--track_thresh', type=float, default=0.5)
    parser.add_argument('--track_buffer', type=int, default=30)
    parser.add_argument("--match_thresh", type=float, default=0.9)

    parser.add_argument('--data_root', type=str, default='/data/wujiapeng/datasets/MOT17/images/{split}/{seq}/{frame_id:06d}.jpg')
    parser.add_argument('--vis', action='store_true')

    parser.add_argument('--save_dir', type=str, default='track_results/{dataset_name}/{split}')

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

    assert img is not None

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

def main(args):

    exp_infos = args.det_path.split('/')
    dataset_name, split = exp_infos[-2], exp_infos[-1]
    
    seq_dets = sorted(os.listdir(args.det_path))

    save_dir = args.save_dir.format(dataset_name=dataset_name, split=split)

    for seq in seq_dets:

        if not '.txt' in seq or 'meta_data' in seq: 
            continue
        
        logger.info(f'\ntracking seq {seq}')

        file_name = os.path.join(args.det_path, seq)        
        file_content = np.loadtxt(file_name, dtype=float, delimiter=',')

        # get max frames 
        max_frame_id = file_content[:, 0].max()
        max_frame_id = int(max_frame_id)

        # init tracker
        trakcer = NaiveTracker(args, )

        results = []

        process_bar = tqdm([i for i in range(1, max_frame_id + 1)], ncols=150)

        for frame_id in range(1, max_frame_id + 1):

            process_bar.update()

            current_det = file_content[file_content[:, 0] == float(frame_id)]
            current_det = current_det[:, 2: 8]

            output_tracklets = trakcer.update(current_det, )
            
            # save results
            cur_tlwh, cur_id, cur_cls, cur_score = [], [], [], []
            for trk in output_tracklets:
                bbox = trk.tlwh
                id = trk.track_id
                cls = 1
                score = trk.score

                # TODO filter box
                cur_tlwh.append (bbox)
                cur_id.append(id)
                cur_cls.append(cls)
                cur_score.append(score)

            results.append((frame_id, cur_id, cur_tlwh, cur_cls, cur_score))

            if args.vis:
                cur_frame_path = os.path.join(args.data_root.format(split=split, seq=seq[:-4], frame_id=frame_id))
               
                cur_frame = cv2.imread(cur_frame_path)
                plot_img(img=cur_frame, frame_id=frame_id, results=[cur_tlwh, cur_id, cur_cls], 
                         save_dir=os.path.join(save_dir, 'vis_results'))

        save_results(folder_name=os.path.join(dataset_name + '_' + args.motion, split), 
                     seq_name=seq[:-4], 
                     results=results)


if __name__ == '__main__':
    args = get_args()
    main(args)