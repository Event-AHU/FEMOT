import argparse
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'third_party', 'ByteTrack'))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'yolox_exps', 'custom'))

import numpy as np
import torch
import cv2 
from PIL import Image
from tqdm import tqdm

from time import gmtime, strftime

from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.data.data_augment import preproc

from loguru import logger

from dual_data import dual_preproc

try:
    from torchvision.ops import roi_align as tv_roi_align
except ImportError:
    tv_roi_align = None

FEAT_STRIDE = 8
FEAT_POOL_SIZE = 7


def extract_det_feats(fpn_feat, tlwhs, ratio, feat_stride=FEAT_STRIDE, pool_size=FEAT_POOL_SIZE):
    """
    RoI-Align on the highest-resolution FPN level, then GAP + L2 norm.
    fpn_feat: (1, C, H, W), tlwhs: (N, 4) in original image coordinates.
    """
    if tlwhs is None or len(tlwhs) == 0:
        return np.zeros((0, fpn_feat.shape[1]), dtype=np.float32)

    if tv_roi_align is None:
        raise ImportError('torchvision is required for --save_feats (roi_align)')

    rois = []
    for tlwh in tlwhs:
        x, y, w, h = tlwh
        x1 = x * ratio
        y1 = y * ratio
        x2 = (x + w) * ratio
        y2 = (y + h) * ratio
        rois.append([0.0, x1, y1, x2, y2])

    rois = torch.tensor(rois, dtype=torch.float32, device=fpn_feat.device)
    pooled = tv_roi_align(
        fpn_feat,
        rois,
        output_size=(pool_size, pool_size),
        spatial_scale=1.0 / feat_stride,
        aligned=True,
    )
    feats = pooled.mean(dim=(2, 3))
    feats = torch.nn.functional.normalize(feats, dim=1)
    return feats.detach().cpu().numpy().astype(np.float32)

def get_args():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', required=True, type=str, default='visdrone')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--data_root', required=True, type=str, default='/data/wujiapeng/datasets/VisDrone2019/VisDrone2019/images')
    parser.add_argument('--exp_file', required=True, type=str, default='')
    parser.add_argument('--model_path', required=True, type=str, default='weights/yolo')

    parser.add_argument('--img_size', nargs='+', type=int, default=[800, 1440], help='[train, test] image sizes')
    parser.add_argument('--high_thresh', type=float, default=0.5)

    parser.add_argument('--save_dir', type=str, default='det_results/{dataset_name}/{split}')

    parser.add_argument('--device', type=str, default='1')

    parser.add_argument('--fp16', type=bool, default=False)

    parser.add_argument('--vis', action='store_true')

    parser.add_argument('--generate_meta_data', action='store_true', help='generate meta data for ssm tracker, such as img h and w')
    parser.add_argument('--img_subdir', type=str, default='', help='image subdir under each sequence, e.g. fusion for FEMOT')
    parser.add_argument('--dual_modal', action='store_true', help='use RGB+event (_aps + _dvs) for FEMOT dual YOLOX')
    parser.add_argument('--save_feats', action='store_true', help='save dual-modal RoI appearance features (.npz per seq)')
    parser.add_argument('--save_modal_feats', action='store_true', help='also save pre-fusion RGB/Event RoI features')

    return parser.parse_args()

def select_device(device):
    """ set device 
    Args:
        device: str, 'cpu' or '0' or '1,2,3'-like

    Return:
        torch.device
    
    """

    if device == 'cpu':
        logger.info('Use CPU for training')

    elif ',' in device:  # multi-gpu
        logger.error('Multi-GPU currently not supported')
    
    else:
        # Keep a device chosen before this process imported torch.
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            logger.info(
                f'keep CUDA_VISIBLE_DEVICES={os.environ["CUDA_VISIBLE_DEVICES"]} '
                f'(ignore --device={device})'
            )
        else:
            logger.info(f'set gpu {device}')
            os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available()

    cuda = device != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    return device

def postprocess_yolox(out, num_classes, conf_thresh, img, ori_img):
    """
    convert out to  -> (tlbr, conf, cls)
    """

    out = postprocess(out, num_classes, conf_thresh, )[0]  # (tlbr, obj_conf, cls_conf, cls)

    if out is None: return out

    # merge conf 
    out[:, 4] *= out[:, 5]
    out[:, 5] = out[:, -1]
    out = out[:, :-1]

    # scale to origin size 

    img_size = [img.shape[-2], img.shape[-1]]  # h, w
    ori_img_size = [ori_img.shape[0], ori_img.shape[1]]  # h0, w0
    img_h, img_w = img_size[0], img_size[1]

    scale = min(float(img_h) / ori_img_size[0], float(img_w) / ori_img_size[1])

    out[:, :4] /= scale 

    return out

def save_results(folder_name, seq_name, result_dict, data_type='default'):
    """
    write results to txt file

    """
    

    if not os.path.exists(folder_name):
        os.makedirs(folder_name) 

    with open(os.path.join(folder_name, seq_name + '.txt'), 'w') as f:
        
        for frame_id, output in result_dict.items():

            for det in output:
                f.write(f'{frame_id},-1,{det[0]:.2f},{det[1]:.2f},{det[2]:.2f},{det[3]:.2f},{det[4]:.2f},-1,-1,-1\n')

    f.close()

    return folder_name

def save_meta_data(folder_name, meta_data):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name) 

    with open(os.path.join(folder_name, 'meta_data.txt'), 'w') as f:
        for k, v in meta_data.items():
            line = k + ','
            for item in v: line += str(item) + ','
            line = line[:-1]  # drop the last comma

            f.write(line + '\n')
    f.close()

def plot_img(img, frame_id, results, save_dir):
    """
    visualization
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if len(img.shape) > 3:
        img = img.squeeze(0)

    img_ = np.ascontiguousarray(np.copy(img))

    for det in results:
        tlwh, s = det[: 4], det[4]
        tlbr = tuple([int(tlwh[0]), int(tlwh[1]), int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])])
        text = '{:.2f}'.format(s)

        cv2.rectangle(img_, tlbr[:2], tlbr[2:], (0, 255, 0), thickness=3, )

        cv2.putText(img_, text, (tlbr[0], tlbr[1]), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, 
                        color=(255, 164, 0), thickness=2)
        
    cv2.imwrite(filename=os.path.join(save_dir, f'{frame_id:05d}.jpg'), img=img_)

def main(args):
    
    exp = get_exp(args.exp_file, args.dataset_name)

    device = select_device(args.device)
    model = exp.get_model().to(device)

    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    # load ckpt
    ckpt = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")
    if args.save_modal_feats:
        if not args.dual_modal or not hasattr(model.backbone, 'capture_modal_features'):
            raise ValueError('--save_modal_feats requires the dual-modal backbone')
        args.save_feats = True
        model.backbone.capture_modal_features = True

    # default fuse the model 
    logger.info("\tFusing model...")
    model = fuse_model(model)

    if args.fp16:
        model = model.half()
    
    data_root = os.path.join(args.data_root, args.split)
    save_dir = args.save_dir.format(dataset_name=args.dataset_name, split=args.split)

    def _is_femot_seq(name):
        seq_dir = os.path.join(data_root, name)
        if not os.path.isdir(seq_dir):
            return False
        if args.dual_modal:
            return os.path.isdir(os.path.join(seq_dir, f'{name}_aps'))
        if args.img_subdir == 'aps' or args.img_subdir == '':
            return os.path.isdir(os.path.join(seq_dir, f'{name}_aps'))
        if args.img_subdir:
            return os.path.isdir(os.path.join(seq_dir, args.img_subdir))
        return True

    seqs = sorted([s for s in os.listdir(data_root) if _is_femot_seq(s)])

    meta_data = dict()  # meta data, including image h and w

    for seq in seqs:
        logger.info(f'detecting seq {seq}')

        seq_img_dir = os.path.join(data_root, seq)
        evt_img_dir = None
        if args.dual_modal:
            seq_img_dir = os.path.join(seq_img_dir, f'{seq}_aps')
            evt_img_dir = os.path.join(data_root, seq, f'{seq}_dvs')
        elif args.img_subdir == 'aps':
            seq_img_dir = os.path.join(seq_img_dir, f'{seq}_aps')
        elif args.img_subdir:
            seq_img_dir = os.path.join(seq_img_dir, args.img_subdir)

        imgs = sorted([
            f for f in os.listdir(seq_img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])

        det_result_dict = dict()
        feat_result_dict = dict()
        rgb_feat_result_dict = dict()
        evt_feat_result_dict = dict()
        frame_id = 1

        img_ori = cv2.imread(os.path.join(seq_img_dir, imgs[0]))
        meta_data[seq] = [img_ori.shape[0], img_ori.shape[1]]  # h, w, 

        for img_name in tqdm(imgs):
            img_ori = cv2.imread(os.path.join(seq_img_dir, img_name))

            if args.dual_modal:
                evt_ori = cv2.imread(os.path.join(evt_img_dir, img_name))
                assert evt_ori is not None, f'missing event frame {evt_img_dir}/{img_name}'
                img, ratio = dual_preproc(
                    img_ori,
                    evt_ori,
                    args.img_size,
                    rgb_means=(0.485, 0.456, 0.406),
                    rgb_std=(0.229, 0.224, 0.225),
                )
            else:
                img, ratio = preproc(img_ori, args.img_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

            img = torch.from_numpy(img).unsqueeze(0).float().to(device)

            if args.fp16:
                img = img.half()

            with torch.no_grad():
                if args.save_feats:
                    fpn_outs = model.backbone(img)
                    head_out = model.head(fpn_outs)
                    output = postprocess_yolox(head_out, exp.num_classes, 0.05, img, img_ori)
                else:
                    output = model(img)
                    output = postprocess_yolox(output, exp.num_classes, 0.05, img, img_ori)

                if output is not None:
                    # tlbr to tlwh
                    output[:, 2] -= output[:, 0]
                    output[:, 3] -= output[:, 1]
                    det_result_dict[frame_id] = output.cpu().numpy()
                    if args.save_feats:
                        tlwhs = det_result_dict[frame_id][:, :4]
                        feat_map = fpn_outs[0]
                        feat_result_dict[frame_id] = extract_det_feats(feat_map, tlwhs, ratio)
                        if args.save_modal_feats:
                            out_rgb, out_evt = model.backbone.last_modal_features
                            rgb_feat_result_dict[frame_id] = extract_det_feats(
                                out_rgb['dark3'], tlwhs, ratio
                            )
                            evt_feat_result_dict[frame_id] = extract_det_feats(
                                out_evt['dark3'], tlwhs, ratio
                            )
                else:
                    det_result_dict[frame_id] = np.zeros((0, 5))
                    if args.save_feats:
                        feat_result_dict[frame_id] = np.zeros(
                            (0, fpn_outs[0].shape[1]), dtype=np.float32
                        )
                        if args.save_modal_feats:
                            out_rgb, out_evt = model.backbone.last_modal_features
                            rgb_feat_result_dict[frame_id] = np.zeros(
                                (0, out_rgb['dark3'].shape[1]), dtype=np.float32
                            )
                            evt_feat_result_dict[frame_id] = np.zeros(
                                (0, out_evt['dark3'].shape[1]), dtype=np.float32
                            )

            if args.vis and output is not None:
                plot_img(img_ori, frame_id, output, save_dir=os.path.join(save_dir, 'vis_results'))

            frame_id += 1


        # write result
        logger.info(f'write results of seq {seq}')

        save_results(save_dir, seq, det_result_dict)

        if args.save_feats:
            feat_path = os.path.join(save_dir, seq + '_feat.npz')
            np.savez_compressed(feat_path, **{f'f{k}': v for k, v in feat_result_dict.items()})
            logger.info(f'saved appearance feats to {feat_path}')
        if args.save_modal_feats:
            rgb_path = os.path.join(save_dir, seq + '_rgb_feat.npz')
            evt_path = os.path.join(save_dir, seq + '_evt_feat.npz')
            np.savez_compressed(
                rgb_path, **{f'f{k}': v for k, v in rgb_feat_result_dict.items()}
            )
            np.savez_compressed(
                evt_path, **{f'f{k}': v for k, v in evt_feat_result_dict.items()}
            )
            logger.info(f'saved modality features to {rgb_path} and {evt_path}')

    # save meta data
    if args.generate_meta_data:
        save_meta_data(save_dir, meta_data)
                


if __name__ == '__main__':
    args = get_args()
    main(args)


