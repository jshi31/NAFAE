# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 14:46:34 2018

@author: skinshi
"""


import pickle
import numpy as np
import pdb
import cv2
import glob
import os
import sys
sys.path.append(os.path.split(os.path.abspath(__file__))[0])
from seq_nms import seq_nms
np.random.seed(0)

# change the det form into seq_nms input form
class_list = ['bacon', 'bean', 'beef', 'blender', 'bowl', 'bread', 'butter', 'cabbage', 'carrot', 'celery', 'cheese', 'chicken', 'chickpea', 'corn', 'cream', 'cucumber', 'cup', 'dough', 'egg', 'flour', 'garlic', 'ginger', 'it', 'leaf', 'lemon', 'lettuce', 'lid', 'meat', 'milk', 'mixture', 'mushroom', 'mussel', 'mustard', 'noodle', 'oil', 'onion', 'oven', 'pan', 'paper', 'pasta', 'pepper', 'plate', 'pork', 'pot', 'potato', 'powder', 'processor', 'rice', 'salad', 'salt', 'sauce', 'seaweed', 'sesame', 'shrimp', 'soup', 'squid', 'sugar', 'that', 'them', 'they', 'tofu', 'tomato', 'vinegar', 'water', 'whisk', 'wine', 'wok']


def change_form(dets):
    """
    :param dets: (frm_i, box_i, 6) (x1, y1, x2, y2, cls_i, score)
    :output dets_out: (cls_i, frm_i, box_i, 5) (x1, y1, x2, y2, score, box_fi)
    : box_fi: to find the original index of each box.
    """
    # what if frm_i, box_i does not belong to the same length?? 
    class_num = len(class_list)
    Nb = 20
    frm_num = len(dets)
    dets_out = [[[] for j in range(frm_num)] for i in range(class_num)]
    for frm_i in range(frm_num):
        dets_frm = dets[frm_i]
        for box_i in range(Nb):
            box = dets_frm[box_i]
            # judge class
            box_cls = int(box[4])
            dets_out[box_cls][frm_i].append(np.hstack((box[:4], [box[5]], [box_i])))
        for cls_i in range(class_num):
            dets_out[cls_i][frm_i] = np.array(dets_out[cls_i][frm_i])
    return dets_out

def get_show_boxes(dets, tbs):
    """
    :param dets (cls_i, frm_i, box_i, 5) (x1, y1, x2, y2, score)
    :param tbs (cls_i, tb_i, 3) (start_frm_i, tube, total_score)
    :param dets_out (frm_i, box_i, 9) (x1, y1, x2, y2, cls_i, score, tb_len, tb_i, tb_progress)
    """
    frm_num = len(dets[0])
    dets_out = [[] for j in range(frm_num)]
    for cls_i, tb_cls in enumerate(tbs):
        for tb_i, tb_tup in enumerate(tb_cls):
            frm_s, tb, s_sum = tb_tup
            tb_len = len(tb)
            s_ave = s_sum/(tb_len + 1e-6)
            for tb_prog, box_i in enumerate(tb):
                frm_i = frm_s + tb_prog
                det_box = dets[cls_i][frm_i][box_i][:4]
                box = np.hstack((det_box, np.array([cls_i, s_ave, tb_len, tb_i, tb_prog])))
                dets_out[frm_i].append(box)
    return dets_out

def get_show_flow(flow):
    """
    :param flow (h, w, 2)
    """
    # Use Hue, Saturation, Value colour model 
    h, w, _ = flow.shape
    img_shape = (h, w, 3)
    hsv = np.zeros(img_shape, dtype=np.uint8)
    hsv[..., 1] = 128
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    bgr = rgb[:, :, ::-1]
    return bgr
def add_bound(img):
    img[:3, :, 2] = 255
    img[-3:, :, 2] = 255
    img[:, :3, 2] = 255
    img[:, -3:, 2] = 255
    return img

def vis_video(img_names, dets, flos=None, dvsa_path=None):
    """
    :param img_names: list of img file
    :param dets (frm_i, box_i, 9) (x1, y1, x2, y2, cls_i, score, tb_len, tb_i, tb_progress)
    """
    colors = (np.random.rand(1000, 3)*255).astype('int')
    img = cv2.imread(img_names[0])
    h_img, w_img, _ = img.shape
    frm_num = len(dets)
    if flos is not None:
        _ , _, h_flo, w_flo = flos.shape
        flos *= h_img/h_flo
        flos = flos.transpose((0, 2, 3, 1))
        # pad last frame
        flos = np.concatenate((flos, flos[-1][np.newaxis]), 0)
    
    for frm_i, img_name in enumerate(img_names):
        rcp_id, vid_id, seg_i, _ = parse_img_name(img_name)
        img = cv2.imread(img_name)
        h, w, _ = img.shape
        boxes = dets[frm_i]
        #load dvsa image
        if dvsa_path is not None:
            dvsa_img_name = os.path.join(dvsa_path, '{:04d}{:06d}.jpg'.format(seg_i, frm_i//16))
            dvsa_img = cv2.imread(dvsa_img_name)
      
        for box_i, box in enumerate(boxes):
            bbox = box[:4].astype('int')
            cls_i, score, tb_len, tb_i, tb_prog = box[4:].astype('int')
            score = box[5]
            if score > 0.1:
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), [i for i in colors[tb_i]], 1)
                cv2.putText(img, '{} {:.2f} {}'.format(class_list[cls_i], score, tb_prog), (bbox[0], bbox[1] + 10), cv2.FONT_HERSHEY_PLAIN, 1.0, colors[tb_i], thickness=1)
            if frm_i%16 == 7 or frm_i%16 == 8:
                img = add_bound(img)
#        cv2.imshow('video', img)
#        cv2.waitKey (0)

        save_name = img_name.replace('sampled', 'vis')
        save_dir = save_name.rsplit('\\', 1)[0]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if flos is not None:
            flo = flos[frm_i]
            flo = get_show_flow(flo)
            flo = cv2.resize(flo, (h_img, w_img))
            color_flo = (255, 255, 255) if frm_i + 1 != frm_num else (0, 0, 255)
            cv2.putText(flo, 'seg {}'.format(seg_i), (0, 0 + 10), cv2.FONT_HERSHEY_PLAIN, 1.0, color_flo, thickness=1)
            img = np.concatenate((img, flo), 1)
        if dvsa_path is not None:
            img = np.concatenate((dvsa_img, img), 1)
            
        cv2.imwrite(save_name, img)


def div_imglst_by_name(image_list):
    """divide image list by frame name
    :param image_list
    :output img_lists
    """
    image_list.sort()
    code_list = [path.split('\\')[-1].split('.')[0] for path in image_list]
    def get_seg_ind(name):
        return int(name[:4])
    seg_cell = []
    start_i = 0
    id = get_seg_ind(code_list[0])
    for i in range(len(code_list)):
        if i == len(code_list) - 1 or get_seg_ind(code_list[i+1]) != id:
            seg_cell.append(image_list[start_i:i + 1])
            if i < len(code_list) - 1:
                id = get_seg_ind(code_list[i + 1])
                start_i = i + 1
    return seg_cell
def parse_img_name(path):
    """parse image by frame name
    :param name [str]
    :output img_lists
    """
    code = path.split('\\')[-1].split('.')[0]
    vid_id = path.split('\\')[-2]
    rcp_id = path.split('\\')[-3]
    seg_id = int(code[:4])
    frm_id = int(code[4:])
    return rcp_id, vid_id, seg_id, frm_id
    
def write_video(vid_path, size):
    """
    :param vid_path path that store the images for a video
    :param size: (h, w) [tuple] for the video
    """
    vid_path = vid_path.replace('sampled', 'vis')
    vid_id = vid_path.split('\\')[-1]
    save_path = vid_path.replace('frames', 'videos')
    save_path = save_path.rsplit('\\', 1)[0]
    print ('save_path', save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print ('vid name', os.path.join(save_path, '{}.mp4'.format(vid_id)))
    save_name = os.path.join(save_path, '{}.mp4'.format(vid_id))


    img_names = glob.glob(os.path.join(vid_path, '*'))
    for img_i, img_name in enumerate(img_names):
        base, code = img_name.rsplit('\\', 1)
        code, tail = code.split('.')
        code_seq = '{:010d}'.format(img_i)
        img_name_seq = os.path.join(base, '.'.join([code_seq, tail]))
        os.rename(img_name, img_name_seq)
    os.system('ffmpeg -r 20 -i  {} -c:v libx264 -vf fps=20 -pix_fmt yuv420p {}'.format(os.path.join(vid_path, '%010d.jpg'), save_name))
    for img_i, img_name in enumerate(img_names):
        base, code = img_name.rsplit('\\', 1)
        code, tail = code.split('.')
        code_seq = '{:010d}'.format(img_i)
        img_name_seq = os.path.join(base, '.'.join([code_seq, tail]))
        os.rename(img_name_seq, img_name)


    
if __name__ == '__main__':

    root = r'E:\YouCookII\sampled_frames_spl-1\validation'
    dvsa_root = r'E:\vid-cap-ground\output\data\YouCookII\sample_frames_spl0_dvsa\validation'
    vid_paths = glob.glob(os.path.join(root, '*', '*'))
    vid_paths.sort()
    for vid_i, vid_path in enumerate(vid_paths):
        rcp_id, vid_id = vid_path.split('\\')[-2:]
        # load det
        det_dict = pickle.load(open(r'E:\vid-cap-ground\output\det\det_dict2.pkl', 'rb'))
        dets = det_dict[vid_id]
        # load image
        img_names = glob.glob(os.path.join(vid_path, '*'))
        img_names.sort()
        img_names_cells = div_imglst_by_name(img_names)
        # load dvsa
        dvsa_vid_path = os.path.join(dvsa_root, rcp_id, vid_id)

        # load flow
        flo_path = r'E:\vid-cap-ground\output\flow\{}'.format(vid_id)
        flo_names = glob.glob(os.path.join(flo_path, '*'))
        flo_names.sort()

        for seg_i, dets_seg in enumerate(dets): 
            # get seg dets 
            dets_seg = change_form(dets_seg)
            # dets_seg (cls_i, frm_i, box_i, 6) box_i num is not cleaned
            # dets_seg_pure (cls_i, frm_i, box_i, 6) box_i num is cleaned
            # tbs (cls_i, tb_i, 3) (start_frm_i, tube, total_score)
            dets_seg, dets_seg_pure, tbs = seq_nms(dets_seg)
            pdb.set_trace()
            dets_seg = get_show_boxes(dets_seg, tbs)
            # get seg flow flo_seg: (bs-1, 2, 64, 64)
            flos_seg = np.load(flo_names[seg_i])
            vis_video(img_names_cells[seg_i], dets_seg, flos_seg, dvsa_vid_path)
        print('write {} ...'.format(vid_path))
        write_video(vid_path, (224, 224*3))
    


            
