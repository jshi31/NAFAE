# ----------------------
# eval script for youcook2
# writen by Jing Shi
# ----------------------
from __future__ import division
import numpy as np
import os
import json
import pickle
from tqdm import tqdm 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pdb

def parse_gt(root, dataset, gt_file, anno_file, vid_list_file, class_file, img_size, sample_num, thr, phase):
    """parse the box annotation
    :param: dataset: dataset Name 
    :param gt_file: file name for store the box gt json file
    :param anno_file: file name for store the segment time json file
    :param vid_list_file: video list file name
    :param class_path: file name for store the class label, stored in ./data/YouCookII/annotations/, each row is the name of a class
    :param img_size: tuple (h, w) to specify the image size.
    :param sample_num: sample number [int], 5: sample 5 frames per segment; 0: sample all frames per segment
    :param thr: iou thresh greater than which the box and gt are matched
    :output recs: record list of dict, each dict for a box, dict['label'], ['bbox'], ['thr'], ['img_idx']
    """
    # load bbox ground truth
    gt_path = os.path.join(root, dataset, 'annotations', gt_file)
    with open(gt_path) as f:
        gt = json.load(f)
    # load anno segment time grond truth
    anno_path = os.path.join(root, dataset, 'annotations', anno_file)
    with open(anno_path) as f:
        anno = json.load(f)

    # get vid_ids:
    vid_list_path = os.path.join(root, dataset, 'split', vid_list_file)
    vid_ids = []
    with open(vid_list_path) as f:
        for line in f:
            vid_ids.append(line.rstrip('\n').split('/')[-1])
    # get class label list
    class_list_path = os.path.join(root, dataset, 'annotations', class_file)
    class_list = []
    with open(class_list_path) as f:
        for line in f:
            class_list.append(line.rstrip('\n'))

    # Parse the box annotation
    print('start parsing box annotation...')
    # iterate over each vid 
    img_count = 0
    # recs: store the all dict of labels. 
    recs = []
    print('')
    for vid_ind, vid_id in enumerate(tqdm(vid_ids)):
        h = gt['database'][vid_id]['height']
        w = gt['database'][vid_id]['width']
        eta_y = img_size[0]/h
        eta_x = img_size[1]/w
        # segs: list of tuple [(start, end)]
        segs = [anno['database'][vid_id]['annotations'][i]['segment'] for i in range(len(anno['database'][vid_id]['annotations']))]
        # seg_list: labeled seg
        seg_list = [int(i) for i in gt['database'][vid_id]['annotations'].keys()]
        seg_list.sort()

        for seg_ind, seg in enumerate(segs):
            # if the curretn segment is labeled
            if seg_ind in seg_list:
                obj_list = [int(i) for i in gt['database'][vid_id]['annotations'][str(seg_ind)].keys()]
                obj_list.sort()
                # frame_length: number of frame in the video segment
                frame_length = seg[1] - seg[0]
                # get the sampled frame index (if we are evaluation, we do not need to sample, all frames are needed)
                # two senarios (now I only have the sampled data, so we need the sampled)
                # 1. iterate over sampled frames
                # 2. iterate over all frames
                frame_list = range(frame_length) if sample_num == 0 else np.round(np.linspace(0, frame_length-1, sample_num)).astype('int')
                # iterate over frame
                for frame_ind in frame_list:
                    rec = {}
                    labels = []
                    bboxes = []
                    thrs = []
                    img_ids = []
                    # iterate over each box, if the frame id is labeled
                    for obj_ind in range(len(gt['database'][vid_id]['annotations'][str(seg_ind)].keys())):
                        # judge if the box should append to annotations
                        if (gt['database'][vid_id]['annotations'][str(seg_ind)][str(obj_ind)]['boxes'][str(frame_ind)]['outside'] == 0 and
                        gt['database'][vid_id]['annotations'][str(seg_ind)][str(obj_ind)]['boxes'][str(frame_ind)]['occluded'] == 0):
                            label = gt['database'][vid_id]['annotations'][str(seg_ind)][str(obj_ind)]['label']
                            # lemmatize the label
                            lemmatizer = WordNetLemmatizer()
                            label = str(lemmatizer.lemmatize(label, pos=wordnet.NOUN))
                            bbox = [gt['database'][vid_id]['annotations'][str(seg_ind)][str(obj_ind)]['boxes'][str(frame_ind)]['{}'.format(axis)] for axis in ['xtl', 'ytl', 'xbr', 'ybr']]
                            bbox[0], bbox[2] = bbox[0]*eta_x, bbox[2]*eta_x
                            bbox[1], bbox[3] = bbox[1]*eta_y, bbox[3]*eta_y
                            bbox = [int(i) for i in bbox]
                            labels.append(label)
                            bboxes.append(bbox)
                            thrs.append(thr)
                            img_ids.append(img_count)
                    rec['label'] = labels
                    rec['bbox'] = bboxes
                    rec['thr'] = thrs
                    rec['img_ids'] = img_ids
                    img_count += 1
                    recs.append(rec)
            # if the segment is not labelled, padded with emtpy recs, but each frame should have one.
            else:
                # frame_length: number of frame in the video segment
                frame_length = seg[1] - seg[0]
                # get the sampled frame index (if we are evaluation, we do not need to sample, all frames are needed)
                # two senarios (now I only have the sampled data, so we need the sampled)
                # 1. iterate over sampled frames
                # 2. iterate over all frames
                frame_list = range(frame_length) if sample_num == 0 else np.round(np.linspace(0, frame_length-1, sample_num)).astype('int')
                # iterate over frame
                for frame_ind in frame_list:
                    rec = {}
                    rec['label'] = []
                    rec['bbox'] = []
                    rec['thr'] = []
                    rec['img_ids'] = []
                    recs.append(rec)
                    img_count += 1
    # write recs to pkl cache file
    cache_path = os.path.join('cache', dataset, 'gtbox_{}_sample_{}.pkl'.format(phase, sample_num))
    with open(cache_path, 'wb') as f:
        pickle.dump(recs, f)
    return recs


def parse_gt_RW(root, anno_file, vid_list_file, thr, phase):
    """
    image shape (224x224), image batch_size=5
    :param: root data
    :param: vid_list_file: list of videos
    :param: anno_file: pkl of word dictionary
    """
    data_path = os.path.join(root, 'RoboWatch')
    # load images
    vids_path = os.path.join(data_path, 'sampled_frames_splnum-1')
    vid_list_file = os.path.join(data_path, 'split', '{}_list.txt'.format(phase))
    vid_ids = []
    vid_names = []
    with open(vid_list_file) as f:
        for line in f:
            vid_path = line.rstrip('\n')
            vid_name = os.path.join(vids_path, vid_path)
            # get video id
            video_id = vid_path.split('/')[-1]
            vid_names.append(vid_name)
            vid_ids.append(video_id)


    # load annotations
    anno_path = os.path.join(data_path, 'annotations', anno_file)

    # load words
    print ('anno_file', anno_path)
    with open(anno_path, 'rb') as f:
        anno_dict = pickle.load(f)

    # load frames with batch size 1
    img_id = 0
    # records of all image
    recs = []
    for vid_i, vid_id in enumerate(vid_ids):
        anno_dict_vid = anno_dict[vid_id]
        # devide to segements:
        frm_segs = anno_dict_vid['frm']
        ent_segs = anno_dict_vid['entity']
        box_dict = anno_dict_vid['box']
        #
        for seg_i in range(len(frm_segs)):
            frm_seg = frm_segs[seg_i]
            ent_seg = ent_segs[seg_i]
            # get entities
            entities = ent_seg['noun']
            # get img_ids
            img_ids = []
            # get img_paths 
            img_paths = [os.path.join(vid_names[vid_i], frm_id + '.jpg') for frm_id in frm_seg]
            valid_img_paths = []
            # get frames 
            imgs = []
            for i, img_path in enumerate(img_paths):
                #  read image
                if not os.path.exists(img_path):
                    img_path = img_path.replace('sampled_frames_splnum-1', 'vis_frames')
                if not os.path.exists(img_path):
                    print ('\nWarning: skip frame {}\n'.format(img_path))
                    continue
                # get img paths 
                valid_img_paths.append(img_path)
                # get boxes
                frm_id = img_path.split('/')[-1].split('.')[0]
                bbox = box_dict[frm_id]['box']
                label = box_dict[frm_id]['label']
                thred = [thr for i in range(len(bbox))]
                img_ids = [img_id for i in range(len(bbox))]
                # update img_id
                img_id += 1
                recs.append({'label': label, 'bbox': bbox, 'thr': thred, 'img_ids': img_ids})

    # write recs to pkl cache file
    cache_dir = os.path.join('cache', 'RoboWatch')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_path = os.path.join(cache_dir, 'gtbox_{}_sample.pkl'.format(phase))

    with open(cache_path, 'wb') as f:
        pickle.dump(recs, f)
    return recs


def phrase_accuracy(recs, dets, class_list):
    """ evaluate the detection result by phrase
    assume the frame number in recs are the same with the frame number in dets
    param: dets: detection result, list of four lists [[img_ids], [obj_labels], [obj_bboxes], [obj_confs]]
    param: recs: record list of gt dict, each dict for a box, dict['label'], ['bbox'], ['thr'], ['img_idx']
    param: class_list: list of all class
    """

    # accuracy calculation method: for each phrase, it will ground the box in the current segmentation.
    img_ids = np.array(dets[0])
    obj_labels = np.array(dets[1])
    obj_bboxes = np.array(dets[2])
    obj_confs = np.array(dets[3])
    # sort by img_ids
    order = np.argsort(img_ids)
    img_ids = img_ids[order]
    obj_labels = obj_labels[order]
    obj_bboxes = obj_bboxes[order]
    obj_confs = obj_confs[order]

    gt_img_id = recs[-1]['img_ids']
    # store labels, confs, bboxes w.r.t image cell
    obj_confs = obj_confs[order]
    num_imgs = np.max(img_ids) + 1
    obj_labels_cell = [None] * num_imgs
    obj_confs_cell = [None] * num_imgs
    obj_bboxes_cell = [None] * num_imgs
    start_i = 0
    id = img_ids[0]
    for i in range(0, len(img_ids)):
        if i == len(img_ids) - 1 or img_ids[i + 1] != id:
            conf = obj_confs[start_i:i + 1]
            label = obj_labels[start_i:i + 1]
            bbox = obj_bboxes[start_i:i + 1, :]
            sorted_inds = np.argsort(-conf)
            obj_labels_cell[id] = label[sorted_inds]
            obj_confs_cell[id] = conf[sorted_inds]
            obj_bboxes_cell[id] = bbox[sorted_inds, :]
            if i < len(img_ids) - 1:
                id = img_ids[i + 1]
                start_i = i + 1

    # calculation accuracy w.r.t box. If the ground of the phrase class is correct, then it is matched.
    match_num = 0
    # construct incremental class label
    # class match list: list of the matched number in each class
    class_match_count = np.zeros(len(class_list), dtype=int)
    # class_count: count how many is counted
    class_count =  np.zeros(len(class_list), dtype=int)

    for img_id in range(num_imgs):
        rec = recs[img_id]
        # first loop det
        obj_dict = {}
        if obj_bboxes_cell[img_id] is None:
            continue
        for obj_label, obj_conf, obj_bbox in zip(obj_labels_cell[img_id], obj_confs_cell[img_id], obj_bboxes_cell[img_id]):
            # second loop gt
            for gt_label, gt_bbox, thr, img_idx in zip(rec['label'], rec['bbox'], rec['thr'], rec['img_ids']):
                # calculate IoU
                if obj_label != gt_label:
                    continue
                # now obj_label == gt_label, check if it is the first time to add obj_label to obj_dict
                if not obj_label in obj_dict.keys():
                    obj_dict[obj_label] = 0
                    class_ind = class_list.index(gt_label)
                    class_count[class_ind] += 1
                elif obj_dict[obj_label] == 1:
                    # if the object has been matched, continue
                    continue 

                bi = [
                np.max((obj_bbox[0], gt_bbox[0])),
                np.max((obj_bbox[1], gt_bbox[1])),
                np.min((obj_bbox[2], gt_bbox[2])),
                np.min((obj_bbox[3], gt_bbox[3]))
                ]
                iw = bi[2] - bi[0] + 1
                ih = bi[3] - bi[1] + 1
                if iw > 0 and ih > 0:
                    # compute overlap as area of intersection / area of union
                    ua = (obj_bbox[2] - obj_bbox[0] + 1.) * (obj_bbox[3] - obj_bbox[1] + 1.) + \
                        (gt_bbox[2] - gt_bbox[0] + 1.) * \
                        (gt_bbox[3] - gt_bbox[1] + 1.) - iw*ih
                    ov = iw * ih / ua
                    # makes sure that this object is detected according
                    # to its individual threshold
                    if ov >= thr:
                        match_num += 1
                        class_match_count[class_ind] += 1
                        # to prevent one phrase matched multiple gt.
                        obj_dict[obj_label] = 1
    # class
    class_accuracy = class_match_count/(class_count + 1e-6)
    cls_mean_accuracy = np.mean(class_accuracy)
    mean_accuracy = np.sum(class_match_count)/np.sum(class_count)
    """ # uncomment to see per category accuracy
    for p in zip(class_list, class_accuracy):
        print ('{:10s}{:0.2%}'.format(p[0]+':', p[1]))
    """
    print ('macro query accuracy: {:0.2%}'.format(cls_mean_accuracy))
    print ('micro query accuracy: {:0.2%}'.format(mean_accuracy))
    return cls_mean_accuracy



def box_accuracy(recs, dets, class_list):
    """ evaluate the detection result by phrase
    assume the frame number in recs are the same with the frame number in dets
    param: dets: detection result, list of four lists [[img_ids], [obj_labels], [obj_bboxes], [obj_confs]]
    param: recs: record list of gt dict, each dict for a box, dict['label'], ['bbox'], ['thr'], ['img_idx']
    param: class_list: list of all class
    """

    # accuracy calculation method: for each phrase, it will ground the box in the current segmentation.
    img_ids = np.array(dets[0])
    obj_labels = np.array(dets[1])
    obj_bboxes = np.array(dets[2])
    obj_confs = np.array(dets[3])
    # sort by img_ids
    order = np.argsort(img_ids)
    img_ids = img_ids[order]
    obj_labels = obj_labels[order]
    obj_bboxes = obj_bboxes[order]
    obj_confs = obj_confs[order]

    gt_img_id = recs[-1]['img_ids']
    # store labels, confs, bboxes w.r.t image cell
    obj_confs = obj_confs[order]
    num_imgs = np.max(img_ids) + 1
    obj_labels_cell = [None] * num_imgs
    obj_confs_cell = [None] * num_imgs
    obj_bboxes_cell = [None] * num_imgs
    start_i = 0
    id = img_ids[0]
    for i in range(0, len(img_ids)):
        if i == len(img_ids) - 1 or img_ids[i + 1] != id:
            conf = obj_confs[start_i:i + 1]
            label = obj_labels[start_i:i + 1]
            bbox = obj_bboxes[start_i:i + 1, :]
            sorted_inds = np.argsort(-conf)
            obj_labels_cell[id] = label[sorted_inds]
            obj_confs_cell[id] = conf[sorted_inds]
            obj_bboxes_cell[id] = bbox[sorted_inds, :]
            if i < len(img_ids) - 1:
                id = img_ids[i + 1]
                start_i = i + 1

    # calculation accuracy w.r.t box. If the ground of the phrase class is correct, then it is matched.
    match_num = 0
    # construct incremental class label
    # class match list: list of the matched number in each class
    class_match_count = np.zeros(len(class_list), dtype=int)
    # class_count: count how many is counted
    class_count = np.zeros(len(class_list), dtype=int)

    for img_id in range(num_imgs):
        rec = recs[img_id]
        # first loop gt
        for gt_label, gt_bbox, thr, img_idx in zip(rec['label'], rec['bbox'], rec['thr'], rec['img_ids']):
            class_ind = class_list.index(gt_label)
            class_count[class_ind] += 1
            if obj_bboxes_cell[img_id] is None:
                continue
            # second loop dets
            for obj_label, obj_conf, obj_bbox in zip(obj_labels_cell[img_id], obj_confs_cell[img_id], obj_bboxes_cell[img_id]):
                # calculate IoU
                if obj_label != gt_label:
                    continue
                bi = [
                np.max((obj_bbox[0], gt_bbox[0])),
                np.max((obj_bbox[1], gt_bbox[1])),
                np.min((obj_bbox[2], gt_bbox[2])),
                np.min((obj_bbox[3], gt_bbox[3]))
                ]
                iw = bi[2] - bi[0] + 1
                ih = bi[3] - bi[1] + 1
                if iw > 0 and ih > 0:
                    # compute overlap as area of intersection / area of union
                    ua = (obj_bbox[2] - obj_bbox[0] + 1.) * (obj_bbox[3] - obj_bbox[1] + 1.) + \
                        (gt_bbox[2] - gt_bbox[0] + 1.) * \
                        (gt_bbox[3] - gt_bbox[1] + 1.) - iw*ih
                    ov = iw * ih / ua
                    # makes sure that this object is detected according
                    # to its individual threshold
                    if ov >= thr:
                        match_num += 1
                        class_match_count[class_ind] += 1
                        # to prevent one phrase matched multiple gt.
                        break
                        # allow several phrase match one box
    # class
    class_accuracy = class_match_count/(class_count + 1e-6)
    cls_mean_accuracy = np.mean(class_accuracy)
    mean_accuracy = np.sum(class_match_count)/np.sum(class_count)
    """ # uncomment to see the per class accuracy
    for p in zip(class_list, class_accuracy):
        print ('{:10s}{:0.2%}'.format(p[0]+':', p[1]))
    """
    print ('macro box accuracy: {:0.2%}'.format(cls_mean_accuracy))
    print ('micro box accuracy: {:0.2%}'.format(mean_accuracy))
    return cls_mean_accuracy


def evaluate_phrase(recs, dets, class_list):
    """ evaluate the detection result
    assume the frame number in recs are the same with the frame number in dets
    param: dets: detection result, list of four lists [[img_ids], [obj_labels], [obj_bboxes], [obj_confs]]
    param: recs: record list of gt dict, each dict for a box, dict['label'], ['bbox'], ['thr'], ['img_idx']
    param: class_list: list of all class
    """

    # accuracy calculation method: for each phrase, it will ground the box in the current segmentation.
    img_ids = np.array(dets[0])
    obj_labels = np.array(dets[1])
    obj_bboxes = np.array(dets[2])
    obj_confs = np.array(dets[3])
    # sort by img_ids
    order = np.argsort(img_ids)
    img_ids = img_ids[order]
    obj_labels = obj_labels[order]
    obj_bboxes = obj_bboxes[order]
    obj_confs = obj_confs[order]

    gt_img_id = recs[-1]['img_ids']

    # calculation accuracy w.r.t phrase. If the ground of the phrase class is correct, then it is matched.
    obj_num = len(obj_labels)
    match_num = 0
    # construct incremental class label
    # class match list: list of the matched number in each class
    class_match_count = np.zeros(len(class_list), dtype=int)
    # class_count: count how many is counted
    class_count = np.zeros(len(class_list), dtype=int)
    for img_id, obj_label, obj_conf, obj_bbox in zip(img_ids, obj_labels, obj_confs, obj_bboxes):
        class_ind = class_list.index(obj_label)
        class_count[class_ind] += 1
        rec = recs[img_id]
        for gt_label, gt_bbox, thr, img_idx in zip(rec['label'], rec['bbox'], rec['thr'], rec['img_ids']):
            # calculate IoU
            if obj_label != gt_label:
                continue
            bi = [
            np.max((obj_bbox[0], gt_bbox[0])),
            np.max((obj_bbox[1], gt_bbox[1])),
            np.min((obj_bbox[2], gt_bbox[2])),
            np.min((obj_bbox[3], gt_bbox[3]))
            ]
            iw = bi[2] - bi[0] + 1
            ih = bi[3] - bi[1] + 1
            if iw > 0 and ih > 0:
                # compute overlap as area of intersection / area of union
                ua = (obj_bbox[2] - obj_bbox[0] + 1.) * (obj_bbox[3] - obj_bbox[1] + 1.) + \
                    (gt_bbox[2] - gt_bbox[0] + 1.) * \
                    (gt_bbox[3] - gt_bbox[1] + 1.) - iw*ih
                ov = iw * ih / ua
                # makes sure that this object is detected according
                # to its individual threshold
                if ov >= thr:
                    match_num += 1
                    class_match_count[class_ind] += 1
                    # to prevent one phrase matched multiple gt.
                    break
    # accuracy: class agnostic accuracy, deprecated
    accuracy = match_num / obj_num
    # class
    class_accuracy = class_match_count/class_count
    mean_accuracy = np.mean(class_accuracy)
    for p in zip(class_list, class_accuracy):
        print ('{:10s}{:0.2%}'.format(p[0]+':', p[1]))
    print ('mean phrase accuracy: {:0.2%}'.format(mean_accuracy))
    return mean_accuracy

def evaluate_box(recs, dets, class_list):
    # first do qeury-level accuracy
    cls_mean_accuracy = phrase_accuracy(recs, dets, class_list)
    # then do box-level accuracy
    cls_mean_accuracy = box_accuracy(recs, dets, class_list)
    return cls_mean_accuracy


def evaluate_box_RW(recs, dets):
    """ evaluate the detection result
    assume the frame number in recs are the same with the frame number in dets
    param: dets: detection result, list of four lists [[img_ids], [obj_labels], [obj_bboxes], [obj_confs]]
    param: recs: record list of gt dict, each dict for a box, dict['label'], ['bbox'], ['thr'], ['img_idx']
    param: class_list: list of all class
    """

    # accuracy calculation method: for each phrase, it will ground the box in the current segmentation.
    img_ids = np.array(dets[0])
    obj_labels = np.array(dets[1])
    obj_bboxes = np.array(dets[2])
    obj_confs = np.array(dets[3])
    # sort by img_ids
    order = np.argsort(img_ids)
    img_ids = img_ids[order]
    obj_labels = obj_labels[order]
    obj_bboxes = obj_bboxes[order]
    obj_confs = obj_confs[order]

    gt_img_id = recs[-1]['img_ids']
    # store labels, confs, bboxes w.r.t image cell
    obj_confs = obj_confs[order]
    num_imgs = np.max(img_ids) + 1
    obj_labels_cell = [None] * num_imgs
    obj_confs_cell = [None] * num_imgs
    obj_bboxes_cell = [None] * num_imgs
    start_i = 0
    id = img_ids[0]
    for i in range(0, len(img_ids)):
        if i == len(img_ids) - 1 or img_ids[i + 1] != id:
            conf = obj_confs[start_i:i + 1]
            label = obj_labels[start_i:i + 1]
            bbox = obj_bboxes[start_i:i + 1, :]
            sorted_inds = np.argsort(-conf)
            obj_labels_cell[id] = label[sorted_inds]
            obj_confs_cell[id] = conf[sorted_inds]
            obj_bboxes_cell[id] = bbox[sorted_inds, :]
            if i < len(img_ids) - 1:
                id = img_ids[i + 1]
                start_i = i + 1

    # calculation accuracy w.r.t box. If the ground of the phrase class is correct, then it is matched.
    match_num = 0
    # total count 
    total_num = 0

    for img_id in range(num_imgs):
        rec = recs[img_id]
        for gt_label, gt_bbox, thr, img_idx in zip(rec['label'], rec['bbox'], rec['thr'], rec['img_ids']):
            if not gt_label:
                continue
            total_num += 1
            if obj_bboxes_cell[img_id] is None:
                continue
            for obj_label, obj_conf, obj_bbox in zip(obj_labels_cell[img_id], obj_confs_cell[img_id], obj_bboxes_cell[img_id]):
                # calculate IoU
                if obj_label != gt_label:
                    continue
                bi = [
                np.max((obj_bbox[0], gt_bbox[0])),
                np.max((obj_bbox[1], gt_bbox[1])),
                np.min((obj_bbox[2], gt_bbox[2])),
                np.min((obj_bbox[3], gt_bbox[3]))
                ]
                iw = bi[2] - bi[0] + 1
                ih = bi[3] - bi[1] + 1
                if iw > 0 and ih > 0:
                    # compute overlap as area of intersection / area of union
                    ua = (obj_bbox[2] - obj_bbox[0] + 1.) * (obj_bbox[3] - obj_bbox[1] + 1.) + \
                        (gt_bbox[2] - gt_bbox[0] + 1.) * \
                        (gt_bbox[3] - gt_bbox[1] + 1.) - iw*ih
                    ov = iw * ih / ua
                    # makes sure that this object is detected according
                    # to its individual threshold
                    if ov >= thr:
                        match_num += 1
                        # to prevent one phrase matched multiple gt.
                        break
    # class
    mean_accuracy = match_num / total_num
    print ('mean box accuracy: {:0.2%}'.format(mean_accuracy))
    return mean_accuracy

def evaluate_phrase_ub(recs, dets, class_list):
    """ evaluate the detection result
    assume the frame number in recs are the same with the frame number in dets
    param: dets: detection result, list of four lists [img_inds_all, seg_inds_all, seg_labels_all, obj_bboxes_all]
        img_inds_all = [] each image has one
        seg_inds_all = [] each image has one 
        seg_labels_all = [] each segment has one, adding all words
        obj_bboxes_all = [] each image has one
    param: recs: record list of gt dict, each dict for a box, dict['label'], ['bbox'], ['thr'], ['img_idx']
    Idea: gt and obj are processed frame by frame in just on for iteration !!!
    [important] if gt in each segment is matched with others, then the gt number will not add up.
    [Assumpution] this code can only deal with the situation where sample number is fixed.
    """

    # accuracy calculation method: for each phrase, it will ground the box in the current segmentation.
    img_inds_all = np.array(dets[0])
    seg_inds_all = np.array(dets[1])
    seg_labels_all = dets[2]
    obj_bboxes_all = np.array(dets[3])

    gt_img_id = recs[-1]['img_ids']

    # calculation accuracy w.r.t phrase. If the ground of the phrase class is correct, then it is matched.
    match_num = 0
    # construct incremental class label
    # class match list: list of the matched number in each class
    class_match_count = np.zeros(len(class_list), dtype=int)
    # class_count: count how many is counted
    class_count =  np.zeros(len(class_list), dtype=int)

    seg_ind_old = -1

    for img_ind in range(len(recs)):
        seg_ind = seg_inds_all[img_ind]
        seg_labels = seg_labels_all[seg_ind]
        img_bboxes = obj_bboxes_all[img_ind]
        rec = recs[img_ind]
        gt_bboxes = np.array(rec['bbox'])
        gt_labels = np.array(rec['label'])
        thrs = np.array(rec['thr'])
        assert img_ind == img_inds_all[img_ind], 'img id unmatched between det and gt'
        # build seg_labels mask to prevent duplicating computation, process each for a segment
        if seg_ind != seg_ind_old:
            seg_label_mask = {e:0 for e in seg_labels}
            for label in seg_labels:
                class_ind = class_list.index(label)
                class_count[class_ind] += 1
        # update seg_ind_old
        seg_ind_old = seg_ind
        # 1. find the index of gt boxes
        for label in seg_labels:
            class_ind = class_list.index(label)
            if seg_label_mask[label]:
                continue
            if label in gt_labels:
                gt_inds = np.where(label == gt_labels)[0]
                # 2. find the iou of all boxes with each of gt boxes
                for i, gt_bbox in enumerate(gt_bboxes[gt_inds]):
                    if seg_label_mask[label]:
                        continue
                    thr = thrs[gt_inds][i]
                    for obj_bbox in img_bboxes:
                        if seg_label_mask[label]:
                            continue
                        bi = [
                        np.max((obj_bbox[0], gt_bbox[0])),
                        np.max((obj_bbox[1], gt_bbox[1])),
                        np.min((obj_bbox[2], gt_bbox[2])),
                        np.min((obj_bbox[3], gt_bbox[3]))
                        ]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            # compute overlap as area of intersection / area of union
                            ua = (obj_bbox[2] - obj_bbox[0] + 1.) * (obj_bbox[3] - obj_bbox[1] + 1.) + \
                                (gt_bbox[2] - gt_bbox[0] + 1.) * \
                                (gt_bbox[3] - gt_bbox[1] + 1.) - iw*ih
                            ov = iw * ih / ua
                            # makes sure that this object is detected according
                            # to its individual threshold
                            if ov >= thr:
                            # if ov >= -1:
                                class_match_count[class_ind] += 1
                                # to prevent one phrase matched multiple gt.
                                seg_label_mask[label] = 1
                                break


    # class
    class_accuracy = class_match_count/class_count
    mean_accuracy = np.mean(class_accuracy)
    for p in zip(class_list, class_accuracy):
        print ('{:10s}{:0.2%}'.format(p[0]+':', p[1]))
    print ('mean accuracy: {:0.2%}'.format(mean_accuracy))
    return mean_accuracy

def evaluate_box_ub(recs, dets, class_list):
    """ evaluate the detection result
    assume the frame number in recs are the same with the frame number in dets
    param: dets: detection result, list of four lists [img_inds_all, seg_inds_all, seg_labels_all, obj_bboxes_all]
        img_inds_all = [] each image has one
        seg_inds_all = [] each image has one
        seg_labels_all = [] each segment has one, adding all words
        obj_bboxes_all = [] each image has one
    param: recs: record list of gt dict, each dict for a box, dict['label'], ['bbox'], ['thr'], ['img_idx']
    Idea: gt and obj are processed frame by frame in just on for iteration !!!
    [important] if gt in each segment is matched with others, then the gt number will not add up.
    [Assumpution] this code can only deal with the situation where sample number is fixed.
    """


    # accuracy calculation method: for each phrase, it will ground the box in the current segmentation.
    img_ids = np.array(dets[0])
    seg_inds = np.array(dets[1])
    seg_labels = dets[2]
    obj_bboxes = np.array(dets[3])
    # sort by img_ids
    order = np.argsort(img_ids)
    img_ids = img_ids[order]
    seg_inds = seg_inds[order]
    obj_bboxes = obj_bboxes[order]

    pdb.set_trace()

    gt_img_id = recs[-1]['img_ids']
    # store labels, bboxes w.r.t image cell
    num_imgs = max(len(recs), img_ids.max()) + 1
    obj_bboxes_cell = [None] * num_imgs
    seg_inds_cell = [0] * num_imgs
    start_i = 0
    id = img_ids[0]
    for i in range(0, len(img_ids)):
        if i == len(img_ids) - 1 or img_ids[i + 1] != id:
            bbox = obj_bboxes[start_i:i + 1, :]
            obj_bboxes_cell[id] = bbox
            seg_inds_cell[id] = seg_inds[start_i]
            if i < len(img_ids) - 1:
                id = img_ids[i + 1]
                start_i = i + 1
    # calculation accuracy w.r.t. box. If the ground of the phrase class is correct, then it is matched.
    match_num = 0
    # construct incremental class label
    # class match list: list of the matched number in each class
    class_match_count = np.zeros(len(class_list), dtype=int)
    # class_count: count how many is counted
    class_count = np.zeros(len(class_list), dtype=int)

    # Initialize matched dets. Does not directly serve the accuracy, just save for use in case.
    dets_matched = [None]*num_imgs

    for img_id in range(num_imgs):
        if obj_bboxes_cell[img_id] is None:
            continue
        rec = recs[img_id]
        seg_label = seg_labels[seg_inds_cell[img_id]]
        frm_dets_matched = []
        for gt_label, gt_bbox, thr, img_idx in zip(rec['label'], rec['bbox'], rec['thr'], rec['img_ids']):
            assert img_id == img_idx, 'img id unmatched between det and gt'
            class_ind = class_list.index(gt_label)
            class_count[class_ind] += 1
            for obj_bbox in obj_bboxes_cell[img_id][0]:
                if gt_label not in seg_label:
                    continue
                # calculate IoU
                # pdb.set_trace()
                bi = [
                np.max((obj_bbox[0], gt_bbox[0])),
                np.max((obj_bbox[1], gt_bbox[1])),
                np.min((obj_bbox[2], gt_bbox[2])),
                np.min((obj_bbox[3], gt_bbox[3]))
                ]
                iw = bi[2] - bi[0] + 1
                ih = bi[3] - bi[1] + 1
                if iw > 0 and ih > 0:
                    # compute overlap as area of intersection / area of union
                    ua = (obj_bbox[2] - obj_bbox[0] + 1.) * (obj_bbox[3] - obj_bbox[1] + 1.) + \
                        (gt_bbox[2] - gt_bbox[0] + 1.) * \
                        (gt_bbox[3] - gt_bbox[1] + 1.) - iw*ih
                    ov = iw * ih / ua
                    # makes sure that this object is detected according
                    # to its individual threshold
                    if ov >= thr:
                        match_num += 1
                        class_match_count[class_ind] += 1
                        # process matched detection
                        frm_dets_matched.append([ii for ii in obj_bbox] + [class_ind])
                        # to prevent one phrase matched multiple gt.
                        break
        dets_matched[img_id] = frm_dets_matched

    # class
    class_accuracy = class_match_count/(class_count + 1e-6)
    mean_accuracy = np.mean(class_accuracy)
    for p in zip(class_list, class_accuracy):
        print ('{:10s}{:0.2%}'.format(p[0]+':', p[1]))
    print ('mean box accuracy: {:0.2%}'.format(mean_accuracy))
    return mean_accuracy, dets_matched

def write_recs(recs, path):
    with open(path, 'wb') as f:
        pickle.dump(recs, f)

# class file name: youcook_cls.txt
if __name__ == '__main__':
    root = 'data'
    dataset = 'YouCookII'
    gt_file = 'yc2_bb_val_annotations.json'
    anno_file = 'youcookii_annotations_trainval.json'
    vid_list_file = 'val_list.txt'
    class_file = 'youcook_cls.txt'
    img_size = (224, 224)
    sample_num = 0
    thr = 0.5
    phase = 'val'
    # recs = parse_gt(root, dataset, gt_file, anno_file, vid_list_file, class_file, img_size, sample_num, thr)
    # anno_file = 'ent_anno.pkl'
    # recs = parse_gt_RW(root, anno_file, vid_list_file, thr, phase)
    
    # load classes list
    class_list_path = os.path.join(root, dataset, 'annotations', class_file)
    class_list = []
    with open(class_list_path) as f:
        for line in f:
            class_list.append(line.rstrip('\n'))
    # evaluation
    det_dir = 'output/result'
    det_path = os.path.join(det_dir, 'ground_res_train_256_8_41.pkl')
    with open(det_path, 'rb') as f:
        dets = pickle.load(f)
    rec_path = os.path.join('cache', dataset, 'gtbox_{}_sample_{}.pkl'.format(phase, sample_num))
    with open(rec_path, 'rb') as f:
        recs = pickle.load(f)
    # accuracy, dets_matched = evaluate_box_ub(recs, dets, class_list)
    # write_recs(dets_matched, os.path.join('output', 'result', 'ground_res_ub.pkl'))
    accuracy = evaluate_box(recs, dets, class_list)
    print('accuracy: {:0.2%}'.format(accuracy))
    pdb.set_trace()


    # evaluation gt
    gt_det_path = os.path.join(det_dir, 'ground_res_allbox_test_132_1_1539.pkl')
    with open(det_path, 'rb') as f:
        dets = pickle.load(f)

    accuracy, dets_matched = evaluate_box_ub(recs, dets, class_list)



