import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import pickle

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.vocab as vocab
from datasets.youcook2 import MPrpDataSet, SubsetSampler, MPrpBatchSampler
from datasets.youcook_eval import parse_gt, evaluate_phrase, evaluate_box
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import save_net, load_net, vis_detections, vis_grounds, vis_box_order, vis_single_det, vis_det
from model.utils.net_utils import save_checkpoint
from model.faster_rcnn.vgg16_rpn import vgg16
from model.transformer.SubLayers import MultiHeadAttention
from model.transformer.Models import position_encoding_general as position_encoding_init
try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

CLASSES = ['bacon', 'bean', 'beef', 'blender', 'bowl', 'bread', 'butter', 'cabbage', 'carrot', 'celery', 'cheese', 'chicken', 'chickpea', 'corn', 'cream', 'cucumber', 'cup', 'dough', 'egg', 'flour', 'garlic', 'ginger', 'it', 'leaf', 'lemon', 'lettuce', 'lid', 'meat', 'milk', 'mixture', 'mushroom', 'mussel', 'mustard', 'noodle', 'oil', 'onion', 'oven', 'pan', 'paper', 'pasta', 'pepper', 'plate', 'pork', 'pot', 'potato', 'powder', 'processor', 'rice', 'salad', 'salt', 'sauce', 'seaweed', 'sesame', 'shrimp', 'soup', 'squid', 'sugar', 'that', 'them', 'they', 'tofu', 'tomato', 'vinegar', 'water', 'whisk', 'wine', 'wok']

EPS = 1e-5

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/vgg16.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='vgg16', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--model', dest='model',
                        help='model_name',
                        default='DVSA', type=str)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models',
                        default="models/vgg16/pretrain")
    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models',
                        default="output/models")
    parser.add_argument('--vid_list_file', dest='vid_list_file',
                        help='directory to load video list file',
                        default="./data/YouCookII/split/dummy_list.txt")
    parser.add_argument('--val_list_file', dest='val_list_file',
                        help='directory to load validation video list file',
                        default="val_list.txt")
    parser.add_argument('--test_list_file', dest='test_list_file',
                        help='directory to load validation video list file',
                        default="test_list.txt")
    parser.add_argument('--word_file', dest='word_file',
                        help='directory to load caption file',
                        default='./data/YouCookII/sampled_entities/train_entities.pkl')
    parser.add_argument('--box_val_anno_file', dest='box_val_anno_file',
                        help='box annotation file name',
                        default='yc2_bb_val_annotations.json')
    parser.add_argument('--box_test_anno_file', dest='box_test_anno_file',
                        help='box annotation file name',
                        default='yc2_bb_test_annotations.json')
    parser.add_argument('--seg_anno_file', dest='seg_anno_file',
                        help='segment annotation file name',
                        default='youcookii_annotations.json')
    parser.add_argument('--root', dest='root',
                        help='root path',
                        default='data')
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset',
                        default='YouCookII')
    parser.add_argument('--class_file', dest='class_file',
                        help='class file name that store class',
                        default='youcook_cls.txt')
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regres Data loadingsion',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=1, type=int)
    parser.add_argument('--checkbatch', dest='checkbatch',
                        help='checkbatch to load network',
                        default=10021, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='training batch_size',
                        default=8, type=int)
    parser.add_argument('--bs_val', dest='batch_size_val',
                        help='validation batch_size',
                        default=1, type=int)
    parser.add_argument('--workers', dest='workers',
                        help='wroker number',
                        default=8, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('--act_trunc', dest='act_trunc',
                        help='truncate action length',
                        default=20, type=int)
    parser.add_argument('--debug', dest='debug',
                        help='debug the code',
                        action='store_true')
    parser.add_argument('--pdb', dest='pdb',
                        help='debug with pdb',
                        action='store_true')
    # data size
    parser.add_argument('--img_h', dest='img_h',
                        help='img height',
                        default=224, type=int)
    parser.add_argument('--img_w', dest='img_w',
                        help='img width',
                        default=224, type=int)

    # training param
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=20, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)
    parser.add_argument('--dropout_rate', dest='dropout_rate',
                        help='dropout probability',
                        default=0.1, type=float)
    parser.add_argument('--clip', dest='clip',
                        help='clip the gradient to prevent overfitting',
                        default=100, type=float)
    parser.add_argument('--weight_decay', dest='weight_decay',
                        help='weight decay for optimizer',
                        default=0.00001, type=float)
    parser.add_argument('--shuffle_train', dest='shuffle_train',
                        help='training shuffle',
                        action='store_true')
    parser.add_argument('--no_shuffle_train', dest='shuffle_train',
                        help='no training shuffle',
                        action='store_false')
    parser.add_argument('--shuffle_val', dest='shuffle_val',
                        help='validation shuffle',
                        default=False, type=bool)

    # vis embedding param
    parser.add_argument('--vis_fc_dim', dest='vis_fc_dim',
                        help='faster rcnn fc layer feature dimension',
                        default=4096, type=int)
    # word embedding param
    parser.add_argument('--glove_dim', dest='glove_dim',
                        help='gloVe dimension',
                        default=200, type=int)
    parser.add_argument('--word_ebd_dim', dest='word_ebd_dim',
                        help='word embedding dimension',
                        default=512, type=int)
    parser.add_argument('--max_ent_len', dest='max_ent_len',
                        help='max number of word in a sentence',
                        default=13, type=int)
    # transformer param
    parser.add_argument('--n_head', dest='n_head',
                        help='number of head',
                        default=8, type=int)
    parser.add_argument('--d_k', dest='d_k',
                        help='dimension of key in each head',
                        default=64, type=int)
    parser.add_argument('--d_v', dest='d_v',
                        help='dimension of value in each head',
                        default=64, type=int)
    parser.add_argument('--n_position', dest='n_position',
                        help='maximum length of a sentence',
                        default=100, type=int)

    # Epoch
    parser.add_argument('--epoch', dest='epoch',
                        help='epoch number for training',
                        default=10, type=int)

    # Loss
    parser.add_argument('--Delta', dest='Delta',
                        help='Delta is for margin in the margin loss',
                        default=1, type=float)
    parser.add_argument('--vis_lam', dest='vis_lam',
                        help='balance visual similarity constraint loss and DVSA loss',
                        default=1, type=float)
    # sample number
    parser.add_argument('--entity_type', dest='entity_type',
                        help='entity type of the word, category, [sentence_raw, sentence, noun, category]',
                        default=['category'], type=list)
    parser.add_argument('--sample_num', dest='sample_num',
                        help='sample number of frame per video segment',
                        default=5, type=int)
    parser.add_argument('--sample_num_val', dest='sample_num_val',
                        help='sample number of frame per video segment or evaluation',
                        default=0, type=int)
    parser.add_argument('--sample_rate', dest='sample_rate',
                        help='sample rate of frame per video segment for training',
                        default=1, type=int)
    parser.add_argument('--sample_rate_val', dest='sample_rate_val',
                        help='sample rate of frame per video segment for evaluation',
                        default=16, type=int)
    parser.add_argument('--fix_seg_len', dest='fix_seg_len',
                        help='if fix the sample number of each video segment for training',
                        action='store_true')
    parser.add_argument('--fix_seg_len_val', dest='fix_seg_len_val',
                        help='if fix the sample number of each video segment for evaluation',
                        action='store_true')
    parser.add_argument('--eval_freq', dest='eval_freq',
                        help='evaluation frequency',
                        default=1, type=int)
    parser.add_argument('--validate', dest='validate',
                        help='just go to validate mode',
                        action='store_true')
    parser.add_argument('--phase', dest='phase',
                        help='phase: train, test, val',
                        choices=['train', 'val', 'test', 'detvis'], type=str)
    parser.add_argument('--resume', dest='resume',
                        help='resume for training',
                        action='store_true')
    parser.add_argument('--iou_thr', dest='iou_thr',
                        help='iou threshold for match gt',
                        default=0.5, type=float)
    parser.add_argument('--ovthr', dest='ovthr',
                        help='overlap threshold for visualized feature',
                        default=0.7, type=float)
    parser.add_argument('--val_vis_freq', dest='val_vis_freq',
                        help='visualization freqency for validation',
                        default=100, type=int)
    parser.add_argument('--train_vis_freq', dest='train_vis_freq',
                        help='visualization freqency for training',
                        default=100, type=int)


    # add file illustration
    parser.add_argument('--statement', dest='statement',
                        help='statement for current training',
                        default='', type=str)

    args = parser.parse_args()
    return args

def v2np(var):
    """variable to numpy
    :param: var: variable(gpu)
    """
    return var.data.cpu().numpy()

def t2np(var):
    """ tensor to numpy
    :param: var: Tensor(gpu)
    """
    return var.cpu().numpy()


def visualize_grounding(D, D_prob, boxes, imgs, word_entities, im_paths, Ns=1, gt_dets=None, gt_classes=None):
    """ Visualize grouding boxes
    This can compile two kind of input (Na*Ns, Ne) for ground entity in each frame,
    and input shape (Na, Ne) for ground entity in each video segement.
    :param: D (Na*Ns, Ne) value scope [0, Na*100) [numpy ndarray]
    :param: D_prob (Na, Ne) [numpy ndarray]
    :param: boxes (Na, 5, 20, 4) [numpy ndarray]
    :param: imgs (Na, 5, 224, 224, 3) [numpy ndarray]
    :param: word_entities (Na, Ne) [list of list of str]
    :param: img_paths [Na*5*20] str
    :param: gt_dets (Na*5, obj_num, 5) [list]
    :param: gt_classes str x (Na*5, obj_num) [list]
    """

    # get image shaep
    h, w = imgs.shape[2:4]
    # get Nas, Ne, where Nas = Na*Ns
    Nas, Ne = D.shape

    # get img_num_per_act, box_num_per_img
    _, img_num_per_act, box_num_per_img, _ = boxes.shape
    # get max length of each act
    action_length = [len(action) for action in word_entities]
    # legends (Na*100,Ne) build legend for each box in each image
    legends = []
    for i in range(Nas*img_num_per_act*box_num_per_img):
        legends.append([''])
    for frm_ind in range(Nas):
        act_ind = frm_ind // img_num_per_act
        for ent_ind in range(action_length[act_ind]):
            # get single legend for each box
            entity = word_entities[act_ind][ent_ind]
            score = D_prob[frm_ind, ent_ind]
            # get the score tilt
            score_tilt = (score - D_prob[:, ent_ind].min()) / (D_prob[:, ent_ind].max() - D_prob[:, ent_ind].min())
            # get the score bar
            score_bar = score_tilt * score
            # legend = '{:2d} {}:{:.2f}'.format(act_ind, entity, score)
            legend = '{}: {:.2f}:{:.2f}:{:.2f}'.format(entity, score, score_tilt, score_bar)
            # get box index
            box_ind = D[frm_ind, ent_ind]
            # append legend to box legend list
            if legends[box_ind][0]:
                legends[box_ind].append(legend)
            else:
                legends[box_ind][0] = legend
    # process data
    boxes = boxes.reshape(-1, box_num_per_img, 4)
    imgs = imgs.reshape(-1, h, w, 3)
    # transfer BGR to RGB
    imgs = vis_grounds(imgs, legends, boxes)

    # draw gt box
    if gt_dets:
        for i in range(len(imgs)):
            imgs[i] = vis_single_det(imgs[i], np.array(gt_dets[i]), gt_classes[i])

    # process the save path
    for im_ind, im2show in enumerate(imgs):
        im_path = im_paths[im_ind]
        head = im_path.split('/')[:-7]
        tail = im_path.split('/')[-7:]
        save_path = head + ['output'] + ['Visualize'] + tail
        save_path = '/'.join(save_path)
        if not os.path.isdir(save_path.rsplit('/', 1)[0]):
            os.makedirs(save_path.rsplit('/', 1)[0])
        cv2.imwrite(save_path, im2show)
        print ('save_path {}'.format(save_path))


def visualize_single_grounding(D, D_prob, boxes, imgs, word_entities, im_paths):
    """ Visualize grouding boxes without other proposals
    This can compile two kind of input (Na*Ns, Ne) for ground entity in each frame,
    and input shape (Na, Ne) for ground entity in each video segement.
    :param: D (Na*Ns, Ne) value scope [0, Na*100) [numpy ndarray]
    :param: D_prob (Na, Ne) [numpy ndarray]
    :param: boxes (Na, 5, 20, 4) [numpy ndarray]
    :param: imgs (Na, 5, 224, 224, 3) [numpy ndarray]
    :param: word_entities (Na, Ne) [list of list of str]
    :param: img_paths [Na*5*20] str
    :param: gt_dets (Na*5, obj_num, 5) [list]
    :param: gt_classes str x (Na*5, obj_num) [list]
    """
    # get image shaep
    h, w = imgs.shape[2:4]
    # get Nas, Ne, where Nas = Na*Ns
    Nas, Ne = D.shape

    # get img_num_per_act, box_num_per_img
    _, img_num_per_act, box_num_per_img, _ = boxes.shape
    # get max length of each act
    action_length = [len(action) for action in word_entities]
    # legends (Na*100,Ne) build legend for each box in each image
    legends = []
    for i in range(Nas*img_num_per_act*box_num_per_img):
        legends.append([''])
    for frm_ind in range(Nas):
        act_ind = frm_ind // img_num_per_act
        for ent_ind in range(action_length[act_ind]):
            # get single legend for each box
            entity = word_entities[act_ind][ent_ind]
            score = D_prob[frm_ind, ent_ind]
            # get the score tilt
            score_tilt = (score - D_prob[:, ent_ind].min()) / (D_prob[:, ent_ind].max() - D_prob[:, ent_ind].min())
            # get the score bar
            score_bar = score_tilt * score
            # legend = '{:2d} {}:{:.2f}'.format(act_ind, entity, score)
            # legend = '{}: {:.2f}:{:.2f}:{:.2f}'.format(entity, score, score_tilt, score_bar)
            legend = '{}'.format(entity)
            # get box index
            box_ind = D[frm_ind, ent_ind]
            # append legend to box legend list
            if legends[box_ind][0]:
                legends[box_ind].append(legend)
            else:
                legends[box_ind][0] = legend
    # process data
    boxes = boxes.reshape(-1, box_num_per_img, 4)
    imgs = imgs.reshape(-1, h, w, 3)

    # divide the above code into frame-level
    out_imgs = []
    for img_i, img in enumerate(imgs):
        legends_img = legends[img_i*box_num_per_img:(img_i + 1)*box_num_per_img]
        legends_valid_img = [legend for i, legend in enumerate(legends_img) if legend[0]]
        boxes_img = boxes[img_i]
        boxes_valid_img = np.array([boxes_img[i] for i, legend in enumerate(legends_img) if legend[0]])
        clses_img = [CLASSES.index(legend[0]) for legend in legends_valid_img]
        legends_valid_img = [','.join(legend) for i, legend in enumerate(legends_valid_img)]
        # plot
        img = vis_det(img, boxes_valid_img, clses_img, legends_valid_img, 1)
        out_imgs.append(img)


    # process the save path
    for im_ind, im2show in enumerate(out_imgs):
        im_path = im_paths[im_ind]
        head = im_path.split('/')[:-7]
        tail = im_path.split('/')[-7:]
        save_path = head + ['output'] + ['Visualize_frm'] + tail
        save_path = '/'.join(save_path)
        if not os.path.isdir(save_path.rsplit('/', 1)[0]):
            os.makedirs(save_path.rsplit('/', 1)[0])
        cv2.imwrite(save_path, im2show)
        print ('save_path {}'.format(save_path))

## Word Related Functions
def get_word(glove, word):
    return glove.vectors[glove.stoi[word]]

def stepRCNN(im_data, im_info, gt_boxes, num_boxes, ground_model):
    """
    If the batch size is too large, split the batch into steps with smaller batch size
    :return:
    """
    Ns = im_data.shape[0]
    # divide Ns into steps, each step with size 160
    step_size = 64
    step = int(Ns/step_size)
    rem = Ns - step*step_size
    splits = [step_size for i in range(step)]
    if rem > 0:
        splits.append(rem)
    rois_lst = []
    roi_feats_lst = []
    fc_feats_lst = []
    for i, stp in enumerate(splits):
        s, e = i*step_size, i*step_size + stp
        rois, roi_scores, roi_feats, fc_feats = ground_model.fasterRCNN(im_data[s:e], im_info[s:e], gt_boxes, num_boxes)
        rois_lst.append(rois)
        roi_feats_lst.append(roi_feats)
        fc_feats_lst.append(fc_feats)
    rois_lst = torch.cat(rois_lst, 0)
    roi_feats_lst = torch.cat(roi_feats_lst, 0)
    fc_feats_lst = torch.cat(fc_feats_lst, 0)
    return rois_lst, roi_feats_lst, fc_feats_lst


def postprocess(D, D_sim, Na, Ns, Nb, Ne):
    """
    :param D: grounding result (Na*Ns, Na*Ne)
    :param D_sim: grounding similarity (Na*Ns, Na*Ne) value scope [0, Nb)
    :return: D (Na, Ns, Ne)
    :return D_sim (Na, Ns, Ne)
    """
    D_temp = D.reshape(Na, Ns, Na, Ne)
    D_sim_temp = D_sim.reshape(Na, Ns, Na, Ne)

    D = np.zeros((Na, Ns, Ne), dtype=int)
    D_sim = np.zeros((Na, Ns, Ne))
    for act_ind in range(Na):
        for spl_ind in range(Ns):
            for ent_ind in range(Ne):
                D[act_ind, spl_ind, ent_ind] = D_temp[act_ind, spl_ind, act_ind, ent_ind] + act_ind*Ns*Nb + spl_ind*Nb
                D_sim[act_ind, spl_ind, ent_ind] = D_sim_temp[act_ind, spl_ind, act_ind, ent_ind]
    return D, D_sim


def record_det(img_inds, obj_labels, obj_bboxes, obj_confs, Nb, vid_entities, D, D_sim, img_ids, infer_boxes):
    Na, Ns, Ne = D.shape
    for act_ind, entities in enumerate(vid_entities):
        for spl_ind in range(Ns):
            for ent_ind, entity in enumerate(entities):
                box_id_offset = D[act_ind][spl_ind][ent_ind]
                img_id_offset = box_id_offset//Nb
                img_inds.append(img_ids[img_id_offset])
                obj_labels.append(entity)
                obj_bboxes.append(infer_boxes[box_id_offset])
                obj_confs.append(D_sim[act_ind][spl_ind][ent_ind])


class DVSA(torch.nn.Module):
    def __init__(self, args, cfg):
        super(DVSA, self).__init__()
        self.args = args
        self.cfg = cfg
        self.slf_attn = MultiHeadAttention(args.n_head, args.word_ebd_dim, args.d_k,
                                           args.d_v, dropout=args.dropout_rate)
        self.position_enc = nn.Embedding(args.n_position, args.word_ebd_dim)
        self.position_enc.weight.data = position_encoding_init(args.n_position, args.word_ebd_dim)
        self.ffn = nn.Linear(2*args.word_ebd_dim, args.sample_num)
        self.phase = ''

    def zero2one(self, x):
        """ if the input is 0, then output 1
        """
        if x == 0:
            x = 1
        return x

    def init_train(self):
        self.Na = self.args.batch_size
        self.phase = 'train'

    def init_eval(self):
        self.Na = self.args.batch_size_val
        self.phase = 'eval'

    def forward(self, vis_feats, word_feats, entities_length):
        """ Process EM part in video level
        :param: vis_feats (Nax100, 512)
        :param: word_feats (NaxNe, 512)
        """
        # Na: action number (segment number)
        Na = self.Na
        # Nb: number for box in each frame
        Nb = cfg.TEST.RPN_POST_NMS_TOP_N
        # Ne: maximum number of entity in an action
        # Ne = max(entities_length)
        Ne = self.args.max_ent_len
        # Ns: sample number for each segment
        Ns = int(vis_feats.size()[0]/Na/Nb)
        # division vector: (Na,)
        div_vec = torch.tensor([self.zero2one(i) for i in entities_length], dtype=torch.float).to(device)
        # S_mask: (NaxNsxNb, Na, Ne)
        S_mask = np.zeros((Na*Ns*Nb, Na, Ne))
        for act_ind, entity_length in enumerate(entities_length):
            S_mask[:, act_ind, entity_length:] = 1
        S_mask = torch.tensor(S_mask, dtype=torch.uint8).to(device)
        S_mask = S_mask.view(Na*Ns*Nb, -1)
        # S_mask_vis: (Na, Ne, Ns, Ns)
        S_mask_vis = torch.zeros(Na, Ne, Ns, Ns, dtype=torch.uint8).to(device)
        for act_i, ent_len in enumerate(entities_length):
            S_mask_vis[act_i, ent_len:] = 1
            for i in range(Ns):
                if ent_len == 0: continue
                S_mask_vis[act_i, :ent_len, i, i] = 1

        # S_: (NaxNsxNb, NaxNe)
        S_ = vis_feats @ (word_feats.permute((1, 0)))

        # S_mask: (NaxNsxNb, Na*Ne)
        S_.masked_fill_(S_mask, 0)

        if self.phase == 'train':
            # S_vis: for visual similarity (Na, Ns, Nb, Ne)
            S_vis_temp = S_.view(Na, Ns, Nb, Na, Ne)
            with torch.no_grad():
                S_vis = torch.zeros(Na, Ns, Nb, Ne).to(device)
                for i in range(Na):
                    S_vis[i] = S_vis_temp[i, :, :, i, :]
                sim_scr, maxind = S_vis.max(2)
                # setup indarr for index selection
                indarr = torch.zeros(Na*Ns*Ne, dtype=torch.long).to(device)
                for i in range(Na):
                    for j in range(Ns):
                        for k in range(Ne):
                            indarr[i*Ns*Ne + j*Ne + k] = maxind[i, j, k]
                sim_scr = (sim_scr - sim_scr.min(1, True)[0])/(sim_scr.max(1, True)[0] - sim_scr.min(1, True)[0] + EPS)
                sim_scr = sim_scr.view(Na, Ns, Ne, -1)
            vis_feats_cls = torch.index_select(vis_feats, 0, indarr).view(Na, Ns, Ne, -1)
            vis_feats_cls = vis_feats_cls/(torch.norm(vis_feats_cls, 2, 3, True) + EPS)
            vis_feats_cls = vis_feats_cls * sim_scr
            vis_feats_cls1 = vis_feats_cls.permute(0, 2, 1, 3).contiguous().view(Na*Ne, Ns, -1)
            vis_feats_cls2 = vis_feats_cls.permute(0, 2, 3, 1).contiguous().view(Na*Ne, -1, Ns)
            vis_feats_cls = 1 - torch.bmm(vis_feats_cls1, vis_feats_cls2).view(Na, Ne, Ns, Ns)
            vis_feats_cls.masked_fill_(S_mask_vis, 0)
            dem = vis_feats_cls.nonzero().shape[0]
            vis_loss = vis_feats_cls.sum()/dem

        # S: (NaxNs, Nb, NaxNe)
        S = S_.view(Na*Ns, Nb, Na*Ne)
        # S_att: (NaxNs, Nb, NaxNe)
        # S: (NaxNs, NaxNe)
        S, _ = S.max(1)
        # s_att (Na, Ns, NaxNe)
        S = S.view(Na, Ns, Na*Ne)
        # with torch.no_grad():
        S_att = (S - S.min(1, True)[0]) / (S.max(1, True)[0] - S.min(1, True)[0] + EPS)
        S = S*S_att
        # average on entities in each action
        Sf = S.view(Na, Ns, Na, Ne).sum(-1)
        # Sf: (Na, Ns, Na)
        Sf = Sf/div_vec
        # extract diagnal vectors
        Sf_diag = torch.zeros(Na, Ns, requires_grad=True).to(device)
        Sf_diag = Sf_diag.clone()
        for i in range(Na):
            Sf_diag[i] = Sf[i, :, i]
        # Sf_diag: (Na, Ns, 1)
        Sf_diag = Sf_diag.unsqueeze(2)

        # frame_score: (Na, Ns)
        # frame_score = Sf.sum(2) + Sf.permute((2, 1, 0)).sum(2)
        frame_score = F.relu(Sf - Sf_diag.permute(2, 1, 0) + self.args.Delta).mean(0).permute(1, 0) + F.relu(Sf - Sf_diag + self.args.Delta).mean(2)

        # rank loss
        margin_loss = (frame_score.mean() + self.args.vis_lam*vis_loss)*10 if self.phase == 'train' else frame_score.mean()*10

        # grounding and only ground the current segment
        # S_sim: (NaxNs, Nb, NaxNe)
        S_sim = S_.view(Na*Ns, -1, Na*Ne)
        # D_sim, D_ind: (Na*Ns, Na*Ne) value scope [0, Nb)
        D_sim, D_ind = S_sim.max(1)

        return D_ind, D_sim, margin_loss

class VisEbd(torch.nn.Module):
    """visual feature embedding module
    """
    def __init__(self, args):
        super(VisEbd, self).__init__()
        self.fc1 = nn.Linear(args.vis_fc_dim, args.word_ebd_dim)
        self.drop = nn.Dropout(p=args.dropout_rate)

    def forward(self, feats):
        vis_feats = feats/100
        vis_feats = self.fc1(vis_feats)
        vis_feats = self.drop(vis_feats)
        vis_feats = torch.tanh(vis_feats)
        return vis_feats

class WordEbd(torch.nn.Module):
    """word feature embedding module
    """
    def __init__(self, args):
        super(WordEbd, self).__init__()
        self.fc1 = nn.Linear(args.glove_dim, args.word_ebd_dim)
        self.drop = nn.Dropout(p=args.dropout_rate)
        self.bn = nn.BatchNorm1d(args.word_ebd_dim)

    def forward(self, feats):
        word_feats = torch.tanh(self.drop(self.bn(self.fc1(feats))))
        return word_feats

class GroundModel(torch.nn.Module):
    def __init__(self, args, cfg):
        super(GroundModel, self).__init__()
        gnome_classes = np.array(['' for _ in range(2501)])
        # initilize fasterRCNN
        self.fasterRCNN = vgg16(gnome_classes, pretrained=False, class_agnostic=args.class_agnostic)
        self.fasterRCNN.create_architecture()
        self.fasterRCNN.eval()
        # init vis embedding layer
        self.vis_ebd = VisEbd(args)
        # init word embedding layer
        self.word_ebd = WordEbd(args)
        # initialize DVSA
        self.DVSA = DVSA(args, cfg)

def save_feat(fc_feats, boxes, file_name):
    """

    :param fc_feats: (num_boxes, 4096)
    :param boxes: (num_boxes, 4)
    :return:
    """
    fc_feats = fc_feats.cpu().numpy()
    boxes = boxes.cpu().numpy()
    pdb.set_trace()
    np.save(fc_feats, file_name)



def train(train_loader, ground_model, glove, criterion, optimizer, epoch, args):
    # output directory
    output_dir = os.path.join(args.save_dir, args.net, args.dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # initilize the tensor holder here.
    im_data = torch.tensor(1, dtype=torch.float).to(device)
    im_info = torch.tensor(1, dtype=torch.float).to(device)
    num_boxes = torch.tensor(1, dtype=torch.float).to(device)
    gt_boxes = torch.tensor(1, dtype=torch.float).to(device)
    ground_model.train()
    ground_model.DVSA.init_train()
    ground_model.fasterRCNN.eval()

    # init loss
    running_loss = 0

    # init times
    model_time = 0
    batch_prev = time.time()
    RCNN_time = 0
    batch_time = 0

    for batch_ind, (im_blobs, entities, entities_length, frm_length, rl_seg_inds, seg_nums, im_paths, img_ids) in enumerate(train_loader):
        if max(entities_length) == 0:
            continue
        """Process visual feature"""
        # default im_scales 1 and shape (5,)
        im_scales = [1]*len(im_blobs)
        # im_info_np (h, w, scale)
        im_info_np = np.array([[im_blob.shape[0], im_blob.shape[1], im_scales[0]] for im_blob in im_blobs], dtype=np.float32)
        im_data_pt = torch.from_numpy(im_blobs.copy())
        # -> (batch, channel, h, w)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        # put im_data to variable
        im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
        # put im_info to variable
        im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.data.resize_(1, 1, 5).zero_()
        num_boxes.data.resize_(1).zero_()

        det_tic = time.time()
        # rois (batch, num_boxes, 5)  roi_feats(batch*num_boxes, 512, 7, 7) fc_feats(batch*num_boxes, 4096)
        with torch.no_grad():
            rois, roi_scores, roi_feats, fc_feats = ground_model.fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
        det_toc = time.time()

        # feats (batch*num_boxes, 512)
        vis_feats = ground_model.vis_ebd(fc_feats)
        # (batch, num_boxes, 4)
        boxes = rois.data[:, :, 1:5]

        # save feature
        save_feat(fc_feats, boxes, im_paths)

        pred_boxes = boxes

        # only use RPN, only use rois
        pred_boxes /= im_scales[0]

        """Process visual feature End"""

        # Ne = max(entities_length)
        Ne = args.max_ent_len
        Na = len(entities_length)
        Ns = args.sample_num
        Nb = cfg.TEST.RPN_POST_NMS_TOP_N

        """Process word feature for each action"""
        # glove_feats (Na, Ne, 200)
        glove_feats = torch.zeros(Na, Ne, args.glove_dim)
        glove_feats = glove_feats.clone()
        # entity point
        ent_p = 0
        for act_ind, entity_length in enumerate(entities_length):
            for ent_ind in range(entity_length):
                entity = entities[ent_p]
                if not entity:
                    continue
                elif entity in glove.stoi.keys():
                    glove_feats[act_ind, ent_ind] = get_word(glove, entity)
                else:
                    glove_feats[act_ind, ent_ind] = torch.zeros(args.glove_dim)
                    raise Exception('{} is not in glove vocabulary'.format(entity))
                ent_p += 1
        # glove_feats (NaxNe, 200)
        glove_feats = glove_feats.view(-1, args.glove_dim)
        glove_feats = glove_feats.to(device)
        # word_feats (NaxNe, 512)
        word_feats = ground_model.word_ebd(glove_feats)
        """Process word feature End"""

        # store visual feature for one video vis_boxes, vis_feats
        vis_boxes = t2np(pred_boxes)
        vid_im_paths = im_paths
        # change series entities to structured entities
        vid_entities = [[] for i in range(len(entities_length))]
        for i, l in enumerate(entities_length):
            vid_entities[i] = [entities.pop(0) for _ in range(l)]

        vid_ims = im_blobs

        optimizer.zero_grad()

        model_tic = time.time()
        # get visual grounding loss
        # Df_sim (Na*Ns, Na*Ne)
        # Df (Na*Ns, Na*Ne) value scope [0, Nb)
        D, D_sim, margin_loss = ground_model.DVSA(vis_feats, word_feats, entities_length)

        # use L1 loss to minimize the margin loss
        loss = criterion(margin_loss, torch.zeros_like(margin_loss))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ground_model.parameters(), args.clip)
        optimizer.step()
        model_toc = time.time()
        batch_time += (model_toc - batch_prev)
        batch_prev = model_toc
        RCNN_time += (det_toc - det_tic)
        model_time += (model_toc - model_tic)
        running_loss += loss.item()
        print('epoch {:2d}/{:2d} batch {:4d}/{:4d} loss {:.3f} RCNN_time {:.3f} model_time {:.3f} batch_time {:.3f}'
              .format(epoch, args.epoch, batch_ind + 1, len(train_loader), running_loss/(batch_ind + 1), RCNN_time/(batch_ind + 1), model_time/(batch_ind + 1), batch_time/(batch_ind + 1)))

        if int((batch_ind + 1)%args.train_vis_freq) == 0:
            # visualize groundings
            vid_ims = vid_ims.reshape(-1, args.sample_num, args.img_h, args.img_w, 3) + 255/2
            vis_boxes = vis_boxes.reshape(-1, args.sample_num, cfg.TEST.RPN_POST_NMS_TOP_N, 4)
            D, D_sim = v2np(D), v2np(D_sim)
            D, D_sim = postprocess(D, D_sim, Na, Ns, Nb, Ne)
            # D_sim (Na*Ns, Ne) D for frame
            # D (Na*Ns, Ne) value scope [0, Nb)
            D_sim = D_sim.reshape(-1, Ne)
            D = D.reshape(-1, Ne)
            visualize_grounding(D, D_sim, vis_boxes, vid_ims, vid_entities, vid_im_paths)

    writer.add_scalar('loss', running_loss/(batch_ind + 1), epoch)



def validate(val_loader, ground_model, glove, criterion, epoch, args):

    if not args.fix_seg_len_val:
        assert args.sample_num_val == 0, 'sample number should be 0 if fix the video segment length in evaluation phase'
        assert args.batch_size_val == 1, 'batch_size should be 1 if fix the video segment length in evaluation phase'

    # load gt
    gt_cache_file = os.path.join('cache', args.dataset, 'gtbox_{}_sample_{}.pkl'.format(args.phase, args.sample_num_val))
    if not os.path.exists(gt_cache_file):
        if not os.path.isdir('/'.join(gt_cache_file.split('/')[:-1])):
            os.makedirs('/'.join(gt_cache_file.split('/')[:-1]))
        print('parse gt label ...')
        list_file = args.val_list_file if args.phase != 'test' else args.test_list_file
        box_anno_file = args.box_val_anno_file if args.phase != 'test' else args.box_test_anno_file
        recs = parse_gt(args.root, args.dataset, box_anno_file, args.seg_anno_file, list_file,
                        args.class_file, (args.img_h, args.img_w), args.sample_num_val, args.iou_thr, args.phase)
    else:
        print('load gt label ...')
        with open(gt_cache_file, 'rb') as f:
            recs = pickle.load(f)

    # load classes list
    class_list_path = os.path.join(args.root, args.dataset, 'annotations', args.class_file)
    class_list = []
    with open(class_list_path) as f:
        for line in f:
            class_list.append(line.rstrip('\n'))

    # initilize the tensor holder here.
    im_data = torch.tensor(1, dtype=torch.float).to(device)
    im_info = torch.tensor(1, dtype=torch.float).to(device)
    num_boxes = torch.tensor(1, dtype=torch.float).to(device)
    gt_boxes = torch.tensor(1, dtype=torch.float).to(device)
    ground_model.eval()
    ground_model.DVSA.init_eval()

    # init loss
    running_loss = 0

    # init times
    batch_prev = time.time()
    model_time = 0
    RCNN_time = 0
    batch_time = 0

    # init dets holder
    img_inds, obj_labels, obj_bboxes, obj_confs = [], [], [], []

    for batch_ind, (im_blobs, entities, entities_length, frm_length, rl_seg_inds, seg_nums, im_paths, img_ids) in enumerate(val_loader):
        if max(entities_length) == 0:
            continue
        if len(im_blobs) > 800:
            im_blobs = im_blobs[:800]
            im_paths = im_paths[:800]
            img_ids = img_ids[:800]
        """Process visual feature"""
        # default im_scales 1 and shape (5,)
        im_scales = [1]*len(im_blobs)
        # im_info_np (h, w, scale)
        im_info_np = np.array([[im_blob.shape[0], im_blob.shape[1], im_scales[0]] for im_blob in im_blobs], dtype=np.float32)
        im_data_pt = torch.from_numpy(im_blobs.copy())
        # -> (batch, channel, h, w)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        # put im_data to variable
        im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
        # put im_info to variable
        im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.data.resize_(1, 1, 5).zero_()
        num_boxes.data.resize_(1).zero_()

        det_tic = time.time()

        # rois (batch, num_boxes, 5)  roi_feats(batch*num_boxes, 512, 7, 7) fc_feats(batch*num_boxes, 4096)
        with torch.no_grad():
            rois, roi_feats, fc_feats = stepRCNN(im_data, im_info, gt_boxes, num_boxes, ground_model)
        det_toc = time.time()
        """ use roi feature
        # get the mean of the roi_feats 
        feats = torch.mean(roi_feats.view(args.batch_size*cfg.TEST.RPN_POST_NMS_TOP_N, 512, -1), -1)
        # feats (batch*num_boxes, 512)
        feats = feats.view(args.batch_size*cfg.TEST.RPN_POST_NMS_TOP_N, -1)
        """
        # use fc feat, and need a visual embedding layer
        # feats (batch*num_boxes, 512)
        vis_feats = ground_model.vis_ebd(fc_feats)

        # (batch, num_boxes, 4)
        boxes = rois.data[:, :, 1:5]

        pred_boxes = boxes

        # only use RPN, only use rois
        pred_boxes /= im_scales[0]

        """Process visual feature End"""
        # Ne = max(entities_length)
        Ne = args.max_ent_len
        Na = len(entities_length)
        Nb = cfg.TEST.RPN_POST_NMS_TOP_N
        Ns = args.sample_num_val if args.fix_seg_len_val else int(vis_feats.shape[0]/Na/Nb)
        """Process word feature for each action"""
        # glove_feats (Na, Ne, 200)
        glove_feats = torch.zeros(Na, Ne, args.glove_dim)
        glove_feats = glove_feats.clone()
        # entity point
        ent_p = 0

        for act_ind, entity_length in enumerate(entities_length):
            for ent_ind in range(entity_length):
                entity = entities[ent_p]
                if not entity:
                    continue
                elif entity in glove.stoi.keys():
                    glove_feats[act_ind, ent_ind] = get_word(glove, entity)
                else:
                    glove_feats[act_ind, ent_ind] = torch.zeros(args.glove_dim)
                    raise Exception('{} is not in glove vocabulary'.format(entity))
                ent_p += 1
        # glove_feats (NaxNe, 200)
        glove_feats = glove_feats.view(-1, args.glove_dim)
        glove_feats = glove_feats.to(device)
        # word_feats (NaxNe, 512)
        word_feats = ground_model.word_ebd(glove_feats)
        """Process word feature End"""

        # store visual feature for one video vis_boxes, vis_feats
        vis_boxes = (t2np(pred_boxes))
        vid_im_paths = im_paths
        # change series entities to structured entities
        vid_entities = [[] for i in range(len(entities_length))]
        ent_lis = entities_length.copy()
        for i, l in enumerate(entities_length):
            vid_entities[i] = [entities.pop(0) for _ in range(l)]

        vid_ims = im_blobs
        model_tic = time.time()

        # get visual grounding loss
        # D_sim (Na*Ns, Na*Ne)
        # D (Na*Ns, Na*Ne) value scope [0, Nb)
        D, D_sim, margin_loss = ground_model.DVSA(vis_feats, word_feats, entities_length)
        D, D_sim = v2np(D), v2np(D_sim)
        D, D_sim = postprocess(D, D_sim, Na, Ns, Nb, Ne)

        # use L1 loss to minimize the margin loss
        loss = criterion(margin_loss, torch.zeros_like(margin_loss))

        model_toc = time.time()
        batch_time += (model_toc - batch_prev)
        batch_prev = model_toc
        # print statistics
        model_time += (model_toc - model_tic)
        RCNN_time += (det_toc - det_tic)
        running_loss += loss.item()

        print('epoch {:2d}/{:2d} batch {:4d}/{:4d} RCNN time {:.3f} model time {:.3f} batch time {:.3f}'
              .format(epoch, args.epoch, batch_ind + 1, len(val_loader), RCNN_time/(batch_ind + 1), model_time/(batch_ind + 1), batch_time/(batch_ind + 1)))
        # det: record detection result, [img_inds, obj_labels, obj_bboxes, obj_confs].
        infer_boxes = np.array(vis_boxes).reshape(-1, 4)
        record_det(img_inds, obj_labels, obj_bboxes, obj_confs, Nb, vid_entities, D, D_sim, img_ids, infer_boxes)

        if int((batch_ind + 1)%args.val_vis_freq) == 0:
            # visualize groundings
            vid_ims = vid_ims.reshape(-1, Ns, args.img_h, args.img_w, 3) + 255/2
            vis_boxes = vis_boxes.reshape(-1, Ns, cfg.TEST.RPN_POST_NMS_TOP_N, 4)
            D_sim = D_sim.reshape(-1, Ne)
            D = D.reshape(-1, Ne)
            visualize_single_grounding(D, D_sim, vis_boxes, vid_ims, vid_entities, vid_im_paths)

    # generate det
    dets = [img_inds, obj_labels, obj_bboxes, obj_confs]

    # write to output result for evaluation
    # output grounding result for evaluation directory
    eval_dir = os.path.join('output', 'result')
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    eval_path = os.path.join(eval_dir, 'ground_res_{}_{}_{}_{}.pkl'.format(args.phase, args.checksession, epoch, args.checkbatch))
    with open(eval_path, 'wb') as f:
        pickle.dump(dets, f)

    accuracy = evaluate_box(recs, dets, class_list)
    print('{} accuracy: {:0.2%}'.format(args.phase, accuracy))

    # add loss to summary
    writer.add_scalar('loss/val_loss', running_loss/(batch_ind + 1), epoch)
    writer.add_scalar('accuracy/val_accu', accuracy, epoch)

    # return the validation accuracy
    return accuracy


def main():
    global writer, device
    args = parse_args()

    print('Called with args:')
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # init summary writer
    summary_path = os.path.join('runs', 'sess_{}_{}'.format(args.checksession, args.statement))
    writer = SummaryWriter(summary_path)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = True if torch.cuda.is_available() else False

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # Process word embedding dict
    glove = vocab.GloVe(name='6B', dim=args.glove_dim)
    print('load {} word'.format(len(glove.itos)))

    # output directory
    output_dir = os.path.join(args.save_dir, args.net, args.dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.

    input_dir = args.load_dir
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)

    # initilize the ground here.
    ground_model = GroundModel(args, cfg)

    start_epoch = 0

    # load ground model
    if args.resume:
        load_name = os.path.join(output_dir, 'vis_ground_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkbatch))
        print("resume checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        start_epoch = checkpoint['epoch'] + 1
        ground_model.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
    elif args.phase == 'val' or args.phase == 'test':
        load_name = os.path.join(output_dir, 'vis_ground_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkbatch))
        print("load checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        start_epoch = checkpoint['epoch']
        ground_model.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
    else:
        load_name = os.path.join(input_dir, 'faster_rcnn_gnome.pth')
        print("load checkpoint %s" % (load_name))
        if torch.cuda.is_available() and args.cuda:
            checkpoint = torch.load(load_name)
        else:
            checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
        ground_model.fasterRCNN.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')

    ground_model.to(device)

    # print attribute of word_ebd
    # if args.debug:
    # 	word_ebd.weight.register_hook(print_grad)

    # init the loss layer
    criterion = nn.L1Loss()
    # init optimizer
    param_list = [
        {'params': ground_model.DVSA.parameters()},
        {'params': ground_model.word_ebd.parameters()},
        {'params': ground_model.vis_ebd.parameters()}
    ]
    optimizer = torch.optim.Adam(param_list, lr=args.lr, weight_decay=args.weight_decay)

    def adjust_learning_rate(optimizer, epoch, drop_rate, step):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = args.lr * (drop_rate ** (epoch // step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # dataloader
    train_dataset = MPrpDataSet(args.root, args.dataset, args.seg_anno_file, args.box_val_anno_file, 'train', args.fix_seg_len, args)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle_train, num_workers=args.workers, drop_last=True,
                                               collate_fn=train_dataset.collate_fn)
    val_dataset = MPrpDataSet(args.root, args.dataset, args.seg_anno_file, args.box_val_anno_file, 'val', args.fix_seg_len_val, args)
    sampler = SubsetSampler(val_dataset.provide_batch_spl_ind(args.batch_size_val, shuffle=args.shuffle_val))
    batch_sampler = MPrpBatchSampler(sampler, args.batch_size_val, drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=args.workers, collate_fn=val_dataset.collate_fn,
                                             batch_sampler=batch_sampler, pin_memory=True)
    test_dataset = MPrpDataSet(args.root, args.dataset, args.seg_anno_file, args.box_test_anno_file, 'test', args.fix_seg_len_val, args)
    sampler = SubsetSampler(test_dataset.provide_batch_spl_ind(args.batch_size_val, shuffle=args.shuffle_val))
    batch_sampler = MPrpBatchSampler(sampler, args.batch_size_val, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=args.workers, collate_fn=test_dataset.collate_fn,
                                              batch_sampler=batch_sampler, pin_memory=True)
    best_accuracy = 0
    for epoch in range(start_epoch, args.epoch):
        if args.phase == 'train':
            # train for one epoch
            train(train_loader, ground_model, glove, criterion, optimizer, epoch, args)
            # adjust learning rate
            adjust_learning_rate(optimizer, epoch, args.lr_decay_gamma, args.lr_decay_step)
            # evaluate on validation set
            if (epoch + 1) % args.eval_freq == 0 or epoch == args.epoch - 1:
                accuracy = validate(val_loader, ground_model, glove, criterion, epoch, args)
                # save model
                is_best = accuracy > best_accuracy
                best_accuracy = max(accuracy, best_accuracy)
                if is_best:
                    save_name = os.path.join(output_dir, 'vis_ground_{}_{}_{}.pth'.format(args.checksession, epoch, args.checkbatch))
                    save_model = ground_model.state_dict()
                    save_checkpoint({
                        'session': args.checksession,
                        'epoch': epoch,
                        'model': save_model,
                        'optimizer': optimizer.state_dict(),
                        'pooling_mode': cfg.POOLING_MODE,
                    }, save_name)
        elif args.phase == 'val':
            accuracy = validate(val_loader, ground_model, glove, criterion, epoch, args)
            break
        elif args.phase == 'test':
            accuracy = validate(test_loader, ground_model, glove, criterion, epoch, args)
            break

    writer.close()

    # write best accuracy to file pythonfile.best
    python_file = __file__.split('/')[-1].split('.')[0]
    if not os.path.exists('.optm'):
        os.makedirs('.optm')
    with open('.optm/{}.best'.format(python_file), 'w') as f:
        f.write('{}'.format(best_accuracy))


if __name__ == '__main__':
    main()
