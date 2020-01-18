import os
import sys
import pdb
import time
import glob
import argparse

import cv2
import pickle
import numpy as np
import torch.utils.data as data
from torch._six import int_classes as _int_classes

pwd = os.getcwd()
sys.path.append(os.path.join(pwd, 'lib'))

from model.utils.blob import im_list_to_blob



class MPrpDataSet(data.Dataset):
    def __init__(self, root, dataset, anno_file, box_anno_file, phase, fix_seg_len, args):
        """ Multi-proposal dataset, in order to accelerate if a video has too many proposals.
        Data structure: video->segment
        Requirements:
        1. training and evaluation data are sampled by frame and stored 
        2. frame number for each segment can be set as variable or fixed with sampling rate
        3. each return is a proposal, with a segment indicator to tell the index of the proposal in the current video.
        4. each frame has format has name with 10 bits, left 4 bits encode segment id, right 6 bits encode frame_id
        :param: root: project root directory
        :param: dataset: dataset Name 
        :param: word_dir: directory holding pkl of word vocabulary
        :param: anno_file: pkl of annotation file 
        :param: phase: 'train' 'val' 'test'
        :param: fix_seg_len[bool]: fix the sample number for each video segment
        :param: args: arguments: act_trunct[int], fix_seg_len[bool], sample_num[int]

        :output: image blob, entities, image_path, new video flag for each action
        :output: action_length: length of each action in a video [list]
        :output: action_ind: action index for each action in a video
        """
        self.args = args
        self.fix_seg_len = fix_seg_len
        self.entity_type = args.entity_type
        self.model = args.model
        self.phase = phase
        phase_dict = {'train':'training', 'test':'testing', 'val':'validation'}
        data_path = os.path.join(root, dataset)
        # load words
        words_path = os.path.join(data_path, 'sampled_entities', '{}_entities.pkl'.format(phase))
        with open(words_path, 'rb') as f:
            self.word_dict = pickle.load(f)
        # load images
        # if self.fix_seg_len:
        spl_path = 'sampled_frames_splnum-1'
        # else:
        #     spl_path = 'sampled_frames_splnum{}'.format(args.sample_num_val)
        self.vids_path = os.path.join(data_path, spl_path)
        self.vid_list_file = os.path.join(data_path, 'split', '{}_list.txt'.format(phase))
        # load annotations
        self.anno_path = os.path.join(data_path, 'annotations', anno_file)
        self.box_anno_path = os.path.join(data_path, 'annotations', box_anno_file)
        # get segment number
        self.seg_num = self.get_segment_num()

    def get_segment_num(self):
        """ get the total number of video segment. each segment can be variable or fixed length
        """
        acc_seg_num = 0
        img_id = -1
        self.seg_accumulate_num = [0]
        self.seg_nums = []
        self.vid_ids = []
        self.vid_paths = []
        self.actions_length = []
        # img_id_dict vid->seg_id->frame_id --> image_id in the total dataset.
        self.img_id_dict = {}
        print('word dict length', len(self.word_dict))
        with open(self.vid_list_file) as f:
            for vid_i, line in enumerate(f):
                vid_img_holder = []
                vid_path = os.path.join(self.vids_path, line.rstrip('\n'))
                # get video id
                video_id = vid_path.split('/')[-1]
                # get entities in video, the entity can be 'sentence_raw', 'sentence', 'noun', 'category'
                if len(self.entity_type) == 2:
                    entities = self.word_dict[video_id][self.entity_type[0]]
                    sents = self.word_dict[video_id][self.entity_type[1]]
                else:
                    entities = self.word_dict[video_id][self.entity_type[0]]
                # process entities for training
                if len(entities) > self.args.act_trunc:
                    entities = entities[:self.args.act_trunc]
                # get action length
                action_length = [len(action) for action in entities]
                # TODO: deal with the empty action
                # print ('action length', action_length)
                acc_seg_num += len(entities)
                self.seg_accumulate_num.append(acc_seg_num)
                self.seg_nums.append(len(entities))
                self.vid_ids.append(video_id)
                self.vid_paths.append(vid_path)
                self.actions_length.append(action_length)
                # deal with img_id_dict
                image_list = glob.glob(os.path.join(vid_path, '*'))
                image_lists = self.div_imglst_by_name(image_list)
                # now we only can parse frame by name (reconstruct the name)
                for seg_ind, image_list in enumerate(image_lists):
                    seg_img_holder = {}
                    f_inds = self.get_frm_inds(image_list)
                    img_list = [image_list[f_ind] for f_ind in f_inds]
                    for img_ind, image in enumerate(img_list):
                        img_id = img_id + 1
                        _, _, _, frm_ind = self.parse_img_path(image)
                        seg_img_holder[frm_ind] = img_id
                    vid_img_holder.append(seg_img_holder)
                self.img_id_dict[video_id] = vid_img_holder

        self.seg_accumulate_num = np.array(self.seg_accumulate_num)
        self.seg_nums = np.array(self.seg_nums)
        return acc_seg_num

    def parse_index(self, index):
        """parse index to video id and segment index
        :param index: index over whole segment
        :output vid_index: video index [int]
        :output vid_id: video id [str]
        :output seg_ind: segment index [int]
        """
        # self.seg_accumulate_num: [0, seg1_num, seg1_num + seg2_num, ..., total_seg_num]
        # self.vid_ids: list of video id
        # self.vid_paths: list of video path
        # index must must be less then seg_num
        vid_index = np.where(self.seg_accumulate_num <= index)[0][-1]
        vid_id = self.vid_ids[vid_index]
        seg_ind = index - self.seg_accumulate_num[vid_index]
        return vid_index, vid_id, seg_ind

    def parse_img_path(self, filename):
        vid_id, code = filename.split('/')[-2:]
        code = code.split('.')[0]
        vid_ind = self.vid_ids.index(vid_id)
        action_ind = int(code[:4])
        try:
            frame_ind = int(code[4:])
        except:
            pdb.set_trace()
        return vid_id, vid_ind, action_ind, frame_ind

    def __len__(self):
        return self.seg_num

    def div_imglst_by_name(self, image_list):
        """divide image list by frame name
        :param image_list
        :output img_lists
        """
        image_list.sort()
        code_list = [path.split('/')[-1].split('.')[0] for path in image_list]
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

    def get_frm_inds(self, image_list):
        """
        :param image_list: list of image in a video segment
        """
        if self.phase == 'train':
            if self.args.fix_seg_len:
                f_inds = np.round(np.linspace(0, len(image_list)-1, self.args.sample_num)).astype('int')
            else:
                f_inds = np.arange(int(self.args.sample_rate/2), len(image_list), self.args.sample_rate, dtype=int)
        else:
            if self.args.fix_seg_len_val:
                f_inds = np.round(np.linspace(0, len(image_list)-1, self.args.sample_num_val)).astype('int')
            else:
                f_inds = np.arange(int(self.args.sample_rate_val/2), len(image_list), self.args.sample_rate_val, dtype=int)
        return f_inds

    def __getitem__(self, index):
        """ get item by index of segment
        """
        # parse segment index to video id and video-level segment id.
        vid_index, vid_id, seg_ind = self.parse_index(index)
        # get segment entities
        if len(self.entity_type) == 2:
            entities = self.word_dict[vid_id][self.entity_type[0]][seg_ind]
            sent = self.word_dict[vid_id][self.entity_type[1]][seg_ind][:20]
        else:
            entities = self.word_dict[vid_id][self.entity_type[0]][seg_ind]
        # get segment images (read n images in series)
        vid_path = self.vid_paths[vid_index]
        image_list = glob.glob(os.path.join(vid_path, '*'))
        image_lists = self.div_imglst_by_name(image_list)
        # now we only can parse frame by name (reconstruct the name)
        image_list = image_lists[seg_ind]
        f_inds = self.get_frm_inds(image_list)
        image_list = [image_list[f_ind] for f_ind in f_inds]
        imgs = []
        img_paths = []
        for i, img_path in enumerate(image_list):
            #  read image
            # print(img_path)
            img = cv2.imread(img_path)
            img = img.astype(np.float32, copy=True)
            img -= 127.5
            # resize image
            if img.shape[0] != self.args.img_h or img.shape[1] != self.args.img_w:
                img = cv2.resize(img, (self.args.img_h, self.args.img_w))
            # append image to ims
            imgs.append(img)
            img_paths.append(img_path)
            # transfer to blob (batch, 3, h, w)
            # preclude the condition that no entity in such action
        blob = im_list_to_blob(imgs)
        # blob = blob.transpose(0, 3, 1, 2)[0]
        # new video: true if current seg_ind is the last segment in a video, else false
        new_vid = True if self.seg_accumulate_num[vid_index] + seg_ind in self.seg_accumulate_num else False
        # get action_length
        action_length = self.actions_length[vid_index]
        action_ind = seg_ind
        # yeild image blob, word entity, image_path and new video flag

        if len(self.entity_type) == 2:
            return blob, entities, sent, img_paths, new_vid, action_length, action_ind
        elif len(self.entity_type) == 1:
            return blob, entities, img_paths, new_vid, action_length, action_ind

    def img_id_mapping(self, filename):
        """given file name, get the image id by file name
        :param filename: file name xx/xx/xx/vid_id/____(act_id)______(seg_id).jpg
        """
        vid_id, vid_ind, action_ind, frame_ind = self.parse_img_path(filename)
        try:
            img_id = self.img_id_dict[vid_id][action_ind][frame_ind]
        except:
            pdb.set_trace()
        return img_id

    def collate_fn(self, data):
        # data is a tuple with len of batch size.   
        if self.model == 'DVSA':
            return self.combine_batches(data)
        return data[0]
        
    def combine_batches(self, datas):
        # to combine batch of video segment, regardless video information, and do not need to consider the interaction between video segments.
        # and better no intersection of the same entity
        # datas contain (blob, entities, im_paths, new_vid, action_lengths, action_ind)
        # blobs: get the blob for all images [Nax5, h, w, c]
        blobs = []
        # entities: store all entities list
        entities = []
        # entity length, each video segment will have a length [list]
        entities_length = []
        # sentences
        sents = []
        # img_paths: get the image paths for all images
        img_paths = []
        # img_ids: get the image id for latter evaluation [list]
        img_ids = []
        # frm_length: get the frame length for a batch of proposals
        frm_length = []
        # rl_seg_inds: get the relative segment index
        rl_seg_inds = []
        # seg_nums: get the number of segment in each video
        seg_nums = []

        if len(self.entity_type) == 1:
            for i, data in enumerate(datas):
                blobs.append(data[0])
                entities.extend(data[1])
                # entities_length.append(len(data[1]))
                seg_nums.append(len(data[4]))
                entities_length.append(data[4][data[5]])
                img_paths.extend(data[2])
                blob_shape = data[0].shape
                img_ids.extend([self.img_id_mapping(img_path) for img_path in data[2]])
                rl_seg_inds.append(data[5])
        elif len(self.entity_type) == 2:
            for i, data in enumerate(datas):
                blobs.append(data[0])
                entities.extend(data[1])
                # entities_length.append(len(data[1]))
                sents.append(data[2])
                seg_nums.append(len(data[5]))
                entities_length.append(data[5][data[6]])
                img_paths.extend(data[3])
                blob_shape = data[0].shape
                img_ids.extend([self.img_id_mapping(img_path) for img_path in data[3]])
                rl_seg_inds.append(data[6])
            
        frm_length = [len(blob) for blob in blobs]
        blobs = np.concatenate(blobs, 0)

        blobs = blobs.reshape(-1, blob_shape[1], blob_shape[2], blob_shape[3])
        if len(self.entity_type) == 1:
            return blobs, entities, entities_length, frm_length, rl_seg_inds, seg_nums, img_paths, img_ids
        elif len(self.entity_type) == 2:
            return blobs, entities, sents, entities_length, frm_length, rl_seg_inds, seg_nums, img_paths, img_ids

    def provide_batch_spl_ind(self, batch_size, shuffle):
        """according batch_size, re-generate the sample index
        seg per action, and seg id, then we know
        :param index: index over whole segment
        :output vid_index: video index [int]
        :output vid_id: video id [str]
        :output seg_ind: segment index [int]
        # self.seg_nums: number for segment in each video
        # self.vid_ids: list of video id
        # self.vid_paths: list of video path
        # index must must be less then seg_num
        vid_index = np.where(self.seg_accumulate_num <= index)[0][-1]
        vid_id = self.vid_ids[vid_index]
        seg_ind = index - self.seg_accumulate_num[vid_index]
        return vid_index, vid_id, seg_ind
        """
        spl_inds = []
        # to satify the shuffle, we must shuffle seg_nums
        seg_nums = self.seg_nums.copy()
        # vid_num: video number
        vid_num = len(seg_nums)
        # vid_inds: video index list
        vid_inds = np.arange(vid_num)
        # if shuffle (only shuffle video level)
        if shuffle:
            np.random.shuffle(vid_inds)
        # construct accumulate segment number
        seg_accumulate_nums = np.add.accumulate(seg_nums)
        seg_accumulate_nums = np.hstack((np.zeros(1, dtype=int), seg_accumulate_nums))
        # construct map from vid_id to segment_id 
        vid2seg = [[i for i in range(seg_accumulate_nums[vid_ind],seg_accumulate_nums[vid_ind+1])] for vid_ind in range(vid_num)]
        # construct re-aranged seg_nums
        seg_nums = [seg_nums[vid_ind] for vid_ind in range(vid_num)]
        # construct re-aranged accumulated seg_nums
        seg_accumulate_nums = np.add.accumulate(seg_nums)
        seg_accumulate_nums = np.hstack((np.zeros(1, dtype=int), seg_accumulate_nums))
        # map vid2seg back to sequence seg indexes with re-aranged vid_inds
        # seg_inds: re-aranged seg indexes list
        seg_inds = []
        for vid_ind in vid_inds:
            seg_inds.extend(vid2seg[vid_ind])
        # seg_num: total number of segment in dataset
        seg_num = np.max(seg_accumulate_nums)
        for seg_ind in range(seg_num):
            spl_ind = seg_inds[seg_ind]
            spl_inds.append(spl_ind)
            if (spl_ind + 1) in seg_accumulate_nums:
                vid_ind = np.where(seg_accumulate_nums==(spl_ind+1))[0][0] - 1
                # current seg_ind is the last element in an video
                # get number for padding a batch
                spl_offset = (spl_ind - seg_accumulate_nums[vid_ind] + 1) % batch_size
                pad_num = batch_size - spl_offset if spl_offset != 0 else 0
                if pad_num > 0:
                    spl_inds.extend([-1]*pad_num)
        return spl_inds


class SubsetSampler(data.sampler.Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (list): a list of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class MPrpBatchSampler(object):
    """Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last):
        if not isinstance(sampler, data.sampler.Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for i, idx in enumerate(self.sampler):
            if idx >= 0:
                batch.append(int(idx))
            if (i+1)%self.batch_size == 0:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--act_trunc', dest='act_trunc',
                        help = 'maximum number of action in a video, for reducing memory',
                        default=20, type=int)
    parser.add_argument('--fix_seg_len', dest='fix_seg_len',
                    help='if fix the sample number of each video segment',
                    default=True, type=bool)
    parser.add_argument('--fix_seg_len_val', dest='fix_seg_len_val',
                    help='if fix the sample number of each video segment',
                    default=False, type=bool)
    parser.add_argument('--sample_num', dest='sample_num',
                    help='frame sample number for each video segment',
                    default=5, type=int)
    parser.add_argument('--sample_num_val', dest='sample_num_val',
                        help='sample number of frame per video segment for evaluation',
                        default=0, type=int)
    parser.add_argument('--sample_rate', dest='sample_rate',
                        help='sample rate of frame per video segment for training',
                        default=1, type=int)
    parser.add_argument('--sample_rate_val', dest='sample_rate_val',
                        help='sample rate of frame per video segment for evaluation',
                        default=16, type=int)
    parser.add_argument('--img_h', dest='img_h',
                    help='img height',
                    default=224, type=int)
    parser.add_argument('--img_w', dest='img_w',
                    help='img width',
                    default=224, type=int)
    parser.add_argument('--workers', dest='workers',
                    help='worker number',
                    default=0, type=int)
    parser.add_argument('--model', dest='model',
                    help='specify ground model',
                    default='DVSA', type=str)
    parser.add_argument('--batch_size', dest='batch_size',
                    help='batch size',
                    default=1, type=int)
    parser.add_argument('--entity_type', dest='entity_type',
                    help='entity type of input language',
                    default=['category'], type=list)
    parser.add_argument('--phase', dest='phase',
                        help='phase: train, test, val',
                        default='val',
                        choices=['train', 'val', 'test'], type=str)

    args = parser.parse_args()
    root = 'data'
    dataset = 'YouCookII'
    anno_file = 'youcookii_annotations.json'
    if args.phase != 'train':
        box_anno_file = 'yc2_bb_{}_annotations.json'.format(args.phase)
    else:
        box_anno_file = 'yc2_bb_val_annotations.json' #pseudo input
    class_file = 'youcook_cls.txt'

    ######################
    ## test MPrpDataSet ##
    ######################
    DataSet = MPrpDataSet(root, dataset, anno_file, box_anno_file, args.phase, args.fix_seg_len, args)
    sampler = SubsetSampler(DataSet.provide_batch_spl_ind(args.batch_size, shuffle=True))
    batch_sampler = MPrpBatchSampler(sampler, args.batch_size, drop_last=False)
    data_loader = data.DataLoader(DataSet, num_workers=args.workers, collate_fn=DataSet.collate_fn,
        batch_sampler=batch_sampler)
    t_s = time.time()
    for i, (blobs, entities, entities_length, frm_length, rl_seg_inds, seg_nums, img_paths, img_ids) in enumerate(data_loader):
        t_e = time.time()
        t_ave = (t_e - t_s)/(i+1)/args.batch_size
        print ('time {:.3f}'.format(t_ave), blobs.shape, entities_length, frm_length, rl_seg_inds, seg_nums, entities, img_ids, img_paths)


    ####################################################
    ## test MPrpDataSet both load entity and sentence ##
    ####################################################
    """
    DataSet = MPrpDataSet(root, dataset, anno_file, box_anno_file, args.phase, args.fix_seg_len, args)
    sampler = SubsetSampler(DataSet.provide_batch_spl_ind(args.batch_size, shuffle=True))
    batch_sampler = MPrpBatchSampler(sampler, args.batch_size, drop_last=False)
    data_loader = data.DataLoader(DataSet, num_workers=args.workers, collate_fn=DataSet.collate_fn,
        batch_sampler=batch_sampler)
    ## data_loader = data.DataLoader(DataSet, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True, 
    ##    collate_fn=DataSet.collate_fn)  
    t_s = time.time()
    for i, (blobs, entities, sents, entities_length, frm_length, rl_seg_inds, seg_nums, img_paths, img_ids) in enumerate(data_loader):
        t_e = time.time()
        t_ave = (t_e - t_s)/(i+1)/args.batch_size
        print ('time {:.3f}'.format(t_ave), sents, blobs.shape, entities_length, frm_length, rl_seg_inds, seg_nums, entities, img_ids, img_paths)
    """

