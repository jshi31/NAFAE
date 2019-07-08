import torch.utils.data as data
from torch._six import int_classes as _int_classes
import os 
import sys
import pdb
import pickle
import json
import argparse
import numpy as np
import glob
import copy
import cv2
import time
pwd = os.getcwd()
sys.path.append(pwd)
sys.path.append(os.path.join(pwd, 'lib'))

from model.utils.blob import im_list_to_blob



def RWDataLoader(root, anno_file, phase, args):
    """
    image shape (224x224), image batch_size=5
    :param: root data 
    :param: vid_list_file: list of videos
    :param: word_file: pkl of word dictionary
    """
    args = args
    entity_type = args.entity_type
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
    for vid_i, vid_id in enumerate(vid_ids):
        anno_dict_vid = anno_dict[vid_id]
        # devide to segements:
        frm_segs = anno_dict_vid['frm']
        ent_segs = anno_dict_vid['entity']
        box_dict = anno_dict_vid['box']

        for seg_i in range(len(frm_segs)):
            frm_seg = frm_segs[seg_i]
            ent_seg = ent_segs[seg_i]
            # get entities
            entities = ent_seg[args.entity_type[0]]
            # get sentence 
            sent = [ent_seg[args.entity_type[1]]]
            # get entities_length 
            entities_length = [len(entities)]
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
                # get img_ids
                img_ids.append(img_id)
                img_id += 1
                # get img paths 
                valid_img_paths.append(img_path)
                img = cv2.imread(img_path)
                try:
                    img = img.astype(np.float32, copy=True)
                except:
                    pdb.set_trace()
                img -= 127.5
                # resize image
                if img.shape[0] != 224 or img.shape[1] != 224:
                    img = cv2.resize(img, (224, 224))
                # append image to ims
                imgs.append(img)
                # transfer to blob (batch, 3, h, w)
            if not imgs:
                continue
            blob = im_list_to_blob(imgs)
            yield blob, entities, sent, entities_length, valid_img_paths, img_ids


class GroundDataSet(data.Dataset):
    def __init__(self, root, dataset, anno_file, box_anno_file, phase, args):
        """
        :param: root: project root directory
        :param: dataset: dataset Name 
        :param: word_dir: directory holding pkl of word vocabulary
        :param: anno_file: pkl of annotation file 
        :param: phase: 'train' 'val' 'test'
        :param: args: arguments
        :output: image blob, entities, image_path, new video flag for each action
        :output: action_length: length of each action in a video [list]
        :output: action_ind: action index for each action in a video
        """
        self.args = args
        self.entity_type = 'category'
        self.model = args.model
        data_path = os.path.join(root, dataset)
        # load words
        words_path = os.path.join(data_path, 'sampled_entities', '{}_entities.pkl'.format(phase))
        with open(words_path, 'rb') as f:
            self.word_dict = pickle.load(f)
        # load images
        self.vids_path = os.path.join(data_path, 'sampled_frames')
        self.vid_list_file = os.path.join(data_path, 'split', '{}_list.txt'.format(phase))

        # load annotations
        self.anno_path = os.path.join(data_path, 'annotations', anno_file)
        self.box_anno_path = os.path.join(data_path, 'annotations', box_anno_file)
        # get segment number
        self.seg_num = self.get_segment_num()

    def get_segment_num(self):
        """ get the total number of video segment. each segment sample fix number of frames.
        """
        seg_num = 0
        self.seg_accumulate_num = [0]
        self.vid_ids = []
        self.vid_paths = []
        self.actions_length = []
        print('word dict length', len(self.word_dict))
        with open(self.vid_list_file) as f:
            for line in f:
                vid_path = os.path.join(self.vids_path, line.rstrip('\n'))
                # get video id
                video_id = vid_path.split('/')[-1]
                # get entities in video, the entity can be 'sentence_raw', 'sentence', 'noun', 'category'
                entities = self.word_dict[video_id][self.entity_type]
                # process entities for training
                if len(entities) > self.args.act_trunc:
                    entities = entities[:self.args.act_trunc]
                # get action length
                action_length = [len(action) for action in entities]
                # TODO: deal with the empty action
                # print ('action length', action_length)
                seg_num += len(entities)
                self.seg_accumulate_num.append(seg_num)
                self.vid_ids.append(video_id)
                self.vid_paths.append(vid_path)
                self.actions_length.append(action_length)
        self.seg_accumulate_num = np.array(self.seg_accumulate_num)
        return seg_num

    # 1. need to get the map from index to video, and video offset
    # 2. need to get the 
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

    def __len__(self):
        return self.seg_num

    def __getitem__(self, index):
        """ get item by index of segment
        """
        # parse segment index to video id and video-level segment id.
        vid_index, vid_id, seg_ind = self.parse_index(index)
        # get segment entities
        entities = self.word_dict[vid_id][self.entity_type][seg_ind]
        # get segment images (read n images in series)
        vid_path = self.vid_paths[vid_index]
        image_list = glob.glob(os.path.join(vid_path, '*'))
        image_list.sort()
        image_list = image_list[seg_ind*self.args.sample_num:(seg_ind + 1)*self.args.sample_num]
        imgs = []
        img_paths = []
        for i, img_path in enumerate(image_list):
            #  read image
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
        return blob, entities, img_paths, new_vid, action_length, action_ind

    def img_id_mapping(self, filename):
        """given file name, get the image id by file name
        :param filename: file name xx/xx/xx/vid_id/__(act_id)__(seg_id).jpg
        """
        vid_id, code = filename.split('/')[-2:]
        code = code.split('.')[0]
        vid_ind = self.vid_ids.index(vid_id)
        action_ind = int(code[:2])
        frame_ind = int(code[2:])
        img_id_offset = action_ind*self.args.sample_num + frame_ind
        img_id = self.seg_accumulate_num[vid_ind]*self.args.sample_num + img_id_offset
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
        # img_paths: get the image paths for all images
        img_paths = []
        # img_ids: get the image id for latter evaluation [list]
        img_ids = []
        for i, data in enumerate(datas):
            blobs.append(data[0])
            entities.extend(data[1])
            entities_length.append(len(data[1]))
            img_paths.extend(data[2])
            blob_shape = data[0].shape
            img_ids.extend([self.img_id_mapping(img_path) for img_path in data[2]])
        blobs = np.array(blobs)
        blobs = blobs.reshape(-1, blob_shape[1], blob_shape[2], blob_shape[3])
        return blobs, entities, entities_length, img_paths, img_ids


class GroundTransDataSet(data.Dataset):
    def __init__(self, root, dataset, anno_file, box_anno_file, phase, args):
        """
        :param: root: project root directory
        :param: dataset: dataset Name 
        :param: word_dir: directory holding pkl of word vocabulary
        :param: anno_file: pkl of annotation file 
        :param: trans_file: pkl of transcript file
        :param: phase: 'train' 'val' 'test'
        :param: args: arguments
        :output: image blob, entities, image_path, new video flag for each action
        :output: action_length: length of each action in a video [list]
        :output: action_ind: action index for each action in a video
        """
        self.args = args
        self.entity_type = 'category'
        self.model = args.model
        data_path = os.path.join(root, dataset)
        # load words
        words_path = os.path.join(data_path, 'sampled_entities', '{}_entities.pkl'.format(phase))
        # load transcripts
        self.trans_path = os.path.join(data_path, 'sampled_entities', '{}_transcripts.pkl'.format(phase))
        with open(words_path, 'rb') as f:
            self.word_dict = pickle.load(f)
        # load images
        self.vids_path = os.path.join(data_path, 'sampled_frames')
        self.vid_list_file = os.path.join(data_path, 'split', '{}_list.txt'.format(phase))

        # load annotations
        self.anno_path = os.path.join(data_path, 'annotations', anno_file)
        self.box_anno_path = os.path.join(data_path, 'annotations', box_anno_file)

        # get segment number
        self.seg_num = self.get_segment_num()
        # get stamped transcript
        self.timed_trans = GroundTransDataSet.get_timed_trans(self.trans_path)

    @staticmethod
    def get_timed_trans(trans_path):
        """ 
        :param trans_path: transcript path
        :output vid_stamped_nouns: time stamp for each noun in transcript within each segment.
        """
        with open(trans_path, 'rb') as f:
            trans = pickle.load(f)
        trans_vid_ids = [i for i in trans.keys()]
        out_stamped_nouns = {}
        for vid_i, vid_id in enumerate(trans_vid_ids):
            trans_dict = trans[vid_id]
            segs = trans_dict['segments']
            timed_nouns = trans_dict['noun']
            stamped_nouns = np.array([(v[0], (v[1] + v[2])/2) for v in timed_nouns])
            stamps = np.array([v[1] for v in stamped_nouns], dtype=float)
            nouns = np.array([v[0] for v in stamped_nouns])
            # find stamps within segments
            vid_stamped_nouns = []
            for seg in segs:
                inds = (seg[0] < stamps) * (stamps < seg[1])
                seg_stamps = (stamps[inds] - seg[0])/(seg[1] - seg[0])
                seg_nouns = nouns[inds]
                vid_stamped_nouns.append([tup for tup in zip(seg_nouns, seg_stamps)])
            out_stamped_nouns[vid_id] = vid_stamped_nouns
        return out_stamped_nouns


    def get_segment_num(self):
        """ get the total number of video segment. each segment sample fix number of frames.
        """
        seg_num = 0
        self.seg_accumulate_num = [0]
        self.vid_ids = []
        self.vid_paths = []
        self.actions_length = []
        print('word dict length', len(self.word_dict))
        with open(self.vid_list_file) as f:
            for line in f:
                vid_path = os.path.join(self.vids_path, line.rstrip('\n'))
                # get video id
                video_id = vid_path.split('/')[-1]
                # get entities in video, the entity can be 'sentence_raw', 'sentence', 'noun', 'category'
                entities = self.word_dict[video_id][self.entity_type]
                # process entities for training
                if len(entities) > self.args.act_trunc:
                    entities = entities[:self.args.act_trunc]
                # get action length
                action_length = [len(action) for action in entities]
                # TODO: deal with the empty action
                # print ('action length', action_length)
                seg_num += len(entities)
                self.seg_accumulate_num.append(seg_num)
                self.vid_ids.append(video_id)
                self.vid_paths.append(vid_path)
                self.actions_length.append(action_length)
        self.seg_accumulate_num = np.array(self.seg_accumulate_num)
        return seg_num

    # 1. need to get the map from index to video, and video offset
    # 2. need to get the 
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

    def __len__(self):
        return self.seg_num

    def __getitem__(self, index):
        """ get item by index of segment
        """
        # parse segment index to video id and video-level segment id.
        vid_index, vid_id, seg_ind = self.parse_index(index)
        # get segment entities
        entities = self.word_dict[vid_id][self.entity_type][seg_ind]
        # get segment timed transcript [(noun, time)] or []
        timed_trans = self.timed_trans[vid_id][seg_ind] if vid_id in self.timed_trans.keys() else []
        # get segment images (read n images in series)
        vid_path = self.vid_paths[vid_index]
        image_list = glob.glob(os.path.join(vid_path, '*'))
        image_list.sort()
        image_list = image_list[seg_ind*self.args.sample_num:(seg_ind + 1)*self.args.sample_num]
        imgs = []
        img_paths = []
        for i, img_path in enumerate(image_list):
            #  read image
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
        return blob, entities, timed_trans, img_paths, new_vid, action_length, action_ind

    def img_id_mapping(self, filename):
        """given file name, get the image id by file name
        :param filename: file name xx/xx/xx/vid_id/__(act_id)__(seg_id).jpg
        """
        vid_id, code = filename.split('/')[-2:]
        code = code.split('.')[0]
        vid_ind = self.vid_ids.index(vid_id)
        action_ind = int(code[:2])
        frame_ind = int(code[2:])
        img_id_offset = action_ind*self.args.sample_num + frame_ind
        img_id = self.seg_accumulate_num[vid_ind]*self.args.sample_num + img_id_offset
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
        # timed_trans [single_batch_timed_trans, ...]
        timed_trans = []
        # img_paths: get the image paths for all images
        img_paths = []
        # img_ids: get the image id for latter evaluation [list]
        img_ids = []
        for i, data in enumerate(datas):
            blobs.append(data[0])
            entities.extend(data[1])
            entities_length.append(len(data[1]))
            timed_trans.append(data[2])
            img_paths.extend(data[3])
            blob_shape = data[0].shape
            img_ids.extend([self.img_id_mapping(img_path) for img_path in data[3]])
        blobs = np.array(blobs)
        blobs = blobs.reshape(-1, blob_shape[1], blob_shape[2], blob_shape[3])
        return blobs, entities, entities_length, timed_trans, img_paths, img_ids


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

class MPrpTransDataSet(data.Dataset):
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
        phase_dict = {'train':'training', 'test':'testing', 'val':'validation'}
        data_path = os.path.join(root, dataset)
        # load words
        words_path = os.path.join(data_path, 'sampled_entities', '{}_entities.pkl'.format(phase))
        with open(words_path, 'rb') as f:
            self.word_dict = pickle.load(f)
        # load transcripts
        self.trans_path = os.path.join(data_path, 'sampled_entities', '{}_transcripts.pkl'.format(phase))
        with open(words_path, 'rb') as f:
            self.word_dict = pickle.load(f)
        # load images
        spl_path = 'sampled_frames_splnum-1'
        self.vids_path = os.path.join(data_path, spl_path)
        self.vid_list_file = os.path.join(data_path, 'split', '{}_list.txt'.format(phase))
        # load annotations
        self.anno_path = os.path.join(data_path, 'annotations', anno_file)
        self.box_anno_path = os.path.join(data_path, 'annotations', box_anno_file)
        # get segment number
        self.seg_num = self.get_segment_num()
        # get stamped transcript
        self.timed_trans = MPrpTransDataSet.get_timed_trans(self.trans_path)
        # optimized time sample window
        self.optwindow_lst = [['bacon', [5, 9], 0.41371393169700166], ['bean', [1, 6], 0.25897013970773897], ['beef', [5, 7], 0.4394933253721144], ['blender', [4, 6], 0.41829053770249713], ['bowl', [5, 10], 0.4338670334155178], ['bread', [4, 9], 0.43447960243597217], ['butter', [1, 5], 0.28077042639770344], ['cabbage', [7, 10], 0.3962063210155055], ['carrot', [3, 8], 0.15974401775162933], ['celery', [1, 3], 0.19293498997680647], ['cheese', [7, 6], 0.28555111364934244], ['chicken', [5, 9], 0.3378052311552939], ['chickpea', [3, 6], 0.29826127619776416], ['corn', [9, 8], 0.5837800134951749], ['cream', [4, 4], 0.20792452339800646], ['cucumber', [6, 10], 0.3759983012332605], ['cup', [2, 3], 0.32641107561235344], ['dough', [2, 9], 0.4144261751572229], ['egg', [2, 4], 0.29384979661222793], ['flour', [2, 9], 0.3112292666370571], ['garlic', [1, 9], 0.1765718995829209], ['ginger', [4, 2], 0.12752248463930116], ['it', [6, 5], 0.3971877560838004], ['leaf', [1, 6], 0.6434065934065922], ['lemon', [3, 4], 0.11033517525893585], ['lettuce', [6, 9], 0.42604252164473466], ['lid', [2, 10], 0.23508668821627943], ['meat', [5, 9], 0.3734175712812706], ['milk', [4, 5], 0.19758345474551103], ['mixture', [2, 9], 0.3411395740284071], ['mushroom', [6, 7], 0.3656262568402578], ['mussel', [1, 8], 0.9360000000000014], ['mustard', [4, 8], 0.1496206048947312], ['noodle', [5, 6], 0.4150520967880819], ['oil', [1, 7], 0.21077869554334766], ['onion', [1, 6], 0.25068708222361463], ['oven', [2, 10], 0.283044809699394], ['pan', [3, 7], 0.44831081478975576], ['paper', [4, 8], 0.4529760946099348], ['pasta', [7, 9], 0.39817882234049373], ['pepper', [1, 6], 0.11132444852941174], ['plate', [1, 9], 0.43137104211496935], ['pork', [1, 7], 0.5560506244164314], ['pot', [3, 7], 0.46334393747030134], ['potato', [8, 10], 0.31610980778937214], ['powder', [1, 8], 0.19168104274487258], ['processor', [1, 8], 0.353982700812599], ['rice', [2, 8], 0.34076886568580894], ['salad', [4, 10], 0.5417850378787878], ['salt', [4, 10], 0.06428592563096089], ['sauce', [2, 6], 0.20667613636363635], ['seaweed', [2, 10], 0.31070977764831537], ['sesame', [1, 3], 0.19138755980861213], ['shrimp', [6, 10], 0.4423107503755421], ['soup', [1, 8], 0.4641387419165185], ['squid', [1, 10], 0.29849027797312966], ['sugar', [1, 6], 0.11143746304607544], ['that', [1, 1], 0], ['them', [4, 10], 0.48081152676261185], ['they', [4, 7], 0.6959115198210496], ['tofu', [5, 10], 0.4372222222222217], ['tomato', [3, 5], 0.2218925247921476], ['vinegar', [3, 6], 0.0924095837912004], ['water', [1, 6], 0.2158833663444971], ['whisk', [9, 1], 0.28004667444574083], ['wine', [1, 9], 0.15981751968178115], ['wok', [5, 9], 0.44878596204856147]]
        self.class_list = [v[0] for v in self.optwindow_lst]

    @staticmethod
    def get_timed_trans(trans_path):
        """ 
        :param trans_path: transcript path
        :output vid_stamped_nouns: time stamp for each noun in transcript within each segment.
        """
        with open(trans_path, 'rb') as f:
            trans = pickle.load(f)
        trans_vid_ids = [i for i in trans.keys()]
        out_stamped_nouns = {}
        for vid_i, vid_id in enumerate(trans_vid_ids):
            trans_dict = trans[vid_id]
            segs = trans_dict['segments']
            timed_nouns = trans_dict['category']
            stamped_nouns = np.array([(v[0], (v[1] + v[2])/2) for v in timed_nouns])
            stamps = np.array([v[1] for v in stamped_nouns], dtype=float)
            nouns = np.array([v[0] for v in stamped_nouns])
            # find stamps within segments
            vid_stamped_nouns = []
            for seg in segs:
                inds = (seg[0] < stamps) * (stamps < seg[1])
                seg_stamps = stamps[inds] - seg[0]
                seg_nouns = nouns[inds]
                vid_stamped_nouns.append([tup for tup in zip(seg_nouns, seg_stamps)])
            out_stamped_nouns[vid_id] = vid_stamped_nouns
        return out_stamped_nouns

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
            for line in f:
                vid_img_holder = []
                vid_path = os.path.join(self.vids_path, line.rstrip('\n'))
                # get video id
                video_id = vid_path.split('/')[-1]
                # get entities in video, the entity can be 'sentence_raw', 'sentence', 'noun', 'category'
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
                    seg_img_holder = []
                    if self.fix_seg_len:
                        f_inds = np.round(np.linspace(0, len(image_list)-1, self.args.sample_num)).astype('int')
                    for img_ind, image in enumerate(image_list):
                        if self.fix_seg_len:
                            img_id = img_id + 1 if (img_ind in f_inds) else img_id
                        else:
                            img_id += 1
                        seg_img_holder.append(img_id)
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


    def __getitem__(self, index):
        """ get item by index of segment
        """
        # parse segment index to video id and video-level segment id.
        vid_index, vid_id, seg_ind = self.parse_index(index)
        # get segment entities
        entities = self.word_dict[vid_id][self.entity_type[0]][seg_ind]
        # get segment timed transcript [(noun, time)] or []
        timed_trans = self.timed_trans[vid_id][seg_ind] if vid_id in self.timed_trans.keys() else []
        # filter by entities
        if timed_trans:
            timed_trans = [v for v in timed_trans if v[0] in entities]
        # map from entity to timed_trans index
        filter_trans = [v[0] for v in timed_trans]
        ent_trans_map = {ent: filter_trans.index(ent) for ent in entities if ent in filter_trans}
        # get segment images (read n images in series)
        vid_path = self.vid_paths[vid_index]
        image_list = glob.glob(os.path.join(vid_path, '*'))
        image_lists = self.div_imglst_by_name(image_list)
        # now we only can parse frame by name (reconstruct the name)
        image_list = image_lists[seg_ind]
        seg_dur = len(image_list)/16

        # init entities bias weight
        ents_weight = []

        if self.fix_seg_len:
            if timed_trans:
                f_inds = []
                # get sample index
                trans_len = len(timed_trans)
                num_ave = self.args.sample_num // trans_len
                num_last = self.args.sample_num - num_ave*(trans_len - 1)
                snips_spl_num = [num_ave for _ in range(trans_len)]
                snips_spl_num[-1] = num_last
                for snip_i, snip_spl_num in enumerate(snips_spl_num):
                    snip_stamp = timed_trans[snip_i][1]
                    # get opt window
                    cls_ind = self.class_list.index(timed_trans[snip_i][0])
                    scope = self.optwindow_lst[cls_ind][1]
                    ### adjust the shift
                    snip_scope = [np.clip(snip_stamp - scope[0], 0, seg_dur), np.clip(snip_stamp + scope[1], 0, seg_dur)]
                    # snip_scope = [np.clip(snip_stamp - 0.5, 0, seg_dur), np.clip(snip_stamp + 0.5, 0, seg_dur)]
                    snip_inds = (snip_scope[0] + np.random.rand(snip_spl_num)*(snip_scope[1] - snip_scope[0]))*16
                    snip_inds = snip_inds.astype('int')
                    f_inds.extend(snip_inds)

                # append ent_weight to ents_weight
                # print('snips_spl_num: ', snips_spl_num)
                for ent in entities:
                    ent_weight = np.zeros(self.args.sample_num)
                    frm_head = 0
                    if ent in ent_trans_map.keys():
                        ent_trans_ind = ent_trans_map[ent]
                        for i in range(ent_trans_ind):
                            frm_head += snips_spl_num[i]
                        frm_num = snips_spl_num[ent_trans_map[ent]]
                        ent_weight[frm_head:frm_head + frm_num] = 1
                        ents_weight.append(ent_weight)
                    else:
                        # entity regard each frame equally
                        ents_weight.append(np.ones(self.args.sample_num)/2)

            else:
                f_inds = np.round(np.linspace(0, len(image_list)-1, self.args.sample_num)).astype('int')
                # append ent_weight to ents_weigth
                for ent in entities:
                    ents_weight.append(np.ones(self.args.sample_num)/2)

            image_list = [image_list[f_ind] for f_ind in f_inds]



        imgs = []
        img_paths = []
        for i, img_path in enumerate(image_list):
            #  read image
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
        return blob, entities, img_paths, new_vid, action_length, action_ind, ents_weight

    def img_id_mapping(self, filename):
        """given file name, get the image id by file name
        :param filename: file name xx/xx/xx/vid_id/____(act_id)______(seg_id).jpg
        """
        vid_id, code = filename.split('/')[-2:]
        code = code.split('.')[0]
        vid_ind = self.vid_ids.index(vid_id)
        action_ind = int(code[:4])
        frame_ind = int(code[4:])
        img_id = self.img_id_dict[vid_id][action_ind][frame_ind]
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
        # img_paths: get the image paths for all images
        img_paths = []
        # img_ids: get the image id for latter evaluation [list]
        img_ids = []
        # frm_length: get the frame length for a batch of proposals
        frm_length = []
        # rl_seg_inds: get the relative segment index
        rl_seg_inds = []
        # ents_weight: get the entitie wights for a batch
        ents_weight = []

        for i, data in enumerate(datas):
            blobs.append(data[0])
            entities.extend(data[1])
            # entities_length.append(len(data[1]))
            entities_length.append(data[4][data[5]])
            img_paths.extend(data[2])
            blob_shape = data[0].shape
            img_ids.extend([self.img_id_mapping(img_path) for img_path in data[2]])
            rl_seg_inds.append(data[5])
            ents_weight.extend(data[6])

        frm_length = [len(blob) for blob in blobs]
        blobs = np.concatenate(blobs, 0)

        blobs = blobs.reshape(-1, blob_shape[1], blob_shape[2], blob_shape[3])
        return blobs, entities, ents_weight, entities_length, frm_length, rl_seg_inds, img_paths, img_ids

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

class MFrmDataSet(data.Dataset):
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
        # load flow
        flo_path = 'sampled_flow_splnum-1'
        self.flo_path = os.path.join(data_path, flo_path)
        # get segment number
        self.seg_num = self.get_segment_num()
        # get frm number
        self.frm_num = self.get_frm_num()

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
                entities = self.word_dict[video_id][self.entity_type]
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

    def get_frm_num(self):
        """ get the total number of image. each segment can be variable or fixed length
        return: reverse image id mapping vector img_id_rev_dict {img_id: vid_id, vid_i, seg_i, frm_i}
        """
        assert self.args.sample_rate == 1, 'sample rate should be 1 but get {}'.format(self.args.sample_rate)
        assert self.args.sample_rate_val == 1, 'sample rate val should be 1 but get {}'.format(self.args.sample_rate_val)
        self.img_id_rev_dict = {}
        self.img_len_dict = {}
        frm_num = 0
        vid_ids = [vid_id for vid_id in self.img_id_dict.keys()]
        for vid_i, vid_id in enumerate(vid_ids):
            vid_img_holder = self.img_id_dict[vid_id]
            frm_lens = []
            for seg_i, seg_img_holder in enumerate(vid_img_holder):
                frm_num += len(seg_img_holder)
                frm_lens.append(len(seg_img_holder))
                self.img_len_dict[vid_id] = frm_lens
                seg_img_lst = [seg_img_holder[i] for i in range(len(seg_img_holder))]
                for frm_i, img_id in enumerate(seg_img_lst):
                    self.img_id_rev_dict[img_id] = [vid_id, vid_i, seg_i, frm_i]
        return frm_num

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
        frame_ind = int(code[4:])
        return vid_id, vid_ind, action_ind, frame_ind

    def __len__(self):
        return self.frm_num

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
        """ get item by index of frm
        """
        # parse segment index to video id and video-level segment id.
        vid_id, vid_index, seg_ind, frm_ind = self.img_id_rev_dict[index]

        # get segment entities
        entities = self.word_dict[vid_id][self.entity_type][seg_ind]
        # get segment images (read n images in series)
        vid_path = self.vid_paths[vid_index]
        img_path = os.path.join(vid_path, '{:04d}{:06d}.jpg'.format(seg_ind, frm_ind))
        #  read image
        img = cv2.imread(img_path)
        img = img.astype(np.float32, copy=True)
        img -= 127.5
        # resize image
        if img.shape[0] != self.args.img_h or img.shape[1] != self.args.img_w:
            img = cv2.resize(img, (self.args.img_h, self.args.img_w))
        blob = im_list_to_blob([img])

        # get action_length
        action_length = self.actions_length[vid_index]
        action_ind = seg_ind
        # get frame length
        frm_length = self.img_len_dict[vid_id][seg_ind]

        # get flow
        flo_path = vid_path.replace('frames', 'flow')
        
        if frm_ind%16 == 0:
            self.flo_seg = seg_ind
            self.seg_flo_f = np.load(os.path.join(flo_path, 'f{:04d}.npy'.format(seg_ind)))
            self.seg_flo_b = np.load(os.path.join(flo_path, 'b{:04d}.npy'.format(seg_ind)))

        assert self.flo_seg == seg_ind, 'loading flow speed in different frame unbalanced'
        if frm_ind + 1 == frm_length:
            flo_f = self.seg_flo_f[frm_ind - 1]
            flo_b = self.seg_flo_b[frm_ind - 1]
        else:
            flo_f = self.seg_flo_f[frm_ind]
            flo_b = self.seg_flo_b[frm_ind]
        flo = [flo_f, flo_b]
        # yeild image blob, word entity, image_path and new video flag
        return blob, flo, img_path, action_length, action_ind, frm_length, frm_ind

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
        # img_paths: get the image paths for all images
        img_paths = []
        # img_ids: get the image id for latter evaluation [list]
        img_ids = []
        # frm_inds: get the relative frame index in a video segment
        frm_inds = []
        # frm_length: get the frame length for a batch of proposals
        frm_length = []
        # rl_seg_inds: get the relative segment index
        seg_inds = []
        # seg_nums: get the number of segment in each video
        seg_length = []
        # flos: get forward and backward flow [flos_f, flos_b]
        flos_f = []
        flos_b = []
        for i, data in enumerate(datas):
            blobs.append(data[0])
            flos_f.append(data[1][0])
            flos_b.append(data[1][1])
            seg_length.append(len(data[3]))
            img_paths.append(data[2])
            blob_shape = data[0].shape
            img_ids.append(self.img_id_mapping(data[2]))
            seg_inds.append(data[4])
            frm_length.append(data[5])
            frm_inds.append(data[6])

        blobs = np.concatenate(blobs, 0)
        blobs = blobs.reshape(-1, blob_shape[1], blob_shape[2], blob_shape[3])
        flos_f = np.array(flos_f, dtype=float)
        flos_b = np.array(flos_b, dtype=float)
        flos = [flos_f, flos_b]
        return blobs, flos, seg_inds, seg_length, frm_inds, frm_length, img_paths, img_ids

def show_ub(dataloader, recs, dets, classes):
    "show upperbound of the video images"
    from model.utils.net_utils import vis_single_det
    for batch_ind, (blobs, entities, entities_length, frm_length, rl_seg_inds, img_paths, img_ids) in enumerate(data_loader):
        for i in range(len(blobs)):
            img = blobs[i][:, :, ::-1]
            img_id = img_ids[i]
            rec = recs[img_id] # 
            det = np.array(dets[img_id]) # (n_box, 5) (x1, y1, x2, y2, cls_ind)
            # vis_single_dets(img, dets[bbox_n, 5], classes[bbox_n]
            img = vis_single_det(img, np.array(rec['bbox']), rec['label'])
            if det.any():
                img = vis_single_det(img, det[:, :4], [classes[int(j)] for j in det[:, -1]], color=(0, 0, 204), thickness=2)
            # process save file
            img_path = img_paths[i]
            img_path = img_path.replace('sampled_frames', 'vis_frames')
            head = img_path.split('/')[:-7]
            tail = img_path.split('/')[-7:]
            save_path = head + ['output'] + tail
            save_path = '/'.join(save_path)
            if not os.path.isdir(save_path.rsplit('/', 1)[0]):
                os.makedirs(save_path.rsplit('/', 1)[0])
            cv2.imwrite(save_path, img)
            print ('save_path {}'.format(save_path))

        print (blobs.shape, entities_length, frm_length, rl_seg_inds, entities, img_ids, img_paths)




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
        box_anno_file = 'yc2_bb_val_annotations.json'
    class_file = 'youcook_cls.txt'


    ########################
    ## test GroundDataSet ##
    ########################

    # DataSet = GroundDataSet(root, dataset, anno_file, box_anno_file, args.phase, args)
    # data_loader = data.DataLoader(DataSet, batch_size=8, shuffle=False, num_workers=args.workers, collate_fn=DataSet.collate_fn)
    # t_s = time.time()
    # for i, (blobs, entities, entities_length, img_paths, img_ids) in enumerate(data_loader):
    #     t_e = time.time()
    #     t_ave = (t_e - t_s)/(i+1)
    #     print ('time {:.3f}'.format(t_ave), blobs.shape, entities_length, entities, img_ids, img_paths)

    #############################
    ## test GroundTransDataSet ##
    #############################

    # DataSet = GroundTransDataSet(root, dataset, anno_file, box_anno_file, args.phase, args)
    # data_loader = data.DataLoader(DataSet, batch_size=8, shuffle=False, num_workers=args.workers, drop_last=True, collate_fn=DataSet.collate_fn)
    # t_s = time.time()
    # for i, (blobs, entities, entities_length, timed_trans, img_paths, img_ids) in enumerate(data_loader):
    #     t_e = time.time()
    #     t_ave = (t_e - t_s)/(i+1)
    #     print ('time {:.3f}'.format(t_ave), blobs.shape, timed_trans, entities_length, entities, img_ids, img_paths)

    ######################
    ## test MPrpDataSet ##
    ######################
    """
    DataSet = MPrpDataSet(root, dataset, anno_file, box_anno_file, args.phase, args.fix_seg_len, args)
    sampler = SubsetSampler(DataSet.provide_batch_spl_ind(args.batch_size, shuffle=True))
    batch_sampler = MPrpBatchSampler(sampler, args.batch_size, drop_last=False)
    data_loader = data.DataLoader(DataSet, num_workers=args.workers, collate_fn=DataSet.collate_fn,
        batch_sampler=batch_sampler)
    # data_loader = data.DataLoader(DataSet, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True, 
    #    collate_fn=DataSet.collate_fn)  
    t_s = time.time()
    for i, (blobs, entities, entities_length, frm_length, rl_seg_inds, seg_nums, img_paths, img_ids) in enumerate(data_loader):
        t_e = time.time()
        t_ave = (t_e - t_s)/(i+1)/args.batch_size
        print ('time {:.3f}'.format(t_ave), blobs.shape, entities_length, frm_length, rl_seg_inds, seg_nums, entities, img_ids, img_paths)
    """


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

    #######################################
    ## test MPropDatset with each other ##
    #####################################
    # args1 = args 
    # args1.fix_seg_len_val = True
    # args2 = copy.copy(args)
    # args2.sample_num_val = 0
    # args2.sample_rate_val = 16
    # args2.fix_seg_len_val = False
    # args2.batch_size = 1
    # DataSet1 = MPrpDataSet(root, dataset, anno_file, box_anno_file, args1.phase, args1.fix_seg_len, args1)
    # DataSet2 = MPrpDataSet(root, dataset, anno_file, box_anno_file, args2.phase, args2.fix_seg_len_val, args2)
    # sampler1 = SubsetSampler(DataSet1.provide_batch_spl_ind(args1.batch_size, shuffle=False))
    # sampler2 = SubsetSampler(DataSet2.provide_batch_spl_ind(args2.batch_size, shuffle=False))
    # batch_sampler1 = MPrpBatchSampler(sampler1, args1.batch_size, drop_last=False)
    # batch_sampler2 = MPrpBatchSampler(sampler2, args2.batch_size, drop_last=False)
    # data_loader1 = data.DataLoader(DataSet1, num_workers=args.workers, collate_fn=DataSet1.collate_fn,
        # batch_sampler=batch_sampler1)
    # data_loader2 = data.DataLoader(DataSet2, num_workers=args.workers, collate_fn=DataSet2.collate_fn,
        # batch_sampler=batch_sampler2)
    # data_iter1 = data_loader1.__iter__()
    # data_iter2 = data_loader2.__iter__()
    # t_s = time.time()
    # pdb.set_trace()
    # for i, (blobs, entities, entities_length, frm_length, rl_seg_inds, seg_nums, img_paths, img_ids) in enumerate(data_loader1):
        # t_e = time.time()
        # t_ave = (t_e - t_s)/(i+1)/args.batch_size
        # print ('time {:.3f}'.format(t_ave), blobs.shape, entities_length, frm_length, rl_seg_inds, seg_nums, entities, img_ids, img_paths)
        # seg_num = seg_nums[0]
        # for seg_i in range(seg_num):
            # blobs, entities, entities_length, frm_length, rl_seg_inds, seg_nums, img_paths, img_ids = data_iter2.__next__()
            # print ('time {:.3f}'.format(t_ave), blobs.shape, entities_length, frm_length, rl_seg_inds, seg_nums, entities, img_ids, img_paths)

    ##################################
    # test MPrpTransDataSet segment level #
    ##################################
    # DataSet = MPrpTransDataSet(root, dataset, anno_file, box_anno_file, args.phase, args.fix_seg_len, args)
    # data_loader = data.DataLoader(DataSet, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True, 
    #     collate_fn=DataSet.collate_fn)
    # t_s = time.time()
    # for i, (blobs, entities, ents_weight, entities_length, frm_length, rl_seg_inds, img_paths, img_ids) in enumerate(data_loader):
    #     t_e = time.time()
    #     t_ave = (t_e - t_s)/(i+1)/args.batch_size
    #     print ('time {:.3f}'.format(t_ave), blobs.shape, ents_weight, entities_length, frm_length, rl_seg_inds, entities, img_ids, img_paths)

    ####################
    # test MFrmDataSet #
    ####################
    # DataSet = MFrmDataSet(root, dataset, anno_file, box_anno_file, args.phase, args.fix_seg_len, args)
    # data_loader = data.DataLoader(DataSet, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True, 
        # collate_fn=DataSet.collate_fn)
    # t_s = time.time()
    # for i, (blobs, flos, seg_inds, seg_length, frm_inds, frm_length, img_paths, img_ids) in enumerate(data_loader):
        # t_e = time.time()
        # t_ave = (t_e - t_s)/(i+1)/args.batch_size
        # print ('time {:.3f}'.format(t_ave), blobs.shape, seg_inds, seg_length, frm_inds, frm_length, img_ids, img_paths)
        # print ('time {:.3f}'.format(t_ave), flos[0].shape, flos[1].shape)

    ######################
    ## show upper bound ##
    ######################
    # DataSet = MPrpDataSet(root, dataset, anno_file, box_anno_file, 'val', False, args)
    # sampler = SubsetSampler(DataSet.provide_batch_spl_ind(1, shuffle=False))
    # batch_sampler = MPrpBatchSampler(sampler, 1, drop_last=False)
    # data_loader = data.DataLoader(DataSet, num_workers=args.workers, collate_fn=DataSet.collate_fn,
    #     batch_sampler=batch_sampler)
    # class_list_path = os.path.join(root, dataset, 'annotations', class_file)
    # class_list = []
    # with open(class_list_path) as f:
    #     for line in f:
    #         class_list.append(line.rstrip('\n'))
    # # dets: lenght: image number, empty detection result will be None
    # dets = pickle.load(open(os.path.join('output', 'result', 'ground_res_ub.pkl'), 'rb'))
    # # recs: 
    # gt_cache_file = os.path.join('cache', dataset, 'gtbox_sample_{}.pkl'.format(0))
    # recs = pickle.load(open(gt_cache_file, 'rb'))
    # show_ub(data_loader, recs, dets, class_list)

    ###################
    ## show sentence ##
    ###################
    """
    args.entity_type = 'sentence_raw'
    DataSet = MPrpDataSet(root, dataset, anno_file, box_anno_file, 'val', False, args)
    sampler = SubsetSampler(DataSet.provide_batch_spl_ind(1, shuffle=False))
    batch_sampler = MPrpBatchSampler(sampler, 1, drop_last=False)
    data_loader = data.DataLoader(DataSet, num_workers=args.workers, collate_fn=DataSet.collate_fn,
        batch_sampler=batch_sampler)
    args.entity_type = 'category'
    DataSet_cate = MPrpDataSet(root, dataset, anno_file, box_anno_file, 'val', False, args)
    args.entity_type = 'sentence'
    DataSet_sen = MPrpDataSet(root, dataset, anno_file, box_anno_file, 'val', False, args)

    for i, (blobs, entities, entities_length, frm_length, rl_seg_inds, img_paths, img_ids) in enumerate(data_loader):
        cate = DataSet_cate[i][1]
        sentence = DataSet_sen[i][1]
        s = []
        if len(sentence) != len(entities):
            print (sentence, cate, entities)
        for ent_i, ent in enumerate(entities):
            if sentence[ent_i] in cate:
                ent = '[' + ent + ']'
            s.append(ent)
        s = ' '.join(s)
        if rl_seg_inds[0] == 0:
            ss = ''
        ss = ss + '{:2d} '.format(rl_seg_inds[0]) + s + '\n'
        if rl_seg_inds[0] + 1 == len(entities_length):
            save_path = img_paths[0].replace('frames', 'sentence')
            head = save_path.split('/')[:-1]
            tail = ['captions.txt']
            save_path = '/'.join(head + tail)
            save_dir = os.path.dirname(save_path)
            print (save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(save_path, 'w') as f:
                f.write(ss)
        print (entities_length, rl_seg_inds, s)
    """
    ###############################
    ## get word entity statistic ##
    ###############################
    """
    # 1. get the entity occurence proportion
    # 2. write to a file showing the video that contain each object.
    args.entity_type = 'category'
    DataSet = MPrpDataSet(root, dataset, anno_file, box_anno_file, 'val', False, args)
    sampler = SubsetSampler(DataSet.provide_batch_spl_ind(1, shuffle=False))
    batch_sampler = MPrpBatchSampler(sampler, 1, drop_last=False)
    data_loader = data.DataLoader(DataSet, num_workers=args.workers, collate_fn=DataSet.collate_fn,
        batch_sampler=batch_sampler)
    # get class file
    class_list_path = os.path.join(root, dataset, 'annotations', class_file)
    class_list = []
    with open(class_list_path) as f:
        for line in f:
            class_list.append(line.rstrip('\n'))
    cls_path_dic = {clss:[] for clss in class_list}
    cls_count = np.zeros(len(class_list))
    # each time load a segment.
    for i, (blobs, entities, entities_length, frm_length, rl_seg_inds, img_paths, img_ids) in enumerate(data_loader):
        for entity in entities:
            cls_path_dic[entity].append('/'.join(img_paths[0].split('/')[-4:-1]) + '/{:4d}'.format(rl_seg_inds[0]))
            cls_ind = class_list.index(entity)
            cls_count[cls_ind] += 1
    cls_pr = cls_count/np.sum(cls_count)
    cls_pr = [i for i in cls_pr]
    print (cls_pr)
    print ([i for i in cls_count])

    # content to write object file index
    cnt = ''
    for cls_name in cls_path_dic.keys():
        cls_line = cls_name + ':\n'
        cls_line += '\n'.join(cls_path_dic[cls_name])
        cls_line += '\n'
        cnt += cls_line
    save_file = os.path.join('output', root, dataset, 'stat', 'obj_search.txt')
    with open(save_file, 'w') as f:
        f.write(cnt)
    """


    ###############################
    ## get word entity statistic ##
    ###############################
    """
    # 1. get the entity occurence proportion
    # 2. write to a file showing the video that contain each object.
    DataSet = GroundDataSet(root, dataset, anno_file, box_anno_file, 'train', args)
    data_loader = data.DataLoader(DataSet, batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=DataSet.collate_fn)
    # get class file
    class_list_path = os.path.join(root, dataset, 'annotations', class_file)
    class_list = []
    with open(class_list_path) as f:
        for line in f:
            class_list.append(line.rstrip('\n'))
    cls_path_dic = {clss:[] for clss in class_list}
    cls_count = np.zeros(len(class_list))
    # each time load a segment.
    t_s = time.time()
    for i, (blobs, entities, entities_length, img_paths, img_ids) in enumerate(data_loader):
        t_e = time.time()
        t_ave = (t_e - t_s)/(i+1)
        print ('time {:.3f}'.format(t_ave), blobs.shape, entities_length, entities, img_ids, img_paths)
    for i, (blobs, entities, entities_length, img_paths, img_ids) in enumerate(data_loader):
        for entity in entities:
            cls_path_dic[entity].append('/'.join(img_paths[0].split('/')[-4:-1]))
            cls_ind = class_list.index(entity)
            cls_count[cls_ind] += 1
    cls_pr = cls_count/np.sum(cls_count)
    cls_pr = [i for i in cls_pr]
    print (cls_pr)
    print ([i for i in cls_count])

    # content to write object file index
    cnt = ''
    for cls_name in cls_path_dic.keys():
        cls_line = cls_name + ':\n'
        cls_line += '\n'.join(cls_path_dic[cls_name])
        cls_line += '\n'
        cnt += cls_line
    save_file = os.path.join('output', root, dataset, 'stat', 'obj_search_train.txt')
    with open(save_file, 'w') as f:
        f.write(cnt)
    """

    ###################################
    ## test RW dataloader
    ###################################
    anno_file = 'ent_anno.pkl'
    phase = 'val'
    loader = RWDataLoader(root, anno_file, phase, args)
    args.entity_type = ['noun', 'sentence_raw']
    for batch_ind, (im_blobs, entities, sents, entities_length, im_paths, img_ids) in enumerate(loader):
        print ('im_blob shape', im_blobs.shape, entities, sents, entities_length, im_paths, img_ids)
