#!/usr/bin/env python
import cv2
import time
import os
import numpy as np
import glob
import json
import argparse
from multiprocessing import Pool

def get_vidfiles(root):
    """
    get all video files from root directory (ex. training, testing)
    """
    clsdir = glob.glob('{}/*/*'.format(root))
    clsdir.sort()
    video_ids = []
    for vidfile in clsdir:
        video_id = vidfile.split("/")[-1].split(".")[0]
        video_ids.append(video_id)
    return clsdir, video_ids

def get_frame(vidfile, time_length, segs, sample_num=0):
    """
    :param vidfile: video file
    :param segs: video segment list: [(start, end)] (second)
    :param sample_num: sampling number for each segment. +num: fix sample num. 0: 1 fps. -1: 16 fps
    """
    if os.path.exists(vidfile):
        video = cv2.VideoCapture(vidfile)
    else:
        raise IOError
    vidsample = vidfile.replace('raw_videos', 'sampled_frames_splnum{}'.format(sample_num))
    # create file for the vidfile subset
    if not os.path.isdir(vidsample):
        os.makedirs(vidsample)

    video_id = vidsample.split("/")[-1].split(".")[0]

    # Start time
    start = time.time()
    # Grab a few frames
    frames = []
    # frames_segs[[frames of seg1], [frames of seg2]]
    frames_segs = []
    num_frames = 0
    print('start extract {} ...'.format(vidfile))
    reshape_size = (224, 224)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # resize
        frame = cv2.resize(frame, reshape_size)
        frames.append(frame)
        num_frames += 1
    if num_frames == 0:
        print ("unreadable video {}".format(video_id))
        with open('unreadable_video.txt', 'a') as f:
            f.write(vidsample + '\n')
    # subsample scheme is to sample the interger point, interval is at least 1s
    # subsample 5 frames per segment
    frames = np.array(frames)
    vidtime = time_length
    fps = num_frames/vidtime
    print('start sample {} ...'.format(video_id))

    # only can get the intergral points.
    for seg in segs:
        t_s = seg[0]
        t_e = seg[1]
        anno_frame_num = t_e - t_s
        # t_inds: bbox annotation frames time indexes
        t_inds = np.arange(t_s, t_e) + 0.5
        # t_sample: frame index of annotated frames
        t_samples = np.round(t_inds * fps).astype('int')
        if sample_num > 0:
            # f_inds: frame index of sampled frames within annotated frames
            f_inds = np.round(np.linspace(0, anno_frame_num-1, sample_num)).astype('int')
            # seg_inds: frame index of sampeld frames in video
            seg_inds = t_samples[f_inds]
        elif sample_num == 0:
            seg_inds = t_samples
        elif sample_num == -1:
            # dense sample, sample rate 16 fps
            t_inds = np.linspace(t_s, t_e, (t_e - t_s)*16)
            t_samples = np.round(t_inds * fps).astype('int')
            seg_inds = t_samples
        else:
            raise ("unimplemented sample num {}".format(sample_num))

        seg_inds = np.clip(seg_inds, 0, len(frames) - 1)
        frames_seg = frames[seg_inds]
        frames_segs.append(frames_seg)

    # save images for each segment with format videofilename/__(seg id)__(frame id).jpg
    for seg_id, frames_seg in enumerate(frames_segs):
        for frame_id, frame in enumerate(frames_seg):
            cv2.imwrite('{}/{:04d}{:06d}.jpg'.format(vidsample, seg_id, frame_id), frame)
    # End time
    end = time.time()
    # Time elapsed
    seconds = end - start
    print ("Time taken : {0} seconds".format(seconds))
    # Calculate frames per second
    fps = num_frames / seconds
    print ("Estimated frame parsing speed (fps): {0}".format(fps))
    # Release video
    video.release()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='parse YouCookII video into frames')
    parser.add_argument('--sample_num', type=int, default=-1,
                        help='''sampled frames number per video segment. choice from (n, 0, -1) 
                        n: sample n frame per segment;
                        0: sample at 1 fps; 
                        -1: sample at 16 fps''')
    parser.add_argument('--annopath', type=str, default='annotations/youcookii_annotations.json',
                        help='annotation file path')
    parser.add_argument('--video_dir', type=str,
                        help='the raw video directory')

    args = parser.parse_args()

    # phase: choose the phrase to parse. (train val test)
    phases = ['testing', 'validation', 'training']

    for phase in phases:
        # get annotation data
        annodata = json.load(open(args.annopath))
        # get video files path
        vidfiles, vid_ids = get_vidfiles(os.path.join(args.video_dir, phase))
        total_num = len(vid_ids)

        # change the start index
        start_vid = 0
        vidfiles = vidfiles[start_vid:]
        vid_ids = vid_ids[start_vid:]
        num_workers = 20
        # multithread build pool
        pool = Pool(num_workers)
        # init job list
        jobs = []
        # class vocabulary
        cls_voc = set()
        for vid_ind, (vidfile, vid_id) in enumerate(zip(vidfiles, vid_ids)):
            print('process {}, {}/{}'.format(vid_id, vid_ind+start_vid, total_num))

            segs = [annodata['database'][vid_id]['annotations'][i]['segment'] for i in range(len(annodata['database'][vid_id]['annotations']))]
            time_length = annodata['database'][vid_id]['duration']
            # append each job with same calling function into jobs
            jobs.append(
                pool.apply_async(get_frame,
                                 args=(vidfile, time_length, segs, args.sample_num)))
            # run
            if (vid_ind + 1) % num_workers == 0 or (vid_ind + 1) == len(vid_ids):
                res = [job.get() for job in jobs]
                pool.close()
                pool.join()
                pool = Pool(num_workers)

