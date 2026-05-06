import numpy as np
import multiprocessing
import os
import sys
from itertools import product
from collections import OrderedDict
from lib.test.evaluation import Sequence, Tracker
import torch


def _save_tracker_output(seq: Sequence, tracker: Tracker, output: dict):
    """Saves the output of the tracker."""

    if not os.path.exists(tracker.results_dir):
        print("create tracking result dir:", tracker.results_dir)
        os.makedirs(tracker.results_dir)
    # if seq.dataset in ['trackingnet', 'got10k']:
    if not os.path.exists(os.path.join(tracker.results_dir, seq.dataset)):
        os.makedirs(os.path.join(tracker.results_dir, seq.dataset))
    '''2021.1.5 create new folder for these two datasets'''
    # if seq.dataset in ['trackingnet', 'got10k']:
    base_results_path = os.path.join(tracker.results_dir, seq.dataset, seq.name)
    # else:
    #     base_results_path = os.path.join(tracker.results_dir, seq.name)

    def save_bb(file, data):
        tracked_bb = np.array(data).astype(int)
        np.savetxt(file, tracked_bb, delimiter='\t', fmt='%d')

    def save_time(file, data):
        exec_times = np.array(data).astype(float)
        np.savetxt(file, exec_times, delimiter='\t', fmt='%f')

    def save_score(file, data):
        scores = np.array(data).astype(float)
        np.savetxt(file, scores, delimiter='\t', fmt='%.2f')

    def _convert_dict(input_dict):
        data_dict = {}
        for elem in input_dict:
            for k, v in elem.items():
                if k in data_dict.keys():
                    data_dict[k].append(v)
                else:
                    data_dict[k] = [v, ]
        return data_dict

    for key, data in output.items():
        # If data is empty
        try:
            if not data:
                continue
        except:
            import pdb
            pdb.set_trace()

        if key == 'target_bbox':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}.txt'.format(base_results_path)
                save_bb(bbox_file, data)

        if key == 'all_boxes':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_all_boxes.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}_all_boxes.txt'.format(base_results_path)
                save_bb(bbox_file, data)

        if key == 'all_scores':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_all_scores.txt'.format(base_results_path, obj_id)
                    save_score(bbox_file, d)
            else:
                # Single-object mode
                print("saving scores...")
                bbox_file = '{}_all_scores.txt'.format(base_results_path)
                save_score(bbox_file, data)

        elif key == 'time':
            if isinstance(data[0], dict):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    timings_file = '{}_{}_time.txt'.format(base_results_path, obj_id)
                    save_time(timings_file, d)
            else:
                timings_file = '{}_time.txt'.format(base_results_path)
                save_time(timings_file, data)
        
        elif key == 'reliablity':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_reliablity.txt'.format(base_results_path, obj_id)
                    save_score(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}_reliablity.txt'.format(base_results_path)
                save_score(bbox_file, data)
        elif key == 'iou':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_iou.txt'.format(base_results_path, obj_id)
                    save_score(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}_iou.txt'.format(base_results_path)
                save_score(bbox_file, data)
        elif key == 'iou_rgb':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_iou_rgb.txt'.format(base_results_path, obj_id)
                    save_score(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}_iou_rgb.txt'.format(base_results_path)
                save_score(bbox_file, data)
        elif key == 'iou_tir':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_iou_tir.txt'.format(base_results_path, obj_id)
                    save_score(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}_iou_tir.txt'.format(base_results_path)
                save_score(bbox_file, data)


def run_sequence(seq: Sequence, tracker: Tracker, debug=False, num_gpu=8):
    """Runs a tracker on a sequence."""
    '''2021.1.2 Add multiple gpu support'''
    def overlap_ratio(rect1, rect2):
        '''
        Compute overlap ratio between two rects
        - rect: 1d array of [x,y,w,h] or
                2d array of N x [x,y,w,h]
        '''
    
        if rect1.ndim==1:
            rect1 = rect1[None,:]
        if rect2.ndim==1:
            rect2 = rect2[None,:]
    
        left = np.maximum(rect1[:,0], rect2[:,0])
        right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
        top = np.maximum(rect1[:,1], rect2[:,1])
        bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])
    
        intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
        union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
        iou = np.clip(intersect / union, 0, 1)
        return iou
    
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    # def _results_exist():
    #     if seq.object_ids is None:
    #         if seq.dataset in ['trackingnet', 'got10k']:
    #             base_results_path = os.path.join(tracker.results_dir, seq.dataset, seq.name)
    #             bbox_file = '{}.txt'.format(base_results_path)
    #         else:
    #             bbox_file = '{}/{}.txt'.format(tracker.results_dir, seq.name)
    #         return os.path.isfile(bbox_file)
    #     else:
    #         bbox_files = ['{}/{}_{}.txt'.format(tracker.results_dir, seq.name, obj_id) for obj_id in seq.object_ids]
    #         missing = [not os.path.isfile(f) for f in bbox_files]
    #         return sum(missing) == 0

    # if _results_exist() and not debug:
    #     print('FPS: {}'.format(-1))
    #     return

    print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))
  
    if debug:
        output = tracker.run_sequence(seq, debug=debug, run_id=tracker.run_id, name=tracker.parameter_name)
    else:
        # try:
        output = tracker.run_sequence(seq, debug=debug, run_id=tracker.run_id, name=tracker.parameter_name)
        # except Exception as e:
        #     print(e)
        #     return
    try:
        iou = overlap_ratio(np.array(seq.ground_truth_rect), np.array(output['target_bbox']))
    except:
        iou = np.zeros(1)
    ave_iou = iou.sum()/len(iou)
    sys.stdout.flush()
    output.update({'iou': iou.tolist()})
    # import pdb
    # pdb.set_trace()
    # iou rgb and tie head
    try:
        iou_rgb = overlap_ratio(np.array(seq.ground_truth_rect), np.array(output['target_bbox_rgb']))
    except:
        iou_rgb = np.zeros(1)
    ave_iou_rgb = iou_rgb.sum()/len(iou_rgb)
    output.update({'iou_rgb': iou_rgb.tolist()})
    try:
        iou_tir = overlap_ratio(np.array(seq.ground_truth_rect), np.array(output['target_bbox_tir']))
    except:
        iou_tir = np.zeros(1)
    ave_iou_tir = iou_tir.sum()/len(iou_tir)
    output.update({'iou_tir': iou_tir.tolist()})
    
    if isinstance(output['time'][0], (dict, OrderedDict)):
        exec_time = sum([sum(times.values()) for times in output['time']])
        num_frames = len(output['time'])
    else:
        exec_time = sum(output['time'])
        num_frames = len(output['time'])

    # print('FPS: {}'.format(num_frames / exec_time))

    if not debug:
        _save_tracker_output(seq, tracker, output)
        
    return iou.tolist(), ave_iou, num_frames / exec_time


def run_dataset(dataset, trackers, debug=False, threads=0, num_gpus=8):
    """Runs a list of trackers on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        trackers: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
    """
    multiprocessing.set_start_method('spawn', force=True)

    print('Evaluating {:4d} trackers on {:5d} sequences'.format(len(trackers), len(dataset)))

    multiprocessing.set_start_method('spawn', force=True)

    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'

    if mode == 'sequential':
        iou_list = []
        num_sqe = 0
        for seq in dataset:
            for tracker_info in trackers:
                num_sqe = num_sqe + 1
                # if num_sqe>=201:
                iou, ave_iou, fps = run_sequence(seq, tracker_info, debug=debug)
                iou_list = iou_list + iou
                print('Tracker: {} {} {}, {}-th Sequence: {}, IOU: {}, FPS: {}, total mIoU: {}'.format(tracker_info.name, tracker_info.parameter_name, tracker_info.run_id, num_sqe, seq.name, ave_iou, fps, sum(iou_list)/len(iou_list)))
                # else:
                #     print('exist!')
    elif mode == 'parallel':
        param_list = [(seq, tracker_info, debug, num_gpus) for seq, tracker_info in product(dataset, trackers)]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_sequence, param_list)
    print('Done')
