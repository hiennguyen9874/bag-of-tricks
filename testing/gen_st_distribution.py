import os
import math
import torch
import copy
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from data import DataManger


def gen_st_distribution(logger, config, datamanager: DataManger):
    list_person_id, list_camera_id, list_frames = get_id(datamanager, config)

    num_camera = datamanager.datasource.get_num_camera('train')
    num_label = datamanager.datasource.get_num_classes('train')

    distribution = spatial_temporal_distribution(
        list_person_id, list_camera_id, list_frames, num_camera, num_label, config)

    print("Smooth histogram...")
    with tqdm(total=(distribution.shape[0] * distribution.shape[1])) as pbar:
        for i in range(0, distribution.shape[0]):
            for j in range(0, distribution.shape[1]):
                distribution[i][j][:] = gaussian_smooth(
                    distribution[i][j][:], variance=config['testing']['variance'])
                pbar.update(1)

    sum_ = np.sum(distribution, axis=2)
    for i in range(num_camera):
        for j in range(num_camera):
            distribution[i][j][:] = distribution[i][j][:] / \
                (sum_[i][j]+config['testing']['eps'])

    with open(os.path.join(config['testing']['ouput_dir'], 'distribution.pt'), 'wb') as f:
        torch.save(distribution, f)


def get_id(datamamager: DataManger, config):
    list_person_id = []
    list_camera_id = []
    list_frames = []

    data = copy.deepcopy(datamamager.datasource.get_data('train'))
    data = sorted(data, key=lambda x: (x[1], x[2], x[3]))

    for _, person_id, camera_id, frame in data:
        list_person_id.append(person_id)
        list_camera_id.append(camera_id)
        list_frames.append(frame)
    return list_person_id, list_camera_id, list_frames


def spatial_temporal_distribution(list_person_id, list_camera_id, list_frames, num_camera, num_label, config):
    class_num = num_label
    max_hist = config['testing']['max_hist']
    eps = config['testing']['eps']
    interval = config['testing']['interval']

    spatial_temporal_sum = np.zeros((class_num, num_camera))
    spatial_temporal_count = np.zeros((class_num, num_camera))

    for i in range(len(list_camera_id)):
        label_k = list_person_id[i]     # not in order, done
        cam_k = list_camera_id[i]-1     # from 1, not 0
        frame_k = list_frames[i]
        spatial_temporal_sum[label_k][cam_k] = spatial_temporal_sum[label_k][cam_k]+frame_k
        spatial_temporal_count[label_k][cam_k] = spatial_temporal_count[label_k][cam_k] + 1
    # spatial_temporal_avg: 751 ids, 8cameras, center point
    spatial_temporal_avg = spatial_temporal_sum/(spatial_temporal_count+eps)

    distribution = np.zeros((num_camera, num_camera, max_hist))
    for i in range(class_num):
        for j in range(num_camera-1):
            for k in range(j+1, num_camera):
                if spatial_temporal_count[i][j] == 0 or spatial_temporal_count[i][k] == 0:
                    continue
                st_ij = spatial_temporal_avg[i][j]
                st_ik = spatial_temporal_avg[i][k]
                if st_ij > st_ik:
                    diff = st_ij-st_ik
                    hist_ = int(diff/interval)
                    distribution[j][k][hist_] = distribution[j][k][hist_]+1
                else:
                    diff = st_ik-st_ij
                    hist_ = int(diff/interval)
                    distribution[k][j][hist_] = distribution[k][j][hist_]+1

    sum_ = np.sum(distribution, axis=2)
    for i in range(num_camera):
        for j in range(num_camera):
            distribution[i][j][:] = distribution[i][j][:]/(sum_[i][j]+eps)

    # [to][from], to xxx camera, from xxx camera
    return distribution


def gaussian_func(x, u, variance=50):
    temp1 = 1.0 / (variance * math.sqrt(2 * math.pi))
    temp2 = -(np.power(x - u, 2)) / (2 * np.power(variance, 2))
    return temp1 * np.exp(temp2)


def gaussian_smooth(arr, variance):
    hist_num = len(arr)
    vect = np.zeros((hist_num, 1))
    for i in range(hist_num):
        vect[i, 0] = i
    # gaussian_vect= gaussian_func2(vect,0,1)
    # variance=50
    # when x-u>approximate_delta, e.g., 6*variance, the gaussian value is approximately equal to 0.
    approximate_delta = 3*variance
    gaussian_vect = gaussian_func(vect, 0, variance)
    matrix = np.zeros((hist_num, hist_num))
    for i in range(hist_num):
        k = 0
        for j in range(i, hist_num):
            if k > approximate_delta:
                continue
            matrix[i][j] = gaussian_vect[j-i]
            k = k+1
    matrix = matrix+matrix.transpose()
    for i in range(hist_num):
        matrix[i][i] = matrix[i][i]/2
    # for i in range(hist_num):
    #     for j in range(i):
    #         matrix[i][j]=gaussian_vect[j]
    xxx = np.dot(matrix, arr)
    return xxx
