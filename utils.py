import sys
from scripts.evaluation_utils import load_model_by_name
import nomenclature
import matplotlib.pyplot as plt
import torch
import numpy as np
import argparse
import yaml
import glob
import os
import math
import wandb
import pandas as pd
from collections import deque
import collections

import cv2
import json
import imageio
import numpy as np
import random
import pickle
from acumen_test.constants import *
from acumen_test.sort import *

coco2openpose = np.array([
    [0, 0],
    [1, 14],
    [2, 15],
    [3, 16],
    [4, 17],
    [5, 2],
    [6, 5],
    [7, 3],
    [8, 6],
    [9, 4],
    [10, 7],
    [11, 8],
    [12, 11],
    [13, 9],
    [14, 12],
    [15, 10],
    [16, 13],
    [17, 1],
])


def normalise_data(poses):
    middle_hips = (poses[:, 12, :2] + poses[:, 11, :2]) / 2
    y_distance = np.sqrt(np.sum((poses[:, 0, :2] - middle_hips) ** 2, axis = 1))

    y_distance[y_distance == 0] = np.mean(y_distance)
    x_distance = y_distance / 2

    if np.any(np.isnan(x_distance)) or np.any(x_distance == 0):
        print(x_distance[x_distance == 0])
        print(x_distance)

    y_distance = y_distance.reshape((poses.shape[0], 1))
    x_distance = x_distance.reshape((poses.shape[0], 1))

    poses[:, :, :2] = poses[:, :, :2] - middle_hips.reshape((poses.shape[0], 1, 2))
    poses[:, :, 0] = poses[:, :, 0] / x_distance
    poses[:, :, 1] = poses[:, :, 1] / y_distance

    return poses


def pose2bbox(p, return_score = False):
    x, y, v = p.transpose()
    x1 = np.min(x[np.argwhere(v)]).astype(np.int32)
    x2 = np.max(x[np.argwhere(v)]).astype(np.int32)
    y1 = np.min(y[np.argwhere(v)]).astype(np.int32)
    y2 = np.max(y[np.argwhere(v)]).astype(np.int32)

    if not return_score:
        return x1, y1, x2, y2

    return x1, y1, x2, y2, np.mean(v)

def combine_pose_tracking(poses, track_bbs_ids):
    output = []

    for i, track_bb in enumerate(track_bbs_ids):
        ious = [iou(pose2bbox(pose), track_bb[:4]) for pose in poses]
        max_iou_idx = np.argmax(ious)
        tracked_pose = poses[max_iou_idx]
        output.append({
            'pose': tracked_pose,
            'track_id': track_bb[-1]
        })
    return output

def centre(data_2d):
    return (data_2d.T - data_2d.mean(1)).T

def centre_all(data):
    return (data.transpose(2, 0, 1) - data.mean(2)).transpose(1, 2, 0)

def normalise_data_old(d2, weights):
    d2 = d2.reshape(d2.shape[0], -1, 2).transpose(0, 2, 1)
    idx_consider = weights[0, :, 0].astype(np.bool)

    d2[:, :, idx_consider] = centre_all(d2[:, :, idx_consider])
    m2 = d2[:, 1, idx_consider].min(1) / 2.0
    m2 -= d2[:, 1, idx_consider].max(1) / 2.0
    crap = m2 == 0
    m2[crap] = 1.0
    d2[:, :, idx_consider] /= m2[:, np.newaxis, np.newaxis]
    return d2, m2

def draw_pose(frame, poses, pcolor = None, linecolor = None):
    l_pair = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (17, 11), (17, 12),  # Body
        (11, 13), (12, 14), (13, 15), (14, 16),
        (11, 17), (12, 17),
        (5, 18), (6, 18), (18, 17)
    ]

    p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
               (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
               (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
    line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                  (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                  (77, 222, 255), (255, 156, 127),
                  (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]

    if linecolor is not None:
        line_color = [linecolor for _ in range(len(l_pair))]

    if pcolor is not None:
        p_color = [pcolor for _ in range(19)]

    img = frame.copy()
    height, width = img.shape[:2]
    img = cv2.resize(img, (width // 2, height // 2))

    middle_hips = (poses[:, 12, :] + poses[:, 11, :]) / 2
    middle_hips = middle_hips.reshape((poses.shape[0], 1, 3))
    poses = np.hstack((poses, middle_hips))

    middle_shoulders = (poses[:, 5, :] + poses[:, 6, :]) / 2
    middle_shoulders = middle_shoulders.reshape((poses.shape[0], 1, 3))
    poses = np.hstack((poses, middle_shoulders))

    for human in poses:
        part_line = {}
        for n in range(human.shape[0]):
            cor_x, cor_y = int(human[n, 0]), int(human[n, 1])
            part_line[n] = (int(cor_x / 2), int(cor_y / 2))
            cv2.circle(img, (int(cor_x / 2), int(cor_y / 2)), 1, p_color[n], -1)

        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                bg = img.copy()

                X = (start_xy[0], end_xy[0])
                Y = (start_xy[1], end_xy[1])
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                stickwidth = (human[start_p, 2] + human[end_p, 2]) + 1
                transparency = max(0, min(1, 0.5 * (human[start_p, 2] + human[end_p, 2])))
                polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), int(stickwidth)), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(bg, polygon, line_color[i])
                img = cv2.addWeighted(bg, transparency, img, 1 - transparency, 0)

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    return img

def visualize(poses, repeat = True):
    while True:
        for i in range(len(poses)):
            canvas = np.zeros((270, 480, 3))
            for idx, (x, y, _) in enumerate(poses[i]):
                cv2.circle(canvas, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.imshow('winname', canvas)
            cv2.waitKey(12)

        if not repeat:
            break

def draw_hud(vis_frame, x1, y1, poses, model_outputs):
    cv2.putText(vis_frame, model_outputs['action']['name'], (int(x1) + 50, int(y1) - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (69, 90, 100), 3, cv2.LINE_AA)
    for pidx, p in enumerate(model_outputs['action']['probabilities']):
        size = int(p * 100)
        cv2.rectangle(vis_frame, (int(x1) + 50, int(y1) + 15 * pidx), (50 + int(x1) + size, int(y1) + 15 * (pidx + 1)), (69, 90, 100), -1)

    for pidx, p in enumerate(model_outputs['gait']['probabilities']):
        size = int(p * 100)
        color = (69, 90, 100) if not model_outputs['gait']['outliers'][pidx] else (35, 199, 172)
        cv2.rectangle(vis_frame, (int(x1) + 50, int(y1) + 130 + 15 * pidx), (50 + int(x1) + size, 130 + int(y1) + 15 * (pidx + 1)), color, -1)

    cv2.putText(vis_frame, model_outputs['gait']['name'], (int(x1) + 50, int(y1) - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (35, 199, 172), 3, cv2.LINE_AA)

    color = None
    color = (66, 126, 245) if model_outputs['gender']['name'] == 'Male' else (245, 66, 170)
    for pidx, p in enumerate(model_outputs['gender']['probabilities']):
        size = int(p * 100)
        cv2.rectangle(vis_frame, (int(x1) + 50, int(y1) + 70 + 15 * pidx), (50 + int(x1) + size, 70 + int(y1) + 15 * (pidx + 1)), (35, 199, 172), -1)

    # cv2.putText(vis_frame, model_outputs['gender']['name'], (int(x1) + 50, int(y1) - 150), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (35, 199, 172), 3, cv2.LINE_AA)

    vis_frame = draw_pose(vis_frame, [poses[-1]], color = color)

    return vis_frame
