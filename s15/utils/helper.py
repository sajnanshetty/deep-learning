import numpy as np
import torch
import json
import pickle


def get_stastics_map():
    stastics_map = {
        "bg": {"mean": [0.5558092594146729, 0.5201340913772583, 0.463156521320343],
               "std": [0.2149990200996399, 0.21596555411815643, 0.23049025237560272]},
        "fg_bg": {"mean": [0.5455222129821777, 0.5086212158203125, 0.45718181133270264],
                  "std": [0.22610004246234894, 0.2249932438135147, 0.23590309917926788]},
        "mask": {"mean": [0.05790501832962036], "std": [0.22068527340888977]},
        "dense": {"mean": [0.40361160039901733], "std": [0.19922664761543274]}
    }
    return stastics_map


def calculate_iou(target, prediction, thresh):
    intersection = np.logical_and(np.greater(target, thresh), np.greater(prediction, thresh))
    union = np.logical_or(np.greater(target, thresh), np.greater(prediction, thresh))
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def calculate_iou(target, prediction, thresh):
    intersection = np.logical_and(np.greater(target, thresh), np.greater(prediction, thresh))
    union = np.logical_or(np.greater(target, thresh), np.greater(prediction, thresh))
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def pickle_data(data_list, name):
    """
    :param data: list of item eg: test_image.pkl
    :param name: file name to pickle
    :return: None
    """
    with open(name, 'wb') as f:
        pickle.dump(data_list, f)


def read_pickle(pickle_name):
    """
    :param name: file name to pickle eg: test_image.pkl
    :return: list
    """
    with open(pickle_name, 'rb') as f:
        pickle_list = pickle.load(f)
    return pickle_list

