import torch
from torchvision import datasets, transforms
from google.colab import drive
from torchsummary import summary
import sys
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, OneCycleLR
import torch.optim as optim
from models.resnet import *
import pickle


class HelperModel(object):

    def __init__(self, model):
        self.model = model

    @staticmethod
    def get_device():
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        return device

    def display_model_summay(self, model, input_image_size):
        summary(model, input_size=input_image_size)

    def get_optimizer(self, lr=0.01, momentum=0.9):
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        return optimizer

    def get_one_cycle_lr(self, lr=0.01, momentum=0.9, optimizer=None, max_lr=0.1, total_steps=20):
        if optimizer is None:
            optimizer = self.get_optimizer(lr=lr, momentum=momentum)
        scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps)
        return scheduler

    def get_step_optimizer(self, lr=0.01, momentum=0.9, step_size=1, gamma=0.1, optimizer=None):
        if optimizer is None:
            optimizer = self.get_optimizer(self.model, lr, momentum)
        scheduler = StepLR(optimizer, step_size, gamma)
        return scheduler

    def get_l2_regularizer(self, weight_decay=0.001, lr=0.01, momentum=0.9):
        l2_regularizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        return l2_regularizer

    @staticmethod
    def apply_l1_regularizer(model, loss, l1_factor=0.0005):
        reg_loss = 0
        parameters = model.parameters()
        for param in parameters:
            reg_loss += torch.sum(param.abs())
        loss += l1_factor * reg_loss
        return loss

    @staticmethod
    def calculate_output_size(input_channel_size, padding, kernel_size, stride=1, dilation=1):
        output_channel_size = ((input_channel_size + 2 * padding - kernel_size - (kernel_size - 1) * (
                dilation - 1)) / stride) + 1
        return output_channel_size

    @staticmethod
    def jump_out(jump_in, stride):
        """
        This calculates jump value of next layer based on current layer jump amd stride value
        :param jump_in: previous cov layer jump value. Default 1 for first input layer
        :param stride: stride of previous value
        :return: Returns next conv block jump value
        """
        return jump_in * stride

    @staticmethod
    def calculate_receptive_field(jump_in, kernel_size, receptive_field_in):
        """

         :param jump_in: Current jump in value of conv layer
         :param kernel_size: current kernel size of block
         :param receptive_field_in: Receptive field of previous conv block
         :return:
         """
        receptive_field_out = receptive_field_in + (kernel_size - 1) * jump_in
        return receptive_field_out

    @staticmethod
    def save_model(model, model_obj, optimizer, file_name):
        """

        :param model: model which needs to be saved
        :param file_name: the file name of model eg: 'model_aug.pth'
        :return: None
        """
        checkpoint = {'model': model_obj,
                      'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, file_name)

    @staticmethod
    def load_checkpoint(file_path):
        checkpoint = torch.load(file_path)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        for parameter in model.parameters():
            parameter.requires_grad = False
        return model

    @staticmethod
    def store_data(file_name, data):
        """
        :param file_name: name of the file
        :param data: python object like list, dict
        :return:
        """
        # initializing data to be stored in db

        # Its important to use binary mode
        dbfile = open(file_name, 'ab')

        # source, destination
        pickle.dump(data, dbfile)
        dbfile.close()

    @staticmethod
    def load_data(file_name):
        """
        :param file_name name to load
        :return:
        """
        dbfile = open(file_name, 'rb')
        db = pickle.load(dbfile)
        for keys in db:
            print(keys, '=>', db[keys])
        dbfile.close()
