import sys
import torch
import cv2
import numpy as np

from torch.autograd import Variable

# cannot import normally since the name contains a hyphen
sys.modules['grad_cam'] = __import__("grad-cam")
from grad_cam import *


class ModelOutputsVideo(ModelOutputs):
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(
            self.model.conv_column, target_layers)

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)

        # averaging features in time dimension
        output = output.mean(-1).mean(-1).mean(-1)

        output = self.model.clf_layers(output)
        return target_activations, output


class GradCamVideo(GradCam):
    def __init__(self, model, target_layer_names, class_dict, use_cuda,
                 input_spatial_size=224):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        self.input_spatial_size = input_spatial_size
        self.class_dict = class_dict
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputsVideo(self.model, target_layer_names)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        print("Predicted index chosen = {} ({})".format(index, self.class_dict[index]))

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.conv_column.zero_grad()
        self.model.clf_layers.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3, 4))[0, :]
        cam = np.ones(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :, :]

        cam = np.maximum(cam, 0)

        clip_size = input.size(2)
        step_size = clip_size // target.shape[1]

        cam_vid = []
        for i in range(cam.shape[0]):
            cam_map = cam[i, :, :]
            cam_map = cv2.resize(cam_map,
                                 (self.input_spatial_size, self.input_spatial_size))
            cam_vid.append(np.repeat(
                                np.expand_dims(cam_map, 0),
                                step_size,
                                axis=0)
                           )

        cam_vid = np.array(cam_vid)
        cam_vid = cam_vid - np.min(cam_vid)
        cam_vid = cam_vid / np.max(cam_vid)
        if cam_vid.shape[0] > 1:
            cam_vid = np.concatenate(cam_vid, axis=0)
        if cam_vid.shape[0] == 1:
            cam_vid = np.squeeze(cam_vid, 0)
        print("Shape of CAM mask produced = {}".format(cam_vid.shape))
        return cam_vid, output
