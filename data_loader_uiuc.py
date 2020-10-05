import av
import torch
import numpy as np

from data_parser import WebmDataset
from data_augmentor import Augmentor
import torchvision
from transforms_video import *
from utils import save_images_for_debug


FRAMERATE = 12  # default value

import cv2
import toml
from PIL import Image

class UiucVideo(torchvision.datasets.VisionDataset):
    def __init__(self, root, transforms=None, transform=None, target_transform=None):
        super(UiucVideo, self).__init__(root, transforms, transform, target_transform)
        self.root = f'{root}/uiuc_combine_camera_action_dataset'
        label_path = f'{self.root}/0.toml'
        label_dist = toml.load(label_path)
        self.action_label_dist = label_dist['Action label']
        self.path_and_label = list(label_dist['Video label'].items())
        self.size = len(self.path_and_label)

    def __getitem__(self, index):
        video_name, label = self.path_and_label[index]
        video_path = f'{self.root}/{video_name}'
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError('No file in path. Check the `root` path.')
        frame_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video = []
        for _ in range(int(frame_number)):
            ok, image = cap.read()
            if not ok: 
                raise ValueError('Failed read video.')
            video.append(image)
        return np.array(video), self.action_label_dist[label]

    def __len__(self):
        return self.size

class UiucVideoV1(UiucVideo):
    def __init__(self, root, train=True, **kwargs):
        super(UiucVideoV1, self).__init__(root, **kwargs)
        partition = self.size * 4 // 5
        if train:
            self.offset = 0
            self.size = partition
        else:
            self.offset = partition
            self.size = self.size - partition

    def __getitem__(self, index):
        return super(UiucVideoV1, self).__getitem__(self.offset + index)

class VideoFolder(torch.utils.data.Dataset):

    def __init__(self, root, json_file_input, json_file_labels, clip_size,
                 nclips, step_size, is_val, transform_pre=None, transform_post=None,
                 augmentation_mappings_json=None, augmentation_types_todo=None,
                 get_item_id=False, is_test=False):
        self.uiuc = UiucVideoV1('/data-ssd1', not is_val)
        self.classes = list(range(len(list(self.uiuc.action_label_dist))))
        self.transform_pre = transform_pre
        self.transform_post = transform_post

        self.augmentor = Augmentor(augmentation_mappings_json,
                                   augmentation_types_todo)

        self.clip_size = clip_size
        self.nclips = nclips
        self.step_size = step_size
        self.is_val = is_val
        self.get_item_id = get_item_id

    def __getitem__(self, index):
        """
        [!] FPS jittering doesn't work with AV dataloader as of now
        """

        imgs, label = self.uiuc[index]

        imgs = [img for img in imgs]

        imgs = self.transform_pre(imgs)
        # imgs, label = self.augmentor(imgs, item.label)
        imgs = self.transform_post(imgs)

        num_frames = len(imgs)
        target_idx = label
        if self.nclips > -1:
            num_frames_necessary = self.clip_size * self.nclips * self.step_size
        else:
            num_frames_necessary = num_frames
        offset = 0
        if num_frames_necessary < num_frames:
            # If there are more frames, then sample starting offset.
            diff = (num_frames - num_frames_necessary)
            # temporal augmentation
            if not self.is_val:
                offset = np.random.randint(0, diff)

        imgs = imgs[offset: num_frames_necessary + offset: self.step_size]

        if len(imgs) < (self.clip_size * self.nclips):
            imgs.extend([imgs[-1]] *
                        ((self.clip_size * self.nclips) - len(imgs)))

        # format data to torch
        data = torch.stack(imgs)
        data = data.permute(1, 0, 2, 3)
        if self.get_item_id:
            return (data, target_idx, target_idx)
        else:
            return (data, target_idx)

    def __len__(self):
        return len(self.uiuc)


if __name__ == '__main__':
    upscale_size = int(84 * 1.1)
    transform_pre = ComposeMix([
            # [RandomRotationVideo(20), "vid"],
            [Scale(upscale_size), "img"],
            [RandomCropVideo(84), "vid"],
            # [RandomHorizontalFlipVideo(0), "vid"],
            # [RandomReverseTimeVideo(1), "vid"],
            # [torchvision.transforms.ToTensor(), "img"],
             ])
    # identity transform
    transform_post = ComposeMix([
                        [torchvision.transforms.ToTensor(), "img"],
                         ])

    loader = VideoFolder(root="/data-ssd1/20bn-something-something-v2/videos",
                         json_file_input="/data-ssd1/20bn-something-something-v2/annotations/something-something-v2-train.json",
                         json_file_labels="/data-ssd1/20bn-something-something-v2/annotations/something-something-v2-labels.json",
                         clip_size=36,
                         nclips=1,
                         step_size=1,
                         is_val=False,
                         transform_pre=transform_pre,
                         transform_post=transform_post,
                         # augmentation_mappings_json="notebooks/augmentation_mappings.json",
                         # augmentation_types_todo=["left/right", "left/right agnostic", "jitter_fps"],
                         )
    # fetch a sample
    # data_item, target_idx = loader[1]
    # save_images_for_debug("input_images_2", data_item.unsqueeze(0))
    # print("Label = {}".format(loader.classes_dict[target_idx]))

    import time
    from tqdm import tqdm

    batch_loader = torch.utils.data.DataLoader(
        loader,
        batch_size=10, shuffle=False,
        num_workers=8, pin_memory=True)

    start = time.time()
    for i, a in enumerate(tqdm(batch_loader)):
        if i > 100:
            break
        pass
    print("Size --> {}".format(a[0].size()))
    print(time.time() - start)
