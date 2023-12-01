import os
import cv2
import torch
import pandas as pd
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.io import read_video
from PIL import Image

class VideoDataset(Dataset):
    def __init__(self, label_path, root, frame_interval=1, video_len=89, transform=None):
        """
        Args:
            info_list (string): Path to the info list file with annotations.
            root_dir (string): Directory with all the video frames.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.frame_interval = frame_interval
        self.transform = transform
        self.label_path = label_path
        self.samples = self._load_video_samples(root, self.label_path)
        self.video_len = video_len
        
        
    def __len__(self):
        return len(self.samples)
    
    
    def _load_video_samples(self, root, label_path):
        samples = []
        with open(label_path, 'r') as f:
            for line in f:
                video_name, target = line.strip().split()  # Split the line into video_name and target
                video_name = video_name[:-4]
                video_path = os.path.join(root, video_name)
                samples.append((video_path, int(target)))
        return samples
    

    def _extract_frames(self, video_path):
        frames = []

        for image_filename in os.listdir(video_path):
            image_path = os.path.join(video_path, image_filename)
            img = Image.open(image_path)
            frames.append(img)
            
        actual_length = len(frames)
            
        if actual_length > self.video_len:
            frames = frames[0:self.video_len]
        # elif actual_length < self.video_len:
        #     # 创建一个空帧（全黑帧）来进行填充
        #     blank_frame = Image.new("RGB", frames[0].size, (0, 0, 0))
            
        #     # 计算需要填充的帧数
        #     frames_to_fill = self.video_len - actual_length
            
        #     # 将空帧添加到frames列表中，以达到desired_length
        #     for _ in range(frames_to_fill):
        #         frames.append(blank_frame)

        return frames
    
    
    def __getitem__(self, index):
        """
            return each video's frames and label
        """
        video_path, target = self.samples[index]
        
        frames = self._extract_frames(video_path)

        if self.transform is not None:
            # Apply the defined transformations to each frame
            frames = [self.transform(frame) for frame in frames]
        
        samples = torch.stack(frames)
        #samples = frames
        #samples = frames[:5]
        return samples, target
    

