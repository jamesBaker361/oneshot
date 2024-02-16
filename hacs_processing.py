import os
import sys
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir
import torch
torch.hub.set_dir("/scratch/jlb638/torch_hub_cache")
from PIL import Image
from datasets import load_dataset, Dataset

def extract_frames(video, start, end)->list:
    '''
    given video and two timestamps, gets a list of pil images for each second
    '''
    return []

def process_clip_dict(clip_dict: object)-> tuple:
    '''
    given clip_dict for example
    {'annotations': [
            {'label': 'Dodgeball','segment': [5.4, 11.6]},
            {'label': 'Dodgeball', 'segment': [12.6, 88.16]}],
        'duration': '92.166667',
        'subset': 'training',
        'url': 'https://www.youtube.com/watch?v=--0edUL8zmA'
    }
    downloads video, and for each segment in annotations, extracts the frames
    for that segment, and if first frame and last frame have the same person in different pose
    then we add start_frame, end_frame_pose and end_frame to their lists
    then we return the 3 lists
    '''
    return [],[],[]

def create_dataset(filepath:str)-> Dataset:
    '''
    given filepath, parses json into object and processes each clip
    '''
    
    return {}