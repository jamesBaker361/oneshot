import os
import sys
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir
import torch
torch.hub.set_dir("/scratch/jlb638/torch_hub_cache")
from PIL import Image
from PIL.ImageOps import pad
from datasets import load_dataset, Dataset
import cv2
import json
from pytube import YouTube
from pose_extraction import get_pose_pair
import argparse

parser = argparse.ArgumentParser(description="making our person data")

def extract_frames(video, start:float, end:float)->list:
    '''
    given video and two timestamps, gets first and last frames
    '''
    fps  = video.get(cv2.CAP_PROP_FPS)
    count=0
    success=1
    first=None
    last=None
    while success:
        success, image = video.read()
        count += 1
        second=count/fps
        if second>start and first is None:
            first=image
        if second >end:
            return first,image

    return first,last

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
    then we add start_frame, end_frame_pose end_frame and label to their lists
    then we return the 4 lists
    '''
    src_image_list,src_pose_list,target_image_list,label_list=[],[],[],[]
    yt=YouTube(clip_dict["url"])
    temp_path="tmp.mp4"
    stream=yt.streams.filter(file_extension='mp4')[0]
    stream.download(filename=temp_path)
    for anno in clip_dict["annotations"]:
        video = cv2.VideoCapture(temp_path)
        [start,end]=anno["segment"]
        first,last=extract_frames(video,start,end)
        if first is None or last is None:
            continue
        first=pad(Image.fromarray(cv2.cvtColor(first, cv2.COLOR_BGR2RGB)), (512,512))
        last=pad(Image.fromarray(cv2.cvtColor(last, cv2.COLOR_BGR2RGB)), (512,512))
        first_black, first_color,first_pred_boxes=get_pose_pair(first)
        if len(first_pred_boxes)==1 and last is not None:
            last_black,_,last_pred_boxes=get_pose_pair(last)
            if len(last_pred_boxes)==1:
                src_image_list.append(first)
                src_pose_list.append(last_black)
                target_image_list.append(last)
                label_list.append(anno["label"])
    return src_image_list,src_pose_list,target_image_list,label_list



def create_dataset(filepath:str,limit:int)-> Dataset:
    '''
    given filepath, parses json into object and processes each clip
    '''
    data_dict={
        "src_image":[],
        "src_pose":[],
        "target_image":[],
        "label":[]
    }
    with open(filepath, "r") as file:
        json_database=json.load(file)["database"]
    for key,clip_dict in json_database.items():
        print(key)
        #try:
        src_image_list,src_pose_list,target_image_list,label_list=process_clip_dict(clip_dict)
        '''print(f"\t{key} success")
        except Exception as err:
            print(f"\t{key} exception")
            print(err)'''
        data_dict["src_image"]+=src_image_list
        data_dict["src_pose"]+=src_pose_list
        data_dict["target_image"]+=target_image_list
        data_dict["label"]+=label_list
        if len(data_dict["src_image"])>limit:
            break

    return Dataset.from_dict(data_dict)

if __name__=='__main__':
    hf_dataset=create_dataset("hacs_segments.json",3)
    hf_dataset.push_to_hub("jlbaker361/hacs-segment-pairs")