"""
Copyright 2022-2023 Zsolt Bedohazi, Andras Biricz, Oz Kilim, Istvan Csabai

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import glob
import os
import random
from importlib.resources import path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

# from genericpath import isfile
from openslide import OpenSlide


def get_slide_file_path(slide_id):
    #train
    data_dir = os.path.join('/','home','ngsci','datasets','brca-psj-path')
    # holdout
    #data_dir = os.path.join('/','home','ngsci','datasets','brca-psj-path', 'holdout')
    slide_dir = os.path.join(data_dir, 'ndpi')
    slide_fp = os.path.join(slide_dir,'*', f'{slide_id}.ndpi')
    return glob.glob(slide_fp)[0]


class PatchGeneratorPixel:
    """
    Class to generate smaller non-empty patches/tiles from a higher resolution image of a slide.
    The patch generation logic is the following:
        - taking a lower resolution image (level 8) a histogram of pixel values are calculated and
          based on the assumption that the most common value is the background
          a mask of non-background coordinates (nx2 matrix) is created
        - the higher resolution image is read in and reshaped to have the same indexing as the
          lower resolution image from which the mask was created
        - given number of random patches are drawn from the higher resolution image taking
          into account the coordinates from the mask
    """

    def __init__(self, slide_path, slide_level) -> None:
        self.slide_path = slide_path
        self.slide_dir_path = os.path.dirname(os.path.realpath(self.slide_path))
        self.slide_name = os.path.splitext(os.path.basename(self.slide_path))[0]
        self.slide_level = slide_level
        self.patch_dir_path = 'save_level7/'
        os.makedirs(self.patch_dir_path, exist_ok=True)
        
        #os.path.join(self.slide_dir_path, "..", f"patch_by_pixel_level_{self.slide_level}_resized_all")

    def get_slide_name_w_ext(self):
        return os.path.basename(self.slide_path)
    

    def read_lvl7_then_write(self):
        """
        Reads level 7 resolution image
        """
        slide_openslide = OpenSlide(self.slide_path)
        img_openslide = slide_openslide.read_region((0, 0), 7, slide_openslide.level_dimensions[7])
        img_openslide_RGB = img_openslide.convert("RGB")
        img_openslide_np = np.array(img_openslide_RGB)
        
        save_slide_folder = os.path.join(self.patch_dir_path, self.slide_name)
        
        np.save( save_slide_folder+'_level'+str(self.slide_level)+'.npy', img_openslide_np ) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--slide_path", type=str, required=True)
    parser.add_argument("--slide_level", type=int, required=True)
    
    parser.add_argument("--num_thread", type=int, required=False, default=0)
    parser.add_argument("--maxnum_threads", type=int, required=False, default=8)

    args = parser.parse_args()
    num = args.num_thread
    maxnum = args.maxnum_threads

    
    # TRAINING
    data_dir = os.path.join('/','home','ngsci','datasets','brca-psj-path')
    # HOLDOUT
    #data_dir = os.path.join('/','home','ngsci','datasets','brca-psj-path', 'holdout')
    
    slide_dir = os.path.join(data_dir, 'ndpi')
    slides_fp = os.path.join(slide_dir,'*','*')
    slides_list = [ j.split('/')[-1].split('.ndpi')[0] for j in sorted(glob.glob(slides_fp)) ]

    # goal: create slide_id : cancer stage mapping
    slide_biop_df = pd.read_csv(
        "/home/ngsci/datasets/brca-psj-path/v2/slide-biopsy-map.csv"
    )
    outcomes_df = pd.read_csv("/home/ngsci/datasets/brca-psj-path/v2/outcomes.csv")
    slide_stage_df = slide_biop_df.merge(outcomes_df, on="biopsy_id")

    # map cancer stage to 0 - 4:
    # outcomes_df["stage"].unique()
    #     ['IA', 'IIB', 'IIA', '0', nan, 'IIIC', 'IV', 'IIIA', 'IIIB', 'IB']
    def stage_to_int(stage):
        if stage == "0":
            return 0
        elif stage == "IA" or stage == "IB":
            return 1
        elif stage == "IIA" or stage == "IIB":
            return 2
        elif stage == "IIIA" or stage == "IIIB" or stage == "IIIC":
            return 3
        elif stage == "IV":
            return 4
        else:
            return np.nan


    slide_stage_df["stage"] = slide_stage_df["stage"].apply(stage_to_int)

    # subset columns, drop nans, reset index
    labels_df = (
        slide_stage_df[["slide_id", "biopsy_id", "stage"]]
        .copy()
        .dropna(how="any")
        .reset_index(drop=True)
    )
    labels_df["stage"] = labels_df["stage"].astype(int)

    sort_idx = np.argsort( labels_df.slide_id.values )
    labels_df = labels_df.loc[sort_idx]
    labels_df.reset_index(inplace=True, drop=True)

    slide_id_index_map = dict(zip(labels_df.slide_id.values, np.arange(labels_df.shape[0])))
    slide_id_index_map_reverse = dict(zip(np.arange(labels_df.shape[0]), labels_df.slide_id.values))

    slides_list_index = np.array([slide_id_index_map[ j ] for j in slides_list if j in slide_id_index_map.keys() ])

    slides_list_staged = np.array([slide_id_index_map_reverse[j] for j in slides_list_index])
    print( slides_list_staged.shape )
    
    # PROCESS FOR HOLDOUT:
    #slides_list_staged = np.array(slides_list)
    #print(slides_list_staged.shape)
    
    nr_of_slides_processed = 0
    
    idx_to_run = np.append(np.arange( 0, len(slides_list_staged), len(slides_list_staged)/maxnum  ).astype(int), len(slides_list_staged))
    idx_to_run_all = np.vstack( (idx_to_run[:-1], idx_to_run[1:]) ).T
    idx_to_run_now = idx_to_run_all[num]

    print( 'start:', idx_to_run_now[0], 'end:', idx_to_run_now[1], slides_list_staged[ idx_to_run_now[0]:idx_to_run_now[1] ].shape ) 
    
    for slide_path in slides_list_staged[ idx_to_run_now[0]:idx_to_run_now[1] ]:
        patch_generator = PatchGeneratorPixel(slide_path=get_slide_file_path(slide_path), slide_level=args.slide_level)
        patch_generator.read_lvl7_then_write()
        print('processed slides:', nr_of_slides_processed)
        nr_of_slides_processed += 1