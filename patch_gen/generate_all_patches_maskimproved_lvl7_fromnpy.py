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
from openslide import OpenSlide
from tqdm import tqdm

class PatchGeneratorPixel:
    def __init__(self, slide_path, slide_level) -> None:
        self.slide_path = slide_path
        self.slide_dir_path = os.path.dirname(os.path.realpath(self.slide_path))
        self.slide_name = os.path.splitext(os.path.basename(self.slide_path))[0].split('_level')[0]
        self.slide_level = slide_level
        self.patch_dir_path = 'save_patches_level4_bags/'#_errored/'
        os.makedirs(self.patch_dir_path, exist_ok=True)

    def _keep_patch(self, patch):

        intensity = patch.mean(axis=2)
        saturation = 1 - patch.min(axis=2) / (intensity + 1e-8)
        saturation_check = (saturation < 0.05).mean() <= 0.75
        intensity_check = (intensity > 245).mean() <= 0.75

        #return saturation_check & intensity_check
        return True

    def _reshape_split(self, image: np.ndarray, kernel_size: int):
        img_height, img_width, channels = image.shape
        tile_height = tile_width = kernel_size

        tiled_array = image.reshape(
            img_height // tile_height, tile_height, img_width // tile_width, tile_width, channels
        )
        tiled_array = tiled_array.swapaxes(1, 2)

        return tiled_array


    def calc_mask_improved(self, patch_size):
        img_openslide_np = np.load('/home/ngsci/project/save_level7_npy/' + self.slide_name + '_level7.npy')

        if self.slide_level != 0:
            scale_factor = 2 ** self.slide_level * (patch_size / 128)

            if patch_size % 128 != 0 and not float.is_integer(scale_factor):
                img_openslide_np = img_openslide_np[
                    : int(np.floor((img_openslide_np.shape[0] // scale_factor) * scale_factor)),
                    : int(np.floor((img_openslide_np.shape[1] // scale_factor) * scale_factor)),
                    :,
                ]
            else:
                img_openslide_np = img_openslide_np[
                    : int((img_openslide_np.shape[0] // scale_factor) * scale_factor),
                    : int((img_openslide_np.shape[1] // scale_factor) * scale_factor),
                    :,
                ]
            width, height = (
                int(img_openslide_np.shape[1] // scale_factor),
                int(img_openslide_np.shape[0] // scale_factor),
            )
            img_openslide_np = cv2.resize(
                img_openslide_np, (width, height), interpolation=cv2.INTER_AREA
            )
        
        current_image = img_openslide_np
        white_start=235
        
    # which color channel to choose

        percentiles = np.zeros((3,99))
        perc_limits = np.ones(3, dtype=int)*-1


        for ch in range(percentiles.shape[0]):
            color_hist = np.cumsum(np.histogram(current_image[:,:,ch].flatten(), bins=np.arange(0,257), density=True)[0])

            for p in range(percentiles.shape[1]):

                perc_threshold = np.arange(256)[color_hist < p*0.01 + 0.01]
                if len(perc_threshold) > 0:
                    percentiles[ch, p] = np.max(perc_threshold)


            perc_filter = percentiles[ch, :] < white_start

            if np.sum(perc_filter) >0:
                perc_limits[ch] = np.argmax(percentiles[ch, np.arange(percentiles.shape[1])[ percentiles[ch, :] < white_start ] ])

        logical_filters = []

        mask_img = np.zeros((current_image.shape[0], current_image.shape[1]))


        if (perc_limits != -1).sum():

            for i in np.arange(3)[perc_limits != -1]:
                logical_filters.append(np.logical_and(current_image[:,:,i], current_image[:, :, i] < percentiles[i, perc_limits[i]] ))

            if len(logical_filters) == 1:
                mask = np.array(np.where(logical_filters[0])).T
                mask_img[mask[:,0], mask[:,1]] = 1


            if len(logical_filters) == 2:
                mask = np.array(np.where(np.logical_or(logical_filters[0], logical_filters[1]))).T
                mask_img[mask[:,0], mask[:,1]] = 1

            if len(logical_filters) == 3:
                mask = np.array(np.where(np.logical_or(np.logical_or(logical_filters[0], logical_filters[1]) , logical_filters[2] ))).T
                mask_img[mask[:,0], mask[:,1]] = 1


            ## Use "opening" morphological operation for clearing some small dots (noise).
            #mask_img_refined = cv2.morphologyEx(
            #    mask_img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
            #)
            # close some remaining gaps in the mask
            #mask_img_refined = cv2.morphologyEx(
            #    mask_img_refined, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
            #)

            #return mask_img_refined

            return mask, width, height

        else:
            return np.argwhere( np.isclose(np.ones((current_image.shape[0], current_image.shape[1])), 1.) ) , width, height
        


    def generate_patches(self, patch_size, level, nr_of_patches: None):
        """
        Generate given number of small patches from the slide with given size.

        Args:
            patch_size (int): width and height of patches
            nr_of_patches (int): number of patches to generate, if not passed all patches will be generated
        """

        # generate mask for the slide
        
        # path 
        save_slide_folder = os.path.join(self.patch_dir_path, self.slide_name)
        
        ## do something only if npy file does not exist
        #if not os.path.exists( save_slide_folder+'_level'+str(self.slide_level)+'.npy' ):

        mask, mask_width, mask_height = self.calc_mask_improved(patch_size=patch_size)

        #print(mask.shape)

        # read higher resolution image (level 0) with tifffile,
        # other resolution can only be read with Openlisde and reshape it
        slide_img = np.load('/home/ngsci/project/save_level'+str(level)+'_npy/' + self.slide_name + '_level'+str(level)+'.npy')

        # if slide_level is other than 0 so that width and/or height of higher resolution image
        # cannot be divided by the patch size without a reminder then the higher resolution image need to be cropped
        if patch_size % 128 != 0:
            slide_img = slide_img[
                : int(patch_size * mask_height),
                : int(patch_size * mask_width),
                :,
            ]

        else:
            slide_img = slide_img[
                : (slide_img.shape[0] // patch_size) * patch_size,
                : (slide_img.shape[1] // patch_size) * patch_size,
                :,
            ]

        slide_img_reshaped = self._reshape_split(image=slide_img, kernel_size=patch_size)

        # select random nr_of_patches coordinates from the mask and create list of patches
        random.seed(425)
        # can only draw given nr of patches if there is at least that much, otherwise
        # generate maximum nr of available patches
        if nr_of_patches is not None and len(mask) > nr_of_patches:
            rand_idx = random.sample(range(1, len(mask)), nr_of_patches)
            mask = mask[rand_idx]

        #print(mask.shape, mask)
        patch_list = [slide_img_reshaped[i, j] for [i, j] in mask]
        #print(len(patch_list), patch_list[0])
        np.save( os.path.join( self.patch_dir_path, self.slide_name), np.array(patch_list, dtype=np.uint8) ) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", type=int, required=True)
    parser.add_argument("--nr_of_patches", type=int, required=False)
    parser.add_argument("--slide_level", type=int, required=True)
    
    parser.add_argument("--num_thread", type=int, required=False, default=0)
    parser.add_argument("--maxnum_threads", type=int, required=False, default=8)

    args = parser.parse_args()
    num = args.num_thread
    maxnum = args.maxnum_threads

    data_dir = '/home/ngsci/project/save_level'+str(args.slide_level)+'_npy/'
    
    slides_list = np.array( sorted([ data_dir+i for i in os.listdir(data_dir) if '.npy' in i ]) )
    print('Number of files found:', slides_list.shape, slides_list[:3])
    
    nr_of_slides_processed = 0
    
    idx_to_run = np.append(np.arange( 0, len(slides_list), len(slides_list)/maxnum  ).astype(int), len(slides_list))
    idx_to_run_all = np.vstack( (idx_to_run[:-1], idx_to_run[1:]) ).T
    idx_to_run_now = idx_to_run_all[num]

    print( 'start:', idx_to_run_now[0], 'end:', idx_to_run_now[1], slides_list[ idx_to_run_now[0]:idx_to_run_now[1] ].shape ) 
    
    for slide_path in tqdm( slides_list[ idx_to_run_now[0]:idx_to_run_now[1] ] ):
    #for slide_path in tqdm( slides_list[ 4045:4046 ] ): 
    ### correcting run: 
    #for slide_path in tqdm( slides_list[ 4044+800:idx_to_run_now[1]+1 ] ): 
        
        
        patch_generator = PatchGeneratorPixel(slide_path=slide_path, slide_level=args.slide_level)
        

        
        patch_number = patch_generator.generate_patches(
            patch_size=args.patch_size, level=args.slide_level,  nr_of_patches=args.nr_of_patches
        )
        
        
        #print(
        #    f"Generated {patch_number} patches from slide {slide_name} - [{nr_of_slides_processed}/{len(slides_list[ idx_to_run_now[0]:idx_to_run_now[1] ])}]"
        #)
        
        #if nr_of_slides_processed >  10:
        #    break
        nr_of_slides_processed += 1
        