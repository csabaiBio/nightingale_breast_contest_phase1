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
    #data_dir = os.path.join('/','home','ngsci','datasets','brca-psj-path')
    # holdout
    data_dir = os.path.join('/','home','ngsci','datasets','brca-psj-path', 'holdout')
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
        self.patch_dir_path = 'save_patches_level5_bags/'
        os.makedirs(self.patch_dir_path, exist_ok=True)
        
        #os.path.join(self.slide_dir_path, "..", f"patch_by_pixel_level_{self.slide_level}_resized_all")

    def get_slide_name_w_ext(self):
        return os.path.basename(self.slide_path)

    def _keep_patch(self, patch):
        """
        Saturation and intensity check on a patch based on function from:
        https://github.com/bsulyok/crc/blob/main/src/patch_creation.py
        to decide whether a patch contains enough data to be kept or not.

        Args:
            patch (np.ndarray): image patch
        Returns:
            (bool): True if patch can be kept, False if not.
        """
        intensity = patch.mean(axis=2)
        saturation = 1 - patch.min(axis=2) / (intensity + 1e-8)
        saturation_check = (saturation < 0.05).mean() <= 0.75
        intensity_check = (intensity > 245).mean() <= 0.75

        #return saturation_check & intensity_check
        return True

    def _reshape_split(self, image: np.ndarray, kernel_size: int):
        """
        Reshapes higher resolution image to have the same indexing
        as the lower resolution image from which the mask is created.

        Args:
            image (np.ndarray): higher resolution image
            kernel_size (int): width and height of patch
        Returns:
            tiled_array (np.array): higher resolution image reshaped
        """
        img_height, img_width, channels = image.shape
        tile_height = tile_width = kernel_size

        tiled_array = image.reshape(
            img_height // tile_height, tile_height, img_width // tile_width, tile_width, channels
        )
        tiled_array = tiled_array.swapaxes(1, 2)

        return tiled_array

    def calc_mask(self, patch_size):
        """
        Reads level 7 resolution image, calcualtes histogram of pixel values
        and creates a mask of non-background coordinates.

        Returns:
            mask (np.ndarray): coordinates of the non-background pixels
        """
        slide_openslide = OpenSlide(self.slide_path)
        img_openslide = slide_openslide.read_region((0, 0), 7, slide_openslide.level_dimensions[7])
        img_openslide_RGB = img_openslide.convert("RGB")
        img_openslide_np = np.array(img_openslide_RGB)

        # If slide_level is other than 0, level 8 image used for masking
        # need to be resized in order to make the indexing of the mask and the
        # higher resolution image match, after higher resolution image is cropped and reshaped.
        # Before resizing, the image need to be cropped also to keep the aspect ration when resizing.

        if self.slide_level != 0:
            #scale_factor = 2 ** self.slide_level * (patch_size / 256)
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

        value1, cnt1 = np.unique(img_openslide_np[:, :, 0], return_counts=True)
        mask = np.array(
            np.where(
                np.logical_or(
                    img_openslide_np[:, :, 0] < value1[np.argmax(cnt1)] - img_openslide_np.std() / 2,
                    img_openslide_np[:, :, 0] > value1[np.argmax(cnt1)] + img_openslide_np.std() / 2,
                )
            )
        ).T

        return mask, width, height
    
    def calc_mask_improved(self, patch_size):
        """
        Reads level 8 resolution image, calculates histogram of pixel values
        and creates a mask of non-background coordinates.

        Returns:
            mask (np.ndarray): coordinates of the non-background pixels
        """
        slide_openslide = OpenSlide(self.slide_path)
        img_openslide = slide_openslide.read_region((0, 0), 7, slide_openslide.level_dimensions[7])
        img_openslide_RGB = img_openslide.convert("RGB")
        img_openslide_np = np.array(img_openslide_RGB)

        # If slide_level is other than 0, level 8 image used for masking
        # need to be resized in order to make the indexing of the mask and the
        # higher resolution image match, after higher resolution image is cropped and reshaped.
        # Before resizing, the image need to be cropped also to keep the aspect ration when resizing.

        if self.slide_level != 0:
            #scale_factor = 2 ** self.slide_level * (patch_size / 256)
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
        


    def generate_patches(self, patch_size, nr_of_patches: None):
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

        print(mask.shape)

        # read higher resolution image (level 0) with tifffile,
        # other resolution can only be read with Openlisde and reshape it
        if self.slide_level == 0:
            slide_img = tifffile.imread(self.slide_path)
        else:
            slide_openslide = OpenSlide(self.slide_path)
            slide_img = slide_openslide.read_region(
                (0, 0), self.slide_level, slide_openslide.level_dimensions[self.slide_level]
            )
            slide_img = slide_img.convert("RGB")
            slide_img = np.array(slide_img)

            ## it not exists - not yet saved
            #if not os.path.exists(save_slide_folder):

            np.save( save_slide_folder+'_level'+str(self.slide_level)+'.npy', slide_img ) 

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

            # generate patches
            save_slide_folder = os.path.join(self.patch_dir_path, self.slide_name)
            print(save_slide_folder)
            #if not os.path.isdir( self.patch_dir_path):
            #    os.mkdir(self.patch_dir_path)
            os.makedirs(save_slide_folder, exist_ok=True)

            kept_nr = 0
            for idx in range(len(patch_list)):
                patch_tmp = patch_list[idx]

                # resize to 512x512
                #patch_tmp = cv2.resize(patch_tmp, (512, 512), interpolation=cv2.INTER_AREA)

                #if self._keep_patch(patch_tmp):
                savename = save_slide_folder+\
                        f"/{self.slide_name}_"+\
                        str(mask[idx][0])+\
                        "_" +\
                        str(mask[idx][1])+ ".jpg"

                #if not os.path.exists(savename): # if not exist write to disk
                kept_nr +=1
                plt.imsave(
                            savename,
                            patch_tmp,
                        )
                print(savename)
                #else:
                #    print('dropping patch due to cheking intensity and saturations ')

            return kept_nr

        #else:
        #     print('passed')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--slide_path", type=str, required=True)
    parser.add_argument("--patch_size", type=int, required=True)
    parser.add_argument("--nr_of_patches", type=int, required=False)
    parser.add_argument("--slide_level", type=int, required=True)
    
    parser.add_argument("--num_thread", type=int, required=False, default=0)
    parser.add_argument("--maxnum_threads", type=int, required=False, default=8)

    args = parser.parse_args()
    num = args.num_thread
    maxnum = args.maxnum_threads

    #    if args.patch_size % 256 != 0:
    #        raise Exception("--patch_size argument can only be 256 or integral multiple of 256!")

    
    # TRAINING
    #data_dir = os.path.join('/','home','ngsci','datasets','brca-psj-path')
    # HOLDOUT
    data_dir = os.path.join('/','home','ngsci','datasets','brca-psj-path', 'holdout')
    
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
    slides_list_staged = np.array(slides_list)
    #print(slides_list_staged.shape)
    
    nr_of_slides_processed = 0
    
    idx_to_run = np.append(np.arange( 0, len(slides_list_staged), len(slides_list_staged)/maxnum  ).astype(int), len(slides_list_staged))
    idx_to_run_all = np.vstack( (idx_to_run[:-1], idx_to_run[1:]) ).T
    idx_to_run_now = idx_to_run_all[num]

    print( 'start:', idx_to_run_now[0], 'end:', idx_to_run_now[1], slides_list_staged[ idx_to_run_now[0]:idx_to_run_now[1] ].shape ) 
    
    for slide_path in slides_list_staged[ idx_to_run_now[0]:idx_to_run_now[1] ]:
        #print( get_slide_file_path(slide_path) )
        patch_generator = PatchGeneratorPixel(slide_path=get_slide_file_path(slide_path), slide_level=args.slide_level)
        
        slide_name = patch_generator.get_slide_name_w_ext()
        patch_number = patch_generator.generate_patches(
            patch_size=args.patch_size, nr_of_patches=args.nr_of_patches
        )
        print(
            f"Generated {patch_number} patches from slide {slide_name} - [{nr_of_slides_processed}/{len(slides_list_staged[ idx_to_run_now[0]:idx_to_run_now[1] ])}]"
        )
        nr_of_slides_processed += 1
    
    
    """
    if os.path.isfile(args.slide_path):
        patch_generator = PatchGeneratorPixel(slide_path=args.slide_path, slide_level=args.slide_level)
        slide_name = patch_generator.get_slide_name_w_ext()
        patch_number = patch_generator.generate_patches(
            patch_size=args.patch_size, nr_of_patches=args.nr_of_patches
        )
        print(f"Generated {patch_number} patches from slide {slide_name}")
    
    elif os.path.isdir(args.slide_path):
        slide_path_list = sorted( glob.glob(args.slide_path + "/*.ndpi") )
        nr_of_slides_processed = 0
        for slide_path in slide_path_list[29+861+1:]:
            patch_generator = PatchGeneratorPixel(slide_path=slide_path, slide_level=args.slide_level)
            slide_name = patch_generator.get_slide_name_w_ext()
            patch_number = patch_generator.generate_patches(
                patch_size=args.patch_size, nr_of_patches=args.nr_of_patches
            )
            print(
                f"Generated {patch_number} patches from slide {slide_name} - [{nr_of_slides_processed}/{len(slide_path_list)}]"
            )
            nr_of_slides_processed += 1
    
    
    else:
        raise Exception(
            "--slide_path argument must either be a path to a slide or a directory containing slide!"
        )
    """