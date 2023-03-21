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

import numpy as np
import os
import pandas as pd
import json
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import zlib
import bz2
import tensorflow as tf



parent_folder = '/home/ngsci/project/'
img_folder_path = parent_folder+'save_patches_level4_bags/'
resnet_emb_folder_path = parent_folder+'save_resnet_embeddings_level4_bags/'

os.makedirs(resnet_emb_folder_path, exist_ok=True)

img_bags = np.array( sorted( [ i for i in os.listdir(img_folder_path) if '.npy' in i ] ) )


backbone = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', pooling='max')
backbone.trainable = False

def get_model():
    inp = tf.keras.layers.Input(shape=(224, 224, 3))
    x = tf.keras.applications.resnet50.preprocess_input(inp)
    out = backbone(x)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    return model



if __name__ == '__main__':

    print(img_bags.shape)
    
    model = get_model()

    model.compile(loss=tf.keras.losses.MeanSquaredError(), metrics=['mse'], optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    
    
    errors = []
    for b in tqdm( range(img_bags.shape[0]) ):
        try:
            if not os.path.exists( resnet_emb_folder_path+'resnet50_'+img_bags[b] ):
                current_bag = np.load( img_folder_path+img_bags[b] ).astype(np.uint8)
                resnet_embedding = model.predict(x=current_bag, verbose=0)
                np.save(resnet_emb_folder_path+'resnet50_'+img_bags[b], resnet_embedding)
        except:
            print(b, current_bag.shape )
            errors.append( [b, current_bag.shape])
