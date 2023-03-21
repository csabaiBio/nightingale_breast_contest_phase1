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
import torch
from torch.nn import MSELoss
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import Feature_attention
import time
from torch import nn
import pickle

from torch.utils.tensorboard import SummaryWriter



class MIL_bag_loader(Dataset):

    def __init__(self, annotations_file):
        self.img_labels = pd.read_csv(annotations_file) 
        
        self.bag_all = [] 

        for k in tqdm( range( len(self.img_labels) ) ):
            whole_slide_name = str(self.img_labels.iloc[k, 1])
            bag_bracs = np.load("/home/ngsci/project/save_resnet_embeddings_level4_biopsy_bags_bracs_float16/"+ whole_slide_name + ".npy")
            

            self.bag_all.append(bag_bracs)
        
        print('DONE LOADING')
        print('Bag size: ', len(self.bag_all))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        try:
            # start = time.process_time()


            bag = self.bag_all[idx]
            
            # print(bag.shape)
            label = self.img_labels.iloc[idx, 2]

        except:
            print("no data")
            bag = np.zeros(shape=(100, 2048))
            label = np.nan

        return bag, label

num_epocs = 80

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu') #device config


print(device)

def trainLoop(train_annotations_file,test_annotations_file):

    # Load data.
    # Train dataloader.
    training_data = MIL_bag_loader(train_annotations_file)
    train_dataloader = DataLoader(training_data,batch_size=1, shuffle=True) 
    # Test dataloader.
    testing_data = MIL_bag_loader(test_annotations_file) #test non non transformed data?...
    test_dataloader = DataLoader(testing_data,batch_size=1, shuffle=True) 
    # Training ---------
    test_accuracy = []
    train_accuracy = []
    # Define model ------
    model = Feature_attention()
    model = nn.DataParallel(model,device_ids=[1])
    model = model.to(device)

    # defining the optimizer
    optimizer = optim.SGD(model.parameters(), lr=float(10**-3), momentum=0.9)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,6,10,15,20,25], gamma=0.5)
    
    criterion = MSELoss()
    # class_balence_list = [0.40415724, 0.33498047, 0.37513201, 0.47175224, 0.60097488]
    for epoch in tqdm(range(num_epocs)):
        # print(epoch)
        training_acc = []
        for step, (x, y) in enumerate(train_dataloader):  # gives batch data
            # start = time.process_time()
            # ------- Training --------
            if np.isnan(y):
                print('data not found...')

                loss = 0
            else:
                x = x.float().to(device)
                y = y.float().to(device)
                
                optimizer.zero_grad()
                
                outputs = model.module.forward(x)

                loss = criterion(outputs,y.float())
                
                # Backward pass
                loss.backward()
                
                optimizer.step()
              
                training_acc.append(loss.cpu().item())
        scheduler.step()
            

        # epoc training evaulation.
        training_acc = np.nanmean(np.array(training_acc))
        print("training MSE:",training_acc)
        train_accuracy.append(training_acc)

        # ------- Evaluations --------
        with torch.no_grad():
            testing_acc = []
            for step, (x_test, y_test) in enumerate(test_dataloader):

                outputs = model.module.forward(x_test.float().to(device))
                loss = criterion(outputs,y_test.float().to(device)) 
                testing_acc.append(loss.cpu().item())

            testing_acc = np.nanmean(np.array(testing_acc)) 
            print("testing MSE:",testing_acc)
            test_accuracy.append(testing_acc)


            if testing_acc < 0.57:
                print("model testing well!")

                PATH = f'./trained_models/biopsy_model_4_bracs_4_{np.round(testing_acc,5)}.pth'.replace('0.', '0_') # save the model
                torch.save(model, PATH)
            

    return test_accuracy , train_accuracy ,model



folds = [0,1,2,3,4]
folds = [0]
for fold in folds:

    train_annotations_file = './final_splits/train_biopsy_unbalenced.csv'  #Load split..
    test_annotations_file = './final_splits/test_biopsy_unbalenced.csv'

    num_repetes = 1
    for i in range(num_repetes):
        test_accuracy , train_accuracy ,model = trainLoop(train_annotations_file,test_annotations_file)
        

# save trained model 
PATH = './trained_models/biopsy_model_level_4_bracs_4_last.pth'
torch.save(model, PATH)

