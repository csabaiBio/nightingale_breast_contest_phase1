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


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms


class Feature_attention(nn.Module):
    def __init__(self):
        super(Feature_attention, self).__init__()
        self.L = 2048
        self.I = 128
        self.D = 64 
        self.K = 1 

        self.feature_extractor = nn.Sequential(
            nn.Linear(self.L, self.L), 
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.L),
            nn.Dropout(0.6),
            nn.Linear(self.L, self.I), 
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.I),
            nn.Dropout(0.6),
            nn.Linear(self.I, self.I), 
            nn.LeakyReLU(),
            nn.Dropout(0.6),
            nn.BatchNorm1d(self.I)
        )

        self.attention = nn.Sequential(
            nn.Linear(self.I, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.I*self.K, self.I), 
            nn.LeakyReLU(),
            nn.Dropout(0.6),
            nn.Linear(self.I, 1), 
        )

    def forward(self, x):
        x = x.squeeze(0)

        x = self.feature_extractor(x)
        A = self.attention(x)
        A = torch.transpose(A, 1, 0) 
        A = F.softmax(A, dim=1) 
        M = torch.mm(A, x)
        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()
        return Y_prob[0]

