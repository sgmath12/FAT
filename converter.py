import torch
import torch.nn as nn
import pdb

class Converter(nn.Module):
    def __init__(self, encoder, mean, std):
        super(Converter, self).__init__()
        self.encoder = encoder
        self.mean = torch.tensor(mean).cuda().view(-1, 1, 1)
        self.std = torch.tensor(std).cuda().view(-1, 1, 1)
        self.class_proto = torch.zeros([10,512]).cuda()
        self.class_example_number = torch.zeros([10]).cuda()    


    def linear(self, feat):
        return self.encoder.linear(feat)

    def normalize(self, x):

        return (x - self.mean) / self.std

    def forward(self, x, feat = None):
        x = self.normalize(x)
        return self.encoder(x, feat)
 
    def extract_feature(self, x, only_feature = False):
        x = self.normalize(x)
        return self.encoder.extract_feature(x,only_feature)

    def forward_with_score(self,x,score,return_feat = True):
        x = self.normalize(x)
        return self.encoder.forward_with_score(x, score, return_feat)

    def forward_from_feature(self,x,layer = 0):
        return self.encoder.forward_from_feature(x, layer)
    
    