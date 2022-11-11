import torch

import numpy as np
import torch.nn.functional as F

""" Esto lo saqué del código de JuanPablo
JuanPablo Heredia (juan1t0 github)
"""
class FusionTransformer(object):
  def __init__(self, modename):
    self.mode = modename

  def __call__(self, sample):
    facedata, audiodata, textdata = sample['face'], sample['audio'], sample['text']
    label, avs, name = sample['label'], sample['availabilities'], sample['name']

    facedata = torch.flatten(torch.from_numpy(facedata))
    audiodata = F.softmax(torch.from_numpy(audiodata),dim=-1)
    textdata = torch.from_numpy(textdata)
    avs = torch.from_numpy(avs)
    label = np.asarray(label)

    return {'face': facedata.float(),
            'audio': audiodata.float(),
            'text': textdata.float(),
            'label': torch.from_numpy(label).long(),
            'availabilities': avs.float(),
            'name': name}