import torchvision.models as models
import torch
import argparse
from ptflops import get_model_complexity_info

from models.Model.main_models.network import SLDDNet

with torch.cuda.device(0):
  # net = models.resnet18()


  net = SLDDNet()

  flops, params = get_model_complexity_info(net, (3, 256, 256), as_strings=True, print_per_layer_stat=True) #不用写batch_size大小，默认batch_size=1
  print('Flops:  ' + flops)
  print('Params: ' + params)
