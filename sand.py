import torch
from create_model import iiwpod
import numpy as np
import torch
import cv2
import src.polygon_iou_loss as p
from torchsummary import summary
from time import time

one = torch.rand((2, 4, 2)).to('cuda')
two = torch.rand((2, 4, 2)).to('cuda')


#t1 = time()
#for i in range(1000):
#    p.c_poly_diou_loss(one[0], two[0])
#    p.c_poly_diou_loss(one[1], two[1])
#t2 = time()
#print('took:', t2-t1)
#t3 = time()
#for i in range(1000):
#    p.batch_poly_diou_loss(one, two)
#t4 = time()
#print('took:', t4-t3)

print(p.torch_wn(one[0], two[0]))
print(p.torch_wn(one[1], two[1]))
print(p.batch_torch_wn(one, two))

#model = iiwpod().to('cuda')
#print(summary(model, (3, 128, 128), 3))