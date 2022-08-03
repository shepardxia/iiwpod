import torch
from create_model import iiwpod
import numpy as np
import torch
import cv2
import src.polygon_iou_loss as p
from torchsummary import summary
from time import time

one = torch.rand((100, 4, 2)).to('cuda')
two = torch.rand((100, 4, 2)).to('cuda')
img = np.zeros((500, 500, 3))


res = torch.zeros((100)).to('cuda')
torch.cuda.synchronize()
t1 = time()
cnt = 0
for j in range((int)(100 / one.shape[0])):
    for i in range((int)(one.shape[0])):
        cnt += 1
        res[i] = p.c_poly_diou_loss(one[i], two[i])
        #print(res[i])
print(res.mean())
torch.cuda.synchronize()
t2 = time()
print('took:', t2-t1)
torch.cuda.synchronize()
t3 = time()
res2 = None
for i in range((int)(100 / one.shape[0])):
    res2 = p.batch_poly_diou_loss(one, two)
print(res2.mean())
torch.cuda.synchronize()
t4 = time()
print('took:', t4-t3)

#print(p.batch_poly_diou_loss(one, two))
