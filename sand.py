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


#one = torch.tensor(([[[0.3751, 0.7144],
#         [0.2090, 0.8723],
#         [0.2006, 0.6045],
#         [0.2051, 0.0682]],
#
#        [[0.8102, 0.1022],
#         [0.7816, 0.2165],
#         [0.4685, 0.7746],
#         [0.5083, 0.4449]]])).to('cuda')
#
#two = torch.tensor(([[[0.7131, 0.5863],
#         [0.9131, 0.9257],
#         [0.3605, 0.7825],
#         [0.5891, 0.6713]],
#
#        [[0.8986, 0.5698],
#         [0.5792, 0.6181],
#         [0.0998, 0.4444],
#         [0.3099, 0.2412]]])).to('cuda')


#zero = torch.zeros((one.shape[0], 4, 2)).to('cuda')
#jim = torch.cat((one, zero), dim=1)
#print(jim)
#comb = jim.abs().mean(dim=-1)[..., None]
#
#max = torch.max(comb, dim=1, keepdim=True)[0]
##head = torch.arange(0, one.shape[0]).to('cuda').T.reshape(-1, 1)
##ind = torch.cat((head, max), dim=1)
##print(ind)
##print(ind.shape)
##print(torch.where(max!=0))
#
#max = max.repeat(1, 8, 1)
#ind = torch.where((comb-max).squeeze(-1)==0)
#
##tim = torch.zeros((one.shape[0], 1, 2)).to('cuda')
#
#alt = jim[ind].reshape(one.shape[0], 1, 2)
#alt = alt.repeat(1, 8, 1)
#
#
#idxs = torch.where(comb.squeeze(-1)==0)
#
#jim[idxs] = alt[idxs]
#print(jim)
#print(torch.where())

#sam = p.batch_clockify(one)
#tom = p.batch_clockify(one.repeat(1, 2 ,1))
#
#print(p.batch_poly_area(sam))
#print(p.batch_poly_area(tom))

res = torch.zeros((100)).to('cuda')
t1 = time()
cnt = 0
for j in range((int)(1000 / one.shape[0])):
    for i in range((int)(one.shape[0])):
        cnt += 1
        res[i] = p.c_poly_diou_loss(one[i], two[i])
        #print(res[i])
print(res.mean())
t2 = time()
print('took:', t2-t1)
t3 = time()
res2 = None
for i in range((int)(1000 / one.shape[0])):
    res2 = p.batch_poly_diou_loss(one, two)
print(res2.mean())

t4 = time()
print('took:', t4-t3)
#clock1 = torch.tensor(([[0.2191, 0.4742],
#        [0.1364, 0.6929],
#        [0.0387, 0.5048],
#        [0.1079, 0.2309]]))
#clock2 = torch.tensor(([[0.4978, 0.3740],
#        [0.6710, 0.4145],
#        [0.0745, 0.6083],
#        [0.0380, 0.4924]]))
#clock3 = torch.tensor(([[0.5862, 0.0775],
#        [0.8610, 0.7860],
#        [0.3452, 0.3129],
#        [0.2930, 0.0840]]))
#clock4 = torch.tensor(([[0.9543, 0.2611],
#        [0.7737, 0.6423],
#        [0.2148, 0.7907],
#        [0.2271, 0.7413]]))
#
#one = torch.cat([clock1.unsqueeze(0), clock2.unsqueeze(0)], dim=0)
#two = torch.cat([clock3.unsqueeze(0), clock4.unsqueeze(0)], dim=0)

#mock = p.clockify(one[0])
#print(mock)
#cv2.fillPoly(img, [(clock1 * 200).to('cpu').numpy().astype(int)], (255, 0, 0))
#cv2.fillPoly(img, [(mock * 200).to('cpu').numpy().astype(int)], (0, 255, 0))

#cv2.imshow('img', img)
#cv2.waitKey(0)

#print(p.torch_wn(one[0], two[0]))
#print(p.torch_wn(one[1], two[1]))
#print(p.batch_torch_wn(one, two))

#one = torch.arange(0, 16, 1).reshape(2, 4, 2)
#empty = torch.zeros((2, 8, 2)).type(one.dtype)
#
#print(one)
#sam = one[...,0]
#print(sam)
#check = sam % 3
#print(check)
#nz = torch.where(check!=0)
#print(nz)
#
#empty[nz] = one[nz]
#print(empty)

#print(one[sam, tom, nz])


#model = iiwpod().to('cuda')
#print(summary(model, (3, 128, 128), 3))