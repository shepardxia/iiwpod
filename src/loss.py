from src.polygon_iou_loss import c_poly_diou_loss, batch_poly_diou_loss
import numpy as np
import torch

device = 'cuda'

def gen_loss(Ytrue, Ypred):
    return poly_loss(Ytrue, Ypred) + clas_loss(Ytrue, Ypred)


def l1(true, pred, szs):
	b,h,w,ch = szs
	res = torch.reshape(true-pred, (b,h*w*ch))
	res = torch.abs(res)
	res = torch.sum(res,1)
	return res

def poly_loss(Ytrue, Ypred):
    Ypred = Ypred.permute(0, 2, 3, 1)
    b = Ytrue.shape[0]
    h = Ytrue.shape[1]
    w = Ytrue.shape[2]

    obj_probs_true = Ytrue[..., 0]
    affine_pred = Ypred[..., 1:]
    pts_true = Ytrue[..., 1:]

    affinex = torch.stack([torch.max(affine_pred[...,0], torch.zeros(affine_pred[...,0].shape).to(device)), affine_pred[...,1], affine_pred[...,2]], 3)
    affiney = torch.stack([affine_pred[...,3], torch.max(affine_pred[...,4], torch.zeros(affine_pred[...,0].shape).to(device)), affine_pred[...,5]], 3)

    v = .5
    base = torch.tensor([[[[-v, -v, 1., v, -v, 1., v, v, 1., -v, v, 1.]]]]).to(device)
    base = base.repeat(b, h, w, 1)

    pts = torch.zeros((b, h, w, 0)).to(device)

    for i in range(0, 12, 3):
        row = base[..., i:(i+3)]
        ptsx = torch.sum(affinex * row, 3)
        ptsy = torch.sum(affiney * row, 3)

        pts_xy = torch.stack([ptsx, ptsy], 3)
        pts = torch.cat([pts, pts_xy], 3)


    idxs = torch.where(obj_probs_true==1)

    #loss = torch.zeros(Ypred.shape[0]).to(device)

    loss = batch_poly_diou_loss(pts[idxs].reshape(-1, 4, 2), pts_true[idxs].reshape(-1, 4, 2)) / b
    print('poly: ', loss)
    flags = torch.reshape(obj_probs_true, (b,h,w,1))
    res   =  1.0*l1(pts_true*flags, pts*flags, (b, h, w, 4*2))
    print('res', res)

    return loss + res


def logloss(Ptrue, Pred, szs, eps=10e-10):
	# batch size, height, width and channels
    b,h,w,ch = szs
    Pred = torch.clip(Pred, eps, 1.0 - eps)
    Pred = -torch.log(Pred)
    Pred = Pred*Ptrue
    Pred = torch.reshape(Pred, (b, h*w*ch))
    Pred = torch.sum(Pred, dim=1)
    return Pred

def clas_loss(Ytrue, Ypred): # classification loss only

    wtrue = 0.5
    wfalse = 0.5
    Ypred = Ypred.permute(0, 2, 3, 1)
    b = Ytrue.shape[0]
    h = Ytrue.shape[1]
    w = Ytrue.shape[2]

    obj_probs_true = Ytrue[...,0]
    obj_probs_pred = Ypred[...,0]

    non_obj_probs_true = 1. - Ytrue[...,0]
    non_obj_probs_pred = 1 - Ypred[...,0]

    res  = wtrue*logloss(obj_probs_true,obj_probs_pred,(b,h,w,1))
    res  += wfalse*logloss(non_obj_probs_true,non_obj_probs_pred,(b,h,w,1))
    return res