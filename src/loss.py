from src.polygon_iou_loss import c_poly_diou_loss
import numpy as np
import torch

def loss(Ytrue, Ypred):
    return poly_loss(Ytrue, Ypred) + clas_loss(Ytrue, Ypred)

def poly_loss(Ytrue, Ypred):
    Ypred = Ypred.permute(0, 2, 3, 1)
    b = Ytrue.shape[0]
    h = Ytrue.shape[1]
    w = Ytrue.shape[2]

    obj_probs_true = Ytrue[..., 0]
    affine_pred = Ypred[..., 1:]
    pts_true = Ytrue[..., 1:]

    affinex = torch.stack([torch.max(affine_pred[...,0], torch.zeros(affine_pred[...,0].shape)), affine_pred[...,1], affine_pred[...,2]], 3)
    affiney = torch.stack([affine_pred[...,3], torch.max(affine_pred[...,4], torch.zeros(affine_pred[...,0].shape)), affine_pred[...,5]], 3)

    v = .5
    base = torch.tensor([[[[-v, -v, 1., v, -v, 1., v, v, 1., -v, v, 1.]]]])
    base = base.repeat(b, h, w, 1)

    pts = torch.zeros((b, h, w, 0))

    for i in range(0, 12, 3):
        row = base[..., i:(i+3)]
        ptsx = torch.sum(affinex * row, 3)
        ptsy = torch.sum(affiney * row, 3)

        pts_xy = torch.stack([ptsx, ptsy], 3)
        pts = torch.cat([pts, pts_xy], 3)

    flags = torch.reshape(obj_probs_true, (b, h, w, 1))
    loss = 0
    
    bb, xx, yy, zz = torch.where(flags==1)
    for i in range(len(bb)):
        b, x, y = bb[i], xx[i], yy[i]
        loss += c_poly_diou_loss(pts[b, x, y, :].T.reshape(4, 2), pts_true[b, x, y, :].T.reshape(4, 2))
    # dimmax = 13
    #loss = 1.0 * c_poly_diou_loss(pts_true * flags, pts * flags)
    # /dimmax
    return loss


def logloss(Ptrue, Pred, szs, eps=10e-10):
	# batch size, height, width and channels
	b,h,w,ch = szs
	Pred = torch.clip_by_value(Pred, eps, 1.0 - eps)
	Pred = -torch.math.log(Pred)
	Pred = Pred*Ptrue
	Pred = torch.reshape(Pred, (b, h*w*ch))
	Pred = torch.reduce_sum(Pred,1)
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