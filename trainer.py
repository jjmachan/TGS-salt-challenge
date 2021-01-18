import time
import os

import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp

from data import build_train_val_loaders
from model import init_model
from utils import timeSince, EarlyStopping


def get_iou_score(outputs, labels):
    A = labels.squeeze().bool()
    pred = torch.where(outputs < 0.,
                       torch.zeros_like(outputs),
                       torch.ones_like(outputs))
    B = pred.squeeze().bool()
    intersection = (A & B).float().sum((1, 2))
    union = (A | B).float().sum((1, 2))
    iou = (intersection + 1e-6)/(union + 1e-6)
    return iou


def train(model, x, y, loss_fn, optimizer, device):
    x, y = x.to(device), y.to(device)
    model.train()

    outputs = model(x)
    loss = loss_fn(outputs.squeeze(), y.bool().float())
    iou = get_iou_score(outputs, y).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), iou.item()


def validate(dataloader, model, loss_fn, device):
    model.eval()
    with torch.no_grad():
        running_iou = []
        running_loss = []
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out.squeeze(), y.bool().float())
            iou = get_iou_score(out, y).mean()
            running_iou.append(iou.item())
            running_loss.append(loss.item())
    return np.mean(running_loss), np.mean(running_iou)


def train_epochs(train_dataloader, val_dataloader, model,
                 loss_fn, num_epochs, save_path):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',
                                                           verbose=True)
    stopper = EarlyStopping(verbose=True,
                            path=os.path.join(save_path, 'unet_model_best.pth'),
                            patience=15, mode='max')
    steps = len(train_dataloader.dataset)//train_dataloader.batch_size
    best_model = model = model.to(device)

    start = time.time()
    train_losses = []
    train_ious = []
    val_losses = []
    val_ious = []
    for epoch in range(1, num_epochs+1):
        print('-'*10)
        print('Epoch {}/{}'.format(epoch, num_epochs))
        running_iou = []
        running_loss = []
        for step, (x, y) in enumerate(train_dataloader):
            loss, iou = train(model, x, y, loss_fn, optimizer, device)
            running_iou.append(iou)
            running_loss.append(loss)
            print('\r{:6.1f} %\tloss {:8.4f}\tIoU {:8.4f}'.format(100*(step+1)/steps, loss,iou), end = "")
        print('\r{:6.1f} %\tloss {:8.4f}\tIoU {:8.4f}\t{}'.format(100*(step+1)/steps,np.mean(running_loss),np.mean(running_iou), timeSince(start)))
        print('running validation...', end='\r')
        val_loss, val_iou = validate(val_dataloader, model, loss_fn, device)
        print('Validation: \tloss {:8.4f} \tIoU {:8.4f}'.format(val_loss, val_iou))
        scheduler.step(np.mean(running_iou))

        train_losses.append(loss)
        train_ious.append(iou)
        val_losses.append(val_loss)
        val_ious.append(val_iou)

        stopper(val_iou, model)
        if stopper.early_stop:
            break

    return (train_losses, val_losses), (train_ious, val_ious), best_model


if __name__ == '__main__':
    # Load the data
    batch_size = 128
    data_path = 'data/competition_data/'
    tloader, vloader = build_train_val_loaders(data_path, batch_size)

    # init model
    unet = init_model(from_scratch=True)

    # define loss
    cross_entropy_loss = nn.BCEWithLogitsLoss()
    lovasz_loss = smp.utils.losses.LovaszLoss(mode='binary')

    # Train Model with cross entropy loss first
    num_epochs = 30
    save_path = 'saved_models/'
    print('Training with CrossEntropy loss')
    losses, ious, best_model = train_epochs(tloader, vloader,
                                            unet, cross_entropy_loss,
                                            num_epochs, save_path)

    # then with lovasz_loss to get the maximum accuracy
    num_epochs = 70
    unet = init_model(best=True)
    print('Training with LovaszLoss')
    losses, ious, best_model = train_epochs(tloader, vloader,
                                            unet, lovasz_loss,
                                            num_epochs)
