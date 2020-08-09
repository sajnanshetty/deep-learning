from tqdm.notebook import tqdm
import torch
import numpy as np
from utils.helper import iou_score

class Train(object):
    def __init__(self):
        self.train_m_losses = []
        self.train_d_losses = []
        self.train_iou_mask = []
        self.train_iou_dense = []
        self.train_losses = []
        self.iou_dense_flg = 0

    def train(self, model, device, train_loader, optimizer, criterion, epoch, scheduler=None, save_model=True):
        model.train()
        pbar = tqdm(iter(train_loader), dynamic_ncols=True)
        avg_loss = 0
        train_m_losses = 0
        train_iou_mask = 0
        train_d_losses = 0
        train_iou_dense = 0
        train_loader_len = len(train_loader)
        for batch_idx, data in enumerate(pbar):
            bg = data["bg"].type(torch.FloatTensor).to(device)
            fg_bg = data["fg_bg"].type(torch.FloatTensor).to(device)
            input_image = data["input_images"].type(torch.FloatTensor).to(device)
            fg_bg_mask = data["mask"].type(torch.FloatTensor).to(device)
            fg_bg_depth = data["dense"].type(torch.FloatTensor).to(device)

            # Init
            optimizer.zero_grad()  #Reset the gradients
            mask_pred, depth_pred = model(input_image)
            mask_pred = torch.sigmoid(mask_pred)
            depth_pred = torch.sigmoid(depth_pred)
            # Calculate loss
            loss_m = criterion(mask_pred, fg_bg_mask)
            loss_d = criterion(depth_pred, fg_bg_depth)
            loss = loss_m + loss_d
            avg_loss +=loss
            self.train_losses.append(loss.item())
            # Backpropagation
            loss.backward()  #gradients calculated for each parameters
            optimizer.step()
            iou_mask = iou_score(mask_pred.detach().cpu().numpy(), data["mask"].detach().cpu().numpy())
            iou_dense = iou_score(depth_pred.detach().cpu().numpy(), data["dense"].detach().cpu().numpy())

            train_m_losses += loss_m.item()
            train_iou_mask += iou_mask
            train_d_losses += loss_d.item()
            train_iou_dense += iou_dense
            if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
            if save_model and (batch_idx + 1) % 250 == 0 and epoch % 3 == 0 and self.iou_dense_flg < train_iou_dense:
                self.iou_dense_flg = train_iou_dense
                print("*******saving model*********")
                torch.save(model.state_dict(), f"saved_model/E_{epoch:02d}_B_{batch_idx:05d}_ID_{iou_dense:0.5f}.pth")


            pbar.set_description(
                 desc=f'Train Set: Epoch : {epoch} | batch_id={batch_idx} | mask loss ={loss_m.item()} | dense depth losss ={loss_d.item()} | total loss = {loss}  | iou mask={iou_mask} |iou dense depth={iou_dense}' )
        self.train_losses.append(avg_loss/train_loader_len)
        self.train_m_losses.append(train_m_losses/train_loader_len)
        self.train_d_losses.append(train_d_losses/train_loader_len)
        self.train_iou_dense.append(train_iou_dense/train_loader_len)
        self.train_iou_mask.append(train_iou_mask/train_loader_len)
        if self.iou_dense_flg == 0:
            self.iou_dense_flg = self.train_iou_dense[-1]
        print(f'Train Set: Epoch : {epoch} | avg mask loss ={self.train_m_losses[-1]} | avg dense depth losss ={self.train_d_losses[-1]} | avg total loss = {self.train_losses[0]} | avg iou mask={self.train_iou_mask[-1]} |avg iou dense depth={self.train_iou_dense[-1]}')