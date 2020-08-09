from tqdm import tqdm
import torch
from utils.helper import iou_score


class Test(object):

    def __init__(self):
        self.test_m_losses = []
        self.test_d_losses = []
        self.test_iou_mask = []
        self.test_iou_dense = []
        self.test_losses = []
        self.test_image = {}
        self.iou_dense_flg = 0

    def test(self, model, device, test_loader, criterion, epoch):
        model.eval()
        pbar = tqdm(iter(test_loader), dynamic_ncols=True)
        test_loader_len = len(test_loader)
        test_avg_loss = 0
        test_m_losses = 0
        test_iou_mask = 0
        test_d_losses = 0
        test_iou_dense = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(pbar):
                bg = data["bg"].to(device)
                fg_bg = data["fg_bg"].to(device)
                input_image = data["input_images"].to(device)
                fg_bg_mask = data["mask"].to(device)
                fg_bg_depth = data["dense"].to(device)
                mask_pred, depth_pred = model(input_image)
                mask_pred = torch.sigmoid(mask_pred)
                depth_pred = torch.sigmoid(depth_pred)
                loss_m = criterion(mask_pred, fg_bg_mask)
                loss_d = criterion(depth_pred, fg_bg_depth)

                loss = loss_m + loss_d

                iou_mask = iou_score(mask_pred.detach().cpu().numpy(), data["mask"].detach().cpu().numpy())
                iou_dense = iou_score(depth_pred.detach().cpu().numpy(), data["dense"].detach().cpu().numpy())

                test_avg_loss += loss
                test_m_losses += loss_m.item()
                test_iou_mask += iou_mask
                test_d_losses += loss_d.item()
                test_iou_dense += iou_dense
                if (batch_idx + 1) % test_loader_len == 0 and epoch % 3 == 0 and self.iou_dense_flg < test_iou_dense:
                    self.iou_dense_flg = test_iou_dense
                    test_image_update = {}
                    print("*******saving test batch images*********")
                    print(
                        f'Test Set: Batch idx = {batch_idx} |epoch : {epoch}, mask loss ={loss_m.item()} | dense depth losss ={loss_d.item()} | total loss = {loss} | iou Mask={iou_mask} |iou Dense depth={iou_dense}')
                    test_image_update.update({"fg_bg": fg_bg})
                    test_image_update.update({"mask_pred": mask_pred})
                    test_image_update.update({"mask_actual": fg_bg_mask})
                    test_image_update.update({"depth_pred": depth_pred})
                    test_image_update.update({"depth_actual": fg_bg_depth})
                    self.test_image.update({f'{epoch}_{batch_idx + 1}_I_{iou_dense:0.5f}': test_image_update})
                pbar.set_description(
                    desc=f'Test Set: Epoch : {epoch} | batch idx = {batch_idx} | mask loss ={loss_m.item()} | dense depth losss ={loss_d.item()} | total loss = {loss} | iou mask={iou_mask} |iou dense depth={iou_dense}')
            self.test_losses.append(test_avg_loss / test_loader_len)
            self.test_m_losses.append(test_m_losses / test_loader_len)
            self.test_d_losses.append(test_d_losses / test_loader_len)
            self.test_iou_dense.append(test_iou_dense / test_loader_len)
            self.test_iou_mask.append(test_iou_mask / test_loader_len)
            if self.iou_dense_flg == 0:
                self.iou_dense_flg = self.test_iou_dense[-1]
            print(
                f'Test Set: Epoch : {epoch} | avg mask loss ={self.test_m_losses[-1]} | avg dense depth losss ={self.test_d_losses[-1]} | avg total loss = {self.test_losses[0]} | avg iou mask={self.test_iou_mask[-1]} |avg iou dense depth={self.test_iou_dense[-1]}')
