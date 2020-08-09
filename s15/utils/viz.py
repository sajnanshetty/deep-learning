import matplotlib.pyplot as plt
import torchvision


def image_show(tensors1):
    grid_tensor1 = torchvision.utils.make_grid(tensors1.detach().cpu())
    grid_image1 = grid_tensor1.permute(1, 2, 0)
    plt.figure(figsize=(50, 50))
    plt.imshow(grid_image1)
    plt.show()


def show_grid_images(fg_bg, mask_actual, mask_pred, depth_actual, depth_pred, strs="test_image"):
    grid_tensor_fg_bg = torchvision.utils.make_grid(fg_bg.detach().cpu(), padding=2, scale_each=True)
    grid_image0 = grid_tensor_fg_bg.permute(1, 2, 0)
    grid_tensor1 = torchvision.utils.make_grid(mask_actual.detach().cpu(), padding=2, scale_each=True)
    grid_image1 = grid_tensor1.permute(1, 2, 0)
    grid_tensor2 = torchvision.utils.make_grid(mask_pred.detach().cpu(), padding=2, scale_each=True)
    grid_image2 = grid_tensor2.permute(1, 2, 0)
    grid_tensor3 = torchvision.utils.make_grid(depth_actual.detach().cpu(), padding=2, scale_each=True)
    grid_image3 = grid_tensor3.permute(1, 2, 0)
    grid_tensor4 = torchvision.utils.make_grid(depth_pred.detach().cpu(), padding=2, scale_each=True)
    grid_image4 = grid_tensor4.permute(1, 2, 0)
    fig = plt.figure(figsize=(100, 100))
    plt.subplot(511)
    plt.xlabel('Foreground + Background input', fontsize=75, color='blue', fontweight="bold")
    plt.imshow(grid_image0)
    plt.subplot(512)
    plt.xlabel('Mask Actual', fontsize=75, color='blue', fontweight="bold")
    plt.imshow(grid_image1)
    plt.subplot(513)
    plt.xlabel('Mask Predicted', fontsize=75, color='red', fontweight="bold")
    plt.imshow(grid_image2)
    plt.subplot(514)
    plt.xlabel('Depth Actual', fontsize=75, color='blue', fontweight="bold")
    plt.imshow(grid_image3)
    plt.subplot(515)
    plt.xlabel('Depth Predicted', fontsize=75, color='red', fontweight="bold")
    plt.imshow(grid_image4)
    fig.subplots_adjust(wspace=0)
