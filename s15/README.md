### Objective
```
Given an image with foreground objects and background image, predict the depth map as well as a mask for the foreground object.
```
### Dataset preparation and statistics
https://github.com/sajnanshetty/deep-learning/blob/master/s14_s15A/README.md

### Notations Used
```
    fg - Foreground 
    bg - Background
    fg_bg - Foreground images overlayed on background images
    mask - Mask 
    dense depth - Depth map
```

### Data Description
A custom dataset  will be used to train this model refer below link for more detail:
[https://github.com/sajnanshetty/deep-learning/blob/master/s14_s15A/README.md](https://github.com/sajnanshetty/deep-learning/blob/master/s14_s15A/README.md)
```
* 100 RGB background images
* 400k RGB foreground overplayed on background images
* 400k gray scale masks images which is created from fg_bg.
* 400k gray scale dense depth images which is created from fg_bg.
```
[Dataset Link]: (https://drive.google.com/drive/folders/1UQpDsJrcUR4PYDo9vEWeWvG5XGormwTk?usp=sharing)

### Model and Architecture
Customized implementation of the [U-Net](https://arxiv.org/abs/1505.04597) in PyTorch for predicting mask and dense images.
* Unet Model architecture:
![Unet](https://github.com/sajnanshetty/deep-learning/blob/master/s15/images/unet-architecture.png)

* Customised Unet architecture to produce 2 targets
![Unet_custon](https://github.com/sajnanshetty/deep-learning/blob/master/s15/images/custom_unet.PNG)

* Input of the model : The concatenated image of bg and fg_bg images
```
bg : (3 x image_size x image_size) + fg_bg: (3 x image_size x image_size) = (6, image_size, image_size)
```
* Customised approach of getting 2 targets using above Unet architecture
```
Image size change: 64 -->  32 --> 16 --> 8 -->  4 --> 8  -->  16 -->  32 --> 64 --> 64
                                                |
                                                |  
                                                |             
                      |--> 64 --> 128 -->256 -->| --> 256 --> 128 --> 64 --> 32 --> 1
                      |                         |                            
channel change: 6-32->|                        512
                      |                         |
                      |--> 64 --> 128-->256 --> | --> 256 --> 128 --> 64 --> 32 --> 1                  |
                      
```

* The Targets of the model are mask and dense depth.
    mask_pred : (1 x image_size x image_size)
    dense_depth_pred : (1 x image_size x image_size)
    [Model file]: (link)
```
class Unet(nn.Module):
    def __init__(self, num_channels=6, num_classes=1, bilinear=True):
        super(Unet, self).__init__()

        factor = 2 if bilinear else 1

        self.num_classes = num_classes
        self.num_channels = num_channels

        # encoder
        self.down_conv_1 = double_conv(self.num_channels, 32)

        # ***********Depth convoluton part*****************
        self.down2_d = Down(32, 64)
        self.down3_d = Down(64, 128)
        self.down4_d = Down(128, 256)
        self.down5_d = Down(256, 512 // factor)

        self.up1_d = Up(512, 256 // factor, bilinear)
        self.up2_d = Up(256, 128 // factor, bilinear)
        self.up3_d = Up(128, 64 // factor, bilinear)
        self.up4_d = Up(64, 32, bilinear)

        self.out_d = nn.Conv2d(
            in_channels=32,
            out_channels=self.num_classes,
            kernel_size=1
        )

        # ***********Mask convoluton part*****************
        self.down2_m = Down(32, 64)
        self.down3_m = Down(64, 128)
        self.down4_m = Down(128, 256)
        self.down5_m = Down(256, 512 // factor)

        self.up1_m = Up(512, 256 // factor, bilinear)
        self.up2_m = Up(256, 128 // factor, bilinear)
        self.up3_m = Up(128, 64 // factor, bilinear)
        self.up4_m = Up(64, 32, bilinear)

        self.out_m = nn.Conv2d(
            in_channels=32,
            out_channels=self.num_classes,
            kernel_size=1
        )

    def forward(self, image):
        # bs, c, h, w
        # encoding starts
        image = image
        x1 = self.down_conv_1(image)  #

        # *************depth starts*******
        # encoder
        x2 = self.down2_d(x1)
        x3 = self.down3_d(x2)
        x4 = self.down4_d(x3)
        x5 = self.down5_d(x4)

        # decoder
        x_d = self.up1_d(x5, x4)
        x_d = self.up2_d(x_d, x3)
        x_d = self.up3_d(x_d, x2)
        x_d = self.up4_d(x_d, x1)
        dense_output = self.out_d(x_d)

        # *************mask starts*******
        # encoder
        x2 = self.down2_m(x1)
        x3 = self.down3_m(x2)
        x4 = self.down4_m(x3)
        x5 = self.down5_m(x4)

        # decoder
        x_m = self.up1_m(x5, x4)
        x_m = self.up2_m(x_m, x3)
        x_m = self.up3_m(x_m, x2)
        x_m = self.up4_m(x_m, x1)
        mask_output = self.out_m(x_m)
        return dense_output, mask_output
```
* Model is built in such a way that it can take any size and number of classes as input.
* In our case we are passing below input:
    * The model performs encoding and decoding the input image
    * takes 6 channel as input(fg_bg + bg)
    * then each unet down sampling and up sampling is follows 2 block.One for mask and one for dense depth.
    * the model returns 2 target images i.e mask and dense depth with each one channel.

### Model Summary
![model_summary](https://github.com/sajnanshetty/deep-learning/blob/master/s15/images/model_summary.PNG)

### Custom data set
* Facilities to pass bg numbers, which returns corresponding fg_bg's, mask and dense depth images.
Which is flexible enough to train any fg_bg overplayed images.
```
class CustomDataset(Dataset):
    def __init__(self, image_path, start=1, end=100):
        self.bg_image_list = []
        self.fg_bg_image_list = []
        self.dense_image_list = []
        self.mask_image_list = []
        for bg_number in range(start, end + 1):
            index_no = (bg_number - 1) * 4000 + 1
            for count in range(1, 4001):
                sub_image_path = f'bg_{bg_number:03d}/{index_no}.jpg'
                self.bg_image_list.append(os.path.join(image_path, "background", f'{bg_number}.jpg'))
                self.fg_bg_image_list.append(os.path.join(image_path, "fg_bg", sub_image_path))
                self.dense_image_list.append(os.path.join(image_path, "fg_bg_dense_depth", sub_image_path))
                self.mask_image_list.append(os.path.join(image_path, "fg_bg_mask", sub_image_path))
                index_no += 1

    def __getitem__(self, idx):
        fg_bg = self.fg_bg_image_list[idx]
        mask = self.mask_image_list[idx]
        dense = self.dense_image_list[idx]
        bg = self.bg_image_list[idx]
        return {"fg_bg": fg_bg, "bg": bg, "mask": mask, "dense": dense}

    def __len__(self):
        return len(self.fg_bg_image_list)


def apply_transform(image_type, resize_image=128, transform=None):
    image_type = get_stastics_map()[image_type]
    mean = image_type["mean"]
    std = image_type["std"]
    apply_transform = [
        transforms.Resize((resize_image, resize_image)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),

    ]
    if transform:
        transform = transform + apply_transform
    transform = apply_transform
    return transforms.Compose(transform)


class SubDatasetTransform(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, custmadataset, resize_image=128, transform=None):
        """
        Args:
            data (string): zipped images and labels.
        """
        self.custmset = custmadataset
        self.bg_transform = apply_transform('bg', resize_image)
        self.fg_bg_transform = apply_transform('fg_bg', resize_image)
        self.mask_transform = apply_transform('mask', resize_image)
        self.dense_transform = apply_transform('dense', resize_image)

    def __len__(self):
        return len(self.custmset)

    def __getitem__(self, idx):
        image_map = self.custmset[idx]
        bg = Image.open(image_map["bg"])
        fg_bg = Image.open(image_map["fg_bg"])
        mask = Image.open(image_map["mask"]).convert("L")
        dense = Image.open(image_map["dense"]).convert("L")
        bg_transformed = self.bg_transform(bg)
        fg_bg_transformed = self.fg_bg_transform(fg_bg)
        mask_transformed = self.mask_transform(mask)
        dense_transformed = self.dense_transform(dense)
        input_images = torch.cat((bg_transformed, fg_bg_transformed), dim=0)
        return {"input_images": input_images, "bg": bg_transformed, "fg_bg": fg_bg_transformed,
                "mask": mask_transformed, "dense": dense_transformed}
```
### Summary:
* Optimizer: Adam with lr 0.01
* Loss function: BCEWithLogitsLoss
* Batch Size: 8
* Epochs: 15
* Parameters =  8,631,810
* Transform: Resize, Normalize and ToTensor

The IOU score is calculated with below code:
```
def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)
```
Below images are the result trained from part of dataset(12k).
 * Result of image size 128
    ```
    Prediction:
    Avg IOU for mask prediction: 0.9499
    Avg IOU for dense depth prediction: 0.7299
    ```
   ```
    order of images:
        1. actual Mask
        2. predicted mask
        3. actual dense depth
        4. predicted dense depth
   ```
 ![image1](https://github.com/sajnanshetty/deep-learning/blob/master/s15/images/128_image_size.png)
  
 * Result of image size 64
    ```
    Prediction:
    Avg IOU for mask prediction: 0.9420
    Avg IOU for dense depth prediction: 0.7454
   ```
 ![image2](https://github.com/sajnanshetty/deep-learning/blob/master/s15/images/predicted_images_64_size.png)
    

The prediction of mask images are clear irrespective of any input image size.
The prediction of dense depth images are more clear when size of the image goes higher.

The saved models are stored in below path:
Saving the model for each best prediction and can be found in below location
[saved_images](https://drive.google.com/drive/folders/1lRZWQ6-SNggtGJ5dxETclky7mKBqxyK2?usp=sharing)
    
Saving batches of images and pickled the data into following location:
[test_image](https://drive.google.com/drive/folders/1UiqP3rXae7iX7hbgB3NXWdFJRpjjrykP?usp=sharing)




    







    