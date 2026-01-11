import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from data.dataset import TrainDataset, TestDataset
from io import BytesIO
import numpy as np
from scipy.ndimage.filters import gaussian_filter

class RobustnessTest(object):
    def __init__(self, jpeg_quality=None, gaussian_sigma=None):
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma

    def __call__(self, img):
        if self.jpeg_quality is not None:
            out = BytesIO()
            img.save(out, format='jpeg', quality=self.jpeg_quality)
            out.seek(0)
            img = Image.open(out)

        if self.gaussian_sigma is not None:
            img_np = np.array(img)
            for i in range(3): 
                gaussian_filter(img_np[:,:,i], output=img_np[:,:,i], sigma=self.gaussian_sigma)
            img = Image.fromarray(img_np)

        return img

def create_dataset(config):   
    robustness_transform = RobustnessTest(jpeg_quality=config['jpeg_quality'], gaussian_sigma=config['gaussian_sigma'])

    train_hsv_preprocess = transforms.Compose([                
        transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda img: img.convert("HSV")),
        transforms.ToTensor(),
        transforms.Normalize((0.3054201304912567, 0.260720431804657, 0.5469616055488586), (0.28788992762565613, 0.24716658890247345, 0.2902427613735199)),
    ]) 
    train_clip_preprocess = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    test_hsv_preprocess = transforms.Compose([ 
        transforms.Resize((config['image_res'], config['image_res']),interpolation=Image.BICUBIC),
        robustness_transform,
        transforms.Lambda(lambda img: img.convert("HSV")),
        transforms.ToTensor(),
        transforms.Normalize((0.3054201304912567, 0.260720431804657, 0.5469616055488586), (0.28788992762565613, 0.24716658890247345, 0.2902427613735199)),
    ]) 
    test_clip_preprocess = transforms.Compose([ 
        transforms.Resize((config['image_res'], config['image_res']),interpolation=Image.BICUBIC),
        robustness_transform,
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ]) 

    train_dataset = TrainDataset(config['train_root'], train_hsv_preprocess, train_clip_preprocess)  
    val_dataset = TrainDataset(config['val_root'], test_hsv_preprocess, test_clip_preprocess)  
    test_dataset = TestDataset(config['test_root'], test_hsv_preprocess, test_clip_preprocess)                
    return train_dataset, val_dataset, test_dataset    


def create_loader(datasets, batch_size, is_trains):
    loaders = []
    for dataset, bs, is_train in zip(datasets, batch_size, is_trains):
        if is_train:
            shuffle = True
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            pin_memory=True,
            shuffle=shuffle,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    