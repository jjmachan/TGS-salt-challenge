import os

from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import pandas as pd

from utils import load_mask, stratified_split

# The albumentations transforms
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightness(p=0.2, limit=0.2),
    A.RandomContrast(p=0.1, limit=0.2),
    A.ShiftScaleRotate(shift_limit=0.1625, scale_limit=0.6, rotate_limit=0, p=0.7),
    A.Resize(128, 128),
    A.ToFloat(always_apply=True),
    ToTensorV2()
])

validation_transform = A.Compose([
    A.Resize(128, 128),
    A.ToFloat(always_apply=True),
    ToTensorV2()
])

downsize_transform = A.Compose([
    A.Resize(101, 101)
])

# Pytorch dataset to load the images
class TGSDatasetAlbu(Dataset):
    def __init__(self, ids, image_path, mask_path, transform=None):
        self.image_path = image_path
        self.mask_path = mask_path
        self.transform = transform

        image_list= ids.values
        sample_names = []
        for img_name in image_list:
            sample_names.append(img_name+'.png')

        self.sample_names = sample_names

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.image_path, self.sample_names[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.imread(os.path.join(self.mask_path, self.sample_names[idx]))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        return image, mask

    def __len__(self):
        return len(self.sample_names)

def build_validation_set(train_df, data_path):
    """
    Build a validation dataset from the train dataset. We are
    using stratified sampling to split the dataset and it is sampled
    by taking into accound the coverage of salt beds in the images
    """
    load_func = lambda x: load_mask(x, data_path+'train/masks')
    train_df['coverage'] = train_df.id.apply(load_func)/pow(101, 2)
    train_df['cuts'] = pd.cut(train_df.coverage, bins=10,
                              labels=list(range(10)))
    train_df = train_df.set_index('id')

    train_set = train_df.drop(stratified_split(train_df)).reset_index().id
    val_set = train_df.loc[stratified_split(train_df)].reset_index().id

    return train_set, val_set

def build_train_val_loaders(data_path, batch_size):
    """
    Build the Train and Validation DataLoaders for training Pytorch models.
    """

    train_df = pd.read_csv(data_path+'train.csv')
    train_set, val_set = build_validation_set(train_df, data_path)
    train_dataset = TGSDatasetAlbu(train_set, data_path+'train/images',
                                   data_path+'train/masks', train_transform)
    val_dataset = TGSDatasetAlbu(val_set, data_path+'train/images',
                                 data_path+'train/masks', validation_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, drop_last=False)

    return train_dataloader, val_dataloader
