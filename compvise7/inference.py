import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from model import build_model
from tqdm import tqdm

def mask2rle(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

class TestDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.image_ids = os.listdir(images_dir)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.images_dir, image_id)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        return image, os.path.splitext(image_id)[0]

def predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model().to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()

    test_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    test_dataset = TestDataset('../../../data/test/test/images', test_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    image_ids = []
    rles = []
    with torch.no_grad():
        for images, ids in tqdm(test_loader, desc='Predicting'):
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()
            for pred, img_id in zip(preds, ids):
                pred_mask = (pred > 0.5).astype(np.uint8).squeeze(0)
                rle = mask2rle(pred_mask)
                image_ids.append(img_id)
                rles.append(rle)

    df = pd.DataFrame({'ImageId': image_ids, 'EncodedPixels': rles})
    df.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    predict()