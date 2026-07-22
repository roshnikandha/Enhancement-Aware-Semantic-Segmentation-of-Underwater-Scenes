import os
import cv2
import torch
from torch.utils.data import Dataset

# RGB Color -> Class ID
COLOR_MAP = {
    (0, 0, 0): 0,
    (0, 0, 255): 1,
    (0, 255, 0): 2,
    (0, 255, 255): 3,
    (255, 0, 0): 4,
    (255, 0, 255): 5,
    (255, 255, 0): 6,
    (255, 255, 255): 7
}

class SUIMDataset(Dataset):

    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_name = self.images[idx]

        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(
            self.mask_dir,
            image_name.replace(".jpg", ".bmp")
        )

        # --------------------
        # Read Image
        # --------------------
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image to fixed size
        image = cv2.resize(image, (512, 512))

        # --------------------
        # Read Mask
        # --------------------
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # Resize mask using nearest neighbour
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        # --------------------
        # Convert image to tensor
        # --------------------
        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute(2, 0, 1) / 255.0

        # --------------------
        # Convert RGB mask to class IDs
        # --------------------
        mask_tensor = torch.tensor(mask, dtype=torch.uint8)

        mask_class = torch.zeros((512, 512), dtype=torch.long)

        for color, class_id in COLOR_MAP.items():
            color = torch.tensor(color, dtype=torch.uint8)
            matches = (mask_tensor == color).all(dim=2)
            mask_class[matches] = class_id

        return image, mask_class