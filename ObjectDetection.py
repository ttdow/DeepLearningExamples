import os
import numpy as np
from PIL import Image

import torch

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms

        # Load all image files, sorting them to ensure they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

        def __getitem__(self, idx):
            # Load images and masks
            img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
            mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
            img = Image.open(img_path).convert("RGB")

            # Note that we haven't converted the mask to RGB, because each color
            # corresponds to a different instance with 0 being the background
            mask = Image.open(mask_path)
            
            # Convert the image to a numpy array
            mask = np.array(mask)

            # Instances are encoded as different colors
            obj_ids = np.unique(mask)

            # First id is the background, so remove it
            obj_ids = obj_ids[1:]

            # Split the color-encoded mask into a set of binary masks
            masks = mask == obj_ids[:, None, None]

            # Get the bounding box coordinates for each mask
            num_objs = len(obj_ids)
            boxes = []
            for i in range(num_objs):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])

            # Convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

            # There is only one class
            labels = torch.ones((num_objs), dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)

            image_id = torch.tensor([idx])
            area = (boxes[:,3] - boxes[:,1]) * (boxes[:, 2] - boxes[:, 0])

            # Suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes

