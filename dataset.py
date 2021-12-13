import os 
import json

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT

# import albumentations as A

from PIL import Image

class CreateDataset(torch.utils.data.Dataset):
    def __init__(self, file_dir, img_dir, img_size, split_size=7, num_boxes=2, num_classes=20, box_format="coco", bb_ratio=False, offset=1, transform=None):
        with open(file_dir) as f:
            info = json.load(f)
            self.data = info["annotations"]
        self.img_dir = img_dir
        self.img_size = img_size
        self.box_format = box_format
        self.bb_ratio = bb_ratio
        self.offset = offset

        self.S = split_size
        self.B = num_boxes
        self.C = num_classes

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_id = self.data[index]["id"]
        boxes = []
        for i in range(len(self.data[index]["category_id"])):
            class_label = self.data[index]["category_id"][i]
            x, y, width, height = self.data[index]["bbox"][i]

            if not self.bb_ratio:
                x /= self.img_size[0]
                y /= self.img_size[1]
                width /= self.img_size[0]
                height /= self.img_size[1]

            boxes.append([class_label, x, y, width, height])
        img_path = os.path.join(self.img_dir, str(self.data[index]["id"])+".png")
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)
        
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label) - self.offset # for starting with 0

            if self.box_format == "coco":
                x_cen, y_cen = x+width/2, y+height/2
            elif self.box_format == "midpoint":
                x_cen, y_cen = x, y
            # i,j represents the cell row and cell column
            i, j = int(self.S * y_cen), int(self.S * x_cen)
            x_cell, y_cell = self.S * x_cen - j, self.S * y_cen - i

            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            if label_matrix[i, j, self.C] == 0:
                label_matrix[i, j, self.C] = 1 # one object per cell
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i, j, self.C+1:self.C+5] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return image, label_matrix


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes



def get_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0,0,0

    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1

    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5

    return mean, std

if __name__=="__main__":
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid

    FILE_DIR =  "datasets/info_all.json"
    IMG_DIR = "datasets/images"
    BATCH_SIZE = 16
    NUM_WORKERS=2
    PIN_MEMORY=True
    IMG_SIZE = [1080, 1920]

    EPOCHS = 10
    S=7
    B=2
    C=4

    transform = Compose([transforms.Resize((448, 448)), 
                        transforms.ToTensor()]) 

    train_dataset = CreateDataset(file_dir=FILE_DIR, img_dir=IMG_DIR, img_size=IMG_SIZE, split_size=S, num_boxes=B, num_classes=C, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=False)

    mean, std = get_mean_std(train_loader)
    print(mean); print(std)

    for images, labels in train_loader:
        plt.figure(figsize=(16,8))
        plt.axis('off')
        plt.imshow(make_grid(images).permute((1, 2, 0)))
        # for idx in range(BATCH_SIZE):
        #     plot_image(images[idx].permute(1,2,0).to("cpu"), labels, labels, show=False)
        break

    plt.show()
    # images, targets = next(iter(train_loader))