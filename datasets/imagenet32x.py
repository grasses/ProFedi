
import os
import pickle
import numpy as np
from torch.utils.data.dataset import Dataset


class ImageNet32x(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, index=False, **kwargs):
        self.transform = transform
        self.target_transform = target_transform
        if train:
            data_list = [
                "train_data_batch_1",
                "train_data_batch_2",
                "train_data_batch_3",
                "train_data_batch_4",
                "train_data_batch_5",
                "train_data_batch_6",
                "train_data_batch_7",
                "train_data_batch_8",
                "train_data_batch_9",
            ]
        else:
            data_list = ["train_data_batch_10"]
        self.index = index
        '''
        TODO: Not support testset
        if not train:
            raise NotImplementedError("Only support train mode!")
        '''

        data = []
        targets = []
        for step, file_name in enumerate(data_list):
            print(f"-> loading {step+1}th.. file for ImageNet32x")
            file_path = os.path.join(root, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
            data.append(entry["data"])
            targets.extend(entry["labels"])
        # Convert data (List) to NHWC (np.ndarray) works with transformations.
        data = np.vstack(data).reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
        self.data = data
        self.targets = np.asarray(targets)
        self.kwargs = kwargs

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]-1
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.index:
            return index, img, target
        else:
            return img, target

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    import torch
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    imagenet_dst = ImageNet32x(root="ImageNet32x", transform=transform)
    loader = torch.utils.data.DataLoader(
            imagenet_dst,
            batch_size=100,
            shuffle=True, num_workers=2
        )

    for step, (idx, x, y) in enumerate(loader):
        print(x)
        exit(1)






