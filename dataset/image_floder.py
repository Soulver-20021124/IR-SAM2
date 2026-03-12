import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset


def Normalized(img, dataset):
    if dataset == "NUDT-SIRST":
        img = (img - 107.80905151367188) / 33.02274703979492
    elif dataset == "NUAA-SIRST":
        img = (img - 101.06385040283203) / 34.619606018066406
    elif dataset == "IRSTD-1k":
        img = (img - 87.4661865234375) / 39.71953201293945
    return img


def random_crop(img, mask, patch_size, pos_prob=None):
    h, w, c = img.shape
    if min(h, w) < patch_size:
        img = np.pad(
            img,
            ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w), (0, 0)),
            mode="constant",
        )
        mask = np.pad(
            mask,
            ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)),
            mode="constant",
        )
        h, w, c = img.shape

    while 1:
        h_start = random.randint(0, h - patch_size)
        h_end = h_start + patch_size
        w_start = random.randint(0, w - patch_size)
        w_end = w_start + patch_size

        img_patch = img[h_start:h_end, w_start:w_end]
        mask_patch = mask[h_start:h_end, w_start:w_end]

        if pos_prob is None or random.random() > pos_prob:
            break
        elif mask_patch.sum() > 0:
            break

    return img_patch, mask_patch


def augumentation(input, target):
    if random.random() < 0.5:
        input = input[::-1, :, :]
        target = target[::-1, :]
    if random.random() < 0.5:
        input = input[:, ::-1, :]
        target = target[:, ::-1]
    if random.random() < 0.5:
        input = input.transpose(1, 0, 2)
        target = target.transpose(1, 0)
    return input, target


def PadImg(img, times=32):
    h, w, c = img.shape
    if not h % times == 0:
        img = np.pad(img, ((0, (h // times + 1) * times - h), (0, 0), (0, 0)), mode="constant")
    if not w % times == 0:
        img = np.pad(img, ((0, 0), (0, (w // times + 1) * times - w), (0, 0)), mode="constant")
    return img


def PadMask(img, times=32):
    h, w = img.shape
    if not h % times == 0:
        img = np.pad(img, ((0, (h // times + 1) * times - h), (0, 0)), mode="constant")
    if not w % times == 0:
        img = np.pad(img, ((0, 0), (0, (w // times + 1) * times - w)), mode="constant")
    return img


# Modify from https://github.com/xdFai/SCTransNet/blob/main/dataset.py and https://github.com/YeRen123455/Infrared-Small-Target-Detection
class ImageFolder(Dataset):
    def __init__(
        self,
        path,
        data_set="NUDT",
        istraining=True,
        base_size=256,
        crop_size=256,
        copy_paste=True,
    ):
        self.path = path
        self.copy_paste = copy_paste
        self.T_masks = os.path.join(path, data_set, "Target_mask")
        self.T_images = os.path.join(path, data_set, "Target_image")
        self.base_size = base_size
        self.crop_size = crop_size
        self.istraining = istraining
        self.data_set = data_set
        self.images, self.masks = [], []
        self.augumentation = augumentation
        configs_path = "train.txt" if istraining else "test.txt"
        self.totenser = transforms.ToTensor()
        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.float() / 255.0),
            ]
        )

        if data_set == "NUAA-SIRST":
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        elif data_set == "IRSTD-1k":
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.28450727, 0.28450724, 0.28450724], [0.22880708, 0.22880709, 0.22880709]),
                ]
            )
        elif data_set == "NUDT-sea":
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.1583, 0.1583, 0.1583], [0.0885, 0.0885, 0.0885]),
                ]
            )
        elif data_set == "SIRST-AUG":
            # 参考代码只做了 /255 处理，ToTensor() 本身就包含这个功能
            # 不需要额外的 Normalize
            self.transform = transforms.ToTensor()
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        # get train set list
        with open(
            os.path.join("./dataset/set_configs", data_set, configs_path),
            encoding="utf-8",
        ) as file:
            lines = file.readlines()

        for line in lines:
            data = line.strip()
            if data_set == "SIRST-UAVB":
                image_path = os.path.join(path, data_set, "images", f"{data}.jpg")
            else:
                image_path = os.path.join(path, data_set, "images", f"{data}.png")
            if data_set == "NUAA":
                mask_path = os.path.join(path, data_set, "masks", f"{data}_pixels0.png")
            elif data_set == "SIRST-UAVB":
                mask_path = os.path.join(path, data_set, "masks", f"{data}.jpg")
            else:
                mask_path = os.path.join(path, data_set, "masks", f"{data}.png")

            self.images.append(image_path)
            self.masks.append(mask_path)

    def __len__(self):
        return len(self.images)

    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        if self.data_set == "SIRST-UAVB" or self.data_set == "IRSTDID-SKY":
            img = img.resize((480, 480), Image.BILINEAR)
            mask = mask.resize((480, 480), Image.NEAREST)
        else:
            img = img.resize((base_size, base_size), Image.BILINEAR)
            mask = mask.resize((base_size, base_size), Image.NEAREST)

        # final transform
        img, mask = np.array(img), np.array(mask, dtype=np.float32)  # img: <class 'mxnet.ndarray.ndarray.NDArray'> (512, 512, 3)
        return img, mask

    def _copy_paste_transform(self, img, mask, CP_num):
        img_path = self.T_images + "/"  # img_id的数值正好补了self._image_path在上面定义的2个空
        label_path = self.T_masks + "/"
        w, h = mask.size
        img_dir = os.listdir(img_path)
        label_dir = os.listdir(label_path)
        range_k = len(img_dir)
        dice = random.randint(0, 1)

        if dice == 0:
            img = img
            mask = mask
        else:
            for i in range(CP_num):
                k = random.randint(0, range_k - 1)
                x = random.randint(0, w - 1)
                y = random.randint(0, h - 1)
                T_I_path = img_path + img_dir[k]
                T_M_path = label_path + label_dir[k]
                T_img = Image.open(T_I_path).convert("RGB")  ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
                T_mask = Image.open(T_M_path)
                img.paste(T_img, (x, y))
                mask.paste(T_mask, (x, y))

        return img, mask

    def _sync_transform(self, img, mask, is_copy_paste=True):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        # final transform
        if is_copy_paste:
            img, mask = self._copy_paste_transform(img, mask, random.randint(1, 100))  # CP
        img, mask = np.array(img), np.array(mask, dtype=np.float32)
        return img, mask

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.masks[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask_array = np.array(mask)
        mask_array[mask_array >= 127] = 255
        mask_array[mask_array < 127] = 0

        mask = Image.fromarray(mask_array)
        image_name = os.path.basename(image_path)

        # 初始化返回变量
        img_tensor = None
        mask_tensor = None

        if self.data_set == "NUDT-sea" or self.data_set == "IRSTD-1k":
            if self.istraining:
                img, mask = self._sync_transform(image, mask, is_copy_paste=self.copy_paste)
                img = self.transform(img)
                mask = np.expand_dims(mask, axis=0).astype("float32") / 255.0
            else:
                img, mask = self._testval_sync_transform(image, mask)
                img = self.transform(img)
                mask = np.expand_dims(mask, axis=0).astype("float32") / 255.0
            return img, torch.from_numpy(mask), image_name
        # === 【新增分支】专门处理 SIRST-AUG ===
        elif self.data_set == "SIRST-AUG":
            # 1. 强制调整到 base_size (对应参考代码的 reshape，但 resize 更安全)
            image = image.resize((self.base_size, self.base_size), Image.BILINEAR)
            mask = mask.resize((self.base_size, self.base_size), Image.NEAREST)

            # 转为 Numpy 以便进行 augumentation 处理
            img_np = np.array(image)   # (H, W, 3)
            mask_np = np.array(mask)   # (H, W)

            if self.istraining:
                # 2. 使用现有的 augumentation 函数 (简单的翻转和转置)
                # 注意：你的全局函数 augumentation 期望输入是 (H, W, C)
                img_np, mask_np = self.augumentation(img_np, mask_np)

            # 3. 手动转 Tensor 并归一化 (/255)
            # 图片: Numpy (H, W, C) -> Tensor (C, H, W)
            img = torch.from_numpy(np.ascontiguousarray(img_np.transpose(2, 0, 1))).float() / 255.0
            
            # 掩码: 处理成 0/1 张量
            # 参考代码是用 mask / max(mask)，这里用 >0 更通用
            if mask_np.max() > 0:
                mask_np = mask_np / mask_np.max() # 归一化到 0-1
            mask = torch.from_numpy(np.ascontiguousarray(mask_np)).float()
            
            # 增加 channel 维度: (H, W) -> (1, H, W)
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)

            return img, mask, image_name
        elif self.data_set == "NUDT-SIRST":
            image = image.resize((self.base_size, self.base_size), Image.BILINEAR)
            mask = mask.resize((self.base_size, self.base_size), Image.BILINEAR)
            if self.istraining:
                image = Normalized(np.array(image, dtype=np.float32), self.data_set)
                mask = np.array(mask, dtype=np.float32) / 255.0
                if len(mask.shape) > 2:
                    mask = mask[:, :, 0]

                img_patch, mask_patch = random_crop(image, mask, self.base_size, pos_prob=0.5)
                img_patch, mask_patch = self.augumentation(img_patch, mask_patch)

                img_patch, mask_patch = (
                    img_patch.transpose(2, 0, 1),
                    mask_patch[np.newaxis, :],
                )
                img = torch.from_numpy(np.ascontiguousarray(img_patch))
                mask = torch.from_numpy(np.ascontiguousarray(mask_patch))

            else:
                image = Normalized(np.array(image, dtype=np.float32), self.data_set)
                mask = np.array(mask, dtype=np.float32) / 255.0
                if len(mask.shape) > 2:
                    mask = mask[:, :, 0]
                img_patch = PadImg(image)
                mask_patch = PadMask(mask)

                img_patch, mask_patch = (
                    img_patch.transpose(2, 0, 1),
                    mask_patch[np.newaxis, :],
                )
                img = torch.from_numpy(np.ascontiguousarray(img_patch))
                mask = torch.from_numpy(np.ascontiguousarray(mask_patch))

            mask = (mask > 0).to(torch.float32)
            return img, mask, image_name
        else:
            if self.istraining:
                img, mask = self._sync_transform(image, mask, is_copy_paste=False)
                img = self.transform(img)
                mask = np.expand_dims(mask, axis=0).astype("float32") / 255.0
            else:
                img, mask = self._testval_sync_transform(image, mask)
                img = self.transform(img)
                mask = np.expand_dims(mask, axis=0).astype("float32") / 255.0
            mask[mask < 1] = 0
            return img, torch.from_numpy(mask), image_name
