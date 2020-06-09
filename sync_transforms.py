import random
from PIL import Image, ImageOps, ImageFilter
import numpy as np


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomScale(object):
    def __init__(self, base_size, crop_size, resize_scale_range):
        self.base_size = base_size
        self.crop_size = crop_size
        self.resize_scale_range = resize_scale_range

    def __call__(self, img, mask):
        w, h = img.size
        # print("img.size:", img.size)
        short_size = random.randint(int(self.base_size * self.resize_scale_range[0]),
                                    int(self.base_size * self.resize_scale_range[1]))
        # print("short_size:", short_size)
        #         if h > w:
        #             ow = short_size
        #             oh = int(1.0 * h * ow / w)
        #         else:
        #             oh = short_size
        #             ow = int(1.0 * w * oh / h)
        ow, oh = short_size, short_size
        # print("ow, oh = ", ow, oh)
        img, mask = img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        w, h = img.size
        img = np.array(img)
        mask = np.array(mask)
        num_crop = 0
        while num_crop < 5:
            x = random.randint(0, w - self.crop_size)
            y = random.randint(0, h - self.crop_size)
            endx = x + self.crop_size
            endy = y + self.crop_size
            patch = img[y:endy, x:endx]
            if (patch == 0).all():
                continue
            else:
                break
        img = img[y:endy, x:endx]
        mask = mask[y:endy, x:endx]
        img, mask = Image.fromarray(img), Image.fromarray(mask)
        return img, mask


class RandomFlip(object):
    def __init__(self, flip_ratio=0.5):
        self.flip_ratio = flip_ratio

    def __call__(self, img, mask):
        if random.random() < self.flip_ratio:
            img, mask = img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            img, mask = img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)
        return img, mask


class RandomGaussianBlur(object):
    def __init__(self, prop):
        self.prop = prop
    def __call__(self, img, mask, prop):
        if random.random() < self.prop:
            img = img.filter(ImageFilter.GaussianBlur)(radius=random.random())
        return img, mask