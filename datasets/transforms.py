# ------------------------------------------------------------------------
# QAHOI
# Copyright (c) 2021 Junwen Chen. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate


def crop(image, target, region):
    """ [fill]

    Args:
        image (PIL.Image): image input
        target (dict): image target dictionary
        region (tuple(int, int, int, int)): tuple of 4 elements consisting of parameters for F.crop()
            these 4 elements are (top, left, height, width), where:
                top - vertical component (y-coord) of the top left corner of the crop box
                left - horizontal component (x-coord) of the top left corner of the crop box
                height - height of the crop box
                width - width of the crop box

    """
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # set the size as the height and width of the crop box
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        ## torch.min, then clamp makes sure that none of the bbox coords go out of bounds
        # the reshape transforms the shape to [N, 2, 2] where cropped_boxes[i, 0, :] is [x_1, y_1] 
        # and cropped_boxes[i, 1, :] is [x_2, y_2]
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        
        # (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]) -> forms a tensor of shape [N, 2],
        # where the rows hold the width and height
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            # keep ones where x_2 > x_1 and y_2 > y_1
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    # hflip horizontally flips the image,
    # image is expected to have shape [W, H] 
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target: # flip object bbox horizontally
        boxes = target["boxes"]
        # remember target["boxes"] consists of a torch tensor of shape (N, 4)
        #   in the format: (x0, y0, x1, y1)
        # flipped horizontally, the y-coordinates remain unchaged (* 1 + 0)
        # but the x-coordinates are changed:
        # result: (w-x1, y0, w-x0, y1)
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target: # by default there are no "masks" in target
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    """
    Resizes the shorter side of the image to the given size, 
        and resize the longer side to maintain aspect ratio,
        resizes object bbox coordinates in the image accordingly,
        and record the new size of the image in target["size"]
    image: PIL image to resize
    target: dictionary referring to the bbox, labels of the objects in the image
    size: int representing the size to resize the image to (or can be a tuple (w,h))
    max_size: upperbound for image size
    """
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        """
        Resizes the shorter side of the image to the given size, 
            and resizes the longer side based on the shorter side to maintain the aspect ratio,
            returns (h, w)
        Args:
            image_size: tuple (w, h) of the current PIL image size
            size: int representing the size to resize the image to
        """
        w, h = image_size
        if max_size is not None:
            ## check if the longer size would exceed max_size, if so bound the resized longer side to max_size
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            # since the shorter side is resized to size,
            # to maintain aspect ratio, the longer side is size * original_aspect_ratio
            # if the resized longer side exceeds max_size,
            # bound size, the reason why we do max_size * min_original_size / max_original_size
            # is because if we resize the shorter side to this, then the longer side is resized to
            # (max_size * min_original_size / max_original_size) * (max_original_size / min_original_size)
            # the two fractions cancel out and we're left with max_size
            # in other words, this ensures that the resized length the longer side is capped at max_size
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        # if the shorter side (width or height) already matches the desired resize size, just return (h, w)
        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h: # if the image is taller than it is wider
            ow = size  # resize the shorter side (the width to size)
            oh = int(size * h / w) # resize the longer size by doing size * aspect ratio
        else:
            oh = size # resize the shorter side (the height to size)
            ow = int(size * w / h) # resize the longer size by doing size * aspect ratio

        # return the resized height and width
        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        """
        Get the resized size as a tuple (h, w)
        image_size: a tuple (w, h) representing the current PIL image size
        size: int or tuple to resize the image to
        """
        if isinstance(size, (list, tuple)): # if explicitly given width AND height to resize to, just return height, width
            return size[::-1]
        else: # otherwise only an int is given, need to decide how to resize, esp given images with width != height
            return get_size_with_aspect_ratio(image_size, size, max_size)

    # get the resized image sizes as a tuple (h, w)
    size = get_size(image.size, size, max_size)
    # resize the PIL image to the given size, here it is always a tuple (h, w)
    # NOTE: even though PIL images' size are (w, h) passing in size as (h, w) is correct. 
    #       For example, if the original size is (250, 201) and we pass (201, 250), 
    #       the resulting PIL image remains unchange (250, 201)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    # get the ratio of resized to original sizes as a tuple in the form:
    # (ratio of resized width to original width, ratio of resized height to original height)
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        # remember target["boxes"] is a torch tensor with shape (N, 4) where each of the 4 values represent:
        #   x0, y0, x1, y1
        # thus multiply x-coordinates by the width ratio to get the new x-coordinate in the resized image
        # and multiply y-coordinates by the height ratio to get the new y-coordinate in the resized image.
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target: # scales area of bbox to match scaled bbox coords
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w]) # stores the resized (h, w)

    if "masks" in target: # N/A
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        # generate a random width and height to crop the image to
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        # get_params is a static method, which returns a tuple of 4 elements 
        # which represent params to provide to crop()
        # h, w are the output size we want
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    """
    For a set probability, p, generate a random number and if it is below p, 
    horizontally flip the img and target["boxes"]
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    """
    Resizes an image and object bbox coords to a random size from a list of given sizes.
    Also records the resized size in target["size"]
    """
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes) # chose randomly between one of the possible sizes
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    """
    Must be passed two arguments, img and target.
    Converts the (by default PIL Image) img to tensor and leaves the target intact
    """
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    """
    Accepts an image (in the form of a tensor), 
    and an optional target (in the form of a dictionary with two keys - boxes and labels, 
        where boxes is a torch tensor of shape (N, 4) and labels is a numpy array of shape (N,))
    Normalizes the image (by the given mean and std), and normalizes the "boxes" key in target (if given)
        by changing it from coordinates to center-point coords and width, height then normalized by image width, height
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        # normalize the image with the set mean and std
        image = F.normalize(image, mean=self.mean, std=self.std)
        # if no target is provided, just return the image and None
        if target is None:
            return image, None
        
        target = target.copy()
        h, w = image.shape[-2:]
        
        if "boxes" in target:
            # fetch the bbox coordinates for every object in the image
            boxes = target["boxes"] # tensor of shape [N, 4]
            # boxes is a torch tensor of shape (N, 4)
            # now each of the N objects has 4 elements: 
            #   x-coordinate of bbox center-point,
            #   y-coordinate of bbox center-point,
            #   width of bbox,
            #   height of bbox
            boxes = box_xyxy_to_cxcywh(boxes)
            # normalize the center-point and sizes by the width and height of the image
            # divide x-coordinate of bbox center-point with image width,
            # divide y-coordinate of bbox center-point with image height,
            # divide width of bbox with image width,
            # divide height of bbox with image height
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            # replace the value of the "boxes" key in target with 
            # the new normalized center-point and size info tensor
            target["boxes"] = boxes
            
        # return the normalized image and target
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        # computes the "official" string representation of the object
        # invoked when doing repr(Compose(...))
        # returns a string like:
        #   Compose(
        #       [transform 1],
        #       [transform 2],
        #       ...
        #   )
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

class ColorJitter(object):
    """Randomly change the brightness, contrast, saturation and hue of an image. 
    If the image is torch Tensor, it is expected to have […, 1 or 3, H, W] shape.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        """
        Args:
            brightness (int, optional): How much to jitter brightness. 
                brightness_factor is chosen uniformly from [max(0, 1-brightness), 1 + brightness]. 
                (i.e. multiplies brightness by brightness_factor), for ex. if brightness = 0.4, range is [0.6, 1.4]
                Defaults to 0.
            contrast (int, optional): How much to jitter contrast. 
                contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
                Defaults to 0.
            saturation (int, optional): How much to jitter saturation. 
                saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]. 
                Defaults to 0.
            hue (int, optional): How much to jitter hue. 
                hue_factor is chosen uniformly from [-hue, hue].
                Defaults to 0.
        """
        self.color_jitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img, target):
        """Perform ColorJitter transformation on the image, leaves the target unchanged.
        """
        return self.color_jitter(img), target
