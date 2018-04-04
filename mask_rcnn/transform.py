import cv2
import numpy as np


def crop_transform2(image, mask, x, y, w, h):

    image = image[y:y+h, x:x+w]
    mask = mask[y:y+h, x:x+w]

    return image, mask


def horizontal_flip_transform2(image, mask):
    image = cv2.flip(image, 1)
    mask = cv2.flip(mask, 1)
    if len(mask.shape) == 2:
        mask = mask[:, :, np.newaxis]
    return image, mask


def vertical_flip_transform2(image, mask):
    image = cv2.flip(image, 0)
    mask = cv2.flip(mask, 0)
    if len(mask.shape) == 2:
        mask = mask[:, :, np.newaxis]
    return image, mask


def rotate90_transform2(image, mask, angle):

    if angle == 90:
        image = image.transpose(1, 0, 2)  #cv2.transpose(img)
        image = cv2.flip(image, 1)
        mask = mask.transpose(1, 0, 2)
        mask = cv2.flip(mask, 1)
    elif angle == 180:
        image = cv2.flip(image, -1)
        mask = cv2.flip(mask, -1)
    elif angle == 270:
        image = image.transpose(1, 0, 2)  #cv2.transpose(img)
        image = cv2.flip(image, 0)
        mask = mask.transpose(1, 0, 2)
        mask = cv2.flip(mask, 0)

    if len(mask.shape) == 2:
        mask = mask[:, :, np.newaxis]
    return image, mask
