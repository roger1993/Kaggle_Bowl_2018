import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

def rotate(image, angle, center=None, scale=1.0):
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # return the rotated image
    return rotated

def translate(image, x, y):
    # define the translation matrix and perform the translation
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # return the translated image
    return shifted

def mirror_border(image, pad=32):
    reflect = cv2.copyMakeBorder(image,pad,pad,pad,pad,cv2.BORDER_REFLECT_101)
    return reflect

if __name__ == '__main__':
    # for angle in range(0, 30, 10):
    # # rotate the image and display it
    #     img = cv2.imread("/Users/roger/Downloads/Kaggle_Bowl_2018/data/stage1_train/0acd2c223d300ea55d0546797713851e818e5c697d073b7f4091b96ce0f3d2fe/images/0acd2c223d300ea55d0546797713851e818e5c697d073b7f4091b96ce0f3d2fe.png")
    #     rotated = rotate(img, angle=angle)
    #     cv2.imshow("Angle=%d" % (angle), rotated)
    #     cv2.waitKey(0)
    img = cv2.imread("/Users/roger/Downloads/Kaggle_Bowl_2018/data/stage1_train/0acd2c223d300ea55d0546797713851e818e5c697d073b7f4091b96ce0f3d2fe/images/0acd2c223d300ea55d0546797713851e818e5c697d073b7f4091b96ce0f3d2fe.png")
    cv2.imshow("img", img)
    cv2.waitKey(0)