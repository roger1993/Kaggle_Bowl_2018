import cv2
import matplotlib.pyplot as plt
import imageio
import skimage
from skimage import color

# img = cv2.imread("/Users/roger/Downloads/Kaggle_Bowl_2018/data/external_data/TCGA-18-5592-01Z-00-DX1/images/TCGA-18-5592-01Z-00-DX1.tif")
# # cv2.imshow("img",img)
# # cv2.waitKey(0)
# print img.shape
# plt.imshow(img)
# plt.show()

def load_image(image_path):
    image = imageio.imread(image_path)
    if image.shape[2] != 3:
    	image = image[:,:,:3]
    # print(image.max())
    # print(image.min())
    image = preprocess(image)
    image = image.astype('float32')
    return image

def preprocess(img):
    gray = skimage.color.rgb2gray(img.astype('uint8'))
    img = skimage.color.gray2rgb(gray)
    img *= 255.
    return img


image = load_image("/Users/roger/Downloads/Kaggle_Bowl_2018/data/stage1_train/0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9/images/0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9.png")
#print(image[1])
print(image.max())
print(image.min())

assert image[0].all() == image[1].all() == image[2].all()