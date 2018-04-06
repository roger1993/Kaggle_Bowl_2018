import cv2
import matplotlib.pyplot as plt
import imageio
import skimage
from skimage import color
import imutils
import imgaug as ia
from imgaug import augmenters as iaa

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

# def scale(img, factor):
# 	h, w = img.shape[:2]
# 	new_img = imutils.resize(img, width=int(factor * img.shape[1]))
# 	center = (int(img.shape[0]/2),int(img.shape[1]/2))
# 	cropScale = (int(center[0]/factor), int(center[1]/factor))
# 	scaled = new_img[int(cropScale[0]):int(center[0] + cropScale[0]), int(cropScale[1]):int(center[1] + cropScale[1])]
# 	return scaled


# image = load_image("/Users/roger/Downloads/Kaggle_Bowl_2018/data/stage1_train/0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9/images/0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9.png")
# #print(image[1])
# print(image.max())
# print(image.min())

# assert image[0].all() == image[1].all() == image[2].all()

from scipy.ndimage.interpolation import zoom
import scipy

img = cv2.imread("/Users/roger/Downloads/Kaggle_Bowl_2018/data/stage1_train/0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9/images/0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9.png")
#print(img.shape)
#img = scale(img,1)
# print(img.shape)
# h, w = img.shape[:2]
# img = scipy.ndimage.interpolation.zoom(img, 1.2)
# print(img.shape)
# plt.imshow(img)
# plt.show()
#cv2.imshow("img", img)
#cv2.waitKey(0)

for i in range(3):
	scaler = iaa.Affine(scale={"x": (0.8,1.2), "y": (0.8,1.2)}) # scale each input image to 80-120% on the y axis
	scaler_de = scaler.to_deterministic()
	img1 = scaler_de.augment_image(img) # scale image 5 by 80-120% on the y axis
	plt.imshow(img1)
	plt.show()
	img2 = scaler_de.augment_image(img)
	plt.imshow(img2)
	plt.show()
#print(img.shape)



    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matricies [height,width,depth]. Images can have
            different sizes.
        Returns 3 Numpy matricies:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image to fit the model expected size
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                max_dim=self.config.IMAGE_MAX_DIM,
                padding=self.config.IMAGE_PADDING)
            molded_image = mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = compose_image_meta(
                0, image.shape, window,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def mold_inputs2(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matricies [height,width,depth]. Images can have
            different sizes.
        Returns 3 Numpy matricies:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image to fit the model expected size
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding = utils.resize_image2(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                max_dim=self.config.IMAGE_MAX_DIM,
                padding=self.config.IMAGE_PADDING)
            molded_image = mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = compose_image_meta(
                0, image.shape, window,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows