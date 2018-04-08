import utils
import os
import skimage.io
import numpy as np
import random
from random import randint
import transform
import warnings
from random import shuffle
import cv2

class BowlDataset(utils.Dataset):
    def load_bowl(self, dataset_dir, img_ids):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("bowl", 1, "cell")

        # Add images
        for img_id in img_ids:
            path = os.path.join(dataset_dir, img_id, "images", "{}.png".format(img_id))
            gray_path = os.path.join(dataset_dir, img_id, "images", "{}_gray.png".format(img_id))
            im =skimage.io.imread(path)
            self.add_image(
                "bowl", image_id=img_id,
                path=os.path.join(dataset_dir, img_id, "images", "{}.png".format(img_id)),
                gray_path=gray_path,
                width=im.shape[1],
                height=im.shape[0],
                mask_dir=os.path.join(dataset_dir, img_id, "masks"))

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        info = self.image_info[image_id]
        mask_dir = info["mask_dir"]
        mask = []
        class_ids = []

        for mask_file in next(os.walk(mask_dir))[2]:
            mask.append(skimage.io.imread(os.path.join(mask_dir, mask_file)))
            class_ids.append(1)

        mask = np.array(mask, dtype=np.uint8)
        mask &= 1
        mask = np.stack(mask, axis=2)
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids.astype(np.int32)

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        return info["image_id"]

    def load_image(self, image_id, gray_scale=True):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        path = self.image_info[image_id]['path']
        if gray_scale:
            gray_path = self.image_info[image_id]['gray_path']
            if os.path.exists(gray_path):
                image = skimage.io.imread(gray_path)
            else:
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                image = image[:, :, np.newaxis]
                image = np.repeat(image, 3, axis=2)
                skimage.io.imsave(gray_path, image)
        else:
            image = skimage.io.imread(path)
            image = image[:, :, :3]
        return image

    def augment(self, augment_dir, num_per_image, max_crop_ratio=0.8):
        image_info = [info for info in self.image_info]
        for _img_id, image_info in enumerate(image_info):
            img_id = image_info['id']
            image = self.load_image(_img_id, gray_scale=False)
            mask, _ = self.load_mask(_img_id)
            H, W = image.shape[:2]

            if not os.path.exists(augment_dir):
                os.makedirs(augment_dir)

            for _ in range(num_per_image):
                new_image = np.copy(image)
                new_mask = np.copy(mask)
                new_img_id = img_id
                horizontal_flip = random.uniform(0, 1) < 0.5
                vertical_flip = random.uniform(0, 1) < 0.5
                rotate90 = random.uniform(0, 1) < 0.5
                crop = random.uniform(0, 1) < 0.5

                if horizontal_flip:
                    new_image, new_mask = transform.horizontal_flip_transform2(new_image, new_mask)
                    new_img_id += '_hor'

                if vertical_flip:
                    new_image, new_mask = transform.vertical_flip_transform2(new_image, new_mask)
                    new_img_id += '_ver'

                if rotate90:
                    angle = randint(1, 3)*90
                    new_image, new_mask = transform.rotate90_transform2(new_image, new_mask, angle)
                    new_img_id += '_rot{}'.format(angle)

                # disable crop and use it later
                if crop:
                    crop_width = round(W * random.uniform(max_crop_ratio, 0.99))
                    crop_height = round(H * random.uniform(max_crop_ratio, 0.99))
                    x = randint(1, W - crop_width)
                    y = randint(1, H - crop_height)
                    new_image, new_mask = transform.crop_transform2(new_image, new_mask, x, y, crop_width, crop_height)
                    new_img_id += '_crop{}|{}|{}|{}'.format(y, x, crop_height, crop_width)

                image_dir = os.path.join(augment_dir, new_img_id)
                path = os.path.join(image_dir, "images", "{}.png".format(new_img_id))
                mask_dir = os.path.join(image_dir, "masks")

                if not os.path.exists(image_dir):
                    os.makedirs(image_dir)
                    os.makedirs(os.path.join(image_dir, "images"))
                    os.makedirs(os.path.join(image_dir, "masks"))
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        skimage.io.imsave(path, new_image)

                    for i in range(new_mask.shape[-1]):
                        _mask = new_mask[:, :, i]
                        _mask[_mask > 0] = 255

                        mask_path = os.path.join(mask_dir, "mask{}.png".format(i+1))
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            skimage.io.imsave(mask_path, _mask)

                self.add_image("bowl", image_id=new_img_id,
                               path=path,
                               width=new_image.shape[1],
                               height=new_image.shape[0],
                               mask_dir=mask_dir)
        shuffle(self.image_info)
