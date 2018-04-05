import sys
import base_loader as base
import numpy as np 
import cv2
import os
import imageio
import matplotlib.pyplot as plt


class NucleiDataset(base.Dataset):
	def add_nuclei(self, root_dir, mode, split_ratio=0.9):
	    # Add classes
	    self.add_class("nuclei", 1, "nuclei") # source, id, name. id = 0s is BG

	    image_names = os.listdir(root_dir)
	    length = len(image_names)
	    
	    np.random.seed(1000)
	    image_names = list(np.random.permutation(image_names))
	    np.random.seed(None)
	    
	    if mode == 'train':
	        image_names = image_names[: int(split_ratio*length)]
	    if mode == 'val':
	        image_names = image_names[int(split_ratio*length):]
	    if mode == 'val_as_test':
	        image_names = image_names[int(split_ratio*length):]     
	        mode = 'test'
	    dirs = [root_dir + img_name + '/images/' for img_name in image_names]
	    mask_dirs = [root_dir + img_name + '/masks/' for img_name in image_names]

	    # Add images
	    for i in range(len(image_names)):
	        self.add_image(
	            source = "nuclei", 
	            image_id = i,
	            path = dirs[i] + image_names[i] + '.png',
	            mask_dir = mask_dirs[i],
	            name = image_names[i]
	            )

	"""
	We need 3 classes, boundary class, foreground class and backgroudn class
	"""
	def add_boundray(self, root_dir, mode, split_ratio=0.9):
		#self.add_class("nuclei", 2, "boundary")
		pass
	  

	def load_image(self, image_id):
	    """Load the specified image and return a [H,W,3] Numpy array.
	    """
	    image = imageio.imread(self.image_info[image_id]['path'])
	    # RGBA to RGB
	    if image.shape[2] != 3:
	        image = image[:,:,:3]
	    return image

	def image_reference(self, image_id):
	    """Return the details of the image."""
	    info = self.image_info[image_id]
	    if info["source"] == "nuclei":
	        return info["path"]
	    else:
	        super(NucleiDataset, self).image_reference(self, image_id)

	def load_mask(self, image_id):
	    """ 
	    Returns:
	        masks: A binary array of shape [height, width, instance count] with
	            a binary mask per instance.
	        class_ids: a 1D array of class IDs of the instance masks.
	    """
	    info = self.image_info[image_id]
	    mask_dir= info['mask_dir'] 
	    mask_names = os.listdir(mask_dir)
	    mask_paths = [mask_dir + mask_name for mask_name in mask_names]
	    
	    count = len(mask_paths)
	    
	    masks = [imageio.imread(path) for path in mask_paths]
	    mask = np.stack(masks, axis=-1)
	#        mask = mask.astype(bool)
	    mask = np.where(mask>128, 1, 0)
	    
	    class_ids = np.ones(count,dtype=np.int32)
	    return mask, class_ids

	def load_semantic(self, image_id):
	    info = self.image_info[image_id]
	    path = info['mask_dir'].replace('masks','images') 
	    mask_path = path + 'mask.png'
	    
	    mask = imageio.imread(mask_path)
	    mask = np.where(mask>128, 1, 0)
	    return mask

if __name__ == "__main__":
	ds = NucleiDataset()
	#ds.add_nuclei('data/stage1_train/','train')
	ds.add_nuclei('../stage1_train/','train')
	ds.prepare()
	print(ds.image_info[0])

	image = ds.load_image(0)
	print(image.shape)
	#plt.imshow(image)
	

	mask, _ = ds.load_mask(0)
	#print(len(_))
	print(mask.shape)

	means = []
	for idx in ds.image_ids:
	    im = ds.load_image(idx)
	    means.append(np.mean(im[:,-1],axis=0))
	print("Per-channel mean value is: ", np.mean(means,axis=0))
