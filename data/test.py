import cv2
import matplotlib.pyplot as plt

img = cv2.imread("/Users/roger/Downloads/Kaggle_Bowl_2018/data/external_data/TCGA-18-5592-01Z-00-DX1/images/TCGA-18-5592-01Z-00-DX1.tif")
# cv2.imshow("img",img)
# cv2.waitKey(0)
print img.shape
plt.imshow(img)
plt.show()