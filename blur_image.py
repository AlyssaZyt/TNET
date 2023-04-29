import cv2
import os

src_image_folder = 'data/val/original/'
blur_folder = 'data/val/blur/'
imglist = sorted(os.listdir(src_image_folder))
for imgname in imglist:
    imgpath = src_image_folder+imgname
    imgnum = imgname.split('.')[0]
    input = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    blur = cv2.GaussianBlur(input, (3,3), 3) # We recommend you to use GaussianBlur for at least 10 times.
    cv2.imwrite(blur_folder+imgnum+'.jpg', blur)