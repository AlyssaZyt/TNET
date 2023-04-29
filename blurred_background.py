import torch.backends.cudnn as cudnn
import torch
import numpy as np
from torch import nn
from tnet import tnet
from utils import *
import time
import cv2
import os

# parameters of the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


if __name__ == '__main__':
    # test images
    img_id = '1'
    imgPath = './results/' + img_id + '.jpg'
    src_image_folder = 'data/test/image/'
    blur_folder = 'data/test/blur/'
    comp_folder = 'data/test/comp/'
    trimap_folder = 'data/test/trimap/'
    imglist = sorted(os.listdir(src_image_folder))

    # load images
    for imgname in imglist:
        imgpath = src_image_folder+imgname
        imgnum = imgname.split('.')[0]
        input = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        blur = cv2.imread(blur_folder+imgnum+'.jpg')
        width = input.shape[1]
        height = input.shape[0]



        # the pretrained model
        checkpoint = "./results/tnet.pth"

        # 加载模型
        checkpoint = torch.load(checkpoint)
        model = tnet()

        model = model.to(device)
        model.load_state_dict(checkpoint['tnet'])

        model.eval()

        # preprocess images
        img = cv2.resize(input, (320, 320), interpolation=cv2.INTER_CUBIC)
        img = (img.astype(np.float32) - (114., 121., 134.,)) / 255.0
        h, w, c = img.shape
        img = torch.from_numpy(img.transpose((2, 0, 1))).view(c, h, w).float()
        img = img.view(1, 3, h, w)

        # report the time
        start = time.time()

        # transport data to the device
        img = img.to(device)

        # inference the model
        with torch.no_grad():
            trimap = model(img)


            # save the trimap
            trimap = trimap.squeeze(0).float().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu',
                                                                                                     torch.uint8).numpy()
            cv2.imwrite(trimap_folder + imgnum + '.png', trimap)

            # resize and save the alpha image
            trimap = cv2.resize(trimap, (width, height), interpolation=cv2.INTER_CUBIC)

            # stack the trimap with the original image, and then generate the image with a blurred background
            trimap_f = trimap / 255.
            comp = input * trimap_f + blur * (1. - trimap_f)
            cv2.imwrite(comp_folder + imgnum + '.png', comp.astype(np.uint8))

        print('We used  {:.3f} seconds'.format(time.time() - start))
