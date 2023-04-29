import os
import cv2
import json

def getFileList(dir, suffix):
    res = []
    names_1 = sorted(os.listdir(dir))
    for name_1 in names_1:
        names_2 = sorted(os.listdir(dir+name_1))
        for name_2 in names_2:
            pic_names = sorted(os.listdir(dir+name_1+'/'+name_2))
            for pic_name in pic_names:
                if pic_name.endswith(suffix):
                    res.append(dir+name_1+'/'+name_2+'/'+pic_name)
    res.sort()
    return res

def erode_dilate(mask, size=(10, 10), smooth=True):
    """
    Erode and dilate in order to generate the trimap
    input mask：binary mask with only one channel
    """
    # generate the kernel
    if smooth:
        size = (size[0] - 4, size[1] - 4)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)

    # dilate
    dilated = cv2.dilate(mask, kernel, iterations=1)
    if smooth:
        dilated[(dilated > 5)] = 255
        dilated[(dilated <= 5)] = 0
    else:
        dilated[(dilated > 0)] = 255

    # erode
    eroded = cv2.erode(mask, kernel, iterations=1)
    if smooth:
        eroded[(eroded < 250)] = 0
        eroded[(eroded >= 250)] = 255
    else:
        eroded[(eroded < 255)] = 0

    res = dilated.copy()
    res[((dilated == 255) & (eroded == 0))] = 128
    return res


def genAiFenGe():
    """
    Generate standard AiFenGe dataset and JSON files
    """
    # set copy paths
    src_img_folder = 'matting_human_half/clip_image/'
    src_alpha_folder = 'matting_human_half/matting/'
    des_img_folder = './data/AiFenGe/img'
    des_alpha_folder = './data/AiFenGe/alpha'
    des_trimap_folder = './data/AiFenGe/trimap'

    if not os.path.exists(des_img_folder):
        os.makedirs(des_img_folder)
    if not os.path.exists(des_alpha_folder):
        os.makedirs(des_alpha_folder)
    if not os.path.exists(des_trimap_folder):
        os.makedirs(des_trimap_folder)

    # retrieve files
    imglist = getFileList(src_img_folder, 'jpg')
    alphalist = getFileList(src_alpha_folder, 'png')

    print('Find ' + str(len(imglist)) + ' original images.')
    print('Find ' + str(len(alphalist)) + ' alpha images.')

    # retrieve every image
    index = 0
    save_img_list = list()
    save_alpha_list = list()
    save_trimap_list = list()

    for imgpath in imglist:
        imgname = os.path.splitext(os.path.basename(imgpath))[0]
        alphaname = imgname + '.png'

        for j in range(len(alphalist)):
            if alphaname in alphalist[j]:
                alphapath = alphalist[j]
                try:
                    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)

                    alpha = cv2.imread(alphapath, cv2.IMREAD_UNCHANGED)
                    alpha = alpha[:, :, 3]  # separate alpha channels
                    ret, alpha = cv2.threshold(alpha, 50, 255, cv2.THRESH_BINARY)

                    # generate the trimap
                    trimap = erode_dilate(alpha)

                    # save
                    cv2.imwrite(des_img_folder + ('/%d.png' % (index)), img)
                    cv2.imwrite(des_alpha_folder + ('/%d.png' % (index)), alpha)
                    cv2.imwrite(des_trimap_folder + ('/%d.png' % (index)), trimap)

                    # record
                    save_img_list.append(des_img_folder + ('/%d.png' % (index)))
                    save_alpha_list.append(des_alpha_folder + ('/%d.png' % (index)))
                    save_trimap_list.append(des_trimap_folder + ('/%d.png' % (index)))

                    index += 1
                    print('We have written %d images.' % (index))

                except Exception as err:
                    print(err)

    # write json files
    with open('./data/aifenge_img.json', 'w') as jsonfile1:
        json.dump(save_img_list, jsonfile1)

    with open('./data/aifenge_alpha.json', 'w') as jsonfile2:
        json.dump(save_alpha_list, jsonfile2)

    with open('./data/aifenge_trimap.json', 'w') as jsonfile3:
        json.dump(save_trimap_list, jsonfile3)

    print('We wrote %d images.' % (index))

genAiFenGe()