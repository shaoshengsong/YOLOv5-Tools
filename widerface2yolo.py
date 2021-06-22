import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

class WiderFaceDetection(data.Dataset):
    def __init__(self, txt_path, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        f = open(txt_path,'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = txt_path.replace('label.txt','images/') + path
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        self.words.append(labels)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        labels = self.words[index]
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            # landmarks
            annotation[0, 4] = label[4]    # l0_x
            annotation[0, 5] = label[5]    # l0_y
            annotation[0, 6] = label[7]    # l1_x
            annotation[0, 7] = label[8]    # l1_y
            annotation[0, 8] = label[10]   # l2_x
            annotation[0, 9] = label[11]   # l2_y
            annotation[0, 10] = label[13]  # l3_x
            annotation[0, 11] = label[14]  # l3_y
            annotation[0, 12] = label[16]  # l4_x
            annotation[0, 13] = label[17]  # l4_y
            if (annotation[0, 4]<0):
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)


def landmark_normalization(x,scale):
    if x < 0: 
        x=0
    if x >= scale: 
        x=x-1
    x = x / scale
    return x

if __name__ == "__main__":
    
    #图片和文件分别独立的文件夹
    save_path = '/media/ubuntu/data/pytorch1.7/widerface/'
    wfd=WiderFaceDetection("/media/ubuntu/data/dataset/original-widerface/WIDER_train/label.txt")

    #按照yolov5的方式创建文件夹结构
    phase = 'train'
    images_path=(save_path + "/images/" + phase)
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    labels_path=(save_path + "/labels/" + phase)
    if not os.path.exists(labels_path):
        os.makedirs(labels_path)
    
    for i in range(len(wfd.imgs_path)):
        print(i, wfd.imgs_path[i])
        img = cv2.imread(wfd.imgs_path[i])
        base_img = os.path.basename(wfd.imgs_path[i])#文件名和扩展名
        base_txt = os.path.basename(wfd.imgs_path[i])[:-4] +".txt"
        save_img_path = os.path.join(images_path, base_img)
        save_txt_path = os.path.join(labels_path, base_txt)
        with open(save_txt_path, "w") as f:
            height, width, _ = img.shape
            labels = wfd.words[i]
            annotations = np.zeros((0, 14))
            if len(labels) == 0:
                continue
            for idx, label in enumerate(labels):
                annotation = np.zeros((1, 14))
                # bbox 防止边框超出图片大小
                label[0] = max(0, label[0])
                label[1] = max(0, label[1])
                label[2] = min(width -  1, label[2])
                label[3] = min(height - 1, label[3])
                #widerface box 转yolov5 box归一化存储
                annotation[0, 0] = (label[0] + label[2] / 2) / width  # cx
                annotation[0, 1] = (label[1] + label[3] / 2) / height  # cy
                annotation[0, 2] = label[2] / width  # w
                annotation[0, 3] = label[3] / height  # h
        
                # landmarks 也跟着归一化， 防止超出图片外 ，原始-1表示没有关键点，这里用10个数都是0，表示没有关键点。
                annotation[0, 4] = landmark_normalization(label[4],width) # 0_x
                annotation[0, 5] = landmark_normalization(label[5], height)  # 0_y
                annotation[0, 6] = landmark_normalization(label[7] , width ) # 1_x
                annotation[0, 7] = landmark_normalization(label[8]  , height) # 1_y
                annotation[0, 8] = landmark_normalization(label[10] , width ) # 2_x
                annotation[0, 9] = landmark_normalization(label[11] , height  )# 2_y
                annotation[0, 10] = landmark_normalization(label[13] , width ) # 3_x
                annotation[0, 11] = landmark_normalization(label[14] , height) # 3_y
                annotation[0, 12] = landmark_normalization(label[16] , width ) # 4_x
                annotation[0, 13] = landmark_normalization(label[17] , height ) # 4_y
                
               
                str_cls="0 "

                for i in range(len(annotation[0])):
                    str_cls =str_cls+" "+str(annotation[0][i])

                str_cls = str_cls + '\n'
                f.write(str_cls)
        cv2.imwrite(save_img_path, img)

