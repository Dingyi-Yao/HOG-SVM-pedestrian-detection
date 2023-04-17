import cv2
from sklearn import svm
import os
import numpy as np
import joblib
from skimage.feature import hog
from skimage import io
from sklearn.utils import shuffle
import sys
import matplotlib.pyplot as plt
import random
import re
import os

def load_gt_ann_single(gt_filepath, ann_file_name):
    """
    加载标签文件
    返回一个字典，key是图片名字，value是数组，没给数组元素是(Xmin, Ymin ,Xmax, Ymax)元组
    """
    dict_gt_ann = dict()
    ann_txt = os.path.join(gt_filepath, ann_file_name)
    with open(ann_txt, 'r') as f:
        # 读取文件内容
        content = f.read()
        # 使用正则表达式匹配所有边界框的坐标信息
        bbox_pattern = re.compile(r'Bounding box for object \d+ "PASperson" \(Xmin, Ymin\) - \(Xmax, Ymax\) : (.+)')
        bbox_matches = bbox_pattern.findall(content)
        # 解析坐标信息
        bbox_list = []
        for bbox_match in bbox_matches:
            bbox = bbox_match.split(' - ')
            bbox = [re.findall(r'\d+', coord) for coord in bbox]
            bbox = tuple([int(coord) for coord in sum(bbox, [])])
            bbox_list.append(bbox)
            # print(bbox_list)
            # (Xmin, Ymin) - (Xmax, Ymax)格式
    return bbox_list

def dataloader(rootpath, annpath):
    """
    根据路径加载图片，返回特征和标签
    """
    feature_list = []
    label = []
    pos_count = 0
    neg_count = 0

    #根据数据集位置获得正负样本文件名
    pospath = rootpath + '\\pos'
    pos_files = os.listdir(pospath)
    negpath = rootpath + '\\neg'
    neg_files = os.listdir(negpath)

    for img_file in pos_files:
        ann_file_name = img_file.replace('.png','.txt')
        img_raw = io.imread(os.path.join(pospath, img_file))
        cropped_pos_img_list = []
        bbox_list_pos = load_gt_ann_single(annpath, ann_file_name)
        bbox_idx = 0
        for bbox in bbox_list_pos :
            # 获取子图像的坐标范围
            xmin, ymin, xmax, ymax = bbox
            # 截取子图像
            sub_img = img_raw[ymin:ymax, xmin:xmax]
            resized_sub_img = cv2.resize(sub_img, (64,128))
            io.imsave("F:\\现代方法2\\作业2\\INRIAPerson\\test_for_class\\pos\\"+str(bbox_idx)+img_file,resized_sub_img)
            bbox_idx += 1
        # HOG特征提取算法针对的是单一通道（灰度）图像，而多通道（RGB）图像需要将其转换为灰度图像或单独处理每个通道的特征
    for img_file in neg_files:
            #负样本随机构建，由于正样本数量1237张，原数据集负样本图片1218张，所以每个图片随机截图两个
            img_raw = io.imread(os.path.join(negpath, img_file))
            h, w , c= img_raw.shape
            if h < 128 or w < 64:
                continue
            h = h - 128
            w = w - 64
            random.seed(1)
            for i in range(2):
                x = random.randint(0, w)
                y = random.randint(0, h)
                cropped_neg_img = img_raw[y:y+128, x:x+64]
                io.imsave("F:\\现代方法2\\作业2\\INRIAPerson\\test_for_class\\neg\\"+str(i)+img_file,cropped_neg_img)
                neg_count += 1
                if neg_count == 589 :
                    return feature_list, label

annpath = "F:\\现代方法2\\作业2\\INRIAPerson\\Test\\annotations"
rootpath = "F:\\INRIAPerson\\Test"
feature_list, label= dataloader(rootpath,annpath)