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
        for bbox in bbox_list_pos :
            # 获取子图像的坐标范围
            xmin, ymin, xmax, ymax = bbox
            # 截取子图像
            sub_img = img_raw[ymin:ymax, xmin:xmax]
            resized_sub_img = cv2.resize(sub_img, (64,128))
            # io.imsave("F:\\现代方法2\\作业2\\论文\\"+img_file,resized_sub_img)
            # HOG特征提取算法针对的是单一通道（灰度）图像，而多通道（RGB）图像需要将其转换为灰度图像或单独处理每个通道的特征
            gray_img = cv2.cvtColor(resized_sub_img , cv2.COLOR_BGR2GRAY)
            ## 可视化hog特征图
            # vis_gray_img = cv2.cvtColor(resized_sub_img, cv2.COLOR_BGR2GRAY)
            # features,hog_image = hog(vis_gray_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", visualize=True)
            # print(hog_image.shape)
            # plt.imshow(hog_image,cmap=plt.cm.gray)
            # plt.show()
            # exit()
            ##
            features = hog(gray_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", transform_sqrt=True, feature_vector=True)
            cropped_pos_img_list.append(resized_sub_img)
            pos_count += 1
            feature_list.append(features)
            label.append(1)
        # 1237张正样本
        
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
            gray_img = cv2.cvtColor(cropped_neg_img, cv2.COLOR_BGR2GRAY)
            features = hog(gray_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", transform_sqrt=True, feature_vector=True)
            neg_count += 1
            feature_list.append(features)
            label.append(0)
            if neg_count == pos_count :
                return feature_list, label


def train():
    annpath = "F:\\现代方法2\\作业2\\INRIAPerson\\Train\\annotations"
    rootpath = "F:\\INRIAPerson\\Train"
    feature_list, label= dataloader(rootpath,annpath)
    feature_list = np.array(feature_list)
    label = np.array(label)
    feature_list, label= shuffle(feature_list, label, random_state=0)
    #设置SVM分类器
    # svm_classifier = svm.LinearSVC(C=0.001, max_iter=1000, class_weight='balanced')
    svm_classifier = svm.SVC(C=1, kernel="rbf",class_weight='balanced', probability=True)
    svm_classifier.fit(feature_list, label)
    #保存分类器
    joblib.dump(svm_classifier, 'svm_classifier_rbf_1.pkl')

if __name__ == '__main__':
    train()