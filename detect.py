import cv2
from sklearn import svm
import os
import numpy as np
import joblib
from skimage.feature import hog
from skimage import io
import time

def dataloader(rootpath):
    """
    根据路径加载图片，返回特征和标签
    """
    pos_feature_list = []
    neg_feature_list = []
    pos_count = 0
    neg_count = 0

    #根据数据集位置获得正负样本文件名
    pospath = rootpath + '\\pos'
    pos_files = os.listdir(pospath)
    negpath = rootpath + '\\neg'
    neg_files = os.listdir(negpath)

    for img_file in pos_files:
        img_raw = io.imread(os.path.join(pospath, img_file))
        # print(img_raw.shape)
        # HOG特征提取算法针对的是单一通道（灰度）图像，而多通道（RGB）图像需要将其转换为灰度图像或单独处理每个通道的特征
        gray_img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

        features = hog(gray_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", transform_sqrt=True, feature_vector=True)
        pos_count += 1
        pos_feature_list.append(features)
    for img_file in neg_files:
        img_raw = io.imread(os.path.join(negpath, img_file))
        img = cv2.resize(img_raw, (64,128))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = hog(gray_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", transform_sqrt=True, feature_vector=True)
        neg_count += 1
        neg_feature_list.append(features)
    return pos_feature_list,neg_feature_list, pos_count,neg_count


def sliding_window(image, window_size, step_size,svm_classifier,probThresh):
    '''
    滑动窗口截取图片
    '''
    prob_list = []
    bboxes_list = []
    img_scale_factor_list = [1.0,1.25,0.5,0.75,0.25]
    for img_scale_factor in img_scale_factor_list:
        image = cv2.resize(image, None,fx=img_scale_factor ,fy=img_scale_factor)
        for y in range(0, image.shape[0] - window_size[1], int(step_size[1]*img_scale_factor)):
            for x in range(0, image.shape[1] - window_size[0], int(step_size[0]*img_scale_factor)):
                cropped_img = image[y:y + window_size[1], x:x + window_size[0]]
                gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                feature = hog(gray_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", transform_sqrt=True, feature_vector=True)
                # prob = svm_classifier.decision_function([feature])
                prob = svm_classifier.predict_proba([feature]).ravel()[1]
                # print(prob)
                if prob> probThresh:
                    prob_list.append(prob)
                    bbox = np.array([x, y, x + window_size[0], y + window_size[1]])/img_scale_factor
                    bboxes_list.append(bbox)
    print("1张结束")
    bboxes_list = np.array(bboxes_list)
    # print(boxes)
    prob_list = np.array(prob_list)
    return prob_list, bboxes_list

def nms(boxes, probs, overlapThresh):
    """
    执行非最大值抑制
    输入包含边界框的数组list，float格式
    """
    if len(boxes) == 0:
        return []
    # 初始化

    # print('boxes',boxes.shape)
    # print('probs',probs.shape)
    # print(probs)
    selected = []
    # 改改框的格式
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # 计算面积，以按照概率排序为索引
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(probs.ravel())
    # print(idxs)

    while len(idxs) > 0:
        # 选择概率最大的索引
        last = len(idxs) - 1
        i = idxs[last]
        selected.append(i)
        # print(selected)
        # 得到相交区域，左上和右下
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # 计算相交面积
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        # 计算iou
        iou = inter / (area[i] + area[idxs[:last]] - inter)
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(iou > overlapThresh)[0])))
    return boxes[selected,:]

def detect():
    #加载分类器参数
    svm_classifier = joblib.load('svm_classifier_rbf_1.pkl')
    rootpath = "F:\\INRIAPerson\\Test"
    window_size = [64,128]
    step_size = [20,20]

    #预测概率阈值
    probThresh = 0.9
    #交并比阈值
    overlapThresh=0.2
    #根据数据集位置获得正负样本文件名
    pospath = rootpath + '\\pos'
    pos_files = os.listdir(pospath)
    for img_file in pos_files:
        img_raw = io.imread(os.path.join(pospath, img_file))
        prob_list, bboxes_list = sliding_window(img_raw, window_size, step_size,svm_classifier, probThresh)
        # print(bboxes_list)
        boxes_selected = nms(bboxes_list, prob_list, overlapThresh)
        # print(boxes_selected)
        img_vis = img_raw.copy()
        # for i in range(bboxes_list.shape[0]):
        #     cv2.rectangle(img_vis, (bboxes_list[i, 0],bboxes_list[i, 1]), (bboxes_list[i, 2], bboxes_list[i, 3]), (0, 0, 255), 2)
        # cv2.imwrite("test_beforenms.png", img_vis)
        for i in range(boxes_selected.shape[0]):
            cv2.rectangle(img_vis, (int(boxes_selected[i, 0]), int(boxes_selected[i, 1])), (int(boxes_selected[i, 2]),int(boxes_selected[i, 3])), (0, 0, 255), 2)
        cv2.imwrite("C:\\Users\\Dean Yao\\Desktop\\detect_9_2\\"+img_file, img_vis)

if __name__ == '__main__':
    detect()