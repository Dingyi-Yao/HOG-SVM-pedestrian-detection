import cv2
from sklearn import svm
import os
import numpy as np
import joblib
from skimage.feature import hog
from skimage import io



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
        pos_feature_list.append(features.tolist())
    for img_file in neg_files:
        img_raw = io.imread(os.path.join(negpath, img_file))
        gray_img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        features = hog(gray_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", transform_sqrt=True, feature_vector=True)
        neg_count += 1
        neg_feature_list.append(features.tolist())
    return pos_feature_list,neg_feature_list, pos_count,neg_count


def eval():
    """
    评估分类器
    """
    #加载分类器参数
    print("svm------------------------")
    svm_classifier = joblib.load('svm_classifier.pkl')
    rootpath = "F:\\现代方法2\\作业2\\INRIAPerson\\test_for_class"
    pos_features, neg_features,pos_num, neg_num= dataloader(rootpath)
    #用训好的分类器进行预测
    pos_pred = svm_classifier.predict(pos_features)
    neg_pred = svm_classifier.predict(neg_features)
    # 标签是1的有多少预测结果为1 ，TP
    TP = cv2.countNonZero(pos_pred)
    FN = pos_pred.shape[0] - TP
    # 标签是0的有多少预测结果为1 ，FP
    FP = cv2.countNonZero(neg_pred)
    TN = neg_pred.shape[0] - FP
    precision = float(TP) / (TP + FP)
    recall = float(TP) / (TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    print ("Precision: " + str(precision), "Recall: " + str(recall))
    print ("F1 Score: " + str(f1))
    #高斯核
    print("rbfsvm------------------------")
    svm_classifier = joblib.load('svm_classifier_rbf_1.pkl')
    rootpath = "F:\\现代方法2\\作业2\\INRIAPerson\\test_for_class"
    pos_features, neg_features,pos_num, neg_num= dataloader(rootpath)
    #用训好的分类器进行预测
    pos_pred = svm_classifier.predict(pos_features)
    neg_pred = svm_classifier.predict(neg_features)
    # 标签是1的有多少预测结果为1 ，TP
    TP = cv2.countNonZero(pos_pred)
    FN = pos_pred.shape[0] - TP
    # 标签是0的有多少预测结果为1 ，FP
    FP = cv2.countNonZero(neg_pred)
    TN = neg_pred.shape[0] - FP
    precision = float(TP) / (TP + FP)
    recall = float(TP) / (TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    print ("Precision: " + str(precision), "Recall: " + str(recall))
    print ("F1 Score: " + str(f1))
    #logstic方法
    print("logstic------------------------")
    logistic_classifier = joblib.load('logistic_classifier.pkl')
    rootpath = "F:\\现代方法2\\作业2\\INRIAPerson\\test_for_class"
    pos_features, neg_features,pos_num, neg_num= dataloader(rootpath)
    #用训好的分类器进行预测
    pos_pred = logistic_classifier.predict(pos_features)
    neg_pred = logistic_classifier.predict(neg_features)
    # 标签是1的有多少预测结果为1 ，TP
    TP = cv2.countNonZero(pos_pred)
    FN = pos_pred.shape[0] - TP
    # 标签是0的有多少预测结果为1 ，FP
    FP = cv2.countNonZero(neg_pred)
    TN = neg_pred.shape[0] - FP
    precision = float(TP) / (TP + FP)
    recall = float(TP) / (TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    print ("Precision: " + str(precision), "Recall: " + str(recall))
    print ("F1 Score: " + str(f1))

if __name__ == '__main__':
    eval()