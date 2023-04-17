import cv2
from sklearn import svm
import os
import numpy as np
import joblib
from skimage.feature import hog
from skimage import io
import time
from sklearn.metrics import auc,roc_curve
import re

import matplotlib.pyplot as plt

global NUM
NUM = 0

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

def dataloader(rootpath):
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
    global NUM
    prob_list = []
    bboxes_list = []
    img_scale_factor_list = [1.0,0.5,0.33,0.25,0.2]
    for img_scale_factor in img_scale_factor_list:
        try:
            image = cv2.resize(image, None, fx=img_scale_factor , fy=img_scale_factor)
        except AttributeError:
                print(NUM+1)
                pass
        
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
    NUM +=  1
    print(str(NUM)+"张结束")
    bboxes_list = np.array(bboxes_list)
    # print(boxes)
    prob_list = np.array(prob_list)
    return prob_list, bboxes_list

def nms(boxes, probs, overlapThresh,prob_list):
    """
    执行非最大值抑制
    输入包含边界框的数组list，float格式
    """
    if len(boxes) == 0:
        return [],[]
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
    return boxes[selected,:],prob_list[selected]

def detect():
    #加载分类器参数
    svm_classifier = joblib.load('svm_classifier_rbf_1.pkl')
    rootpath = "F:\\INRIAPerson\\Test"
    annpath = "F:\\现代方法2\\作业2\\INRIAPerson\\Test\\annotations"
    window_size = [64,128]
    step_size = [12,12]

    #预测概率阈值
    probThresh = 0.95
    #交并比阈值
    overlapThresh=0.05
    #根据数据集位置获得正负样本文件名
    pospath = rootpath + '\\pos'
    pos_files = os.listdir(pospath)
    gt_label_list = []
    prob_list_final = []
    for img_file in pos_files:
        img_raw = io.imread(os.path.join(pospath, img_file))

        prob_list, bboxes_list = sliding_window(img_raw, window_size, step_size,svm_classifier, probThresh)
        # print(bboxes_list)
        boxes_selected,prob_list_selected = nms(bboxes_list, prob_list, overlapThresh,prob_list)
        if len(prob_list_selected) !=0:
            gt_label_list_single, prob_list_single = eval(img_file, annpath, boxes_selected, prob_list_selected)
            
            gt_label_list.extend(gt_label_list_single)
            prob_list_final.extend(prob_list_single)
            # print(boxes_selected)
            index = [2,1,0]
            img_vis = img_raw[:,:,index].copy()
            # for i in range(bboxes_list.shape[0]):
            #     cv2.rectangle(img_vis, (bboxes_list[i, 0],bboxes_list[i, 1]), (bboxes_list[i, 2], bboxes_list[i, 3]), (0, 0, 255), 2)
            # cv2.imwrite("test_beforenms.png", img_vis)
            for i in range(boxes_selected.shape[0]):
                cv2.rectangle(img_vis, (int(boxes_selected[i, 0]), int(boxes_selected[i, 1])), (int(boxes_selected[i, 2]),int(boxes_selected[i, 3])), (0, 0, 255), 2)
            save_path = "C:\\Users\\Dean Yao\\Desktop\\detect_95_2_new\\"
            cv2.imwrite(save_path+img_file, img_vis)
        #绘制曲线
    fpr, tpr, thresholds = roc_curve(gt_label_list, prob_list_final)

    # 计算AUC
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    path = "C:\\Users\\Dean Yao\\Desktop\\auc"
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    plt.savefig(path)
def caliou(bbox, bbox_list):
    """
    输入一个框，和所有gt框计算iou，并返回最大值
    """
    max = 0
    x1_p = bbox[0]
    y1_p = bbox[1]
    x2_p = bbox[2]
    y2_p = bbox[3]
    area_p = (x2_p - x1_p) *  (y2_p - y1_p)
    for i in range(len(bbox_list)):
        x1_gt = bbox_list[i][0]
        y1_gt = bbox_list[i][1]
        x2_gt = bbox_list[i][2]
        y2_gt = bbox_list[i][3]
        area_gt = (x2_gt - x1_gt) *  (y2_gt - y1_gt)
        xx1 = np.maximum(x1_p, x1_gt)
        yy1 = np.maximum(y1_p, y1_gt)
        xx2 = np.minimum(x2_p, x2_gt)
        yy2 = np.minimum(y2_p, y2_gt)

        # 计算相交面积
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        # 计算iou
        iou = inter / (area_p + area_gt- inter)
        if iou > max:
            max = iou
    return max

def eval(img_file, annpath, boxes_selected,prob_list):
    """
    读入标签，和候选框计算IOU，IOU即为预测正确与否的概率值，再调用sklearn里的AUC函数，获得图像
    """

    gt_label_list = []
    ann_file_name = img_file.replace('.png','.txt')
    bbox_list_pos = load_gt_ann_single(annpath, ann_file_name)
    for box_selected in boxes_selected:
        prob = caliou(box_selected, bbox_list_pos)
        if prob > 0.1:
            gt_label_list.append(1)
        else :
            gt_label_list.append(0)
    # print(prob_list)
    prob_list_tolist= prob_list.tolist()
    return gt_label_list, prob_list_tolist
 
if __name__ == '__main__':
    detect()