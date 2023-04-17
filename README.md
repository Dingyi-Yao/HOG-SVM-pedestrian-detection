# HOG-SVM-pedestrian-detection
西安交通大学自动化钱专业人工智能课作业。行人检测，采用HOG特征，和SVM分类器，在INRIAPerson数据集上进行检测。

由于老师要求有一些模糊，代码中加上了自己对作业的理解。

在本实验中，AUC为0.582，略高于0.5。效果不好的原因是没时间仔细调预测概率阈值(probThresh)和交并比阈值(overlapThresh)，以及img_scale_factor取值，可以根据实际情况合理调参。
