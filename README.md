# YOLOv5-Tools

## general_json2yolo.py
COCO格式转YOLOv5格式

## check_yolov5_label_format.py
在各种格式转到YOLOv5格式之后，防止转换错误，最后检查一下，可视化一下标注结果。

## look_up_anchor.py
查看anchor的数值是多少

## voc2yolo.py
VOC格式转YOLOv5格式

## 混淆矩阵
根据检测框和GT boxes输出混淆矩阵（TP，FN，FP，TN）据此可以计算模型指标
实现文件general.py
使用方法confusion_matrix_test.py


## widerface2yolo.py
widerface人脸数据集转yolov5格式
