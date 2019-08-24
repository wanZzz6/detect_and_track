# -*- coding: utf-8 -*-

import os
from timeit import time
import warnings
import imutils
import cv2
import numpy as np
import argparse
from PIL import Image
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from collections import deque
from keras import backend

backend.clear_session()
warnings.filterwarnings('ignore')

url = 'rtmp://58.200.131.2:1935/livetv/hunantv'
video = os.path.join('video', 'chaplin.mp4')

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="path to input video", default=0)
ap.add_argument("-p", "--prototxt", required=False,
                default='model_data/MobileNetSSD_deploy.prototxt.txt',
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=False,
                default='model_data/MobileNetSSD_deploy.caffemodel',
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
                help="minimum probability to filter weak detections")

args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

NEED_CLASSES = {'car', 'person'}
# NEED_CLASSES = set(CLASSES)

# 记录运动轨迹坐标
pts = [deque(maxlen=30) for _ in range(9999)]

# initialize a list of colors to represent each possible class label
np.random.seed(100)
# 能标记200个目标的颜色
COLORS = np.random.randint(0, 255, size=(200, 3),
                           dtype="uint8")


def main():
    global frame_index, out, list_file, count, class_name, track

    time.sleep(2.0)
    start = time.time()

    # 参数定义
    max_cosine_distance = 0.5  # 0.9 余弦距离的控制阈值
    nn_budget = None
    nms_max_overlap = 0.3  # 非极大抑制的阈值
    # 是否保存识别结果
    write_video_flag = True

    counter = []

    # load our serialized model from disk
    # print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    # deep_sort
    model_filename = 'model_data/market1501.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    video_capture = cv2.VideoCapture(args["input"])
    obj_count_txt_filename = 'counter.txt'
    count_file = open(obj_count_txt_filename, 'a')
    count_file.write('\n')

    if write_video_flag:
        # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(os.path.join('video', 'output.avi'), fourcc, 15,
                              (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1
    # 帧率计数
    fps = 0.0

    while True:

        ret, frame = video_capture.read()  # frame shape 640*480*3
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        time1 = time.time()

        # image = Image.fromarray(frame)
        frame = imutils.resize(frame, width=800)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)
        # predictions 检测
        time2 = time.time()

        net.setInput(blob)
        detections = net.forward()

        # detections.shape
        # >>> (1, 1, n, 7)
        # eg:(1, 1, 2, 7)
        # [[[[0.          9.          0.42181703  0.4647404   0.610577
        #     0.6360997   0.8479532]
        #    [0.         15.          0.8989926   0.21603307  0.42735672
        #    0.58441484  0.8699994]]]]
        time3 = time.time()
        print('detect cost is', time3 - time2)
        boxs = []
        class_names = []
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            # greater than the minimum confidence
            if confidence > args["confidence"]:
                idx = int(detections[0, 0, i, 1])
                class_name = CLASSES[idx]
                class_names.append(class_name)
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                # 转为整形坐标
                (startX, startY, endX, endY) = box.astype("int")
                startX = 0 if startX < 0 else startX
                startY = 0 if startY < 0 else startY

                boxs.append([startX, startY, endX - startX, endY - startY])

        print(boxs, class_names)
        time3 = time.time()
        print('detect cost is', time3 - time2)

        # 特征提取
        features = encoder(frame, boxs)
        # score to 1.0 here).
        detections = [Detection(bbox, class_name, 1.0, feature) for bbox, class_name, feature in
                      zip(boxs, class_names, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        time4 = time.time()
        print('features extract is', time4 - time3)

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        time5 = time.time()
        print('update tracker cost:', time5 - time4)

        i = 0
        # 跟踪器id
        indexIDs = []

        for track in tracker.tracks:

            # todo and or
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            # boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(track.track_id)
            counter.append(track.track_id)
            bbox = track.to_tlbr()
            color = COLORS[indexIDs[i] % len(COLORS)].tolist()
            # 画目标跟踪框、id标注
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 3)
            cv2.putText(frame, track.class_name + str(track.track_id), (int(bbox[0]), int(bbox[1] - 40)), 0, 0.75,
                        color, 2)

            i += 1
            # 画运动轨迹 draw motion path
            center = int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2)
            pts[track.track_id].append(center)
            thickness = 5
            cv2.circle(frame, center, 1, color, thickness)

            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                    continue
                thickness = int(np.sqrt(64 / (j + 1.0)) * 2)
                cv2.line(frame, (pts[track.track_id][j - 1]), (pts[track.track_id][j]), color, thickness)
                # cv2.putText(frame, str(class_names[j]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (255,255,255),2)

        # 画目标检测白框
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)

        count = len(set(counter))
        cv2.putText(frame, "Total Object Counter: " + str(count), (20, 120), 0, 0.75, (0, 255, 0), 2)
        cv2.putText(frame, "Current Object Counter: " + str(i), (20, 80), 0, 0.75, (0, 255, 0), 2)
        cv2.putText(frame, "FPS: %f" % fps, (20, 40), 0, 1.0, (0, 255, 0), 2)
        time6 = time.time()
        print('Draw Rectangle and Text cost:', time6 - time5)

        cv2.namedWindow("YOLO3_Deep_SORT", 0)
        cv2.resizeWindow('YOLO3_Deep_SORT', 1024, 768)
        cv2.imshow('YOLO3_Deep_SORT', frame)

        if write_video_flag:
            # save a frame
            out.write(frame)
            frame_index += 1
            list_file.write(str(frame_index) + ' ')
            if len(boxs) != 0:
                for i in range(0, len(boxs)):
                    list_file.write(
                        str(boxs[i][0]) + ' ' + str(boxs[i][1]) + ' ' + str(boxs[i][2]) + ' ' + str(boxs[i][3]) + ' ')
            list_file.write('\n')
        fps = (fps + (1. / (time.time() - time1))) / 2
        # print(set(counter))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print("[Finish]")
    end = time.time()

    if len(pts[track.track_id]):
        print(str(args["input"]) + ": " + str(count) + 'target Found')
        count_file.write(str("[VIDEO]: " + args["input"]) + " " + (
            str(count)) + " " + "[MODEL]: yolo_cc_0612.h5" + " " + "[TIME]:" + (str('%.2f' % (end - start))))
    else:
        print("[No Found]")

    video_capture.release()
    count_file.write('\n')
    count_file.close()
    if write_video_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
