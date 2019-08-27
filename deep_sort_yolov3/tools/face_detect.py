import os
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
face_save_path = os.path.join(BASE_DIR, 'data', 'face')
if not os.path.exists(face_save_path):
    os.makedirs(face_save_path)

# 人脸计数
face_num = len(os.listdir(face_save_path))

# 加载人脸特征库
if __name__ == '__main__':
    face_cascade_name = os.path.join('..', 'model_data', 'haarcascade_frontalface_default.xml')
else:
    face_cascade_name = os.path.join('model_data', 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier()
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)


def face_detect(img, save=True, show=False):
    """检测并保存人脸"""
    global face_num
    if img is None:
        return False
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        cv2.imwrite('aa.jpg', img)
        exit(0)
    rects = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in rects:
        print('Face Location:', x, y, x + w, y + h)
        # 保存人脸
        if save:
            x = x - 20 if x - 20 > 0 else 0
            y = y - 20 if y - 20 > 0 else 0
            w += 40
            h += 40
            cv2.imwrite(os.path.join(face_save_path, str(face_num) + '.jpg'), img[y:y + h, x:x + w, :])
            face_num += 1
            print('Save Face', face_num)
        # 用矩形圈出人脸
        if show:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if show and len(rects) > 0:
        cv2.imshow('face', img)
    return len(rects) > 0


# def draw_rects(img, rects, color):
#     for x1, y1, x2, y2 in rects:
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


if __name__ == '__main__':

    url = 'rtmp://58.200.131.2:1935/livetv/hunantv'
    camera_device = 0
    # -- 2. Read the video stream
    cap = cv2.VideoCapture(0)
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)

    while True:
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
        face_detect(frame, save=True, show=True)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
