import cv2

url = 'rtsp://admin:admin777@10.86.77.35:554/h264/ch1/sub/av_stream'
cap = cv2.VideoCapture(url)

while (1):
    ret, img = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()  # 释放摄像头
cv2.destroyAllWindows()  # 释放窗口资源