from multiprocessing import shared_memory, Semaphore

import cv2
import numpy as np
import time
import mediapipe as mp


class BoundingBox:
    def __init__(self, box: list):
        self.x1, self.y1, self.width, self.height = box[0], box[1], box[2], box[3]
        self.x2, self.y2 = (self.x1 + self.width), (self.y1 + self.height)

    def __str__(self):
        return "LeftTop = " + str((self.x1, self.y1)) + "RightDown = " + str((self.x2, self.y2))


def check_available(human_box: BoundingBox, machine_box: BoundingBox):
    x, x2, y, y2 = human_box.x1, human_box.x2, human_box.y1, human_box.y2
    tx, ty, th, tw = machine_box.x1, machine_box.y1, machine_box.height, machine_box.width
    if x > x2:
        x, x2 = x2, x
    if y > y2:
        y, y2 = y2, y
    if x < tx + tw and x2 > tx and y < ty + th and y2 > ty:
        a = max(x, tx)
        b = max(y, ty)
        a2 = min(x2, tx + tw)
        b2 = min(y2, ty + th)
        area = (a2 - a) * (b2 - b)
        if area > tw * th / 2:
            return True
        else:
            return False
    else:
        return False


def get_bounding_box_of_human(camera_num: int, process_title: str = None, shared: str = None,
                              shape=None, datatype=None, sem: Semaphore = None):
    cap = cv2.VideoCapture(camera_num)
    yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

    ret, frame = cap.read()

    while True:
        cv2.imwrite('output' + process_title + '.jpg', frame)
        src = cv2.imread('output' + process_title + '.jpg', cv2.IMREAD_COLOR)
        roi = cv2.selectROI(src)
        print('roi = ', roi)
        break

    cv2.destroyAllWindows()

    # using gpu
    yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    with open("yolo.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = yolo_net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in yolo_net.getUnconnectedOutLayers()]

    while True:
        flag = False
        ret, frame = cap.read()
        # using gpu
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)

        h, w, c = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        yolo_net.setInput(blob)
        outs = yolo_net.forward(output_layers)
        class_ids = []
        confidences = []
        list_of_boxes = []  # 탐지된 오브젝트의 바운딩 박스들을 저장해두는 리스트
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # 사람만 입력 받겠다
                if str(classes[class_id]) == "person":
                    if confidence > 0.5:
                        center_x = int(detection[0] * w)
                        center_y = int(detection[1] * h)
                        dw = int(detection[2] * w)
                        dh = int(detection[3] * h)
                        x = int(center_x - dw / 2)
                        y = int(center_y - dh / 2)

                        # 탐지된 바운딩 박스를 리스트에 추가
                        list_of_boxes.append([x, y, dw, dh])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(list_of_boxes, confidences, 0.45, 0.4)
        tx, ty, tw, th = roi

        for i in range(len(list_of_boxes)):
            if i in indexes:
                x, y, w, h = list_of_boxes[i]
                label = str(classes[class_ids[i]])
                score = confidences[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
                cv2.putText(frame, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, (0, 0, 0), 1)
                if check_available(BoundingBox(list_of_boxes[i]), BoundingBox(roi)):
                    cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), (0, 0, 255), 5)
                    flag = True
                else:
                    cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 5)
                    flag = False

        if len(list_of_boxes) == 0:
            cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), (255, 255, 0), 5)

        if __name__ != "__main__":
            sem.acquire()
            connect_shared = shared_memory.SharedMemory(name=shared)  # name으로 지정된 공유메모리 연결
            temp_arr = np.ndarray(shape=shape, dtype=datatype, buffer=connect_shared.buf)
            # TEST CODE
            for i in range(shape[0]):
                temp_arr[i] = flag

            sem.release()

        cv2.imshow(process_title, frame)
        if cv2.waitKey(10) > 0:
            break


if __name__ == "__main__":
    get_bounding_box_of_human(0, '0')
