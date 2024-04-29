import paddlehub as hub
import sys
sys.path.append('E:\FR_Paddle\insight-face-paddle')
import insightface_paddle as face
import logging
logging.basicConfig(level=logging.INFO)
import cv2
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
from src.VideoStream import VideoStream

face_detector = hub.Module(name="pyramidbox_lite_mobile")
parser = face.parser()
args = parser.parse_args()

args.use_gpu = True
args.det = False
args.rec = True
args.rec_thresh = 0.45
args.index = "Dataset/index.bin"
args.rec_model = "Models/mobileface_v1.0_infer"
recognizer = face.InsightFace(args)

def detect_face(image):
    result = face_detector.face_detection(images=[image], use_gpu=True)
    box_list = result[0]['data']
    return box_list

def recognize_face(image, box_list):
    img = image[:, :, ::-1]
    res = list(recognizer.predict(images=[img], boxes=[box_list], print_info=True))
    box_list = res[0]['box_list']
    labels = res[0]['labels']
    return box_list, labels

def draw_boundary_boxes(image, box_list, labels):
    for box,label in zip(box_list,labels):
        score = "{:.2f}".format(box['confidence'])
        x_min, y_min, x_max, y_max = int(box['left']), int(box['top']), int(box['right']), int(box['bottom'])
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        label = label+str(score)
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

def detection_video_stream():
    cap = cv2.VideoCapture(0)  # 0 for the default webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
        box_list = detect_face(resized_frame)
        box_list, labels = recognize_face(resized_frame, box_list)
        draw_boundary_boxes(resized_frame, box_list, labels)
        cv2.imshow("Webcam", resized_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detection_video_stream()
