import cv2
import paddlehub as hub
import sys
sys.path.append('E:\FR_Paddle\insight-face-paddle')
import insightface_paddle as face
import os

def detect_face(image, detector):
    result = detector.face_detection(images=[image], use_gpu=False)
    return result[0]['data']

def recognize_face(image, box_list, recognizer):
    img = image[:, :, ::-1]
    res = list(recognizer.predict(img, box_list))
    return res[0]['box_list'], res[0]['labels']

def draw_boundary_boxes(image, box_list, labels):
    for box, label in zip(box_list, labels):
        score = "{:.2f}".format(box['confidence'])
        x_min, y_min, x_max, y_max = int(box['left']), int(box['top']), int(box['right']), int(box['bottom'])
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        label = label + str(score)
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

def capture_dataset(detector, recognizer, dataset_dir, num_images_per_label=10):
    labels = set()
    current_label = None
    current_label_count = 0

    webcam = cv2.VideoCapture(0)

    while True:
        ret, frame = webcam.read()

        if ret:
            resized_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
            box_list = detect_face(resized_frame, detector)
            box_list, labels = recognize_face(resized_frame, box_list, recognizer)

            for label in labels:
                labels.add(label)

            if current_label is not None:
                if current_label_count < num_images_per_label:
                    image_filename = f"{current_label}_{current_label_count}.jpg"
                    image_path = os.path.join(dataset_dir, current_label, image_filename)
                    cv2.imwrite(image_path, frame)
                    current_label_count += 1
                else:
                    current_label_count = 0
                    current_label = None

            cv2.imshow("Webcam", resized_frame)
        else:
            print("Failed to capture frame")
            break

        if cv2.waitKey(1) == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()

def main():
    face_detector = hub.Module(name="pyramidbox_lite_mobile")
    parser = face.parser()
    args = parser.parse_args()

    args.use_gpu = False
    args.enable_mkldnn = True
    args.cpu_threads = 4
    args.det = False
    args.rec = True
    args.rec_thresh = 0.45
    args.index = "Dataset/index.bin"
    args.rec_model = "Models/mobileface_v1.0_infer"
    recognizer = face.InsightFace(args)

    dataset_dir = "Dataset"  # Change this to your desired dataset directory
    os.makedirs(dataset_dir, exist_ok=True)

    capture_dataset(face_detector, recognizer, dataset_dir)

if __name__ == '__main__':
    main()
