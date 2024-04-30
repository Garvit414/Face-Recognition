import paddlehub as hub
import cv2
import os
import time

# Use the default webcam (index 0)
camera_url = 0

face_detector = hub.Module(name="pyramidbox_lite_mobile")

def draw_bounding_boxes(image, faces):
    # Draw bounding boxes on the image
    for face in faces:
        left = int(face['left'])
        right = int(face['right'])
        top = int(face['top'])
        bottom = int(face['bottom'])
        confidence = face['confidence']

        # Draw a rectangle around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 1)
        label = f": {confidence:.2f}"
        cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)
    return image

def detect_faces_from_webcam(camera_url):
    # Start the webcam
    cap = cv2.VideoCapture(camera_url)

    while True:
        # Capture frame-by-frame from the webcam
        ret, frame = cap.read()

        if ret:
            # Perform face detection on the frame
            result = face_detector.face_detection(images=[frame])
            box_list = result[0]['data']
            
            # Draw bounding boxes on the frame
            img_with_boxes = draw_bounding_boxes(frame, box_list)
            
            # Display the frame with bounding boxes
            cv2.imshow("Webcam", img_with_boxes)
        
        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_faces_from_webcam(camera_url)
