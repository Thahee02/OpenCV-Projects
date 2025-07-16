import cv2 as cv
import mediapipe as mp
import time

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection

# Create a Face Detection object with a minimum detection confidence
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# Initialize MediaPipe Drawing Utils
mpDraw = mp.solutions.drawing_utils

# Start video capture
cam = cv.VideoCapture(0)

# Initialize timing variables for FPS calculation
pTime = 0
cTime = 0

while True:
    success, img = cam.read()
    if not success:
        break

    # Convert the image to RGB for MediaPipe processing
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # Process the image and detect faces
    results = face_detection.process(imgRGB)

    if results.detections:
        for detection in results.detections:

            # Draw the detection results on the image
            bboxC = detection.location_data.relative_bounding_box
            
            # Get the bounding box coordinates
            ih, iw, _ = img.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # Draw rectangle around the face
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw landmarks and confidence score
            cv.putText(img, f'{int(detection.score[0] * 100)}% - Hathil Thahee', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Display FPS and number of detected faces
    cv.putText(img, f'FPS: {int(fps)}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv.putText(img, f'Faces: {len(results.detections) if results.detections else 0}', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

    cv.imshow("Face Detection", img)

    # Exit on 'q' key press
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cam.release()
cv.destroyAllWindows()