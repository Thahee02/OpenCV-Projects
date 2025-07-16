import cv2 as cv

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture from the default camera
cap = cv.VideoCapture(0)

while True:
    success, img = cap.read()

    # Check if the frame was captured successfully
    if not success:
        break

    # Convert the image to grayscale for face detection
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the count of detected faces
    cv.putText(img, f'Faces: {len(faces)}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame with detected faces
    cv.imshow("Face Detection", img)

    # Break the loop if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()