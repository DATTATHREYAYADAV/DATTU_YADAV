import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
known_face = cv2.imread('known_face.jpg')
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()  # Read a frame from the camera

    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a rectangle around each detected face

        if known_face is not None:
            # Compare the detected face with the known face
            face_roi = gray_frame[y:y + h, x:x + w]
            result = cv2.matchTemplate(face_roi, known_face, cv2.TM_CCOEFF_NORMED)
            match = np.unravel_index(result.argmax(), result.shape)

            if result[match] > 0.8:
                cv2.putText(frame, "Known Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
