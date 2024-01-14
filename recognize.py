import cv2

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []

    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        # Predicting the id of the user
        id, _ = clf.predict(gray_img[y:y+h, x:x+w])

        # Check for id of user and label the rectangle accordingly
        if id == 1:
            cv2.putText(img, "Ali", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]

    return coords

def recognize(img, clf, face_cascade):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
    coords = draw_boundary(img, face_cascade, 1.1, 10, color["white"], "Face", clf)
    return img

# Loading classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Loading custom classifier to recognize
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.yml")

# Capturing real-time video stream. 0 for built-in web-cams, 0 or -1 for external web-cams
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Reading image from video stream
    _, img = video_capture.read()

    if img is None:
        print("Error: Failed to capture frame.")
        break

    # Call method we defined above
    img = recognize(img, clf, face_cascade)

    # Writing processed image in a new window
    cv2.imshow("Face Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing web-cam
video_capture.release()
# Destroying output window
cv2.destroyAllWindows()
