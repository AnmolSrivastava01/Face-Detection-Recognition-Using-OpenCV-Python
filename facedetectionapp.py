import cv2

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []

    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords.append([x, y, w, h])

    return coords

def detect(img, faceCascade, eyeCascade, noseCascade, mouthCascade):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Face")

    if len(coords) == 1:
        roi_img = img[coords[0][1]:coords[0][1] + coords[0][3], coords[0][0]:coords[0][0] + coords[0][2]]
        draw_boundary(roi_img, eyeCascade, 1.1, 12, color['red'], "Eye")
        draw_boundary(roi_img, noseCascade, 1.1, 4, color['green'], "Nose")
        draw_boundary(roi_img, mouthCascade, 1.1, 20, color['white'], "Mouth")

    return img

# Loading classifiers
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
noseCascade = cv2.CascadeClassifier('Nariz.xml')
mouthCascade = cv2.CascadeClassifier('Mouth.xml')

# Check if classifiers are loaded successfully
if faceCascade.empty() or eyeCascade.empty() or noseCascade.empty() or mouthCascade.empty():
    print("Error: Unable to load classifiers.")
    exit()

# Capturing real-time video stream
video_capture = cv2.VideoCapture(0)  # Change the camera index to 0

while True:
    ret, img = video_capture.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    img = detect(img, faceCascade, eyeCascade, noseCascade, mouthCascade)
    cv2.imshow("Face Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture
video_capture.release()
cv2.destroyAllWindows()
