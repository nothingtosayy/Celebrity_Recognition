import cv2
def get_cropped_image_if_2_eyes(image_path):
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        'D:/Users/Py-MASTER/py-master/DataScience/CelebrityFaceRecognition/model/opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(
        'D:/Users/Py-MASTER/py-master/DataScience/CelebrityFaceRecognition/model/opencv/haarcascades/haarcascade_eye.xml')
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)

    cropped_faces = []
    for (x, y, w, h) in faces:
        face_img = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = image_gray[y:y + h, x:x + w]
        roi_color = face_img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    return cropped_faces