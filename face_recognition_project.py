import face_recognition
import cv2
import numpy as np

# Get a reference to camera.
video_capture = cv2.VideoCapture(0)

pvsindhu_image = face_recognition.load_image_file("Images/PV Sindhu1.jpg")
pvsindhu_face_encoding1 = face_recognition.face_encodings(pvsindhu_image)[0]

pvsindhu_image = face_recognition.load_image_file("Images/PV Sindhu2.jpg")
pvsindhu_face_encoding2 = face_recognition.face_encodings(pvsindhu_image)[0]

pvsindhu_image = face_recognition.load_image_file("Images/PV Sindhu4.jpg")
pvsindhu_face_encoding4 = face_recognition.face_encodings(pvsindhu_image)[0]

pvsindhu_image = face_recognition.load_image_file("Images/PV Sindhu5.jpg")
pvsindhu_face_encoding5 = face_recognition.face_encodings(pvsindhu_image)[0]

kohli_image = face_recognition.load_image_file("Images/Virat Kohli2.jpg")
kohli_face_encoding2 = face_recognition.face_encodings(kohli_image)[0]

kohli_image = face_recognition.load_image_file("Images/Virat Kohli4.jpg")
kohli_face_encoding4 = face_recognition.face_encodings(kohli_image)[0]

ronaldo_image=face_recognition.load_image_file("Images/Ronaldo1.jpg")
ronaldo_face_encoding1 = face_recognition.face_encodings(ronaldo_image)[0]

ronaldo_image=face_recognition.load_image_file("Images/Ronaldo2.jpg")
ronaldo_face_encoding2 = face_recognition.face_encodings(ronaldo_image)[0]

ronaldo_image=face_recognition.load_image_file("Images/Ronaldo3.jpg")
ronaldo_face_encoding3 = face_recognition.face_encodings(ronaldo_image)[0]

ronaldo_image=face_recognition.load_image_file("Images/Ronaldo4.jpg")
ronaldo_face_encoding4 = face_recognition.face_encodings(ronaldo_image)[0]

ronaldo_image=face_recognition.load_image_file("Images/Ronaldo5.jpg")
ronaldo_face_encoding5 = face_recognition.face_encodings(ronaldo_image)[0]


# Create arrays of known face encodings
known_face_encodings = [
    pvsindhu_face_encoding1,
    pvsindhu_face_encoding2,
    pvsindhu_face_encoding4,
    pvsindhu_face_encoding5,
    kohli_face_encoding2,
    kohli_face_encoding4,
    ronaldo_face_encoding1,
    ronaldo_face_encoding2,
    ronaldo_face_encoding3,
    ronaldo_face_encoding4,
    ronaldo_face_encoding5
]
# Create arrays of names
known_face_names = [
    "PV Sindhu",
    "PV Sindhu",
    "PV Sindhu",
    "PV Sindhu",
    "Virat Kohli",
    "Virat Kohli",
    "Ronaldo",
    "Ronaldo",
    "Ronaldo",
    "Ronaldo",
    "Ronaldo"
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video.
    ret, frame = video_capture.read()

    # Only process one out of two frame of video to save time.
    if process_this_frame:
        # Resize frame of video to 1/2 size for faster face recognition processing.
        small_frame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses).
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # Find all the faces and face encodings in the current frame of video.
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Using the known face with the smallest distance to the new face.
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_matched_index = np.argmin(face_distances)
            if matches[best_matched_index]:
                name = known_face_names[best_matched_index]
            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (x1,y2, x2,y1), name in zip(face_locations, face_names):
        # Scale back up face locations.
        x1 *= 2
        y2 *= 2
        x2 *= 2
        y1 *= 2

        # Draw a box around the faces
        cv2.rectangle(frame, (y1, x1), (y2, x2), (0, 255, 0), 3)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (y1, x2 - 30), (y2, x2), (0, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (y1 + 8, x2 - 8), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'esc' on the keyboard to escape.
    if cv2.waitKey(1)==27:
        break

# Release handle to the camera
video_capture.release()
cv2.destroyAllWindows()