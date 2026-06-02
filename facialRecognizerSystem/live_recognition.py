import cv2
import face_recognition


# PART 1: THE MEMORY (Training the system)

print("Learning known face...")
# Load your saved picture
img_known = cv2.imread("faces/face1.jpg")
img_known = cv2.cvtColor(img_known, cv2.COLOR_BGR2RGB)

# Calculate the 128-number fingerprint for your saved picture.
# We add [0] because we want the fingerprint of the first face it finds in the photo.
encode_known = face_recognition.face_encodings(img_known)[0]


# PART 2: THE LIVE MATCHING

print("Starting webcam...")
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. Find the locations of all faces in the live webcam
    faces_cur_frame = face_recognition.face_locations(img_rgb)

    # 2. Calculate the 128-number fingerprints for those live faces
    # Notice we pass the locations into the function to save CPU power
    encodes_cur_frame = face_recognition.face_encodings(img_rgb, faces_cur_frame)

    # 3. Loop through every face found on the webcam
    # zip() lets us loop through two lists at the exact same time
    for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):

        # Compare the live fingerprint against our known fingerprint
        # This returns a True or False
        matches = face_recognition.compare_faces([encode_known], encode_face)

        # If it's a match (True)
        if matches[0]:
            name = "ABDULLAH"  # This is what we will print on the screen

            # Draw the standard box around the face
            cv2.rectangle(img, (face_loc[3], face_loc[0]), (face_loc[1], face_loc[2]), (0, 255, 0), 2)

            # Draw a solid green rectangle at the bottom to act as a name tag background
            cv2.rectangle(img, (face_loc[3], face_loc[2] - 35), (face_loc[1], face_loc[2]), (0, 255, 0), cv2.FILLED)

            # Write the name in white text over the solid green background
            cv2.putText(img, name, (face_loc[3] + 6, face_loc[2] - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Live Face Detection', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()