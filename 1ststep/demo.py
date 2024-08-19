import cv2
import face_recognition as fr

# Load the input image
input_image = cv2.imread("wp.jpg")

# Convert the image to RGB format (required by face_recognition)
input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Detect face locations in the image
face_locations = fr.face_locations(input_image_rgb)

# Draw rectangles around detected faces
for (top, right, bottom, left) in face_locations:
    cv2.rectangle(input_image, (left, top), (right, bottom), (34, 125, 0), 2)

# Display the result
cv2.imshow("Face Detection Result", input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
