import cv2
import face_recognition
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import os

# Define paths for your images
base_directory = 'C:\\Users\\Sarang T\\Documents\\sarang\\known_faces_directory\\'

# Load training images and their corresponding labels
training_images = [
    cv2.imread(os.path.join(base_directory, 'aharan1.jpg')),
    cv2.imread(os.path.join(base_directory, 'sarang1.png'))
]
labels = ['aharan', 'sarang']  # Corresponding labels

# Function to extract face embeddings using face_recognition
def extract_face_embeddings(image):
    # Convert the image from BGR (OpenCV format) to RGB (face_recognition format)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Find face locations and embeddings
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    # Return the face encodings
    return face_encodings

# Example of training a classifier with face embeddings
def train_classifier(training_images, labels):
    face_encodings_list = []
    
    for img in training_images:
        encodings = extract_face_embeddings(img)
        if encodings:
            face_encodings_list.append(encodings[0])  # Only take the first encoding
        
    X_train = np.array(face_encodings_list)
    y_train = np.array(labels)
    
    # Label encoding
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    
    # Train a classifier (e.g., SVM)
    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(X_train, y_train)
    
    return classifier, le

# Function to predict the identity of a face from the test image
def recognize_face(test_image, classifier, label_encoder):
    encodings = extract_face_embeddings(test_image)
    if encodings:
        features_flat = np.array(encodings).reshape(1, -1)
        prediction = classifier.predict(features_flat)
        predicted_label = label_encoder.inverse_transform(prediction)
        return predicted_label
    return None

# Train the classifier
classifier, le = train_classifier(training_images, labels)

# Start real-time video capture from the webcam
cap = cv2.VideoCapture(0)  # 0 means the default camera

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Recognize faces in the current frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Iterate over the faces detected in the frame
    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Predict the identity of the face
        predicted_identity = recognize_face(frame, classifier, le)
        
        if predicted_identity:
            cv2.putText(frame, f"Identity: {predicted_identity[0]}", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Unknown", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the frame with bounding boxes and labels
    cv2.imshow("Real-time Face Recognition", frame)

    # Press 'q' to exit the loop and stop the webcam
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
