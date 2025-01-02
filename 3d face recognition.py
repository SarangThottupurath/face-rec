import cv2
import dlib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import open3d as o3d
from scipy.spatial import distance

# Load the face detector and face landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Function to extract 3D features from a face
def extract_3d_face_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        # Extract 3D features (as an example, use 2D landmarks + depth estimation methods)
        features = []
        for n in range(36, 48):  # Only the nose area for simplicity
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            features.append([x, y])
        # Convert to 3D by adding some depth (simple example, adjust as needed)
        features_3d = np.array(features)
        features_3d = np.column_stack((features_3d, np.random.rand(features_3d.shape[0])))  # Random depth
        return features_3d
    return None

# Example of training a classifier with 3D features
def train_classifier(training_images, labels):
    features_list = []
    for img in training_images:
        features = extract_3d_face_features(img)
        if features is not None:
            features_list.append(features.flatten())
    
    X_train = np.array(features_list)
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
    features = extract_3d_face_features(test_image)
    if features is not None:
        features_flat = features.flatten().reshape(1, -1)
        prediction = classifier.predict(features_flat)
        predicted_label = label_encoder.inverse_transform(prediction)
        return predicted_label
    return None

# Load some example images and labels
# Example: You need to load a set of images and their corresponding labels (e.g., filenames or identities)
training_images = [cv2.imread('C:\\Users\\Sarang T\\Documents\\sarang\\known_faces_directory\\aharan1.jpg'), cv2.imread('C:\\Users\\Sarang T\\Documents\\sarang\\known_faces_directory\\sarang1.png')]  # Add your images here
labels = ['aharan', 'sarang']  # Corresponding labels

# Train the classifier
classifier, le = train_classifier(training_images, labels)

# Test with a new image
test_image = cv2.imread('C:\\Users\\Sarang T\\Documents\\sarang\\known_faces_directory\\sarang2.png')
predicted_identity = recognize_face(test_image, classifier, le)
if predicted_identity:
    print(f"Predicted identity: {predicted_identity[0]}")
else:
    print("No face detected or unable to extract features.")
