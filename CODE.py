import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('/Users/adithibalajee/fer2013.csv')

# Extract features and labels
pixels = data['pixels'].tolist()
emotions = data['emotion'].values

# Convert pixels to numpy arrays and reshape
X = np.array([np.fromstring(pixel, dtype=int, sep=' ') for pixel in pixels])
#X = np.array([np.frombuffer(bytearray(pixel, 'utf-8'), dtype=int, sep=' ') for pixel in pixels])
X = X.reshape(X.shape[0], 48, 48, 1)
# Normalize the pixel values
X = X / 255.0

# Convert labels to categorical
y = to_categorical(emotions, num_classes=7)

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')

])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Plot the training and validation accuracy and loss
'''
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')

plt.show()
'''

# Function to predict emotion from image
def predict_emotion(image, model):
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    image = cv2.resize(image, (48, 48))
    image = image.reshape(1, 48, 48, 1)
    image = image / 255.0
    prediction = model.predict(image)
    emotion = np.argmax(prediction)
    return emotion_labels[emotion]

# Real-time emotion recognition
def real_time_emotion_recognition():
    # Start video capture
    cap = cv2.VideoCapture(1)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]

            # Predict emotion
            emotion = predict_emotion(roi_gray, model)

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Put text of predicted emotion
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the resulting frame
        #cv2.imshow('Emotion Recognition', frame)
        cv2.imshow('Webcam', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# Run the real-time emotion recognition
real_time_emotion_recognition()