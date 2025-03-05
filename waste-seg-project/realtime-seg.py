import cv2
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('model.keras')

# Define class labels
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']  

# ip_camera_url = 'http://192.168.18.4:4747/video'
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing
    resized_frame = cv2.resize(frame, (224, 224))
    # normalized_frame = resized_frame / 255.0
    input_frame = resized_frame.reshape(1, 224, 224, 3)

    # Prediction
    prediction = model.predict(input_frame)
    predicted_class = class_labels[prediction.argmax()]

    # Overlay result
    cv2.putText(frame, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display
    cv2.imshow('Waste Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
