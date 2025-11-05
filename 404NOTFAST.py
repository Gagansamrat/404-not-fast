import cv2
from deepface import DeepFace

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Detect and analyze emotions
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Handle both new (dict) and old (list) formats
        if isinstance(result, list):
            
            result = result[0]

        # Extract region safely
        region = result.get('region', {})
        x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
        emotion = result.get('dominant_emotion', 'Unknown')

        # Draw rectangle & label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    except Exception as e:
        print("Error:", e)

    cv2.imshow('Emotion Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
