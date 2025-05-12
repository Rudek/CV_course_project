from ultralytics import YOLO
import cv2

model = YOLO('best.pt')

video = 'final-video.mp4'
cap = cv2.VideoCapture(video)

ret = True

labels = {0: "Tank"}

while ret:
    ref, frame = cap.read()

    results = model.track(frame, persist=True)
    boxes = results[0].boxes
    annotated_frame = frame.copy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(boxes.cls[0])
        conf = float(boxes.conf[0])
        label = labels[cls_id] if cls_id == 0 else model.names[cls_id]
        
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    cv2.imshow('frame', annotated_frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
