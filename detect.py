from ultralytics import YOLO
import cv2

model = YOLO("yolo_custom.pt")

# Inicializa a webcam USB
cap = cv2.VideoCapture(4)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha na captura do frame.")
        break

    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf >= 0.5:  # ✅ Só desenha se confiança for ≥ 0.7
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = f"{model.names[cls]} {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Detector de Garrafas - YOLOv11", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
