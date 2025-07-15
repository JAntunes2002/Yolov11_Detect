import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# Carrega o modelo
model = YOLO("yolo_custom.pt")

# Configura a câmara RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Inicia o pipeline (somente uma vez)
profile = pipeline.start(config)

# Obter parâmetros intrínsecos da câmera
depth_sensor = profile.get_stream(rs.stream.depth).as_video_stream_profile()
intrinsics = depth_sensor.get_intrinsics()  # Parâmetros intrínsecos da câmera

# Nome fixo para a janela
WINDOW_NAME = "Deteção com RealSense + YOLO"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

try:
    while True:
        # Captura e alinha os frames
        frames = pipeline.wait_for_frames()
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            print("Frame não capturado!")
            continue

        # Converte para arrays numpy
        color_image = np.asanyarray(color_frame.get_data())

        # Aplica o modelo YOLO
        results = model(color_image, stream=True)

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf >= 0.5:
                    # Coordenadas do retângulo
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    label_name = model.names[cls]

                    # Centro do bounding box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # Obter profundidade em metros (z)
                    depth = depth_frame.get_distance(cx, cy)

                    # Converter para coordenadas 3D (x, y, z)
                    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth)
                    x, y, z = point_3d  # x e y são as coordenadas em metros, z é a profundidade

                    # Texto para exibir: classe, confiança e coordenadas 3D
                    text_top = f"{label_name} {conf:.2f}// Prof {depth:.2f}m"
                    text_bottom = f"X: {x:.2f}m, Y: {y:.2f}m, Z: {z:.2f}m"

                    # Desenhar o retângulo em volta do objeto
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Exibir o texto acima do retângulo
                    cv2.putText(color_image, text_top, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Exibir as coordenadas 3D abaixo do retângulo
                    cv2.putText(color_image, text_bottom, (x1, y2 + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Exibir a imagem capturada
        cv2.imshow(WINDOW_NAME, color_image)

        # Permite sair com a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Libertar recursos
    pipeline.stop()
    cv2.destroyAllWindows()
