import cv2
import numpy as np
import tensorflow as tf

# --------------------------
# 1. Carregar o modelo
# --------------------------
model_path = 'modelo_cnn_ckplus_finetuned.h5'
model = tf.keras.models.load_model(model_path)

class_names = ['bom', 'ruim']

# --------------------------
# 2. Configurar Haar Cascade
# --------------------------
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if face_cascade.empty():
    print("Erro ao carregar o arquivo 'haarcascade_frontalface_default.xml'. Verifique o caminho.")
    exit(1)

# --------------------------
# 3. Abrir webcam
# --------------------------
cap = cv2.VideoCapture(1)  # ajuste o índice caso tenha mais de uma webcam

if not cap.isOpened():
    print("Não foi possível abrir a webcam.")
    exit(1)

THRESHOLD = 0.5
print("Pressione 'q' para encerrar...")

# --------------------------
# 4. Loop principal
# --------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converter para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        # Extrair a ROI do rosto
        face_crop = gray[y:y+h, x:x+w]

        # Redimensionar para 48x48
        face_crop = cv2.resize(face_crop, (48, 48))

        # ============== Ajuste de iluminação (CLAHE) ==============
        # 1) Criar o objeto CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # 2) Aplicar no rosto redimensionado
        face_crop = clahe.apply(face_crop)
        # ==========================================================

        # Normalizar para [0, 1]
        face_crop = face_crop.astype('float32') / 255.0

        # Expandir dimensões p/ (1, 48, 48, 1)
        face_input = np.expand_dims(face_crop, axis=-1)
        face_input = np.expand_dims(face_input, axis=0)

        # Predição
        pred = model.predict(face_input)[0][0]

        if pred >= THRESHOLD:
            label_index = 1  # “ruim”
        else:
            label_index = 0  # “bom”

        label_text = class_names[label_index]

        # Probabilidades (apenas ilustrativo)
        prob_ruim = pred
        prob_bom  = 1 - pred

        # Desenho do retângulo e rótulo
        color = (0, 255, 0) if label_index == 0 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame,
                    f'{label_text} ({prob_ruim:.2f})',
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2)

    cv2.imshow('Deteccao de Emocao (Tempo Real)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
