import cv2
import numpy as np
import tensorflow as tf

# ==============================
# Carregar o modelo treinado
# ==============================
modelo = tf.keras.models.load_model(r'C:\Users\Caio Burton\Desktop\trab_final\ck+\modelo_cnn_ckplus.h5')

# Mapeamento dos índices para rótulos (ajuste se necessário)
# Aqui, assumindo que o mapeamento foi: 'bom' -> 0, 'ruim' -> 1 (ou vice-versa, confira o seu mapping)
class_names = {0: 'bom', 1: 'ruim'}  

# ==============================
# Carregar o classificador Haar Cascade para detecção facial
# ==============================
# Se necessário, informe o caminho completo para o arquivo haarcascade_frontalface_default.xml
haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# ==============================
# Parâmetros de pré-processamento
# ==============================
IMG_HEIGHT, IMG_WIDTH = 48, 48

# ==============================
# Inicializar a captura de vídeo pela webcam
# ==============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao abrir a câmera!")
    exit()

print("Pressione 'q' para sair.")

while True:
    # Captura o frame a partir da webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Falha na captura do frame")
        break

    # Converter o frame para escala de cinza para a detecção facial
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar faces na imagem
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Para cada face detectada (nesse exemplo, processamos apenas a primeira)
    for (x, y, w, h) in faces:
        # Extrair a região de interesse (ROI) da face
        face_roi = gray_frame[y:y+h, x:x+w]
        
        # Pré-processar a ROI:
        # - Redimensionar para (48, 48)
        # - Normalizar a imagem (0-1)
        face_resized = cv2.resize(face_roi, (IMG_WIDTH, IMG_HEIGHT))
        face_normalized = face_resized.astype("float32") / 255.0
        
        # Expandir as dimensões para adequar ao formato (batch_size, altura, largura, canais)
        face_input = np.expand_dims(face_normalized, axis=0)  # adiciona dimensão do batch
        face_input = np.expand_dims(face_input, axis=-1)        # adiciona dimensão do canal (1)
        
        # Realizar a predição
        pred = modelo.predict(face_input)
        
        # Como o modelo utiliza ativação sigmoid, um limiar de 0.5 é utilizado para distinguir as classes
        pred_label = 0 if pred[0] < 0.5 else 1
        
        # Obter o rótulo textual
        label_text = class_names[pred_label]
        
        # Exibir o retângulo da face e o rótulo na imagem
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)
        
        # Processa apenas a primeira face detectada para evitar múltiplos desenhos
        break

    # Exibir o frame com as anotações
    cv2.imshow('Reconhecimento de Emoção (bom vs ruim)', frame)

    # Sair se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura e fecha as janelas
cap.release()
cv2.destroyAllWindows()
