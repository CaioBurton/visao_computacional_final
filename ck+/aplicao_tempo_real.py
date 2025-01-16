import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# ===============================
# Configurações
# ===============================
IMG_HEIGHT, IMG_WIDTH = 48, 48  # Dimensões esperadas pelo modelo
CLASS_NAMES = ['bom', 'ruim']   # Ajuste conforme suas classes
MODEL_PATH = 'modelo_cnn_ckplus.h5'

# Carregar o modelo treinado
model = tf.keras.models.load_model(MODEL_PATH)

# Carregar o classificador Haarcascade para detecção de rosto
haarcascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haarcascade_path)

# ===============================
# Função para pré-processar o rosto
# ===============================
def preprocess_face(face):
    """
    Pré-processa o rosto para o formato esperado pelo modelo.
    """
    face_resized = cv2.resize(face, (IMG_HEIGHT, IMG_WIDTH))  # Redimensionar
    face_array = img_to_array(face_resized) / 255.0           # Normalizar [0,1]
    face_array = np.expand_dims(face_array, axis=0)           # Adicionar dimensão de batch
    return face_array

# ===============================
# Main: Classificação em tempo real
# ===============================
def real_time_classification():
    """
    Detecta o rosto da pessoa em tempo real usando a webcam,
    classifica a expressão facial e exibe o resultado na tela.
    """
    # Iniciar captura de vídeo (0 = webcam padrão)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Não foi possível acessar a webcam.")
        return

    while True:
        # Ler quadro (frame) da webcam
        ret, frame = cap.read()
        if not ret:
            print("Não foi possível capturar o quadro de vídeo.")
            break

        # Converter para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostos na imagem
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Para cada rosto detectado, vamos classificar
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]      # Recorta apenas a região do rosto
            processed_face = preprocess_face(face_img)

            # Fazer a previsão
            prediction = model.predict(processed_face)

            # No caso binário, usando > 0.5 como limiar
            predicted_class = CLASS_NAMES[int(prediction[0] > 0.5)]

            # Desenhar retângulo ao redor do rosto
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Escrever a classe prevista acima do rosto
            cv2.putText(
                frame, 
                predicted_class, 
                (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.9, 
                (36, 255, 12), 
                2
            )

        # Exibir o frame resultante
        cv2.imshow('Classificador de Expressoes Faciais - Pressione "q" para sair', frame)

        # Se o usuário pressionar 'q', interrompa o loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar a captura e destruir janelas
    cap.release()
    cv2.destroyAllWindows()

# Executar a função de classificação em tempo real
if __name__ == "__main__":
    real_time_classification()
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# ===============================
# Configurações
# ===============================
IMG_HEIGHT, IMG_WIDTH = 48, 48  # Dimensões esperadas pelo modelo
CLASS_NAMES = ['bom', 'ruim']   # Ajuste conforme suas classes
MODEL_PATH = 'modelo_cnn_ckplus.h5'

# Carregar o modelo treinado
model = tf.keras.models.load_model(MODEL_PATH)

# Carregar o classificador Haarcascade para detecção de rosto
haarcascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haarcascade_path)

# ===============================
# Função para pré-processar o rosto
# ===============================
def preprocess_face(face):
    """
    Pré-processa o rosto para o formato esperado pelo modelo.
    """
    face_resized = cv2.resize(face, (IMG_HEIGHT, IMG_WIDTH))  # Redimensionar
    face_array = img_to_array(face_resized) / 255.0           # Normalizar [0,1]
    face_array = np.expand_dims(face_array, axis=0)           # Adicionar dimensão de batch
    return face_array

# ===============================
# Main: Classificação em tempo real
# ===============================
def real_time_classification():
    """
    Detecta o rosto da pessoa em tempo real usando a webcam,
    classifica a expressão facial e exibe o resultado na tela.
    """
    # Iniciar captura de vídeo (0 = webcam padrão)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Não foi possível acessar a webcam.")
        return

    while True:
        # Ler quadro (frame) da webcam
        ret, frame = cap.read()
        if not ret:
            print("Não foi possível capturar o quadro de vídeo.")
            break

        # Converter para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostos na imagem
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Para cada rosto detectado, vamos classificar
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]      # Recorta apenas a região do rosto
            processed_face = preprocess_face(face_img)

            # Fazer a previsão
            prediction = model.predict(processed_face)

            # No caso binário, usando > 0.5 como limiar
            predicted_class = CLASS_NAMES[int(prediction[0] > 0.5)]

            # Desenhar retângulo ao redor do rosto
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Escrever a classe prevista acima do rosto
            cv2.putText(
                frame, 
                predicted_class, 
                (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.9, 
                (36, 255, 12), 
                2
            )

        # Exibir o frame resultante
        cv2.imshow('Classificador de Expressoes Faciais - Pressione "q" para sair', frame)

        # Se o usuário pressionar 'q', interrompa o loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar a captura e destruir janelas
    cap.release()
    cv2.destroyAllWindows()

# Executar a função de classificação em tempo real
if __name__ == "__main__":
    real_time_classification()
