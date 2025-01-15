import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

# ===============================
# Configurações
# ===============================
IMG_HEIGHT, IMG_WIDTH = 48, 48  # Dimensões esperadas pelo modelo
CLASS_NAMES = ['bom', 'ruim']  # Ajuste conforme suas classes
MODEL_PATH = 'modelo_cnn_ckplus.h5'

# Carregar o modelo treinado
model = tf.keras.models.load_model(MODEL_PATH)

# Carregar o classificador Haarcascade para detecção de rosto
haarcascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haarcascade_path)

# ===============================
# Função para detectar e recortar o rosto
# ===============================
def detect_face(image_path):
    """
    Detecta rostos na imagem e retorna a região do rosto recortada e as coordenadas do rosto.
    """
    # Carregar a imagem
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detectar rostos na imagem
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("Nenhum rosto detectado na imagem.")
        return None, img, None

    # Selecionar o primeiro rosto detectado
    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]

    return face, img, (x, y, w, h)

# ===============================
# Função para processar a imagem do rosto
# ===============================
def preprocess_face(face):
    """
    Pré-processa o rosto para o formato esperado pelo modelo.
    """
    face_resized = cv2.resize(face, (IMG_HEIGHT, IMG_WIDTH))  # Redimensionar
    face_array = img_to_array(face_resized) / 255.0  # Normalizar
    face_array = np.expand_dims(face_array, axis=0)  # Adicionar dimensão de batch
    return face_array

# ===============================
# Função para classificar a imagem
# ===============================
def classify_image(image_path):
    """
    Detecta o rosto na imagem, processa e classifica.
    """
    face, original_img, bbox = detect_face(image_path)

    if face is None or bbox is None:
        return None

    # Pré-processar o rosto detectado
    processed_face = preprocess_face(face)
    prediction = model.predict(processed_face)
    predicted_class = CLASS_NAMES[int(prediction[0] > 0.5)]  # Limiar para classificação binária

    # Desenhar o retângulo verde na imagem original com base no bbox
    x, y, w, h = bbox
    cv2.rectangle(original_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Exibir a imagem original com o rosto detectado e a classificação
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Classificação: {predicted_class}')
    plt.axis('off')
    plt.show()

    return predicted_class

# ===============================
# Caminho da imagem de teste
# ===============================
image_path = r'C:\Users\LIMCI\Desktop\trab_final\teste\teste_boa8.jpg'

# Classificar a imagem
resultado = classify_image(image_path)
if resultado:
    print(f'A classe prevista é: {resultado}')
