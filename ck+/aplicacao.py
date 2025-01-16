import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# ===============================
# Configurações
# ===============================
IMG_HEIGHT, IMG_WIDTH = 48, 48  # Dimensões esperadas pelo modelo
CLASS_NAMES = ['bom', 'ruim']   # Ajuste conforme as classes do seu modelo
MODEL_PATH = 'modelo_cnn_ckplus_finetuned.h5'

# Carregar o modelo treinado
model = tf.keras.models.load_model(MODEL_PATH)

# Carregar o classificador Haarcascade para detecção de rosto
haarcascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haarcascade_path)
if face_cascade.empty():
    print("Erro ao carregar o arquivo haarcascade_frontalface_default.xml.")
    exit(1)

# ===============================
# 1. Função para detectar o rosto
# ===============================
def detect_face(image_path):
    """
    Detecta rostos na imagem e retorna a região do rosto recortada, a imagem original
    e as coordenadas (x, y, w, h) do primeiro rosto encontrado.
    """
    # Carregar a imagem
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro ao ler a imagem {image_path}. Verifique o caminho.")
        return None, None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detectar rostos na imagem
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        print(f"Nenhum rosto detectado em: {image_path}")
        return None, img, None

    # Selecionar o primeiro rosto detectado
    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]

    return face, img, (x, y, w, h)

# ===============================
# 2. Função para pré-processar o rosto
# ===============================
def preprocess_face(face):
    """
    Pré-processa o rosto para o formato esperado pelo modelo (48x48, normalizado).
    """
    face_resized = cv2.resize(face, (IMG_HEIGHT, IMG_WIDTH))  # Redimensionar
    
    # (Opcional) Ajuste de iluminação via Equalização de Histograma:
    # face_resized = cv2.equalizeHist(face_resized)
    #
    # (Ou) usando CLAHE:
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # face_resized = clahe.apply(face_resized)

    # Converter em array e normalizar
    face_array = img_to_array(face_resized) / 255.0
    face_array = np.expand_dims(face_array, axis=0)  # (1, 48, 48, 1)
    return face_array

# ===============================
# 3. Função para classificar UMA imagem
# ===============================
def classify_image(image_path):
    """
    Detecta o rosto em 'image_path', processa e classifica.
    Retorna a string da classe prevista ('bom' ou 'ruim') ou None se não há rosto.
    """
    face, original_img, bbox = detect_face(image_path)
    if face is None or bbox is None:
        return None

    # Pré-processar o rosto detectado
    processed_face = preprocess_face(face)

    # Fazer a predição
    prediction = model.predict(processed_face)[0][0]
    # Limiar para classificação binária (0.5, 0.6 ou 0.7)
    predicted_class = CLASS_NAMES[int(prediction > 0.5)]

    # Desenhar o retângulo na imagem original
    x, y, w, h = bbox
    cv2.rectangle(original_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Exibir a imagem original
    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Classificação: {predicted_class} (score: {prediction:.2f})")
    plt.axis('off')
    plt.show()

    return predicted_class

# ===============================
# 4. Função para classificar MÚLTIPLAS imagens
# ===============================
def classify_images(image_paths):
    """
    Recebe uma lista de caminhos de imagens e classifica cada uma.
    """
    results = {}
    for img_path in image_paths:
        print(f"\nProcessando imagem: {img_path}")
        result_class = classify_image(img_path)
        results[img_path] = result_class
    return results

# ===============================
# Exemplo de uso
# ===============================
if __name__ == "__main__":
    # Defina o(s) caminho(s) das imagens de teste
    # Exemplos individuais:
    # image_path = "C:/Users/.../teste_boa1.png"
    # classify_image(image_path)
    
    # Para classificar várias imagens em um diretório:
    test_dir = "C:/Users/Caio Burton/Documents/dataset_visao/teste"
    valid_exts = ['.png', '.jpg', '.jpeg']
    images_to_classify = [
        os.path.join(test_dir, f) 
        for f in os.listdir(test_dir) 
        if os.path.splitext(f)[1].lower() in valid_exts
    ]

    # Chamar a função para processar múltiplas imagens
    results_dict = classify_images(images_to_classify)

    print("\nResultados Finais:")
    for path, cls in results_dict.items():
        print(f"{path} => {cls}")
