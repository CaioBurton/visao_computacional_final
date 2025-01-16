import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Definir as classes
classes = {1: 'Bad', 0: 'Good'}

# Carregar o modelo treinado
model_path = 'fer2013_cnn_model.h5'  # Atualize este caminho se necessário
model = load_model(model_path)

# Carregar o classificador Haar Cascade para detecção de faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(1)  # 0 geralmente é a webcam padrão

# Verificar se a webcam está aberta
if not cap.isOpened():
    print("Erro: Não foi possível acessar a webcam.")
    exit()

while True:
    # Capturar frame-a-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Erro: Não foi possível ler o frame da webcam.")
        break

    # Converter para escala de cinza (necessário para o Haar Cascade)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar CLAHE para corrigir a iluminação
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Detectar faces no frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Desenhar um retângulo em torno da face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extrair a região da face
        face_roi = gray[y:y + h, x:x + w]

        # Pré-processamento: redimensionar para 48x48, como no treinamento
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi.astype('float32') / 255.0  # Normalizar
        face_roi = np.expand_dims(face_roi, axis=0)    # Adicionar dimensão de batch
        face_roi = np.expand_dims(face_roi, axis=-1)   # Adicionar canal

        # Prever a emoção
        prediction = model.predict(face_roi)
        # O modelo retorna um valor entre 0 e 1. Vamos salvar esse valor para exibir.
        inference_value = float(prediction[0][0])
        
        # Arredondar para 0 ou 1 para decidir entre 'Bad' e 'Good'
        class_idx = int(np.round(inference_value))
        emotion = classes[class_idx]

        # Definir a cor do texto com base na classe
        color = (0, 0, 255) if class_idx == 1 else (0, 255, 0)  # Verde para 'Good', Vermelho para 'Bad'

        # Criar o texto a ser exibido, incluindo o valor previsto
        text = f"{emotion} ({inference_value:.2f})"

        # Exibir a classificação na imagem
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, color, 2, cv2.LINE_AA)

    # Exibir o frame resultante
    cv2.imshow('Detecção de Emoções - Pressione "q" para sair', frame)

    # Sair do loop quando a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Quando tudo estiver feito, liberar a captura e fechar janelas
cap.release()
cv2.destroyAllWindows()
