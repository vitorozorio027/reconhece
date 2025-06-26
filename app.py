import cv2
import face_recognition
import os
import requests
import io
from datetime import datetime, timedelta

#---------------------------------#
KNOWN_FACES_DIR  = 'known_faces'
TOLERANCE        = 0.50
FRAME_RESIZE     = 0.5
MODEL            = 'hog'
NUM_JITTERS      = 1
CONSEC_FRAMES    = 0     # contagens de frames em consequencia
ALERT_COOLDOWN   = timedelta(minutes=1) 

# --- botttttt ---
BOT_TOKEN        = "BOT TOKEN coloque aqui" #---substitua por token de bot do Telegram
CHAT_ID          = "CHAT ID coloque aqui" #---substitua por chat id do Telegram
TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

# === carrega encodings na memória a cada execuaooooooo ===
known_names     = []
known_encodings = []

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.lower().endswith(('.jpg', '.png')):
        name, _ = os.path.splitext(filename)
        img = face_recognition.load_image_file(os.path.join(KNOWN_FACES_DIR, filename))
        encs = face_recognition.face_encodings(img, num_jitters=NUM_JITTERS)
        if encs:
            known_names.append(name)
            known_encodings.append(encs[0])

print(f"Carregadas {len(known_names)} pessoas conhecidas.")

# === inicializa captura de vídeo na Iriun Webcam ===
cap = cv2.VideoCapture("video=Iriun Webcam", cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Iriun Webcam não abriu por nome, tentando índice 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir a Iriun Webcam (nome ou índice).")

# Dicionários de apoio:
frame_counts   = {}  # conta quantos frames consecutivos cada nome apareceu
last_alert     = {}  # armazena datetime do último alerta enviado para cada nome

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # redimensiona e converte cor
    small = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE, fy=FRAME_RESIZE)
    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    # detecta e codifica faces
    locations = face_recognition.face_locations(rgb, model=MODEL)
    encodings = face_recognition.face_encodings(rgb, locations, num_jitters=NUM_JITTERS)

    for loc, enc in zip(locations, encodings):
        matches = face_recognition.compare_faces(known_encodings, enc, TOLERANCE)
        name = 'Desconhecido'

        if True in matches:
            idx = matches.index(True)
            name = known_names[idx]
        else:
            # tenta pela menor distância
            dists = face_recognition.face_distance(known_encodings, enc)
            best  = dists.argmin()
            if dists[best] < TOLERANCE:
                name = known_names[best]

        # se for rosto desconhecido, ignora
        if name == 'Desconhecido':
            continue

        # conta detecção consecutiva
        frame_counts[name] = frame_counts.get(name, 0) + 1

        # quando atingir X frames consecutivos, tenta enviar alerta
        if frame_counts[name] >= CONSEC_FRAMES:
            agora = datetime.now()
            # checa cooldown de 1 minuto
            if name in last_alert:
                diff = agora - last_alert[name]
            else:
                diff = ALERT_COOLDOWN  # força envio na primeira vez

            if diff >= ALERT_COOLDOWN:
                # ajusta box ao tamanho original
                top, right, bottom, left = [int(v / FRAME_RESIZE) for v in loc]
                face_img = frame[top:bottom, left:right]
                _, buf = cv2.imencode('.jpg', face_img)
                photo = io.BytesIO(buf)

                timestamp = agora.strftime('%Y-%m-%d %H:%M:%S')
                caption   = f"🔔 {name} em Vitória/ES, Brazil\n🕒 {timestamp}"

                resp = requests.post(
                    TELEGRAM_API_URL,
                    data={'chat_id': CHAT_ID, 'caption': caption},
                    files={'photo': ('face.jpg', photo.getvalue())}
                )
                if resp.ok:
                    print(f"Alerta enviado: {name} em Vitória/ES, Brazil")
                else:
                    print("Falha no envio:", resp.text)

                # registra o momento do alerta e zera contagem de frames
                last_alert[name] = agora
                frame_counts[name] = 0
            else:
                # ainda dentro do intervalo de 1 minuto: não faz nada, só zera a contagem de frames
                frame_counts[name] = 0

    # desenha caixas e nomes na imagem (usando o último 'name' detectado neste frame)
    for loc in locations:
        t, r, b, l = [int(v / FRAME_RESIZE) for v in loc]
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, name, (l, b + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Reconhecimento Facial', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
