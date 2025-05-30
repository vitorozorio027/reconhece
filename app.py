import cv2
import face_recognition
import os
import requests
import io
from datetime import datetime
from geopy.geocoders import Nominatim
import geocoder

#---------------------------------#
KNOWN_FACES_DIR  = 'known_faces'
TOLERANCE        = 0.50
FRAME_RESIZE     = 0.5
MODEL            = 'hog'
NUM_JITTERS      = 2
CONSEC_FRAMES    = 3     # contagens de frames consecutivos

# --- telegram Bot ---
BOT_TOKEN        = "8055665716:AAFoRpjHDYBq__72ViRQVoXblNQ3--U_fuc"
CHAT_ID          = "6149329556"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

# === carrega encodings na memória a cada execução ===
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

# === geolocalização via IP ===
def get_neighborhood():
    try:
        g = geocoder.ip('me')
        latlng = g.latlng
        loc = Nominatim(user_agent="app_face").reverse(latlng, language='pt')
        addr = loc.raw.get('address', {})
        return addr.get('suburb') or addr.get('neighbourhood') or addr.get('city')
    except Exception:
        return 'Bairro desconhecido'

# === inicializa captura de vídeo na Iriun Webcam ===
# Tente primeiro pelo nome no Windows DirectShow; se falhar, tenta índice 1
cap = cv2.VideoCapture("video=Iriun Webcam", cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Iriun Webcam não abriu por nome, tentando índice 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir a Iriun Webcam (nome ou índice).")

frame_counts = {}

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

        # conta detecção consecutiva
        if name != 'Desconhecido':
            frame_counts[name] = frame_counts.get(name, 0) + 1
        else:
            continue

        # alerta quando confirmado em X frames
        if frame_counts[name] >= CONSEC_FRAMES:
            # ajusta box ao tamanho original
            top, right, bottom, left = [int(v / FRAME_RESIZE) for v in loc]
            face_img = frame[top:bottom, left:right]
            _, buf = cv2.imencode('.jpg', face_img)
            photo = io.BytesIO(buf)

            bairro    = get_neighborhood()
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            caption   = f"👤 {name} em Vitória/ES, Brazil\n🕒 {timestamp}"

            resp = requests.post(
                TELEGRAM_API_URL,
                data={'chat_id': CHAT_ID, 'caption': caption, 'parse_mode':'Markdown'},
                files={'photo': ('face.jpg', photo.getvalue())}
            )
            if resp.ok:
                print(f"Alerta enviado: {name} em Vitória/ES, Brazil")
            else:
                print("Falha no envio:", resp.text)

            cap.release()
            cv2.destroyAllWindows()
            exit(0)

    # desenha caixas e nomes
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