import cv2
import numpy as np
from ultralytics import YOLO
import time
from PIL import Image, ImageDraw, ImageFont
from collections import deque, Counter  # <<< NOVO: Importações para o sistema de votação

# ===================================================================
#           CLASSE V2E AVANÇADA
# ===================================================================
class V2E_Simulator_Pro:
    """
    Uma classe V2E avançada com limiar adaptativo, simulação de ruído
    e saída de lista de eventos no formato (x, y, t, p).
    """

    def __init__(self, contrast_threshold=0.2, noise_probability=0.001):
        self.contrast_threshold = contrast_threshold
        self.noise_probability = noise_probability
        self.prev_log_luminance = {}
        self.last_timestamp = {}

    def generate_event_list(self, frame, identifier='eye'):
        h, w = frame.shape[:2]
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        current_log_luminance = np.log(gray_frame.astype(np.float32) / 255.0 + 1e-5)

        if identifier not in self.prev_log_luminance:
            self.prev_log_luminance[identifier] = current_log_luminance
            return [], np.full((h, w, 3), 128, dtype=np.uint8)

        log_diff = current_log_luminance - self.prev_log_luminance[identifier]
        adaptive_threshold = np.abs(self.prev_log_luminance[identifier] * self.contrast_threshold)
        adaptive_threshold[adaptive_threshold < 0.02] = 0.02

        on_events_coords = np.where(log_diff > adaptive_threshold)
        off_events_coords = np.where(log_diff < -adaptive_threshold)

        vis_frame = np.full((h, w, 3), 128, dtype=np.uint8)
        vis_frame[on_events_coords] = [255, 255, 255]  # Eventos ON em branco
        vis_frame[off_events_coords] = [0, 0, 0]  # Eventos OFF em preto

        self.prev_log_luminance[identifier] = current_log_luminance

        return [], vis_frame


# ===================================================================
#           FUNÇÃO AUXILIAR PARA DESENHAR TEXTO
# ===================================================================
def draw_text_pil(image, text, position, font, color):
    color_rgb = (color[2], color[1], color[0])
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    draw.text(position, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


# ===================================================================
#           CONFIGURAÇÕES E CARREGAMENTO DOS MODELOS
# ===================================================================
try:
    print("Carregando modelos...")
    model_pose = YOLO('yolov8n-pose.pt')
    model_classifier = YOLO(
        r'C:\Users\netinhoklz\Downloads\Code\Apresentação Visão Computacional\train41\weights\best.pt')
except Exception as e:
    print(f"Erro ao carregar um dos modelos YOLO: {e}")
    exit()

v2e_pro_simulator = V2E_Simulator_Pro(contrast_threshold=0.2, noise_probability=0.0)
LEFT_EYE_INDEX = 2
BOX_SIZE = 90
HALF_BOX = BOX_SIZE // 2

# ===================================================================
#           PARÂMETROS DE ESTABILIDADE, CONFIANÇA E FONTE
# ===================================================================
ALPHA, frames_since_detection = 0.3, 5
last_center_x, last_center_y = None, None
PERSISTENCE_FRAMES, CLASSIFICATION_CONF_THRESHOLD = 3, 0.1

# --- PARÂMETROS PARA O VOTING CLASSIFIER ---
VOTING_WINDOW_SIZE = 75  # <<< NOVO: Número de frames para considerar na votação
recent_predictions = deque(maxlen=VOTING_WINDOW_SIZE) # <<< NOVO: Fila para armazenar as últimas predições
voted_label = "Analisando..." # <<< NOVO: Rótulo final baseado na votação

COLOR_FOCO, COLOR_DISTRACAO, COLOR_OUTRO = (0, 255, 0), (0, 0, 255), (0, 255, 255)
box_color = COLOR_OUTRO

try:
    font_path = "arial.ttf"
    font_status = ImageFont.truetype(font_path, 20)
    font_timer = ImageFont.truetype(font_path, 18)
    font_timer_title = ImageFont.truetype(font_path, 22)
    print(f"Fonte '{font_path}' carregada com sucesso.")
except IOError:
    print(f"AVISO: Fonte '{font_path}' não encontrada. Usando fonte padrão do OpenCV.")
    font_status = None

time_per_class, last_frame_time = {}, time.time()

# ====================================================================
#           LOOP PRINCIPAL DA APLICAÇÃO
# ====================================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro: Não foi possível abrir a câmera.")
    exit()

print(f"\nPipeline iniciado. Pressione 'q' para sair.")
v2e_eye_image = np.full((BOX_SIZE, BOX_SIZE, 3), 128, dtype=np.uint8)

while True:
    ret, frame = cap.read()
    if not ret: break

    current_time = time.time()
    delta_time = current_time - last_frame_time
    last_frame_time = current_time

    frame = cv2.flip(frame, 1)
    frame_height, frame_width = frame.shape[:2]

    pose_results = model_pose(frame, stream=True, verbose=False)
    eye_detected_this_frame = False
    for r in pose_results:
        if r.keypoints.xy.shape[1] > 0:
            person_kpts, person_confs = r.keypoints.xy[0], r.keypoints.conf[0]
            if person_confs[LEFT_EYE_INDEX] > 0.5:
                eye_detected_this_frame = True
                frames_since_detection = 0
                current_center_x, current_center_y = int(person_kpts[LEFT_EYE_INDEX][0]), int(
                    person_kpts[LEFT_EYE_INDEX][1])
                if last_center_x is None:
                    last_center_x, last_center_y = current_center_x, current_center_y
                else:
                    last_center_x = int(ALPHA * current_center_x + (1 - ALPHA) * last_center_x)
                    last_center_y = int(ALPHA * current_center_y + (1 - ALPHA) * last_center_y)
                break
    if not eye_detected_this_frame:
        frames_since_detection += 1

    clean_class_name = None
    if voted_label.startswith("Status:"): # <<< MODIFICADO para usar o novo rótulo
        clean_class_name = voted_label.split(' ')[1]

    if last_center_x is not None and frames_since_detection < PERSISTENCE_FRAMES:
        x1, y1, x2, y2 = last_center_x - HALF_BOX, last_center_y - HALF_BOX, last_center_x + HALF_BOX, last_center_y + HALF_BOX
        safe_x1, safe_y1, safe_x2, safe_y2 = max(0, x1), max(0, y1), min(frame_width, x2), min(frame_height, y2)

        # --- LÓGICA DE COR CORRIGIDA ---
        if clean_class_name:
            class_name_lower = clean_class_name.lower()
            if 'focus' in class_name_lower:
                box_color = COLOR_FOCO
            elif 'distracted' in class_name_lower: # Verifique se o nome da classe é 'distracted'
                box_color = COLOR_DISTRACAO
            else:
                box_color = COLOR_OUTRO
        else:
            box_color = COLOR_OUTRO

        cv2.rectangle(frame, (safe_x1, safe_y1), (safe_x2, safe_y2), box_color, 2)

        eye_crop_bgr = frame[safe_y1:safe_y2, safe_x1:safe_x2]
        if eye_crop_bgr.size > 0:
            eye_crop_resized = cv2.resize(eye_crop_bgr, (BOX_SIZE, BOX_SIZE))
            _, v2e_eye_image = v2e_pro_simulator.generate_event_list(eye_crop_resized, identifier='left_eye')
            class_results = model_classifier(v2e_eye_image, verbose=False)

            # <<< INÍCIO DA NOVA LÓGICA DE VOTAÇÃO >>>
            if class_results:
                result, top1_conf = class_results[0], class_results[0].probs.top1conf
                if top1_conf >= CLASSIFICATION_CONF_THRESHOLD:
                    # Adiciona a predição atual à fila de predições recentes
                    class_name = model_classifier.names[result.probs.top1]
                    recent_predictions.append(class_name)

            # Realiza a votação se houver predições na fila
            if recent_predictions:
                # Conta a ocorrência de cada classe na janela de tempo
                vote_counts = Counter(recent_predictions)
                # Pega a classe mais votada
                top_class, top_count = vote_counts.most_common(1)[0]
                # Calcula a "confiança" da votação (quão dominante é a classe mais votada)
                vote_confidence = top_count / len(recent_predictions)
                voted_label = f"Status: {top_class} ({vote_confidence:.2f})"
            else:
                # Mantém um status inicial se ainda não houver predições
                voted_label = "Analisando..."
            # <<< FIM DA NOVA LÓGICA DE VOTAÇÃO >>>

        if font_status:
            frame = draw_text_pil(frame, voted_label, (safe_x1, safe_y1 - 25), font_status, box_color) # <<< MODIFICADO
        else:
            cv2.putText(frame, voted_label, (safe_x1, safe_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2) # <<< MODIFICADO

    else:
        last_center_x, last_center_y = None, None
        voted_label = "Olho nao detectado" # <<< MODIFICADO
        recent_predictions.clear() # <<< NOVO: Limpa o histórico de predições quando o olho é perdido
        box_color = COLOR_OUTRO

    if clean_class_name:
        time_per_class[clean_class_name] = time_per_class.get(clean_class_name, 0) + delta_time

    y_pos = 10
    if font_status:
        frame = draw_text_pil(frame, "Tempo por estado:", (10, y_pos), font_timer_title, (255, 255, 255))
        y_pos += 30
        for name, total_time in time_per_class.items():
            text = f"- {name}: {int(total_time)}s"
            frame = draw_text_pil(frame, text, (10, y_pos), font_timer, (255, 255, 255))
            y_pos += 25
    else:
        cv2.putText(frame, "Tempo por estado:", (10, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 45
        for name, total_time in time_per_class.items():
            text = f"- {name}: {int(total_time)}s"
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_pos += 25

    cv2.imshow('Detector de Olhos', frame)
    cv2.imshow('Olho em V2E (Entrada do Classificador)', v2e_eye_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Fechando o programa.")
cap.release()
cv2.destroyAllWindows()