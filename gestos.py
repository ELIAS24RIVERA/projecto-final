import cv2
import mediapipe as mp

class HandGestureRecognizer:
    def __init__(self):
        # Inicialización del modelo de MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.95,  # Aumentamos la confianza de detección
            min_tracking_confidence=0.9     # Aumentamos la confianza de seguimiento
        )
        self.mp_draw = mp.solutions.drawing_utils

    def recognize_gesture(self, hand_landmarks):
        # Obtener el estado de los dedos (levantados o bajados) con una mejor precisión
        thumb = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].y
        index = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        middle = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
        ring = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP].y
        pinky = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP].y

        # Mejorar la precisión del reconocimiento de gestos
        if thumb and not index and not middle and not ring and not pinky:
            return "👍", "Aprobación / Bien hecho"
        elif not thumb and index and not middle and not ring and not pinky:
            return "👎", "Desaprobación / No me gusta"
        elif not thumb and index and middle and not ring and not pinky:
            return "✌️", "Paz"
        elif not thumb and index and middle and ring and not pinky:
            return "🤘", "Actitud de rock"
        elif not thumb and index and middle and ring and pinky:
            return "👊", "Fuerza / Luchar"
        elif thumb and index and middle and ring and pinky:
            return "👌", "Todo bien"
        elif not thumb and not index and not middle and not ring and not pinky:
            return "🙏", "Por favor / Gracias"
        elif thumb and index and not middle and not ring and not pinky:
            return "🤲", "Ofrecer algo"
        elif thumb and index and pinky and not middle and not ring:
            return "💪", "Fuerza / Poder"
        elif thumb and not index and not middle and not ring and pinky:
            return "🤚", "Alto / Detente"
        elif not thumb and index and not middle and not ring and pinky:
            return "✋", "Alto / Detente"
        elif not thumb and not index and middle and not ring and not pinky:
            return "👋", "Hola / Adiós"
        elif not thumb and not index and not middle and ring and not pinky:
            return "🖐", "Adiós"
        elif not thumb and not index and not middle and not ring and pinky:
            return "🖖", "Larga vida y prosperidad"
        elif not thumb and index and not middle and ring and not pinky:
            return "👐", "Receptividad / Recibir"
        elif thumb and index and not middle and ring and pinky:
            return "🤗", "Abrazo"
        # Añadimos más gestos que se pueden reconocer según la combinación de los dedos
        else:
            return "Desconocido", "Gestos no reconocidos"

    def process_frame(self, frame):
        # Convertir el fotograma de BGR a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujar los puntos de referencia de la mano
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Reconocer y mostrar el gesto
                gesture, description = self.recognize_gesture(hand_landmarks)
                cv2.putText(
                    frame,
                    f"Gesto: {gesture}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    frame,
                    f"Descripción: {description}",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
        
        return frame

def main():
    cap = cv2.VideoCapture(0)  # Abrir la cámara
    recognizer = HandGestureRecognizer()  # Crear el reconocedor de gestos

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Procesar el fotograma
        frame = recognizer.process_frame(frame)
        
        # Mostrar el fotograma procesado
        cv2.imshow('Reconocimiento de Gestos de Mano', frame)
        
        # Terminar el bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # Liberar la cámara
    cv2.destroyAllWindows()  # Cerrar la ventana

if __name__ == "__main__":
    main()
