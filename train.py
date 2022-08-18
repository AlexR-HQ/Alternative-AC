# importando librerias
import cv2
import mediapipe as mp
import numpy as np

# Variables para definir los metodos de las librerias
mp_drawing = mp.solutions.mediapipe.python.solutions.drawing_utils
mp_pose = mp.solutions.mediapipe.python.solutions.pose

## Calculo de Angulos
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

## Fuente de video (camara)
cap = cv2.VideoCapture(0)
## Fuente de video (Videos de prueba)
#cap = cv2.VideoCapture("PoseVideos/1.mp4")

# Variables del contador de Curl
counter = 0 
stage = None

## Configurar instancia de mediapipe
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Cambiar el color de la imagen a RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Hacer detección
        results = pose.process(image)
    
        # Volver a colorear a BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extraer puntos de referencia
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Obtener coordenadas
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Calcular ángulo
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # Visualizar ángulo
            cv2.putText(image, str(angle), 
                           tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            
            # Lógica del contador del Curl
            if angle > 160:
                stage = "down "
            if angle < 30 and stage =='down ':
                stage="up "
                counter +=1
                print(counter)
                       
        except:
            pass
        
        # Contador de curl de renderizado
        # Cuadro de estado de configuración
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
        # Datos de las reps
        cv2.putText(image, 'REPS', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # Datos de la etapa
        cv2.putText(image, 'STAGE', (65,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, 
                    (60,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        
        # Procesar detecciones
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
            )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()