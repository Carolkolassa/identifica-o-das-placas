import cv2
import cvzone
from ultralytics import YOLO

cap = cv2.VideoCapture('video7.mp4')
modelo = YOLO('TCC10.pt')

classNames = ['de a preferencia', 'pare', 'permitido estacionar', 'ponto de onibus', 'proibido estacionar',
              'proibido parar e estacionar', 'proibido virar a direita', 'proibido virar a esquerda',
              'saliencia ou lombada', 'sentido obrigatorio', 'sentido proibido', 'siga em frente ou a direita',
              'siga em frente ou a esquerda', 'transito de pedestres', 'velocidade maxima permitida',
              'vire a direita', 'vire a esquerda']

cor = (0,0,255)

while True:
    success,img = cap.read()
    results = modelo(img,stream=True)
    for r in results:
        for box in r.boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            #Confiança
            conf = int(box.conf[0] * 100)
            #Extrair Classe
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # print(x,y,w,h,cls,conf)
            if conf >=30:
                if currentClass == 'de a preferencia' or currentClass =='pare' or currentClass =='permitido estacionar' or currentClass == 'proibido estacionar' or currentClass =='proibido parar e estacionar' or currentClass =='proibido virar a direita' or currentClass =='proibido virar a esquerda'  or currentClass == 'sentido obrigatorio' or currentClass == 'sentido proibido' or currentClass == 'siga em frente ou a direita' or currentClass == 'siga em frente ou a esquerda'  or currentClass == 'velocidade maxima permitida' or currentClass == 'vire a direita' or currentClass =='vire a esquerda':
                    cor = (0,0,255)
                elif currentClass == 'ponto de onibus' or 'saliencia ou lombada' or 'transito de pedestres':
                    cor = (39, 186, 186)

                cvzone.putTextRect(img,f'{classNames[cls]} {conf}',(x1,y1-5),scale=1,thickness=1,colorB=cor, colorT=(255,255,255), colorR=cor)
                cv2.rectangle(img, (x1, y1), (x2, y2), cor,2)
    cv2.imshow('Image',img)
    cv2.waitKey(1)