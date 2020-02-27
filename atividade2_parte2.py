import cv2
import auxiliar as aux
import numpy as np

#cap = cv2.VideoCapture('hall_box_battery_1024.mp4')
cap = cv2.VideoCapture(0)

rosa="#ff004d"
azul="#0173fe"

 

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret == False:
        print("Codigo de retorno FALSO - problema para capturar o frame")

    # Our operations on the frame come here
    rgb = frame #  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cap_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Display the resulting frame
    # cv2.imshow('frame',frame)
    cv2.imshow('gray', rgb)
    
   
    #Tentando detctar as cores - Lais
    #rosa
    cap1r, cap2r = aux.ranges(rosa)
    maskrosa = cv2.inRange(cap_hsv, cap1r, cap2r)
    
    #azul
    cap1a, cap2a = aux.ranges(azul)
    maskazul = cv2.inRange(cap_hsv, cap1a, cap2a)
    

    mask = maskrosa + maskazul

    cv2.imshow("mask", mask)
    
    # MARCAR ROSA
    segmentado_cor_rosa = cv2.morphologyEx(maskrosa,cv2.MORPH_CLOSE,np.ones((10, 10)))
    selecao_rosa = cv2.bitwise_and(frame, frame, mask=segmentado_cor_rosa)
    
    
    
    # MARCAR AZUL
    segmentado_cor_azul = cv2.morphologyEx(maskazul,cv2.MORPH_CLOSE,np.ones((10, 10)))
    selecao_azul = cv2.bitwise_and(frame, frame, mask=segmentado_cor_azul)
    
    selecao = selecao_rosa + selecao_azul
    cv2.imshow("selecao", selecao)
    
    


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the captureq
cap.release()
cv2.destroyAllWindows()
