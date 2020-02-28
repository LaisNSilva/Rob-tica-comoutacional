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
    
    
    # Reconhecendo os contornos ROSA
    
    
    segmentado_caprosa = cv2.morphologyEx(maskrosa,cv2.MORPH_CLOSE,np.ones((4, 4)))
    
    img_out_rosa, contornos_rosa, arvore_rosa = cv2.findContours(segmentado_caprosa.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contornos_img_rosa = frame.copy() 
    cv2.drawContours(contornos_img_rosa, contornos_rosa, -1, [0, 0, 255], 3);
    
    maior_rosa = None
    maior_area_rosa = 0
    for c in contornos_rosa:
        area_rosa = cv2.contourArea(c)
        if area_rosa > maior_area_rosa:
            maior_area_rosa = area_rosa
            maior_rosa = c
            
    cv2.drawContours(contornos_img_rosa, [maior_rosa], -1, [0, 255, 255], 5);
    
    
    #cv2.imshow("contornos_img_rosa", contornos_img_rosa)
    
    # Reconhecendo os contornos AZUL
    

    segmentado_capazul = cv2.morphologyEx(maskazul,cv2.MORPH_CLOSE,np.ones((4, 4)))

    img_out_azul, contornos_azul, arvore_azul = cv2.findContours(segmentado_capazul.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contornos_img_azul = frame.copy()
    cv2.drawContours(contornos_img_azul, contornos_azul, -1, [0, 0, 255], 3);

    maior_azul = None
    maior_area_azul = 0
    for e in contornos_azul:
        area_azul = cv2.contourArea(e)
        if area_azul > maior_area_azul:
            maior_area_azul = area_azul
            maior_azul = e

    cv2.drawContours(contornos_img_azul, [maior_azul], -1, [0, 255, 255], 5);


    #cv2.imshow("contornos_img_azul", contornos_img_azul)
    
    contornos = contornos_img_azul + contornos_img_rosa
    cv2.imshow("contornos", contornos)
    
    # PONTO MÉDIO ROSA
    for p in range(maior_rosa.shape[0]):
        if len(maior_rosa)%2==0:
            if p == len(maior_rosa)/2:
                x_rosa = maior_rosa[p][0]
                y_rosa = maior_rosa[p][1]
        if len(maior_rosa)%2!=0:
            if p== (len(maior_rosa)+1)/2:
                x_rosa = maior_rosa[p][0]
                y_rosa = maior_rosa[p][1]
        
    # PONTO MÉDIO AZUL
    for o in range(maior_azul.shape[0]):
        if len(maior_azul)%2==0:
            if o == len(maior_azul)/2:
                x_azul = maior_azul[o][0]
                y_azul = maior_azul[o][1]
        if len(maior_azul)%2!=0:
            if o== (len(maior_azul)+1)/2:
                x_azul = maior_azul[o][0]
                y_azul = maior_azul[o][1]

    # Distância entre os centros das circunferências:
    h = ((x_azul-x_rosa)**2+(y_azul-y_rosa)**2)**0.5
    
    #Distância entre a folha e a camera do computador
    D = 624*14/h
    

    print ("A distância entre a folha e a camera é de {0}cm".format(D))
    


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()