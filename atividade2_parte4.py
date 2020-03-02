import cv2
import auxiliar as aux
import numpy as np
import math

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
    
    mascara_blur = cv2.blur(mask, (3,3))
    
    mask = mascara_blur

    cv2.imshow("mask", mask)
    
    
     # Contornos ROSA:
    
    frame_out_rosa, contornos_rosa, arvore = cv2.findContours(maskrosa, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    
    mask_rgb_rosa = cv2.cvtColor(maskrosa, cv2.COLOR_GRAY2RGB) 
    contornos_frame_rosa = mask_rgb_rosa.copy() 
    
    cv2.drawContours(contornos_frame_rosa, contornos_rosa, -1, [0, 255, 255], 5);
    
    maior_rosa = None
    maior_area_rosa = 0
    for c in contornos_rosa:
        area_rosa = cv2.contourArea(c)
        if area_rosa > maior_area_rosa:
            maior_area_rosa = area_rosa
            maior_rosa = c
            
            
    cv2.drawContours(contornos_frame_rosa, [maior_rosa], -1, [255, 0, 0], 5);
   
    
    
    # Contornos AZUL:
    
    frame_out_azul, contornos_azul, arvore = cv2.findContours(maskazul, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    
    mask_rgb_azul = cv2.cvtColor(maskazul, cv2.COLOR_GRAY2RGB) 
    contornos_frame_azul = mask_rgb_azul.copy() 
    
    cv2.drawContours(contornos_frame_azul, contornos_azul, -1, [0, 255, 255], 5);
    
    maior_azul = None
    maior_area_azul = 0
    for c in contornos_azul:
        area_azul = cv2.contourArea(c)
        if area_azul > maior_area_azul:
            maior_area_azul = area_azul
            maior_azul = c
            
            
    cv2.drawContours(contornos_frame_azul, [maior_azul], -1, [255, 0, 0], 5);
    
    
    
    # ACHANDO OS CENTROS DOS CONTORNOS
    
    def center_of_contour(contorno):
        """ Retorna uma tupla (cx, cy) que desenha o centro do contorno"""
        M = cv2.moments(contorno)
    # Usando a expressão do centróide definida em: https://en.wikipedia.org/wiki/Image_moment
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (int(cX), int(cY))
    
    def crosshair(img, point, size, color):
        """ Desenha um crosshair centrado no point.
            point deve ser uma tupla (x,y)
            color é uma tupla R,G,B uint8
        """
        x,y = point
        cv2.line(img,(x - size,y),(x + size,y),color,5)
        cv2.line(img,(x,y - size),(x, y + size),color,5)

    font = cv2.FONT_HERSHEY_SIMPLEX

    def texto(img, a, p):
        #"""Escreve na img RGB dada a string a na posição definida pela tupla p"""
        cv2.putText(img, str(a), p, font,1,(0,50,100),2,cv2.LINE_AA)

    def auto_canny(image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image)

        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)

        # return the edged image
        return edged
    
    
    for c in contornos_rosa: 
        a = cv2.contourArea(c) # área
        p = center_of_contour(c) # centro de massa
        crosshair(contornos_frame_rosa, p, 20, (128,128,0))
        texto(contornos_frame_rosa, np.round(a,2),p)
        
    for e in contornos_azul: 
        b = cv2.contourArea(e) # área
        q = center_of_contour(e) # centro de massa
        crosshair(contornos_frame_azul, q, 20, (128,128,0))
        texto(contornos_frame_azul, np.round(b,2),q)  
        
    
    contornos_frame = contornos_frame_rosa + contornos_frame_azul
    cv2.imshow("contornos_frame", contornos_frame)
    
    # DESENHANDO LINHA DE UM CENTRO A OUTRO
    cv2.line(contornos_frame,(p[0],P[1]),(q[0],q[1]),(255,0,0),5)
    
    # DESENHANDO LINHA HORIZONTAL
    horizontal = [p[0]+5,p[1]]
    cv2.line(contornos_frame,(p[0],p[1]),(horizontal[0],horizontal[1]),(255,0,0),5)
    
    # CALCULANDO O ÂNGULO ENTRE AS DUAS LINHA
    def calculo_angulo(p1, p2, p3):
        p2p3 = ((p2[0]-p3[0])**2+(p2[1]-p3[1])**2)**0.5
        p1p2 = ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5
        p1p3 = ((p1[0]-p3[0])**2+(p1[1]-p3[1])**2)**0.5
        cos_angulo = ((p2p3**2)-(p1p2**2)-(p1p3**2))/(-2*p1p2*p1p3)
        angulo = math.acos(cos_angulo)
        return angulo
    
    real_angulo = calculo_angulo(p, q, horizontal)
    
    print ("O ângulo entre a reta e a horizontal é de {0}".format(real_angulo))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(contornos_frame,"O ângulo entre a reta e a horizontal é de {0}".format(real_angulo),(0,50), font, 1,(255,255,255),2,cv2.LINE_AA)
    
    
    
    
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()