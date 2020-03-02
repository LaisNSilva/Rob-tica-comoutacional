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
    
    mascara_blur = cv2.blur(mask, (3,3))
    
    mask = mascara_blur

    cv2.imshow("mask", mask)
    
    def auto_canny(image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image)

        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)

        # return the edged image
        return edged
    
    #Transformada de Hough Circles
    retorno, mask_limiar = cv2.threshold(mask, 100 ,255, cv2.THRESH_BINARY)

    bordas = auto_canny(mask_limiar)
    circles=cv2.HoughCircles(image=bordas,method=cv2.HOUGH_GRADIENT,dp=2.5,minDist=40,param1=50,param2=100,minRadius=5,maxRadius=50)
    mask_limiar_rgb = cv2.cvtColor(mask_limiar, cv2.COLOR_GRAY2RGB)
    bordas_rgb = cv2.cvtColor(bordas, cv2.COLOR_GRAY2RGB)
    
    
    if circles is not None:        
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            print(i)
            # draw the outer circle
            # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
            cv2.circle(bordas_rgb,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(bordas_color,(i[0],i[1]),2,(0,0,255),3)
    
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
