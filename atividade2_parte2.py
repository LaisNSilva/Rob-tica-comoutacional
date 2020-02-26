import cv2

#cap = cv2.VideoCapture('hall_box_battery_1024.mp4')
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret == False:
        print("Codigo de retorno FALSO - problema para capturar o frame")

    # Our operations on the frame come here
    rgb = frame #  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    # cv2.imshow('frame',frame)
    cv2.imshow('gray', rgb)
    
    # Detctando circulos
    
    bordas = auto_canny(blur)
    
    circles = []
    
    bordas_color = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)
    
    circles = None
    circles=cv2.HoughCircles(bordas,cv2.HOUGH_GRADIENT,2,40,param1=50,param2=100,minRadius=5,maxRadius=60)
    
    if circles is not None:        
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            print(i)
            # draw the outer circle
            # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
            cv2.circle(bordas_color,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(bordas_color,(i[0],i[1]),2,(0,0,255),3)
        
        #Tentando detctar as cores - Lais
        for circles in cap:
            if value == #ef1864:
                cap1, cap2 = aux.ranges(colorpicker.value)
                mask = cv2.inRange(cap, cap1, cap2)
                #MARCAR NA IMAGEM O CILULO ROSA
            if value == #0f14ea:
                cap1, cap2 = aux.ranges(colorpicker.value)
                mask = cv2.inRange(cap, cap1, cap2)
                # MARCAR NA IMAGEM O CIRCULO AZUL
            

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
