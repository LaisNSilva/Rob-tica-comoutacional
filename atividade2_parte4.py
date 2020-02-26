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
            
    # CONTORNANDO OS CIRCULOS
    
    segmentado_circle = cv2.morphologyEx(circle,cv2.MORPH_CLOSE,np.ones((4, 4)))
    
    img_out, contornos, arvore = cv2.findContours(segmentado_circle.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    contornos_img = circle.copy()
    cv2.drawContours(contornos_img, contornos, -1, [0, 0, 255], 3);
    
    x,y,w,h = cv2.cv2.boundingRect(opening)
    
    cv2.cv2.circle(frame,(int(x+w/2), int(y+h/2)), 5,(0,0,255), -1)
    
    # Calculando a distancia do centro de um circulo para outro
    
    h = ((x1-x2)**2 + (y1-y2)**2)**0.5
    
    # Trançando uma linha de um centro ao outro
    
    #Traçando uma linha horizondo
    
    # Pegando o ângulo entre essas linhas
    
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
