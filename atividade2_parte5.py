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
                cap1r, cap2r = aux.ranges(colorpicker.value)
                mask1 = cv2.inRange(cap, cap1r, cap2r)
                #MARCAR NA IMAGEM O CILULO ROSA
                
                
                hough_rosa = canny_img.copy() 

                lines = cv2.HoughLinesP(hough_img, 10, math.pi/180.0, 100, np.array([]), 45, 5)

                a,b,c = lines.shape

                hough_rosa_rgb = cv2.cvtColor(hough_rosa, cv2.COLOR_GRAY2BGR)

                for i in range(a):
                    # Faz um circulo ligando o ponto inicial ao ponto final, com a cor rosa (BGR)
                    cv2.circle(hough_rosa_rgb, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 5, cv2.LINE_AA)
                    
                    
            if value == #0f14ea:
                cap1a, cap2a = aux.ranges(colorpicker.value)
                mask2 = cv2.inRange(cap, cap1a, cap2a)
                # MARCAR NA IMAGEM O CIRCULO AZUL
                
                hough_azul = mask2.copy() 

                lines = cv2.HoughLinesP(hough_azul, 10, math.pi/180.0, 100, np.array([]), 45, 5)

                a,b,c = lines.shape

                hough_azul_rgb = cv2.cvtColor(hough_azul, cv2.COLOR_GRAY2BGR)

                for i in range(a):
                    # Faz uma linha ligando o ponto inicial ao ponto final, com a cor vermelha (BGR)
                    cv2.line(hough_azul_rgb, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 5, cv2.LINE_AA)
                 
               
            

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()