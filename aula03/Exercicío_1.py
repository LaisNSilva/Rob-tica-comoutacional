from __future__ import division

import cv2
import os
import math
import numpy as np

# def ponto_fuga (p1, p2, q1, q2):
#     m1 = (p2[1]-p1[1])/(p2[0]-p1[0])
#     m2 = (q2[1]-q1[1])/(q2[0]-q1[0])
#     h1 = p1[1] - m1*p1[0]
#     h2 = q1[1] - m2*q1[0]
    
#     x_fuga = (h2-h1)/(m1-m2)
#     y_fuga = m1*x_fuga +h1
    
#     return (x_fuga, y_fuga)
def ponto_de_fuga(ang1, ang2, linear1, linear2):
    if ang2 == ang1:
        ang2 = ang1 + 1 # para nao dar div por zero
    x_fuga = (-(linear2 - linear1))/(ang2 - ang1)
    y_fuga = ang1*x_fuga + linear1
    return (x_fuga, y_fuga)  

    

def calcula_coef_ang(A1, A2):
    coef_ang = (A2[1]-A1[1])/(A2[0]-A1[0])
    return coef_ang
                      
def calcula_coef_linear (A1, A2):
    coef_ang = (A2[1]-A1[1])/(A2[0]-A1[0])
    coef_linear = A1[1]-(coef_ang*A1[0])
    return coef_linear
    


cap = cv2.VideoCapture('VID_20200302_063445951.mp4')


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
    #cv2.imshow('gray', gray)
    
    # Mascara para o branco
    
    cap1 = np.array([249])
    cap2 = np.array([255])
    mask = cv2.inRange(gray, cap1, cap2)
    #cv2.imshow('mask', mask)
    
    hough_img_rgb2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    min_contrast = 100
    max_contrast = 200
    linhas = cv2.Canny(hough_img_rgb2, min_contrast, max_contrast)
    
    lines = cv2.HoughLinesP(linhas, 10, math.pi/180.0, 100, np.array([]), 45, 5)

    a,b,c = lines.shape

    hough_img_rgb = cv2.cvtColor(linhas, cv2.COLOR_GRAY2BGR)

    coeficientes_angular=[]
    coeficientes_linear=[]
        
    
    for i in range(a):
        # Faz uma linha ligando o ponto inicial ao ponto final, com a cor vermelha (BGR)
        cv2.line(hough_img_rgb, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 5, cv2.LINE_AA)
    
    
    
        
        coef_angular = calcula_coef_ang((lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]))
        coeficientes_angular.append(coef_angular)
                      
        coef_linear = calcula_coef_linear((lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]))
        coeficientes_linear.append(coef_linear)
                      
        
    menor = min(coeficientes_angular) 
    maior = max(coeficientes_angular) 
    i_menor = coeficientes_angular.index(menor)
    i_maior = coeficientes_angular.index(maior)
                      
    linear_menor = coeficientes_linear[i_menor]                  
    linear_maior = coeficientes_linear[i_maior]
    
    
    PG= ponto_de_fuga(maior, menor, linear_maior, linear_menor)
    
    # DESENHANDO PONTO DE FUGA
    cor=(255, 255, 0)
    cv2.circle(hough_img_rgb, (int(PG[0]), int(PG[1])), 20, cor)
    cv2.circle(hough_img_rgb, (int(PG[0]),int(PG[1])), 4, cor, 1)
    
    #para x=400
    x1=400
    y_menor = menor*x1 + linear_menor
    
    
    #para x=700
    x2=600
    y_maior = maior*x2 +linear_maior
    print(y_maior)
    
    verde=(0, 255, 0)
    #cv2.line(hough_img_rgb, (x1, y_menor), (int(PG[0]),int(PG[1]), verde, 5)
    #cv2.line(hough_img_rgb, (x2, y_maior), (int(PG[0]),int(PG[1]), verde, 5)

        
    cv2.imshow('hough_img_rgb', hough_img_rgb)
    
#     print("shape", hough_img_rgb.shape)
        
        
            

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



    
    
    