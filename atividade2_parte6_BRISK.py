import numpy as np
import cv2
from matplotlib import pyplot as plt
from math import pi
import matplotlib.cm as cm


cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret == False:
        print("Codigo de retorno FALSO - problema para capturar o frame")

    # Our operations on the frame come here
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    # cv2.imshow('frame',frame)
    cv2.imshow('gray', gray)
    
    insper_bgr = cv2.imread('insper.png')
    insper_gray = cv2.cvtColor(insper_bgr, cv2.COLOR_BGR2GRAY)
    
    brisk = cv2.BRISK_create() # Nota: numa versão anterior era a BRISK
    kpts = brisk.detect(insper_gray)
    x = [k.pt[0] for k in kpts]
    y = [k.pt[1] for k in kpts]
    # s will correspond to the neighborhood area
    s = [(k.size/2)**2 * pi for k in kpts]

    
    def find_homography_draw_box(kp1, kp2, img_cena):
    
        out = img_cena.copy()

        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)


        # Tenta achar uma trasformacao composta de rotacao, translacao e escala que situe uma imagem na outra
        # Esta transformação é chamada de homografia 
        # Para saber mais veja 
        # https://docs.opencv.org/3.4/d9/dab/tutorial_homography.html
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()



        h,w = img_original.shape
        # Um retângulo com as dimensões da imagem original
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

        # Transforma os pontos do retângulo para onde estao na imagem destino usando a homografia encontrada
        dst = cv2.perspectiveTransform(pts,M)


        # Desenha um contorno em vermelho ao redor de onde o objeto foi encontrado
        img2b = cv2.polylines(out,[np.int32(dst)],True,(255,255,0),5, cv2.LINE_AA)

        return img2b
    
    
    # Número mínimo de pontos correspondentes
    MIN_MATCH_COUNT = 10

    cena_bgr = frame # Imagem do cenario
    original_bgr = insper_bgr

    # Versões RGB das imagens, para plot
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    cena_rgb = cv2.cvtColor(cena_bgr, cv2.COLOR_BGR2RGB)
    
    # Versões grayscale para feature matching
    img_original = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    img_cena = cv2.cvtColor(cena_bgr, cv2.COLOR_BGR2GRAY)

    framed = None

    # Imagem de saída
    out = cena_rgb.copy()


    # Cria o detector BRISK
    brisk = cv2.BRISK_create()

    # Encontra os pontos únicos (keypoints) nas duas imagems
    kp1, des1 = brisk.detectAndCompute(img_original ,None)
    kp2, des2 = brisk.detectAndCompute(img_cena,None)

    # Configura o algoritmo de casamento de features que vê *como* o objeto que deve ser encontrado aparece na imagem
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)


    # Tenta fazer a melhor comparacao usando o algoritmo
    matches = bf.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)


    if len(good)>MIN_MATCH_COUNT:
        # Separa os bons matches na origem e no destino
        print("Matches found")    
        framed = find_homography_draw_box(kp1, kp2, cena_rgb)
    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))

    
    cv2.imshow('framed', framed)

    
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
