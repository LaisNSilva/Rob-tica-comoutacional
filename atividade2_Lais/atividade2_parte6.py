import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import pi
import matplotlib.cm as cm

#cap = cv2.VideoCapture('hall_box_battery_1024.mp4')
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
    cv2.imshow('gray', rgb)
    
    # Padronizando a palavra insper
    insper = cv2.imread('insper.png',0)
    
    sift = cv2.xfeatures2d.SIFT_create()
    kpts = sift.detect(insper)
    x = [k.pt[0] for k in kpts]
    y = [k.pt[1] for k in kpts]
    # s will correspond to the neighborhood area
    s = [(k.size/2)**2 * pi for k in kpts]
    
    # Esta função não é importante, apenas desenha as features na imagem
    def draw_points(matches, img_original, img_cena, kp1, kp2): 
        for mat in matches:
                # Get the matching keypoints for each of the images
                img1_idx = mat[0].queryIdx
                img2_idx = mat[0].trainIdx

                # x - columns
                # y - rows
                (x1,y1) = kp1[img1_idx].pt
                (x2,y2) = kp2[img2_idx].pt

                # Draw a small circle at both co-ordinates
                # radius 4
                # colour blue
                # thickness = 1
                cv2.circle(img_original, (int(x1),int(y1)), 4, (255, 0, 0), 1)
                #cv2.circle(img_cena, (int(x2),int(y2)), 4, (255, 0, 0), 1)
                
                
    # Número mínimo de pontos correspondentes
    MIN_MATCH_COUNT = 10

    img_original = cv2.imread('insper.png',0)      # Gabarito / Imagem a procurar
    img_cena = gray # Imagem do cenario - puxe do video para fazer isto

    # Versões RGB das imagens, para plot
    original_rgb = cv2.cvtColor(img_original, cv2.COLOR_GRAY2RGB)
    cena_rgb = cv2.cvtColor(img_cena, cv2.COLOR_GRAY2RGB)

    # Imagem de saída
    out = cena_rgb.copy()


    # Cria o detector SIFT
    sift = cv2.xfeatures2d.SIFT_create()

    # Encontra os pontos únicos (keypoints) nas duas imagems
    kp1, des1 = sift.detectAndCompute(img_original ,None)
    kp2, des2 = sift.detectAndCompute(img_cena,None)

    # Configurações do algoritmo FLANN que compara keypoints e ver correspondências - não se preocupe com isso
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    # Configura o algoritmo de casamento de features que vê *como* o objeto que deve ser encontrado aparece na imagem
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Tenta fazer a melhor comparacao usando o algoritmo
    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)


    if len(good)>MIN_MATCH_COUNT:
        # Separa os bons matches na origem e no destino
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
        img2b = cv2.polylines(out,[np.int32(dst)],True,(255,0,0),3, cv2.LINE_AA)

    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)

    # Vocês não precisam disto: desenham os pontos
    draw_points(matches, img_original, img_cena, kp1, kp2)
    
    cv2.imshow('out', out)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
