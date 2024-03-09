import cv2
import numpy as np

def ordenarPuntos(puntos):
    n_puntos = np.concatenate([puntos[0], puntos[1], puntos[2], puntos[3]]).tolist()
    y_order = sorted(n_puntos,key=lambda n_puntos:n_puntos[1])
    x1_order = y_order[:2]
    x1_order = sorted(x1_order,key=lambda x1_order:x1_order[0])
    x2_order = y_order[2:4]
    x2_order= sorted(x2_order, key=lambda x2_order:x2_order[0])
    return[x1_order[0], x1_order[1], x2_order[0], x2_order[1]]

def alineamiento(imagen, ancho, alto):
    imagen_alineada=None
    grices= cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    tipo_umbral, umbral = cv2.threshold(grices, 150, 255, cv2.THRESH_BINARY)
    cv2.imshow("umbral", umbral )
    contorno = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contornos = sorted(contorno, key=cv2.contourArea, reverse=True)[:1]
    
    for i in contorno:
        epsilon = 0.01*cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, epsilon, True)
        if len(approx) == 4:
            puntos= ordenarPuntos(approx)
            puntos1 = np.float32(puntos)
            puntoss2 = np.float32([[0, 0], [ancho, 0], [0, alto], [ancho, alto]])
            M = cv2.getPerspectiveTransform(puntos1,puntoss2)
            imagen_alineada= cv2.warpPerspective(imagen, M, (ancho, alto))
    return imagen_alineada

capuraVIdeo = cv2.VideoCapture(0)

while True:
    tipoCamara,camara = capuraVIdeo.read() 
    if tipoCamara == False:
        print("no se dectto la camara")
        break
    imagen_A6 = alineamiento(camara, ancho=720, alto=480)
    if imagen_A6 is not None: 
        puntos = []
        imagen_gris = cv2.cvtColor(imagen_A6, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(imagen_gris, (5, 5), 1)
        _,umbral2 = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
        cv2.imshow("umbral", umbral2)
        contorno2 = cv2.findContours(umbral2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(imagen_A6, contorno2, -1, (255,0,0),1)
        suma1= 0.0
        suma2= 0.0
        
        for i_2 in contorno2:
            area = cv2.contourArea(i_2)
            moments = cv2.moments(i_2)
            
            if moments["m00"] == 0:
                moments["m00"] = 1.0
            x = int(moments["m10"] / moments["m00"])
            y = int(moments["m01"] / moments["m00"])
            
            if 2500 < area < 3000:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(imagen_A6, "100", (x, y), font, 0.75, (0, 255, 1), 2)
                suma1 += 0.2
            
            if 1000 < area < 2000:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(imagen_A6, "50", (x, y), font, 0.75, (0, 255, 1), 2)
                suma2 += 0.1

        total = suma1 + suma2
        print(f"suma total: {total:.2}")
        cv2.imshow("imagen", imagen_A6)
        cv2.imshow("camara", camara)

        
    if cv2.waitKey(1)== ord(" "): 
        break

capuraVIdeo.release()
cv2.destroyAllWindows()