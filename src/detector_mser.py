import cv2
import numpy as np

class PanelDetectorMSER: 
    def __init__(self):
        #Inciar el detecto MSER
        #Los parámetros (delta, min_area, max_area...) tengo que ajustarlos para mejorar el resultado. 
        self.mser = cv2.MSER_create( delta =5, min_area=600, max_area= 50000)
        
    def detect(self, image): 
        """
        Recibe imágen BGR y devuelve lista de boundig boxes : [x1, y1, x2, y2, score]
        Score temporal 1.0 (Rubén calcula el real)
        """
        
        boxes = []
        
        #1. Pasar la imagen a niveles de gris
        gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        #Mejorar contrastes si MSER no lo detecta bien 
        #gray = cv2.equalizeHist(gray)
        
        #2. Detectar regiones MSER
        regions, _ = self.mser.detectRegions(gray)
        
        alto_img, ancho_img = image.shape[:2]
        
        for p in regions: 
            #3. Pasar los píxeles de la región a un rectángulo
            x, y, w, h = cv2.boundingRect(p)
            
            #4. Filtrar por aspecto y tamaño
            aspect_ratio = w / float(h)
            if aspect_ratio < 0.8 or aspect_ratio > 4.5:
                continue
            
            #5. Agrandar un poco el rectángulo
            exp_w = int(w * 0.15)
            exp_h = int(h * 0.15)
            
            x1 = max(0, x - exp_w)
            y1 = max(0, y - exp_h)
            x2 = min(ancho_img , x + w + exp_w)
            y2 = min(alto_img, y + h + exp_h)
            
            #Añadir la caja a la lista.
            #SCORE TEMPORAL 1.0
            boxes.append([x1, y1, x2, y2, 1.0])
    
        return boxes