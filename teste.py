import cv2, json
import numpy as np
import pyautogui
from ultralytics import YOLO
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon, mapping
import matplotlib.pyplot as plt
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image

# Caminho do ortomosaico TIFF
#file_path = "M:\\EMBRAPA\\ORTOMOSAICOS\\Sede_B3B4_1,5_190312\\3_dsm_ortho\\2_mosaic\\Sede_B3B4_1,5_190312_mosaic_group1.tif"
file_path = "M:\\EMBRAPA\\ORTOMOSAICOS\\SEDEB3B4_0-5CM_19-03-12_1-4\\3_dsm_ortho\\2_mosaic\\SEDEB3B4_0-5CM_19-03-12_1-4_transparent_mosaic_group1.tif"


model = YOLO("best.pt")
image = cv2.imread(file_path)
cv2.imwrite('img_cropped_before.png', image)
    
        
aoi = np.array([[860, 715], [1449, 951], [1326, 1314], [963,1152],[822,1542],[594,1452]], np.int32)
#aoi = np.array([[737948.53,737657.71],[737950.01,7737651.01],[7379494.12,7737647.85],[737947.59,7737648.25]],np.int32)
pts = aoi.reshape((-1, 1, 2))
mask = np.zeros(image.shape[:2], dtype=np.uint8)
cv2.fillPoly(mask, [pts], 255)
masked_image = cv2.bitwise_and(image, image, mask=mask)
x, y, w, h = cv2.boundingRect(pts)
cropped_image = masked_image[y:y+h, x:x+w]
image = cropped_image
cv2.imwrite('img_cropped.png', image)
    

#results = model.predict(source=image, show_labels=True, show_conf=True, max_det=190, save=True, save_crop=True, save_txt=True)
results = model.predict(source=image)

cont = 0
for result in results:
    for poly in result.masks.xy:
        poly = np.array(poly, dtype=np.int32)
        mask = np.zeros_like(image)

        cv2.fillPoly(mask, [poly], (255,) * mask.shape[2])
        imagem_mascarada = cv2.bitwise_and(image, mask)

        x, y, w, h = cv2.boundingRect(poly)
        recorte = imagem_mascarada[y:y+h, x:x+w]
        cv2.imwrite(f'./parcelas1/parcela-{cont}.png', recorte)
        cont += 1

