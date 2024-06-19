# import numpy as np
# import rasterio
# import matplotlib.pyplot as plt

# # Função para calcular NGRDI
# def calculate_NGRDI(red_band, green_band):
#     return (green_band - red_band) / (green_band + red_band)

# # Carregar o ortomosaico TIFF
# file_path = "caminho_para_o_ortomosaico.tiff"

# with rasterio.open(file_path) as dataset:
#     # Ler as bandas vermelha e verde
#     red_band = dataset.read(1).astype(np.float32)
#     green_band = dataset.read(2).astype(np.float32)

#     # Calcular NGRDI
#     ngrdi = calculate_NGRDI(red_band, green_band)

#     # Definir área de interesse (exemplo: coordenadas de pixel)
#     # Substitua pelos valores reais de sua área de interesse
#     x1, y1, x2, y2 = 100, 100, 200, 200

#     # Calcular estatísticas dentro da área de interesse
#     ngrdi_roi = ngrdi[y1:y2, x1:x2]
#     mean_ngrdi = np.mean(ngrdi_roi)
#     max_ngrdi = np.max(ngrdi_roi)
#     min_ngrdi = np.min(ngrdi_roi)

#     print("Média de NGRDI na área de interesse:", mean_ngrdi)
#     print("Valor máximo de NGRDI na área de interesse:", max_ngrdi)
#     print("Valor mínimo de NGRDI na área de interesse:", min_ngrdi)

#     # Plotar NGRDI
#     plt.imshow(ngrdi, cmap='RdYlGn')  # Escolha de colormap
#     plt.colorbar(label='NGRDI')
#     plt.title('NGRDI Map')
#     plt.show()

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


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    

# Função para calcular NGRDI com tratamento para divisão por zero
def calculate_NGRDI(red_band, green_band):
    with np.errstate(divide='ignore', invalid='ignore'):
        ngrdi = (green_band - red_band) / (green_band + red_band)
        ngrdi[~np.isfinite(ngrdi)] = 0  # Substitui NaNs e infinitos por 0
    return ngrdi

# Caminho do ortomosaico TIFF
file_path = "M:\\EMBRAPA\\ORTOMOSAICOS\\Sede_B3B4_1,5_190312\\3_dsm_ortho\\2_mosaic\\Sede_B3B4_1,5_190312_mosaic_group1.tif"
#file_path = "M:\\EMBRAPA\\ORTOMOSAICOS\\SEDEB3B4_0-5CM_19-03-12_1-4\\3_dsm_ortho\\2_mosaic\\SEDEB3B4_0-5CM_19-03-12_1-4_transparent_mosaic_group1.tif"

# Carregar o ortomosaico TIFF
with rasterio.open(file_path) as dataset:
    
    # Ler as bandas vermelha e verde
    red_band = dataset.read(1).astype(np.float32)
    green_band = dataset.read(2).astype(np.float32)

    # Calcular NGRDI apenas uma vez
    ngrdi = calculate_NGRDI(red_band, green_band)

    # Definir o polígono de interesse
    # modificar para que seja possível selecionar
    coords = [(737946.78,7737650.20),(737948.24,7737649.87),(737947.09,7737647.10),(737945.79,7737647.52)]
    #coords = [(737948.53,777657.71),(737950.01,7737651.01),(7379494.12,7737647.85),(737947.59,7737648.25)]
    polygon = Polygon(coords)
    
    # Verificar se o polígono está dentro dos limites do raster manualmente
    raster_left, raster_bottom, raster_right, raster_top = dataset.bounds
    poly_left, poly_bottom, poly_right, poly_top = polygon.bounds
    
    if not (raster_left <= poly_left <= raster_right and raster_left <= poly_right <= raster_right and
            raster_bottom <= poly_bottom <= raster_top and raster_bottom <= poly_top <= raster_top):
        raise ValueError("O polígono de interesse está fora dos limites do raster")

    # Definir a janela do recorte
    window = rasterio.windows.from_bounds(*polygon.bounds, transform=dataset.transform)
    
    # Ler apenas a área de interesse
    subset = dataset.read(window = window, indexes = [1,2])

    subset_ngrdi = calculate_NGRDI(subset[0], subset[1]) 

    # Calcular NGRDI só da area de interesse
    plt.figure(figsize=(10, 10))
    plt.imshow(subset_ngrdi, cmap='RdYlGn')
    plt.colorbar(label='NGRDI')
    plt.title('NGRDI Map (Area of Interest)')
    # plt.show()

    ngrdi_rgb = np.stack([ngrdi, ngrdi, ngrdi], axis=-1) * 255  # Convert NGRDI to RGB (0-255)


    # # Modify subset_ngrdi to replace non-vegetation pixels with black
    # non_veg_mask = np.where(subset_ngrdi == 0, 1, 0)
    # black_pixel_array = np.zeros_like(subset_ngrdi, dtype=np.uint8)
    # black_pixel_array[:] = np.full((54, 42), [0, 0, 0], dtype=np.uint8)   # Repeat and expand
    # subset_ngrdi[non_veg_mask] = black_pixel_array

    # plt.figure(figsize=(10, 10))
    # plt.imshow(subset_ngrdi)
    # plt.colorbar(label='NGRDI (RGB)')
    # plt.title('NGRDI with Non-Vegetation Pixels Represented as Black')
    # plt.show()

    
    #
    #Considerar apenas onde possui planta dentro do polígono
    #
    #

    out_image, out_transform = mask(dataset, [mapping(polygon)], crop=True)
    ngrdi_masked = calculate_NGRDI(out_image[0], out_image[1])

    # ... (código para calcular NGRDI e criar máscara)

    # Plotar o NGRDI mascarado
    plt.figure(figsize=(10, 10))  # Ajuste o tamanho da figura conforme desejado
    plt.imshow(ngrdi_masked, cmap='RdYlGn')  # Utilize um mapa de cores adequado para NGRDI (ex: 'RdYlGn')
    plt.colorbar(label='NGRDI')  # Adicione a barra de cores com o rótulo 'NGRDI'
    plt.title('NGRDI Map (Área de Interesse)')  # Defina o título da figura
    # plt.show()  # Exiba a figura


    # Calcular estatísticas dentro da área de interesse
    ngrdi_valid_infinite = ngrdi_masked[np.isfinite(ngrdi_masked)]
    ngrdi_valid = ngrdi_valid_infinite[ngrdi_valid_infinite > 150] #comparar com o resultado do modelo e 
    mean_ngrdi = np.mean(ngrdi_valid)
    max_ngrdi = np.max(ngrdi_valid)
    min_ngrdi = np.min(ngrdi_valid)

    print("Média de NGRDI na área de interesse:", mean_ngrdi)
    print("Valor máximo de NGRDI na área de interesse:", max_ngrdi)
    print("Valor mínimo de NGRDI na área de interesse:", min_ngrdi)

    # Plotar a área recortada
    plt.figure(figsize=(8, 8))
    plt.imshow(ngrdi_masked, cmap='RdYlGn')
    plt.colorbar(label='NGRDI')
    plt.title('NGRDI Map - Área Recortada')
    
    # Plotar a imagem original com a máscara sobreposta
    plt.figure(figsize=(10, 10))
    plt.imshow(ngrdi, cmap='RdYlGn', extent=(dataset.bounds.left, dataset.bounds.right, dataset.bounds.bottom, dataset.bounds.top))
    plt.colorbar(label='NGRDI')
    plt.title('NGRDI Map')

    # Adicionar o polígono na imagem
    x, y = polygon.exterior.xy
    plt.plot(x, y, color='blue', linewidth=2, solid_capstyle='round', zorder=2)

    # plt.show()

    


    model = YOLO("best.pt")
    image = cv2.imread(file_path)
        
            
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
      

    results = model.predict(source=image, show_labels=True, show_conf=True, max_det=190, save=True, save_crop=True, save_txt=True)
    cont = 0
    for polygon in results[0].masks.xy:
        polygon = np.array(polygon, dtype=np.int32)
        #print("polígono : "+str(cont) + " = " + str(polygon))
        mask = np.zeros_like(image)

        cv2.fillPoly(mask, [polygon], (255,) * mask.shape[2])

        imagem_mascarada = cv2.bitwise_and(image, mask)

        x, y, w, h = cv2.boundingRect(polygon)
        recorte = imagem_mascarada[y:y+h, x:x+w]
        cv2.imwrite(f'./parcelas/parcela-{cont}.jpg', recorte)
        cont += 1

    # list = []
    
    # for result in results:
    #         boxes = result.boxes.cpu().numpy()
    #         for box in boxes:
    #             cls = int(box.cls[0])
    #             path = result.path
    #             class_name = model.names[cls]
    #             conf = int(box.conf[0]*100)
    #             bx = box.xywh.tolist()
    #             df = pd.DataFrame({'path': path,'class_name': class_name, 'class_id': cls, 'confidence': conf, 'box_coord': bx})
    #             list.append(df)

    #             # dumped = json.dumps(result.masks.xy, cls=NumpyEncoder)
    #             # with open('data.json', 'w') as f:
    #             #     json.dump(dumped, f)

    #         print(len(result.masks.xy))
    #         for polygon in result.masks.xy:
    #             cont = 0
    #             coordinates = polygon
    #             pyauto_coordinates = [(x,y) for x,y in coordinates]
    #             # pyautogui.click(pyauto_coordinates[0], button='left', interval=0.1)  # Start at first point, press and hold
    #             # for i in range(1, len(pyauto_coordinates)):  # Connect remaining points
    #             #     pyautogui.moveTo(pyauto_coordinates[i])
    #             # pyautogui.click(pyauto_coordinates[0], button='left')  # Release at first point to close polygon
    #             # if cont < 1:
    #             #     print(pyauto_coordinates)
    #             #     cont +=1
    #             # print("\n\n")
    #         # y = result.masks.xy
    #         # p = Polygon(y)
    #         # fig, ax = plt.subplots()

    #         # ax.add_patch(p)
    #         # ax.set_xlim([0,3])
    #         # ax.set_ylim([0,3])
    #         # plt.show()

    #         # image = Image.new("RGB", (640, 480))

    #         # draw = ImageDraw.Draw(image)

    #         # # points = ((1,1), (2,1), (2,2), (1,2), (0.5,1.5))
    #         # points = result.masks.xy
    #         # draw.polygon((points), fill=200)

    #         # image.show()
    #         # x , y = result.masks.xy[0]
    #         # plt.plot(x,y, color='red', linewidth=2, solid_capstyle='round', zorder=2)

    #         # plt.show()

    #         #
    #         # Definir nomenclatura conforme definido anteriormente - linha coluna - é preciso processamento
    #         #


    # # df = pd.concat(list)

    # # df.to_csv('predicted_labels.csv', index=False)
  
    # #results.save(save_dir='/results')  
    # #print('Predição salva em /results'

