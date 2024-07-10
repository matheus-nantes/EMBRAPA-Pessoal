import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import shutil
from screeninfo import get_monitors
from tkinter import Tk, filedialog, simpledialog
from index_calculation import IndexCalculation

import csv

def select_file():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")])
    if not file_path:
        print("Nenhum arquivo selecionado. Encerrando o programa.")
        sys.exit(0)
    return file_path

file_path = select_file()
print(f"Arquivo selecionado: {file_path}")

runs_dir = './runs'
if os.path.exists(runs_dir):
    shutil.rmtree(runs_dir)

output_dir = './parcelas'
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

output_indices_dir = './parcelas_indice'
if os.path.exists(output_indices_dir):
    shutil.rmtree(output_indices_dir)
os.makedirs(output_indices_dir)

image = cv2.imread(file_path)
clone = image.copy()
aoi_points = []

def undo_last_point():
    if aoi_points:
        aoi_points.pop()

def select_aoi(image):
    fig, ax = plt.subplots()
    monitor = get_monitors()[0]
    fig.set_size_inches((monitor.width*0.9) / fig.dpi, (monitor.height*0.7) / fig.dpi)
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title("Clique para selecionar os pontos e delimitar a Área de Interesse.\nPressione 'u' para desfazer o último ponto.\n Pressione 'f' para finalizar.")

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            ix, iy = int(event.xdata), int(event.ydata)
            aoi_points.append((ix, iy))
            ax.plot(ix, iy, 'ro')
            if len(aoi_points) > 1:
                ax.plot([aoi_points[-2][0], aoi_points[-1][0]], [aoi_points[-2][1], aoi_points[-1][1]], 'b-')
            if len(aoi_points) > 2:
                ax.plot([aoi_points[-1][0], aoi_points[0][0]], [aoi_points[-1][1], aoi_points[0][1]], 'b--')
            fig.canvas.draw()

    def onkey(event):
        if event.key == 'u':
            undo_last_point()
            ax.clear()
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax.set_title("Clique para selecionar pontos. Pressione 'u' para desfazer o último ponto. Pressione 'f' para finalizar.")
            if aoi_points:
                ax.plot(*zip(*aoi_points), 'ro-')
                if len(aoi_points) > 1:
                    ax.plot([aoi_points[-1][0], aoi_points[0][0]], [aoi_points[-1][1], aoi_points[0][1]], 'b--')
            fig.canvas.draw()
        elif event.key == 'f':
            if len(aoi_points) > 2:
                aoi_points.append(aoi_points[0])
                fig.canvas.mpl_disconnect(cid)
                fig.canvas.mpl_disconnect(kid)
                plt.close(fig)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    kid = fig.canvas.mpl_connect('key_press_event', onkey)
    plt.show()

def show_cropped_image(image):
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title("Imagem recortada. Pressione 'p' para prosseguir com a predição. Pressione 'c' para cancelar.")

    def onkey(event):
        if event.key == 'p':
            fig.canvas.mpl_disconnect(kid)
            plt.close()
        elif event.key == 'c':
            print("Proceso encerrado espontaneamente")
            sys.exit(0)

    kid = fig.canvas.mpl_connect('key_press_event', onkey)
    plt.show()

def calculate_indices(red, green, blue, nir=None):
    def bi(red, green, blue):
        return (2 * green - red - blue) / (2 * green + red + blue)

    def gli(red, green, blue):
        return (2 * red - green - blue) / (2 * red + green + blue)

    def ngrdi(red, green, nir):
        return (green - red) / (green + red - nir)

    index_functions = {
        "BI": bi,
        "GLI": gli,
        "NGRDI": lambda r, g, nir: ngrdi(r, g, nir)
    }

    index_results = {}
    for index_name, func in index_functions.items():
        if index_name == "BI" or index_name == "GLI":
            index_results[index_name] = func(red, green, blue)
        elif index_name == "NGRDI" and nir is not None:
            index_results[index_name] = func(red, green, nir)

    return index_results

def process_parcelas(input_dir, output_dir):
    parcelas_files = os.listdir(input_dir)
    for filename in parcelas_files:
        if filename.endswith(".png"):
            input_path = os.path.join(input_dir, filename)
            image = cv2.imread(input_path)
            blue, green, red = cv2.split(image)

            index_calculator = IndexCalculation(red=red, green=green, blue=blue)
            hue_index = index_calculator.ngrdi()
            cv2.imwrite("./hue.jpg", hue_index)

            indices = calculate_indices(red, green, blue)

            for index_name, result in indices.items():
                output_index_dir = os.path.join(output_dir, index_name)
                os.makedirs(output_index_dir, exist_ok=True)
                output_filename = f'{os.path.splitext(filename)[0]}_{index_name}.png'
                output_path = os.path.join(output_index_dir, output_filename)
                cv2.imwrite(output_path, image)
                print(f"Imagem com índice '{index_name}' salva em: {output_path}")

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def get_rotation_angle():
    root = Tk()
    root.withdraw()
    angle = simpledialog.askfloat("Input", "Digite o ângulo de rotação (em graus):")
    root.destroy()
    if angle is None:
        print("Nenhum ângulo de rotação fornecido. Encerrando o programa.")
        sys.exit(0)
    return angle

def expand_polygon(polygon, scale=1.1):
    M = cv2.moments(polygon)
    if M['m00'] == 0:
        return polygon
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    center = np.array([cx, cy])
    expanded_polygon = scale * (polygon - center) + center
    return expanded_polygon.astype(np.int32)

def sort_polygons(polygons, zigzag):
    sorted_polygons = []
    line_count = 0

    while polygons:
        # Encontrar o polígono mais superior à esquerda
        top_left_polygon = find_top_left_polygon(polygons)
        if top_left_polygon is None:
            break
        #Adiciona o primeiro da linha
        line_count += 1
        sorted_polygons.append((line_count, 1, top_left_polygon))

        #remove o pivo
        polygons = [poly for poly in polygons if not np.array_equal(poly, top_left_polygon)]
        
        pivot = top_left_polygon
        same_line_polygons = []
        
        # Determinar os limites verticais do polígono pivô
        x, y, w, h = cv2.boundingRect(pivot)
        y_min = y
        y_max = y + h
        
        # Encontrar polígonos cujos centroides estejam dentro dos limites do pivô
        for poly in polygons:
            centroid = find_centroid(poly)
            if y_min <= centroid[1] <= y_max:
                same_line_polygons.append(poly)
        
        # Encontrar o polígono mais próximo em x do pivô na mesma linha
        closest_polygon = find_closest_polygon(same_line_polygons, pivot)
        
        col_count = 1
        
        # Adicionar os polígonos mais próximos à lista de polígonos ordenados
        while closest_polygon is not None:
            sorted_polygons.append((line_count, col_count, closest_polygon))
            col_count += 1
            # Remover o 
            # polígono processado
            polygons = [poly for poly in polygons if not np.array_equal(poly, closest_polygon)]

            # Definir o polígono mais próximo como novo pivô
            pivot = closest_polygon

            # Reencontrar polígonos na mesma linha com o novo pivô
            same_line_polygons = []
            x, y, w, h = cv2.boundingRect(pivot)
            y_min = y
            y_max = y + h
            for poly in polygons:
                centroid = find_centroid(poly)
                if y_min <= centroid[1] <= y_max:
                    same_line_polygons.append(poly)
            
            # Encontrar novo polígono mais próximo em x
            closest_polygon = find_closest_polygon(same_line_polygons, pivot)

    return sorted_polygons
# y_min, y_max = y_coords.min(), y_coords.max()
    # line_height = (y_max - y_min) / n_lines
    # line_bounds = [(y_min + i * line_height, y_min + (i + 1) * line_height) for i in range(n_lines)]

    # sorted_polygons = []
    # for line_idx, (lower, upper) in enumerate(line_bounds):
    #     line_polygons = [poly for poly in polygons if lower <= find_centroid(poly)[1] < upper]
    #     line_polygons.sort(key=lambda poly: find_centroid(poly)[0])
        
    #     if zigzag and line_idx % 2 == 1:
    #         line_polygons.reverse()
        
    #     for col, poly in enumerate(line_polygons):
    #         sorted_polygons.append((line_idx + 1, col + 1, poly))  # Começa em (1,1)

    # return sorted_polygons


def find_top_left_polygon(polygons):
    if not polygons:
        return None

    top_left_polygon = polygons[0]
    x_tl, y_tl, _, _ = cv2.boundingRect(top_left_polygon)

    for poly in polygons[1:]:
        x, y, _, _ = cv2.boundingRect(poly)
        if x < x_tl or (x == x_tl and y < y_tl):
            top_left_polygon = poly
            x_tl, y_tl = x, y
    
    return top_left_polygon

def find_centroid(poly):
    M = cv2.moments(poly)
    if M['m00'] == 0:
        return np.array([0, 0])
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return np.array([cx, cy])

def find_closest_polygon(polygons, pivot):
    if not polygons:
        return None

    pivot_centroid = find_centroid(pivot)
    closest_polygon = polygons[0]
    closest_distance = abs(find_centroid(closest_polygon)[0] - pivot_centroid[0])

    for poly in polygons[1:]:
        centroid = find_centroid(poly)
        distance = abs(centroid[0] - pivot_centroid[0])
        if distance < closest_distance:
            closest_distance = distance
            closest_polygon = poly
    
    return closest_polygon
    

select_aoi(clone)

if len(aoi_points) < 3:
    raise ValueError("Área de interesse precisa ter ao menos 3 pontos")

aoi = np.array(aoi_points, np.int32)
pts = aoi.reshape((-1, 1, 2))
mask = np.zeros(clone.shape[:2], dtype=np.uint8)
cv2.fillPoly(mask, [pts], 255)
cropped_image = cv2.bitwise_and(clone, clone, mask=mask)

x, y, w, h = cv2.boundingRect(pts)
cropped_image = cropped_image[y:y+h, x:x+w]

angle = get_rotation_angle()
rotated_image = rotate_image(cropped_image, angle)
show_cropped_image(rotated_image)
cv2.imwrite("./rotated.png", rotated_image)

model_path = "best.pt"
model = YOLO(model_path)

results = model.predict(source=rotated_image, show_labels=True, save=True)

os.makedirs(output_dir, exist_ok=True)

cont = 0
polygons = [np.array(poly, dtype=np.int32) for result in results for poly in result.masks.xy if len(poly) > 0]
zigzag = simpledialog.askstring("Input", "Deseja ordenar em zigue-zague? (sim/não)").lower() == 'sim'
sorted_polygons = sort_polygons(polygons, zigzag)

annotated_image = rotated_image.copy()
data = []

y_min, y_max = np.min([find_centroid(poly)[1] for poly in polygons]), np.max([find_centroid(poly)[1] for poly in polygons])
  
for row, col, poly in sorted_polygons:
    mask = np.zeros_like(rotated_image)
    cv2.fillPoly(mask, [expand_polygon(poly, scale=1.1)], (255,) * mask.shape[2])
    imagem_mascarada = cv2.bitwise_and(rotated_image, mask)

    x, y, w, h = cv2.boundingRect(expand_polygon(poly, scale=1.1))
    recorte = imagem_mascarada[y:y+h, x:x+w]

    cv2.imwrite(f'{output_dir}/parcela-{row}-{col}.png', recorte)
    cont += 1
    print(f'Salvando parcela {cont} na posição (linha {row}, coluna {col})')
    
    centroid = find_centroid(poly)
    cv2.putText(annotated_image, f'{row},{col}', (centroid[0], centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
    
    data.append([row, col, f'parcela-{row}-{col}.png'])

cv2.imwrite("./rotated_annotated.png", annotated_image)

with open('data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Linha', 'Coluna', 'Arquivo'])
    writer.writerows(data)

#process_parcelas(output_dir, output_indices_dir)


#
#
# Carregar planilha de campo e correlacionar com parcelas extraídos
#
#

#
#
# Implementar opção de nomeação em zig zague
#
#