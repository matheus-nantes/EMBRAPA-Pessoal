import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import shutil
from screeninfo import get_monitors
from tkinter import Tk, filedialog, simpledialog  # Importa o módulo filedialog do tkinter
from index_calculation import IndexCalculation
from PIL import Image
from PIL import ImageFile

# Configurações do Pillow para lidar com imagens grandes
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

def select_file():
    root = Tk()
    root.withdraw()  # Esconde a janela principal do tkinter

    # Abre a janela de seleção de arquivo
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

# Carrega a imagem usando Pillow
image_pil = Image.open(file_path)
image = np.array(image_pil)
clone = image.copy()

# Redimensiona a imagem para evitar problemas de memória
scale_percent = 10  # Reduz a imagem para 10% do tamanho original
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
resized_clone = cv2.resize(clone, dim, interpolation=cv2.INTER_AREA)

aoi_points = []

# Função para desfazer o último ponto
def undo_last_point():
    if aoi_points:
        aoi_points.pop()

def select_aoi(image):
    fig, ax = plt.subplots()
    
    monitor = get_monitors()[0]
    fig.set_size_inches((monitor.width*0.9) / fig.dpi, (monitor.height*0.7) / fig.dpi)
    
    ax.imshow(image)
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
            ax.imshow(image)
            ax.set_title("Clique para selecionar pontos. Pressione 'u' para desfazer o último ponto. Pressione 'f' para finalizar.")
            if aoi_points:
                ax.plot(*zip(*aoi_points), 'ro-')
                if len(aoi_points) > 1:
                    ax.plot([aoi_points[-1][0], aoi_points[0][0]], [aoi_points[-1][1], aoi_points[0][1]], 'b--')
            fig.canvas.draw()
        elif event.key == 'f':
            if len(aoi_points) > 2:
                aoi_points.append(aoi_points[0])  # Fecha o polígono
                fig.canvas.mpl_disconnect(cid)
                fig.canvas.mpl_disconnect(kid)
                plt.close(fig)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    kid = fig.canvas.mpl_connect('key_press_event', onkey)
    plt.show()

def show_cropped_image(image):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Imagem recortada. Pressione 'p' para prosseguir com a predição. Pressione 'c' para cancelar.")

    def onkey(event):
        if event.key == 'p':
            fig.canvas.mpl_disconnect(kid)
            plt.close()
        elif event.key == 'c':
            print("Processo encerrado espontaneamente")
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
                cv2.imwrite(output_path, result)
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

select_aoi(resized_image)

if len(aoi_points) < 3:
    raise ValueError("Área de interesse precisa ter ao menos 3 pontos")

aoi = np.array(aoi_points, np.int32)
pts = aoi.reshape((-1, 1, 2))
mask = np.zeros(resized_clone.shape[:2], dtype=np.uint8)
cv2.fillPoly(mask, [pts], 255)
masked_image = cv2.bitwise_and(resized_clone, resized_clone, mask=mask)
x, y, w, h = cv2.boundingRect(pts)
cropped_image = masked_image[y:y+h, x:x+w]
image = cropped_image
cv2.imwrite('img_cropped.png', image)

show_cropped_image(image)

angle = get_rotation_angle()
rotated_image = rotate_image(image, angle)
cv2.imwrite('img_rotated.png', rotated_image)

show_cropped_image(rotated_image)


if rotated_image.shape[2] == 4:
    rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_RGBA2RGB)
    
# Carrega o modelo YOLO
model = YOLO("best.pt")

results = model.predict(source=rotated_image, save=True, show_labels=False)

output_dir = './parcelas'
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

def expand_polygon(polygon, scale=1.1):
    # Calcula o centroide do polígono
    M = cv2.moments(polygon)
    if M['m00'] == 0:  # Evita divisão por zero
        return polygon
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    center = np.array([cx, cy])

    # Expande o polígono em relação ao centroide
    expanded_polygon = scale * (polygon - center) + center
    return expanded_polygon.astype(np.int32)

cont = 0
for result in results:
    for poly in result.masks.xy:
        poly = np.array(poly, dtype=np.int32)
        expanded_poly = expand_polygon(poly, scale=1.1)
        
        mask = np.zeros_like(rotated_image)

        # Certifique-se de que a máscara tenha 3 canais
        if len(mask.shape) == 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        cv2.fillPoly(mask, [expanded_poly], (255, 255, 255))
        parcela = cv2.bitwise_and(rotated_image, mask)
        cv2.imwrite(f"./parcelas/parcela_{cont}.png", parcela)
        cont += 1

process_parcelas('./parcelas', './indices')
