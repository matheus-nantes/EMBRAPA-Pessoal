import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import shutil
from screeninfo import get_monitors
from tkinter import Tk, filedialog  # Importa o módulo filedialog do tkinter
from index_calculation import IndexCalculation 



def select_file():
    root = Tk()
    root.withdraw()  # Esconde a janela principal do tkinter

    # Abre a janela de seleção de arquivo
    file_path = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")])

    if not file_path:
        print("Nenhum arquivo selecionado. Encerrando o programa.")
        sys.exit(0)

    return file_path

# Caminho do ortomosaico TIFF
#file_path = "M:\\EMBRAPA\\ORTOMOSAICOS\\SEDEB3B4_0-5CM_19-03-12_1-4\\3_dsm_ortho\\2_mosaic\\SEDEB3B4_0-5CM_19-03-12_1-4_transparent_mosaic_group1.tif"
#file_path = "M:\\EMBRAPA\\ORTOMOSAICOS\\Sede_B3B4_1,5_190312\\3_dsm_ortho\\2_mosaic\\Sede_B3B4_1,5_190312_mosaic_group1.tif"
file_path = select_file()
print(f"Arquivo selecionado: {file_path}")


# Verifica se a pasta "runs/segment/predict" existe, cria se não existir e limpa se existir
runs_dir = './runs'
if os.path.exists(runs_dir):
    shutil.rmtree(runs_dir)

# Carrega a imagem
image = cv2.imread(file_path)
clone = image.copy()
aoi_points = []

# Função para desfazer o último ponto
def undo_last_point():
    if aoi_points:
        aoi_points.pop()

# Função para selecionar a AOI
def select_aoi(image):
    fig, ax = plt.subplots()
    
    # Obtém a resolução da tela e ajusta o tamanho da janela
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
                aoi_points.append(aoi_points[0])  # Fecha o polígono
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
    # Definir funções de cálculo para diferentes índices
    def bi(red, green, blue):
        return (2 * green - red - blue) / (2 * green + red + blue)

    def gli(red, green, blue):
        return (2 * red - green - blue) / (2 * red + green + blue)

    def ngrdi(red, green, nir):
        return (green - red) / (green + red - nir)

    # Dicionário de funções de cálculo de índice
    index_functions = {
        "BI": bi,
        "GLI": gli,
        "NGRDI": lambda r, g, nir: ngrdi(r, g, nir)
        # Adicione mais funções de índice conforme necessário
    }

    # Calcular os índices selecionados
    index_results = {}
    for index_name, func in index_functions.items():
        if index_name == "BI" or index_name == "GLI":
            index_results[index_name] = func(red, green, blue)
        elif index_name == "NGRDI" and nir is not None:
            index_results[index_name] = func(red, green, nir)

    return index_results


# Função para processar cada imagem na pasta parcelas
def process_parcelas(input_dir, output_dir):
    # Listar todos os arquivos na pasta parcelas
    parcelas_files = os.listdir(input_dir)

    # Iterar sobre cada imagem na pasta parcelas
    for filename in parcelas_files:
        if filename.endswith(".png"):  # Apenas processa arquivos PNG (ajuste conforme necessário)
            # Caminho completo para a imagem de entrada
            input_path = os.path.join(input_dir, filename)

            # Carregar a imagem
            image = cv2.imread(input_path)

            # Dividir em canais RGB
            blue, green, red = cv2.split(image)

            # # Salvar os canais Red, Green e Blue
            # cv2.imwrite(os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_red.png"), red)
            # cv2.imwrite(os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_green.png"), green)
            # cv2.imwrite(os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_blue.png"), blue)

            index_calculator = IndexCalculation(red=red, green=green, blue=blue)

            hue_index = index_calculator.hue()
            cv2.imwrite("./hue.jpg", hue_index)

            # Calcular os índices desejados
            indices = calculate_indices(red, green, blue)

            # Salvar as imagens com os índices calculados na pasta parcelas_indice
            for index_name, result in indices.items():
                # Criar pasta parcelas_indice se não existir
                output_index_dir = os.path.join(output_dir, index_name)
                os.makedirs(output_index_dir, exist_ok=True)

                # Salvar imagem com o nome indicando o índice calculado
                output_filename = f'{os.path.splitext(filename)[0]}_{index_name}.png'
                output_path = os.path.join(output_index_dir, output_filename)
                cv2.imwrite(output_path, image)
                print(f"Imagem com índice '{index_name}' salva em: {output_path}")


# Chama a função para selecionar a AOI
select_aoi(clone)

if len(aoi_points) < 3:
    raise ValueError("Área de interesse precisa ter ao menos 3 pontos")

# Converte a lista de pontos para um array numpy
aoi = np.array(aoi_points, np.int32)
pts = aoi.reshape((-1, 1, 2))
mask = np.zeros(clone.shape[:2], dtype=np.uint8)
cv2.fillPoly(mask, [pts], 255)
masked_image = cv2.bitwise_and(clone, clone, mask=mask)
x, y, w, h = cv2.boundingRect(pts)
cropped_image = masked_image[y:y+h, x:x+w]
image = cropped_image
cv2.imwrite('img_cropped.png', image)

show_cropped_image(image)

# Carrega o modelo YOLO
model = YOLO("best.pt")

# Realiza a predição na imagem recortada
results = model.predict(source=image, save=True, show_labels=False)

# Verifica se a pasta "parcelas" existe, cria se não existir e limpa se existir
output_dir = './parcelas'
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# Salva as parcelas detectadas
cont = 0
for result in results:
    for poly in result.masks.xy:
        poly = np.array(poly, dtype=np.int32)
        mask = np.zeros_like(image)

        cv2.fillPoly(mask, [poly], (255,) * mask.shape[2])
        imagem_mascarada = cv2.bitwise_and(image, mask)

        x, y, w, h = cv2.boundingRect(poly)
        recorte = imagem_mascarada[y:y+h, x:x+w]
        cv2.imwrite(f'{output_dir}/parcela-{cont}.png', recorte)
        cont += 1
        print(f'Salvando parcela {cont}')


output_indices_dir = './parcelas_indice'
if os.path.exists(output_indices_dir):
        shutil.rmtree(output_indices_dir)
os.makedirs(output_indices_dir)
process_parcelas(output_dir, output_indices_dir)