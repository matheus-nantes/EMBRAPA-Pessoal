import os
import sys

# Substitua 'M:\\Program Files' pelo diretório onde o GDAL está instalado
gdal_path = 'M:\\Program Files'
os.environ['PATH'] = os.path.join(gdal_path, 'bin') + ';' + os.environ['PATH']

# Adicione o diretório dos dados do GDAL ao caminho de busca do Python
os.environ['GDAL_DATA'] = os.path.join(gdal_path, 'data')
os.environ['GDAL_DRIVER_PATH'] = os.path.join(gdal_path, 'gdalplugins')
os.environ['PROJ_LIB'] = os.path.join(gdal_path, 'projlib')

# Agora, importe o módulo gdal
from osgeo import gdal
