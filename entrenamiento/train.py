# Instalamos Roboflow y la librería de YOLOv8
!pip install roboflow ultralytics

from roboflow import Roboflow

# NOTA: Reemplaza TU_CLAVE_AQUI con las letras y números que ocultaste en tu imagen
rf = Roboflow(api_key="35AvfIqQdQKskTUCpFiJ")
project = rf.workspace("danna-marcela-balta-espinel").project("ppe-detection-vpw6m-gz52b")
version = project.version(1)
dataset = version.download("yolov8")


from ultralytics import YOLO

# 1. Cargar un modelo YOLOv8 pre-entrenado (recomendado: tamaño pequeño 's' para que entrene rápido)
model = YOLO('yolov8s.pt')

# 2. Entrenar el modelo
# Le pasamos el archivo data.yaml que descargó Roboflow, definimos 25 épocas y el tamaño de imagen
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=25,
    imgsz=640,
    plots=True # Esto generará automáticamente las gráficas de rendimiento que pide tu profesor
)
