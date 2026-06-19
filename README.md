# PPE_Segurity-IA

Sistema de visión artificial con YOLOv8 para la auditoría automática de Equipos de Protección Personal (EPP) en entornos industriales, con integración de reportes HSE en tiempo real vía n8n.

## Base de Datos Utilizada (Obligatorio)
* **Nombre del Dataset:** PPE Detection
* **Enlace de acceso:** [Acceder al Dataset en Roboflow](https://app.roboflow.com/danna-marcela-balta-espinel/ppe-detection-vpw6m-gz52b/train)
* **Descripción:** El dataset está compuesto por imágenes de entornos industriales, anotadas con cajas delimitadoras. Contiene las clases necesarias para detectar personal operativo y validar el uso de: cascos, chalecos y botas.

## Despliegue en Hugging Face Spaces
👉 [Visitar Aplicación Interactiva en Hugging Face](https://huggingface.co/spaces/dbaltae/Proyecto_PDI)

## Entregables Obligatorios
Con base en los requerimientos del proyecto, el repositorio está estructurado de la siguiente manera:

* **`/entrenamiento`**: Todo el código fuente del entrenamiento del modelo.
* **`/despliegue_hf`**: El código de configuración y despliegue del espacio en Hugging Face Spaces.
* **`/executorch_docs`**: La documentación del proceso de exportación del modelo (utilizando ExecuTorch) para su posterior integración en Hugging Face Spaces.
