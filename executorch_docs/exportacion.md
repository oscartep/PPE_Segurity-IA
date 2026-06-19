# Documentación de Exportación de Modelo a ExecuTorch

## Introducción
El objetivo de esta documentación es establecer el flujo de trabajo para exportar el modelo YOLOv8 entrenado (PyTorch) hacia entornos Edge (dispositivos móviles, cámaras inteligentes o hardware de bajos recursos) utilizando **ExecuTorch**.

## Proceso Teórico de Exportación
Para llevar nuestro detector de EPP a hardware de bajo consumo, el flujo consta de los siguientes pasos mediante la API de PyTorch:

1. **Captura del Grafo (Exportación a ATen):** Se utiliza `torch.export.export()` para trazar el modelo de PyTorch (`best.pt`) a fin de capturar su grafo computacional estático, garantizando que no existan dependencias de Python en tiempo de ejecución.
2. **Cuantización (Opcional):** Reducción de la precisión de los pesos (por ejemplo, de FP32 a INT8) para optimizar el tamaño del modelo y acelerar la velocidad de inferencia en dispositivos de recursos limitados.
3. **Compilación (Backend Delegation):** El modelo capturado se compila y optimiza para un backend delegado como XNNPACK o un procesador neuronal (NPU) específico del hardware.
4. **Generación del Binario:** Se genera el archivo final con extensión `.pte`, el cual es consumido directamente por el *runtime* nativo en C++ de ExecuTorch en el dispositivo final.

## Implementación en este Proyecto
Para nuestro despliegue actual en la nube (Hugging Face Spaces), la inferencia se ejecuta sobre un servidor que soporta nativamente el framework completo. Por lo tanto, el sistema consume directamente el modelo en formato PyTorch (`best.pt`) a través de la librería `ultralytics`. La vía de ExecuTorch queda aquí documentada como el pipeline oficial de escalabilidad para futuras instalaciones locales directamente en el hardware de cámaras CCTV sin conexión a internet.
