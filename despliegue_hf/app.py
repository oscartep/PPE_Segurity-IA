import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime, timezone, timedelta
import os
import requests
import json

# 1. Cargar el modelo entrenado
model = YOLO('best.pt')

# Configura aquí tu URL de producción o test de n8n
N8N_WEBHOOK_URL = "https://baltae.app.n8n.cloud/webhook/auditoria-epp"

# Definir la zona horaria de Colombia (UTC-5) para usarla en todo el código
colombia_tz = timezone(timedelta(hours=-5))

# Función para renderizar la tabla histórica en formato Markdown
def generar_tabla_markdown(historial):
    if not historial:
        return "| Hora | Persona # | ¿Cumple? | Detalle / EPP Faltante |\n| --- | --- | --- | --- |\n| *Sin registros* | *Espere análisis* | *---* | *---* |"
    
    tabla = "| Hora | Persona # | ¿Cumple? | Detalle / EPP Faltante |\n| --- | --- | --- | --- |\n"
    for reg in historial:
        tabla += f"| {reg['hora']} | **{reg['persona']}** | {reg['cumple']} | {reg['detalle']} |\n"
    return tabla

# 1. BOTÓN ANALIZAR: Procesa la foto actual y acumula en la tabla
def analizar_foto(imagen, historial, contador_personas):
    if imagen is None:
        return None, historial, contador_personas, generar_tabla_markdown(historial)
        
    img_bgr = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR).copy()
    resultados = model.predict(img_bgr, imgsz=640)
    boxes = resultados[0].boxes
    clases = model.names
    
    id_humano = [k for k, v in clases.items() if v.lower() in ['human', 'humano', 'person', 'persona', 'man']]
    id_casco = [k for k, v in clases.items() if v.lower() in ['helmet', 'casco', 'hard-hat', 'hat']]
    id_chaleco = [k for k, v in clases.items() if v.lower() in ['vest', 'chaleco', 'safety-vest']]
    id_botas = [k for k, v in clases.items() if v.lower() in ['boots', 'botas', 'shoes', 'boot']]
    
    detected_humans = []
    detected_helmets = []
    detected_vests = []
    detected_boots = []
    
    for box in boxes:
        coords = box.xyxy[0].cpu().numpy().tolist()
        cls_id = int(box.cls.item())
        if cls_id in id_humano: detected_humans.append(coords)
        elif cls_id in id_casco: detected_helmets.append(coords)
        elif cls_id in id_chaleco: detected_vests.append(coords)
        elif cls_id in id_botas: detected_boots.append(coords)
            
    def item_en_persona(box_item, box_persona):
        ix1, iy1 = max(box_item[0], box_persona[0]), max(box_item[1], box_persona[1])
        ix2, iy2 = min(box_item[2], box_persona[2]), min(box_item[3], box_persona[3])
        if ix2 > ix1 and iy2 > iy1:
            area_interseccion = (ix2 - ix1) * (iy2 - iy1)
            area_item = (box_item[2] - box_item[0]) * (box_item[3] - box_item[1])
            if area_item == 0: return False
            return (area_interseccion / area_item) > 0.55
        return False

    # HORA CON ZONA HORARIA DE COLOMBIA
    hora_actual = datetime.now(colombia_tz).strftime("%H:%M:%S")
    nuevo_contador = contador_personas
    
    if detected_humans:
        for p_box in detected_humans:
            tiene_c = any(item_en_persona(c_box, p_box) for c_box in detected_helmets)
            tiene_ch = any(item_en_persona(ch_box, p_box) for ch_box in detected_vests)
            tiene_b = any(item_en_persona(b_box, p_box) for b_box in detected_boots)
            
            faltantes = []
            if not tiene_c: faltantes.append("Casco 🪖")
            if not tiene_ch: faltantes.append("Chaleco 🦺")
            if not tiene_b: faltantes.append("Botas 🥾")
            
            px1, py1, px2, py2 = map(int, p_box)
            nombre_persona = f"Persona {nuevo_contador}"
            
            if faltantes:
                cumple_status = "❌ No cumple"
                detalle_status = f"Le falta: {', '.join(faltantes)}"
                color = (0, 0, 255)
                if not tiene_c:
                    alto_cabeza = int((py2 - py1) * 0.23)
                    cv2.rectangle(img_bgr, (px1, py1), (px2, py1 + alto_cabeza), (0, 0, 255), 2)
                    cv2.putText(img_bgr, "ALERTA: SIN CASCO", (px1, py1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            else:
                cumple_status = "✅ Cumple"
                detalle_status = "Equipo Completo"
                color = (0, 255, 0)
            
            cv2.rectangle(img_bgr, (px1, py1), (px2, py2), color, 3)
            cv2.putText(img_bgr, nombre_persona, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            historial.append({
                "hora": hora_actual,
                "persona": nombre_persona,
                "cumple": cumple_status,
                "detalle": detalle_status
            })
            nuevo_contador += 1

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tabla_actualizada = generar_tabla_markdown(historial)
    return img_rgb, historial, nuevo_contador, tabla_actualizada

# 2. BOTÓN SIGUIENTE: Solo limpia las imágenes del visor actual para permitir una nueva carga sin tocar la tabla
def siguiente_persona():
    return None, None

# 3. BOTÓN LIMPIAR: Borra la imagen si el usuario se equivocó al subirla antes de procesar
def limpiar_pantalla():
    return None, None

# 4. BOTÓN TERMINAR: Exporta un archivo de datos .txt descargable, envía reporte a n8n y resetea todo de inmediato
def terminar_y_descargar(historial, correo_destino):
    nombre_archivo = "reporte_auditoria_epp.txt"
    
    if not correo_destino or "@" not in correo_destino:
        return gr.skip(), gr.skip(), None, historial, gr.skip(), generar_tabla_markdown(historial), "⚠️ Ingrese un correo electrónico válido."
        
    ahora_col = datetime.now(colombia_tz)
    
    contenido = "=== REPORTE DE AUDITORÍA DE SEGURIDAD HSE ===\n"
    contenido += f"Fecha de exportación: {ahora_col.strftime('%Y-%m-%d %H:%M:%S')}\n"
    contenido += "-"*50 + "\n"
    
    if not historial:
        contenido += "No se registraron datos en esta sesión.\n"
    else:
        for reg in historial:
            contenido += f"Hora: {reg['hora']} | {reg['persona']} | Estado: {reg['cumple']} | Detalle: {reg['detalle']}\n"
            
    with open(nombre_archivo, "w", encoding="utf-8") as f:
        f.write(contenido)
    
    if historial:
        asunto_dinamico = f"🚨 CONTROL HSE: Reporte Técnico de Auditoría de EPP ({ahora_col.strftime('%Y-%m-%d %H:%M')})"
        
        html_cuerpo = f"""
        <div style="background-color: #f8fafc; padding: 40px 20px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
            <div style="max-width: 650px; margin: 0 auto; background-color: #ffffff; border-radius: 16px; overflow: hidden; box-shadow: 0 10px 25px rgba(30, 41, 59, 0.05); border: 1px solid #e2e8f0;">
                <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 35px 30px; text-align: center; border-bottom: 4px solid #3b82f6;">
                    <div style="font-size: 45px; margin-bottom: 12px;">🚧</div>
                    <h1 style="color: #ffffff; margin: 0; font-size: 26px; font-weight: 700; letter-spacing: -0.5px;">REPORTE DE AUDITORÍA HSE</h1>
                    <p style="color: #94a3b8; margin: 6px 0 0 0; font-size: 14px; font-weight: 500;">Sistema de Visión Artificial • Control de EPP</p>
                </div>
                <div style="padding: 35px 30px;">
                    <p style="font-size: 16px; color: #1e293b; font-weight: 600; margin-top: 0;">Estimado Supervisor de Seguridad,</p>
                    <p style="font-size: 14px; color: #475569; line-height: 1.6; margin-bottom: 25px;">
                        Se ha completado la verificación visual automatizada mediante el modelo **YOLOv8**. A continuación se anexa el estado de cumplimiento del personal auditado:
                    </p>
                    <table style="width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 14px; text-align: left;">
                        <thead>
                            <tr style="background-color: #f1f5f9; border-bottom: 2px solid #cbd5e1;">
                                <th style="padding: 12px 10px; color: #1e293b; font-weight: 700;">Hora</th>
                                <th style="padding: 12px 10px; color: #1e293b; font-weight: 700;">Trabajador</th>
                                <th style="padding: 12px 10px; color: #1e293b; font-weight: 700; text-align: center;">Estado</th>
                                <th style="padding: 12px 10px; color: #1e293b; font-weight: 700;">Detalle / EPP Faltante</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        for reg in historial:
            is_cumple = "✅" in reg['cumple']
            bg_badge = "#dcfce7" if is_cumple else "#fee2e2"
            text_badge = "#166534" if is_cumple else "#991b1b"
            border_badge = "#bbf7d0" if is_cumple else "#fca5a5"
            
            html_cuerpo += f"""
                            <tr style="border-bottom: 1px solid #f1f5f9;">
                                <td style="padding: 12px 10px; color: #64748b;">{reg['hora']}</td>
                                <td style="padding: 12px 10px; font-weight: 600; color: #334155;">{reg['persona']}</td>
                                <td style="padding: 12px 10px; text-align: center;">
                                    <span style="background-color: {bg_badge}; color: {text_badge}; border: 1px solid {border_badge}; padding: 4px 10px; border-radius: 12px; font-weight: 700; font-size: 11px; display: inline-block;">
                                        {reg['cumple']}
                                    </span>
                                </td>
                                <td style="padding: 12px 10px; color: #475569;">{reg['detalle']}</td>
                            </tr>
            """
        html_cuerpo += f"""
                        </tbody>
                    </table>
                    <div style="background-color: #fbf7f7; padding: 15px; border-radius: 8px; border-left: 4px solid #ef4444; margin-top: 25px;">
                        <p style="margin: 0; font-size: 13px; color: #991b1b; line-height: 1.5;">
                            <strong>⚠️ Plan de Acción Obligatorio:</strong> Todo operario marcado en estado de infracción (No cumple) debe regularizar su equipo de forma obligatoria antes de ingresar a zonas de riesgo.
                        </p>
                    </div>
                    <div style="margin-top: 35px; padding-top: 15px; border-top: 1px solid #e2e8f0; text-align: center;">
                        <p style="font-size: 11px; color: #94a3b8; margin: 0;">Reporte automatizado generado el {ahora_col.strftime('%Y-%m-%d a las %H:%M:%S')}.</p>
                    </div>
                </div>
            </div>
        </div>
        """
        
        payload = {"correo_destino": correo_destino, "asunto": asunto_dinamico, "html_email": html_cuerpo}
        try:
            requests.post(N8N_WEBHOOK_URL, data=json.dumps(payload), headers={"Content-Type": "application/json"}, timeout=10)
            status_envio = f"🚀 Reporte despachado con éxito a {correo_destino} y n8n."
        except Exception as e:
            status_envio = f"⚠️ Guardado localmente, pero falló la conexión con n8n: {str(e)}"
    else:
        status_envio = "ℹ️ Sistema reiniciado. No se transmitieron datos vacíos."

    tabla_vacia = generar_tabla_markdown([])
    return None, None, nombre_archivo, [], 1, tabla_vacia, status_envio


# --- INTERFAZ GRÁFICA (Gradio Blocks) ---
with gr.Blocks(theme=gr.themes.Soft()) as interfaz:
    gr.Markdown("# 🚧 Sistema de Auditoría HSE Acumulativo en Tiempo Real")
    gr.Markdown("Flujo de control secuencial para el análisis y registro normativo de EPP en campo.")
    
    memoria_historial = gr.State([])
    contador_global_personas = gr.State(1)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="Cargar Fotografía de Campo")
            
            with gr.Row():
                btn_analizar = gr.Button("🔍 1. Analizar Foto", variant="primary")
                btn_siguiente = gr.Button("📸 2. Siguiente Persona", variant="secondary")
            with gr.Row():
                btn_limpiar = gr.Button("🧹 3. Limpiar Foto", variant="secondary")
                btn_terminar = gr.Button("🏁 4. Terminar y Exportar", variant="stop")
        
        with gr.Column():
            output_image = gr.Image(type="numpy", label="Monitor de Infracciones Visuales")
            file_output = gr.File(label="📥 Descargar Base de Datos Registrada (.TXT)", interactive=False)
            
            txt_correo = gr.Textbox(
                label="📧 Correo Destinatario (Supervisor HSE)", 
                placeholder="supervisor@empresa.com"
            )
            txt_estado_webhook = gr.Textbox(label="📡 Estado del Envío Automático", interactive=False)
        
    gr.Markdown("## 📊 Historial Log de Auditoría")
    output_table = gr.Markdown(value=generar_tabla_markdown([]))
    
    # --- ENLACE DE ACCIONES ---
    btn_analizar.click(
        fn=analizar_foto,
        inputs=[input_image, memoria_historial, contador_global_personas],
        outputs=[output_image, memoria_historial, contador_global_personas, output_table]
    )
    
    btn_siguiente.click(
        fn=siguiente_persona,
        inputs=[],
        outputs=[input_image, output_image]
    )
    
    btn_limpiar.click(
        fn=limpiar_pantalla,
        inputs=[],
        outputs=[input_image, output_image]
    )
    
    btn_terminar.click(
        fn=terminar_y_descargar,
        inputs=[memoria_historial, txt_correo],
        outputs=[input_image, output_image, file_output, memoria_historial, contador_global_personas, output_table, txt_estado_webhook]
    )

interfaz.launch()
