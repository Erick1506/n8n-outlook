from flask import Flask, request, jsonify
import base64
import io
import fitz  # PyMuPDF
import docx
import zipfile
import pandas as pd
from PIL import Image
import pytesseract
import re
import numpy as np
from pytesseract import image_to_data, Output
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# Configurar Tesseract en Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Inicializar modelo NER
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

app = Flask(__name__)

# ----------- OCR estructurado para formularios -----------
def ocr_preservar_estructura(img):
    try:
        ocr_data = image_to_data(img, lang='spa', output_type=Output.DATAFRAME)
        ocr_data = ocr_data.dropna(subset=['text'])
        ocr_data = ocr_data[ocr_data['text'].str.strip() != '']
        if ocr_data.empty:
            return "📷 Imagen procesada, pero no se detectó texto."

        lineas = ocr_data.groupby(['block_num', 'par_num', 'line_num'])['text'].apply(lambda x: ' '.join(x)).tolist()
        return "\n".join(lineas).strip()
    except Exception as e:
        return f"⚠️ OCR estructurado falló: {str(e)}"

# ----------- Clasificación de contexto -----------
def clasificar_contexto(texto):
    texto = texto.lower()
    if "firma" in texto or "sello" in texto:
        return "🖋️ Posible firma o sello detectado."
    elif "certifica" in texto or "autorizo" in texto:
        return "📄 Declaración o autorización."
    elif "fecha" in texto and "nombre" in texto:
        return "🗂️ Documento formal."
    elif len(texto.split()) < 5:
        return "📷 Imagen con poco texto."
    return "📌 Contexto general."

# ----------- Extracción de entidades -----------
def extraer_entidades(texto):
    if not texto.strip():
        return []
    try:
        return ner_pipeline(texto)
    except Exception as e:
        return [f"⚠️ Error NER: {str(e)}"]

# ----------- Conversión a pares clave:valor -----------
def extraer_campos_estructura(texto):
    campos = []
    for linea in texto.splitlines():
        if ':' in linea:
            partes = linea.split(':', 1)
            clave = partes[0].strip()
            valor = partes[1].strip()
            if clave and valor:
                campos.append({"campo": clave, "valor": valor})
    return campos

# ----------- Lectores de archivos -----------
def leer_pdf(file_bytes):
    texto = ""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for page in doc:
        texto += page.get_text()
        img_list = page.get_images(full=True)
        for img in img_list:
            xref = img[0]
            base_img = doc.extract_image(xref)
            img_bytes = base_img["image"]
            img = Image.open(io.BytesIO(img_bytes))
            ocr_text = ocr_preservar_estructura(img)
            if ocr_text.strip():
                texto += f"\n📷 OCR imagen: {ocr_text}"
    return texto.strip()

def leer_docx(file_bytes):
    texto = ""
    doc = docx.Document(file_bytes)
    texto += "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    file_bytes.seek(0)
    try:
        with zipfile.ZipFile(file_bytes) as docx_zip:
            for file in docx_zip.namelist():
                if file.startswith("word/media/") and file.lower().endswith((".png", ".jpg", ".jpeg")):
                    with docx_zip.open(file) as image_file:
                        image = Image.open(image_file)
                        ocr_text = ocr_preservar_estructura(image)
                        if ocr_text.strip():
                            texto += f"\n📷 OCR imagen: {ocr_text.strip()}"
    except Exception as e:
        texto += f"\n⚠️ Error OCR docx: {str(e)}"
    return texto.strip()

def leer_excel(file_bytes):
    texto = ""
    xl = pd.ExcelFile(file_bytes)
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        texto += f"[Hoja: {sheet}]\n"
        texto += df.to_string(index=False)
        texto += "\n\n"
    return texto.strip()

def leer_txt(file_bytes):
    return file_bytes.read().decode("utf-8", errors="ignore")

def leer_imagen(file_bytes):
    try:
        img = Image.open(file_bytes)
        texto = ocr_preservar_estructura(img)
        contexto = clasificar_contexto(texto)
        return f"{texto}\n\n🔎 {contexto}"
    except Exception as e:
        return f"⚠️ Error imagen: {str(e)}"

# ----------- Extraer texto de HTML (firmas digitales) -----------
def extraer_texto_de_imagenes_embebidas(html):
    resultados = []
    matches = re.findall(r'<img[^>]+src="data:image/[^;]+;base64,([^\"]+)', html)
    for i, base64_img in enumerate(matches):
        try:
            img_data = base64.b64decode(base64_img)
            img = Image.open(io.BytesIO(img_data))
            texto = ocr_preservar_estructura(img)
            if texto.strip():
                resultados.append(f"📎 Firma embebida {i+1}:\n{texto.strip()}")
        except Exception as e:
            resultados.append(f"⚠️ Imagen embebida {i+1} error: {str(e)}")
    return "\n".join(resultados)

# ----------- Ruta principal -----------
@app.route('/procesar-todo', methods=['POST'])
def procesar_todo():
    data = request.get_json()
    archivos = data.get('archivos', [])
    html = data.get('html', '')
    resultados = []

    for archivo in archivos:
        nombre = archivo.get('nombre', 'documento.bin')
        base64_content = archivo.get('contenido')
        if not base64_content:
            resultados.append({'nombre': nombre, 'error': '❌ Sin contenido base64'})
            continue
        try:
            file_bytes = io.BytesIO(base64.b64decode(base64_content))
            file_bytes.seek(0)
            ext = nombre.split('.')[-1].lower()

            if ext == 'pdf':
                texto = leer_pdf(file_bytes)
            elif ext == 'docx':
                texto = leer_docx(file_bytes)
            elif ext in ['xlsx', 'xls']:
                texto = leer_excel(file_bytes)
            elif ext == 'txt':
                texto = leer_txt(file_bytes)
            elif ext in ['jpg', 'jpeg', 'png']:
                texto = leer_imagen(file_bytes)
            else:
                texto = "❌ Tipo de archivo no soportado"

            entidades = extraer_entidades(texto)
            campos_detectados = extraer_campos_estructura(texto)

            resultados.append({
                "nombre": nombre,
                "tipo": ext,
                "texto_extraido": texto,
                "entidades": entidades,
                "campos": campos_detectados
            })

        except Exception as e:
            resultados.append({"nombre": nombre, "error": f"⚠️ Error: {str(e)}"})

    if html:
        html_texto = extraer_texto_de_imagenes_embebidas(html)
        if html_texto:
            entidades = extraer_entidades(html_texto)
            campos = extraer_campos_estructura(html_texto)
            resultados.append({
                "nombre": "firmas_html",
                "tipo": "html",
                "texto_extraido": html_texto,
                "entidades": entidades,
                "campos": campos
            })

    def convertir_tipos_serializables(obj):
        if isinstance(obj, dict):
            return {k: convertir_tipos_serializables(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convertir_tipos_serializables(i) for i in obj]
        elif isinstance(obj, np.float32):
            return float(obj)
        else:
            return obj

    return jsonify(convertir_tipos_serializables(resultados))

# ----------- Iniciar servidor -----------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
