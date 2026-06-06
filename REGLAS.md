# Reglas de trabajo — TP3 Clasificador de Emociones

Reglas para guiar el desarrollo del notebook y el código asociado a este trabajo.

## 1. Compatibilidad con Google Colab

Todo el código debe poder ejecutarse en **Google Colab** sin modificaciones manuales.

- Usar rutas de Colab (`/content/...`) para datos, modelos guardados e imágenes de prueba.
- Evitar rutas locales del entorno de desarrollo (por ejemplo, rutas de OneDrive, `~/Desktop`, etc.).
- Si hace falta instalar una dependencia que no viene preinstalada, hacerlo con `!pip install` en una celda del notebook (por ejemplo, `gdown`).
- Preferir librerías disponibles en Colab: PyTorch, `torchvision`, `scikit-learn`, `matplotlib`, `seaborn`, `opencv-python`, `PIL`, `numpy`.
- Activar GPU en Colab cuando el entrenamiento lo requiera (`Runtime > Change runtime type > GPU`).
- Probar el flujo completo con **Run all** antes de entregar.

## 2. Formato de entrega

- Entregar **un solo notebook** en Google Colab.
- Nombre del archivo: `APELLIDO-NOMBRE-DL-TP3-Co24.ipynb` (ej.: `Luna-Marcelo-DL-TP3-CO24.ipynb`).
- Compartir con `gvilcamiza.ext@fi.uba.ar` y **habilitar comentarios**.
- Código, resultados, gráficos y explicaciones deben quedar **guardados y visibles** en el notebook (no depender de outputs que solo existen en una sesión local).

## 3. Stack y restricciones técnicas

- Framework: **PyTorch** para la CNN.
- **No usar modelos pre-entrenados** (ResNet, VGG, etc.); la arquitectura debe construirse desde cero.
- Preprocesamiento con `torchvision.transforms` cuando corresponda.
- Métricas y reportes con `scikit-learn`; visualizaciones con `matplotlib` o `seaborn`.
- Detección de rostros (punto 5) con OpenCV y el código de referencia del enunciado.

## 4. Estructura del notebook

- Responder las consignas en el orden del TP (secciones 1 a 5).
- Incluir celdas markdown con **justificaciones** y **conclusiones** donde el enunciado lo pida.
- Citar fuentes externas (link o referencia bibliográfica) cuando se tomen ideas de afuera.

## 5. Datos e imágenes de prueba

- Dataset: descargar desde el link de Google Drive del enunciado (recomendado: `gdown`).
- Imágenes del punto 4 y 5: subirlas al entorno de Colab o montarlas de forma accesible desde `/content/`; deben ser **12 imágenes propias**, al menos una por emoción, ajenas al dataset de entrenamiento/validación.

## 6. Desarrollo en este repositorio

- El notebook principal del trabajo es `Luna-Marcelo-DL-TP3-CO24.ipynb`.
- Los cambios hechos localmente deben mantenerse alineados con lo que se ejecutará en Colab.
- No commitear archivos pesados (dataset, checkpoints, zip); solo el notebook y documentación liviana.
