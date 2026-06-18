# Análisis de reseñas de Skyscanner — Trustpilot

Trabajo final del módulo de **Deep Learning · NLP** del Master de Data Science (Evolve).

Sobre el dataset público de Trustpilot (123k reseñas, 1.680 empresas, 22 sectores) se analiza una empresa con 100 reseñas — **Skyscanner** (`www.skyscanner.net`, sector *Travel & Vacation*) — y se compara contra sus competidores más cercanos para responder cuatro preguntas:

1. ¿La mayoría de las reseñas son positivas o negativas? ¿Y en la competencia?
2. ¿Qué temas tratan estas reseñas? ¿Y en la competencia?
3. Para cada topic, ¿en qué somos mejores o peores que la competencia?
4. ¿Qué áreas de mejora se identifican?

---

## Pipeline NLP

| Paso | Decisión | Por qué |
|------|----------|---------|
| **Limpieza** | Dos modos: `clean_soft` (preserva negaciones, signos, mayúsculas) para sentimiento; `clean_hard` (lowercase + stopwords + lematización + filtro de nombres propios) para topics y wordclouds. | Las negaciones son críticas para sentiment; el ruido morfológico arruina los topics. |
| **Sentimiento** | Dos modelos comparados: `distilbert-base-uncased-finetuned-sst-2-english` (binario) y `cardiffnlp/twitter-roberta-base-sentiment-latest` (3 clases). Cardiff queda como referencia. | El enunciado prohíbe usar `stars` (sesgado por imputación a la mediana). El modelo de 3 clases recoge reseñas tibias / mixtas que SST-2 fuerza a un extremo. |
| **Competencia** | Embeddings (`sentence-transformers/all-MiniLM-L6-v2`) sobre la columna `description`, similitud coseno contra Skyscanner, top-15 más cercanos. | La categoría Travel & Vacation mezcla aerolíneas, hoteles, cottages y metabuscadores. Sin un filtro semántico la "competencia" sería ruido. |
| **Topics** | `BERTopic` con pipeline embeddings + UMAP + KMeans (K=8) y c-TF-IDF (modelo preferido del profesor). `NMF + TF-IDF` (8 tópicos) como sanity check interpretable. | El corpus de reviews de viaje es muy homogéneo en vocabulario; HDBSCAN density-based colapsaba 95% del corpus en un solo cluster, así que se forzó K=8 para garantizar granularidad alineada con el baseline. |
| **Sentimiento × Topic** | Cross-tab de % positivos / neutrales / negativos por (topic, grupo). Filtro: solo topics con ≥4 reseñas de Skyscanner. | Si Sky no participa en un topic, no es un "área de mejora" sino una ausencia natural (Sky es metabuscador, no agencia de holidays). |

---

## Hallazgos clave

- **Sentimiento global** — Skyscanner: 57% neg / 12% neu / 31% pos. Competencia: 49% / 10% / 41% (Cardiff). ~10 pp menos de reseñas positivas que sus pares.
- **Topics encontrados (BERTopic, K=8)** — flight/booking, price/easy/flights, helpful/service, holiday, hotel, room/staff/food, flight/chat/change, travel/tour/itinerary.
- **Distribución de Skyscanner** — 74% en `price/easy/flights` y 20% en `flight/booking/booked` (coincide con su propuesta de valor de búsqueda + comparación).
- **Sentimiento × Topic** — Sky queda por debajo de la competencia en sus dos topics dominantes: −46 pts en search/price y −28 pts en flight/booking.
- **Áreas de mejora priorizadas** — auditar el journey *búsqueda → booking* (cambios de precio, disponibilidad) y reforzar customer care post-booking (cancelaciones, refunds, cambios de fecha).

---

## Estructura del repositorio

```
proyecto_deep_learning/
├── analisis_skyscanner.ipynb      # Notebook principal — entregable
├── Analisis_Skyscanner.pptx       # Presentación draft con figuras del notebook
├── README.md
├── requirements.txt
└── .gitignore
```

Al ejecutar el notebook se genera (gitignored) la carpeta `artifacts/` con `reviews_labeled.csv`, `topic_sentiment_compare.csv`, `competition_similarity.csv` y `bertopic_info.csv`.

Los archivos fuente del enunciado (`trustpilot-reviews-123k.csv`, plantillas .pptx/.ipynb del profesor, `Caso Práctico DL.pdf`) **no se versionan** — están en `.gitignore`.

---

## Cómo ejecutar

### 1. Descargar el dataset

`trustpilot-reviews-123k.csv` (124 MB) **no está en el repo** porque excede el límite de 100 MB de GitHub. Descargarlo desde la plataforma del curso y colocarlo en la raíz del proyecto.

### 2. Crear el entorno

Usar **Python 3.11**. En Windows con OneDrive es importante crear el venv fuera de OneDrive para evitar errores por límite de 260 caracteres en rutas (`OSError [Errno 2]` al instalar paquetes con árboles profundos como jupyterlab/jedi):

```powershell
py -3.11 -m venv "C:\Users\<usuario>\.venvs\sky_dl"
& "C:\Users\<usuario>\.venvs\sky_dl\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
& "C:\Users\<usuario>\.venvs\sky_dl\Scripts\python.exe" -m pip install -r requirements.txt
```

Para `torch` CPU (Windows), usar el índice de PyTorch:

```powershell
& "C:\Users\<usuario>\.venvs\sky_dl\Scripts\python.exe" -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
```

### 3. Registrar el kernel y abrir el notebook

```powershell
& "C:\Users\<usuario>\.venvs\sky_dl\Scripts\python.exe" -m ipykernel install --user --name=sky_dl --display-name "Python 3.11 (sky_dl)"
```

Abrir `analisis_skyscanner.ipynb` en VS Code o Jupyter y elegir el kernel **"Python 3.11 (sky_dl)"**. En la primera corrida NLTK descarga stopwords/wordnet y Hugging Face cachea los modelos (~500 MB en `~/.cache/huggingface/`).

### 4. Ejecutar

`Run All` en el notebook. Tiempos aproximados (CPU):

- Embeddings de descripción: ~5 s.
- Sentimiento DistilBERT (1.544 reseñas): ~1–2 min.
- Sentimiento RoBERTa Cardiff (1.544 reseñas): ~3–5 min.
- BERTopic + NMF: <1 min combinado.

---

## Notas técnicas

- El `kernel sky_dl` está configurado en el notebook. Si querés cambiarlo, edita `metadata.kernelspec` o usa "Select Kernel" en Jupyter/VS Code.
- Las visualizaciones de BERTopic (`visualize_topics`, `visualize_barchart`, `visualize_heatmap`) son Plotly y se renderizan inline. En GitHub no aparecen — abrir el notebook localmente para verlas.
- El sesgo de la columna `stars` se documenta en el notebook (mediana por empresa colapsa en muy pocos valores). Por eso se ignora como label de sentimiento.
- En Windows, evitar instalar el metapackage `jupyter` dentro de OneDrive — su sub-package `jupyterlab` tiene rutas que pasan los 260 caracteres y rompen pip. `ipykernel + nbformat + nbclient` alcanzan.

---

## Licencia y datos

El dataset de Trustpilot pertenece a sus respectivos propietarios. Este repositorio contiene solo código y análisis derivados con fines académicos.
