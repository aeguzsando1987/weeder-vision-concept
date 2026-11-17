# Prueba exploratoria: modelo de segmentación de moringa sobre `moringa_sin_ai`

**Fecha:** 25 de octubre de 2026. **Alcance:** experimento aislado, sin relación con
ninguna etapa del plan y sin aprobación. No modifica código del producto, datos ni
anotaciones de E0.

## Montaje

- **Entorno propio:** `.venv/` (fuera de `uv.lock`). Versiones en
  `requirements.lock.txt`: ultralytics 8.4.189, torch 2.14.1+cu130 y OpenCV 5.0.0.
  ultralytics es AGPL-3.0.
- **Modelo:** `.context/Moringa_Train_Annotations/Train/weights/best.pt`, YOLOv8n-seg
  de una clase (SHA-256 `4996a2c1…d5f1`). Se entrenó con 44 imágenes de Roboflow
  (CC BY 4.0).
- **Inferencia:** GPU RTX 4050 (`cuda:0`), `imgsz` 640, `conf` ≥ 0.25, sobre los 972
  cuadros. Comando:
  `experiments/moringa_seg_probe/.venv/bin/python experiments/moringa_seg_probe/run_probe.py`.
- **Salidas** en `output/`:
  - `detections.csv`: caja, confianza, área y centroide de la máscara, en píxeles
    originales;
  - `summary.json`;
  - `samples/`: las 24 muestras de E0 anotadas;
  - `moringa_sin_ai_detections.mp4`: fuera de Git.

## Resultados

| Medida | Valor |
|---|---|
| Cuadros con detección | 972/972 |
| Detecciones por cuadro | mediana 7, máximo 12 |
| Confianza | mediana 0.86, p10 0.53 |
| Área de máscara | p5 1 296, mediana 6 368, p95 20 268 px² |
| Puntos de eje anotados a mano en E0 (P01–P03) | **9/9 dentro de una máscara** del mismo cuadro |
| Plantas de E0 con base oculta (20 filas) | 20/20 con alguna detección en su línea. El contraste es débil, porque siempre hay varias detecciones por línea |

## Revisión visual

Cuadros revisados: 0, 353, 400, 779 y 931, más un recorte del cuadro 0.

- **Aciertos:**
  - Segmenta las copas en ambas líneas, desde plántulas de ~35–50 px (cuadro 400,
    con confianza 0.55–0.63) hasta plantas grandes.
  - En las muestras revisadas no hay detecciones sobre mangueras, anillos blancos
    vacíos, terrones ni restos secos.
- **Hallazgo sobre E0.** La detección del cuadro 0 (borde inferior izquierdo,
  confianza 0.91) **es una planta real** que entra al encuadre. La evaluación
  manual de E0 describe los cuadros 0–9 como «sin plantas en hilera». Es una
  imprecisión de esa descripción, no un falso positivo. No se modifica aquí la
  anotación aprobada; queda para el Director.
- **Duplicados.** A veces hay dos cajas solapadas sobre una misma planta (cuadro
  931: 0.82/0.44; cuadro 353: 0.95/0.55). Para el mapa hace falta fusionarlas, por
  ejemplo con NMS más estricto o uniendo máscaras solapadas.
- **Máscaras gruesas.** La silueta es poligonal y aproximada (resolución de máscara
  del modelo a 640 px). Para proteger plantas habría que inflarla con un margen
  declarado.
- **Sin base de tallo.** El modelo segmenta la copa. El centroide de la máscara es
  un proxy de la posición de la planta, no el contacto tallo–suelo. La puerta
  `PENDING_BASE_VISIBILITY` sigue igual.

## No evaluado todavía

- Precisión y exhaustividad formales: haría falta anotar las copas de un conjunto de
  cuadros independiente.
- Estabilidad temporal por planta (parpadeos y pérdidas), que requiere seguimiento
  entre cuadros (E5).
- Plantas omitidas en cuadros no revisados.
- La detección de baja confianza del cuadro 6 (0.44, en u≈1356, v≈355) no se
  inspeccionó.
