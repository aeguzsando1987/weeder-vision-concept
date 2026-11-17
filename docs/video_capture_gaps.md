# Brechas de captura y documentación — E0

**Fecha:** 24 de octubre de 2026; revisado el 25 de octubre de 2026 (dictamen E0
rev. 1). **Base:** `docs/video_characterization.md` (clip `moringa_sin_ai`, clase B;
puerta de datos `PENDING_BASE_VISIBILITY`).

Regla de la orden E0 §1: se piden nuevas tomas **solo** cuando una pregunta
necesaria no puede resolverse con el material existente. Por eso cada brecha indica
primero si basta con documentación o con el archivo original.

## Resumen por prioridad

| Prioridad | Qué se pide | ¿Nueva captura? | Desbloquea |
|---|---|---|---|
| 1 | Ficha de montaje y del recorrido (G3, G5, G8) | No: documentación y fotos | Orientación, sentido de avance y metadatos de la sesión |
| 2 | Archivo original de la cámara, sin editar (G1, G2) | No: archivo ya existente | Tiempos de captura reales y posible mejora del detalle de tallos |
| 3 | Clip corto de calibración (G4) | **Sí**, ligado a E4 | Escala métrica y homografía validable |
| 4 | Geometría cámara-herramienta (G6) | Documentación; captura solo si no existe registro | Validación de anticipación |
| 5 | Toma específica de bases (G1) | **Solo si** la documentación y el material existente (prioridades 1–2) no permiten bases observables; no se exige para aprobar el software de diagnóstico | Bases tallo-suelo identificables; cierre de la puerta de datos |

## Brechas

### G1 — El contacto tallo-suelo no es observable

- **Pregunta bloqueada:** ¿se pueden localizar bases reales de varias plantas para
  la clase A?
- **Evidencia:** ninguna de las 11 plantas inspeccionadas muestra el contacto
  tallo-suelo en los cuadros anotados; no se afirma para el resto del clip. Solo 3
  tienen un proxy de eje (`stem_axis_proxy`) localizable desde arriba (±12–20 px),
  que no equivale a una base en el suelo. Las causas observadas son la oclusión por
  la copa y, en las plántulas, el tamaño (~30–60 px) junto con la compresión.
- **Estado de la puerta:** `PENDING_BASE_VISIBILITY` (disposición del Director). No
  se cierra cambiando `partial` por `visible`, relajando el criterio ni interpolando.
  Una base parcialmente ocluida sería utilizable si el punto de suelo tiene soporte
  observable y una incertidumbre justificada.
- **Con material existente:** primero, la documentación y el archivo original, si
  existen (G2). No se presume que exista un original de más calidad ni que resuelva
  la oclusión: más detalle podría ayudar con tallos de 1–3 px, pero la copa sigue
  cubriendo la base.
- **Captura, justificada solo si lo anterior no permite bases observables:** una
  pasada sobre plantas de distintos tamaños, en la que la base tenga soporte
  observable. Por ejemplo, con alguno de estos cambios:
  - cámara más baja o con más resolución;
  - menos compresión;
  - una segunda vista oblicua lateral sincronizada que muestre el cuello del tallo.
  Las cifras orientativas de la entrega 1 (≥ 30 Mb/s, ≥ 10 plantas) no son
  garantías técnicas. La aceptación depende de la visibilidad y la evidencia, no de
  una tasa de bits aislada.

### G2 — El archivo está recodificado

- **Pregunta bloqueada:** ¿los PTS reflejan los instantes de captura? ¿Hay más
  detalle en el original?
- **Evidencia:** el contenedor declara `te_is_reencode=1` y `encoder=Lavf61.1.100`.
  Son indicios de procesamiento, no prueba independiente de un editor concreto ni
  de toda la cadena de adquisición. Los PTS son perfectamente constantes y el
  cuadro 2 es un candidato a imagen casi repetida, compatible con una conversión a
  frecuencia constante pero no certificado como duplicado. No se infieren cuadros
  perdidos.
- **Solicitud:** el archivo tal como lo grabó la cámara, con su nombre y fecha
  originales y el modelo de cámara. **No es una nueva captura.**

### G3 — Montaje de la cámara desconocido

- **Pregunta bloqueada:** solidaridad con el vehículo, orientación (qué borde de la
  imagen mira hacia delante), altura e inclinación. Sin ellas no se interpreta
  físicamente el flujo «hacia arriba» ni se puede fijar `ImageVAxisOnGround` (E1, R7).
- **Evidencia:** los gradientes del flujo muestran que no es una traslación
  uniforme. Son compatibles con inclinación, perspectiva o un giro entre cuadros, y
  no fijan el montaje.
- **Solicitud (documentación):**
  - fotos del soporte;
  - altura al suelo medida con cinta;
  - ángulo aproximado;
  - el borde de la imagen que mira hacia el frente del tractor;
  - si el soporte es rígido.

### G4 — Sin escala métrica ni calibración

- **Pregunta bloqueada:** conversión píxel→suelo, precisión en cm y homografía
  `H_ground_from_undistorted_image`. Corresponde a E4; aquí se identifica la
  necesidad.
- **Evidencia:** no hay ninguna referencia de dimensión conocida y documentada en
  el clip. Las candidatas visibles (manguera, cinta, anillos, neumáticos) no tienen
  medida documentada.
- **Primero, documentación:** diámetro nominal de la manguera y longitud de las
  marcas de cinta, si existen. Aun así, esto solo daría una escala **supuesta**, no
  validada.
- **Captura, necesaria para E4:** con la misma cámara, ajustes y montaje que el
  video:
  1. un tablero de calibración (por ejemplo, un damero de tamaño medido) en varias
     posiciones, para obtener intrínsecos y distorsión;
  2. ≥ 4 marcas en el suelo dentro del campo visual, con distancias medidas con
     cinta, más 2 puntos de control independientes que se reserven para la
     validación;
  3. una pasada a la velocidad de trabajo sobre esas marcas.

### G5 — Sentido de avance y velocidad

- **Pregunta bloqueada:** ¿el vehículo avanzaba hacia delante? ¿A qué velocidad?
  El flujo da −25.6 px/cuadro, sin escala.
- **Solicitud (documentación):** sentido de la marcha durante la grabación y
  velocidad aproximada (tacómetro, GPS o tiempo cronometrado en un tramo medido).
  Con G4, permite contrastar la escala.

### G6 — Geometría cámara-herramienta y anticipación

- **Pregunta bloqueada:** ¿la planta se observa antes o después de la intervención?
  ¿Con cuánto margen?
- **Evidencia:** la herramienta no aparece en el encuadre. La franja oscura del
  borde inferior no es identificable.
- **Solicitud:** distancia longitudinal y lateral medida entre el campo visual (o
  el eje óptico) y la cuchilla, más la latencia de grabación, si se conoce. Solo si
  no existe ese registro: una toma en la que la herramienta, o una marca solidaria a
  ella, sea visible en el encuadre.

### G7 — Vibración

- **Pregunta bloqueada:** separar vibración de correcciones de dirección (jitter du
  1.51 px y dv 0.96 px por par).
- **Solicitud:** no es necesaria para E0. Si E5 la requiere, registro de un IMU
  solidario a la cámara durante la toma de G4.

### G8 — Metadatos de sesión y sitio

- **Pregunta bloqueada:** `session_id`, `site_id`, fecha y hora local, estado del
  cultivo. Hoy están en `unknown` en el manifiesto.
- **Solicitud (documentación):** nombre del lote o sitio, fecha, edad de las plantas
  y si el surco del video corresponde a otros clips.

## Condiciones no presentes en el clip

El clip no contiene sombras móviles, maleza densa, cambios de iluminación ni
desenfoque de movimiento apreciable. **No se piden ahora:** solo serán necesarias
para evaluar la robustez de la percepción (E6). No se simularon.
