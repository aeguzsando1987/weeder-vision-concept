# Caracterización del video real — `moringa_sin_ai`

**Etapa:** E0, conforme a `docs/orders/E0_technical_order.md`. **Fecha:** 24 de octubre de 2026;
revisado el 25 de octubre de 2026 por el dictamen E0 rev. 1 (E0-R1 a E0-R3).
**Estado:** entrega 2 para auditoría; no aprobado.

> **Cambios de la entrega 2.** No cambia ninguna medición. Se corrigen
> interpretaciones (§1, §2, §5, §8), se tipan los puntos anotados como proxies de
> eje (§4) y la puerta de datos queda `PENDING_BASE_VISIBILITY` por disposición del
> Director. Detalle en `docs/reviews/E0.md` §7.

Todas las magnitudes están en **píxeles** del cuadro decodificado, salvo que se
indique otra unidad. Ejes de imagen: `u` hacia la derecha, `v` hacia abajo, sin
rotación de visualización. Ninguna cifra de este documento es una medida métrica.

Reproducción:

```bash
uv sync --frozen --extra dev
.venv/bin/python scripts/characterize_videos.py --config configs/video.yaml
.venv/bin/python scripts/render_stem_observations.py \
  --assessment experiments/phase0/moringa_sin_ai/manual_assessment.yaml
```

Resultados deterministas en `experiments/phase0/moringa_sin_ai/characterization.json`
(clave `results`). Los datos de ejecución están aparte, en `run_moringa_sin_ai.json`,
y el hash de cada derivado en `derived_manifest.json`.

## 1. Ficha técnica

| Campo | Valor | Fuente |
|---|---|---|
| Archivo | `resources/moringa_sin_ai.mp4` (original sin modificar) | — |
| SHA-256 | `63f43e111c3a8031aafcdc738eef32ed47f2e12b877e87fcce0817a226e9e083` | calculado; coincide con la orden |
| Tamaño | 43 744 466 bytes | sistema de archivos |
| Contenedor | `mov,mp4,m4a,3gp,3g2,mj2` (QuickTime/MOV), marca `isom` | PyAV |
| Códec / tag / perfil | `h264` / `avc1` / `Main`, `yuv420p` | flujo 0 |
| Resolución | 1920 × 1080 (único tamaño observado en los 972 cuadros) | decodificación |
| Rotación declarada | 0° (`frame.rotation`, matriz de visualización; 0 si no se declara) | decodificación |
| Duración declarada | 32.4 s | flujo |
| Cuadros declarados / decodificados | 972 / 972 | flujo / recorrido |
| FPS nominal | 30 (`stream.average_rate`) | flujo |
| `time_base` | 1/15360 | flujo |
| Tasa de bits | 10.8 Mb/s (el contenedor declara `bitrate=12000000`) | PyAV |
| Otros flujos | `1:audio:aac` (no analizado) | contenedor |
| Metadatos de codificación | `encoder=Lavf61.1.100`, `te_is_reencode=1`, `Hw=1`, `creation_time=2026-07-01T17:08:48Z` | contenedor |
| Decodificador | PyAV 19.0.2; libavcodec 63.1.102; libavformat 63.1.102; OpenCV 5.0.0 | `backend_versions()` |

`te_is_reencode=1` y `encoder=Lavf…` son metadatos **declarados por el contenedor**
que indican procesamiento o recodificación con libavformat. Son un indicio, no una
prueba independiente: no identifican un editor concreto ni describen toda la cadena
de adquisición. Aun así, condicionan la interpretación temporal (§2).

## 2. Decodificación y tiempo

- **Recorrido secuencial hasta EOF:** `eof_reached`, 972 cuadros, sin errores ni
  advertencias de FFmpeg y sin truncamiento sospechado. Los 972 cuadros tienen PTS.
- **PTS:** de 0 a 32.3667 s, en pasos exactos de 512 ticks (1/30 s). La secuencia es
  estrictamente monótona, sin duplicados, regresiones ni saltos (umbral 1.5 ×
  mediana).
- **Intervalos:** mín. = p05 = mediana = p95 = máx. = 0.03333 s; desviación 1.3e-15 s.
- **FPS efectivo:** 30.000, por el método `(pts) (N−1)/(t_último − t_primero)`.
- **Duración observada:** 32.4 s, por el método `(pts) t_último − t_primero + mediana`.
- **Estructura:** solo cuadros I/P, con un cuadro clave cada 150 (5 s).
- **Candidato a imagen casi repetida:** el cuadro 2 difiere muy poco del cuadro 1
  (diferencia media 0.37 frente a una mediana de 16.8 niveles) pese a tener PTS
  regular. El par 1→2 tiene flujo ≈ 0, y el par 2→3 vuelve a ≈ −25.8 px.
  - Es el único candidato del clip; queda registrado como `content_repeat_candidate`
    en `timestamp_anomalies.csv`.
  - Es compatible con una repetición introducida al codificar a frecuencia
    constante, pero **no se certifica** como duplicado de captura.
  - **No** se infiere que se perdiera un cuadro durante la captura: no hay
    referencia externa.
- **Diagnóstico temporal (entrega 2):** se calcula sobre los PTS observados (972 de
  972) y por separado de la línea temporal usada. `timestamps.csv` incluye
  `timestamp_observed_s` (PTS) junto a `timestamp_s` y `timestamp_method`. Para este
  clip ambas coinciden, porque no falta ningún PTS.

**Consecuencia:** los PTS regulares describen este archivo, no necesariamente los
instantes de captura. El tiempo de este clip sirve para el replay offline; no
acredita la temporización de la cámara original.

## 3. Muestras y condiciones visuales

Hay 24 muestras deterministas (`samples.csv`, `frames/`, `contact_sheet.jpg`): 12
uniformes (cuadros 0, 88, 177, 265, 353, 441, 530, 618, 706, 794, 883 y 971) y 12
extremas de métricas indicativas (nitidez baja, salto de brillo, vegetación alta y
baja, dispersión de flujo y pocos vectores). Cada imagen lleva en una banda
inferior, que no tapa píxeles, el clip, el hash, el índice, el PTS, el tiempo, el
método temporal y el motivo de selección.

**Contenido.** Toma casi cenital de un surco con dos líneas de goteo (mangueras
negras con marcas de cinta blanca) y plántulas de moringa junto a cada manguera.
Muchas tienen un anillo blanco alrededor; también hay anillos sin planta. El suelo
es desnudo, con terrones y huellas de neumático. El cuadro 0 muestra el inicio del surco con una planta parcialmente visible
en el borde inferior izquierdo; no debe describirse el intervalo 0–9 como ausencia
de plantas. Véase la [adenda editorial E0](reviews/E0_editorial_addendum.md).

| Condición | Observada | Evidencia |
|---|---|---|
| Sol y sombra | sí | Sol pleno todo el clip; sombras cortas y estables; sin sombras móviles. |
| Desenfoque | no | Las muestras de menor nitidez (280, 584) no muestran desenfoque de movimiento. La compresión limita el detalle fino. |
| Oclusión | sí | La copa ocluye el contacto tallo-suelo en las 11 plantas inspeccionadas, en los cuadros de la tabla (§4). No se afirma para el resto del clip. |
| Vibración | no concluyente | Jitter del flujo: du 1.51 px, dv 0.96 px por par. No se separa de las correcciones de dirección. |
| Maleza densa | no | Fracción verde (ExG) ≤ 0.055. |
| Cambio de iluminación | no | Brillo medio entre 101 y 115; salto máximo de 1.37 niveles entre cuadros. |
| Entrada y salida de plantas | sí | Entran por abajo y salen por arriba en unos 40 cuadros (~1.4 s). |

**Objeto en el borde inferior.** En algunos cuadros aparece una franja oscura de
pocos píxeles en el borde inferior (u ≈ 900–1030). No es identificable: puede ser
una sombra o parte del vehículo, y nada demuestra que sea la herramienta. La ROI de
flujo excluye el 3 % inferior del cuadro.

## 4. Bases de tallo (observación manual)

Las observaciones son **manuales**: lectura sobre rejilla de recortes a resolución
completa y ampliaciones ×2–×4. No son salida de un detector. Tabla:
`experiments/phase0/moringa_sin_ai/stem_observations.csv`, con 29 eventos y 11
plantas. Evidencia visual reproducible en `manual_evidence/` (tiras ampliadas por
planta, con el círculo de incertidumbre, y cuadros completos con la línea virtual).

Desde la entrega 2, cada fila declara qué punto marca (`observed_target`):
`ground_contact` (contacto tallo-suelo), `stem_axis_proxy` (eje o convergencia de
ramas vista desde arriba, a altura desconocida) o `none`. El anotador fijó el
objetivo fila a fila; no se infiere del texto libre. Los demás valores de la tabla
no cambiaron. La tabla y la evaluación están vinculadas al clip por `video_id` y
SHA-256, y se validan contra los cuadros decodificados antes de publicar nada.

| Resultado | Detalle |
|---|---|
| Contacto tallo-suelo (`ground_contact`) | **En ninguna** de las 11 plantas inspeccionadas, en los cuadros anotados. Las plántulas pequeñas (~30–60 px) son matas de hojas sin tallo resoluble; en las grandes, la copa cubre la base y la manguera. No se extrapola al resto del clip. |
| Proxy de eje (`stem_axis_proxy`) | 3 plantas (P01, P02 y P03) con el punto de convergencia de ramas localizado en 3 cuadros cada una, con ±12–20 px y confianza media o baja. **No es una base en el suelo.** |
| Sin punto localizable (`none`) | 8 plantas (P04–P11), registradas como `hidden` sin posición: la tabla no inventa coordenadas. |
| Visibilidad global | `mostly_occluded` |

**Línea virtual de cruce:** horizontal en v = 540 px (fila central), con u de 0 a
1920 y sentido `up`. Es una línea en píxeles: no representa la cuchilla ni el
instante físico de una intervención. Cruces del **proxy de eje**, interpolados
linealmente entre dos anotaciones (`crossing_events.csv`). No son eventos de base
medidos. Se conserva el intervalo de soporte, y la incertidumbre propaga la de
lectura en `v`:

| Planta | Objetivo | Cruce [s] | Soporte [s] (cuadros) | Incertidumbre de lectura [s] |
|---|---|---|---|---|
| P02 | `stem_axis_proxy` | 11.749 | 11.567–11.767 (347 → 353) | ±0.016 |
| P01 | `stem_axis_proxy` | 13.551 | 13.533–13.733 (406 → 412) | ±0.018 |
| P03 | `stem_axis_proxy` | 29.476 | 29.433–29.633 (883 → 889) | ±0.017 |

## 5. Movimiento aparente

**Método:** esquinas Shi-Tomasi en la ROI (v < 1048 px), Lucas-Kanade piramidal
sobre el cuadro a 0.25 y verificación ida/vuelta ≤ 1 px. Se excluyen los vectores
sobre vegetación (ExG). Se procesan los 971 pares consecutivos.

| Resultado | Valor |
|---|---|
| Dirección del suelo en imagen | **up** en 970 de 971 pares. El restante es `none`: es el par 1→2 de la repetición. |
| Mediana dv / du | −25.65 / +1.57 px por cuadro (original) |
| p05–p95 dv / du | −26.90 a −24.19 / −1.32 a +4.42 px |
| Vectores válidos de suelo (mediana) | 375 por par; rechazados por ida/vuelta: 1 por par (mediana) |
| Vectores de vegetación (mediana) | 23 por par, analizados aparte. |

**El flujo no es uniforme en el cuadro.** El ajuste afín sobre las 6 muestras de
vectores (`flow_examples[].affine_fit`) da gradientes estables:

- dv/dv ≈ −0.0045 a −0.0074: el suelo sube más rápido en la parte inferior.
- dv/du ≈ −0.003 a −0.005: más rápido a la derecha.
- du/du ≈ −0.0016 a −0.0047.

Los residuos robustos son de 0.14 a 1.04 px. Los gradientes muestran que **el flujo
no es una traslación uniforme**. No identifican su causa. Un giro en el plano entre
cuadros, con escala uniforme (`x' = R·x + b`), da un flujo `(R − I)·x + b` que
también tiene gradientes no nulos; la perspectiva o la inclinación son otras
explicaciones. Por tanto, el ajuste afín **no determina** inclinación ni calibración,
y **no descarta** una escala uniforme. Lo que se afirma es que **no se ha
identificado ni validado ninguna conversión métrica** (§7). Corrección de la entrega
2: antes se concluía que no existía una escala única, una inferencia más fuerte que
la evidencia.

**Contraste manual en 3 intervalos.** Marcas blancas de cinta sobre las mangueras
seguidas a ojo, comparadas con el flujo LK local a lo largo del mismo recorrido y en
los mismos pares:

| Intervalo | Manual du / dv [px/cuadro] | Automático local du / dv | Tolerancia declarada | Resultado |
|---|---|---|---|---|
| c1: 150→156 | −0.63 / −28.17 | −1.09 / −27.66 | ±1.67 | coincide |
| c2: 388→400 | −1.25 / −27.60 | −1.21 / −26.75 | ±0.50 | sentido coincide; **dv difiere en 0.86, fuera de tolerancia** |
| c3: 850→856 | +3.33 / −27.25 | +2.90 / −26.29 | ±1.00 | coincide |

La dirección coincide en los tres intervalos. `same_vertical_sign` solo acredita esa
coherencia de dirección; el único criterio de magnitud es `within_tolerance`, y c2
lo incumple. Por tanto, **la magnitud no está validada**. La magnitud manual supera
a la automática en un 2–4 % en los tres intervalos. La causa sigue sin conocerse:
podría ser un sesgo de LK a escala 0.25, la altura de la manguera sobre el suelo o
que la máscara ExG etiquete como suelo objetos no verdes. Una marca de cinta y una
mediana de puntos locales no son necesariamente el mismo punto ni el mismo plano.
Antes de usar este contraste para validar magnitud habría que fijar
correspondencias y ROI comparables. Las cifras son píxeles por paso, no m/s.

**Plantas frente a suelo.** Los ejes de tallo de P01–P03 se desplazan entre un 2.5 y
un 5 % más que el suelo local. Con su incertidumbre (±2 px/cuadro) y el sesgo
manual anterior, **no es posible demostrar** paralaje de la copa con este material.

**Interpretación física: ninguna.** El flujo hacia arriba es compatible con varias
combinaciones de orientación de cámara y sentido de marcha. No se traduce a avance
+Y (E1, R7).

## 6. Montaje, temporalidad y anticipación

| Aspecto | Estado | Fuente |
|---|---|---|
| Cámara solidaria al vehículo | `unknown` | Sin ficha de montaje. El flujo estable es compatible con ello, pero no lo prueba. |
| Orientación | `unknown` | El flujo hacia arriba no fija la orientación. |
| Altura | `unknown` | Los 0.5 m del plan son una hipótesis. |
| Posición respecto de la herramienta | `unknown` | La herramienta no es visible y no hay documentación. |
| Observación antes/después de la intervención | `unknown` | — |
| **Replay offline** | **posible (B)** | Seguimiento por copa y línea virtual en píxeles. |
| **Validación temporal anticipativa** | **no posible con este material** | Herramienta no visible y montaje desconocido. Ver plantas entrar en el encuadre no prueba anticipación. |

## 7. Escala y geometría

**Escala: `unknown`.** No hay ninguna referencia métrica independiente visible ni
documentada. Candidatas examinadas y descartadas:

- diámetro de la manguera: no documentado;
- longitud de las marcas blancas: desconocida;
- anillos blancos: irregulares;
- dibujo de los neumáticos: desconocido;
- espaciamiento entre plantas y tamaño de hojas: excluidos por la orden.

Sin intrínsecos ni distorsión. Las coordenadas quedan en píxeles y no se informa
ninguna precisión en cm.

**Homografía: `possible_with_additional_data`.** El campo de flujo es suave y bien
descrito por un modelo afín. Eso es compatible con un suelo aproximadamente plano en
la ROI, pero no lo prueba, y también es compatible con varios modelos de cámara y
movimiento (§5). Surcos, terrones y huellas aportan relieve local. Una homografía exigiría:

- ≥ 4 puntos de control con coordenadas métricas independientes en el suelo;
- intrínsecos y distorsión;
- una pose estable.

Hoy solo sería posible una «estimación asumida» con medidas supuestas. La
calibración validada pertenece a E4.

## 8. Clasificación

| Intervalo (cuadros) | Clase | Motivo |
|---|---|---|
| 0–9 (0.0–0.3 s) | B | Inicio del surco; planta parcial en el borde inferior izquierdo del cuadro 0. Útil para flujo aparente de suelo. No se acredita base ni seguimiento completo de esa planta en este intervalo; véase la adenda editorial E0. |
| 10–971 (0.33–32.4 s) | B | Plantas identificables y seguibles como objetos (copa) durante ~40 cuadros. El contacto tallo-suelo está ocluido en las plantas inspeccionadas; solo en algunas medianas se localiza un proxy de eje. |

Corrección de la entrega 2: el intervalo 0–9 figuraba como C y a la vez como «útil
para flujo». Según la definición de la orden, esa utilidad limitada corresponde a B.
Es un cambio de interpretación, no de medición. La ausencia de plantas se declara
aparte y no implica inutilidad global.

**Clase global: B** (todo el clip). Usos admitidos:

1. Flujo aparente del suelo en píxeles.
2. Seguimiento de plantas como objetos y replay offline con línea virtual v = 540 px.
3. Anotación de proxies de eje en 3 plantas, con incertidumbre declarada y cruces
   interpolados con intervalo de soporte.

**No es A** porque el contacto tallo-suelo no es identificable en ninguna de las 11
plantas inspeccionadas, y un proxy de eje o copa no lo sustituye. La clase B no
depende de la falta de escala ni de montaje: son restricciones independientes.

**Puerta de datos E0: `PENDING_BASE_VISIBILITY`** (disposición del Director, dictamen
E0 rev. 1). Hay ≥ 3 plantas seguidas en varios cuadros y una línea virtual
reproducible, pero a nivel de proxy de eje visto desde arriba, no de base en el
suelo. La puerta no se cierra con este material ni cambiando etiquetas. Sigue
pendiente hasta disponer de observaciones con soporte de suelo (ver
`docs/video_capture_gaps.md`, G1).

## 9. Límites de este análisis

- **Una sola persona anotó:** sin segunda anotación ni acuerdo entre anotadores, y
  con incertidumbres estimadas, no medidas.
- **Cobertura:** la inspección cubre 11 plantas y ventanas seleccionadas, no todo el
  clip.
- **Compresión:** H.264 a ~10.8 Mb/s, con indicios de procesamiento/recodificación (§1); limita el detalle de tallos de
  1–3 px.
- **Flujo:** las métricas de selección son indicativas, y el flujo se calcula a 0.25
  de la resolución.
- **Determinismo:** comprobado en Linux x86-64 con PyAV 19.0.2 y OpenCV 5.0.0. Otras
  versiones del decodificador o del JPEG podrían cambiar los bytes de las imágenes;
  las tablas numéricas deberían coincidir dentro de tolerancia.
