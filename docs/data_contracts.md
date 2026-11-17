# Contratos de datos

**Etapa:** E1, revisado tras el dictamen `docs/reviews/E1_review.md` (R1–R12).
**Referencia normativa:** `docs/technical_plan.md` §3.2.
**Versión de esquema:** 1 (`core.config.SCHEMA_VERSION`, `scenarios.synthetic.SCENE_SCHEMA_VERSION`).

Este documento especifica los contratos del proyecto y marca cuáles están
**implementados** en E1 y cuáles quedan **especificados** para su etapa. No se
crean módulos vacíos para simular avance (§4).

## 1. Marcos de coordenadas

| Marco | Símbolo | Definición | Unidad |
|---|---|---|---|
| Imagen | `I` | `(u, v)` en píxeles, origen arriba-izquierda, `u` a la derecha, `v` **hacia abajo**. Distinguir imagen cruda de imagen sin distorsión. | px |
| Suelo / mundo | `W` | `(x, y)`; `x` lateral a la derecha del sentido de avance, `y` longitudinal **hacia delante**. Fijo durante cada segmento local. | según dominio |
| Vehículo | `V` | Origen y orientación documentados respecto a `W` mediante `T_WV(t)`. | según dominio |
| Herramienta | `B` | Origen en el pivote declarado; polígono local. Pose `T_VB(q(t))`. | según dominio |

Composición: `T_WB(t) = T_WV(t) · T_VB(q(t))`. **El avance del tractor entra una
sola vez, por `T_WV`.** `simulation.tool.world_footprints` rechaza recibir `T_WV`
para un estado ya expresado en `W`.

Las transformaciones son rígidas (rotación + traslación): no hay escalado ni
reflexión, de modo que ningún eje puede invertirse de forma silenciosa. Tolerancia
de ida/vuelta en fixtures bien condicionados: `1e-9`.

**Marco y dominio se comprueban juntos (R3).** Toda operación entre geometrías
exige el mismo marco **y** el mismo dominio, o una transformación explícita previa:
`Corridor.contains_point` rechaza otro marco (`FrameMismatchError`) u otro dominio
(`UnitMismatchError`); `SyntheticScene` exige un único marco y dominio en todos sus
componentes; `Transform2D.allclose` devuelve `False` entre dominios distintos.

## 2. Dominios espaciales y unidades

| Dominio | Unidad | Métrico | Uso |
|---|---|---|---|
| `image_px` | px | no | Exploración en píxeles. |
| `ground_assumed` | m (supuestos) | **no** | Exploración con escala supuesta; no certifica nada. |
| `ground_metric_synthetic` | m | sí (por construcción) | Banco sintético. **No** equivale a escala verificada. |
| `ground_metric_verified` | m | sí (por medición) | Solo con calibración verificada y evidencia de escala. |
| `graphic_unscaled` | `svg_unit` | no | Escena del SVG; nunca se convierte a metros. |

Reglas, implementadas en `core.units`:

- `require_same_domain(...)` rechaza mezclar dominios en una misma operación.
- `require_verified_metric(...)` rechaza cualquier dominio que no sea
  `ground_metric_verified`, incluido el sintético.
- `mode: calibrated` exige `calibration.status == verified` con evidencia de
  escala; su ausencia es fallo explícito, nunca degradación.

## 3. Tiempo

Tiempo en segundos desde el inicio del clip, derivado de PTS; latencias en
segundos. La hora de ejecución del programa es un reloj distinto. Un índice de
fotograma no reemplaza tiempo en videos de FPS variable. Las secuencias
temporales rechazan tiempos no finitos, negativos o decrecientes
(`errors.TemporalOrderError`).

## 4. Contratos implementados en E1

### 4.1 `Point2` — `core.types`

| Campo | Tipo | Regla |
|---|---|---|
| `x`, `y` | `float` | Finitos; NaN/inf rechazados. |
| `frame` | `Frame` | `W`, `V`, `B` o `I`. |
| `domain` | `SpatialDomain` | Determina la unidad (`unit`). |

Inmutable (`frozen`). `to_dict()` exporta marco, dominio y unidad.

### 4.1 bis Polígonos — `core.types.validate_polygon_points`

Devuelve una **copia propia de solo lectura** `(n, 2)` (memoria respaldada por
`bytes`; el indicador `writeable` no puede reactivarse). Reglas (R10): un vértice
final **exactamente igual** al primero se toma como cierre y se descarta; no se
fusionan vértices por proximidad relativa, de modo que la validez es invariante a
traslación; se exigen ≥ 3 vértices distintos, validez sin autointersección y área
> 0.

### 4.2 `UncertaintyBound` — `core.types`

| Campo | Tipo | Regla |
|---|---|---|
| `localization`, `calibration`, `tracking` | `float` | ≥ 0, finitos. |
| `domain` | `SpatialDomain` | Unidad de las cotas. |
| `provenance` | `Provenance` | Origen declarado. |
| `k_sigma` | `float \| None` | > 0 si se declara. |

`total` es la **suma conservadora** de las tres cotas (§3.3). `within_budget(b)`
compara contra el presupuesto; superarlo invalida la intervención. No incluye el
error de ejecución del actuador (E3).

### 4.2 bis `UncertaintyPolicy` e `InflationTarget` — `core.types` (R1)

| Campo | Tipo | Regla |
|---|---|---|
| `bound` | `UncertaintyBound` | Cotas declaradas; `total` es la protección mínima exigida. |
| `budget` | `float` | ≥ 0 y ≥ `bound.total`. Es un **máximo**, no el margen inflado. |
| `inflate` | `InflationTarget` | `exclusions` o `tool`; nunca ambas. |

Fuente única de la incertidumbre efectiva de una escena. Se obtiene de
`safety.uncertainty` mediante `UncertaintyConfig.to_policy(domain)`, que rechaza
aplicar cotas en metros a dominios no métricos.
`covariance_to_bound(Σ, k) = k·sqrt(λ_max(Σ))` rechaza matrices no simétricas o no
semidefinidas positivas. Es un modelo de incertidumbre, no una garantía
probabilística; una confianza de detector **no** es una desviación espacial.

### 4.3 `CalibrationReference` — `core.types`

| Campo | Tipo | Regla |
|---|---|---|
| `calibration_id` | `str` | No vacío. |
| `status` | `CalibrationStatus` | `absent` \| `unverified` \| `verified`. |
| `scale_evidence` | `str \| None` | Se normalizan espacios; **obligatorio y no vacío** si `status == verified`. |
| `independent_point_error` | `float \| None` | ≥ 0; error en puntos independientes. |
| `domain` | `SpatialDomain` | Dominio que autoriza; `ground_metric_verified` si `status == verified`. |

`require_verified()` lanza `MissingCalibrationError` si no está verificada. En E4
este contrato se amplía con intrínsecos, distorsión, homografía
`H_ground_from_undistorted_image`, dominio espacial, resolución, versión y
vigencia temporal. `ProjectConfig` valida la referencia declarada en todos los
modos y exige que esté verificada en `mode: calibrated` (R2).

### 4.4 `PlantReference` — `core.types`

| Campo | Tipo | Regla |
|---|---|---|
| `plant_id` | `str` | No vacío, único en la escena. |
| `position` | `Point2` | Base del tallo, **verdad de referencia**. |
| `stem_radius` | `float` | > 0. |
| `protection_radius` | `float` | > 0 y ≥ `stem_radius`. |
| `provenance` | `Provenance` | `synthetic` en E1. |

Es verdad de referencia, no una detección: se mantiene separada del mapa estimado
(§3.1). Detectar follaje no equivale a localizar la base.

### 4.5 `CircularExclusion` — `core.types`

| Campo | Tipo | Regla |
|---|---|---|
| `plant_id` | `str` | Debe corresponder a una planta declarada. |
| `center` | `Point2` | Centro de la protección. |
| `protection_radius` | `float` | > 0; `P_i`. |
| `uncertainty_margin` | `float` | ≥ 0; `m_i`. |
| `polygon_segments` | `int` | ≥ 8. |

`total_radius = protection_radius + uncertainty_margin`, que es exactamente la
suma de Minkowski `E_i = P_i ⊕ D(m_i)` en el caso circular.
`discretization_excess = total_radius·(sec(π/n) − 1)` es el exceso radial
conservador del polígono, exportado en el JSON (R12). La regla única
`discretization_excess_violation` lo compara con
`safety.max_conservative_discretization_excess_m` en `ProjectConfig` y, mediante
`require_discretization_budget`, sobre cada exclusión construida de una escena
métrica (también las del catálogo); las escenas no métricas quedan exentas por
dominio. `as_polygon()` entrega
una aproximación **exterior**: el radio se escala por `1/cos(π/n)`, de modo que el
círculo exacto queda cubierto y la exclusión nunca se subestima (§3.4).

### 4.6 `Corridor` — `core.types`

| Campo | Tipo | Regla |
|---|---|---|
| `x_min < x_max`, `y_min < y_max` | `float` | Finitos; rectángulo no vacío. |
| `frame`, `domain`, `provenance` | enums | Declarados. |

Delimita el dominio permitido y conocido. `contains_point` rechaza puntos de otro
dominio.

### 4.7 `Transform2D` y `Pose2` — `geometry.frames`

| Campo | Tipo | Regla |
|---|---|---|
| `parent`, `child` | `Frame` | `compose` exige `self.child == other.parent`. |
| `tx`, `ty`, `theta` | `float` | Finitos. |
| `domain` | `SpatialDomain` | `compose` exige dominio común. |

`inverse()` es exacta; `allclose` compara con tolerancia `1e-9` y ángulo
normalizado. `Pose2` prohíbe `parent == child`; `Transform2D.identity` lo permite
solo para la identidad.

`advance_sign_under_declared_mounting(Δv, v_axis_on_ground)` devuelve **solo un
signo** bajo un montaje declarado (R7): `forward` (`+v` → `+y_W`) hace que el flujo
hacia arriba (`Δv < 0`) implique avance en `+y_W`; `backward` invierte la
correspondencia; `unknown` lanza `MissingCalibrationError`. Supone escena estática
y flujo dominado por la traslación del vehículo. No da velocidad, escala ni
anticipación.
`metric_ground_displacement_from_image` exige calibración verificada y, aun con
ella, señala que la conversión se implementa en E4.

### 4.8 `ToolGeometry` (= `ToolProfile`) — `geometry.tool_profile`

Desde A1 vive en `geometry/tool_profile.py` y `simulation.tool` la reexporta: es la misma clase, sin copias. Añade `content_hash()`, que es el SHA-256 de los polígonos, el dominio y la procedencia, `is_convex()` e `is_cut_convex()`, y `from_dict()`, que verifica la identidad y todos los campos derivados (A1E2-R1). El campo heredado `orientation_mode` no gobierna la tarea; la orientación se pide en `OrientationRequirement`.

| Campo | Tipo | Regla |
|---|---|---|
| `tool_id` | `str` | No vacío. |
| `collision_polygon` | `(n,2)` | Válido, simple, área > 0, finito; copia propia de solo lectura. |
| `cutting_polygon` | `(m,2)` | Ídem, **cubierto exactamente** por el de colisión almacenado (sin tolerancia; contacto de bordes admitido, protrusión no) (R11). |
| `pivot` | `Point2` | En el marco `B` y **exactamente** `(0, 0)` (R5); el offset de la cuchilla va en los polígonos. En `rectangular`, `pivot_offset` exige forma exacta `(2,)` y componentes finitos (N1). |
| `orientation_mode` | `OrientationMode` | `fixed` en H1. |
| `domain` | `SpatialDomain` | Común con el pivote. |

Colisión y corte **pueden diferir**: la primera gobierna la no colisión, la segunda
la cobertura. `circumscribed_radius()` se mide desde el origen de `B`, el mismo
respecto del cual se rota la huella, y es la aproximación conservadora admitida en
H1 para inflar obstáculos; `inflated_collision_shape(d)` infla la huella con
esquinas en inglete (contiene el buffer circular exacto) cuando la política es
`inflate: tool`; el validador final usa el polígono completo, nunca solo
el centro o los vértices.

### 4.9 `ToolState` — `simulation.tool`

| Campo | Tipo | Regla |
|---|---|---|
| `time_s` | `float` | ≥ 0, finito. |
| `position` | `Point2` | En `V` o `W`; es la posición del **pivote**. |
| `theta`, `theta_rate` | `float` | Finitos. |
| `velocity`, `acceleration` | `(float, float)` | **Exactamente** 2 componentes finitos, validados antes de indexar (R9). |
| `contact` | `bool` | Booleano real (`bool`/`numpy.bool_`); cadenas y enteros se rechazan (R9). |
| `machine_state` | `SimulationState` | `IDLE`, `WORKING`, `RETREATING`, `ABSTAINED`, `UNSAFE_STATE`. |
| `depth_z` | `None` | Reservado; **debe** ser `None` en H1. |

### 4.10 `SyntheticScene` — `scenarios.synthetic`

| Campo | Regla |
|---|---|
| `scene_id` | No vacío. |
| `frame`, `domain` | Únicos para corredor, plantas, exclusiones e incertidumbre (R3); nunca `ground_metric_verified` (R2). |
| `provenance` | Siempre `synthetic`, también en cada planta (R2). |
| `plants`, `exclusions`, `notes` | Tuplas propias aunque se reciban listas; tipos verificados (R4). |
| correspondencia | Exactamente una exclusión por planta, sin duplicados ni huérfanas, con el mismo centro y `protection_radius` ≥ el de la planta (R4). |
| `uncertainty` | `UncertaintyPolicy`; la protección aplicada (margen de exclusión o `tool_inflation`) cubre `uncertainty.total`, en un solo lugar (R1). |
| `tool_inflation` | ≥ 0; 0 con `inflate: exclusions`. |
| `free_space_certified` | `True` **solo** con lista de plantas vacía; una escena sin plantas y sin certificado se rechaza (R4). |
| `advance_speed`, `advance_heading` | Finitos; velocidad > 0 (R4). |
| `seed` | Determinismo: misma semilla ⇒ escena idéntica. |

`target_region()` es el corredor evaluable menos las exclusiones de referencia:
es el denominador fijo de cobertura y **no** depende de decisiones del
planificador (§6). Si su área es 0, la cobertura se reporta `null` con razón
`NO_TARGET_AREA`, nunca 100 % automático.

**Una lista vacía de plantas no certifica suelo libre.** La afirmación requiere
`free_space_certified=True`, y la configuración rechaza el caso contrario.

### 4.11 Enumeraciones de resultado (estables)

| Enum | Valores |
|---|---|
| `PlanStatus` | `ACCEPTED`, `ABSTAIN`, `INVALID_INPUT`, `UNCERTIFIED`, `UNSUPPORTED` (A1: capacidad no implementada, distinta de una imposibilidad). |
| `AbstentionReason` | `NO_PATH_FOUND`, `INSUFFICIENT_LOOKAHEAD`, `UNKNOWN_SCALE`, `STALE_MAP`, `EXCESS_UNCERTAINTY`, `LIMIT_VIOLATION`, `COLLISION`, `NO_SAFE_TRANSITION`, `NO_TARGET_AREA`; desde A1/E2, además: `UNSUPPORTED_FEATURE`, `START_INADMISSIBLE`, `GOAL_INADMISSIBLE`, `UNKNOWN_SPACE`, `CERTIFICATION_BUDGET_EXHAUSTED`. |
| `SimulationState` | `IDLE`, `WORKING`, `RETREATING`, `ABSTAINED`, `UNSAFE_STATE`. |

`NO_PATH_FOUND` significa que el algoritmo no encontró ruta; no es una prueba
universal de imposibilidad. La abstención es un resultado del dominio, no una
excepción de Python.

## 4 bis. Contratos de video implementados en E0 — `precision_weeder.video`

### `VideoProbe` y `DecodedFrame` — `video.reader`

`VideoProbe` contiene los metadatos **declarados**: contenedor, códec y tag, perfil,
tamaño, duración, número de cuadros, FPS nominal y su fuente, `time_base`, flujos
adicionales y metadatos. Un campo no expuesto vale `None`; nunca 0 ni un valor
inferido de la extensión.

`DecodedFrame` cumple el papel de `FramePacket`: índice, `pts`, `dts`, `time_base`,
`timestamp_s = pts·time_base`, tipo de imagen, clave, tamaño, rotación declarada y,
opcionalmente, el array de píxeles.

`DecodeReport` registra cuadros decodificados, `eof_status`
(`eof_reached`/`decode_error`), error, advertencias capturadas de FFmpeg, tamaños
observados, cuadros sin PTS, `truncation_suspected` y la limitación de
detectabilidad del truncamiento.

### `TimestampReport` — `video.timing`

Separa dos cosas (revisión E0-R1):

- **Diagnóstico de los PTS observados.** Incluye `pts_observed`/`pts_missing`, la
  monotonía (`None` con menos de dos PTS), los duplicados, regresiones y saltos con
  su umbral, y la distribución de intervalos (mín., p05, mediana, media, p95, máx.,
  desviación, pares comparables).
  - Duplicados y regresiones se buscan entre PTS observados consecutivos, aunque
    medie un cuadro sin PTS.
  - Saltos y estadísticas solo se calculan entre cuadros adyacentes con PTS.
  - Cada ausencia genera una fila `missing_pts` (`frame_index`).
  - Toda anomalía produce su fila, también cuando no hay intervalos positivos.
- **Línea temporal usada** (`timeline_s`, origen en `source`: `pts` o
  `index_fps_estimate`, siempre etiquetado). La estimación por índice no sustituye
  al diagnóstico ni demuestra monotonía original.

El FPS efectivo y la duración observada solo se certifican con PTS completos,
monótonos y con intervalos positivos. En otro caso valen `None` y su método empieza
por `no certificado`. Se rechazan con error explícito:

- PTS no finitos o no numéricos, aunque falten otros;
- `fps_nominal` no finito o ≤ 0;
- `gap_factor` ≤ 1 o no finito;
- PTS ausentes sin estimación autorizada.

`timestamps.csv` publica `timestamp_observed_s` (PTS) junto a `timestamp_s` y
`timestamp_method`. Los candidatos a imagen casi repetida con PTS regular se
registran aparte como `content_repeat_candidate`
(`video.frame_metrics.detect_content_repeats`): no certifican un duplicado de
captura.

### `FlowEstimate` — `video.apparent_motion`

Contiene los vectores detectados, válidos y rechazados, separados en suelo y
vegetación. Para el suelo da medianas, MAD, módulo, fracciones por sentido y
dirección en imagen (`up/down/left/right/mixed/none/unknown`). Las unidades son
píxeles del cuadro original por paso de cuadros. **Nunca** se traduce a avance
físico; para eso existe `geometry.frames.advance_sign_under_declared_mounting`,
que exige un montaje declarado. `affine_flow_fit` informa los gradientes del campo
como descriptor. Los gradientes no nulos solo muestran que el flujo no es una
traslación uniforme: un giro con escala uniforme también los produce. No determinan
inclinación ni calibración, ni descartan una escala uniforme.

### `VideoManifestRecord` — `video.manifest`

Tiene las 32 columnas de la orden E0 §6, en ese orden. `unknown` es el único token
de valor desconocido, y 0 se rechaza donde significaría «desconocido». Las rutas
son relativas, sin `..`, y resuelven contra la raíz. Los hashes son SHA-256 en
minúscula. A/B/C exigen motivo y evidencia; A exige además bases observadas y
cuadros decodificados. `pending` es el único estado sin evaluación. `verified`
exige evidencia de escala. La escritura es determinista, ordenada por `video_id`.

### `StemBaseObservation`, `ManualAssessment`, `CrossingLine` — `video.observations`

- **Observación de base.** `source` es siempre `manual_visual`.
  - `observed_target` declara qué punto marca la fila:
    - `ground_contact`: contacto tallo-suelo;
    - `stem_axis_proxy`: eje o copa visto desde arriba, a altura desconocida; no
      equivale a una base en el suelo;
    - `none`: no se localizó ningún punto.

    El anotador lo fija; no se infiere del texto libre.
  - Visibilidad `visible`/`partial`/`hidden` del punto anotado. `hidden` exige
    `observed_target = none` y no lleva posición (y viceversa). Los demás valores
    exigen `u_px`, `v_px` y una incertidumbre > 0.
  - Tiempo con método (`pts`/`index_fps_estimate`) y confianza `high/medium/low`.
  - Se rechazan IDs duplicados y la misma planta repetida en un mismo cuadro.
- **Evaluación manual, esquema 2.** Es la única vía para fijar visibilidad, montaje,
  escala, homografía, temporalidad y aptitud.
  - Se vincula al clip por `video_id` y `video_sha256`, y a su tabla por
    `stem_observations_sha256`.
  - La versión 1 se rechaza con instrucciones de migración explícitas: añadir los
    dos hashes y la columna `observed_target`.
  - En E0 no puede declarar escala `verified` ni homografía `validated`.
  - `before_intervention` y una anticipación validable exigen conocer la posición
    entre cámara y herramienta.
  - La aptitud A exige ≥ 3 plantas con `ground_contact` en ≥ 2 cuadros, una línea
    virtual de cruce y decodificación hasta EOF. Los proxies no cuentan.
  - Una discrepancia entre la dirección manual y la automática es un error, no se
    elige una en silencio.
- **Validación contra el clip** (`validate_against_clip` con `ClipFacts`). La usan
  el caracterizador y el renderer **antes de escribir nada**, y comprueba:
  - ID y SHA-256;
  - índices de cuadro existentes;
  - `timestamp_s` igual al tiempo del cuadro decodificado, dentro de
    `timestamps.annotation_tolerance_s` (1e-6 s, para el redondeo a 6 decimales del
    CSV). La tolerancia debe ser menor que medio intervalo entre cuadros;
  - el método temporal;
  - posiciones de observaciones y de contrastes de flujo dentro de la imagen;
  - línea virtual dentro de la imagen;
  - intervalos dentro del clip y sin solaparse, y cuadros de condiciones
    existentes.

  `ManualAssessment.load` solo valida esquema, tabla y la regla A: no basta para
  publicar.
- **Línea de cruce.** Es horizontal en píxeles. El cruce se obtiene por
  interpolación lineal entre observaciones consecutivas del mismo objetivo, sin
  mezclar objetivos, o se marca `not_bracketed`. Cada evento publica
  `observed_target`, el intervalo de soporte (`support_from_s`/`support_to_s`) y
  `time_uncertainty_s`, la propagación lineal de la incertidumbre de lectura en
  `v`. Es una interpolación de anotaciones, no un instante físico medido.

## 4 ter. Contratos de tarea implementados en A1/E2

Su decisión de base es el [ADR 0001](adr/0001_task_space_planning.md).

- **Identidad (A1E2-R1):** los seis contratos (`LocalMap`, `ToolGeometry`,
  `TaskPath`, `PlanRequest`, `PlanResult` y `ValidationReport`) llevan
  `schema_version` y un `content_hash`, el SHA-256 del JSON canónico de
  `core.identity` (claves ordenadas, sin NaN/inf). Cambiar la geometría, el mapa o
  una tolerancia cambia el hash.
- **Reconstrucción:** cada uno tiene un `from_dict` tipado que rechaza por su causa
  una versión, unidad, campo numérico, estado o hash alterados.
- **Propiedad de datos:** los certificados terminados son inmutables.
  `core.identity.freeze` congela sus colecciones (mapas de solo lectura y tuplas) y
  `to_dict` exporta copias profundas independientes. Ni modificar una exportación ni
  la entrada original altera el artefacto construido.

**Capacidades del perfil E2.** Fuera de ellas, planificador y validador responden
`UNSUPPORTED_FEATURE`, que es una capacidad no implementada y no una imposibilidad
geométrica:

- orientación `FIXED`;
- huella de colisión **y** huella de corte convexas (A1E2-R2);
- rol `pasada_principal` (`core.trajectory.E2_SUPPORTED_ROLES`, A1E2-R3);
- `working_contact = true` y `task_tolerance_m = 0`.

El contrato sigue admitiendo otros roles para A3. El constructor de `ToolGeometry`
de E1 sigue aceptando cortes cóncavos: el rechazo es de capacidad, no de esquema.

### `TaskPath`, `PathSegment`, `OrientationRequirement` — `core.trajectory`

| Campo | Regla |
|---|---|
| `frame`, `domain` | Marco `W`. El dominio es métrico de suelo (`ground_metric_synthetic`, `ground_assumed` o `ground_metric_verified`, que son modos separados); `image_px` y `graphic_unscaled` se rechazan (`UnitMismatchError`). |
| `segments` | `PathSegment`s que particionan `s ∈ [0, 1]` sin huecos ni solapes: cada `s_end` es exactamente el `s_start` siguiente. Posición continua en las uniones (≤ 1e-9 m). |
| `PathSegment` | Intervalo `[s_start, s_end]` no vacío. Tiene 6 coeficientes por coordenada en base de potencias ascendente sobre `u = (s − s_start)/L` en `[0, 1]`, en copias propias no modificables. `d^k p/ds^k = (d^k p/du^k)/L^k`. |
| `orientation` | `FIXED` (θ en rad, en `W`), `INTERVAL` (límites ordenados) o `FREE` (sin θ; no significa cero). |
| `profile_id/hash`, `map_id/hash` | Identidad del perfil y del mapa. El validador rechaza otra combinación. |
| `role`, `working_contact` | `pasada_principal` y contacto de trabajo (booleano). |
| `task_tolerance_m` | En E2 vale 0: se valida y se entrega la curva exacta. |
| `continuity_declared` | `C0`, `C1` o `C2`. El validador la comprueba; E2 entrega `C2`. |

`s` es progreso adimensional: **no** es tiempo ni la coordenada `y`, y sus
derivadas no son velocidades. No hay `q`, tiempos ni identidad de mecanismo.

### `LocalMap` — `mapping.local_map`

| Campo | Regla |
|---|---|
| `map_id`, `version`, `segment_id` | Identidad. Dos `W` de orígenes distintos no son el mismo marco. |
| `corridor` | `Corridor` en `W`. |
| `known_region`, `known_region_basis` | Anillo del suelo declarado conocido y la razón de esa declaración (p. ej. `synthetic_ground_truth`, `certified_free`). No se infiere de la ausencia de plantas. |
| `unknown_regions` | Anillos de suelo desconocido interior; para la huella son **prohibidos**. |
| `exclusions` | `CircularExclusion` en `W`, ya con su margen efectivo. |
| `uncertainty`, `tool_inflation` | Política E1: infla exclusiones **o** herramienta, nunca ambas, y cubre la cota declarada. |
| `free_space_certified` | Obligatorio si no hay exclusiones; incompatible con exclusiones. |
| `static`, `valid_from_s/valid_to_s` | Un fixture estático no lleva vigencia; uno no estático exige una ventana no vacía. |

El dominio permitido es `corredor ∩ región_conocida − desconocido`, y es cerrado.
La región objetivo de cobertura es el dominio permitido menos las exclusiones.

### `PlanRequest`, `PlanResult`, `CandidateRecord` — `planning.base`

- **`PlanRequest`:** identidades de mapa y perfil, inicio y meta del origen `B`
  (`Point2` en `W`, dominio métrico), `OrientationRequirement` y rol. Exige
  `goal.y > start.y` (progreso). No admite `ToolState`, `q`, mecanismo, video ni
  calibración.
- **`PlanResult`:**
  - Contiene estado, razón, detalle, hashes de solicitud, mapa, perfil y
    configuración, `TaskPath` (**solo** si `ACCEPTED`, y siempre en ese caso), el
    informe de validación, el candidato elegido, los candidatos rechazados como
    diagnóstico y estadísticas de búsqueda.
  - El tiempo de pared va aparte (`to_dict(include_timing=False)` es
    determinista).
  - `ACCEPTED` exige un informe `ACCEPTED`, y lleva `validation_scope=geometric` y
    `mechanical_feasibility=not_evaluated`.
  - **Ligadura (A1E2-R1):** el informe se reconstruye y verifica (versión,
    coherencia de estado y `content_hash`) y debe referirse a **esta** ruta, mapa y
    perfil. Sustituir la ruta, o presentar un informe de otra ruta, incompleto,
    rechazado o no certificado, se rechaza. Un resultado no aceptado no puede llevar
    un informe `ACCEPTED`.
  - **Dos hashes de configuración distintos a propósito:** `config_hash` del
    resultado (`planner + validation + safety`) y `config_hash` del informe
    (`validation + safety`). Cada exportación declara su alcance en
    `config_hash_scope`.
- **Correspondencia de estados:**

  | Situación | Estado y razón |
  |---|---|
  | Entrada inválida | Excepción tipada en la API; código 2 en el CLI |
  | Capacidad no implementada | `UNSUPPORTED` / `UNSUPPORTED_FEATURE` (código 2) |
  | Inicio o meta en colisión o fuera del dominio | `ABSTAIN` / `START_INADMISSIBLE` o `GOAL_INADMISSIBLE` |
  | Sin candidatos | `ABSTAIN` / `NO_PATH_FOUND`, o `UNKNOWN_SPACE` si solo cruzando desconocido habría candidatos |
  | Candidatos rechazados por colisión | `ABSTAIN` / `COLLISION` |
  | Candidatos no certificados | `UNCERTIFIED` / `CERTIFICATION_BUDGET_EXHAUSTED` |

  Los cuatro últimos casos devuelven código 3 en el CLI.

### `ValidationReport` — `validation.swept_volume`

- **Identidad:** hashes de ruta, mapa, perfil y configuración (`validation` +
  `safety`). El esquema v2 es inmutable, con `from_dict` y `content_hash`.
- **Estado:** `ACCEPTED`, `REJECTED` o `UNCERTIFIED`.
- **Diagnóstico:** violaciones con evidencia (tipo, `s`, planta, distancia), e
  intervalos no certificados con su causa (`max_subdivision_depth` o
  `max_intervals`).
- **Envolvente:** método, máximo `delta` usado y presupuesto declarado.
- **Despeje:** `clearance_lower_bound_m`, una **cota inferior certificada**, no una
  distancia exacta.
- **Comprobaciones:** continuidad (saltos máximos de p, dp/ds y d²p/ds²), progreso
  (mínimo analítico de dy/ds por tramo) y extremos.
- **Cobertura:** exacta en tramos rectilíneos con corte convexo. En curvas es una
  estimación rotulada: barridos de cuerdas a ≤ `chord_deviation_max_m` de la curva,
  **sin** cota de área garantizada (`error_bound_m2 = null`) (A1E2-R2).
- **Presupuesto de discretización (A1E2-R4):** antes de validar, cada exclusión
  efectiva se contrasta con `safety.max_conservative_discretization_excess_m`
  mediante la regla única de E1 (`discretization_excess_violation`: se rechaza si
  `r·(sec(π/n)−1) > límite`). Superarlo es una entrada incompatible (`ConfigError`),
  no una ausencia de ruta. El planificador aplica la misma comprobación antes de
  buscar, y el CLI sale con 2 sin escribir artefactos.
- **Geometrías y aproximaciones:** las geometrías efectivas que llegaron al
  validador (huellas de colisión y de corte orientadas, `tool_inflation`, radios de
  exclusión) y las aproximaciones (exceso de discretización y buffer en inglete).

## 4 quater. Contratos A3/E3 implementados (paso 1 de la orden A3/E3)

Implementan parte de §5 bis. El compilador, `ExecutableTrajectory`, `ExecutionTrace`,
`InfeasibilityReport` completo y `ValidatedExecution` siguen pendientes (pasos 2–3).

| Contrato | Módulo | Contenido e invariantes |
|---|---|---|
| `TimePolynomial` | `core.temporal` | `f(t)` vectorial polinómica por tramos (grado ≤ 5, base ascendente en `u` local); nodos estrictamente crecientes; continuidad declarada C0/C1/C2 comprobada en uniones (1e-9 / 1e-6 / 1e-6); sin extrapolación fuera de `[t_start, t_end]`; extremos exactos de cualquier derivada por raíces, no por muestreo. |
| `TaskTrajectory` | `core.temporal` | Solo `TIMED_REFERENCE`. `TaskPath` E2 íntegro + hash, ley `s(t)` escalar en `[0,1]` y no decreciente en todo el intervalo, reloj, `WorkRequirement` (avance y cobertura mínimos, región objetivo en W fijada antes), tubo `tolerance_m`. `p_d`, `ṗ_d`, `p̈_d` por regla de la cadena. Hash sensible a camino, mapa, perfil, ley, reloj, trabajo y tolerancia. |
| `VehicleMotion` | `core.temporal` | `T_WV(t) = (x_V, y_V, ψ)` con derivadas exactas, reloj, horizonte, cotas de incertidumbre, origen referencia/predicción. `to_vehicle` incluye traslación y rotación del vehículo en las derivadas. Dominios de píxel o gráficos rechazados. |
| `TaskBinding` / `RoleBinding` | `core.temporal` | Rol → `tool_id` + hash del perfil; política `exclusive`/`simultaneous`/`sequence`. Solo un rol exclusivo es resoluble por el compilador inicial; lo demás será `UNSUPPORTED`. |
| `CompileStatus`, `InfeasibilityCode` | `core.compilation` | Estados y códigos de §5 bis. |
| `MechanismModel` | `kinematics.base` | Esquema de juntas (tipo, unidades, topes, límites de referencia), cuerpos de espesor finito por `body_id` con huella de corte para herramientas, pares estructurales permitidos explícitos, `forward(q)` → poses en V, `validate_configuration`, `inverse` opcional (`UNSUPPORTED` por defecto), identidad por hash de todos los parámetros. `dim(q)` independiente del número de herramientas. |
| `SinglePivotSwing` | `kinematics.models.single_pivot_swing`, `configs/mechanism.yaml` | Baseline 1R (§5 de la adaptación): FK de brazo y cuchilla, IK exacta sobre el arco con rama continua desde semilla y topes, `RADIAL_REACH`/`JOINT_LIMIT`/`ORIENTATION_CONFLICT` con certificado analítico, `q̇`/`q̈` desde `r`, `ṙ`, `r̈`. `L = 0` es configuración inválida. |

## 5. Contratos especificados, pendientes de su etapa

| Contrato | Campos mínimos | Etapa |
|---|---|---|
| `FramePacket` | `clip_id`, índice, `pts`, `time_base`, `timestamp_s`, fuente temporal, tamaño, fotograma. | E0/E4 |
| `StemObservation` | ID de observación, frame, `(u,v)`, covarianza 2×2 o cota, visibilidad, confianza, fuente, calidad de anotación. | E4 |
| `Calibration` | ID, intrínsecos/distorsión si existen, homografía, evidencia de escala, error en puntos independientes, dominio, estado. | E4 |
| `PlantTrack` | `track_id`, tiempo, posición, marco/unidad, covarianza/cota, último avistamiento, estado, procedencia. | E5 |

### 5 bis. Contratos de compilación documentados en A1 (sin implementar)

Su implementación real es A3. No existe un compilador ficticio: hasta entonces,
ninguna salida E2 se presenta como ejecutable. Esquemas tomados de la
[adaptación arquitectónica](architecture_adaptation_plan.md) §4.

| Contrato | Campos y estados |
|---|---|
| `TaskTrajectory` | Unión discriminada por modo temporal. **`TIMED_REFERENCE`** (primero): referencia a un `TaskPath` por hash, ley `s(t)` explícita con reloj y horizonte, y restricciones; el compilador no cambia tiempos. `PATH_WITH_WINDOWS` (extensión): `p(s)` con ventanas temporales. Versión/ID, mapa y perfil versionados, tolerancia de tarea (tubo), roles y condición mínima de trabajo. |
| `TaskBinding` | Rol → herramienta `B_i` y perfil compatible; exclusividad, simultaneidad o secuencia explícitas. Si no se soporta: `UNSUPPORTED`. |
| `MechanismModel` | `configuration_schema` (articulaciones, tipo, unidades, límites, acoplamientos), `validate_configuration(q, assembly_mode)`, `forward(q, assembly_mode)` → poses por `body_id`/`tool_id` en `V`; opcionales `inverse_seeds` y `jacobian`. ID, versión y hash de los parámetros. |
| `TrajectoryCompiler` | Entradas: `TaskTrajectory`, `TaskBinding`, `VehicleMotion`, modelo, límites, estado inicial y latencias. Estados: `FEASIBLE_CANDIDATE`, `INFEASIBLE_PROVEN`, `NOT_FOUND_WITHIN_LIMIT`, `UNSUPPORTED`, `INVALID_INPUT`, `UNCERTIFIED`. |
| `InfeasibilityReport` | Versión, estado, código (`RADIAL_REACH`, `ORIENTATION_CONFLICT`, `JOINT_LIMIT`, `VELOCITY_LIMIT`, `ACCELERATION_LIMIT`, `CLOSURE_ERROR`, `BRANCH_DISCONTINUITY`, `COLLISION`, `INSUFFICIENT_LOOKAHEAD`, `UNKNOWN_SPACE`), etapa, rol/cuerpo/junta, intervalo en `t`/`s`, valor y límite, residuo, método, alcance del certificado, presupuesto, hashes y condición inicial. |

## 6. Reglas transversales de validación

Todos los esquemas rechazan:

1. Valores no finitos (NaN, ±inf) — `NonFiniteValueError`.
2. Dimensiones nulas o negativas — `InvalidDimensionError`.
3. Polígonos vacíos, degenerados, autointersectados o de área no positiva — `InvalidPolygonError`.
4. Tiempos no finitos, negativos o decrecientes — `TemporalOrderError`.
5. Mezcla de unidades o dominios — `UnitMismatchError`.
6. Cadenas de marcos incompatibles — `FrameMismatchError`.
7. Covarianzas no simétricas o no semidefinidas positivas — `UncertaintyBudgetError`.
8. Resultados métricos verificados sin calibración — `MissingCalibrationError`.
9. Claves desconocidas en configuración — `UnknownConfigKeyError`.

## 7. Serialización

`SyntheticScene.to_json()` es determinista (claves ordenadas, ASCII) e incluye
`unit`, `domain`, `is_metric`, `metric_scale_verified` (siempre `false` en escenas
sintéticas), `provenance`, `seed`, `target_area`, la política de incertidumbre con
los márgenes y la inflación **efectivamente aplicados**, el exceso de
discretización por exclusión y las notas de la escena. La demo añade la
procedencia de la configuración (ruta y SHA-256 de cada YAML). `ProjectConfig.resolved_yaml()` materializa
**todos** los valores por defecto. CSV y JSON bastan para el MVP.
