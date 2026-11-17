# Registro de supuestos

**Fecha de corte:** 24 de octubre de 2026. **Etapa:** E1 (con correcciones R1–R12
del dictamen `docs/reviews/E1_review.md`). **Estado:** vivo; cada etapa posterior
añade o corrige filas.

Este registro cumple `docs/technical_plan.md` §1.3. Cada parámetro declara valor,
unidad, procedencia, incertidumbre y responsable de validarlo.

**Procedencias** (`src/precision_weeder/core/units.py::Provenance`):

| Valor | Significado |
|---|---|
| `measured` | Medido físicamente con instrumento y procedimiento registrados. |
| `synthetic` | Elegido para construir el banco sintético; no describe el mundo. |
| `assumed` | Hipótesis de trabajo plausible, sin medición que la respalde. |
| `unknown` | Falta evidencia; no se sustituye por un número inventado. |

> **Ningún parámetro de este repositorio tiene procedencia `measured` a día de
> hoy.** No se han inventado valores medidos para cerrar decisiones abiertas.

## 1. Escena sintética (`configs/synthetic.yaml`)

| Parámetro | Valor | Unidad | Procedencia | Incertidumbre | Responsable de validar |
|---|---|---|---|---|---|
| `scene.corridor.width_m` | 1.0 | m | `synthetic` | n/a (definición del banco) | Responsable agronómico + mecánico |
| `scene.corridor.length_m` | 5.0 | m | `synthetic` | n/a | Responsable del banco |
| `scene.plants.positions_y_m` | 1.0, 2.0, 3.0 | m | `synthetic` | n/a | — |
| `scene.plants.stem_radius_m` | 0.02 | m | `synthetic` | desconocida en campo | Responsable agronómico |
| `scene.exclusion.protection_radius_m` | 0.15 | m | `synthetic` | **criterio agronómico pendiente** | Responsable agronómico |
| `scene.exclusion.uncertainty_margin_m` | 0.03 | m | `assumed` | cota, no desviación típica | Percepción + calibración |
| `scene.exclusion.polygon_segments` | 64 | — | `synthetic` | exceso radial **conservador** 2.171e-4 m para r = 0.18 m (§3) | Desarrollo |
| `scene.advance.speed_mps` | 0.20 | m/s | `assumed` | desconocida; es **entrada**, no variable libre | Operación |
| `scene.seed` | 20261024 | — | `synthetic` | determinista | Desarrollo |

El radio de protección de 0.15 m **no** es una dimensión agronómica validada: la
protección de raíces requiere criterios todavía pendientes (§1.1). El valor
existe para que el banco geométrico tenga una exclusión no trivial.

## 2. Herramienta (`configs/tool.yaml`)

| Parámetro | Valor | Unidad | Procedencia | Incertidumbre | Responsable de validar |
|---|---|---|---|---|---|
| `tool.length_x_m` | 0.20 | m | `synthetic` | n/a | Diseño mecánico |
| `tool.width_y_m` | 0.08 | m | `synthetic` | n/a | Diseño mecánico |
| `tool.pivot_offset_m` | (0, 0) | m | `synthetic` | n/a | Diseño mecánico |
| `tool.cutting_inset_m` | 0.005 | m | `assumed` | n/a | Diseño mecánico |
| `tool.orientation_mode` | `fixed` | — | `synthetic` | decisión de alcance H1 (§3.3) | Arquitectura |
| `tool.depth_z_m` | `null` | m | `unknown` | interfaz reservada (§2) | Diseño mecánico |

La huella de corte se modela 0.005 m hacia dentro de la de colisión únicamente
para ejercitar que ambas geometrías pueden diferir. No describe un filo real.

## 3. Seguridad e incertidumbre (`configs/safety.yaml`)

| Parámetro | Valor | Unidad | Procedencia | Incertidumbre | Responsable de validar |
|---|---|---|---|---|---|
| `clearance_epsilon_m` | 1e-4 | m | `synthetic` | tolerancia de software (§6) | Desarrollo |
| `geometric_approximation_tolerance_m` | 1e-5 | m | `synthetic` | presupuesto de discretización | Desarrollo |
| `limit_comparison_tolerance` | 1e-8 | — | `synthetic` | comparación numérica | Desarrollo |
| `max_conservative_discretization_excess_m` | 5e-4 | m | `synthetic` | cota del exceso de polígonos circunscritos (R12) | Desarrollo |
| `uncertainty.localization_bound_m` | 0.01 | m | `assumed` | sin medición de percepción | Percepción (E6) |
| `uncertainty.calibration_bound_m` | 0.01 | m | `assumed` | sin calibración (E4) | Calibración |
| `uncertainty.tracking_bound_m` | 0.01 | m | `assumed` | sin seguimiento (E5) | Seguimiento |
| `uncertainty.bound_budget_m` | 0.03 | m | `assumed` | igual a la suma de cotas | Arquitectura |
| `uncertainty.sigma_multiplier_k` | 3.0 | — | `assumed` | **modelo, no garantía probabilística** | Percepción |
| `uncertainty.inflate` | `exclusions` | — | `synthetic` | el error infla exclusiones **o** herramienta, nunca ambas (§3.3) | Arquitectura |

Las tolerancias numéricas no son márgenes agronómicos. `k = 3` no implica una
cobertura del 99,7 %: esa afirmación exigiría validar la distribución.

**Política de incertidumbre (R1).** `safety.uncertainty` es la única fuente de las
cotas. Su suma `u = 0.01 + 0.01 + 0.01 m` es la protección mínima exigida y
`bound_budget_m` es solo el **máximo permitido** para `u`, no el margen que se
infla. Con `inflate: exclusions`, `scene.exclusion.uncertainty_margin_m` debe ser
≥ `u` (puede ser mayor); con `inflate: tool`, ese margen debe ser 0 y la escena
registra `tool_inflation = u`. Nunca se inflan ambas geometrías. Estas cotas no
incluyen el error de ejecución del actuador, que se modelará por separado en E3.

**Tolerancias y discretización (R12).** Se mantienen tres magnitudes separadas:

| Magnitud | Valor | Gobierna |
|---|---|---|
| `clearance_epsilon_m` | 1e-4 m | Despeje estricto exigido; contacto y tangencia se rechazan. |
| `geometric_approximation_tolerance_m` | 1e-5 m | Errores de aproximación que **podrían subestimar** una geometría (barrido futuro, buffers interiores). La aproximación de exclusiones es exterior: su error de subprotección es 0. |
| `max_conservative_discretization_excess_m` | 5e-4 m | Exceso **conservador** de los polígonos circunscritos: `r·(sec(π/n) − 1)`. Para r = 0.18 m y n = 64 vale 2.171e-4 m. Cumplir 1e-5 m exigiría n ≥ 299; no se afirma ese cumplimiento. |

Se exige `limit_comparison_tolerance < geometric_approximation_tolerance_m <
clearance_epsilon_m`. Efecto del exceso conservador: el hueco del caso
`narrow_passage` medido entre polígonos es 0.22 − 2·2.171e-4 ≈ 0.21957 m, y el área
objetivo es ligeramente menor que la analítica; ambas desviaciones van del lado
seguro y reducen cobertura, no protección.

## 4. Cinemática y latencias (`configs/kinematics.yaml`)

| Parámetro | Valor | Unidad | Procedencia | Incertidumbre | Responsable de validar |
|---|---|---|---|---|---|
| `position_limits_m.x_min/x_max` | ∓0.35 | m | `assumed` | sin mecanismo definido | Diseño mecánico |
| `position_limits_m.y_min/y_max` | ∓0.20 | m | `assumed` | sin mecanismo definido | Diseño mecánico |
| `velocity_limits_mps.x / .y` | 0.60 / 0.40 | m/s | `assumed` | sin actuador definido | Diseño mecánico |
| `acceleration_limits_mps2.x / .y` | 3.0 / 2.0 | m/s² | `assumed` | sin actuador definido | Diseño mecánico |
| `latencies.acquisition_s` | 0.03 | s | `assumed` | sin cámara elegida | Percepción |
| `latencies.inference_s` | 0.05 | s | `assumed` | sin modelo entrenado | Percepción |
| `latencies.planning_s` | 0.02 | s | `assumed` | sin medición de cómputo | Desarrollo |
| `latencies.communication_s` | 0.01 | s | `assumed` | sin bus definido | Electrónica |
| `latencies.mechanical_response_s` | 0.08 | s | `assumed` | sin actuador definido | Diseño mecánico |
| `horizon_s` | 2.0 | s | `assumed` | n/a | Arquitectura |
| `start_state` | (0, 0), sin contacto, estacionada | m | `synthetic` | n/a | Arquitectura |

`x_max = 0.35 m` se eligió para que exista **al menos un caso factible**: con
exclusión total de 0.18 m y media cuchilla de 0.10 m, el pivote necesita alcanzar
0.28 m. No se modificó ningún caso adverso para que aprobara.

## 5. Escena gráfica del SVG (`.context/descarga.svg`)

| Parámetro | Valor | Unidad | Procedencia | Nota |
|---|---|---|---|---|
| `viewBox` | 340 × 330 | unidad gráfica | `synthetic` (lectura directa) | Sin escala física ni metadatos de calibración. |
| Corredor | x ∈ [90, 250], y ∈ [0, 330] | unidad gráfica | `synthetic` | Franja conceptual; no determina alcance mecánico. |
| Centros de planta | (169, 65), (169, 165), (169, 265) | unidad gráfica | `synthetic` | Separación gráfica de 100 unidades. |
| Radio de protección | 32 | unidad gráfica | `synthetic` | **No** son centímetros ni dimensión agronómica. |
| Radio de tallo (centro) | 4.5 | unidad gráfica | `synthetic` | Círculo central del dibujo. |
| Cuchilla | 30 × 10 en (104, 243) | unidad gráfica | `synthetic` | Motiva huella finita; no define articulación. |
| Pivote | (119, 248) | unidad gráfica | `synthetic` | Centro del rectángulo dibujado. |
| Curva Bézier | no importada | — | — | No es una trayectoria aprobada ni certificada. |

El dominio de esta escena es `graphic_unscaled`: las transformaciones a metros
están prohibidas por contrato y hay un test que lo comprueba.

## 6. Decisiones pendientes (sin valor asignado)

| Tema | Estado | Qué desbloquearía |
|---|---|---|
| Dimensiones reales de cuchilla y mecanismo | `unknown` | Diseño mecánico documentado. |
| Radio de raíces a proteger | `unknown` | Criterio agronómico escrito. |
| Velocidad real de avance y su variabilidad | `unknown` | Medición en campo o especificación de operación. |
| Óptica, montaje y geometría cámara–herramienta | `unknown` | Ficha de montaje; condiciona anticipación física. |
| Orientación del eje `v` de la cámara sobre el suelo | `unknown` | Ficha de montaje. Sin ella, el desplazamiento aparente «hacia arriba» de los videos **no** determina el sentido de avance (R7): `advance_sign_under_declared_mounting` rechaza `unknown`. |
| Disponibilidad y aptitud de videos | `unknown` | E0; no existen clips en el repositorio a esta fecha. |
| Escala métrica verificada | `unknown` | E4: homografía con referencia métrica independiente. |
| Latencias reales del lazo | `unknown` | Medición con hardware. |

Mientras estas filas sigan en `unknown`, ninguna salida del banco puede
etiquetarse como anticipación física validada ni como MVP físico completo.

## 7. Caracterización de video E0 (`configs/video.yaml`)

Parámetros de **análisis exploratorio** en píxeles. No describen la escena ni la
cámara: deciden cómo se mide. Ninguno es una medición física.

| Parámetro | Valor | Unidad | Procedencia | Nota |
|---|---|---|---|---|
| `roi` | u ∈ [0, 1], v ∈ [0, 0.97] | fracción del cuadro | `assumed` | Excluye una franja oscura de pocos píxeles, no identificada, del borde inferior. |
| `sampling.uniform_count` | 12 | muestras | `synthetic` | Mínimo de la orden E0 §3. |
| `sampling.analysis_scale` | 0.25 | — | `assumed` | Métricas y flujo sobre 480×270; resultados reescalados a píxeles originales. |
| `flow.fb_max_error_px` | 1.0 | px de análisis | `assumed` | Rechazo ida/vuelta de Lucas-Kanade. |
| `flow.direction_dominance` | 0.7 | fracción | `assumed` | Umbral para declarar un sentido dominante. |
| `flow.min_motion_px` | 0.5 | px de análisis | `assumed` | Por debajo, el par se clasifica `none`. |
| `timestamps.gap_factor` | 1.5 | × mediana | `assumed` | Umbral de salto de PTS. |
| `timestamps.content_repeat_ratio` | 0.1 | × mediana | `assumed` | Umbral de candidato a imagen casi repetida (no certifica duplicado). |
| `timestamps.annotation_tolerance_s` | 1e-6 | s | `assumed` | Diferencia máxima entre el tiempo anotado y el del cuadro decodificado; cubre el redondeo a 6 decimales del CSV y debe ser menor que medio intervalo entre cuadros. |
| Máscara de vegetación ExG | `2G − R − B > 20` | niveles 0–255 | `assumed` | Proxy para separar vectores de suelo y de vegetación; no es un detector. |

**Hechos del clip `moringa_sin_ai` que siguen como `unknown`:** sesión, sitio,
montaje de la cámara (solidaridad, orientación, altura y posición respecto de la
herramienta), escala métrica, intrínsecos y distorsión. La relación temporal entre
observación e intervención también es desconocida. Ver
`docs/video_characterization.md` y `docs/video_capture_gaps.md`.

## 8. Planificador y validador geométrico A1/E2 (`configs/planner.yaml`, `configs/validation.yaml`)

Todos son parámetros de software con procedencia `assumed`, no mediciones. El
despeje exigido (`clearance_epsilon_m`, 1e-4 m) y el presupuesto de error de
envolvente (`geometric_approximation_tolerance_m`, 1e-5 m) se toman de
`configs/safety.yaml` y no se duplican.

| Parámetro | Valor | Unidad | Procedencia | Nota |
|---|---|---|---|---|
| `planner.generation_margin_m` | 0.01 | m | `assumed` | Holgura solo para **proponer** candidatos; el validador no la usa. Puede rechazar pasos factibles más estrechos (p. ej. `narrow_passage`, hueco de 0.22 m para 0.20 m de cuchilla). |
| `planner.level_padding_m` | 0.05 | m | `assumed` | Separación de los niveles longitudinales respecto de cada obstáculo de configuración. |
| `planner.lateral_candidates_per_interval` | 3 | — | `assumed` | Posiciones laterales por intervalo libre. |
| `planner.max_candidates` | 8 | — | `assumed` | Candidatos ordenados que se suavizan y validan. |
| `planner.max_graph_edges` | 200 000 | aristas | `assumed` | Presupuesto determinista de búsqueda. |
| `planner.weights` | 1.0 / 1.0 / 0.25 | — | `assumed` | Longitud, desviación y suavidad. La seguridad no es un término de costo. |
| `planner.normalizers` | 0.25 m; 1 rad² | — | `assumed` | Escalas de desviación y de suavidad. |
| `demo` | escena nominal, inicio `(0, 0.10)`, meta `(0, 4.90)` m, θ = 0 | m, rad | `assumed` | Extremos del origen `B` en el perfil de demo; no modifican la escena E1. |
| `random_scenarios` | semillas 0–99; ancho 0.8–1.4 m; 0–6 plantas | — | `synthetic` | Generador congelado de E2-08. |
| `validation.max_subdivision_depth` | 16 | niveles | `assumed` | Al agotarse da `UNCERTIFIED`, nunca aprobación. |
| `validation.max_intervals` | 400 000 | intervalos | `assumed` | Presupuesto total por validación. |
| `validation.continuity_tolerance` | 1e-8 | abs. | `assumed` | Saltos en p, dp/ds y d²p/ds² en las uniones. |
| `validation.endpoint_tolerance_m` | 1e-9 | m | `assumed` | Extremos del camino frente al inicio y la meta pedidos. |
