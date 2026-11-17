# Plan técnico del prototipo de desmalezado intrafilar de precisión en moringa

**Fecha:** 24 de octubre de 2026. **Versión:** 1.0. **Estado:** especificación de implementación y revisión; no constituye validación de maquinaria.

**Destinatarios:** el desarrollador y el arquitecto/supervisor técnico como revisor. Este documento define contratos, entregables y puertas de aceptación; los módulos y comandos descritos son trabajo futuro, no funcionalidades ya implementadas.

## 1. Fundamento y análisis de las referencias

### 1.1 Documento funcional

Referencia principal: [Plan_Desmalezadora_Precision_v02.md](../.context/Plan_Desmalezadora_Precision_v02.md). Sus requisitos rectores son:

- Prototipo exclusivamente offline: videos, escenarios sintéticos, trayectoria de una cuchilla virtual y evaluación; sin control del tractor, actuadores o herramienta enterrada.
- Proteger bases de tallos y sus alrededores; detectar follaje no equivale a localizar la base. La protección de raíces requiere criterios agronómicos todavía pendientes.
- Separar percepción, geometría, seguimiento, planificación, simulación y evaluación. Poder sustituir detecciones por referencias manuales.
- Evaluar la geometría completa de una cuchilla alargada y su movimiento continuo; una trayectoria central libre no basta.
- Empezar con posiciones sintéticas o manuales y zonas de exclusión antes de entrenar visión. La fase 0 de videos se inicia primero, pero la ausencia de videos no bloquea el banco sintético.
- Las plantas se desplazan aparentemente de abajo hacia arriba en los videos existentes. Esto es movimiento relativo; no permite deducir por sí solo orientación del mundo, velocidad física ni anticipación respecto a la cuchilla.
- La cámara posterior puede observar después de la actuación. La línea virtual de cruce sirve para ensayos offline; solo una geometría cámara-herramienta documentada permite evaluar anticipación física.
- Altura de 0,5 m, geometría, márgenes, velocidades y latencias son hipótesis, no mediciones. Una homografía sin referencia métrica no proporciona escala física verificable.
- El MVP físico descrito en la fuente requiere videos, posiciones de tallos y escala verificables. Si estos faltan, se entrega un demostrador sintético o exploratorio con la brecha abierta, sin declarar ese MVP completo.

La hoja de ruta de la fuente ubica dataset y geometría antes del planificador, pero su primer hito recomienda empezar por planificación sintética. Aquí se compatibilizan ambas instrucciones: iniciar caracterización, construir contratos y banco geométrico autónomo, validar cinemática y después integrar dataset, calibración y visión.

### 1.2 Análisis del SVG

Referencia: [descarga.svg](../.context/descarga.svg), inspeccionado mediante su estructura vectorial completa.

| Elemento | Evidencia en el SVG | Consecuencia técnica |
|---|---|---|
| Lienzo | `viewBox="0 0 340 330"` | Unidades gráficas, sin escala física ni metadatos de calibración. |
| Corredor | Rectángulo `x=90`, ancho `160`, altura `330` | Representa una franja de trabajo conceptual; no determina alcance mecánico. |
| Plantas | Centros `(169,65)`, `(169,165)`, `(169,265)` | Tres obstáculos alineados, separación gráfica de 100 unidades. Útiles como caso visual de regresión. |
| Protección | Círculos de radio `32`; vegetación de radios `17` y `11`; centro de radio `4.5` | El círculo exterior puede inspirar zonas de exclusión. Los radios no son dimensiones agronómicas. |
| Trayectoria | Curvas Bézier desde `(169,330)` hacia `(169,0)`, con desvíos hacia `x=115` y `x=224` | Sugiere esquivar plantas a lados alternos. No obliga a alternar: el planificador debe poder mantener un lado si es más viable. |
| Herramienta | Rectángulo redondeado de `30×10`, ubicado en `(104,243)`, pivote en `(119,248)` | Motiva huella finita y punto de referencia explícito; no define orientación dinámica ni articulación. |
| Flechas/textos | “Vista cenital”, “Simulación conceptual”, avance hacia arriba | Ilustración de intención; no resuelve la relación entre avance real y flujo óptico. |

El SVG no especifica tiempos, límites, incertidumbre ni volumen barrido. La curva y el rectángulo se dibujan independientemente; no deben asumirse congruentes con una pose exacta ni certificados contra colisión. Crear una escena inspirada en su disposición, conservando las unidades gráficas si se replica literalmente. Las escenas métricas tendrán parámetros sintéticos propios. No convertir 32 unidades gráficas en centímetros ni reutilizar la curva como trayectoria aprobada.

### 1.3 Situación inicial y decisiones pendientes

En la inspección inicial del repositorio se encontraron las dos referencias y sus archivos auxiliares `Zone.Identifier`; no se encontraron videos, dataset, código o configuración de implementación. Los auxiliares no son fuentes de diseño.

Registrar en `docs/assumptions.md` cada parámetro con valor, unidad, procedencia (`measured`, `synthetic`, `assumed`, `unknown`), incertidumbre y responsable de validarlo. Quedan pendientes dimensiones de cuchilla y raíces a proteger, configuración mecánica, velocidad, óptica, posición relativa cámara-herramienta y disponibilidad de videos. No inventar valores medidos para cerrar estas decisiones.

## 2. Alcance, hitos y dependencias

**H1 — Banco sintético:** plantas conocidas, exclusiones, cuchilla finita, ruta continua, simulación cinemática y reporte reproducible. Se alcanza al cerrar E1–E3. Es el primer hito de desarrollo.

**H2 — Referencia manual sobre video:** ingesta temporal, anotaciones, proyección y seguimiento; simulación comparada con referencias. Puede ser exploratoria en píxeles. Se alcanza con E0 y E4–E5, además de H1.

**H3 — MVP evaluado:** pipeline de video con escala verificable, referencia manual, exportaciones físicas y factibilidad explícita; percepción automática evaluada por separado. E6 agrega visión y E7 integra y compara. La aptitud del detector se declara con métricas, sin presumirla por completar el pipeline.

Orden: `E0 inicio → E1 → E2 → E3 → E4 → E5 → E6 → E7`. La documentación de E0 puede completarse mientras avanza E1–E3. Si no hay clips, E0 queda `blocked_data`, E1–E3 continúan y E4–E7 conservan sus dependencias reales. No iniciar entrenamiento ni optimización avanzada antes de H1.

**Extra autorizado por el usuario (25-10-2026):** [M0-P](orders/M0_exploratory_technical_order.md) adelanta un preview `ground_assumed` entre el cierre A1/E2 y A3/E3, reutilizando un modelo existente sin entrenamiento. No altera las dependencias/puertas de E3–E7 ni cierra H1. El M0 evaluado conserva su ubicación posterior; el preview no acredita escala, bases, mecánica ni anticipación.

No forman parte del alcance: diseño mecánico, control en campo, profundidad optimizada, navegación autónoma, eficacia agronómica, reproducción de mecanismos comerciales o garantías de seguridad física. Reservar una interfaz de profundidad Z, inicialmente `null`; no simular elevaciones instantáneas.

## 3. Arquitectura modular

### 3.1 Flujo y responsabilidades

```text
Escenario sintético ───────────────────────────┐
Video → ingesta → observaciones manuales/modelo│
                     ↓                        │
         calibración → seguimiento → mapa local
                                              ↓
                               planificador geométrico
                                              ↓
                                temporalización y simulador
                                              ↓
                                  validador independiente
                                              ↓
                            evaluación → gráficos/video/exportación
```

El orquestador conecta componentes; el planificador no abre videos, importa modelos de visión ni dibuja. El validador decide aceptación y nunca confía en el indicador de éxito del planificador. La evaluación con referencia de verdad se mantiene separada del mapa estimado: una planta omitida puede no existir en ese mapa y aun así sufrir una colisión real simulada.

| Módulo | Entrada y salida | Responsabilidad y límite |
|---|---|---|
| `core` | Esquemas, configuración, errores | Unidades, identificadores, validación y serialización; no depende de visión. |
| `scenarios` | Semilla/config → escenario | Plantas, corredor, verdad de referencia, avance y perturbaciones conocidas. |
| `video` | Archivo → `FramePacket` | PTS, decodificación, manifiesto y trazabilidad temporal. |
| `perception` | Fotograma → `StemObservation[]` | Adaptadores manual, clásico y aprendido intercambiables. |
| `geometry` | Observaciones + calibración → coordenadas | Distorsión, homografías, transformaciones e incertidumbre. |
| `tracking` | Observaciones temporales → `PlantTrack[]` | Asociación, predicción y caducidad; movimiento del terreno explícito. |
| `mapping` | Tracks + región observada → `LocalMap` | Exclusiones, zona desconocida y suelo objetivo. |
| `planning` | Mapa + estado → `PlanResult` | Candidatos geométricos, costos y razón de abstención. |
| `simulation` | Ruta + avance + mecanismo → ejecución | Temporalización, retardo, límites, estado de contacto y trayectoria ejecutada. |
| `validation` | Trayectorias + geometría + límites → `ValidationReport` | Barrido continuo, holgura, límites y anticipación. |
| `evaluation` | Ejecución + referencia → métricas | Cobertura, colisiones, errores, abstención y tiempos. |
| `visualization` | Estados + transformaciones → artefactos | Vista cenital, overlays, gráficas y MP4. |
| `pipeline` / `cli` | Config + fuentes → ejecución | Conexión, registro de artefactos y códigos de salida. |

Interfaces mediante `typing.Protocol` y dataclasses tipadas: `ObservationProvider.observe(frame)`, `Tracker.update(observations, motion, timestamp)`, `Planner.plan(request, local_map, tool_profile) -> PlanResult` (espacio de tarea en `W`, sin `ToolState` articular; [ADR 0001](adr/0001_task_space_planning.md)), `Simulator.rollout(timed_plan, vehicle_motion)`, `Validator.validate(execution, local_map, reference=None)`. El método sintético entra directamente al mapa y usa el mismo planificador y validador. La compilación de la tarea al mecanismo (`TrajectoryCompiler`, `MechanismModel`) pertenece a A3/E3 según la [adaptación arquitectónica](architecture_adaptation_plan.md).

### 3.2 Coordenadas y contratos de datos

- Imagen `I`: `(u,v)` en píxeles, origen arriba a la izquierda, `u` derecha y `v` abajo; distinguir imagen cruda de imagen sin distorsión.
- Suelo `W`: `(x,y)` con `x` lateral derecha respecto al sentido de avance, `y` longitudinal hacia delante, metros cuando la escala está verificada o es sintética; marco fijo durante cada segmento local.
- Vehículo `V`: origen y orientación documentados respecto a `W`; herramienta `B`: origen en su pivote definido y polígono local. `T_WV(t)` y `T_VB(q(t))` sitúan la huella en el mundo. No sumar avance del tractor dos veces.
- En exploración, usar `image_px` o `ground_assumed`; no mezclar con `ground_metric_verified`. Una vista estabilizada sin escala sigue sin ser métrica. La geometría sintética métrica se etiqueta `ground_metric_synthetic`.
- `H_ground_from_undistorted_image` explicita la dirección. La transformación inversa se usa para overlays. Cada homografía lleva dominio espacial, resolución, versión y vigencia temporal.
- Tiempo en segundos desde el inicio del clip, derivado de PTS; latencias en segundos. La hora de ejecución del programa es un reloj distinto. Un índice de frame no reemplaza tiempo en videos de FPS variable.

| Contrato | Campos mínimos |
|---|---|
| `FramePacket` | `clip_id`, índice, `pts`, `time_base`, `timestamp_s`, fuente temporal, tamaño y fotograma. |
| `StemObservation` | ID de observación, frame, `(u,v)`, covarianza 2×2 o cota de error, visibilidad, confianza, fuente y calidad de anotación. |
| `Calibration` | ID, intrínsecos/distorsión si existen, homografía, evidencia de escala, error de puntos independientes, dominio y estado. |
| `PlantTrack` | `track_id`, tiempo, posición, marco/unidad, covarianza/cota, último avistamiento, estado y procedencia. |
| `LocalMap` | Tiempo, marco, corredor, región observada, objetivo, exclusiones identificadas, zonas desconocidas, validez temporal. |
| `ToolGeometry` | Polígono de colisión, huella activa de corte, pivote, orientación fija/variable, unidad; pueden diferir colisión y corte. |
| `ToolState` | Tiempo, posición/orientación, velocidad, aceleración, contacto, marco y profundidad opcional. |
| `PlanResult` | Estado, razón, `TaskPath` geométrico (solo si `ACCEPTED`), identidades de mapa/perfil/solicitud/configuración, candidatos rechazados y costos; un fallo no contiene una ruta implícita. Implementado en A1/E2 (`planning.base`). |
| `ValidationReport` | Aprobación, unidades, mínimo despeje, intervalos de colisión/no certificables, violaciones y tolerancias utilizadas. |

Los esquemas rechazan valores no finitos, tiempos decrecientes, dimensiones negativas, polígonos inválidos y covarianzas no semidefinidas positivas. Distinguir lista vacía de plantas de región certificada como libre: ausencia de detecciones no prueba ausencia de cultivo.

### 3.3 Exclusiones, incertidumbre y condiciones estrictas

Para cada planta, `P_i` representa protección agronómica nominal; en el caso circular incluye radio de tallo y separación configurada. Construir `E_i = P_i ⊕ D(m_i)`, donde `⊕` es suma de Minkowski y `D` un disco. Primera versión: sumar cotas conservadoras de localización, calibración y seguimiento, expresadas en la misma unidad. El error de ejecución infla la herramienta o las exclusiones, pero no ambos.

Si se usan covarianzas, convertirlas a una cota declarada, por ejemplo `k·sqrt(λ_max(Σ))`, con `k` registrado; esto es un modelo de incertidumbre, no garantía probabilística sin validar la distribución. Una confianza de detector no equivale a desviación espacial. Oclusión o pérdida de track incrementa la cota; superar el presupuesto invalida la intervención.

Condiciones duras para todo instante del horizonte aprobado:

1. La huella de colisión transformada, con error de ejecución, permanece a distancia mayor que `clearance_epsilon` de todas las exclusiones. Contacto/tangencia se rechaza.
2. Su huella permanece dentro del dominio permitido y conocido; no se extrapola trabajo a suelo desconocido.
3. Posición, velocidad y aceleración del actuador cumplen sus límites en el marco `V`; orientación también si está habilitada.
4. El plan enlaza con el estado actual sin salto y tiene información y tiempo de anticipación suficientes para el modo declarado.

La cobertura es un objetivo secundario; nunca se compensa una colisión mediante una mejora de costo. E2 certifica la huella con orientación fija pedida en `W` (`OrientationRequirement` `FIXED`); `INTERVAL`/`FREE` se rechazan como capacidad no soportada. **Sustituido por [ADR 0001](adr/0001_task_space_planning.md):** el supuesto anterior de «movimiento X-Y relativo al vehículo» no describe el mecanismo inicial (brazo oscilante 1R); si una tarea es ejecutable lo decide la compilación al mecanismo en A3/E3, no el planificador.

### 3.4 Planificador inicial y validación continua

**Algoritmo base determinista:** construir obstáculos de configuración para la huella de orientación fija; generar niveles longitudinales alrededor de cada exclusión y estados laterales candidatos (centro y corredores a izquierda/derecha); conectar estados hacia delante, validar aristas y buscar una ruta por costo con desempate estable. Esto permite desvíos del mismo lado o alternados. No exigir completitud: `NO_PATH_FOUND` significa que el algoritmo no encontró ruta, no una prueba universal de imposibilidad.

Suavizar candidatos con segmentos quinticos o splines C2, manteniendo posición, tangente y aceleración compatibles en uniones. Volver a validar toda la curva: interpolar vértices libres puede crear colisiones entre ellos. Si el suavizado falla, intentar otro candidato; una polilínea con esquinas no se declara cinemáticamente aceptada.

En H1 se permite inflar obstáculos por el disco circunscrito a la cuchilla como aproximación conservadora del espacio de configuración. Registrar su conservadurismo, que puede rechazar rutas válidas. El validador final considera el polígono completo; nunca usar solo su centro o vértices.

**Certificación entre muestras:** para traslación lineal y orientación fija de polígono convexo, comprobar el barrido exacto como envolvente convexa de las huellas de los extremos. Para curva o rotación, subdividir intervalos y usar una envolvente conservadora con cota de desviación de la curva y del desplazamiento angular de sus puntos. Para una curva con cota de aceleración `A`, el error de la cuerda se acota por `A·Δt²/8`; añadir la cota rotacional correspondiente si aplica. Si la envolvente toca una exclusión, subdividir; al llegar al límite de subdivisión sin poder certificar, rechazar como `UNCERTIFIED`, nunca aprobar por muestreo escaso. Aproximaciones de círculos y buffers deben ser exteriores o compensar explícitamente su error de discretización.

Validar continuidad y máximos de derivadas en los segmentos, no solo diferencias finitas en instantes de video. Los polinomios permiten evaluar extremos en raíces de sus derivadas dentro de cada intervalo. La prueba independiente incluye oráculos analíticos sencillos y un caso de colisión entre dos muestras libres.

### 3.5 Tiempo, avance y abstención

La trayectoria geométrica `p(s)` se temporaliza usando `s(t)` y el avance del vehículo. Para comprobar límites relativos: `q̇ = q'(s)ṡ` y `q̈ = q''(s)ṡ² + q'(s)s̈`. No reducir arbitrariamente la velocidad del tractor para que una ruta pase: es una entrada. Solo un escenario explícitamente distinto puede cambiarla. La simulación propaga retardo y respuesta limitada del actuador y valida también la trayectoria ejecutada, no solo la solicitada.

Para un punto observado por delante del trabajo, calcular tiempo hasta encuentro integrando avance; `d/v` sirve únicamente como aproximación con velocidad positiva constante y distancia longitudinal con signo conocido. Registrar adquisición, inferencia, planificación, comunicación y respuesta mecánica. El inicio y duración de la maniobra deben caber antes del encuentro. Velocidad cero, inversión, distancia desconocida o negativa se manejan explícitamente. En modo exploratorio, línea virtual y velocidad supuesta no autorizan la etiqueta de anticipación física validada.

Estados del resultado: `ACCEPTED`, `ABSTAIN`, `INVALID_INPUT`, `UNCERTIFIED`. Razones estables: `NO_PATH_FOUND`, `INSUFFICIENT_LOOKAHEAD`, `UNKNOWN_SCALE`, `STALE_MAP`, `EXCESS_UNCERTAINTY`, `LIMIT_VIOLATION`, `COLLISION`, `NO_SAFE_TRANSITION`. La máquina de simulación distingue `IDLE`, `WORKING`, `RETREATING`, `ABSTAINED` y `UNSAFE_STATE`.

Abstenerse no hace desaparecer una cuchilla que ya está trabajando. Una retirada lateral requiere ruta temporal validada desde el estado actual. Mantenerla quieta respecto al tractor todavía la mueve en el suelo. Si no existe transición segura, registrar `UNSAFE_STATE`, terminar ese episodio y declarar fallo; no fingir frenado, repliegue o elevación instantánea. Cuando se empieza sin contacto en una zona de estacionamiento conocida, se puede omitir la región, con cobertura cero y causa registrada.

## 4. Tecnologías y estructura del repositorio

Base propuesta: Python 3.12 como versión inicial de compatibilidad, entorno virtual, `pyproject.toml` y un archivo de bloqueo reproducible; validar resolución conjunta en E1 sin asumir compatibilidad por número de versión. CPU suficiente para H1/H2; GPU opcional solo para entrenamiento.

| Tecnología | Uso propuesto y justificación |
|---|---|
| NumPy | Vectores, matrices y trayectorias; `float64` en geometría y validación. |
| OpenCV, variante headless | Calibración, corrección de distorsión, flujo óptico y overlays. Elegir un único paquete OpenCV. |
| SciPy | Interpolación, optimización limitada y asociación lineal de tracks. |
| Shapely/GEOS | Polígonos, buffers, intersecciones y uniones de cobertura 2D; envolver tolerancias y comprobar geometrías inválidas. |
| PyAV y FFmpeg/ffprobe | PTS, inspección de streams y codificación de video. Evitar depender de FPS nominal para sincronización. |
| Matplotlib | Gráficas estáticas, vista cenital y reporte exportable sin interfaz gráfica. |
| Dataclasses, typing, Pydantic y PyYAML | Dominio tipado y configuración validada; YAML mediante carga segura. |
| pytest e Hypothesis | Casos deterministas, propiedades e invariantes geométricos. |
| Ruff y mypy | Estilo y contratos entre módulos; incorporar verificaciones al flujo de revisión. |
| PyTorch + adaptador de modelo, opcional | Detector/segmentador/keypoints solo tras disponer de dataset. Evaluar licencia, costo y calidad antes de elegir pesos, incluido cualquier YOLO. |

CSV y JSON bastan para el MVP; Parquet/PyArrow es opcional si el volumen lo justifica. No introducir ROS, servicios web, base de datos ni infraestructura distribuida en el primer prototipo. La anotación puede empezar con un script OpenCV; una herramienta externa es opcional si exporta el esquema acordado.

Estas decisiones se apoyan en capacidades documentadas de [OpenCV para calibración](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html), [Shapely para geometría planar](https://shapely.readthedocs.io/en/stable/manual.html), [SciPy para interpolación](https://docs.scipy.org/doc/scipy/reference/interpolate.html) y [PyAV para PTS y bases temporales](https://pyav.org/docs/stable/api/time.html), consultadas el 24-10-2026. Son elecciones arquitectónicas del proyecto; las versiones concretas se fijarán y probarán en E1.

```text
.
├── .context/                         # Referencias originales; conservar sin modificar
├── docs/
│   ├── technical_plan.md
│   ├── assumptions.md
│   ├── architecture.md
│   ├── data_contracts.md
│   ├── video_characterization.md
│   ├── video_capture_gaps.md
│   ├── annotation_guide.md
│   ├── calibration_report.md
│   ├── evaluation_protocol.md
│   ├── evaluation_report.md
│   ├── reviews/                      # Evidencia de aceptación E0…E7
│   └── adr/                          # Decisiones y cambios de arquitectura
├── configs/
│   ├── synthetic.yaml
│   ├── tool.yaml
│   ├── safety.yaml
│   ├── kinematics.yaml
│   ├── video.yaml
│   ├── camera.example.yaml
│   ├── perception.yaml
│   └── evaluation.yaml
├── data/
│   ├── video_manifest.csv
│   ├── raw/                          # Videos originales, fuera de Git
│   ├── annotations/                  # Exportaciones versionadas y procedencia
│   ├── calibration/
│   └── splits/                       # IDs por sesión/sitio
├── src/precision_weeder/
│   ├── __init__.py
│   ├── cli.py
│   ├── core/                         # types.py, config.py, units.py, errors.py
│   ├── scenarios/                    # synthetic.py, fixtures.py
│   ├── video/                        # reader.py, manifest.py, timing.py
│   ├── perception/                   # base.py, manual.py, classical.py, learned.py
│   ├── geometry/                     # frames.py, calibration.py, uncertainty.py
│   ├── tracking/                     # association.py, filter.py, motion.py
│   ├── mapping/                      # exclusions.py, local_map.py
│   ├── planning/                     # base.py, geometric.py, smoothing.py
│   ├── simulation/                   # tool.py, timing.py, actuator.py, rollout.py
│   ├── validation/                   # swept_volume.py, kinematics.py, temporal.py
│   ├── evaluation/                   # metrics.py, reference.py, reports.py
│   ├── visualization/                # topdown.py, overlay.py, plots.py
│   └── pipeline/                     # synthetic.py, replay.py, runner.py
├── tests/
│   ├── unit/
│   ├── property/
│   ├── integration/
│   ├── regression/
│   └── fixtures/                     # Datos pequeños y oráculos conocidos
├── scripts/                          # Demos delgadas sobre la API pública
├── experiments/
│   ├── phase0/
│   └── benchmarks/                   # Protocolos/configs; resultados pesados aparte
├── runs/                             # Artefactos generados, fuera de Git
├── models/                           # Pesos fuera de Git; manifiestos versionados
├── pyproject.toml
├── uv.lock                           # Si se adopta uv; una sola estrategia de bloqueo
├── .gitignore
└── README.md
```

No crear módulos vacíos para simular avance: materializar cada archivo al implementar su etapa. Datos y pesos grandes tendrán manifiesto, hash y ubicación recuperable. Los scripts no duplican algoritmos del paquete.

## 5. Etapas de implementación y aceptación

### E0 — Caracterización de videos y brechas

**Objetivo:** determinar qué material existente permite observar, medir y validar sin exigir nuevas capturas de entrada.

**Entregables:** `video/manifest.py`, script `scripts/characterize_videos.py`, `configs/video.yaml`, manifiesto y documentos de fase 0 indicados en la estructura; fotogramas/flujo óptico en `experiments/phase0/`; `tests/unit/test_video_manifest.py` y fixture temporal de video generado.

El manifiesto registra ID/ruta/hash, sesión/sitio, códec, resolución, duración, FPS nominal/efectivo, origen de timestamps, intervalos anómalos, visibilidad de tallos, referencia de escala, montaje conocido, observación previa/posterior/desconocida y aptitud A/B/C con justificación. A: percepción y seguimiento; B: experimentos limitados especificados; C: no apto. Estas categorías no implican calibración métrica.

**Aceptación:**

- Cada clip disponible tiene ficha y muestras de condiciones relevantes, incluyendo sombras, vibración y oclusión cuando existan. Registrar anomalías temporales observables; sin referencia externa no afirmar cuántos cuadros se perdieron durante captura.
- Se documenta dirección aparente, visibilidad de bases, posibilidad de homografía, escala y anticipación, usando `unknown` cuando falte evidencia.
- Para cerrar la fase con datos: al menos un clip permite identificar y seguir varias plantas y definir una línea virtual. Para ese clip se declara por separado si son posibles calibración física y validación temporal.
- Si no hay archivos o ninguno cumple, entregar diagnóstico y brechas, marcar la puerta de datos pendiente y continuar H1. No generar un manifiesto ficticio.
- El fixture prueba lectura de metadatos y errores de archivo inexistente/corrupto; las tomas adicionales se solicitan únicamente vinculadas a brechas concretas.

### E1 — Contratos, configuración y escenas sintéticas

**Objetivo:** establecer coordenadas, unidades y entradas reproducibles independientes de visión.

**Entregables:** paquete instalable, módulos `core`, `scenarios`, `geometry/frames.py`, `simulation/tool.py`; `configs/{synthetic,tool,safety,kinematics}.yaml`; contratos y supuestos; `scripts/demo_scene.py`; pruebas de esquemas, transformaciones y geometría de herramienta.

Crear escena principal de 3–5 plantas y casos: corredor vacío conocido, planta aislada, espaciamiento irregular, planta descentrada, exclusiones superpuestas, paso estrecho y corredor bloqueado. Añadir escena gráfica basada en el SVG como fixture sin escala.

Valores iniciales exclusivamente sintéticos sugeridos: corredor de 1 m de ancho y 5 m de largo; plantas centradas en `y=1,2,3 m`; exclusión nominal de `0.15 m`; margen de incertidumbre `0.03 m`; cuchilla rectangular `0.20×0.08 m`, lado largo sobre X; avance `0.20 m/s`. Ajustar límites y arranque en configuración documentada para que exista al menos un caso factible, sin modificar tests adversos para aprobarlos. Estos valores no recomiendan dimensiones reales.

**Aceptación:**

- Instalación limpia y tests sin GPU ni videos privados; misma semilla genera exactamente las mismas entradas.
- Rechazo de NaN/inf, dimensiones inválidas, polígonos autointersectados, unidades mezcladas y calibración inexistente solicitada como verificada.
- Transformaciones identidad, traslación y rotación conocida pasan con tolerancia `1e-9` en fixtures bien condicionados; ida/vuelta preserva puntos y huella. El desplazamiento aparente hacia arriba se prueba sin invertir silenciosamente ejes.
- El demo dibuja polígonos, pivote, plantas, exclusiones, ejes y unidad, y exporta escena JSON. Todos los parámetros físicos de esta etapa figuran como sintéticos.

### E2 — Planificador geométrico y verificación del área barrida

**Objetivo:** encontrar rutas continuas libres para una cuchilla finita con datos perfectos, antes de introducir percepción.

**Entregables:** `mapping`, `planning/base.py`, `geometric.py`, `smoothing.py`, `validation/swept_volume.py`, vista cenital; `scripts/demo_planner.py`; tests unitarios de exclusión, colisión y suavizado; propiedades y regresiones adversas.

Implementar candidatos, orden de costos normalizados y desempates deterministas. Usar primero longitud/desviación/suavidad; medir cobertura aunque aún no se optimice exhaustivamente. Separar el costo de las condiciones estrictas. Guardar candidatos rechazados y causas para depuración.

**Aceptación:**

- La escena principal y fixtures analíticamente factibles tienen una ruta aprobada con progreso longitudinal completo, sin intersecciones, sin salir del corredor y con continuidad C2 en segmentos de trabajo. Rechazar todo no satisface esta puerta.
- Corredor bloqueado, inicio en colisión y paso más angosto que la huella producen rechazo explícito. `NO_PATH_FOUND` no se etiqueta como imposibilidad matemática demostrada.
- Una ruta cuyo centro es libre pero cuyo borde toca una planta falla. También fallan tangencia, esquina que invade una zona y colisión entre muestras consecutivas libres.
- Validación continua según §3.4, incluida la curva suavizada. Intervalos no certificables se rechazan; nunca se sustituyen por una animación visual como evidencia.
- Al menos 100 escenarios de semillas fijas y pruebas de propiedades: ninguna ruta aprobada viola el validador; ampliar una exclusión no puede volver libre la misma trayectoria antes inválida. Contrastar casos simples con oráculo analítico distinto del algoritmo de planificación.
- Demo exporta ruta, mapa, estados y gráfico con mínimo despeje y motivos de fallo. La aprobación de E2 es geométrica, todavía no temporal.

### E3 — Temporalización, cinemática y respuesta simulada

**Orden de ejecución vigente:** [A3/E3 v1.0](orders/A3_E3_technical_order.md), emitida el 2 de noviembre de 2026 tras aprobar M0-P.

> Adaptada por la [adaptación arquitectónica](architecture_adaptation_plan.md) §10 A3 y el [ADR 0001](adr/0001_task_space_planning.md): E3 compila `TaskPath`/`TaskTrajectory` al brazo 1R mediante `TrajectoryCompiler` y `MechanismModel`; los límites X–Y de `kinematics.yaml` quedan como fixture cartesiano heredado.

**Objetivo:** completar H1 mediante trayectorias factibles bajo avance, límites y latencia explícitos.

**Entregables:** `simulation/{timing,actuator,rollout}.py`, validadores cinemático/temporal, máquina de estados; `scripts/demo_synthetic.py`; configs de velocidad constante/variable, retardo, saturación y fallo; tests de derivadas, transformaciones móviles y transición de abstención.

**Aceptación:**

- La escena nominal ejecuta una maniobra completa con la huella finita, cero colisiones y límites de posición, velocidad y aceleración satisfechos a lo largo de cada segmento, incluida entrada/salida.
- Se exportan trayectoria solicitada y ejecutada, retardo y máximos cinemáticos. Un plan geométrico válido pero demasiado rápido se rechaza o replanifica respetando el avance impuesto.
- Tests de velocidad variable, tractor detenido, timestamps irregulares, retardo insuficiente/excesivo y estado inicial incompatible. Aumentar artificialmente retardo hasta consumir anticipación debe impedir aprobación temporal.
- Ejecución con saturación o error de seguimiento se vuelve a validar; no se oculta una colisión porque la referencia era válida.
- Abstención desde estacionamiento cubre cero; retirada desde trabajo valida el recorrido completo; si no hay escape, aparece `UNSAFE_STATE`. No se cuenta esa ejecución como segura.
- H1 produce animación cenital, JSON de métricas, CSV de trayectorias y configuración resuelta. Repetir la ejecución conserva métricas dentro de tolerancias registradas.

### E4 — Dataset, anotación y geometría de cámara

**Objetivo:** reemplazar plantas sintéticas por observaciones manuales trazables sin confundir proyección con escala verificada.

**Dependencias:** H1 para integración; videos caracterizados de E0. Si no hay datos, pueden probarse utilidades con fixtures, pero no cerrar validación real.

**Entregables:** lector temporal, adaptador manual, herramienta/script de anotación, guía de anotación, dataset y splits; `geometry/{calibration,uncertainty}.py`; `configs/camera.example.yaml`; `docs/calibration_report.md`; `scripts/demo_manual_map.py`; tests de proyección, distorsión, timestamps y división de datos.

Anotar base del tallo, identidad temporal, visible/ocluida/ambigua, incertidumbre y región de planta opcional. No colocar bases inventadas como referencia cierta. Separar sesiones y sitios entre entrenamiento/validación/prueba; si solo existe una sesión, documentar que no permite una prueba independiente equivalente. Revisar al menos 10% de anotaciones y todos los casos ambiguos mediante segunda revisión, registrando desacuerdos y resolución.

**Aceptación:**

- Cada anotación referencia frame/tiempo, clip y versión; el conjunto de prueba queda congelado antes de ajustar visión. Cero solapamiento de sesiones entre splits cuando se declare separación por sesión.
- Fixtures sintéticos con homografía conocida recuperan posiciones y covarianzas dentro de tolerancias configuradas. Puntos degenerados, homografía singular y regiones fuera de calibración se rechazan.
- Calibración real usa referencias físicas documentadas y puntos de control independientes de los usados para estimar. Reportar error mediano/p95/máximo y presupuesto de error previamente declarado en `evaluation.yaml`; aprobar solo si el máximo está dentro del presupuesto en el dominio de uso.
- Si no hay escala verificable, todas las salidas muestran “exploratorio” y unidades píxel o escala supuesta; no se exportan centímetros como mediciones. Esto permite un demo exploratorio, no aprobación del MVP físico.
- Overlay ida/vuelta consistente tras resize/recorte/corrección óptica. PTS variable y gaps conservan su tiempo; una exportación CFR conserva un mapa de correspondencias con el original.

### E5 — Seguimiento, movimiento y replay con referencia manual

**Objetivo:** alimentar el planificador mediante un mapa causal y temporalmente consistente sobre video.

**Entregables:** `tracking`, mapa local con vigencia, `pipeline/replay.py`, estimación de flujo/homografía entre frames y línea virtual; `scripts/demo_manual_replay.py`; pruebas de asociación, pérdida/reaparición, movimiento y prevención de uso de datos futuros.

Asociar con distancia/covarianza y asignación lineal; filtro sencillo con modelo de planta estática en `W` y movimiento de cámara/vehículo separado. Si se sigue en imagen, declarar ese marco y usar movimiento aparente. Estimar movimiento con características de suelo y rechazo de outliers; evaluar residual e inliers antes de aceptar una homografía. No suponer que follaje y suelo comparten plano.

**Aceptación:**

- Fixtures de plantas estáticas y cámara móvil mantienen posiciones mundiales dentro de tolerancia; no asignan velocidad física a plantas por el flujo visual.
- IDs estables en secuencias sintéticas sin ambigüedad; en clips reales reportar cambios de identidad, error de localización y de cruce de línea con referencia anotada.
- En pérdida breve, incertidumbre crece según el modelo; al superar timeout/presupuesto se protege la región desconocida o se abstiene, sin borrar inmediatamente el riesgo.
- Cambios de pose, homografía de baja calidad y mapa vencido impiden trabajo con geometría obsoleta. Replanificar conserva continuidad desde el estado ejecutado actual.
- Replay causal solo consulta observaciones con tiempo menor o igual al disponible tras latencia. Añadir anotaciones futuras no cambia decisiones pasadas. Un modo offline con toda la verdad puede existir como oráculo separado y rotulado.
- Completar un replay manual reproducible, registrar oportunidades omitidas y validar anticipación únicamente cuando montaje, escala, velocidad y timestamps lo permitan. Material posterior sigue siendo válido para desarrollo offline.

### E6 — Percepción automática intercambiable

**Objetivo:** medir la capacidad de localizar bases de tallos y sus consecuencias sobre planificación con H1 ya aprobado.

**Entregables:** adaptadores intercambiables manual, clásico exploratorio y aprendido; YOLO con los pesos existentes de moringa como primer proveedor aprendido candidato, scripts de inferencia/evaluación (entrenamiento solo si la evaluación justifica hacerlo), `configs/perception.yaml`, manifiesto de pesos/dataset y ficha del modelo; pruebas de contrato, oclusión y falsas detecciones; comparación contra adaptador manual.

Se acuerda utilizar un modelo aprendido, manteniendo proveedores intercambiables y el planner independiente. El primer candidato es el YOLO de una clase disponible en `.context/Moringa_Train_Annotations/`; sus máscaras observan copa, no base de tallo. Comparar contra referencia manual y una línea base sencilla con OpenCV; esa comparación no obliga a descartar el modelo aprendido. Cambiar a keypoints, segmentación más estimador de base u otro modelo según etiquetas y evaluación. No tomar automáticamente centroide/caja como base; conservar objetivo observado, incertidumbre y abstención. El experimento aislado y M0-P no aprueban E6. El [protocolo E6 de la orden M0-P §8](orders/M0_exploratory_technical_order.md#8-puerta-posterior-e6-evaluación-aprendida-antes-de-decisiones-operativas) exige anotaciones propias, separación por sesión y criterios congelados antes de decisiones operativas.

**Aceptación:**

- Cambiar `manual` por `learned` no modifica API ni código del planificador. Inferencia conserva marco, tiempo, visibilidad y fuente; no convierte confianza en certeza espacial.
- Reportar error de base mediano/p95, recall de tallos evaluables, falsos positivos y errores por oclusión/luz/escena; declarar denominadores y bases no evaluables. Métricas físicas solo donde hay escala fiable.
- Antes de abrir el test, fijar en validación umbrales de confianza, error admisible, recall mínimo y presupuesto de incertidumbre. Si faltan datos para fijarlos o no se alcanzan, el modelo queda experimental; no inventar porcentajes de eficacia.
- Evaluar colisiones contra plantas de referencia, incluidas las omitidas por el detector. Una omisión no se puede considerar segura solo porque el mapa estimado no contiene obstáculos.
- Fixtures con tallo oculto, cuadro vacío, detección falsa, salto espacial y frame perdido ejercitan políticas de incertidumbre. Con baja confianza no se reduce el margen de protección.
- Comparación manual/automática bajo misma geometría, escenarios y avance; hashes de pesos, splits, semillas y versiones acompañan resultados. Sin modelo apto, conservar baseline manual funcional y estado de percepción no aprobado.

### E7 — Integración, evaluación comparativa y entrega

**Objetivo:** producir el MVP reproducible y una conclusión sustentada sobre sus límites.

**Entregables:** CLI única, exports, reporte de evaluación y guía de reproducción; tests de integración de escenario sintético y video pequeño; protocolo congelado; `scripts/run_benchmarks.py`. Comparar planificador base con al menos una variante acotada (por ejemplo, optimización local de sus puntos de control); horizonte móvil avanzado queda opcional si no aporta a la comparación.

**Aceptación:**

- Una ejecución genera MP4 anotado, CSV de observaciones/tracks/órdenes/ejecución, JSON de métricas, config resuelta, calibración y registro de eventos. Video incluye herramienta finita, zonas, trayectoria, tiempo, unidad y estado; no oculta fallos.
- Para declarar MVP físico: al menos un clip real con tallos de referencia y escala verificables, trayectorias en metros, colisiones simuladas reportadas y tramos no factibles señalados. Si falta ese clip, entregar los hitos alcanzados y la brecha, sin usar la etiqueta de MVP físico validado.
- Cero colisiones y cero violaciones duras en tramos declarados aprobados respecto a la referencia disponible. Ensayos adversos pueden fallar y deben aparecer como fallos; colisiones con referencia no observada por percepción invalidan la aprobación de esos tramos.
- Escena sintética nominal tiene progreso completo y cobertura positiva; reportar cobertura y abstención sobre todo el conjunto para impedir que abstenerse siempre aparente buen desempeño.
- Comparación con los mismos datos, semillas, huella, márgenes, restricciones y presupuestos de cómputo; publicar resultados por escenario además de agregados. No ajustar parámetros con el test.
- Instalación desde entorno limpio y un comando reproducen los fixtures incluidos. Videos privados usan instrucciones y manifiestos; no son necesarios para CI básica. Documentar hardware y tiempos sin afirmar tiempo real por medir solo throughput offline.
- Revisión final enumera requisitos satisfechos, pendientes, resultados y decisiones para un eventual banco físico; no autoriza pruebas mecánicas.

## 6. Métricas, tolerancias y protocolo de prueba

| Métrica | Definición operativa |
|---|---|
| Colisiones | Episodios/intervalos conectados con intersección o contacto, IDs afectados y duración; no contar cada frame como incidente distinto. Separar mapa estimado y referencia. |
| Despeje mínimo | Mínimo continuo entre huella de colisión y exclusiones; informar también distancia al tallo si existe referencia. No confundir ambas medidas. |
| Cobertura | `área(unión de huellas activas ∩ objetivo) / área(objetivo)`; objetivo fijo = corredor evaluable menos exclusiones de referencia. No encoger denominador según decisiones del planificador. |
| Objetivo vacío | Cobertura `null`, razón `NO_TARGET_AREA`; nunca dividir por cero ni asignar éxito automático del 100%. |
| Factibilidad | Trayectorias aceptadas / propuestas, con conteos de solicitudes sin propuesta y abstenciones. Separar validación geométrica y cinemática. |
| Abstención | Regiones omitidas / oportunidades predefinidas y proporción de tiempo/área omitida; publicar definición de oportunidad. |
| Dinámica | Máximos absolutos de posición relativa, velocidad, aceleración, error de seguimiento y retardo; orientación si aplica. |
| Percepción | Error por base evaluable, tasa de omisiones y falsos positivos con regla de asociación congelada; resultados por condición. |
| Seguimiento | Cambios de ID y error de predicción de cruce en segundos; error espacial físico solo con escala. |
| Cómputo | Tiempo por módulo y total p50/p95/máximo, calentamiento separado, equipo, resolución y número de frames; medir también antigüedad de la observación. |

La cobertura es proxy geométrico, no porcentaje de maleza eliminada. Comparar ejecuciones en dominio común; regiones sin referencia se reportan como no evaluables y no permiten certificar ausencia de daño.

**Tolerancias iniciales de software para fixtures métricos sintéticos:** `clearance_epsilon=1e-4 m`, cota máxima de aproximación geométrica `1e-5 m` y tolerancia de comparación numérica de límites `1e-8` en la unidad correspondiente. La tolerancia numérica no reemplaza un margen agronómico. Configurar equivalentes propios en píxeles, sin convertir mediante escala inventada. Si el error de discretización supera su presupuesto, la validación no aprueba. Ajustes futuros se justifican en ADR y mantienen regresiones adversas.

Además de los fixtures de cada etapa, cubrir: ruido creciente, planta en límite del corredor, obstáculos duplicados, coordenadas extremas, herramienta orientada, polígono inválido, gaps largos de video, cambio de resolución, mapa sin región observada y terminación de horizonte. Ensayar límites justo por debajo/en/por encima de los umbrales.

Las pruebas de propiedades deben usar oráculos independientes donde sea posible: distancia segmento-círculo, rectángulos con solapamiento conocido, trayectorias polinómicas con máximos analíticos. Verificar solamente que el planificador se aprueba a sí mismo no basta. Una malla temporal densa sirve como diagnóstico adicional, no sustituye certificado del barrido.

## 7. Configuración, ejecución y trazabilidad

Cada ejecución crea `runs/<run_id>/` con `resolved_config.yaml`, `run_manifest.json`, `events.jsonl`, `metrics.json`, `trajectory.csv`, `executed_trajectory.csv` y artefactos visuales. El manifiesto guarda versión de esquema, revisión de código si existe, estado de cambios locales, versiones del entorno, semillas, hashes de inputs/config/modelo/calibración, parámetros supuestos y hardware. No exigir igualdad binaria de MP4 entre plataformas; comparar datos numéricos con tolerancias y contenido temporal.

La configuración distingue `mode: synthetic | exploratory | calibrated`, fuente de plantas, marco, geometría, márgenes, algoritmo, avance, límites, latencias, horizonte, tolerancias y exportación. Las claves desconocidas son error; todo valor por defecto se materializa en la configuración resuelta. En modo `calibrated`, falta de escala/calibración válida causa fallo explícito, no degradación silenciosa.

CLI objetivo, disponible progresivamente:

```bash
python -m precision_weeder.cli characterize --input data/raw --config configs/video.yaml
python -m precision_weeder.cli synthetic --config configs/synthetic.yaml --output runs/demo
python -m precision_weeder.cli replay --config configs/video.yaml --observations manual --output runs/manual
python -m precision_weeder.cli replay --config configs/video.yaml --observations learned --output runs/learned
python -m precision_weeder.cli evaluate --run runs/manual --config configs/evaluation.yaml
pytest tests/unit tests/property tests/integration tests/regression
ruff check .
mypy src/precision_weeder
```

Configurar explícitamente las fuentes manuales/modelo y calibraciones en los YAML. Cada comando valida dependencias opcionales y explica qué falta. Diferenciar códigos de salida: error de entrada/ejecución, evaluación con violaciones y ejecución terminada con abstenciones esperadas; guardar siempre el estado estructurado. La abstención es un resultado del dominio, no una excepción accidental de Python.

## 8. Revisión por etapas y definición de terminado

Para cada E0–E7, el desarrollador entregará un cambio acotado con:

1. Objetivo, archivos implementados, configuración y demo reproducible.
2. Tests relevantes ejecutados, resultados y evidencia de criterios de aceptación.
3. `docs/reviews/E<n>.md` con cada criterio marcado `PASS`, `FAIL`, `BLOCKED` o `NOT_APPLICABLE` justificado; enlace a artefactos/manifiestos.
4. Supuestos añadidos o cambiados y limitaciones pendientes; ADR si cambia un contrato, sistema de coordenadas, política de incertidumbre o algoritmo principal.
5. README actualizado con el comando que funciona al cierre de esa etapa.

El supervisor verifica especialmente: unidades/marcos, referencia de verdad separada, huella finita y barrido, continuidad después del suavizado, límites entre muestras, causalidad temporal, incertidumbre, transición de abstención y métricas con denominadores completos. Una demo visual atractiva no reemplaza estas comprobaciones.

No declarar cerrada una etapa con criterios obligatorios bloqueados por datos. Se puede avanzar trabajo independiente, manteniendo explícita la dependencia pendiente. No introducir optimizadores complejos, modelos grandes o integración de hardware para compensar carencias del banco geométrico.

## 9. Trazabilidad y riesgos prioritarios

| Requisito de la referencia | Implementación / evidencia |
|---|---|
| Caracterizar material existente y pedir nuevas tomas solo por brechas | E0, manifiesto, caracterización y documento de capturas. |
| Primer hito con datos sintéticos y zonas de exclusión | E1–E3, H1 y demo sintético reproducible. |
| Base de tallo, referencia manual e incertidumbre | E4/E6, guía de anotación y contratos de observación. |
| Homografía/escala y cámara no necesariamente cenital | E4, calibración validada y modos de unidad separados. |
| Seguimiento, avance y ubicación posterior de cámara | E5, replay causal, flujo y evidencia de anticipación. |
| Cuchilla finita, margen y no colisión estricta | E2/E3, barrido conservador y validador independiente. |
| Restricciones del mecanismo (1R), velocidad, aceleración y retardo | A3/E3 (ADR 0001), trayectoria ejecutada, límites y tests temporales. |
| Abstención ante incertidumbre o falta de factibilidad | E3/E5/E6, estados y transición segura simulada. |
| Comparación reproducible y exports | E7, protocolo fijo, manifiestos y métricas completas. |

Los riesgos principales son base oculta, escala ausente, suelo no planar, vibración, retraso excesivo, planta omitida y trayectoria aparentemente libre pero inviable para la huella. Las respuestas son incertidumbre explícita, calibración por dominio, invalidación del mapa, anticipación medida, evaluación contra referencia y validación continua. Ninguno se resuelve únicamente aumentando una penalización de costo.

**Primer trabajo de implementación:** iniciar inventario E0 con los archivos que realmente estén disponibles y ejecutar E1; después implementar E2 y E3 hasta demostrar H1. Integrar el primer video con anotaciones manuales solo cuando ese banco permita distinguir un fallo de percepción de un fallo del planificador.

### Decisiones adicionales del usuario — 25-10-2026

El extra [M0-P](orders/M0_exploratory_technical_order.md) es una excepción exploratoria explícita a la integración de video posterior a H1; no sustituye la secuencia principal. Percepción aprendida queda acordada para E6, con YOLO como primer candidato intercambiable. Licenciamiento de Ultralytics (AGPL declarada) y procedencia/licencia de datos se registran como riesgo pendiente, fuera de alcance de esta prueba por instrucción del usuario; no constituyen una nueva puerta para M0-P. Conservar manifiestos y atribución disponible.
