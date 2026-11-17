# ADR 0001 — Planificación en espacio de tarea y compilación al mecanismo

**Estado:** aceptada para A1/E2 (25 de octubre de 2026).
**Fuentes:** [orden A1/E2](../orders/A1_E2_technical_order.md) §3;
[adaptación arquitectónica](../architecture_adaptation_plan.md) §3–§4 y §10;
[plan maestro](../technical_plan.md) §3.

Este ADR **aplica** la separación ya decidida en la adaptación arquitectónica. No
crea otro plan ni cambia objetivos ni el orden de etapas. Su función es resolver
las contradicciones que el plan maestro arrastraba sobre el control X–Y y la
orientación fija del mecanismo.

## Contexto

El plan maestro suponía una cuchilla que se mueve en X–Y respecto del vehículo con
orientación fija (§3.3) y un planificador que recibe `ToolState`
(`Planner.plan(local_map, tool_state, horizon)`, §3.1). Esa hipótesis no vale
para el mecanismo inicial definido por el usuario, un brazo oscilante de una
articulación (1R). En un 1R, el punto de referencia recorre un arco respecto del
vehículo y la orientación depende del mismo ángulo. Con un único grado de
libertad no se controlan a la vez X, Y y la orientación.

## Decisión

1. **E2 planifica en espacio de tarea.** El planificador produce un `TaskPath`
   geométrico en `W`, dentro de un `PlanResult`. No conoce articulaciones,
   longitudes, actuadores, límites de motor ni video, y no importa `kinematics`,
   el simulador ni OpenCV/PyAV. Hay una prueba automática de esto (A1-03).
2. **E3/A3 compila la tarea al mecanismo.** Un `TrajectoryCompiler` combina
   `TaskTrajectory`, `TaskBinding`, el movimiento del vehículo, el
   `MechanismModel` y los límites, y decide si la tarea es ejecutable y cómo. Si
   no lo es, emite un `InfeasibilityReport`. Esta entrega **no** implementa esas
   piezas: solo documenta su esquema (`docs/data_contracts.md` §5 bis). No hay un
   compilador ficticio que devuelva éxito.
3. **Marcos y unidades.**
   - `T_AB` transforma coordenadas de B a A.
   - `W` es suelo en metros: `x` lateral a la derecha del avance, `y` longitudinal
     hacia delante.
   - `I` es imagen en píxeles.
   - `TaskPath` y `LocalMap` solo admiten dominios métricos de suelo
     (`ground_metric_synthetic`, `ground_assumed`, `ground_metric_verified`), que
     son modos separados.
   - `image_px` y el SVG (`graphic_unscaled`) se rechazan como coordenadas de
     tarea (`UnitMismatchError`).
4. **Progreso y tiempo son contratos distintos.**
   - `TaskPath = p(s)`, con `s ∈ [0, 1]` adimensional. `s` no es tiempo ni la
     coordenada `y`, y las derivadas respecto de `s` no son velocidades.
   - `TaskTrajectory = p(t)` es un contrato futuro que referenciará un `TaskPath`
     y una ley temporal explícita. Su primer modo será `TIMED_REFERENCE`.
   - E2 no asigna tiempos a partir de FPS ni de la velocidad del tractor.
5. **Orientación explícita.**
   - `OrientationRequirement` acepta `FIXED`, `INTERVAL` o `FREE`; `FREE` no
     significa cero.
   - E2 implementa solo `FIXED`. Las otras dos devuelven
     `PlanStatus.UNSUPPORTED` / `UNSUPPORTED_FEATURE`, que se distingue de una
     imposibilidad geométrica.
   - Que E2 certifique una orientación fija **no implica** que el brazo 1R pueda
     mantenerla físicamente; eso lo decide la compilación.
6. **`ToolProfile` = `ToolGeometry`.**
   - Contiene los polígonos locales de colisión y de corte en `B`, unidades,
     procedencia e identidad de contenido (SHA-256). No contiene `q`, topología,
     montaje ni pivote físico del brazo.
   - Se extrajo a `geometry/tool_profile.py` para que el planificador no dependa del
     simulador. `simulation.tool` la reexporta, de modo que hay una sola
     implementación y las importaciones de E1 siguen funcionando.
   - El campo heredado `orientation_mode` no gobierna la tarea.
7. **Identidad y versión.** Mapa, perfil, camino, solicitud, resultado e informe
   llevan versión de esquema y SHA-256 del contenido canónico. El validador rechaza
   como entrada inválida una ruta ligada a otro mapa u otro perfil. Cambiar la
   geometría, el mapa o una tolerancia cambia el hash e invalida el informe
   anterior.
8. **Aceptación geométrica, no mecánica.** `ACCEPTED` se conserva del enum E1, pero
   el resultado y el informe declaran `validation_scope=geometric` y
   `mechanical_feasibility=not_evaluated`. Nunca se exporta como ejecución
   aprobada.

## Compatibilidad con mecanismos futuros

Para un mismo mapa, perfil, requisitos y semilla, el `TaskPath` es idéntico sea
cual sea el mecanismo que lo consuma después. Una prueba con dos dobles de test
(esquemas `q` de dimensión 1 y 3) consume el mismo artefacto serializado sin
modificarlo (A1-03). Los dobles no son modelos mecánicos.

Las extensiones, por ejemplo giro, elevación o dos brazos en cadena abierta o
cerrada, se añaden en el compilador y en el modelo mecánico
(adaptación §7) sin editar el planificador. Una capacidad que falte se declara
`UNSUPPORTED`; nunca se presenta como imposibilidad.

## Consecuencias

- `docs/technical_plan.md` §3.1, §3.2, §3.3 y E3 remiten a este ADR. El
  planificador ya no recibe `ToolState`, y el «movimiento X–Y relativo al
  vehículo» deja de ser la hipótesis del mecanismo.
- `KinematicsConfig` (límites X–Y de E1) queda como fixture cartesiano heredado.
  No se mapea automáticamente a `q`.
- Los contratos de tarea quedan materializados y probados en `core/trajectory.py`,
  `mapping/local_map.py`, `planning/base.py` y `validation/swept_volume.py`.
