# Arquitectura — estado tras A1/E2

**Fecha:** 25 de octubre de 2026. **Decisión de base:**
[ADR 0001](adr/0001_task_space_planning.md). **Etapas aprobadas:** E1 y la
entrega técnica de E0. **A1/E2: APPROVED el 26 de octubre de 2026; R1–R4 cerrados en la [auditoría](reviews/A1_E2_review.md). M0-P autorizado para ejecución.**

## Flujo implementado

```text
escena sintética (scenarios) ──► LocalMap (mapping) ──► GeometricPlanner (planning)
                                     │                         │ candidatos ordenados
ToolProfile (geometry.tool_profile) ─┤                         ▼
                                     └──────────────► validador independiente (validation)
                                                               │
                                     PlanResult: TaskPath + ValidationReport
                                     (scope geométrico; mecánica no evaluada)
```

Pendiente según el plan: temporalización y compilación al mecanismo (A3/E3),
calibración y replay de video (E4/E5), percepción (E6) e integración (E7).

## Módulos y dependencias

| Módulo | Contenido | Depende de | No depende de |
|---|---|---|---|
| `core.trajectory` | `TaskPath`, `PathSegment`, `OrientationRequirement` | `core` | geometría de herramienta, mapa |
| `core.identity` | JSON canónico y SHA-256 | — | — |
| `geometry.tool_profile` | `ToolGeometry` (= `ToolProfile`) | `core`, `geometry.frames` | simulador |
| `mapping.local_map` | `LocalMap` (corredor, región conocida, desconocido, exclusiones, incertidumbre aplicada) | `core` | escenarios, simulador |
| `mapping.synthetic_adapter` | escena E1 → `LocalMap` | `scenarios`, `mapping.local_map` | — |
| `planning` | `PlanRequest`, `PlanResult`, `GeometricPlanner`, suavizado | `core`, `geometry.tool_profile`, `mapping.local_map`, `validation` | cinemática, actuadores, simulador, escenarios, video, matplotlib |
| `validation.swept_volume` | `validate_task_path`, `ValidationReport` | `core`, `geometry.tool_profile`, `mapping.local_map` | planificador |
| `scenarios.planning_cases` | casos de aceptación E2 y generador de 100 semillas | `scenarios`, `mapping` | — |
| `scripts/demo_planner.py` | CLI/demo E2 | API pública | — |

Una prueba en un subproceso limpio comprueba que importar `planning`,
`validation`, `mapping.local_map` y `core.trajectory` no carga `cv2`, `av`,
`precision_weeder.{kinematics,simulation,video,scenarios}` ni `matplotlib`
(A1-03).

## Separación entre proponer y aceptar

El planificador propone y el validador decide. El validador reconstruye las
huellas a partir de la curva, el perfil y el mapa, ligados por SHA-256, y no
recibe aristas, costos ni flags del planificador. Las holguras de generación
(`generation_margin_m`) y las aproximaciones de los obstáculos de configuración
solo afectan a qué candidatos se proponen; la aceptación usa la geometría real.

## Parámetros y generalización

Todo parámetro ajustable vive en `configs/planner.yaml`, `configs/validation.yaml`
o en los YAML de E1 (`safety.yaml` da `clearance_epsilon_m` y el presupuesto de
envolvente; no se duplican). No hay ramas por nombre de escena, clip, hash,
resolución, duración ni coordenadas concretas. Los nombres de caso existen solo
en el catálogo de pruebas.

## Extra autorizado y percepción posterior

[M0-P](orders/M0_exploratory_technical_order.md) añade un preview `ground_assumed` tras el cierre A1/E2 y antes de retomar E3. Sus adaptadores permanecen aislados y consumen la API geométrica; no incorporan YOLO al planner. E6 adopta explícitamente proveedores manual/clásico/aprendido intercambiables, con el YOLO existente como primer candidato aprendido, sometido a evaluación por sesión antes de uso operativo.

## Estado de M0-P — 02-11-2026

El preview M0-P está **APROBADO**, según [dictamen M0-P, revisión 3](reviews/M0_P_review.md). Se cierran R1–R5 y se retoma A3/E3 conforme a su orden específica. La orden v1.2 incorpora la petición explícita del usuario de estabilidad entre cuadros. C2 espacial no acredita continuidad temporal; la compilación desde estado previo y límites de actuadores permanece en A3/E3. A1/E2 sigue aprobada.
