# Estado actual del proyecto

**Actualización:** 16 de noviembre de 2026, cambio de alcance a solo visión y orden V1.
**Dirección técnica e implementación:** equipo del proyecto.
**Aprueba:** el usuario.

| Etapa o puerta | Estado | Continuidad |
|---|---|---|
| E0 | **APPROVED** (entrega técnica y diagnóstico). Puerta positiva de datos **PENDING_BASE_VISIBILITY**. | Clip `moringa_sin_ai`: uso exploratorio, sin verdad de base ni escala métrica. |
| E1 | **APPROVED**. | Mantener contratos. |
| A1/E2 | **APPROVED**. | Planner y validador geométrico sin cambios. |
| M0-P | **APPROVED** como preview exploratorio ([dictamen rev. 3](reviews/M0_P_review.md), 2-11-2026). | Línea base de V1; sus perfiles y salidas se conservan. |
| A3/E3 | **PAUSED** (16-11-2026). Implementado solo el paso 1: contratos temporales (`core/temporal.py`, `core/compilation.py`), `kinematics/base.py`, `single_pivot_swing.py`, `configs/mechanism.yaml` y sus pruebas. | Se retoma cuando el alcance vuelva a incluir mecánica; [orden](orders/A3_E3_technical_order.md) conservada. |
| V1 | **EN CURSO**: ruta zigzag persistente y señal para el brazo, solo visión. | [Orden V1](orders/V1_vision_order.md). |
| E4–E7 | Planificadas; dependen de datos y de la reanudación de A3/E3. | [Plan maestro](technical_plan.md). |

Documentos vigentes: [memoria de dirección](project_memory.md), [orden V1](orders/V1_vision_order.md),
[plan maestro](technical_plan.md), [contratos](data_contracts.md), [supuestos](assumptions.md).

Video: `resources/moringa_sin_ai.mp4`, 43.744.466 bytes, SHA-256
`63f43e111c3a8031aafcdc738eef32ed47f2e12b877e87fcce0817a226e9e083`. Escala y montaje
desconocidos; los PTS del archivo no equivalen a reloj de captura certificado.
