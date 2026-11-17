# Desmalezadora intrafilar de precisión — banco de concepto (offline)

Banco de software para estudiar cómo guiar una herramienta de desmalezado **entre las
plantas de una misma hilera** a partir de video: se detectan las plantas, se construye un
mapa local del suelo, se traza una ruta en zigzag que rodea cada tallo y se exporta la
consigna lateral que seguiría el brazo.

Es un concepto, no una máquina: no controla tractor ni actuadores y no valida mecánica.

> **Ninguna cifra de este repositorio es una medición de campo.** Escala, dimensiones de
> herramienta, protecciones y velocidades son supuestos declarados
> ([`docs/assumptions.md`](docs/assumptions.md)). Los resultados sobre video llevan el
> dominio `ground_assumed`.

## 1. Planteamiento

Una hilera de plantas separadas una distancia $\Delta y$ avanza bajo una cámara rígida al
vehículo. La herramienta tiene una huella convexa $K$ y debe trabajar el suelo de los
espacios entre plantas (*gaps*) sin invadir una zona protegida alrededor de cada tallo.
El problema se separa en cuatro partes independientes:

1. **Percepción y mapa:** dónde están las plantas en el suelo.
2. **Planificación:** una curva suave para la herramienta.
3. **Validación:** una prueba geométrica de que la huella barrida no toca ninguna zona
   protegida. Quien propone la ruta no es quien la acepta.
4. **Consigna:** la posición lateral que debe tener la herramienta en cada cuadro.

## 2. Marcos y unidades

| Marco | Definición |
|---|---|
| `I` | Imagen: $(u, v)$ en píxeles, origen arriba a la izquierda, $v$ hacia abajo. |
| `W` | Suelo: $x$ lateral (derecha del avance), $y$ longitudinal (hacia delante), en metros. |
| `V` | Vehículo, con pose $T_{WV}(t)$. |
| `B` | Herramienta, con origen en su pivote. |

Los dominios espaciales no se mezclan: `image_px`, `ground_assumed`,
`ground_metric_synthetic`, `ground_metric_verified`. Una ruta nunca se compila desde
píxeles como si fueran metros.

## 3. De la imagen al suelo

**Proyección supuesta.** Suelo plano paralelo a la imagen y escala uniforme $s$ [m/px].
El suelo se mueve en la imagen como una traslación acumulada $D_t$ desde el cuadro ancla
del segmento: un punto fijo del suelo visto en $q$ en el ancla se ve en $p = q + D_t$ en
el cuadro $t$. Sus coordenadas en `W` son

$$x = s\,(q_u - c_u), \qquad y = s\,\sigma\,(q_v - c_v),$$

con $c$ el centro de la imagen y $\sigma = \pm 1$ según el sentido de avance. Es una
transformación afín, no una homografía calibrada.

**Movimiento del suelo.** $D_t$ se integra a partir de flujo óptico disperso (esquinas
Shi-Tomasi, Lucas-Kanade piramidal). Un vector vale solo si pasa la verificación
ida/vuelta; los que caen sobre vegetación (máscara de exceso de verde, ExG) se descartan
para que el viento no sesgue la estimación. La traslación entre cuadros es la **mediana**
de los vectores de suelo. Se invalida, y con ella la ruta, si hay pocos vectores, si su
dispersión (MAD) indica giro o paralaje, o si el salto es excesivo.

**Seguimiento.** Las máscaras de copa de una red de segmentación se proyectan a `W`. Los
duplicados se fusionan por IoU o contención, y cada observación se asocia uno a uno con
una planta ya seguida por distancia de centroides con compuerta. Una planta sin
observación conserva su protección durante un número fijo de cuadros; después su zona
pasa a ser suelo desconocido, que es zona prohibida.

## 4. Mapa local

El dominio permitido para la huella es

$$\mathcal{A} = \text{corredor} \cap \text{región conocida} \setminus \text{regiones desconocidas}.$$

Que no haya plantas no certifica espacio libre. Cada planta aporta una exclusión
circular de radio

$$r_{tot} = r_{tallo} + r_{inc},$$

donde $r_{inc}$ es la incertidumbre de posición declarada. La incertidumbre se aplica
una sola vez: infla las exclusiones o la herramienta, nunca ambas.

## 5. Planificador geométrico

Determinista y no completo: si no encuentra ruta, no afirma que no exista.

1. **Obstáculos de configuración.** Para huella convexa $K$ y exclusión convexa $E$, el
   punto de referencia de la herramienta no puede entrar en la suma de Minkowski
   $E \oplus (-K)$, que es la envolvente convexa de las diferencias de vértices.
2. **Niveles longitudinales** en los bordes y centros de los obstáculos, y **candidatos
   laterales** muestreados en los intervalos libres de cada nivel.
3. **Grafo acíclico** con aristas solo hacia delante. Se buscan los mejores caminos por
   programación dinámica con costo

$$J = w_L\,\frac{L}{L_0} + w_D\,\frac{\bar{A}}{d_0} + w_S\,\frac{\sum_i \Delta\psi_i^2}{\psi_0^2},$$

   (longitud, desviación lateral media respecto de la recta y cambios de rumbo). La
   seguridad nunca es un término del costo: es una restricción que decide el validador.

## 6. Curva C2: cambio de carril quíntico

Cada arista $(x_i, y_i) \to (x_{i+1}, y_{i+1})$ se convierte en

$$x(u) = x_i + \Delta x\,h(u), \qquad y(u) = y_i + \Delta y\,u, \qquad h(u) = 10u^3 - 15u^4 + 6u^5,$$

con $u \in [0, 1]$. Como $h'(0) = h'(1) = h''(0) = h''(1) = 0$, la pendiente y la
curvatura laterales se anulan en cada unión y la cadena completa es $C^2$. El avance en
$y$ es estrictamente creciente y, por ser $h$ monótona, cada tramo queda dentro del
rectángulo de sus extremos. La pendiente lateral máxima se alcanza en $u = 1/2$:

$$\left|\frac{dx}{dy}\right|_{max} = h'\!\left(\tfrac12\right)\frac{|\Delta x|}{\Delta y} = 1{,}875\,\frac{|\Delta x|}{\Delta y}.$$

## 7. Validador independiente del barrido

El validador reconstruye las huellas desde la curva, el perfil de herramienta y el mapa
(ligados por SHA-256). No recibe aristas, costos ni indicadores del planificador.

Para orientación fija y huella convexa $K$, en un subintervalo $[a, b]$ del parámetro:

- La cuerda $p(a) \to p(b)$ barre exactamente
  $H = \operatorname{conv}\big((K + p(a)) \cup (K + p(b))\big)$.
- La curva se separa de la cuerda a lo sumo

$$\delta = \frac{M\,(b-a)^2}{8}, \qquad M \ge \max_{[a,b]} \lVert p''(u) \rVert .$$

  $M$ se obtiene de forma analítica: cada componente de $p''$ es un polinomio cúbico y
  su máximo absoluto está en un extremo o en una raíz de su derivada. No hay muestreo.
- El barrido real está contenido en $H \oplus D(\delta)$. El intervalo se certifica si

$$\delta \le \delta_{max}, \qquad \operatorname{dist}(H, \text{exclusiones}) > \varepsilon + \delta, \qquad H \oplus D(\delta) \subseteq \mathcal{A}.$$

- Si no se cumple, el intervalo se biseca. Al agotar la profundidad o el presupuesto
  queda **no certificado**, y la ruta no se acepta.

El despeje que se informa es una cota inferior certificada ($\operatorname{dist} - \delta$),
no una distancia exacta.

## 8. Patrón alternante entre plantas

Para trabajar los gaps, la ruta pasa cada planta por un costado y cruza la línea de
plantación antes de la siguiente. Se coloca un **nudo** al costado de cada planta, a la
altura de su centro, desplazado

$$d = r_{tot} + \tfrac{w}{2} + c,$$

con $w$ el ancho lateral de la herramienta y $c$ una holgura. Los nudos consecutivos se
unen con el cambio de carril quíntico, de modo que cruzar exige un recorrido lateral
$|\Delta x| = 2d$ en el avance $\Delta y$ entre plantas. Con la cota de pendiente
$m_{max}$:

$$1{,}875\,\frac{2d}{\Delta y} \le m_{max} \quad\Longrightarrow\quad \Delta y \ge \frac{3{,}75\,d}{m_{max}}.$$

Con los supuestos actuales ($r_{tot} = 0{,}08$ m, $w = 0{,}10$ m, $c = 0{,}03$ m, por
tanto $d = 0{,}16$ m, y $m_{max} = 2{,}5$) resulta $\Delta y \ge 0{,}24$ m. Las plantas
del clip están a unos 0,25 m: el cruce queda en el límite, y esa es la razón geométrica
de que solo una parte de los gaps se alterne.

Reglas del patrón:

- El costado de cada planta se decide una vez y se recuerda.
- Solo se conserva el tramo que está bajo la herramienta; los nudos por delante se
  vuelven a anclar a su planta en cada cuadro.
- Un cruce nuevo solo se compromete si su certificado tiene una holgura mínima; uno
  aceptado por milímetros deja de certificarse cuando la planta se reobserva.
- Si el cruce supera la pendiente o el validador no lo certifica, la planta se pasa por
  el mismo costado y la omisión se registra con su causa.
- Todo tramo publicado lleva el certificado del validador contra el mapa del cuadro.

## 9. Consigna para el brazo

En cada cuadro se muestrea la ruta vigente en la línea de herramienta $y_h(t)$:

$$x_{ref}(t) = x\big(y_h(t)\big), \qquad \dot{x}_{ref} = \frac{dx}{dy}\,v \le m_{max}\,v,$$

con $v$ la velocidad de avance. Se exporta una fila por cuadro (posición lateral en
metros supuestos y en píxeles, desvío respecto del eje de la hilera, paso y tasa entre
cuadros). Sin ruta vigente no se inventa consigna: el estado lo declara. La tasa es la
que exigiría la consigna; no es una velocidad ejecutada ni una comprobación mecánica.

## 10. Resultados sobre el clip de referencia

Clip de 972 cuadros, dos hileras. La hilera derecha no se usó para ajustar nada.

| Medida | Hilera izquierda | Hilera derecha |
|---|---:|---:|
| Cuadros con ruta vigente | 937 | 952 |
| Cuadros con discontinuidad declarada | 5 | 5 |
| Racha más larga con ruta continua (cuadros) | 250 | 299 |
| Gaps admisibles alternados | 22 de 72 (31%) | 18 de 69 (26%) |
| Paso lateral máximo de la consigna entre cuadros | 0,045 m | 0,051 m |
| Tasa lateral exigida, máxima (avance supuesto 0,77 m/s) | 1,4 m/s | 1,5 m/s |

## 11. Limitaciones conocidas

- **Techo geométrico de la alternancia** (§8). Solo lo suben una herramienta más
  angosta, una protección menor, plantas más separadas o no cruzar del todo.
- **La proyección supuesta no describe bien la cámara:** las plantas seguidas se atrasan
  entre 10 y 18% del avance en el mapa (perspectiva o altura de la copa). La ruta lo
  absorbe re-anclándose; corregirlo requiere una toma con referencia métrica.
- El tallo se supone en el centroide de la copa; la base no es visible en el clip.
- Sin escala real ni mecánica: la compilación al brazo y sus límites no están hechos.

## 12. Estructura

```
src/precision_weeder/
  core/         contratos tipados, unidades, TaskPath (curva polinómica por tramos)
  geometry/     marcos, transformaciones, perfil de herramienta
  mapping/      mapa local (corredor, región conocida, exclusiones)
  planning/     planificador geométrico y suavizado quíntico
  validation/   validador independiente del barrido continuo
  scenarios/    escenas sintéticas y casos adversos
  video/        lectura de video, flujo aparente, caracterización
experiments/m0_exploratory/   pipeline de visión sobre video (m0p) y evaluación
configs/        parámetros (planificador, validación, seguridad, video)
tests/          pruebas unitarias, de propiedad, integración y regresión
docs/           plan técnico, arquitectura, contratos de datos y supuestos
```

## 13. Uso

Requiere Python 3.12 y [uv](https://docs.astral.sh/uv/).

```bash
uv sync --frozen --extra dev

# Pruebas (sin GPU, sin videos, sin red) y estilo
uv run pytest
uv run ruff check .

# Planificador sobre una escena sintética
uv run python scripts/demo_planner.py --output runs/demo_planner

# Ruta y consigna sobre video (requiere el clip y la exportación de detecciones)
uv run python experiments/m0_exploratory/render_route_video.py \
  --input resources/<clip>.mp4 \
  --profile experiments/m0_exploratory/profiles/real_left_row_v1.yaml \
  --output experiments/m0_exploratory/output/v1/<clip>_left_v1.mp4
```

Los videos, los pesos de la red y las salidas generadas no se versionan.
