# Transformada de Fourier de Lathi con SymPy

`fourier_transf.py` proporciona las dos interfaces solicitadas:

```python
fourier_transf(g, t, f, **hints)  # calcula la transformada
FourierTransf(g, t, f)           # conserva la operación sin evaluar
```

La definición es la ecuación (3.1a) del PDF compartido:

```math
G(f)=\int_{-\infty}^{\infty}g(t)e^{-j2\pi ft}\,dt.
```

Esta convención, con frecuencia expresada en Hz, coincide con la de SymPy.
La extensión añade pares distribucionales y reglas operacionales. No modifica
la instalación de SymPy. `FourierTransf` hereda de `FourierTransform` y conserva
su representación simbólica y su integral definitoria.

## Instalación y uso

Requiere Python 3.10 o posterior y SymPy 1.13 o posterior de la serie 1.x.
Se verificó con Python 3.13 y SymPy 1.14.0.

Desde esta carpeta puede importarse directamente. Para instalarlo en otro entorno:

```bash
python -m pip install .
```

```python
from sympy import DiracDelta, Function, Heaviside, I, cos, exp, pi, sin, symbols
from fourier_transf import fourier_transf, FourierTransf

t, f, f0, t0 = symbols('t f f0 t0', real=True)
a = symbols('a', positive=True)
g = Function('g')

fourier_transf(DiracDelta(t-t0), t, f)
# exp(-2*I*pi*f*t0)

fourier_transf(1, t, f)
# DiracDelta(f)

fourier_transf(cos(2*pi*f0*t), t, f)
# DiracDelta(f-f0)/2 + DiracDelta(f+f0)/2

fourier_transf(sin(2*pi*f0*t), t, f)
# (DiracDelta(f-f0) - DiracDelta(f+f0))/(2*I)

fourier_transf(Heaviside(t), t, f)
# DiracDelta(f)/2 + 1/(2*pi*I*f)

fourier_transf(exp(-a*t)*Heaviside(t), t, f)
# 1/(a + 2*pi*I*f)

fourier_transf(exp(2*pi*I*f0*t)*g(t-t0), t, f)
# exp(-2*pi*I*(f-f0)*t0)*FourierTransf(g(t), t, f-f0)
```

La forma algebraica impresa puede diferir de estos comentarios sin cambiar
el resultado. SymPy usa `I` para la unidad imaginaria que Lathi escribe como `j`.

## Objeto sin evaluar y opciones

```python
from sympy import Integral

F = FourierTransf(DiracDelta(t), t, f)
F.doit()                    # 1
F.rewrite(Integral)         # Integral(exp(-2*I*pi*f*t)*DiracDelta(t), (t, -oo, oo))
F.as_integral               # misma integral
F.function                  # DiracDelta(t)
F.function_variable         # t
F.transform_variable        # f
```

| Opción | Valor por defecto | Efecto |
| --- | --- | --- |
| `simplify` | `True` | Simplifica respetando la separación de impulsos y términos singulares. |
| `noconds` | `True` | Omite las condiciones sobre los parámetros de convergencia. |
| `needeval` | `False` | Si es `True`, lanza `IntegralTransformError` cuando quedan transformadas sin evaluar. |
| `deep` | aceptado | Conserva integrales, sumas y derivadas de entrada para reconocer sus propiedades antes de evaluarlas. |

```python
rate = symbols('rate', real=True)
fourier_transf(exp(-rate*t)*Heaviside(t), t, f, noconds=False)
# (1/(rate + 2*pi*I*f), rate > 0)
```

Como en SymPy, un caso completamente desconocido se mantiene como
`FourierTransf(...)`, incluso con `noconds=False`. Una condición sobre la propia
frecuencia se conserva en un `Piecewise`, también con `noconds=True`, para
evitar presentar como espectro completo un resultado válido solo en una región.

## Pares y propiedades reconocidos

Se cubren los 22 pares de la tabla 3.1 del material: exponenciales causales y
anticausales, exponencial bilateral, sus potencias de tiempo, impulsos,
constantes, exponenciales complejas, senos, cosenos, escalón, signo, sinusoides
causales y amortiguadas, pulsos rectangulares y triangulares, `sinc`, `sinc**2`,
tren de impulsos y gaussiana. También se admiten derivadas de impulsos y su
multiplicación por funciones suaves.

Las reglas se componen y también producen resultados formales para funciones
no especificadas, como `g = Function('g')`:

| Propiedad | Entrada representativa | Resultado |
| --- | --- | --- |
| Linealidad | `A*g(t) + B*h(t)` | `A*G(f) + B*H(f)` |
| Corrimiento y escala | `g(a*t+b)` | `exp(2*pi*I*f*b/a)*G(f/a)/Abs(a)` |
| Modulación compleja | `exp(2*pi*I*f0*t)*g(t)` | `G(f-f0)` |
| Modulación con coseno | `cos(2*pi*f0*t)*g(t)` | `(G(f-f0)+G(f+f0))/2` |
| Modulación con seno | `sin(2*pi*f0*t)*g(t)` | `(G(f-f0)-G(f+f0))/(2*I)` |
| Derivación temporal | `Derivative(g(t), (t,n))` | `(2*pi*I*f)**n*G(f)` |
| Multiplicación por tiempo | `t**n*g(t)` | `(I/(2*pi))**n*diff(G(f), f, n)` |
| Conjugación | `conjugate(g(t))` | `conjugate(G(-f))` |
| Dualidad | `FourierTransf(g(tau), tau, t)` | `g(-f)` |
| Convolución temporal | `Integral(g(tau)*h(t-tau), (tau,-oo,oo))` | `G(f)*H(f)` |
| Convolución frecuencial | `g(t)*h(t)` | `Integral(G(nu)*H(f-nu), (nu,-oo,oo))` |
| Integración temporal | `Integral(g(tau), (tau,-oo,t))` | `G(f)/(2*pi*I*f) + G(0)*DiracDelta(f)/2` |

Aquí `G(f)` abrevia `FourierTransf(g(t), t, f)`. La convolución frecuencial
formal reconoce productos y potencias enteras positivas de funciones no
especificadas. Las otras reglas reconocen las formas simbólicas mostradas;
no intentan identificar todas las expresiones algebraicamente equivalentes.
La regla para `t**n*g(t)` admite grados enteros concretos; el par causal
`t**n*exp(-a*t)*Heaviside(t)` admite además `n` simbólico entero no negativo.

## Pulsos, sinc y señales periódicas

La `sinc` de SymPy coincide con la del material: `sinc(x) = sin(x)/x`,
con valor 1 en cero. El pulso triangular de Lathi tiene **anchura total uno**:

```python
from sympy import Abs, Piecewise, S, Sum, oo, sinc

T = symbols('T', positive=True)
n = symbols('n', integer=True)

rectangular = Piecewise((1, Abs(t/T) <= S.Half), (0, True))
triangular = Piecewise((1-2*Abs(t/T), Abs(t/T) <= S.Half), (0, True))

fourier_transf(rectangular, t, f)  # T*sinc(pi*T*f)
fourier_transf(triangular, t, f)   # T*sinc(pi*T*f/2)**2/2

ventana = Heaviside(t+T/2) - Heaviside(t-T/2)
fourier_transf(ventana, t, 0)     # T, sin singularidad removible

tren = Sum(DiracDelta(t-n*T), (n, -oo, oo))
fourier_transf(tren, t, f)
# Sum(DiracDelta(f-n/T), (n,-oo,oo))/T
```

En los bordes del rectángulo obtenido al transformar `sinc`, se usa el valor
medio, coherente con `Heaviside(0)=1/2`. El valor asignado en puntos aislados no
cambia la distribución ni su transformada. Las series de Fourier escritas
como sumas de exponenciales complejas también se transforman término a término.

## Supuestos matemáticos y alcance

- Las coordenadas de tiempo y frecuencia se tratan como reales, incluso si se
  crearon mediante `symbols('t f')`. Para los demás parámetros deben declararse
  los supuestos: `real=True` para retardos y frecuencias; `positive=True` para
  anchuras y tasas de decaimiento; `real=True, nonzero=True` para escalas.
- Los términos `1/f` en las transformadas de escalón y signo representan
  **valor principal de Cauchy**. Las potencias singulares obtenidas por
  derivación se interpretan como partes finitas distribucionales. Esta
  semántica está documentada, no encapsulada en un nuevo tipo simbólico:
  SymPy no la conoce al aplicar después operaciones arbitrarias al resultado.
  No debe evaluarse una distribución puntualmente en su singularidad ni
  integrarse como si fuera una función ordinaria.
- Las reglas formales presuponen la existencia de las transformadas y las
  operaciones correspondientes. La integración desde menos infinito exige
  que la integral acumulada exista y que `G(0)` esté definido. No se aplicará
  esa fórmula cuando el espectro calculado contenga impulsos o sea singular
  en cero. Para funciones no especificadas estas hipótesis son responsabilidad
  del usuario.
- No se asignan valores a productos como `DiracDelta(t)**2`, ni a un impulso
  multiplicado por un escalón en su discontinuidad. Tampoco se fuerza el
  producto de dos espectros con impulsos al usar la regla de convolución.
- Las identidades de sumas infinitas se aplican a trenes de impulsos y series
  de Fourier explícitas; su existencia distribucional es una hipótesis de uso.
- Los casos ordinarios restantes se delegan a `sympy.fourier_transform`.
  Las expresiones no resueltas conservan el nombre `FourierTransf`. No se
  pretende resolver toda señal ni todo producto de distribuciones.

## Frecuencia angular e inversión

El PDF también presenta la convención angular de la ecuación (3.9). Puede
obtenerse sin cambiar la interfaz:

```python
omega = symbols('omega', real=True)
fourier_transf(exp(-a*t)*Heaviside(t), t, omega/(2*pi))
# 1/(a + I*omega)
```

El nombre del símbolo por sí solo no cambia la convención. Para simplificar
impulsos escalados en `omega`, puede usarse `expand(diracdelta=True, wrt=omega)`.
La inversa en Hz puede obtenerse mediante `fourier_transf(G, f, -t)` cuando
el espectro esté dentro de los casos soportados. No se añade una API inversa
independiente.

## Verificación y referencias

```bash
python -m unittest discover -s tests -v
python -m doctest fourier_transf.py
python examples.py
```

Las pruebas contrastan los 22 pares, propiedades y composiciones, escalas
negativas, fases, condiciones, singularidades y la interfaz simbólica. Incluyen
comparaciones independientes con integrales de señales regulares.

- Lathi, B. P. & Ding, Zhi. (2019). *Modern digital and analog communication systems*, capítulo 3, páginas
  impresas 93–123; ecuaciones (3.1a), (3.9), tablas 3.1 y 3.2.
- [Documentación oficial de transformadas de SymPy](https://docs.sympy.org/latest/modules/integrals/integrals.html#integral-transforms).
