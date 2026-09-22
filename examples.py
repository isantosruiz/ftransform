"""Ejemplos ejecutables: python examples.py."""

from sympy import (
    Abs, Derivative, DiracDelta, Function, Heaviside, I, Integral, Piecewise,
    S, Sum, cos, exp, oo, pi, sin, symbols,
)
from fourier_transf import FourierTransf, fourier_transf


def main():
    t, f, f0, t0, tau = symbols("t f f0 t0 tau", real=True)
    a, T = symbols("a T", positive=True)
    n = symbols("n", integer=True)
    g, h = Function("g"), Function("h")
    examples = {
        "Impulso desplazado": DiracDelta(t-t0),
        "Constante": S.One,
        "Coseno": cos(2*pi*f0*t),
        "Seno": sin(2*pi*f0*t),
        "Escalón": Heaviside(t),
        "Exponencial causal": exp(-a*t)*Heaviside(t),
        "Retardo y modulación": exp(2*pi*I*f0*t)*g(t-t0),
        "Escala y retardo": g(-2*t+t0),
        "Derivada temporal": Derivative(g(t), t),
        "Convolución temporal": Integral(g(tau)*h(t-tau), (tau, -oo, oo)),
        "Integración temporal": Integral(g(tau), (tau, -oo, t)),
        "Pulso rectangular": Heaviside(t+T/2)-Heaviside(t-T/2),
        "Pulso triangular": Piecewise((1-2*Abs(t/T), Abs(t/T) <= S.Half), (0, True)),
        "Tren de impulsos": Sum(DiracDelta(t-n*T), (n, -oo, oo)),
    }
    for label, signal in examples.items():
        print(f"{label}:\n  {signal}\n  -> {fourier_transf(signal, t, f)}\n")
    pending = FourierTransf(DiracDelta(t), t, f)
    print("Sin evaluar:", pending)
    print("Evaluada:", pending.doit())
    print("Integral:", pending.rewrite(Integral))


if __name__ == "__main__":
    main()
