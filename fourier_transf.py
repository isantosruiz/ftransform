"""Fourier transforms in the convention of B. P. Lathi (frequency in Hz).

The kernel is exp(-2*pi*I*t*f), exactly as in SymPy.  This module adds
distributional pairs and operational rules; it does not patch SymPy.
Singular rational terms in distributional answers use the principal-value
(and, after differentiation, finite-part) convention.  See README.md.
"""

from dataclasses import dataclass

from sympy import (
    Abs, Add, And, Derivative, DiracDelta, Dummy, Heaviside, I, Integral,
    Mul, Piecewise, Poly, S, Sum, Symbol, binomial, cancel, conjugate, cos, diff, exp,
    expand_mul, factorial, oo, pi, re, sign, simplify as sympy_simplify, sin, sinc, sqrt,
    sympify,
)
from sympy.core.function import AppliedUndef
from sympy.polys.polyerrors import PolynomialError
from sympy.integrals.transforms import (
    FourierTransform, IntegralTransformError, fourier_transform,
)

__all__ = ["fourier_transf", "FourierTransf"]


@dataclass(frozen=True)
class _Result:
    value: object
    condition: object = S.true


def _affine(expr, x):
    """Return (a, b) for a*x+b, including a=0."""
    a = diff(expr, x)
    if a.has(x):
        return None
    b = sympy_simplify(expr - a*x)
    if b.has(x):
        return None
    return a, b


def _real_affine(expr, x):
    ab = _affine(expr, x)
    if ab and ab[0].is_real is True and ab[0].is_zero is False \
            and ab[1].is_real is True:
        return ab
    return None


def _scale(result, coefficient):
    return _Result(coefficient*result.value, result.condition)


def _combine(results):
    return _Result(Add(*(r.value for r in results)),
                   And(*(r.condition for r in results)))


def _reduce_delta_weights(value, f):
    """Apply smooth-multiplier identities before ordinary simplification."""
    terms = []
    for term in Add.make_args(expand_mul(value)):
        deltas = [v for v in Mul.make_args(term) if v.func is DiracDelta]
        if len(deltas) != 1:
            terms.append(term)
            continue
        delta = deltas[0]
        ab = _real_affine(delta.args[0], f)
        coefficient = term/delta
        order = delta.args[1] if len(delta.args) == 2 else S.Zero
        if ab is None or not order.is_Integer \
                or coefficient.has(DiracDelta, Heaviside, sign, Abs, Piecewise):
            terms.append(term)
            continue
        a, b = ab
        root = -b/a
        if coefficient.as_numer_denom()[1].subs(f, root).is_zero is not False:
            terms.append(term)
            continue
        reduced = Add(*(
            (-1)**j*binomial(order, j)/a**j
            * diff(coefficient, f, j).subs(f, root)
            * DiracDelta(delta.args[0], order-j)
            for j in range(int(order)+1)
        ))
        terms.append(term if reduced.has(S.NaN, S.ComplexInfinity, oo, -oo) else reduced)
    return Add(*terms)


class FourierTransf(FourierTransform):
    r"""Unevaluated Lathi Fourier transform: ``FourierTransf(g, t, f)``.

    Construction never evaluates the transform.  Use ``.doit()`` to evaluate
    or ``.rewrite(Integral)`` / ``.as_integral`` for its defining integral.
    The independent time/frequency coordinates are real; scalar parameters
    retain their SymPy assumptions.  Frequencies are in Hz, regardless of the
    name given to the third argument.

    Examples
    --------
    >>> from sympy import DiracDelta, symbols
    >>> t, f = symbols('t f', real=True)
    >>> FourierTransf(DiracDelta(t), t, f).doit()
    1
    """

    _name = "FourierTransf"
    nargs = 3

    @property
    def free_symbols(self):
        return ((self.function.free_symbols - {self.function_variable})
                | self.transform_variable.free_symbols)

    def doit(self, **hints):
        """Evaluate using ``simplify=True, noconds=True, needeval=False``.

        ``noconds=False`` returns ``(expression, condition)`` for evaluated
        results.  A wholly unknown transform remains an unevaluated object,
        just as in SymPy.  Conditions concern ordinary integral convergence;
        distributional pairs are interpreted in the sense explained above.
        ``needeval=True`` raises IntegralTransformError if any transform remains.
        ``deep`` is accepted for compatibility, without forcing inner
        integrals, derivatives, or sums to evaluate before applying the rules.
        """
        do_simplify = hints.pop("simplify", True)
        noconds = hints.pop("noconds", True)
        needeval = hints.pop("needeval", False)
        hints.pop("deep", None)
        if hints:
            raise TypeError("Unsupported transform hints: " + ", ".join(hints))
        x, k = self.function_variable, self.transform_variable
        if not isinstance(x, Symbol):
            raise TypeError("The time variable must be a SymPy Symbol.")
        if x.is_real is False or k.is_real is False:
            raise ValueError("Fourier time and frequency coordinates must be real.")

        # Work with fresh real coordinates, even for plain symbols.  Distinct
        # dummies also make calls with the same input/output symbol safe.
        t, f = Dummy("t", real=True), Dummy("f", real=True)
        g = self.function.xreplace({x: t})
        result = _transform(g, t, f)
        value = result.value
        if do_simplify and not value.has(Integral, Sum, FourierTransf):
            # Combining delta and PV terms into one rational fraction would
            # create undefined products at the delta's support.
            if value.has(DiracDelta):
                value = _reduce_delta_weights(value, f)
                regular = Add(*(term for term in Add.make_args(value)
                                if not term.has(DiracDelta)))
                singular = Add(*(term for term in Add.make_args(value)
                                 if term.has(DiracDelta)))
                value = singular + sympy_simplify(regular)
            else:
                value = sympy_simplify(value)
        value = value.xreplace({t: x, f: k})
        condition = result.condition.xreplace({t: x, f: k})
        unresolved = value.has(FourierTransform)
        if needeval and unresolved:
            raise IntegralTransformError(self._name, self.function, "needeval")
        if noconds or (unresolved and condition is S.true):
            return value
        return value, condition


def fourier_transf(g, t, f, **hints):
    r"""Compute ``Integral(g*exp(-2*pi*I*t*f), (t, -oo, oo))``.

    Calling convention: ``fourier_transf(g, t, f, **hints)``.  Supported hints
    are ``simplify``, ``noconds``, ``needeval`` and ``deep``; see
    :meth:`FourierTransf.doit`.  Unknown cases retain ``FourierTransf``.

    >>> from sympy import cos, pi, symbols
    >>> t, f, f0 = symbols('t f f0', real=True)
    >>> fourier_transf(cos(2*pi*f0*t), t, f)
    DiracDelta(f - f0)/2 + DiracDelta(f + f0)/2
    """
    return FourierTransf(sympify(g), sympify(t), sympify(f)).doit(**hints)


def _unknown(g, t, f):
    return _Result(FourierTransf(g, t, f))


def _pulse(g, t, f):
    """Compact centered/affine rectangular and triangular Piecewise pulses."""
    if g.is_Add and len(g.args) == 2:
        left, right = (term.as_independent(t, as_Add=False) for term in g.args)
        c1, h1 = left
        c2, h2 = right
        if c1 == -c2 and h1.func is Heaviside and h2.func is Heaviside:
            ab1 = _real_affine(h1.args[0], t)
            ab2 = _real_affine(h2.args[0], t)
            if ab1 and ab2 and sign(ab1[0]) == sign(ab2[0]):
                r1, r2 = -ab1[1]/ab1[0], -ab2[1]/ab2[0]
                width = r2-r1
                return _Result(c1*sign(ab1[0])*width*sinc(pi*f*width)
                               * exp(-pi*I*f*(r1+r2)))
    if not isinstance(g, Piecewise) or len(g.args) != 2:
        return None
    (body, cond), (outside, otherwise) = g.args
    if outside != 0 or otherwise is not S.true:
        return None
    if getattr(cond, "rel_op", None) not in ("<", "<="):
        return None
    absolute_factors = [v for v in cond.lhs.atoms(Abs) if v.has(t)]
    if len(absolute_factors) != 1 or cond.rhs.has(t):
        return None
    absolute = absolute_factors[0]
    coefficient = sympy_simplify(cond.lhs/absolute)
    if coefficient.has(t) or coefficient.is_positive is not True:
        return None
    width = cond.rhs/coefficient
    ab = _real_affine(absolute.args[0], t)
    if ab is None or width.is_positive is not True:
        return None
    a, b = ab
    amplitude = body.subs(t, -b/a)
    duration = 2*width/Abs(a)
    phase = exp(2*pi*I*f*b/a)
    if not body.has(t):
        return _Result(body*duration*sinc(pi*f*duration)*phase)
    if sympy_simplify(body-amplitude*(1-absolute/width)) == 0:
        return _Result(amplitude*duration/2*sinc(pi*f*duration/2)**2*phase)
    return None


def _impulse_product(g, t, f):
    """Sifting, including affine delta derivatives and smooth multipliers."""
    factors = Mul.make_args(g)
    deltas = [v for v in factors if v.func is DiracDelta and v.has(t)]
    if len(deltas) != 1:
        return None
    delta = deltas[0]
    ab = _real_affine(delta.args[0], t)
    if ab is None:
        return None
    a, b = ab
    n = delta.args[1] if len(delta.args) == 2 else S.Zero
    if not n.is_Integer:
        return None
    multiplier = g/delta
    # Products of distributions, and multiplication at a discontinuity, need
    # extra definitions.  Do not silently assign a value to those products.
    if multiplier.has(DiracDelta, Heaviside, sign, Piecewise, Abs):
        return None
    sampled = diff(multiplier*exp(-2*pi*I*f*t), t, n).subs(t, -b/a)
    if sampled.has(S.NaN, S.ComplexInfinity, oo, -oo):
        return None
    return _Result((-1)**n*sampled/(Abs(a)*a**n))


def _sinc_pair(g, t, f):
    power = S.One
    base = g
    if g.is_Pow and g.exp == 2:
        base, power = g.base, S(2)
    if base.func is not sinc:
        # Recognize literal sin(argument)/argument, as in the textbook.
        for wave in sorted(g.atoms(sin), key=str):
            arg = wave.args[0]
            for n in (1, 2):
                coefficient = cancel(g*(arg/wave)**n)
                if not coefficient.has(t):
                    pair = _sinc_pair(sinc(arg)**n, t, f)
                    return _scale(pair, coefficient) if pair else None
        return None
    ab = _real_affine(base.args[0], t)
    if ab is None:
        return None
    a, b = ab
    if power == 1:
        band = Abs(a)/(2*pi)
        value = pi/Abs(a)*(Heaviside(f+band)-Heaviside(f-band))
    else:
        band = Abs(a)/pi
        value = pi/Abs(a)*(1-Abs(f)/band)*Heaviside(band-Abs(f))
    return _Result(exp(2*pi*I*f*b/a)*value)


def _exponential_pair(g, t, f):
    """Gaussian, two-sided exponential, and one-sided exponential pairs."""
    if g.func is exp:
        exponent = g.args[0]
        absolutes = [v for v in exponent.atoms(Abs) if v.has(t)]
        if len(absolutes) == 1:
            absolute = absolutes[0]
            c = exponent.expand().coeff(absolute)
            d = sympy_simplify(exponent-c*absolute)
            ab = _real_affine(absolute.args[0], t)
            if ab and not c.has(t) and not d.has(t):
                condition = re(c) < 0
                if condition is not S.false:
                    a, b = ab
                    rate = -c*Abs(a)
                    return _Result(exp(d+2*pi*I*f*b/a)*2*rate /
                                   (rate**2+(2*pi*f)**2), condition)
        try:
            p = Poly(exponent, t)
        except PolynomialError:
            p = None
        if p is not None and p.degree() == 2:
            rate, b, c = -p.nth(2), p.nth(1), p.nth(0)
            condition = re(rate) > 0
            if condition is not S.false:
                return _Result(sqrt(pi/rate)*exp(c+(b-2*pi*I*f)**2/(4*rate)),
                               condition)

    gates = [v for v in Mul.make_args(g) if v.func is Heaviside]
    if len(gates) == 1:
        gate = gates[0]
        ab = _real_affine(gate.args[0], t)
        rest = g/gate
        n = rest.as_powers_dict().get(t, S.Zero)
        if n.is_integer is not True or n.is_nonnegative is not True:
            return None
        exponential = rest/t**n
        if ab is None or exponential.func is not exp:
            return None
        cd = _affine(exponential.args[0], t)
        if cd is None:
            return None
        a, b = ab
        c, d = cd
        orientation = sign(a)
        condition = -orientation*re(c) > 0
        if condition is S.false:
            return None
        if n != 0 and b != 0:
            return None
        root = -b/a
        value = (exp(d+(c-2*pi*I*f)*root)*factorial(n)*orientation**(n+1)
                 / (2*pi*I*f-c)**(n+1))
        return _Result(value, condition)
    return None


def _rational_pair(g, t, f):
    """Dual of the two-sided decaying exponential, with a quadratic denominator."""
    if not g.is_Pow or g.exp != -1:
        return None
    try:
        polynomial = Poly(g.base, t)
    except PolynomialError:
        return None
    if polynomial.degree() != 2:
        return None
    a, b, c = polynomial.nth(2), polynomial.nth(1), polynomial.nth(0)
    if not all(v.is_real is True for v in (a, b, c)) or a.is_positive is not True:
        return None
    center = -b/(2*a)
    width_squared = sympy_simplify(c/a-center**2)
    if width_squared.is_positive is not True:
        return None
    width = sqrt(width_squared)
    return _Result(pi/(a*width)*exp(-2*pi*width*Abs(f)-2*pi*I*f*center))


def _integral_rule(g, t, f):
    if not isinstance(g, Integral) or len(g.limits) != 1:
        return None
    limit = g.limits[0]
    if len(limit) != 3:
        return None
    tau, low, high = limit
    if tau == t or low != -oo:
        return None
    if high == t and not g.function.has(t):
        spectrum = _transform(g.function.subs(tau, t), t, f)
        if spectrum.value.has(DiracDelta):
            return None
        dc = spectrum.value.subs(f, 0)
        if dc.has(S.NaN, S.ComplexInfinity, oo, -oo):
            return None
        return _Result(spectrum.value/(2*pi*I*f)+dc*DiracDelta(f)/2,
                       spectrum.condition)
    if high != oo:
        return None
    first, second = g.function.as_independent(t, as_Add=False)
    shifted = second.subs(t, t+tau)
    if not first.has(tau) or not shifted.has(t) or shifted.has(tau):
        return None
    left = _transform(first.subs(tau, t), t, f)
    right = _transform(shifted, t, f)
    if left.value.has(DiracDelta) and right.value.has(DiracDelta):
        return None
    return _Result(left.value*right.value, And(left.condition, right.condition))


def _sum_rule(g, t, f):
    if not isinstance(g, Sum) or len(g.limits) != 1:
        return None
    n, low, high = g.limits[0]
    term = g.function
    if low == -oo and high == oo and term.func is DiracDelta \
            and len(term.args) == 1:
        ab = _real_affine(term.args[0], t)
        if ab:
            a, b = ab
            shift = _affine(-b/a, n)
            if shift:
                period, offset = shift
                if period.is_real is True and period.is_zero is False \
                        and offset.is_real is True:
                    spacing = Abs(period)
                    return _Result(exp(-2*pi*I*f*offset)/(Abs(a)*spacing)
                                   * Sum(DiracDelta(f-n/spacing), (n, -oo, oo)))
    # Finite sums and explicit Fourier series can be transformed termwise.
    finite = low.is_finite is True and high.is_finite is True
    coefficient, wave = term.as_independent(t, as_Add=False)
    series = wave.func is exp and _affine(wave.args[0], t)
    if finite or (series and (series[0]/I).is_real is True):
        result = _transform(term, t, f)
        if not result.condition.has(n):
            return _Result(Sum(result.value, g.limits[0]), result.condition)
    return None


def _transform(g, t, f):
    if not isinstance(f, Symbol):
        # Frequency differentiation must happen before substituting a shifted
        # or scaled frequency expression.
        frequency = Dummy("frequency", real=True)
        result = _transform(g, t, frequency)
        return _Result(result.value.subs(frequency, f),
                       result.condition.subs(frequency, f))
    if not g.has(t):
        return _Result(g*DiracDelta(f))

    pulse = _pulse(g, t, f)
    if pulse is not None:
        return pulse
    if g.is_Add:
        return _combine([_transform(term, t, f) for term in g.args])

    coefficient, rest = g.as_independent(t, as_Add=False)
    if coefficient != 1:
        return _scale(_transform(rest, t, f), coefficient)

    if isinstance(g, FourierTransform) and g.transform_variable == t \
            and g.function_variable != t:
        return _Result(g.function.subs(g.function_variable, -f))

    if isinstance(g, Derivative) and all(v == t for v, _ in g.variable_count):
        order = sum(n for _, n in g.variable_count)
        return _scale(_transform(g.expr, t, f), (2*pi*I*f)**order)

    for rule in (_impulse_product, _sinc_pair, _exponential_pair, _rational_pair,
                 _integral_rule, _sum_rule):
        result = rule(g, t, f)
        if result is not None:
            return result

    if g.func in (Heaviside, sign):
        ab = _real_affine(g.args[0], t)
        if ab:
            a, b = ab
            value = sign(a)/(I*pi*f)
            if g.func is Heaviside:
                value = (DiracDelta(f)+value)/2
            return _Result(exp(2*pi*I*f*b/a)*value)

    if g.func is conjugate:
        result = _transform(g.args[0], t, -f)
        return _Result(conjugate(result.value), result.condition)

    if isinstance(g, AppliedUndef) and len(g.args) == 1:
        ab = _real_affine(g.args[0], t)
        if ab and ab != (S.One, S.Zero):
            a, b = ab
            return _Result(exp(2*pi*I*f*b/a)/Abs(a)
                           * FourierTransf(g.func(t), t, f/a))
        return _unknown(g, t, f)

    # Polynomial multiplication corresponds to frequency differentiation,
    # including factored polynomials such as (t-t0)**2.
    polynomial = Mul(*(v for v in Mul.make_args(g) if v.is_polynomial(t)))
    if polynomial != 1:
        result = _transform(g/polynomial, t, f)
        value = Add(*(coefficient*(I/(2*pi))**power[0]
                      * diff(result.value, f, power[0])
                      for power, coefficient in Poly(polynomial, t).terms()))
        return _Result(value, result.condition)

    # Strip one carrier at a time, so products and phase-shifted carriers work.
    for carrier in Mul.make_args(g):
        if carrier.is_Pow and carrier.base.func in (sin, cos) \
                and carrier.exp.is_Integer and carrier.exp > 0:
            ab = _affine(carrier.base.args[0], t)
            if ab and all(v.is_real is True for v in ab):
                a, b = ab
                degree = int(carrier.exp)
                terms = []
                for j in range(degree+1):
                    harmonic = degree-2*j
                    weight = binomial(degree, j)*exp(I*harmonic*b)
                    weight *= (S.Half**degree if carrier.base.func is cos
                               else (-1)**j/(2*I)**degree)
                    terms.append(_scale(_transform(g/carrier, t, f-harmonic*a/(2*pi)), weight))
                return _combine(terms)
        if carrier.func not in (exp, sin, cos):
            continue
        ab = _affine(carrier.args[0], t)
        if ab is None:
            continue
        a, b = ab
        other = g/carrier
        if carrier.func is exp and (a/I).is_real is True:
            return _scale(_transform(other, t, f-a/(2*pi*I)), exp(b))
        if carrier.func in (sin, cos) and a.is_real is True and b.is_real is True:
            positive = _scale(_transform(other, t, f-a/(2*pi)), exp(I*b))
            negative = _scale(_transform(other, t, f+a/(2*pi)), exp(-I*b))
            if carrier.func is cos:
                return _scale(_combine([positive, negative]), S.Half)
            return _scale(_combine([positive, _scale(negative, -1)]), 1/(2*I))

    # Frequency convolution for formal products of unspecified signals.
    factors = list(Mul.make_args(g))
    if g.is_Pow and isinstance(g.base, AppliedUndef) and g.exp.is_Integer \
            and g.exp > 1:
        factors = [g.base, g.base**(g.exp-1)]
    if len(factors) > 1 and all(
        isinstance(v, AppliedUndef) or
        (v.is_Pow and isinstance(v.base, AppliedUndef)
         and v.exp.is_Integer and v.exp > 0) for v in factors
    ):
        nu = Dummy("nu", real=True)
        left = _transform(factors[0], t, nu)
        right = _transform(Mul(*factors[1:]), t, f-nu)
        return _Result(Integral(left.value*right.value, (nu, -oo, oo)),
                       And(left.condition, right.condition))

    # These objects require distributional rules.  Ordinary quadrature can
    # return a misleading zero or drop boundary impulses, so do not delegate.
    if g.has(DiracDelta, Heaviside, sign, Integral, Sum, AppliedUndef):
        return _unknown(g, t, f)
    if g.func is exp:
        if g.has(Abs) or _affine(g.args[0], t):
            return _unknown(g, t, f)
        try:
            if Poly(g.args[0], t).degree() == 2:
                return _unknown(g, t, f)
        except PolynomialError:
            pass

    ordinary = fourier_transform(g, t, f, noconds=False)
    if isinstance(ordinary, tuple):
        value, condition = ordinary
    else:
        value, condition = ordinary, S.true
    value = value.replace(
        lambda node: type(node) is FourierTransform,
        lambda node: FourierTransf(*node.args),
    )
    if condition.has(f):
        # A result valid only on half the frequency axis is not a complete
        # spectrum.  Keep that restriction even when noconds=True.
        value = Piecewise((value, condition), (FourierTransf(g, t, f), True))
        condition = S.true
    return _Result(value, condition)
