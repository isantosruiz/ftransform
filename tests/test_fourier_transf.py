"""Reference pairs, operational identities, and symbolic API regressions."""
import unittest

from sympy import (
    Abs, Derivative, DiracDelta, Function, Heaviside, I, Integral, Piecewise,
    S, Sum, conjugate, cos, diff, exp, factorial, integrate, limit, oo, pi,
    simplify, sin, sinc, sqrt, symbols,
)
from sympy.integrals.transforms import FourierTransform, IntegralTransformError

from fourier_transf import FourierTransf, fourier_transf

t, f, f0, t0, phase = symbols("t f f0 t0 phase", real=True)
a, T, B, sigma = symbols("a T B sigma", positive=True)
tau = symbols("tau", real=True)
n = symbols("n", integer=True)
g, h = Function("g"), Function("h")


def rect(x):
    return Piecewise((1, Abs(x) <= S.Half), (0, True))


def tri(x):
    # Lathi uses full width one, hence 1-2*Abs(x).
    return Piecewise((1-2*Abs(x), Abs(x) <= S.Half), (0, True))


class FourierTests(unittest.TestCase):
    def assertEquivalent(self, actual, expected):
        self.assertEqual(simplify((actual-expected).expand()), 0,
                         f"\nactual: {actual}\nexpected: {expected}")

    def check_pair(self, signal, spectrum):
        actual = fourier_transf(signal, t, f)
        self.assertFalse(actual.has(FourierTransform), str(actual))
        self.assertEquivalent(actual, spectrum)

    def test_table_3_1_pairs_1_to_17(self):
        order = symbols("order", integer=True, nonnegative=True)
        p, m = DiracDelta(f-f0), DiracDelta(f+f0)
        pairs = [
            (exp(-a*t)*Heaviside(t), 1/(a+2*pi*I*f)),
            (exp(a*t)*Heaviside(-t), 1/(a-2*pi*I*f)),
            (exp(-a*Abs(t)), 2*a/(a*a+(2*pi*f)**2)),
            (t*exp(-a*t)*Heaviside(t), 1/(a+2*pi*I*f)**2),
            (t**order*exp(-a*t)*Heaviside(t), factorial(order)/(a+2*pi*I*f)**(order+1)),
            (DiracDelta(t), S.One),
            (S.One, DiracDelta(f)),
            (exp(2*pi*I*f0*t), p),
            (cos(2*pi*f0*t), (p+m)/2),
            (sin(2*pi*f0*t), (p-m)/(2*I)),
            (Heaviside(t), DiracDelta(f)/2+1/(2*pi*I*f)),
            (2*Heaviside(t)-1, 1/(pi*I*f)),
            (cos(2*pi*f0*t)*Heaviside(t), (p+m)/4+2*pi*I*f/((2*pi*f0)**2-(2*pi*f)**2)),
            (sin(2*pi*f0*t)*Heaviside(t), (p-m)/(4*I)+2*pi*f0/((2*pi*f0)**2-(2*pi*f)**2)),
            (exp(-a*t)*sin(2*pi*f0*t)*Heaviside(t), 2*pi*f0/((a+2*pi*I*f)**2+(2*pi*f0)**2)),
            (exp(-a*t)*cos(2*pi*f0*t)*Heaviside(t), (a+2*pi*I*f)/((a+2*pi*I*f)**2+(2*pi*f0)**2)),
            (rect(t/T), T*sinc(pi*f*T)),
        ]
        for i, (signal, spectrum) in enumerate(pairs, 1):
            with self.subTest(pair=i):
                self.check_pair(signal, spectrum)

    def test_table_3_1_pair_18_sinc(self):
        actual = fourier_transf(2*B*sinc(2*pi*B*t), t, f)
        for value, expected in [(0, 1), (B/2, 1), (2*B, 0), (-2*B, 0), (B, S.Half)]:
            self.assertEqual(simplify(actual.subs(f, value)), expected)

    def test_table_3_1_pair_19_triangle(self):
        self.check_pair(tri(t/T), T/2*sinc(pi*f*T/2)**2)

    def test_table_3_1_pair_20_sinc_squared(self):
        actual = fourier_transf(B*sinc(pi*B*t)**2, t, f)
        for value in [0, B/2, -B/2, B, -B, 2*B]:
            self.assertEquivalent(actual.subs(f, value), tri(value/(2*B)))

    def test_table_3_1_pair_21_comb(self):
        actual = fourier_transf(Sum(DiracDelta(t-n*T), (n, -oo, oo)), t, f)
        expected = Sum(DiracDelta(f-n/T), (n, -oo, oo))/T
        self.assertEqual(actual, expected)

    def test_table_3_1_pair_22_gaussian(self):
        self.check_pair(exp(-t**2/(2*sigma**2)), sigma*sqrt(2*pi)*exp(-2*(sigma*pi*f)**2))

    def test_impulse_shifting_scaling_and_derivatives(self):
        for scale in [2, -3, a, -a]:
            for order in range(4):
                with self.subTest(scale=scale, order=order):
                    self.check_pair(DiracDelta(scale*t-t0, order),
                                    (2*pi*I*f)**order*exp(-2*pi*I*f*t0/scale)
                                    / (Abs(scale)*scale**order))

    def test_sifting_with_smooth_multiplier(self):
        self.check_pair(exp(-t**2)*DiracDelta(t-t0), exp(-t0**2-2*pi*I*f*t0))
        self.check_pair(t*DiracDelta(t, 1), -S.One)
        self.check_pair(t**2*DiracDelta(t, 2), S(2))

    def test_undefined_distribution_products_remain_unevaluated(self):
        for signal in [DiracDelta(t)**2, Heaviside(t)*DiracDelta(t), DiracDelta(t)/t]:
            with self.subTest(signal=signal):
                self.assertTrue(fourier_transf(signal, t, f).has(FourierTransf))

    def test_linearity_and_unknown_terms(self):
        self.assertEquivalent(fourier_transf(3*g(t)-2*h(t)+7, t, f),
                              3*FourierTransf(g(t), t, f)-2*FourierTransf(h(t), t, f)+7*DiracDelta(f))

    def test_affine_unknown_signal(self):
        for scale in [2, -3, a, -a]:
            self.assertEquivalent(fourier_transf(g(scale*t-t0), t, f),
                                  exp(-2*pi*I*f*t0/scale)/Abs(scale)*FourierTransf(g(t), t, f/scale))

    def test_unknown_scale_sign_but_nonzero(self):
        scale = symbols("scale", real=True, nonzero=True)
        self.assertEquivalent(fourier_transf(g(scale*t), t, f),
                              FourierTransf(g(t), t, f/scale)/Abs(scale))

    def test_shift_and_modulation_compose(self):
        signal = exp(2*pi*I*f0*t)*g(t-t0)
        expected = exp(-2*pi*I*(f-f0)*t0)*FourierTransf(g(t), t, f-f0)
        self.assertEquivalent(fourier_transf(signal, t, f), expected)

    def test_phase_modulation(self):
        for trig in [sin, cos]:
            p = exp(I*phase)*FourierTransf(g(t), t, f-f0)
            m = exp(-I*phase)*FourierTransf(g(t), t, f+f0)
            expected = (p+m)/2 if trig == cos else (p-m)/(2*I)
            self.assertEquivalent(fourier_transf(g(t)*trig(2*pi*f0*t+phase), t, f), expected)

    def test_carrier_products(self):
        self.check_pair(cos(2*pi*f0*t)**2,
                        DiracDelta(f)/2+(DiracDelta(f-2*f0)+DiracDelta(f+2*f0))/4)

    def test_time_derivative(self):
        for order in [1, 2, 3]:
            self.assertEquivalent(fourier_transf(Derivative(g(t), (t, order)), t, f),
                                  (2*pi*I*f)**order*FourierTransf(g(t), t, f))

    def test_frequency_derivative(self):
        self.assertEquivalent(fourier_transf(t**2*g(t), t, f),
                              -diff(FourierTransf(g(t), t, f), f, 2)/(4*pi**2))
        self.check_pair(t**3, (I/(2*pi))**3*DiracDelta(f, 3))

    def test_modulated_frequency_derivative(self):
        self.check_pair(t**2*exp(2*pi*I*f0*t), -DiracDelta(f-f0, 2)/(4*pi**2))

    def test_shifted_polynomial_times_causal_exponential(self):
        signal = (t-t0)**2*exp(-a*(t-t0))*Heaviside(t-t0)
        self.check_pair(signal, 2*exp(-2*pi*I*f*t0)/(a+2*pi*I*f)**3)

    def test_conjugation(self):
        self.assertEqual(fourier_transf(conjugate(g(t)), t, f),
                         conjugate(FourierTransf(g(t), t, -f)))

    def test_duality(self):
        self.assertEqual(fourier_transf(FourierTransf(g(tau), tau, t), t, f), g(-f))
        self.assertEqual(fourier_transf(FourierTransform(g(tau), tau, t), t, f), g(-f))

    def test_time_convolution(self):
        convolution = Integral(g(tau)*h(t-tau), (tau, -oo, oo))
        self.assertEqual(fourier_transf(convolution, t, f),
                         FourierTransf(g(t), t, f)*FourierTransf(h(t), t, f))

    def test_frequency_convolution(self):
        result = fourier_transf(g(t)*h(t), t, f)
        self.assertIsInstance(result, Integral)
        nu = result.variables[0]
        self.assertEqual(result.function,
                         FourierTransf(g(t), t, nu)*FourierTransf(h(t), t, f-nu))
        self.assertEqual(result.limits, ((nu, -oo, oo),))

    def test_time_integration(self):
        result = fourier_transf(Integral(g(tau), (tau, -oo, t)), t, f)
        expected = FourierTransf(g(t), t, f)/(2*pi*I*f)+FourierTransf(g(t), t, 0)*DiracDelta(f)/2
        self.assertEquivalent(result, expected)

    def test_time_integration_known_function(self):
        signal = Integral(exp(-a*Abs(tau)), (tau, -oo, t))
        self.check_pair(signal, 2*a/(a*a+(2*pi*f)**2)/(2*pi*I*f)+DiracDelta(f)/a)

    def test_invalid_cumulative_integral_not_forced(self):
        signal = Integral(S.One, (tau, -oo, t))
        self.assertTrue(fourier_transf(signal, t, f).has(FourierTransf))

    def test_shifted_comb(self):
        signal = Sum(DiracDelta(t-t0-n*T), (n, -oo, oo))
        self.assertEqual(fourier_transf(signal, t, f),
                         exp(-2*pi*I*f*t0)*Sum(DiracDelta(f-n/T), (n, -oo, oo))/T)

    def test_fourier_series(self):
        c = Function("c")
        signal = Sum(c(n)*exp(2*pi*I*n*t/T), (n, -oo, oo))
        self.assertEqual(fourier_transf(signal, t, f),
                         Sum(c(n)*DiracDelta(f-n/T), (n, -oo, oo)))

    def test_shifted_exponential(self):
        self.check_pair(exp(-a*Abs(t-t0)),
                        exp(-2*pi*I*f*t0)*2*a/(a*a+(2*pi*f)**2))
        self.check_pair(exp(-a*(t-t0))*Heaviside(t-t0),
                        exp(-2*pi*I*f*t0)/(a+2*pi*I*f))

    def test_shifted_scaled_pulses(self):
        self.check_pair(rect((-2*t+t0)/T), T/2*sinc(pi*f*T/2)*exp(-pi*I*f*t0))
        self.check_pair(tri((t-t0)/T), T/2*sinc(pi*f*T/2)**2*exp(-2*pi*I*f*t0))

    def test_pulse_dc_value(self):
        self.assertEqual(fourier_transf(rect(t/T), t, 0), T)
        self.assertEqual(fourier_transf(tri(t/T), t, 0), T/2)

    def test_window_written_with_steps(self):
        window = Heaviside(t+T/2)-Heaviside(t-T/2)
        self.check_pair(window, T*sinc(pi*f*T))
        self.assertEqual(fourier_transf(window, t, 0), T)
        self.check_pair(Heaviside(-t)-Heaviside(-t-T),
                        T*sinc(pi*f*T)*exp(pi*I*f*T))

    def test_distributional_derivative_of_step(self):
        self.check_pair(Derivative(Heaviside(t), t, evaluate=False), S.One)
        self.check_pair(Derivative(t, (t, 2), evaluate=False), S.Zero)

    def test_delta_terms_are_not_put_over_singular_denominators(self):
        spectrum = fourier_transf(Heaviside(t)*cos(2*pi*f0*t), t, f)
        for term in spectrum.args:
            if term.has(DiracDelta):
                self.assertFalse(term.as_numer_denom()[1].has(f))

    def test_undefined_convolution_product_is_not_forced(self):
        convolution = Integral(Heaviside(tau)*Heaviside(t-tau), (tau, -oo, oo))
        self.assertTrue(fourier_transf(convolution, t, f).has(FourierTransf))

    def test_literal_sinc(self):
        self.check_pair(sin(a*t)/(a*t), pi/a*(Heaviside(f+a/(2*pi))-Heaviside(f-a/(2*pi))))

    def test_conditions(self):
        rate = symbols("rate", real=True)
        result, condition = fourier_transf(exp(-rate*t)*Heaviside(t), t, f, noconds=False)
        self.assertEquivalent(result, 1/(rate+2*pi*I*f))
        self.assertEqual(condition, rate > 0)
        self.assertEqual(fourier_transf(DiracDelta(t), t, f, noconds=False), (1, S.true))

    def test_growth_is_not_assumed_to_decay(self):
        for signal in [exp(a*t)*Heaviside(t), exp(t), exp(a*Abs(t)), exp(a*t**2)]:
            with self.subTest(signal=signal):
                self.assertTrue(fourier_transf(signal, t, f).has(FourierTransf))

    def test_missing_parameter_assumptions(self):
        scale = symbols("scale")
        for signal in [g(scale*t), DiracDelta(scale*t), sin(scale*t)]:
            with self.subTest(signal=signal):
                self.assertTrue(fourier_transf(signal, t, f).has(FourierTransf))

    def test_symbolic_api(self):
        transform = FourierTransf(g(t), t, f)
        self.assertIsInstance(transform, FourierTransform)
        self.assertEqual(transform.function, g(t))
        self.assertEqual(transform.function_variable, t)
        self.assertEqual(transform.transform_variable, f)
        self.assertEqual(transform.doit(), transform)
        self.assertEqual(transform.as_integral, Integral(g(t)*exp(-2*pi*I*f*t), (t, -oo, oo)))
        self.assertEqual(transform.rewrite(Integral), transform.as_integral)
        self.assertEqual(FourierTransf(1, t, f).args, (S.One, t, f))

    def test_free_symbols_and_substitution(self):
        self.assertEqual(FourierTransf(g(t), t, f/a).free_symbols, {f, a})
        self.assertEqual(FourierTransf(exp(-t*t), t, t).free_symbols, {t})
        self.assertEqual(FourierTransf(DiracDelta(t-t0), t, f).subs(t0, 2).doit(), exp(-4*pi*I*f))

    def test_plain_symbols_and_shared_coordinate(self):
        x, k = symbols("x k")
        self.assertEqual(fourier_transf(DiracDelta(x), x, k), 1)
        self.assertEqual(fourier_transf(1, x, x), DiracDelta(x))
        self.assertEquivalent(fourier_transf(cos(2*pi*x), x, k),
                              (DiracDelta(k-1)+DiracDelta(k+1))/2)

    def test_needeval_and_hints(self):
        with self.assertRaises(IntegralTransformError):
            fourier_transf(1+g(t), t, f, needeval=True)
        with self.assertRaises(TypeError):
            fourier_transf(g(t), t, f, typo=True)
        self.assertEqual(fourier_transf(DiracDelta(t), t, f, deep=False, simplify=False), 1)

    def test_invalid_coordinates(self):
        with self.assertRaises(TypeError):
            fourier_transf(g(t), t+1, f)
        with self.assertRaises(ValueError):
            fourier_transf(g(t), t, I)

    def test_dual_exponential_pair(self):
        self.check_pair(1/(1+t**2), pi*exp(-2*pi*Abs(f)))

    def test_ordinary_fallback_keeps_frequency_restriction(self):
        signal = 1/(1+t**2)**2
        result = fourier_transf(signal, t, f)
        self.assertEquivalent(result.subs(f, 1).rewrite(exp), pi*(1+2*pi)*exp(-2*pi)/2)
        self.assertTrue(result.subs(f, -1).has(FourierTransf))

    def test_unknown_fallback_uses_new_class(self):
        from sympy import cosh
        result = fourier_transf(1/cosh(t), t, f)
        self.assertIs(type(result), FourierTransf)

    def test_reference_integrals_for_regular_signals(self):
        # Independent integral checks at nontrivial numeric frequencies.
        signals = [exp(-t*t), exp(-2*t)*Heaviside(t), rect(t/2), tri(t/2)]
        for signal in signals:
            value = fourier_transf(signal, t, S(1)/3)
            reference = integrate(signal*exp(-2*pi*I*t/3), (t, -oo, oo))
            self.assertLess(abs(complex((value-reference).evalf())), 1e-10)


if __name__ == "__main__":
    unittest.main()
