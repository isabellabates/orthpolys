import numpy as np
import matplotlib.pyplot as plt

# three-term recurrence for legendre polynomials
def legendre_polynomials(n):
    polynomials = [np.poly1d([1]), np.poly1d([1, 0])]  # P_0(x) = 1, P_1(x) = x
    for k in range(2, n + 1):
        P_k = ((2 * k - 1) * np.poly1d([1, 0]) * polynomials[-1] - (k - 1) * polynomials[-2]) / k
        polynomials.append(P_k)
    return polynomials

# three-term recurrence for chebyshev polynomials (first kind)
def chebyshev_polynomials(n):
    polynomials = [np.poly1d([1]), np.poly1d([1, 0])]  # T_0(x) = 1, T_1(x) = x
    for k in range(2, n + 1):
        T_k = 2 * np.poly1d([1, 0]) * polynomials[-1] - polynomials[-2]
        polynomials.append(T_k)
    return polynomials

# plot legendre and chebyshev polynomials
def plot_polynomials(polynomials, title):
    x = np.linspace(-1, 1, 500)
    plt.figure(figsize=(8, 6))
    for i, poly in enumerate(polynomials):
        plt.plot(x, poly(x), label=f"Degree {i}")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("P(x)")
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.legend()
    plt.grid()
    plt.show()

# gauss quad using roots of orth polys
def gauss_quadrature(f, poly):
    roots = np.roots(poly)
    weights = []

    # compute weights using the formula for orth polys
    for r in roots:
        w = 2 / ((1 - r**2) * (np.polyval(np.polyder(poly), r)**2))  # Gauss weights
        weights.append(w)

    # compute the integral approximation
    integral = sum(w * f(r) for w, r in zip(weights, roots))
    return integral, roots, weights

# example function
def exp(x):
    return np.exp(-x**2)

n = 5  # deg of the polynomial

# derive legendre and chebyshev polys
legendre_polys = legendre_polynomials(n)
chebyshev_polys = chebyshev_polynomials(n)

# plot legendre and chebyshev polys
plot_polynomials(legendre_polys, "Legendre polynomials")
plot_polynomials(chebyshev_polys, "Chebyshev polynomials (first kind)")

# apply gauss quad using legendre polynomial roots
legendre_poly = legendre_polys[-1]
legendre_integral, legendre_roots, legendre_weights = gauss_quadrature(exp, legendre_poly)

# plot legendre roots
plt.figure(figsize=(8, 6))
plt.scatter(legendre_roots, [0] * len(legendre_roots), color='blue', label='Legendre Roots', zorder=5)
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.title("Legendre Polynomial Roots")
plt.xlabel("x")
plt.ylabel("P_n(x)")
plt.grid()
plt.legend()
plt.show()

print("legendre gauss quadrature:")
print("approximate integral: ", legendre_integral)
print("roots: ", legendre_roots)
print("weights: ", legendre_weights)

# apply gauss quadrature using chebyshev polynomial roots
chebyshev_poly = chebyshev_polys[-1]
chebyshev_integral, chebyshev_roots, chebyshev_weights = gauss_quadrature(exp, chebyshev_poly)

# plot chebyshev roots
plt.figure(figsize=(8, 6))
plt.scatter(chebyshev_roots, [0] * len(chebyshev_roots), color='red', label='Chebyshev Roots', zorder=5)
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.title("Chebyshev Polynomial Roots")
plt.xlabel("x")
plt.ylabel("T_n(x)")
plt.grid()
plt.legend()
plt.show()

print("chebyshev gauss quadrature:")
print("approximate integral: ", chebyshev_integral)
print("roots: ", chebyshev_roots)
print("weights: ", chebyshev_weights)