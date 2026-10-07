# How it works

Cycles has no diffraction closure and shader nodes have no random-number node. The node group gets correct,
continuous-spectrum diffraction out of stock nodes in five steps.

**1. Grating equation.** With `wo` the view direction, `wi` the light direction, `g` the grating vector
(perpendicular to the grooves), `g2 = N × g` and `_t` the part tangent to the surface, order `(m, n)` at wavelength
`λ` reflects towards

```
wi_t = −wo_t + (λ/d)(m g + n g2)        wi_z = √(1 − |wi_t|²)        (n = 0 for 1D grooves)
```

In the plane of `g` this is the textbook `sin θi + sin θo = mλ/d`. If `|wi_t| ≥ 1` the order is evanescent (it does
not exist at that angle).

**2. Half-vector trick.** For `h = normalize(wi + wo)`, `reflect(wo, h) = wi`. A stock Glossy BSDF (GGX) whose
Normal is `h` therefore puts its highlight exactly on that order. It is an ordinary closure, so light sampling and
MIS work unchanged on CPU and GPU. Roughness blurs it.

**3. Weights.** For fixed `λ`, the map `wi_t → wo_t` is a translation in direction cosines, so projected solid
angle is preserved and each lobe's weight is simply its efficiency `η` (no `1/cos` terms). Each wavelength gets the
colour `c(λ)`: CIE 1931 × D65 → linear Rec.709, negatives clipped, white-balanced so the spectrum averages to
exactly white.

**4. Stochastic orders and wavelengths.** Summing every order at every wavelength would exceed Cycles' 64-closure
limit, so each sample evaluates K random lobes (Quality 1–4 → 4, 8, 12, 16) plus the mirror order, weighted so
the average is the full integral:

```
weight_k = η(m_k, n_k) · c(λ_k) / ( K · P(m_k, n_k) · p(λ_k) )
```

Orders are drawn with stratified sampling from a distribution that matches the efficiencies; wavelengths are
importance-sampled (`p ∝ R+G+B of c(λ)`, inverse CDF baked into a Float Curve). The estimate is unbiased: it
converges to a 288-lobe brute-force reference with ≤ 0.6 % mean bias. The random numbers come from White Noise 4D
on Geometry > Position: it hashes the float bits, so every sample (every jittered ray) gets a fresh value.
Unused lobes sit behind Mix Shader factor 0, which Cycles skips at no cost.

**5. Energy.** Every diffracted order keeps its efficiency. The share that cannot propagate at the current angle
and wavelength, computed in closed form for 1D and 2D gratings, goes into the mirror order:
`order 0 = 1 − D · f_prop`. A lossless foil then stays white in a white furnace (0.98–1.00 up to 85°) for every
setting, and turns into a mirror at grazing angles, as real gratings do. Metal Fresnel (Schlick, per lobe) and an
optional clear coat sit on top. The physical mode (Groove Depth > 0) uses `η_m = J_m(a)²` for sinusoidal grooves;
it is approximate and not energy-exact.

**Why "Balanced" differs.** It sums 30 fixed lobes (orders ±1..3 at five wavelengths) and tilts the shading normal
by `asin(sin θo − |m|λ/d)` instead of solving the grating equation. Tilting the normal deflects light by twice the
angle, it uses the total view angle instead of its component across the grooves, five wavelengths draw five
separate copies of a lamp, the weights are not normalised (about 40 % of white light is reflected), and it is 1D
only. The tests in [`scripts/tests`](../scripts/tests) check both shaders against the textbook equation.

References: J. Stam, "Diffraction Shaders", SIGGRAPH 1999. J. E. Harvey et al., "Diffracted radiance", Applied
Optics 38(31), 1999. A. Toisoul, D. S. Dhillon, A. Ghosh, "Acquiring spatially varying appearance of printed
holographic surfaces", ACM TOG 37(6), 2018.
