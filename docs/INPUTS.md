# Inputs

Node group **Diffraction Grating BSDF** (in `holo_foil.blend`). Output: **BSDF**, a complete foil.
Inside a panel Blender hides the panel name, so "Grating Angle" shows as **Angle** and "Grating Type" as **Type**.

| panel | input | default | what it does |
|---|---|---|---|
| | Color | aluminium | Metal colour at normal incidence (F0). Each order gets metal Fresnel at its own angle. White = lossless. |
| Grating | Pitch (nm) | 1500 | Groove or hole spacing. Holo foil 700–1500, CD 1600, DVD 740, Blu-ray 320, "49 copies" film 2000–5000. |
| Grating | Pitch Multiplier | 1 | Multiplies Pitch. A pattern's Pitch Multiplier plugs in here. |
| Grating | Angle | 0° | Turns the grooves about the normal, starting from Tangent. A pattern's Angle plugs in here. |
| Grating | Tangent | hidden | Reference direction. Unconnected: the UV tangent (or world X without UVs). |
| Grating | Type | 0 | 0 = 1D grooves. 1 = 2D square lattice ("one light, 49 copies"). |
| Grating | Mask | 1 | Where the grating is embossed; 0 = plain mirror foil. A pattern's Mask plugs in here. |
| Orders | Orders | 3 | Highest diffraction order simulated (1–8). |
| Orders | Order 0 Strength | 0.3 | Share of the light in the plain mirror reflection. Light of orders that cannot exist at the current angle is added to it. |
| Orders | Efficiency Falloff | 0.6 | Brightness of each order relative to the previous one. |
| Orders | Groove Depth (nm) | 0 | 0 = art mode (the two inputs above). > 0 = physical sinusoidal grooves (holo foil ≈ 100–200); approximate, not energy-exact. |
| Appearance | Roughness | 0.15 | Blur of every rainbow (light size, foil imperfection). |
| Appearance | Spectral Saturation | 1 | 1 = spectral colours, 0 = silver diffraction. |
| Appearance | Coat, Coat Roughness, Coat IOR, Coat Tint | 0, 0.03, 1.5, white | Clear lacquer on top. A yellow Coat Tint over aluminium = gold foil; translucent ink also goes here. |
| Advanced | Quality | 3 | Lobes per sample: 1 = 4, 2 = 8, 3 = 12, 4 = 16. Higher = less colour noise near small lamps (about 5 % render time per step). Use 2 for HDRI-only scenes. |
| Advanced | Seed | 0 | Decorrelates two stacked gratings. |
| Advanced | Normal | hidden | Shading normal; unconnected = geometry normal. |

Opaque printed ink: mix the BSDF with a Principled BSDF, using the ink opacity as the factor.

## Patterns

Each **Holo Pattern** group outputs a grating direction and mask that vary across the surface:

| output | plug into |
|---|---|
| Angle | Angle |
| Pitch Multiplier | Pitch Multiplier |
| Mask | Mask |
| Roughness Multiplier | multiply with your Roughness |
| Normal (Diagonal Beams, Sparkle Stars, Layer) | Normal (embossed relief, optional) |

Inputs: Vector (unconnected = UV; on a 63 × 88 mm card use `UV × (1, 88/63)`), Scale, Seed and per-pattern
controls. Patterns: Linear Rainbow, Radial CD, Scratched Metal, Glitter Patches, Cosmos, Cracked Ice, Sparkle Stars,
Diagonal Beams, Dot Matrix, Image Driven (hue of a painted map = groove direction) and Layer (stacks two patterns).

**Holo Card Shader** wraps the BSDF with printed art, a foil mask, white ink and a laminate. The **Holo Card**
material is a ready template: swap in your art and masks, and swap the pattern node for any other.

## Denoising

OIDN tends to darken sparse rainbow highlights. Below about 256 samples use the **DG Foil Denoise** compositor group
(the demo scenes are already set up):

1. View Layer > Passes: add the Shader AOV `dg_foil` (Value) and tick Data > Denoising Data.
2. Render > Denoise: off.
3. Compositor: Render Layers → DG Foil Denoise (Image, Denoising Albedo, Denoising Normal, `dg_foil` → Foil Mask)
   → output.

Without the compositor, Denoise > Passes = None works better on foil than Albedo + Normal.

## Limits

* Colour noise near small, bright lamps at low sample counts (raise Quality or samples, or use the denoiser).
* Rec.709 (Blender default) working space only. EEVEE is untested.
* Order efficiencies in art mode are an art control, not a measured groove profile.
