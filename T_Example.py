# STP Texture Function Code (TFC) example
# Archivo: T_Example.py
# Descripción: Ejemplo autocontenido de un 'Texture Function Code' que representa
# una textura procedimental generada por el motor (Smart Texture Pooling).
# El archivo incluye:
# - Estructura de descriptor (TextureDescriptor)
# - Un intérprete simple (TexNet-lite) que genera mapas: albedo, normal, roughness
# - Export a PNG for each map (requiere numpy + pillow)

# Nota: Este script es un ejemplo educativo y prototipo. En un motor real el
# TFC sería mucho más compacto y la reconstrucción estaría optimizada
# (hardware/SIMD/SPU/GPU). Aquí se muestra la lógica completa para incluir
# en documentación técnica.

import math
import json
import os
from dataclasses import dataclass, asdict
from typing import Tuple

import numpy as np
from PIL import Image

# -----------------------------
# Utilities: simple noise funcs
# -----------------------------

def lerp(a, b, t):
    return a + (b - a) * t


def fade(t):
    # 6t^5 - 15t^4 + 10t^3 (Perlin fade)
    return t * t * t * (t * (t * 6 - 15) + 10)


def value_noise_2d(shape, scale=8.0, seed=0):
    """Genera ruido de valor 2D simple (no optimizado)."""
    np.random.seed(seed)
    w, h = shape
    # grid size
    gx = int(max(2, w / scale))
    gy = int(max(2, h / scale))
    grid = np.random.rand(gx + 1, gy + 1)

    xs = np.linspace(0, gx, w, endpoint=False)
    ys = np.linspace(0, gy, h, endpoint=False)
    ix = xs.astype(int)
    iy = ys.astype(int)
    fx = xs - ix
    fy = ys - iy

    fx_f = fade(fx)
    fy_f = fade(fy)

    out = np.zeros((h, w), dtype=np.float32)
    for j in range(h):
        for i in range(w):
            xix = ix[i]
            yiy = iy[j]
            v00 = grid[xix, yiy]
            v10 = grid[xix + 1, yiy]
            v01 = grid[xix, yiy + 1]
            v11 = grid[xix + 1, yiy + 1]
            sx = fx_f[i]
            sy = fy_f[j]
            a = lerp(v00, v10, sx)
            b = lerp(v01, v11, sx)
            out[j, i] = lerp(a, b, sy)
    return out


def fbm(shape, octaves=4, lacunarity=2.0, gain=0.5, scale=8.0, seed=0):
    """Fractional Brownian Motion based on value noise."""
    h, w = shape
    out = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    freq = scale
    for o in range(octaves):
        n = value_noise_2d((w, h), scale=freq, seed=seed + o * 13)
        out += n * amp
        amp *= gain
        freq *= lacunarity
    # normalize
    out = (out - out.min()) / (out.max() - out.min() + 1e-9)
    return out


# -----------------------------
# Texture Descriptor (TFC)
# -----------------------------

@dataclass
class TextureDescriptor:
    name: str
    type: str  # 'albedo', 'fabric', 'metal', etc.
    base_color: Tuple[float, float, float]
    variation_seed: int
    pattern_scale: float
    grunge_amount: float
    roughness_mean: float
    roughness_variation: float
    normal_strength: float
    resolution: int

    def to_json(self):
        return json.dumps(asdict(self), indent=2)


# -----------------------------
# TexNet-lite: simple interpreter
# -----------------------------

class TexNetLite:
    """Intérprete muy simple que genera mapas a partir de TextureDescriptor.
    Está pensado para ilustrar el concepto STP: un motor puede almacenar
    únicamente el descriptor (pocos KB) y regenerar texturas.
    """

    def __init__(self, desc: TextureDescriptor):
        self.desc = desc
        self.res = desc.resolution

    def generate_albedo(self) -> np.ndarray:
        r, g, b = self.desc.base_color
        base = np.ones((self.res, self.res, 3), dtype=np.float32)
        base[:, :, 0] *= r
        base[:, :, 1] *= g
        base[:, :, 2] *= b

        # base pattern - fBM
        pattern = fbm((self.res, self.res), octaves=5, scale=self.desc.pattern_scale, seed=self.desc.variation_seed)
        pattern = pattern[..., None]

        # grunge overlay
        grunge = value_noise_2d((self.res, self.res), scale=self.desc.pattern_scale * 0.25, seed=self.desc.variation_seed + 97)
        grunge = grunge[..., None]

        color = base * (0.7 + 0.3 * pattern)  # modulate base by pattern
        color = color * (1.0 - self.desc.grunge_amount * 0.6) + grunge * (self.desc.grunge_amount * 0.6)

        # subtle hue variation
        hue_shift = fbm((self.res, self.res), octaves=3, scale=self.desc.pattern_scale * 2.0, seed=self.desc.variation_seed + 42)
        color[:, :, 0] = np.clip(color[:, :, 0] * (0.9 + 0.2 * hue_shift), 0.0, 1.0)
        color[:, :, 1] = np.clip(color[:, :, 1] * (0.9 + 0.15 * hue_shift), 0.0, 1.0)
        color[:, :, 2] = np.clip(color[:, :, 2] * (0.95 + 0.1 * hue_shift), 0.0, 1.0)

        return (color * 255.0).astype(np.uint8)

    def generate_roughness(self) -> np.ndarray:
        base = np.ones((self.res, self.res), dtype=np.float32) * self.desc.roughness_mean
        var = fbm((self.res, self.res), octaves=4, scale=self.desc.pattern_scale * 1.5, seed=self.desc.variation_seed + 11)
        rough = np.clip(base * (1.0 + (var - 0.5) * self.desc.roughness_variation), 0.0, 1.0)
        return (rough * 255.0).astype(np.uint8)

    def generate_normal(self) -> np.ndarray:
        # generate height via fbm
        height = fbm((self.res, self.res), octaves=6, scale=self.desc.pattern_scale * 0.5, seed=self.desc.variation_seed + 77)
        # compute normals from height map
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = sobel_x.T

        # pad and convolve
        h = height
        gx = np.zeros_like(h)
        gy = np.zeros_like(h)
        pad = 1
        hp = np.pad(h, pad, mode='edge')
        for y in range(self.res):
            for x in range(self.res):
                region = hp[y:y + 3, x:x + 3]
                gx[y, x] = np.sum(region * sobel_x)
                gy[y, x] = np.sum(region * sobel_y)

        strength = self.desc.normal_strength
        nz = np.ones_like(gx) * (1.0 / strength)
        nx = -gx * strength
        ny = -gy * strength

        # normalize
        norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
        nx /= norm
        ny /= norm
        nz /= norm

        normal = np.zeros((self.res, self.res, 3), dtype=np.uint8)
        normal[:, :, 0] = ((nx * 0.5 + 0.5) * 255).astype(np.uint8)
        normal[:, :, 1] = ((ny * 0.5 + 0.5) * 255).astype(np.uint8)
        normal[:, :, 2] = ((nz * 0.5 + 0.5) * 255).astype(np.uint8)
        return normal

    def reconstruct(self):
        """Genera los tres mapas principales: albedo, normal, roughness."""
        albedo = self.generate_albedo()
        normal = self.generate_normal()
        rough = self.generate_roughness()
        return albedo, normal, rough


# -----------------------------
# Example descriptor y export
# -----------------------------

def save_png(arr: np.ndarray, path: str):
    im = Image.fromarray(arr)
    im.save(path)


def example_usage(out_dir="stp_output"):
    os.makedirs(out_dir, exist_ok=True)

    desc = TextureDescriptor(
        name="urban_brick_01",
        type="masonry",
        base_color=(0.58, 0.24, 0.18),  # RGB 0..1
        variation_seed=12345,
        pattern_scale=24.0,
        grunge_amount=0.45,
        roughness_mean=0.75,
        roughness_variation=0.35,
        normal_strength=1.8,
        resolution=512,
    )

    print("Descriptor:\n", desc.to_json())

    tex = TexNetLite(desc)
    albedo, normal, rough = tex.reconstruct()

    save_png(albedo, os.path.join(out_dir, desc.name + "_albedo.png"))
    save_png(normal, os.path.join(out_dir, desc.name + "_normal.png"))
    save_png(rough, os.path.join(out_dir, desc.name + "_roughness.png"))

    # also export a small metadata TFC file (compact JSON)
    tfc = {
        "name": desc.name,
        "type": desc.type,
        "seed": desc.variation_seed,
        "pattern_scale": desc.pattern_scale,
        "grunge": desc.grunge_amount,
        "roughness_mean": desc.roughness_mean,
        "normal_strength": desc.normal_strength,
        "resolution": desc.resolution,
    }
    with open(os.path.join(out_dir, desc.name + "_tfc.json"), "w") as f:
        json.dump(tfc, f, indent=2)

    print("Texturas generadas en:", out_dir)


if __name__ == '__main__':

    example_usage()
