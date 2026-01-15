// Prototipo STP: Texture Function Code (TFC) runtime generator (C++17).
// - Genera albedo, normal y roughness maps desde un descriptor compacto.
// - Exporta PNGs usando stb_image_write (single-header).
// Nota: Este código es demo/prototipo; en producción se reemplazaría TexNetLite
// por un decoder optimizado (GPU/SPU) y fBM/noise por SIMD/vectorized kernels.

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "T_Example.py"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <array>
#include <algorithm>

namespace fs = std::filesystem;

// ------------------------- utilidades -------------------------
inline float clampf(float v, float a = 0.0f, float b = 1.0f) {
    return (v < a) ? a : (v > b) ? b : v;
}
inline int to8(float v) { return static_cast<int>(clampf(v) * 255.0f + 0.5f); }

// mezclas y easing (Perlin fade)
inline float lerp(float a, float b, float t) { return a + (b - a) * t; }
inline float fade(float t) { return t * t * t * (t * (t * 6 - 15) + 10); }

// ------------------------- simple value-noise 2D -------------------------
// grid-based value noise (simple, deterministic via seed)
struct ValueNoise2D {
    int gridW, gridH;
    std::vector<float> grid; // (gridW+1)*(gridH+1)
    ValueNoise2D() : gridW(0), gridH(0) {}
    ValueNoise2D(int gx, int gy, unsigned int seed) { init(gx, gy, seed); }

    void init(int gx, int gy, unsigned int seed) {
        gridW = gx; gridH = gy;
        grid.resize((gridW + 1) * (gridH + 1));
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (int j = 0; j <= gridH; ++j)
            for (int i = 0; i <= gridW; ++i)
                grid[i + j * (gridW + 1)] = dist(rng);
    }
    float sample(float x, float y) const {
        // x in [0, gridW), y in [0, gridH)
        int ix = static_cast<int>(std::floor(x));
        int iy = static_cast<int>(std::floor(y));
        float fx = x - ix;
        float fy = y - iy;
        ix = std::max(0, std::min(ix, gridW - 1));
        iy = std::max(0, std::min(iy, gridH - 1));
        int i00 = ix + iy * (gridW + 1);
        int i10 = (ix + 1) + iy * (gridW + 1);
        int i01 = ix + (iy + 1) * (gridW + 1);
        int i11 = (ix + 1) + (iy + 1) * (gridW + 1);
        float v00 = grid[i00], v10 = grid[i10], v01 = grid[i01], v11 = grid[i11];
        float sx = fade(fx);
        float sy = fade(fy);
        float a = lerp(v00, v10, sx);
        float b = lerp(v01, v11, sx);
        return lerp(a, b, sy);
    }
};

// fbm usando value noise (no optimizado pero claro)
static void fbm_fill(std::vector<float>& out, int W, int H, int octaves, float baseScale, float lacunarity, float gain, unsigned int seed) {
    std::fill(out.begin(), out.end(), 0.0f);
    float amplitude = 1.0f, frequency = baseScale;
    for (int o = 0; o < octaves; ++o) {
        int gx = std::max(2, static_cast<int>(std::round(W / frequency)));
        int gy = std::max(2, static_cast<int>(std::round(H / frequency)));
        ValueNoise2D v(gx, gy, seed + o * 1315423911u);
        for (int j = 0; j < H; ++j) {
            for (int i = 0; i < W; ++i) {
                float sx = (static_cast<float>(i) / static_cast<float>(W)) * gx;
                float sy = (static_cast<float>(j) / static_cast<float>(H)) * gy;
                float val = v.sample(sx, sy);
                out[j*W + i] += val * amplitude;
            }
        }
        amplitude *= gain;
        frequency *= lacunarity;
    }
    // normalize to 0..1
    float minv = out[0], maxv = out[0];
    for (auto &v : out) { if (v < minv) minv = v; if (v > maxv) maxv = v; }
    float range = (maxv - minv) > 1e-9f ? (maxv - minv) : 1e-9f;
    for (auto &v : out) v = (v - minv) / range;
}

// ------------------------- Descriptor (TFC) -------------------------
struct TextureDescriptor {
    std::string name;            // id
    std::string type;            // p.ej. "masonry", "metal", "fabric"
    std::array<float,3> baseColor;// RGB 0..1
    unsigned int seed;           // variation seed
    float patternScale;          // control de escala del patrón
    float grunge;                // 0..1
    float roughnessMean;         // 0..1
    float roughnessVar;          // 0..1
    float normalStrength;        // >0
    int resolution;              // ancho=alto
    TextureDescriptor()
      : name("tex"), type("generic"), baseColor{0.8f,0.8f,0.8f},
        seed(1337), patternScale(16.0f), grunge(0.2f), roughnessMean(0.5f),
        roughnessVar(0.2f), normalStrength(1.2f), resolution(512) {}
};

// ------------------------- TexNetLite (prototipo CPU) -------------------------
class TexNetLite {
public:
    TexNetLite(const TextureDescriptor& d) : desc(d) {
        W = desc.resolution;
        H = desc.resolution;
    }

    // outputs: albedo (W*H*3 uint8), normal (W*H*3 uint8), roughness (W*H uint8)
    void reconstruct(std::vector<uint8_t>& albedo, std::vector<uint8_t>& normal, std::vector<uint8_t>& roughness) {
        albedo.assign(W*H*3, 0);
        normal.assign(W*H*3, 0);
        roughness.assign(W*H, 0);

        // generate pattern (fbm)
        std::vector<float> pattern(W*H);
        fbm_fill(pattern, W, H, 5, desc.patternScale, 2.0f, 0.5f, desc.seed);

        // grunge/noise
        std::vector<float> grunge(W*H);
        fbm_fill(grunge, W, H, 3, desc.patternScale * 0.25f, 2.0f, 0.6f, desc.seed + 97);

        // hue shift / variation
        std::vector<float> hueShift(W*H);
        fbm_fill(hueShift, W, H, 3, desc.patternScale * 2.0f, 2.0f, 0.5f, desc.seed + 42);

        // compute albedo
        for (int j = 0; j < H; ++j) {
            for (int i = 0; i < W; ++i) {
                int idx = j*W + i;
                float p = pattern[idx];
                float g = grunge[idx];
                float hs = hueShift[idx];

                float r = desc.baseColor[0];
                float gcol = desc.baseColor[1];
                float b = desc.baseColor[2];

                // base modulation
                float m = 0.7f + 0.3f * p;
                float dar = r * m;
                float dag = gcol * m;
                float dab = b * m;

                // combine grunge
                float ga = desc.grunge * 0.6f;
                dar = dar * (1.0f - ga) + g * ga;
                dag = dag * (1.0f - ga) + g * ga * 0.9f;
                dab = dab * (1.0f - ga) + g * ga * 0.95f;

                // subtle hue variation
                dar = clampf(dar * (0.9f + 0.2f * hs));
                dag = clampf(dag * (0.9f + 0.15f * hs));
                dab = clampf(dab * (0.95f + 0.1f * hs));

                albedo[idx*3 + 0] = static_cast<uint8_t>(dar * 255.0f + 0.5f);
                albedo[idx*3 + 1] = static_cast<uint8_t>(dag * 255.0f + 0.5f);
                albedo[idx*3 + 2] = static_cast<uint8_t>(dab * 255.0f + 0.5f);
            }
        }

        // roughness map
        std::vector<float> var(W*H);
        fbm_fill(var, W, H, 4, desc.patternScale * 1.5f, 2.0f, 0.5f, desc.seed + 11);
        for (int idx = 0; idx < W*H; ++idx) {
            float base = desc.roughnessMean;
            float v = clampf(base * (1.0f + (var[idx] - 0.5f) * desc.roughnessVar));
            roughness[idx] = static_cast<uint8_t>(v * 255.0f + 0.5f);
        }

        // height map (for normal generation)
        std::vector<float> height(W*H);
        fbm_fill(height, W, H, 6, desc.patternScale * 0.5f, 2.0f, 0.5f, desc.seed + 77);

        // sobel for gradients
        for (int j = 0; j < H; ++j) {
            for (int i = 0; i < W; ++i) {
                // sample neighbors with clamp
                auto sampleH = [&](int x, int y)->float {
                    int sx = std::max(0, std::min(x, W-1));
                    int sy = std::max(0, std::min(y, H-1));
                    return height[sy*W + sx];
                };
                float gx = -sampleH(i-1, j-1) + sampleH(i+1, j-1)
                           -2.0f*sampleH(i-1, j) + 2.0f*sampleH(i+1, j)
                           -sampleH(i-1, j+1) + sampleH(i+1, j+1);
                float gy = -sampleH(i-1, j-1) -2.0f*sampleH(i, j-1) - sampleH(i+1, j-1)
                           + sampleH(i-1, j+1) + 2.0f*sampleH(i, j+1) + sampleH(i+1, j+1);

                float strength = desc.normalStrength;
                float nx = -gx * strength;
                float ny = -gy * strength;
                float nz = 1.0f / strength;

                // normalize
                float len = std::sqrt(nx*nx + ny*ny + nz*nz);
                if (len < 1e-6f) len = 1e-6f;
                nx /= len; ny /= len; nz /= len;

                int idx = j*W + i;
                normal[idx*3 + 0] = static_cast<uint8_t>(clampf(nx * 0.5f + 0.5f) * 255.0f + 0.5f);
                normal[idx*3 + 1] = static_cast<uint8_t>(clampf(ny * 0.5f + 0.5f) * 255.0f + 0.5f);
                normal[idx*3 + 2] = static_cast<uint8_t>(clampf(nz * 0.5f + 0.5f) * 255.0f + 0.5f);
            }
        }
    }

private:
    TextureDescriptor desc;
    int W, H;
};

// ------------------------- PNG export helpers (stb_image_write) -------------------------
static bool write_png_rgb(const std::string& path, int w, int h, const std::vector<uint8_t>& data) {
    // data size must be w*h*3
    return stbi_write_png(path.c_str(), w, h, 3, data.data(), w*3) != 0;
}
static bool write_png_gray(const std::string& path, int w, int h, const std::vector<uint8_t>& data) {
    // data size must be w*h
    return stbi_write_png(path.c_str(), w, h, 1, data.data(), w) != 0;
}

// ------------------------- Example usage -------------------------
int main(int argc, char** argv) {
    TextureDescriptor desc;
    desc.name = "urban_brick_01";
    desc.type = "masonry";
    desc.baseColor = {0.58f, 0.24f, 0.18f};
    desc.seed = 12345;
    desc.patternScale = 24.0f;
    desc.grunge = 0.45f;
    desc.roughnessMean = 0.75f;
    desc.roughnessVar = 0.35f;
    desc.normalStrength = 1.8f;
    desc.resolution = 512;

    fs::path outdir = "stp_output";
    try { fs::create_directories(outdir); } catch(...) {}

    std::vector<uint8_t> albedo, normal, rough;
    TexNetLite gen(desc);
    std::cout << "Generating texture '" << desc.name << "' (" << desc.resolution << "x" << desc.resolution << ")...\n";
    gen.reconstruct(albedo, normal, rough);

    std::string albedoPath = (outdir / (desc.name + "_albedo.png")).string();
    std::string normalPath = (outdir / (desc.name + "_normal.png")).string();
    std::string roughPath = (outdir / (desc.name + "_roughness.png")).string();

    if (!write_png_rgb(albedoPath, desc.resolution, desc.resolution, albedo)) {
        std::cerr << "Error writing albedo PNG\n";
    } else std::cout << "Albedo saved: " << albedoPath << "\n";

    if (!write_png_rgb(normalPath, desc.resolution, desc.resolution, normal)) {
        std::cerr << "Error writing normal PNG\n";
    } else std::cout << "Normal saved: " << normalPath << "\n";

    if (!write_png_gray(roughPath, desc.resolution, desc.resolution, rough)) {
        std::cerr << "Error writing roughness PNG\n";
    } else std::cout << "Roughness saved: " << roughPath << "\n";

    // export small TFC descriptor JSON
    std::string jsonPath = (outdir / (desc.name + "_tfc.json")).string();
    std::ofstream ofs(jsonPath);
    if (ofs) {
        ofs << "{\n";
        ofs << "  \"name\": \"" << desc.name << "\",\n";
        ofs << "  \"type\": \"" << desc.type << "\",\n";
        ofs << "  \"seed\": " << desc.seed << ",\n";
        ofs << "  \"patternScale\": " << desc.patternScale << ",\n";
        ofs << "  \"grunge\": " << desc.grunge << ",\n";
        ofs << "  \"roughnessMean\": " << desc.roughnessMean << ",\n";
        ofs << "  \"normalStrength\": " << desc.normalStrength << ",\n";
        ofs << "  \"resolution\": " << desc.resolution << "\n";
        ofs << "}\n";
        ofs.close();
        std::cout << "Descriptor JSON saved: " << jsonPath << "\n";
    } else {
        std::cerr << "Failed writing descriptor JSON\n";
    }

    std::cout << "Done.\n";
    return 0;

}
