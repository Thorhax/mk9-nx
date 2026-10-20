// banjo-nx (deko3d renderer, docs/deko3d-renderer.md section 5): what the host shaders need when compiled for deko3d
// (-DBANJO_DEKO3D, through SPIRV-Cross and uam).
#pragma once

// Exact unsigned division. Maxwell has no integer divide, and uam lowers a non-constant one to float math that can be
// off by one (it warns "floating point emulation with resulting loss of precision"). Here: a float estimate scaled
// just below the quotient (its error is ~2e-7 relative, the scale takes 1e-5 off), so q * b never passes a and never
// wraps; a second estimate from the remainder; then integer steps, at most two for any 32-bit a and b > 0.
uint BanjoUDiv(uint a, uint b)
{
    const float below = 0.99999;
    uint q = uint(float(a) / float(b) * below);
    uint r = a - q * b;
    q += uint(float(r) / float(b) * below);
    r = a - q * b;
    [loop] while (r >= b)
    {
        q += 1u;
        r -= b;
    }
    return q;
}

uint BanjoUMod(uint a, uint b)
{
    return a - BanjoUDiv(a, b) * b;
}

#ifdef BANJO_DEKO3D
#define BANJO_UDIV(a, b) BanjoUDiv((a), (b))
#define BANJO_UMOD(a, b) BanjoUMod((a), (b))
#else
#define BANJO_UDIV(a, b) ((a) / (b))
#define BANJO_UMOD(a, b) ((a) % (b))
#endif
