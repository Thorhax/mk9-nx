// peggle2-nx (2026-10-20): the translator's pch.h (app/thirdparty/reblue-XenosRecomp/XenosRecomp/pch.h) without DXC,
// smol-v and zstd, for tools/renut_hlsl_frontend.cpp (the web installer's WebAssembly build, which translates only).
// Force-included (-include) before shader_recompiler.cpp, as CMake's precompiled header is natively.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <execution>
#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fmt/core.h>
#include <xxhash.h>

template<typename T>
static T byteSwap(T value)
{
    if constexpr (sizeof(T) == 1)
        return value;
    else if constexpr (sizeof(T) == 2)
        return static_cast<T>(__builtin_bswap16(static_cast<uint16_t>(value)));
    else if constexpr (sizeof(T) == 4)
        return static_cast<T>(__builtin_bswap32(static_cast<uint32_t>(value)));
    else if constexpr (sizeof(T) == 8)
        return static_cast<T>(__builtin_bswap64(static_cast<uint64_t>(value)));

    assert(false && "Unexpected byte size.");
    return value;
}

template<typename T>
struct be
{
    T value;

    T get() const
    {
        if constexpr (std::is_enum_v<T>)
            return T(byteSwap(std::underlying_type_t<T>(value)));
        else
            return byteSwap(value);
    }

    operator T() const
    {
        return get();
    }
};
