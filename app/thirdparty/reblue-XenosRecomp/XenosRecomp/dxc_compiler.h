#pragma once

struct DxcCompiler
{
    IDxcCompiler3* dxcCompiler = nullptr;

    DxcCompiler();
    ~DxcCompiler();

    // banjo-nx: deko3d = SPIR-V for the deko3d renderer (docs/deko3d-renderer.md): -DBANJO_DEKO3D, D3D clip space
    // (no -fvk-invert-y).
    IDxcBlob* compile(const std::string& shaderSource, bool compilePixelShader, bool compileLibrary, bool compileSpirv,
                      bool deko3d = false);
};
