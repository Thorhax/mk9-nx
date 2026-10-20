#!/usr/bin/env python3
"""Writes the shader cache of Renut's renderer for the Switch build from a PC build's.

The Switch build is a cross build: Renut's XenosRecomp (a host tool) runs on the PC, whose build generates
app/out/pc-me/renut_gpu/shader_cache.cpp (DXIL and SPIR-V). The console reads only the SPIR-V: this copy keeps the
DXIL symbols with empty arrays (the renderer references them; Windows alone reads them), so the NRO does not carry
the DXIL bytes. The compressed SPIR-V (about 4 MB) is written to a binary file next to it and embedded by the
assembler (.incbin): GCC needs minutes and gigabytes for an initializer list of millions of numbers.
Output: app/out/renut_switch/shader_cache_switch.cpp and spirv_cache.zst (cmake/banjo_renut_gpu.cmake,
BANJO_RENUT_SHADER_CACHE_CPP). The cache is game data: never committed or distributed apart from the NRO.

Usage: python tools/renut_switch_shader_cache.py [--pc app/out/pc-me]
"""
import hashlib
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    pc = sys.argv[sys.argv.index('--pc') + 1] if '--pc' in sys.argv else os.path.join(ROOT, 'app', 'out', 'pc-me')
    src = os.path.join(pc, 'renut_gpu', 'shader_cache.cpp')
    if not os.path.isfile(src):
        sys.exit(f'renut_switch_shader_cache: no {src}: build the PC tree with BANJO_RENUT_RENDERER=ON first')
    text = open(src, encoding='utf-8', errors='strict').read()
    m = re.search(r'(const uint8_t g_compressedDxilCache\[\]\s*=\s*\{)(.*?)(\};)', text, re.S)
    if m:
        text = text[:m.start(2)] + '0' + text[m.end(2):]
        text = re.sub(r'(const size_t g_dxilCacheCompressedSize\s*=\s*)\d+', r'\g<1>0', text)
        text = re.sub(r'(const size_t g_dxilCacheDecompressedSize\s*=\s*)\d+', r'\g<1>0', text)

    out_dir = os.path.join(ROOT, 'app', 'out', 'renut_switch')
    os.makedirs(out_dir, exist_ok=True)
    m = re.search(r'const uint8_t g_compressedSpirvCache\[\]\s*=\s*\{(.*?)\};', text, re.S)
    if not m:
        sys.exit('renut_switch_shader_cache: no g_compressedSpirvCache in the PC shader cache')
    blob = bytes(int(v) for v in m.group(1).replace('\n', '').split(',') if v.strip())
    size = re.search(r'const size_t g_spirvCacheCompressedSize\s*=\s*(\d+)', text)
    if not size or int(size.group(1)) != len(blob):
        sys.exit('renut_switch_shader_cache: the SPIR-V array and its size disagree')
    blob_path = os.path.join(out_dir, 'spirv_cache.zst')
    open(blob_path + '.tmp', 'wb').write(blob)
    os.replace(blob_path + '.tmp', blob_path)
    incbin = blob_path.replace('\\', '/')
    asm = ('// banjo-nx: the compressed SPIR-V, embedded by the assembler (sha-256 ' +
           hashlib.sha256(blob).hexdigest()[:16] + ', ' + str(len(blob)) + ' bytes).\n'
           '__asm__(".section .rodata.banjo_renut_spirv,\\"a\\"\\n"\n'
           '        ".global g_compressedSpirvCache\\n"\n'
           '        ".type g_compressedSpirvCache, %object\\n"\n'
           '        ".balign 16\\n"\n'
           '        "g_compressedSpirvCache:\\n"\n'
           '        ".incbin \\"' + incbin + '\\"\\n"\n'
           '        ".size g_compressedSpirvCache, .-g_compressedSpirvCache\\n"\n'
           '        ".previous\\n");')
    text = text[:m.start()] + asm + text[m.end():]

    out = os.path.join(out_dir, 'shader_cache_switch.cpp')
    open(out + '.tmp', 'w', encoding='utf-8').write(text)
    os.replace(out + '.tmp', out)
    print(f'renut_switch_shader_cache: {os.path.getsize(src) >> 20} MB -> {os.path.getsize(out) >> 10} KB of source '
          f'+ {len(blob) >> 10} KB of SPIR-V ({out})')


if __name__ == '__main__':
    main()
