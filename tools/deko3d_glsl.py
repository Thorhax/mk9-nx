"""banjo-nx (deko3d renderer, docs/deko3d-renderer.md section 5): SPIRV-Cross's Vulkan GLSL -> GLSL that uam takes.

    python tools/deko3d_glsl.py in.vk.glsl out.glsl [--stage vert|frag|geom|comp]

The input is `spirv-cross x.spv --version 460 --vulkan-semantics --relax-nan-checks` of a shader DXC compiled with
-DBANJO_DEKO3D and without -fvk-invert-y. What changes:

- Descriptor-heap textures (runtime arrays of texture2D / texture3D / textureCube / texture2DArray / texture2DMS and of
  sampler, as Renut's heaps and the host shaders' tables are) become bindless handles: `sampler2D(T[t], S[s])` ->
  `sampler2D(dkh(t, s))` and a texture used alone (texelFetch, textureSize) `T[t]` -> `sampler2D(dkh(t, 0u))`. deko3d's
  handle is the image id with the sampler id at bit 20 (dkMakeTextureHandle), both relative to the bound image and
  sampler descriptor sets, which the back end binds where the plume sets are.
- Constant buffers: the k-th constant buffer of the plume layout is UBO binding k; the game shaders' sets 5, 6, 7 are
  bindings 0, 1, 2 (--ubo-map). Push constants are a std140 UBO at binding 5 (PUSH_UBO_BINDING), updated by the back end
  with dkCmdBufPushConstants.
- Storage buffers: in the order of their (set, binding), consecutive bindings (an array takes one per element); set 15
  (the deko3d raw-address window of shader_common.h) is binding 15.
- Specialization constants: `#ifndef SPIRV_CROSS_CONSTANT_ID_n` defaults, so a variant is the same text with its values
  defined first (variant_header()).
- gl_VertexIndex / gl_InstanceIndex -> gl_VertexID / gl_InstanceID.

Exit status 1 with a message when the shader has something this does not handle (it is listed, not guessed at).
"""
import argparse
import re
import sys

PUSH_UBO_BINDING = 5
RAW_WINDOW_SET = 15
RAW_WINDOW_BINDING = 15
DEFAULT_UBO_MAP = {5: 0, 6: 1, 7: 2}

SAMPLER_OF = {
    'texture1D': 'sampler1D', 'texture2D': 'sampler2D', 'texture3D': 'sampler3D', 'textureCube': 'samplerCube',
    'texture2DArray': 'sampler2DArray', 'textureCubeArray': 'samplerCubeArray', 'texture2DMS': 'sampler2DMS',
    'texture2DMSArray': 'sampler2DMSArray', 'texture1DArray': 'sampler1DArray',
    'utexture2D': 'usampler2D', 'itexture2D': 'isampler2D', 'utexture3D': 'usampler3D', 'utexture2DMS': 'usampler2DMS',
}
DROPPED_EXTENSIONS = ('GL_EXT_nonuniform_qualifier', 'GL_EXT_samplerless_texture_functions',
                      'GL_EXT_spirv_intrinsics', 'GL_EXT_scalar_block_layout')


class Unsupported(Exception):
    pass


def matching(text, open_index, open_ch, close_ch):
    """Index of the bracket closing the one at open_index."""
    depth = 0
    for i in range(open_index, len(text)):
        c = text[i]
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i
    raise Unsupported(f'unbalanced {open_ch}{close_ch} at {open_index}')


def strip_nonuniform(text):
    out = []
    i = 0
    key = 'nonuniformEXT('
    while True:
        j = text.find(key, i)
        if j < 0:
            out.append(text[i:])
            return ''.join(out)
        out.append(text[i:j])
        close = matching(text, j + len(key) - 1, '(', ')')
        out.append('(' + text[j + len(key):close] + ')')
        i = close + 1


def array_index(text, name_end):
    """For `NAME[expr]` with NAME ending at name_end: (expr, index after ']') or None."""
    k = name_end
    while k < len(text) and text[k] == ' ':
        k += 1
    if k >= len(text) or text[k] != '[':
        return None
    close = matching(text, k, '[', ']')
    return text[k + 1:close], close + 1


def strip_index_mask(expr):
    """Renut's descriptor index mask (renutDescriptor: & 0x03FFFFFF, the bits above being the resolution scale and the
    stacked texture flags) taken off an index used without a sampler: those operations read only the low 20 bits."""
    e = expr.strip()
    for _ in range(4):
        while e.startswith('(') and matching(e, 0, '(', ')') == len(e) - 1:
            e = e[1:-1].strip()
        m = re.match(r'^(.*?)\s*&\s*67108863u$', e, re.S)
        if not m:
            break
        left = m.group(1).strip()
        depth = 0
        balanced = True
        for c in left:
            depth += c in '([' 
            depth -= c in ')]'
            if depth < 0:
                balanced = False
        if not balanced or depth != 0:
            break
        e = left
    return e


SHARED_WORD_RE = re.compile(r'^g_UboShared\.v\[(\d+)u?\]\.([xyzw])$')


def shared_word_offset(expr, temps):
    """The byte offset in SharedConstants (the UBO g_UboShared) of a texture or sampler word read as it is (through a
    temporary or not, Renut's index mask taken off), or None."""
    e = strip_index_mask(expr)
    for _ in range(3):
        if e in temps:
            e = strip_index_mask(temps[e])
        else:
            break
    m = SHARED_WORD_RE.match(e)
    if not m:
        return None
    return int(m.group(1)) * 16 + 'xyzw'.index(m.group(2)) * 4


def convert(src, stage, ubo_map, sampler_words='ids'):
    text = src.replace('\r\n', '\n')
    notes = []
    # Game shaders (sampler words in handle form): textures whose words are plain SharedConstants reads are bound
    # textures, binding k = (sampler type, texture word, sampler word); the back end binds their handles per draw from
    # the uploaded constants (plume_deko3d: dkCmdBufBindTextures). The others stay bindless (dkh).
    bound = sampler_words == 'handle'
    bindings = {}  # (sampler type, texture word offset, sampler word offset or None) -> binding
    temps = {m.group(1): m.group(2).strip() for m in re.finditer(r'^\s*uint (_\d+) = ([^;]+);', src, re.M)}

    def bind(stype, t_off, s_off):
        key = (stype, t_off, s_off)
        if key not in bindings:
            if len(bindings) >= 32:
                return None
            bindings[key] = len(bindings)
        return f'dk_tex_b{bindings[key]}'

    # ---- texture and sampler tables
    textures = {}  # name -> sampler type
    samplers = set()
    decl_re = re.compile(r'^layout\(set = (\d+), binding = (\d+)\) uniform (\w+) (\w+)(\[\d*\])?;\n', re.M)

    def take_decl(m):
        typ, name = m.group(3), m.group(4)
        if typ == 'sampler':
            samplers.add(name)
            return ''
        if typ in SAMPLER_OF:
            textures[name] = SAMPLER_OF[typ]
            return ''
        if typ.startswith('sampler') or typ.startswith('image'):
            raise Unsupported(f'combined sampler or image declaration {typ} {name}')
        raise Unsupported(f'unknown uniform declaration {typ} {name}')

    text = decl_re.sub(take_decl, text)
    text = strip_nonuniform(text)

    if textures:
        # sampler2D(T[t], S[s]) -> sampler2D(dkh(t, s))
        ctor_re = re.compile(r'\b(u|i)?sampler(1D|2D|3D|Cube|2DArray|CubeArray|2DMS|2DMSArray|1DArray)\(')
        pos = 0
        out = []
        while True:
            m = ctor_re.search(text, pos)
            if not m:
                out.append(text[pos:])
                break
            open_i = m.end() - 1
            close_i = matching(text, open_i, '(', ')')
            inner = text[open_i + 1:close_i]
            tm = re.match(r'\s*(\w+)\s*\[', inner)
            if tm and tm.group(1) in textures:
                t_expr, after = array_index(inner, tm.end(1))
                rest = inner[after:].lstrip()
                if not rest.startswith(','):
                    raise Unsupported('sampler constructor without a sampler: ' + inner[:80])
                rest = rest[1:].lstrip()
                sm = re.match(r'(\w+)\s*\[', rest)
                if not sm or sm.group(1) not in samplers:
                    raise Unsupported('sampler constructor with a non-table sampler: ' + inner[:80])
                s_expr, s_after = array_index(rest, sm.end(1))
                if rest[s_after:].strip():
                    raise Unsupported('unexpected sampler constructor arguments: ' + inner[:80])
                out.append(text[pos:m.start()])
                var = None
                if bound:
                    t_off = shared_word_offset(t_expr, temps)
                    s_off = shared_word_offset(s_expr, temps)
                    if t_off is not None and s_off is not None:
                        var = bind(m.group(0)[:-1], t_off, s_off)
                out.append(var if var else f'{m.group(0)}dkh({t_expr}, {s_expr}))')
                pos = close_i + 1
            else:
                out.append(text[pos:close_i + 1])
                pos = close_i + 1
        text = ''.join(out)

        # T[t] used alone (texelFetch, textureSize, textureQueryLevels) -> sampler2D(uvec2(t, 0u))
        name_re = re.compile(r'\b(' + '|'.join(map(re.escape, textures)) + r')\b')
        pos = 0
        out = []
        while True:
            m = name_re.search(text, pos)
            if not m:
                out.append(text[pos:])
                break
            idx = array_index(text, m.end())
            if idx is None:
                raise Unsupported(f'texture table {m.group(1)} used without an index')
            expr, after = idx
            out.append(text[pos:m.start()])
            var = None
            if bound:
                t_off = shared_word_offset(expr, temps)
                if t_off is not None:
                    var = bind(textures[m.group(1)], t_off, None)
            # A texture alone (texelFetch, textureSize, textureQueryLevels) uses no sampler: bound with sampler 0, or
            # bindless with the word as it is, whose low 20 bits are the image id (the hardware ignores the sampler half
            # for these), a handle Ryujinx's shader translator follows straight to the constant buffer.
            out.append(var if var else f'{textures[m.group(1)]}(uvec2(uint({strip_index_mask(expr)}), 0u))')
            pos = after
        text = ''.join(out)
    for s in samplers:
        if re.search(r'\b' + re.escape(s) + r'\b', text):
            raise Unsupported(f'sampler table {s} used outside a sampler constructor')

    # ---- push constants -> std140 UBO
    def push_block(m):
        return f'layout(std140, binding = {PUSH_UBO_BINDING}) uniform {m.group(1)}'
    text, n_push = re.subn(r'^layout\(push_constant, std430\) uniform (\w+)', push_block, text, flags=re.M)
    if n_push:
        # std430 and std140 differ for arrays of scalars / 2-vectors and for structs: refuse those in a push block.
        m = re.search(r'layout\(std140, binding = %d\) uniform \w+\n\{\n(.*?)\n\}' % PUSH_UBO_BINDING, text, re.S)
        if m:
            for line in m.group(1).split('\n'):
                lm = re.match(r'\s*(?:layout\(offset = \d+\) )?(\w+) \w+(\[\d+\])?;', line)
                if not lm:
                    raise Unsupported('push constant member not understood: ' + line.strip())
                if lm.group(2) and lm.group(1) in ('float', 'uint', 'int', 'vec2', 'uvec2', 'ivec2', 'bool'):
                    raise Unsupported('push constant array with a std430-only stride: ' + line.strip())

    # ---- constant buffers
    def ubo_decl(m):
        s, b = int(m.group(1)), int(m.group(2))
        if b != 0 or s not in ubo_map:
            raise Unsupported(f'constant buffer at set {s} binding {b} has no UBO binding (--ubo-map)')
        return f'layout(binding = {ubo_map[s]}, std140) uniform'
    text = re.sub(r'layout\(set = (\d+), binding = (\d+), std140\) uniform', ubo_decl, text)

    # ---- a constant buffer vector's component chosen at run time (2026-10-18, the deko3d water)
    # uam compiles NAME.v[a][b] with b not a constant as a load of component x of the vector its byte offset falls in:
    # it divides a * 16 + b * 4 by 16 and keeps .x (seen in its TGSI). The vector is loaded whole and the component
    # picked from it instead (dk_pick: a select chain). Renut's generic vertex fetch reads its fetch constant's slot so
    # (vertex_fetch_slots[index], the tessellated water's patch corners): it read slot 92 for slot 95.
    vec_blocks = {m.group(2): m.group(1) for m in
                  re.finditer(r'uniform \w+\n\{\n\s*(u?vec4|ivec4) v\[\d+\];\n\} (\w+);', text)}
    picks = set()
    if vec_blocks:
        name_re = re.compile(r'\b(' + '|'.join(map(re.escape, vec_blocks)) + r')\.v\[')
        pos = 0
        out = []
        while True:
            m = name_re.search(text, pos)
            if not m:
                out.append(text[pos:])
                break
            first_open = m.end() - 1
            first_close = matching(text, first_open, '[', ']')
            k = first_close + 1
            if k < len(text) and text[k] == '[':
                second_close = matching(text, k, '[', ']')
                comp = text[k + 1:second_close].strip()
                if not re.fullmatch(r'\d+u?', comp):
                    vtype = vec_blocks[m.group(1)]
                    picks.add(vtype)
                    out.append(text[pos:m.start()])
                    out.append(f'dk_pick_{vtype}({text[m.start():first_close + 1]}, uint({comp}))')
                    pos = second_close + 1
                    continue
            out.append(text[pos:first_close + 1])
            pos = first_close + 1
        text = ''.join(out)
    if picks:
        scalar = {'vec4': 'float', 'uvec4': 'uint', 'ivec4': 'int'}
        helpers = ''.join(f'{scalar[t]} dk_pick_{t}({t} v, uint i) {{ return v[i]; }}\n' for t in sorted(picks))
        at = text.find('\nvoid main()')
        if at < 0:
            raise Unsupported('no main() for the component pick helpers')
        text = text[:at + 1] + helpers + text[at + 1:]

    # ---- storage buffers
    ssbo_re = re.compile(r'layout\(set = (\d+), binding = (\d+), std430\) (readonly |writeonly |restrict )*buffer (\w+)\n\{(.*?)\n\} (\w+)(\[(\d+)\])?;',
                         re.S)
    found = sorted((int(m.group(1)), int(m.group(2)), m.start()) for m in ssbo_re.finditer(text))
    slots = {}
    next_slot = 0
    for s, b, _ in found:
        if s == RAW_WINDOW_SET:
            slots[(s, b)] = b  # the address window (15) and the memexport buffer (14): fixed bindings
            continue
        slots[(s, b)] = next_slot
        m = [m for m in ssbo_re.finditer(text) if int(m.group(1)) == s and int(m.group(2)) == b][0]
        next_slot += int(m.group(8)) if m.group(8) else 1
    if next_slot > 15:
        raise Unsupported(f'{next_slot} storage buffers')

    def ssbo_decl(m):
        slot = slots[(int(m.group(1)), int(m.group(2)))]
        quals = m.group(3) or ''
        arr = m.group(7) or ''
        return f'layout(binding = {slot}, std430) {quals}buffer {m.group(4)}\n{{{m.group(5)}\n}} {m.group(6)}{arr};'
    text = ssbo_re.sub(ssbo_decl, text)

    # ---- specialization constants
    spec_re = re.compile(r'^layout\(constant_id = (\d+)\) const (\w+) (\w+) = ([^;]+);', re.M)

    def spec(m):
        n = m.group(1)
        return (f'#ifndef SPIRV_CROSS_CONSTANT_ID_{n}\n#define SPIRV_CROSS_CONSTANT_ID_{n} {m.group(4)}\n#endif\n'
                f'const {m.group(2)} {m.group(3)} = SPIRV_CROSS_CONSTANT_ID_{n};')
    text = spec_re.sub(spec, text)

    # ---- builtins
    text = re.sub(r'\bgl_VertexIndex\b', 'gl_VertexID', text)
    text = re.sub(r'\bgl_InstanceIndex\b', 'gl_InstanceID', text)
    for bad in ('gl_BaseInstance', 'gl_BaseVertex', 'subpassInput', 'spirv_instruction', 'buffer_reference',
                'push_constant', 'layout(set =', 'GL_EXT_buffer_reference', 'gl_DrawID'):
        if bad in text:
            raise Unsupported(f'{bad} remains')

    # ---- vertex inputs: uam's GLSL frontend (Mesa's) takes locations 0..15 only; Renut's vertex shaders use up to 22
    # (TEXCOORD4..10 are 16..22) but at most 16 at once: renumbered in order, the map in the banjo-dk line (vin).
    vin = []
    if stage == 'vert':
        in_re = re.compile(r'^layout\(location = (\d+)\) in (\w+) (\w+)(\[\d+\])?;', re.M)
        decls = in_re.findall(text)
        if any(d[3] for d in decls):
            raise Unsupported('vertex input array')
        vin = sorted(int(d[0]) for d in decls)
        if len(vin) > 16:
            raise Unsupported(f'{len(vin)} vertex inputs')
        new_of = {old: i for i, old in enumerate(vin)}
        text = in_re.sub(lambda m: f'layout(location = {new_of[int(m.group(1))]}) in {m.group(2)} {m.group(3)};', text)

    # ---- what the back end binds for this shader (plume_deko3d reads the banjo-dk line)
    ubo_mask = 0
    for m in re.finditer(r'layout\((?:std140, )?binding = (\d+)(?:, std140)?\) uniform', text):
        ubo_mask |= 1 << int(m.group(1))
    ssbo_mask = 0
    for m in re.finditer(r'layout\(binding = (\d+), std430\) (?:\w+ )*buffer \w+\n\{.*?\n\} \w+(?:\[(\d+)\])?;', text, re.S):
        for k in range(int(m.group(2) or 1)):
            ssbo_mask |= 1 << (int(m.group(1)) + k)
    bindless = textures and ('dkh(' in text or 'uvec2(uint(' in text)
    meta = f'// banjo-dk ubo=0x{ubo_mask:x} ssbo=0x{ssbo_mask:x} tex={1 if bindless else 0}'
    if vin:
        meta += ' vin=' + ','.join(map(str, vin))
    if bindings:
        # bound textures: binding:texture word:sampler word (byte offsets in SharedConstants; 65535 = no sampler)
        meta += ' texb=' + ','.join(f'{k}:{t}:{65535 if s is None else s}'
                                    for (_, t, s), k in sorted(bindings.items(), key=lambda kv: kv[1]))

    # ---- extensions
    lines = text.split('\n')
    lines = [l for l in lines if not any(l.startswith('#extension ' + e) for e in DROPPED_EXTENSIONS)]
    if not lines[0].startswith('#version'):
        raise Unsupported('no #version line')
    extra = [meta]
    if bindless:
        extra.append('#extension GL_ARB_bindless_texture : require')
        if sampler_words == 'handle':
            # Game shaders: the renderer writes sampler words with the id at bit 20 (draw.cpp SamplerWord) and texture
            # words with flags above bit 20 (the resolution scale): the masked form, which Ryujinx's shader translator
            # also follows back to the constant buffer.
            extra.append('#define dkh(t, s) uvec2((uint(t) & 0xFFFFFu) | (uint(s) & 0xFFF00000u), 0u)')
        else:
            # Host shaders: sampler ids (the push constants' samplerIndex); the shifted form, which Ryujinx follows too.
            extra.append('#define dkh(t, s) uvec2(uint(t) | (uint(s) << 20u), 0u)')
    for (stype, _, _), k in sorted(bindings.items(), key=lambda kv: kv[1]):
        extra.append(f'layout(binding = {k}) uniform {stype} dk_tex_b{k};')
    lines[1:1] = extra
    return '\n'.join(lines), notes


def variant_header(spec_values):
    """Lines that select a variant: {constant_id: value}, inserted after #version."""
    return ''.join(f'#define SPIRV_CROSS_CONSTANT_ID_{k} {v}u\n' for k, v in sorted(spec_values.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('output')
    ap.add_argument('--stage', default='')
    ap.add_argument('--ubo-map', default='5:0,6:1,7:2', help='constant buffer set:binding pairs')
    ap.add_argument('--sampler-words', default='ids', choices=('ids', 'handle'),
                    help='ids: sampler ids (host shaders); handle: ids at bit 20 already (game shaders, draw.cpp)')
    a = ap.parse_args()
    ubo_map = {int(k): int(v) for k, v in (p.split(':') for p in a.ubo_map.split(',') if p)}
    src = open(a.input, encoding='utf-8').read()
    try:
        out, _ = convert(src, a.stage, ubo_map, a.sampler_words)
    except Unsupported as e:
        print(f'deko3d_glsl: {a.input}: {e}', file=sys.stderr)
        return 1
    with open(a.output, 'w', encoding='utf-8', newline='\n') as f:
        f.write(out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
