#!/usr/bin/env python3
"""Writes the premade enhancement configurations from app/sa2_switch.toml (the default), each differing from it only
in what its name says (docs/enhancements.md):

  sa2_cpu_on.toml       the default, profiler on                   (A/B: the CPU group on)
  sa2_cpu_off.toml      sa2_cpu_enhancements = false, profiler on  (A/B: the CPU group off, non-CPU on)
  sa2_noncpu_off.toml   sa2_noncpu_enhancements = false            (the non-CPU group off, CPU on)
  sa2_all_off.toml      both masters false                         (the original paths everywhere)
  sa2_cpu_verify.toml   sa2_cpu_verify = true                      (every enhancement on, every use checked; slow)
  sa2_profile.toml      sa2_gpu_marks_categories = true     (GPU time per pass category)
  sa2_test_dedupe_off.toml        sa2_native_constants_dedupe = false   (flicker test: constants dedupe off)
  sa2_test_copies_off.toml        the 52 unguarded localized copies off, profiler on (A/B against sa2_cpu_on.toml)
  sa2_test_upload_uncached.toml   the upload buffer in uncached memory as before, profiler on (A/B)
  sa2_test_driver_paths_off.toml  the nfsmw NVK driver paths off         (flicker test: set 4 by differences etc.)
  sa2_test_deferred_off.toml      deferred Vulkan recording off (on by default since round 5), profiler on (A/B)
  sa2_test_direct_state_off.toml  the direct renderer's state records off: the ring parses PM4 as before, profiler on (A/B)
  sa2_test_worker_core2.toml      the deferred-recording worker on core 2 only (default: core 0 only), profiler on
  sa2_test_worker_core_any.toml   the deferred-recording worker on any of cores 0-2, profiler on
  sa2_test_main_priority_on.toml  the drawing game thread at 0x3A, profiler on (A/B against sa2_cpu_on.toml)
  sa2_test_sync_epochs_off.toml   round 7's dedupe sync epochs off (the old address-only reuse), profiler on (A/B: cost)
  sa2_test_direct_flush_off.toml  the direct renderer step 2 off: Direct3D's flushes write PM4 again, profiler on (A/B)
  sa2_test_gpu_round_off.toml     the GPU round's three keys off (resolve draws, blend skip, resolve aliases), profiler on
  sa2_all_on.toml                 both masters on, said explicitly (= sa2.toml: every CPU and GPU enhancement), profiler on
  sa2_test_api_off.toml           the native Direct3D layer off (draws and Clear write PM4 again), profiler on (A/B)
  sa2_test_gpu_round2_off.toml    GPU round 2's six keys off (push once, structured CF, index base, three spec bits)
  sa2_test_log_writer_off.toml    the log writer thread off: the synchronous file log as before, profiler on (A/B)
  sa2_test_render_scale_off.toml  the render resolution off: 1280x720 docked too, as before, profiler on (A/B: docked GPU)
  sa2_test_msaa2.toml             2x MSAA of the game's render targets (sa2_msaa = 2), profiler on
  sa2_test_msaa4.toml             4x MSAA of the game's render targets (sa2_msaa = 4), profiler on
  sa2_test_vsync_relaxed_off.toml the old fixed vsync (a frame that misses 60 waits for the next vblank), profiler on
  sa2_test_round13_off.toml       round 13's hand-written vertex loaders, njCalcPoint and njMultiMatrix off, profiler on
  sa2_test_vs_fetch_reorder_off.toml  vertex shaders Direct3D reordered dropped again unless the package has them, profiler on

The profiler is on in sa2.toml itself and in every configuration except sa2_cpu_verify.toml (2026-10-06, the user's
choice: every test run brings its profile; verify is slow anyway): sa2_gpu_marks_categories = true (GPU time per
pass category and the idle gap in the "GPU per Swap" lines) and switch_profile_stacks = true (the CPU sampler,
rex_profile.log). Every A/B side pays the same profiling cost. The variants below still set both keys explicitly, so a
config stays profiled even if the default changes.

  python tools/make_enhancement_configs.py [--out <folder>]     (default: out/nx/configs)
"""
import argparse
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROFILING = {'sa2_gpu_marks_categories': 'true', 'switch_profile_stacks': 'true'}
NO_PROFILING = {'sa2_gpu_marks_categories': 'false', 'switch_profile_stacks': 'false'}
PROFILING_NOTE = ('# Profiler on: switch_profile_stacks = true makes the CPU sampler record where each thread spends its\n'
                  '# time (logs/rex/rex_profile.log); the GPU time per pass category is in\n'
                  '# the "GPU per Swap" log lines. Every config except sa2_cpu_verify profiles, so A/B runs compare\n'
                  '# fairly.\n')

VARIANTS = {
    'sa2_cpu_on.toml': dict(PROFILING),
    'sa2_cpu_off.toml': {'sa2_cpu_enhancements': 'false', **PROFILING},
    'sa2_noncpu_off.toml': {'sa2_noncpu_enhancements': 'false', **PROFILING},
    'sa2_all_off.toml': {'sa2_cpu_enhancements': 'false', 'sa2_noncpu_enhancements': 'false', **PROFILING},
    # Verify mode without the profiler (it checks, it does not measure).
    'sa2_cpu_verify.toml': {'sa2_cpu_verify': 'true', **NO_PROFILING},
    'sa2_profile.toml': dict(PROFILING),
    # Truck-light flicker tests (2026-10-06): the two paths that write shader constants differently from before.
    'sa2_test_dedupe_off.toml': {'sa2_native_constants_dedupe': 'false', **PROFILING},
    # A/B of the 53 unguarded localized copies (round 1's 15, round 3's 20 and round 4's 18; on by default since round
    # 4, smart sync since round 3), profiler on. The two guarded round-4 leaves keep their default.
    'sa2_test_copies_off.toml': {**{f'sa2_cpu_native_sub_{a}': 'false' for a in (
        '8213A4D8', '82139DA0', '8214E130', '82555CC8', '821BC520', '82549178', '82960CE0', '82141420',
        '82555B70', '82960928', '8295F9C0', '82552408', '82960808', '8214D6D8', '8214B6C8', '8214B520', '8213D1F8',
        '829601B0', '82832B78', '8213D7F0', '8214ACA8', '828326F8', '82833E20', '828243A0', '821BA890', '82833938',
        '82834458', '82816B40', '82139930', '82834538', '821472E0', '82565758', '8213E040', '82833568', '82565C60',
        '82556A60', '82555E18', '8251D2A0', '82141050', '821390C8', '8217EB38', '82141240', '825632F8', '8251D1A8',
        '82556718', '82227C98', '8254DED0', '8254D0E0', '82139C90', '8217F048', '8214EAF0', '8213D5C8',)}, **PROFILING},
    # A/B of round 3's cached upload buffer: the uncached memory type, as before.
    'sa2_test_upload_uncached.toml': {'sa2_native_upload_memory': '0', **PROFILING},
    # A/B of deferred Vulkan recording (round 2; on by default since round 5): the ring thread queues the vkCmd* calls
    # and a worker records them, same commands in the same order (masseffect-nx's console: ring thread 62 -> 40 %).
    # This one records on the ring thread as before. Profiler on, so it compares with sa2_cpu_on.toml.
    'sa2_test_deferred_off.toml': {'sa2_native_deferred_recording': 'false',
                                   'sa2_deferred_native_fast': 'false',
                                   'sa2_deferred_native_update': 'false', **PROFILING},
    # A/B of the direct renderer step 1 (round 5): the ring parses every state packet as PM4, as before.
    'sa2_test_direct_state_off.toml': {'sa2_cpu_direct_state': 'false', **PROFILING},
    # Round 6, scheduling A/B: the deferred-recording worker is pinned to the process's default core (core 0) at a
    # priority above the game threads'; these let it run on core 2 only, or on any core.
    'sa2_test_worker_core2.toml': {'sa2_native_deferred_worker_core': '2', **PROFILING},
    'sa2_test_worker_core_any.toml': {'sa2_native_deferred_worker_core': '-1', **PROFILING},
    # A/B of the drawing game thread at priority 0x3A (round 2, from nfsmw-nx), profiler on.
    'sa2_test_main_priority_on.toml': {'sa2_cpu_main_thread_priority': 'true', **PROFILING},
    'sa2_test_sync_epochs_off.toml': {'sa2_native_dedupe_sync_epochs': 'false', **PROFILING},
    # A/B of the direct renderer step 2 (2026-10-07): Direct3D's state flush writes PM4 and the hook decodes it again.
    'sa2_test_direct_flush_off.toml': {'sa2_cpu_direct_flush': 'false', **PROFILING},
    # A/B of the GPU round (2026-10-07): resolves as copies on NVK's copy engine, no blend skip, every resolve copied.
    'sa2_test_gpu_round_off.toml': {'sa2_gpu_resolve_draw': 'false', 'sa2_gpu_blend_skip': 'false',
                                    'sa2_gpu_resolve_alias': 'false', **PROFILING},
    # Every CPU and GPU enhancement on (2026-10-07, the user's request): the default already is, this says it. MSAA
    # stays off as in sa2.toml (2x was in here for one console test; the user set it back the same day).
    'sa2_all_on.toml': {'sa2_cpu_enhancements': 'true', 'sa2_noncpu_enhancements': 'true', **PROFILING},
    # A/B of the native Direct3D layer (2026-10-07, docs/api-renderer.md): Direct3D's draws and Clear write PM4 again.
    'sa2_test_api_off.toml': {'sa2_api_draws': 'false', 'sa2_api_clears': 'false', **PROFILING},
    # A/B of GPU round 2 (2026-10-07): per-draw push constants, the skipTo control flow, the per-draw vertex index base
    # and the three per-vertex/pixel tests, as before (the shader package stays the same).
    'sa2_test_gpu_round2_off.toml': {'sa2_gpu_push_once': 'false', 'sa2_gpu_structured_cf': 'false',
                                     'sa2_gpu_index_base_held': 'false', 'sa2_gpu_texcoords_unswapped': 'false',
                                     'sa2_gpu_no_texture_signs': 'false', 'sa2_gpu_identity_remap': 'false',
                                     **PROFILING},
    # A/B of the log writer thread (2026-10-07, from futari-nx): the synchronous rotating file log, as before.
    'sa2_test_log_writer_off.toml': {'log_file_writer': 'false', **PROFILING},
    # Render resolution and MSAA (2026-10-07, docs/enhancements.md): docked at the game's 1280x720 as before (what
    # 1080p costs the GPU), and the two MSAA levels (MSAA works on the console: the user's test, 2026-10-07).
    'sa2_test_render_scale_off.toml': {'sa2_gpu_render_scale': 'false', **PROFILING},
    'sa2_test_msaa2.toml': {'sa2_msaa': '2', **PROFILING},
    'sa2_test_msaa4.toml': {'sa2_msaa': '4', **PROFILING},
    # Round 13 (2026-10-07): relaxed vsync off (the fall from 60 straight to 30 comes back), and the hand-written
    # vertex loaders / njCalcPoint / njMultiMatrix off (njCalcPoint and njMultiMatrix back to the original, not to their
    # older localized copies: the keys are the same).
    'sa2_test_vsync_relaxed_off.toml': {'sa2_vsync_relaxed': 'false', **PROFILING},
    'sa2_test_round13_off.toml': {'sa2_cpu_native_vertex_normals': 'false', 'sa2_cpu_native_vertex_points': 'false',
                                  'sa2_cpu_native_calc_point': 'false', 'sa2_cpu_native_multi_matrix': 'false',
                                  **PROFILING},
    # 2026-10-07: vertex shaders Direct3D loads with their fetches reordered are no longer identified by the run-time
    # proof (the Chao garden's stats window: drawn only if the package has a container repaired for that order).
    'sa2_test_vs_fetch_reorder_off.toml': {'sa2_gpu_vs_fetch_reorder': 'false', **PROFILING},
    'sa2_test_driver_paths_off.toml': {'sa2_native_set4_differences': 'false',
                                       'sa2_native_nvk_emission': 'false', 'sa2_native_nvk_cbufs': 'false',
                                       'sa2_native_nvk_dynamic': 'false', 'sa2_native_nvk_preload': 'false',
                                       **PROFILING},
}


# The Switch build has logging off (app/sa2_switch_defaults.toml): every test configuration turns the log back on.
LOGGING = {'log_level': '"info"', 'log_to_file': 'true', 'switch_profile_report': 'true'}


def apply(text, changes, name):
    changes = {**LOGGING, **changes}
    for key, value in changes.items():
        pattern = re.compile(r'^' + re.escape(key) + r'\s*=.*$', re.M)
        if pattern.search(text):
            text = pattern.sub(f'{key} = {value}', text, count=1)
        else:
            # Top-level keys go before the first [table] (sa2.toml ends with [input_p1] / [input_p2]): after it they
            # would belong to that table.
            table = re.search(r'^\[', text, re.M)
            line = f'{key} = {value}\n'
            text = text[:table.start()] + line + '\n' + text[table.start():] if table else text + '\n' + line
    header = (f'# {name}: app/sa2_switch.toml with ' + ', '.join(f'{k} = {v}' for k, v in changes.items()) +
              ' (tools/make_enhancement_configs.py). Copy it over sa2.toml next to the NRO.\n')
    if changes.get('sa2_gpu_marks_categories') == 'true':
        header += PROFILING_NOTE
    return header + text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(ROOT, 'out', 'nx', 'configs'))
    a = ap.parse_args()
    base = open(os.path.join(ROOT, 'app', 'sa2_switch.toml'), encoding='utf-8').read()
    os.makedirs(a.out, exist_ok=True)
    for name, changes in VARIANTS.items():
        with open(os.path.join(a.out, name), 'w', encoding='utf-8', newline='\n') as f:
            f.write(apply(base, changes, name))
        print(os.path.join(a.out, name))


if __name__ == '__main__':
    main()
