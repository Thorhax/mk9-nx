# Mortal Kombat Komplete Edition (Xbox 360): static recompilation host application.
#
# The game code is not in this repository: run the code generator first (tools/codegen.sh), which writes generated/
# from your own default.xex together with the CMake glue (generated/rexglue.cmake, made by `rexglue codegen`).
#
# Stage 3, PC (Linux):
#   cmake -S app -B out/pc -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_COMPILER=clang \
#         -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_FLAGS=-march=x86-64-v2 -DCMAKE_CXX_FLAGS=-march=x86-64-v2
#   cmake --build out/pc

cmake_minimum_required(VERSION 3.25)
project(mk9 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The SDK is built from source in this repository unless another tree is given.
set(REXSDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../sdk" CACHE PATH "Path to the ReXGlue SDK source tree")

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/rexglue.cmake")
    message(FATAL_ERROR "app/generated/rexglue.cmake is missing: run tools/codegen.sh first")
endif()
include(generated/rexglue.cmake)

# The build re-runs the generator by itself whenever an input in codegen.d changes (manifest, toml, xex, the generator
# binary). That rewrites app/generated and would silently drop the codegen flavour's post-codegen steps (no-volatile
# pch + loop barriers, direct calls): they are appended to the same command (sa2-nx). The Switch cross build has no
# generator target, so this only applies on the PC.
if(TARGET mk9_codegen)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_custom_command(OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/generated/default/codegen.build.stamp" APPEND
        COMMAND "${CMAKE_COMMAND}" -E env "REXSDK_DIR=${REXSDK_DIR}"
                "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../tools/post_codegen.py")
endif()

set(MK9_SOURCES
    src/main.cpp
    # Linux: logs fatal host signals the SDK does not report (illegal instruction, ...) with the executable offset.
    src/mk9_crash_log.cpp
    # Still snapshots of the game image for testing (cvar mk9_snapshot_s).
    src/mk9_snapshot.cpp
    # Scripted gamepad on player 1 for automated tests (cvars mk9_input_script, mk9_input_script_file).
    src/mk9_script_input.cpp
    # Presented frames per second in the log (hook on the game's Direct3D swap, sub_82BF6D58).
    src/mk9_frame_counter.cpp
    src/mk9_movie.cpp
    # Switch: every thread's stack in logs/rex/rex_crash.log when no frame is presented for a while (mk9_hang_dump_s).
    src/mk9_hang_watch.cpp
    # Recordings of the game image for frame-by-frame comparisons (sa2-nx's recorder: sa2_video_s, lockstep).
    src/sa2_capture.cpp
    # Stage 4: the native renderer (src/me, sa2-nx's copy of masseffect-nx's PM4-ring Vulkan renderer as adapted by
    # banjo-nx and tuned in usf4-nx; files kept as there so fixes can be taken over). It replaces the emulated Xenos GPU
    # with sa2_renderer_native = true. MK9's own Direct3D hooks are in src/me/mk9_me_hooks.cpp
    # (docs/mk9-notes.md); usf4-nx's are at USF4 addresses and are not built here.
    src/me/me_native_system.cpp
    src/me/me_ring_wait.cpp
    src/me/me_renderdoc.cpp
    src/me/native/sa2_native_draws.cpp
    src/me/native/sa2_deferred_recording.cpp
    src/me/native/sa2_native_targets.cpp
    src/me/native/sa2_native_shaders.cpp
    src/me/native/sa2_native_textures_pool.cpp
    src/me/native/sa2_shader_library.cpp
    src/me/native/sa2_native_glue.cpp
    # MK9's Direct3D addresses (shader load, Draw*, shader container parse) and the shader dump (mk9_dump_shaders).
    src/me/mk9_me_hooks.cpp
    # CPU enhancement: Direct3D's GPU wait (sub_82BF5680) sleeps until the ring progresses (from usf4-nx).
    src/mk9_gpu_wait.cpp
    # CPU enhancement: the sound engine's mix-matrix routine (sub_82E1E148, float16 gain ramps) native.
    src/mk9_native_mix.cpp
    src/mk9_native_constants.cpp
    # The SA2-only parts of the renderer glue that are not ported (native Direct3D layer, census): off.
    src/me/mk9_api_stubs.cpp
    # Render resolution per console mode (docked / handheld) and MSAA: sa2_render_height_*, sa2_msaa.
    src/me/sa2_render_mode.cpp
    # Switchable enhancements (sa2-nx docs/enhancements.md): masters sa2_cpu_enhancements / sa2_noncpu_enhancements.
    src/cpu/sa2_enhancements.cpp
    src/cpu/sa2_gpu_boost.cpp
)
if(NOT REXGLUE_PLATFORM_SWITCH)
    # Switch profiler counters the native renderer calls (no-ops on PC).
    list(APPEND MK9_SOURCES src/me/me_pc_stubs.cpp)
    # Sampling CPU profiler for PC tests (from nfsmw-nx via banjo-nx and sa2-nx).
    list(APPEND MK9_SOURCES src/profiler/sa2_cpu_profile_pc.cpp)
endif()

# Diagnostic builds only (temporary hooks generated under out/diag): extra sources, empty by default.
set(MK9_EXTRA_SOURCES "" CACHE STRING "Extra source files for diagnostic builds (empty for normal builds)")
if(MK9_EXTRA_SOURCES)
    list(APPEND MK9_SOURCES ${MK9_EXTRA_SOURCES})
endif()

if(WIN32)
    add_executable(mk9 WIN32 ${MK9_SOURCES})
else()
    add_executable(mk9 ${MK9_SOURCES})
endif()

# The codegen flavour (tools/codegen.sh --flavor) names the build in the first "[enh]" log line (sa2-nx).
set(MK9_FLAVOR_FILE "${CMAKE_CURRENT_SOURCE_DIR}/generated/codegen_flavor.txt")
set(MK9_BUILD_FLAVOR "classic codegen")
if(EXISTS "${MK9_FLAVOR_FILE}")
    file(READ "${MK9_FLAVOR_FILE}" MK9_BUILD_FLAVOR)
    string(STRIP "${MK9_BUILD_FLAVOR}" MK9_BUILD_FLAVOR)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${MK9_FLAVOR_FILE}")
endif()
set_source_files_properties(src/cpu/sa2_enhancements.cpp PROPERTIES
    COMPILE_DEFINITIONS "SA2_BUILD_FLAVOR=\"${MK9_BUILD_FLAVOR}\"")

# Link the SDK. Stage 3 draws with ReXGlue's Xenos GPU emulation; the native renderer replaces it in stage 4.
rexglue_setup_target(mk9 GPU_PLUGINS xenos)

# mk9-nx (2026-10-24): the SDK's FFmpeg (its WMV3 decoder) for the cutscenes: src/mk9_movie.cpp.
target_link_libraries(mk9 PRIVATE libavcodec)
# An emulator (Eden, Ryujinx) starts NROs without argv[0]: the SDK then takes the NRO's usual place on the SD for its
# folder (sdk/src/core/filesystem_posix.cpp, from banjo-nx through sa2-nx). hbloader on the console always passes argv[0].
if(REXGLUE_PLATFORM_SWITCH AND TARGET rexcore)
    target_compile_definitions(rexcore PRIVATE REX_SWITCH_FALLBACK_EXE="/switch/mk9-nx/mk9.nro")
endif()

# SPIR-V initializers of the native renderer's EDRAM transfer / utility shaders (src/me/native/shaders), compiled
# on the host with glslangValidator (built from the SDK's glslang into out/glslang: cmake -S sdk/thirdparty/glslang -B out/glslang, target glslangValidator). The Switch build
# uses the same host tool: SPIR-V is platform independent.
set(SA2_GLSLANG_VALIDATOR "${CMAKE_CURRENT_SOURCE_DIR}/../out/glslang/StandAlone/glslangValidator${CMAKE_EXECUTABLE_SUFFIX}"
    CACHE FILEPATH "Host glslangValidator")
set(SA2_GENERATED_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated-shaders")
set(SA2_EDRAM_SHADER_INCLUDES)
foreach(_shader sa2_edram_7e3_to_rgba8 sa2_edram_rgba8_to_7e3 sa2_edram_16f_to_16f me_resolve_exp_bias me_edram_depth_to_rgba8 me_edram_depth_to_16f me_edram_import me_edram_color_to_depth me_edram_depth_to_depth me_edram_depth_to_stencil me_edram_color_to_stencil me_edram_depth_to_depth_msaa2 me_edram_depth_msaa2_to_depth_1x me_edram_depth_to_stencil_msaa2 me_edram_depth_msaa2_to_stencil_1x me_edram_raw64_to_rgba8 me_edram_rgba8_to_raw64 me_edram_raw64_to_16f me_edram_16f_to_raw64 me_edram_raw64_to_raw64 me_edram_depth_to_raw64 me_edram_raw64_to_depth me_depth_resolve_guestspace me_depth_resolve_guestspace_msaa2 me_edram_stencil_to_buffer me_edram_stencil_msaa2_to_buffer me_edram_color_stencil_to_buffer me_edram_color_to_color me_edram_r64_to_r64 sa2_resolve_copy sa2_depth_resolve_ms)
    set(_extension comp)
    if(_shader STREQUAL "me_edram_import")
        set(_extension vert)
    elseif(_shader MATCHES "^me_edram_(color_to_depth|depth_to_depth|depth_to_stencil|color_to_stencil|depth_to_depth_msaa2|depth_msaa2_to_depth_1x|depth_to_stencil_msaa2|depth_msaa2_to_stencil_1x|raw64_to_depth|color_to_color|r64_to_r64)$" OR _shader STREQUAL "sa2_resolve_copy" OR
           _shader STREQUAL "sa2_depth_resolve_ms")
        set(_extension frag)
    endif()
    set(_source "${CMAKE_CURRENT_SOURCE_DIR}/src/me/native/shaders/${_shader}.${_extension}")
    set(_output "${SA2_GENERATED_SHADER_DIR}/${_shader}.inc")
    set(_target_flags)
    if(_shader MATCHES "^me_edram_(stencil_to_buffer|color_stencil_to_buffer|stencil_msaa2_to_buffer|depth_to_depth_msaa2|depth_msaa2_to_depth_1x|depth_to_stencil_msaa2|depth_msaa2_to_stencil_1x)$" OR _shader STREQUAL "me_depth_resolve_guestspace_msaa2")
        set(_target_flags --target-env vulkan1.2)
        set(_output "${SA2_GENERATED_SHADER_DIR}/${_shader}_generated.inc")
    endif()
    add_custom_command(OUTPUT "${_output}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${SA2_GENERATED_SHADER_DIR}"
        COMMAND "${SA2_GLSLANG_VALIDATOR}" -V ${_target_flags} -x -o "${_output}" "${_source}"
        DEPENDS "${_source}" VERBATIM)
    list(APPEND SA2_EDRAM_SHADER_INCLUDES "${_output}")
endforeach()
add_custom_target(sa2_edram_shaders DEPENDS ${SA2_EDRAM_SHADER_INCLUDES})
add_dependencies(mk9 sa2_edram_shaders)
target_include_directories(mk9 PRIVATE
    "${SA2_GENERATED_SHADER_DIR}"
    src
    src/me
    src/me/native
    src/me/native/shaders
    "${REXSDK_DIR}/src/core"
    # Precompiled SPIR-V of the SDK presenter (vulkan_spirv/guest_output_*): the renderer draws into its output.
    "${REXSDK_DIR}/src/ui/shaders")

# CPU-only SDK sources the native renderer reuses (draw-extent estimation, shader microcode interpretation): the
# GPU plugin's private symbols cannot satisfy application-side use.
target_sources(mk9 PRIVATE
    "${REXSDK_DIR}/src/graphics/register_file.cpp"
    "${REXSDK_DIR}/src/graphics/format/ucode.cpp"
    "${REXSDK_DIR}/src/graphics/pipeline/shader/interpreter.cpp")

# The renderer includes xxhash.h and renderdoc_app.h directly; when the SDK is built from source they are private to
# its libraries.
foreach(_dep xxhash renderdoc)
    if(TARGET ${_dep})
        target_include_directories(mk9 PRIVATE $<TARGET_PROPERTY:${_dep},INTERFACE_INCLUDE_DIRECTORIES>)
    endif()
endforeach()

# Settings template next to the executable, only when there is none yet (the copy there is the user's to edit).
if(NOT REXGLUE_PLATFORM_SWITCH)  # the Switch writes its mk9.toml itself (DefaultConfigFile)
    add_custom_command(TARGET mk9 POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -DSRC=${CMAKE_CURRENT_SOURCE_DIR}/mk9.toml -DDST=$<TARGET_FILE_DIR:mk9>/mk9.toml
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/copy_if_missing.cmake"
        VERBATIM)
endif()

# The SDK compiles itself with these (directory scope), but they do not reach the generated game code, which is
# built in this directory. Fused multiply-add from separate PPC fmul/fadd drifts from the Xbox 360's results, and
# strict aliasing is unsafe for recompiled code (as in sa2-nx).
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    foreach(target mk9 mk9_recomp)
        if(TARGET ${target})
            target_compile_options(${target} PRIVATE -ffp-contract=off -fno-strict-aliasing)
        endif()
    endforeach()
endif()

# --- Nintendo Switch (stage 5) ------------------------------------------------------------------------------------
# Configure with the devkitA64 toolchain and the Mesa Horizon SDK (docs/building.md, "Switch"):
#   cmake -S app -B out/nx -G Ninja -DCMAKE_BUILD_TYPE=Release
#         -DCMAKE_TOOLCHAIN_FILE=tools/switch/cmake/switch-devkitA64.cmake
#         -DREXGLUE_SWITCH_NVK_SDK=<mesa-...-horizon-sdk>/opt/devkitpro/portlibs/switch -DSA2_GLSLANG_VALIDATOR=<host tool>
# Ported from sa2-nx's app/CMakeLists.txt (the parts that apply: no stage DLL, no PGO profile and no hot-function
# order for MK9 yet).
if(REXGLUE_PLATFORM_SWITCH)
    # mk9_register.cpp is one generated function of ~40,000 SetFunction calls that runs once at startup: GCC -O3
    # spends many minutes on it for nothing measurable (sa2-nx, from sasrt-nx).
    set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/generated/default/mk9_register.cpp" APPEND PROPERTY
                 COMPILE_OPTIONS -O0)

    # Link-time optimisation of the game code and the app (sa2-nx: nfsmw-nx / UnleashedRecomp-NX's set). The link
    # repeats -ffp-contract=off and -fno-strict-aliasing; xxHash without its aliasing-unsafe default under LTO.
    option(MK9_SWITCH_LTO "Link-time optimisation of the game code and the app (Switch)" ON)
    set(MK9_SWITCH_LTO_JOBS "4" CACHE STRING "Parallel LTRANS jobs of the Switch LTO link")
    target_compile_definitions(mk9 PRIVATE XXH_FORCE_MEMORY_ACCESS=0)
    if(MK9_SWITCH_LTO)
        foreach(target mk9_recomp mk9)
            if(TARGET ${target})
                target_compile_options(${target} PRIVATE -flto -fno-fat-lto-objects -fipa-pta)
            endif()
        endforeach()
        target_link_options(mk9 PRIVATE -flto=${MK9_SWITCH_LTO_JOBS} -flto-partition=balanced -fipa-pta
                            -ffp-contract=off -fno-strict-aliasing)
        message(STATUS "mk9: Switch LTO on (${MK9_SWITCH_LTO_JOBS} LTRANS jobs)")
    endif()
    # Line tables for the app's own code, so console crash and profile addresses map to source lines (addr2line on
    # the archived ELF); DWARF 4, as sa2-nx found DWARF 5 mixed up files under LTO. The game code stays without.
    target_compile_options(mk9 PRIVATE -g1 -gdwarf-4)
    target_link_options(mk9 PRIVATE -g1 -gdwarf-4)

    # Fixed settings compiled in, and the player's mk9.toml written next to the NRO when missing.
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/mk9_switch_defaults.toml" MK9_BUILTIN_CONFIG)
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/mk9_switch.toml" MK9_PLAYER_CONFIG)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                 "${CMAKE_CURRENT_SOURCE_DIR}/mk9_switch_defaults.toml" "${CMAKE_CURRENT_SOURCE_DIR}/mk9_switch.toml")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/mk9_builtin_config.h.in"
                   "${CMAKE_CURRENT_BINARY_DIR}/builtin_config/mk9_builtin_config.h" @ONLY)
    target_include_directories(mk9 PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/builtin_config")
    target_compile_definitions(mk9 PRIVATE MK9_BUILTIN_CONFIG=1)

    # The homebrew menu's entry (app/icon.jpg: an original 256x256 JPEG made for this port by tools/make_icon.py, no game artwork). A personal
    # app/icon_local.jpg (256x256 JPEG, git-ignored: it may be game artwork) replaces it when present.
    set(MK9_NRO_ICON "${MK9_NRO_ICON}")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/icon_local.jpg")
        set(MK9_NRO_ICON "${CMAKE_CURRENT_SOURCE_DIR}/icon_local.jpg")
    endif()
    rexglue_switch_add_nro(mk9
        NAME "Mortal Kombat Komplete Edition"
        AUTHOR "Thorhax"
        VERSION "0.0.1"
        ICON "${MK9_NRO_ICON}")
endif()

# mk9-nx (2026-10-20): Renut's Direct3D-level renderer (reNut by masterspike52, through banjo-nx and sa2-nx 1.0.3;
# docs/mk9-renut-census.md): no PM4 ring and no EDRAM model, plume's deko3d back end on the Switch. Chosen at boot with
# sa2_renderer = "renut" (the renderer keeps sa2-nx's key names, as src/me keeps its own).
option(MK9_RENUT_RENDERER "Link Renut's Direct3D-level renderer (sa2_renderer = renut)" ON)
if(MK9_RENUT_RENDERER)
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/mk9_renut_gpu.cmake")
endif()
