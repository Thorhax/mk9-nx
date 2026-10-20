# mk9-nx (2026-10-20): sa2-nx's cmake/sa2_renut_gpu.cmake for MK9 (target mk9, MK9's shader dump; docs/mk9-renut-census.md).
# sa2-nx (2026-10-20): Renut's Direct3D-level renderer (github.com/masterspike52/reNut, as banjo-nx carries it in
# src/renut_engine/gpu, retargeted to SA2's Direct3D: docs/renut-renderer.md), chosen at boot with
# sa2_renderer = "renut" (src/renut_engine/gpu/banjo_renut_select.h). From banjo-nx's cmake/banjo_renut_gpu.cmake: Renut's
# gpu.cmake as banjo-nx uses it, with its dependencies where it looks for them: thirdparty/plume (with banjo-nx's deko3d
# back end), thirdparty/reblue-XenosRecomp (the translator, DXC, zstd, smol-v, xxHash), thirdparty/imgui,
# cmake/collect_captured_shaders.cmake.
#
# An OBJECT library, not a static one: its hooks replace the generated code's weak sub_X definitions, and a static
# archive's members are only pulled in for undefined symbols, so the hooks would silently not be linked.

# The shader containers its shader cache is built from. mk9-nx: the containers MK9 registers at run time (mk9_dump_shaders
# and Renut's own captures, out/renut_shader_dump). A normal variable, so it shadows gpu.cmake's cache entry.
set(RENUT_SHADER_DUMP_DIR "${CMAKE_SOURCE_DIR}/../out/renut_shader_dump")
set(RENUT_ASSETS_DIR "${CMAKE_SOURCE_DIR}/assets" CACHE PATH "Game assets (default.xex and Bundle/)")

# Renut's XenosRecomp is built as an ExternalProject (gpu.cmake): it gets this build's whole host toolchain, not only
# the compiler paths (tools/renut_gpu_import.py adds the variable to gpu.cmake). On the PC, clang for the MSVC ABI.
set(BANJO_RENUT_XENOS_RECOMP_ARGS "")
foreach(_banjo_var CMAKE_C_COMPILER_TARGET CMAKE_CXX_COMPILER_TARGET CMAKE_LINKER_TYPE CMAKE_C_FLAGS CMAKE_CXX_FLAGS
                   CMAKE_AR CMAKE_RC_COMPILER CMAKE_MT)
    if(DEFINED ${_banjo_var} AND NOT "${${_banjo_var}}" STREQUAL "")
        list(APPEND BANJO_RENUT_XENOS_RECOMP_ARGS "-D${_banjo_var}=${${_banjo_var}}")
    endif()
endforeach()

# Renut's SDK exports o1heap as rex::o1heap; banjo-nx's SDK fork names it o1heap.
if(TARGET o1heap AND NOT TARGET rex::o1heap)
    add_library(rex::o1heap ALIAS o1heap)
endif()

# The Switch build is a cross build on a Windows host: Renut's XenosRecomp (a host tool) cannot be built with the
# console's compiler. Its shader cache comes from a PC build of the same tree (tools/renut_switch_shader_cache.py
# writes a copy without the DXIL arrays, which only Windows reads).
if(REXGLUE_PLATFORM_SWITCH)
    # plume's Switch backend (UnleashedRecomp-NX's patches, thirdparty/plume): VK_NN_vi_surface on libnx's window and
    # Mesa's static NVK, the same driver the SDK links (REXGLUE_SWITCH_NVK_SDK).
    set(PLUME_PLATFORM_SWITCH ON CACHE BOOL "" FORCE)
    set(PLUME_SWITCH_DEVKITPRO "${DEVKITPRO}" CACHE PATH "" FORCE)
    set(PLUME_SWITCH_NVK_ROOT "${REXGLUE_SWITCH_NVK_SDK}" CACHE PATH "" FORCE)
    set(PLUME_SWITCH_NVK_LIBRARY "${REXGLUE_SWITCH_NVK_SDK}/lib/libvulkan.a" CACHE FILEPATH "" FORCE)
    # sa2-nx (2026-10-20): REXGLUE_SWITCH_NO_NVK (the test builds): no NVK driver in the NRO, for plume too.
    set(PLUME_SWITCH_NO_NVK "${REXGLUE_SWITCH_NO_NVK}" CACHE BOOL "" FORCE)
    # banjo-nx (2026-10-17): plume's deko3d back end (renut_gpu_backend = "deko3d", docs/deko3d-renderer.md) and the
    # uam it compiles shader variants with on the console.
    set(PLUME_SWITCH_UAM_DIR "${CMAKE_SOURCE_DIR}/thirdparty/uam" CACHE PATH "" FORCE)
    set(PLUME_DEKO3D ON CACHE BOOL "deko3d back end (Switch)" FORCE)
    # The host shaders' deko3d blobs (gpu.cmake renut_host_shader) need the PC tools tools/deko3d_build_tools.sh builds.
    set(BANJO_DEKO3D_HOST_SHADERS ON)
    # mk9-nx: the host tools have no suffix on a Linux host (the Docker build); sa2-nx builds on Windows.
    if(CMAKE_HOST_WIN32)
        set(_mk9_host_exe ".exe")
    else()
        set(_mk9_host_exe "")
    endif()
    set(BANJO_DEKO3D_SPIRV_CROSS "${CMAKE_SOURCE_DIR}/../out/deko3d_tools/spirv-cross-build/spirv-cross${_mk9_host_exe}" CACHE FILEPATH
        "SPIRV-Cross for the deko3d host shaders (tools/deko3d_build_tools.sh)")
    set(BANJO_DEKO3D_UAM "${CMAKE_SOURCE_DIR}/../out/deko3d_tools/uam-build/uam${_mk9_host_exe}" CACHE FILEPATH
        "uam (thirdparty/uam built for the PC) for the deko3d host shaders (tools/deko3d_build_tools.sh)")
    foreach(_banjo_tool BANJO_DEKO3D_SPIRV_CROSS BANJO_DEKO3D_UAM)
        if(NOT EXISTS "${${_banjo_tool}}")
            message(FATAL_ERROR "No ${${_banjo_tool}}: run tools/deko3d_build_tools.sh first")
        endif()
    endforeach()
    # The game shaders' GLSL for deko3d (tools/deko3d_shader_package.py), embedded next to the SPIR-V cache.
    # sa2-nx 1.0.3 (from peggle2-nx): ON = the files of tools/renut_release_shaders.py (app/out/renut_switch_release):
    # no translated game shader built in and the shader table without SPIR-V; every game shader comes from
    # deko3d_glsl.bin beside the NRO, which the web installer makes from the player's own files. OFF = every translated
    # shader built in (development and test builds: game data, not for distribution).
    option(SA2_RENUT_RELEASE_SHADERS "Switch: no translated game shader in the NRO (deko3d_glsl.bin beside it)" OFF)
    if(SA2_RENUT_RELEASE_SHADERS)
        set(_sa2_renut_shaders "${CMAKE_SOURCE_DIR}/out/renut_switch_release")
    else()
        set(_sa2_renut_shaders "${CMAKE_SOURCE_DIR}/out/renut_switch")
    endif()
    set(BANJO_DEKO3D_GLSL_CPP "${_sa2_renut_shaders}/deko3d_glsl_switch.cpp" CACHE FILEPATH
        "The deko3d GLSL package's source (tools/deko3d_shader_package.py)" FORCE)
    set(BANJO_RENUT_SHADER_CACHE_CPP "${_sa2_renut_shaders}/shader_cache_switch.cpp" CACHE FILEPATH
        "Renut's shader cache for the Switch build (from a PC build: tools/renut_switch_shader_cache.py)" FORCE)
    function(banjo_renut_shader_cache_override TARGET)
        if(NOT EXISTS "${BANJO_RENUT_SHADER_CACHE_CPP}")
            message(FATAL_ERROR "No ${BANJO_RENUT_SHADER_CACHE_CPP}: build the PC tree first, then run "
                                "tools/renut_switch_shader_cache.py (Renut's XenosRecomp runs on the host)")
        endif()
        target_sources(${TARGET} PRIVATE "${BANJO_RENUT_SHADER_CACHE_CPP}")
        set_source_files_properties("${BANJO_RENUT_SHADER_CACHE_CPP}" PROPERTIES SKIP_PRECOMPILE_HEADERS ON)
    endfunction()
endif()

# sa2-nx: the shader steps run Python (the early alpha pass, and the host shaders' deko3d blobs on the Switch build):
# found here too, as the Switch configure has no generator target that would find it first.
if(NOT Python3_EXECUTABLE)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()
if(NOT PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
endif()

include("${CMAKE_SOURCE_DIR}/src/renut_engine/gpu/gpu.cmake")

add_library(mk9_renut_gpu OBJECT
    "${CMAKE_SOURCE_DIR}/src/renut_engine/gpu/banjo_renut_select.cpp"
    "${CMAKE_SOURCE_DIR}/src/renut_engine/gpu/banjo_renut_tuning.cpp"
    "${CMAKE_SOURCE_DIR}/src/renut_engine/gpu/banjo_renut_checks.cpp"
    "${CMAKE_SOURCE_DIR}/src/renut_engine/gpu/banjo_renut_record.cpp"
    # banjo-nx (2026-10-19, latency batch 3): the display's refresh for the frame limiter (it reads the presents' fences)
    "${CMAKE_SOURCE_DIR}/src/renut_engine/display_sync.cpp"
    "${CMAKE_SOURCE_DIR}/src/renut_engine/perf/frame_stats.cpp"
    # sa2-nx: banjo-nx's SDK and generated-code pieces the renderer uses (the frame latency probe from futari-nx, the
    # store cards switched off), app-side here.
    "${CMAKE_SOURCE_DIR}/src/renut_engine/sa2_compat/frame_latency.cpp"
    "${CMAKE_SOURCE_DIR}/src/renut_engine/sa2_compat/sa2_compat.cpp"
    # sa2-nx: the game's frames at its presentation interval (the console's vertical blank).
    "${CMAKE_SOURCE_DIR}/src/renut_engine/gpu/sa2_renut_pacing.cpp")
target_include_directories(mk9_renut_gpu PRIVATE
    "${CMAKE_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/src"
    # sa2-nx: banjo-nx's headers the renderer's tuning includes, answered by SA2's (cpu/banjo_enhancements.h, ...).
    "${CMAKE_SOURCE_DIR}/src/renut_engine/sa2_compat"
    "${REXGLUE_ENTRYPOINT_INCLUDE_DIR}"
    # banjo_renut_tuning.cpp: the Switch thread helpers (threading_switch.h, through src/me/me_platform_compat.h).
    "${REXSDK_DIR}/src/core")
target_link_libraries(mk9_renut_gpu PRIVATE rex::runtime)
if(REXGLUE_PLATFORM_SWITCH)
    # banjo-nx (2026-10-17): the deko3d back end's glue and the game shaders' GLSL package (docs/deko3d-renderer.md).
    target_sources(mk9_renut_gpu PRIVATE "${CMAKE_SOURCE_DIR}/src/renut_engine/gpu/banjo_renut_deko3d.cpp")
    if(EXISTS "${BANJO_DEKO3D_GLSL_CPP}")
        target_sources(mk9_renut_gpu PRIVATE "${BANJO_DEKO3D_GLSL_CPP}")
        set_source_files_properties("${BANJO_DEKO3D_GLSL_CPP}" PROPERTIES SKIP_PRECOMPILE_HEADERS ON)
    elseif(SA2_RENUT_RELEASE_SHADERS)
        message(STATUS "sa2: no GLSL package built in (SA2_RENUT_RELEASE_SHADERS): deko3d_glsl.bin beside the NRO")
    else()
        message(WARNING "No ${BANJO_DEKO3D_GLSL_CPP}: renut_gpu_backend = deko3d will have no game shaders "
                        "(tools/deko3d_shader_package.py)")
    endif()
endif()
if(COMMAND rexglue_apply_target_settings)
    rexglue_apply_target_settings(mk9_renut_gpu)
endif()
renut_gpu_setup(mk9_renut_gpu)
target_compile_definitions(mk9_renut_gpu PRIVATE SA2_RENUT_RENDERER=1)
if(REXGLUE_PLATFORM_SWITCH AND REXGLUE_SWITCH_NO_NVK)
    # sa2-nx (2026-10-20): no Vulkan driver in this build: Renut on deko3d whatever sa2_renderer / renut_gpu_backend
    # say (per source, so nothing else recompiles).
    set_property(SOURCE "${CMAKE_SOURCE_DIR}/src/renut_engine/gpu/banjo_renut_select.cpp"
                        "${RENUT_GPU_DIR}/backend/backend.cpp"
                 APPEND PROPERTY COMPILE_DEFINITIONS SA2_SWITCH_NO_NVK=1)
endif()
if(REXGLUE_PLATFORM_SWITCH)
    # sa2-nx (2026-10-20): line tables for the renderer's code, so the console profile's samples
    # (tools/switch_profile_report.py on the build's ELF) resolve to lines. GCC's code does not change with -g; the NRO
    # is built from the stripped ELF.
    target_compile_options(mk9_renut_gpu PRIVATE -g1)
endif()
# The game's hooks into it (src/me, src/cpu: sa2_renut_yield.h) and the app's wiring (sa2_app.h).
target_link_libraries(mk9 PRIVATE mk9_renut_gpu)
target_compile_definitions(mk9 PRIVATE SA2_RENUT_RENDERER=1)
