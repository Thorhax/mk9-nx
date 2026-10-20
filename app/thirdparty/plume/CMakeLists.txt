cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

if(APPLE AND NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    # 10.15 because of our use of supportsFamily()
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING "Minimum macOS deployment target")
endif()

project(plume)

if(APPLE)
    enable_language(OBJC OBJCXX)
endif()

string(COMPARE EQUAL ${CMAKE_SYSTEM_NAME} "Linux" IS_LINUX)

# Project options
include(CMakeDependentOption)
cmake_dependent_option(PLUME_SDL_VULKAN_ENABLED "Enable SDL Vulkan integration" OFF IS_LINUX OFF)
cmake_dependent_option(PLUME_D3D12_AGILITY_SDK_ENABLED "Enable D3D12 Agility SDK" OFF WIN32 OFF)
cmake_dependent_option(PLUME_APPLE_RETINA_ENABLED "Enable Apple Retina display support" ON APPLE ON)
option(PLUME_PLATFORM_SWITCH "Enable Nintendo Switch/libnx Vulkan integration" OFF)
option(PLUME_BUILD_EXAMPLES "Build example applications" OFF)

if(PLUME_PLATFORM_SWITCH)
    set(PLUME_SWITCH_DEVKITPRO "/opt/devkitpro" CACHE PATH "devkitPro root for Switch builds")
    set(PLUME_SWITCH_NVK_ROOT "/opt/nvk-switch" CACHE PATH "Mesa/NVK Switch build or install root")
    set(PLUME_SWITCH_NVK_LIBRARY "${PLUME_SWITCH_NVK_ROOT}/lib/libvulkan.a"
        CACHE FILEPATH "Mesa/NVK Switch Vulkan static archive")
    # sa2-nx (2026-10-20): the driver not linked (the parent defines vk_icdGetInstanceProcAddr and
    # vk_icdNegotiateLoaderICDInterfaceVersion as "no driver": ReXGlue's REXGLUE_SWITCH_NO_NVK); the Vulkan back end
    # still compiles, and cannot start.
    option(PLUME_SWITCH_NO_NVK "Do not link the NVK driver (the parent provides failing ICD entry points)" OFF)
endif()

# Windows-specific definitions
if(WIN32)
    add_compile_definitions(NOMINMAX)
endif()

# Enable SDL Vulkan support
if(PLUME_SDL_VULKAN_ENABLED)
    if (NOT TARGET SDL2::SDL2)
        find_package(SDL2 REQUIRED)
    else()
        set(SDL2_INCLUDE_DIRS "$<TARGET_PROPERTY:SDL2::SDL2,INTERFACE_INCLUDE_DIRECTORIES>")
    endif()

    message(STATUS "Plume - Building with SDL2_INCLUDE_DIRS: ${SDL2_INCLUDE_DIRS}")
endif()

# Enable D3D12 Agility SDK support
if(PLUME_D3D12_AGILITY_SDK_ENABLED)
    find_package(directx-headers CONFIG REQUIRED)
    find_package(directx12-agility CONFIG REQUIRED)
endif()

# Print status messages
message(STATUS "Plume - Building with backends: Vulkan=1 Metal=${APPLE} D3D12=${WIN32}")
message(STATUS "Plume - SDL Vulkan integration: ${PLUME_SDL_VULKAN_ENABLED}")
message(STATUS "Plume - D3D12 Agility SDK: ${PLUME_D3D12_AGILITY_SDK_ENABLED}")
message(STATUS "Plume - Switch/libnx integration: ${PLUME_PLATFORM_SWITCH}")
message(STATUS "Plume - Building examples: ${PLUME_BUILD_EXAMPLES}")

# Basic source files that are always included
set(PLUME_SOURCES
    plume_vulkan.cpp
    plume_vulkan.h
)

# Platform-specific files
if(APPLE)
    list(APPEND PLUME_SOURCES
        plume_metal.cpp
        plume_metal.h
        plume_apple.mm
        plume_apple.h
    )
elseif(WIN32)
    list(APPEND PLUME_SOURCES
        plume_d3d12.cpp
        plume_d3d12.h
        ${CMAKE_CURRENT_SOURCE_DIR}/contrib/D3D12MemoryAllocator/src/D3D12MemAlloc.cpp
    )
endif()

# banjo-nx: UnleashedRecomp-NX built LSFG-VK frame generation into plume here; banjo-nx does not (plume_lsfg_stub.cpp).

# Create target
add_library(plume STATIC ${PLUME_SOURCES})

# Include directories
target_include_directories(plume PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/contrib/volk
    ${CMAKE_CURRENT_SOURCE_DIR}/contrib/Vulkan-Headers/include
    ${CMAKE_CURRENT_SOURCE_DIR}/contrib/VulkanMemoryAllocator/include
)

if(WIN32)
    target_include_directories(plume PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/contrib/D3D12MemoryAllocator/include
    )
endif()

if(PLUME_SDL_VULKAN_ENABLED)
    target_compile_definitions(plume PUBLIC PLUME_SDL_VULKAN_ENABLED)
    target_include_directories(plume PUBLIC ${SDL2_INCLUDE_DIRS})
endif()

if(PLUME_D3D12_AGILITY_SDK_ENABLED)
    target_compile_definitions(plume PUBLIC PLUME_D3D12_AGILITY_SDK_ENABLED)
    target_compile_definitions(plume PRIVATE D3D12MA_USING_DIRECTX_HEADERS)
    target_link_libraries(plume PRIVATE Microsoft::DirectX-Headers Microsoft::DirectX-Guids Microsoft::DirectX12-Agility)
endif()

if(PLUME_PLATFORM_SWITCH)
    if(NOT EXISTS "${PLUME_SWITCH_NVK_LIBRARY}")
        message(FATAL_ERROR "PLUME_SWITCH_NVK_LIBRARY does not exist: ${PLUME_SWITCH_NVK_LIBRARY}")
    endif()
    target_compile_definitions(plume PUBLIC
        __SWITCH__
        NX
        VK_USE_PLATFORM_VI_NN
    )
    target_include_directories(plume PUBLIC
        "${PLUME_SWITCH_NVK_ROOT}/include"
        "${PLUME_SWITCH_DEVKITPRO}/libnx/include"
        "${PLUME_SWITCH_DEVKITPRO}/portlibs/switch/include"
    )
    # banjo-nx: no LSFG-VK; plume_lsfg_stub.cpp declines frame generation (the bridge header is next to plume's sources).
    target_include_directories(plume PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_sources(plume PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/plume_lsfg_stub.cpp")
    target_compile_features(plume PUBLIC cxx_std_20)

    # Libraries the static driver may need besides libnx, zlib and zstd:
    #  - expat: Mesa built with xmlconfig (build-unified.sh enables it; the Docker build
    #    enables it when devkitPro's switch-libexpat is installed);
    #  - drm_nouveau: only older mesa-switch builds used switch-libdrm_nouveau; the current
    #    Horizon backend does not. Linked when present, as before.
    # Each is linked only if its archive exists; one nobody references adds nothing.
    # devkitPro's location: the parent project's DEVKITPRO when it passes one (UnleashedRecomp
    # passes the native Windows path), otherwise PLUME_SWITCH_DEVKITPRO.
    set(PLUME_SWITCH_PORTLIBS_ROOT "${PLUME_SWITCH_DEVKITPRO}")
    if(DEFINED DEVKITPRO AND NOT "${DEVKITPRO}" STREQUAL "")
        set(PLUME_SWITCH_PORTLIBS_ROOT "${DEVKITPRO}")
    endif()
    set(PLUME_SWITCH_NVK_EXTRA_LIBRARIES "")
    foreach(PLUME_SWITCH_LIBRARY_NAME IN ITEMS expat drm_nouveau)
        foreach(PLUME_SWITCH_LIBRARY_DIR IN ITEMS "${PLUME_SWITCH_NVK_ROOT}/lib" "${PLUME_SWITCH_PORTLIBS_ROOT}/portlibs/switch/lib")
            if(EXISTS "${PLUME_SWITCH_LIBRARY_DIR}/lib${PLUME_SWITCH_LIBRARY_NAME}.a")
                list(APPEND PLUME_SWITCH_NVK_EXTRA_LIBRARIES "${PLUME_SWITCH_LIBRARY_DIR}/lib${PLUME_SWITCH_LIBRARY_NAME}.a")
                break()
            endif()
        endforeach()
    endforeach()
    message(STATUS "plume: NVK driver ${PLUME_SWITCH_NVK_LIBRARY}, extra libraries: ${PLUME_SWITCH_NVK_EXTRA_LIBRARIES}")

    # One group, so the linker rescans the driver and its dependencies until nothing new
    # resolves (the driver and its helper archives reference each other).
    if(PLUME_SWITCH_NO_NVK)
        message(STATUS "plume: NVK driver not linked (PLUME_SWITCH_NO_NVK)")
        target_link_libraries(plume PUBLIC zstd z nx)
    else()
    target_link_libraries(plume PUBLIC
        "-Wl,-u,vk_icdGetInstanceProcAddr"
        "-Wl,-u,vk_icdNegotiateLoaderICDInterfaceVersion"
        "-Wl,--start-group"
        "${PLUME_SWITCH_NVK_LIBRARY}"
        ${PLUME_SWITCH_NVK_EXTRA_LIBRARIES}
        zstd
        z
        nx
        "-Wl,--end-group"
    )
    endif()

    # banjo-nx (2026-10-17): the deko3d back end (plume_deko3d*.cpp, banjo-nx docs/deko3d-renderer.md): devkitPro's
    # deko3d and uam (PLUME_SWITCH_UAM_DIR: banjo-nx vendors SwitchWakerHD's patched copy, whose CMake builds the static
    # uamlib for the Switch when NX_ROOT is set).
    option(PLUME_DEKO3D "deko3d back end (Switch)" ON)
    set(PLUME_SWITCH_UAM_DIR "" CACHE PATH "uam source (banjo-nx: app/thirdparty/uam)")
    if(PLUME_DEKO3D)
        if(NOT EXISTS "${PLUME_SWITCH_UAM_DIR}/CMakeLists.txt")
            message(FATAL_ERROR "PLUME_DEKO3D needs PLUME_SWITCH_UAM_DIR (uam's source): ${PLUME_SWITCH_UAM_DIR}")
        endif()
        if(NOT TARGET uamlib)
            set(NX_ROOT "${PLUME_SWITCH_DEVKITPRO}/libnx")
            add_subdirectory("${PLUME_SWITCH_UAM_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/uam")
        endif()
        target_sources(plume PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/plume_deko3d.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/plume_deko3d_commands.cpp")
        target_compile_definitions(plume PUBLIC PLUME_DEKO3D=1)
        # uam's Mesa (19) and the static NVK's (a newer Mesa) define functions of the same names with other signatures
        # (ralloc, the linear allocator, hash tables): the linker gave uam's preprocessor NVK's linear_alloc_child, which
        # returned null (banjo-nx's first emulator runs, 2026-10-17). uam is linked into one object whose strong symbols
        # are local but its API (uam::*), so each copy of Mesa calls its own (plume_uam_localize.cmake).
        set(PLUME_DEKO3D_BIN "${PLUME_SWITCH_DEVKITPRO}/devkitA64/bin")
        set(PLUME_DEKO3D_HOST_EXE "")
        if(CMAKE_HOST_WIN32)
            set(PLUME_DEKO3D_HOST_EXE ".exe")
        endif()
        set(plume_uam_local "${CMAKE_CURRENT_BINARY_DIR}/uam_local.o")
        add_custom_command(
            OUTPUT "${plume_uam_local}"
            COMMAND "${CMAKE_COMMAND}" "-DLD=${PLUME_DEKO3D_BIN}/aarch64-none-elf-ld${PLUME_DEKO3D_HOST_EXE}"
                    "-DNM=${PLUME_DEKO3D_BIN}/aarch64-none-elf-nm${PLUME_DEKO3D_HOST_EXE}"
                    "-DOBJCOPY=${PLUME_DEKO3D_BIN}/aarch64-none-elf-objcopy${PLUME_DEKO3D_HOST_EXE}"
                    "-DARCHIVE=$<TARGET_FILE:uamlib>" "-DOUT=${plume_uam_local}"
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/plume_uam_localize.cmake"
            DEPENDS uamlib "${CMAKE_CURRENT_SOURCE_DIR}/plume_uam_localize.cmake"
            COMMENT "uam with its strong symbols local but its API (uam_local.o)"
            VERBATIM)
        add_custom_target(plume_uam_local DEPENDS "${plume_uam_local}")
        add_dependencies(plume plume_uam_local)
        target_include_directories(plume PRIVATE "${PLUME_SWITCH_UAM_DIR}")
        # PLUME_DEKO3D_DEBUG: devkitPro's debug deko3d (libdeko3dd.a), which checks every call's input and reports through
        # the device's debug callback (plume_deko3d's log): for tests, slower.
        option(PLUME_DEKO3D_DEBUG "Link the debug deko3d library (input checks)" OFF)
        if(PLUME_DEKO3D_DEBUG)
            set(plume_deko3d_lib "${PLUME_SWITCH_DEVKITPRO}/libnx/lib/libdeko3dd.a")
            # and guard bytes after every memory block, checked at each submission (plume_deko3d.cpp makeBlock)
            target_compile_definitions(plume PRIVATE PLUME_DEKO3D_GUARDS=1)
        else()
            set(plume_deko3d_lib "${PLUME_SWITCH_DEVKITPRO}/libnx/lib/libdeko3d.a")
        endif()
        target_link_libraries(plume PUBLIC "${plume_uam_local}" "${plume_deko3d_lib}" nx)
        message(STATUS "plume: deko3d back end with uam from ${PLUME_SWITCH_UAM_DIR}")
    endif()
endif()

# Platform-specific includes and shader compilation
if(APPLE)
    if(PLUME_APPLE_RETINA_ENABLED)
        target_compile_definitions(plume PRIVATE PLUME_APPLE_RETINA_ENABLED)
    endif()

    target_include_directories(plume PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/contrib/metal-cpp
    )
endif()

# Add examples if requested
if(PLUME_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
