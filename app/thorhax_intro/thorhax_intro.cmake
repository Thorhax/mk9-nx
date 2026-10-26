# Thorhax intro: adds the startup video player (thorhax_intro.h) to a target.
#   include(thorhax_intro/thorhax_intro.cmake)
#   thorhax_intro_add(<target>)
function(thorhax_intro_add target)
    set(_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
    target_sources(${target} PRIVATE "${_dir}/thorhax_intro.cpp" "${_dir}/pl_mpeg_impl.cpp")
    target_include_directories(${target} PRIVATE "${_dir}")
    # The video and the sound are linked in with .incbin: rebuild when they change, and no LTO for that file so the
    # assembler's symbols reach the linker.
    set_source_files_properties("${_dir}/thorhax_intro.cpp" PROPERTIES
        COMPILE_DEFINITIONS "THORHAX_INTRO_DIR=\"${_dir}\""
        OBJECT_DEPENDS "${_dir}/thorhax_intro.mpg;${_dir}/thorhax_intro.pcm"
        COMPILE_OPTIONS "-fno-lto")
endfunction()
