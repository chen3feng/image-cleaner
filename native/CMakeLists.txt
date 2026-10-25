cmake_minimum_required(VERSION 3.10.2)
project(pixelup_core CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

# ncnn prebuilt binaries, fetched by tools/setup_ncnn.sh
set(NCNN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/ncnn/${ANDROID_ABI}")
if(NOT EXISTS "${NCNN_ROOT}/include/ncnn/net.h")
    message(FATAL_ERROR
        "ncnn prebuilt not found for ${ANDROID_ABI} under ${NCNN_ROOT}.\n"
        "Run: tools/setup_ncnn.sh")
endif()

add_library(ncnn STATIC IMPORTED)
set_target_properties(ncnn PROPERTIES
        IMPORTED_LOCATION "${NCNN_ROOT}/lib/libncnn.a"
        INTERFACE_INCLUDE_DIRECTORIES "${NCNN_ROOT}/include")

# Vulkan build of ncnn is statically linked against glslang/SPIRV; those archives
# reference each other circularly, so wrap the whole group in --start-group.
set(NCNN_EXTRA_LIB_NAMES
        glslang-default-resource-limits glslang MachineIndependent GenericCodeGen
        OSDependent OGLCompiler SPIRV)
set(NCNN_EXTRA_LIBS "")
foreach(_lib ${NCNN_EXTRA_LIB_NAMES})
    if(EXISTS "${NCNN_ROOT}/lib/lib${_lib}.a")
        list(APPEND NCNN_EXTRA_LIBS "${NCNN_ROOT}/lib/lib${_lib}.a")
    endif()
endforeach()

add_library(pixelup_core SHARED
        core/jni_bridge.cpp
        core/pipeline_capi.cpp
        core/pipeline.cpp
        core/model_manifest.cpp
        core/scrfd.cpp
        core/codeformer.cpp
        core/colorize.cpp
        core/inpaint.cpp
        core/realesrgan.cpp
        core/image_utils.cpp
        core/json_min.cpp
)

target_include_directories(pixelup_core PRIVATE core)

# cppFlags do not reach the linker; ncnn's static lib needs libomp.
set_property(TARGET pixelup_core APPEND_STRING PROPERTY
        LINK_FLAGS " -fopenmp -static-openmp")

find_library(log-lib log)
find_library(android-lib android)

target_link_libraries(pixelup_core
        "-Wl,--start-group"
        ncnn
        ${NCNN_EXTRA_LIBS}
        "-Wl,--end-group"
        vulkan
        ${log-lib}
        ${android-lib})
