find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--ptxas-options=-v;-use_fast_math)

if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
	# CUDA 12.x doesn't officially support GCC 13 yet, allow it explicitly.
	list(APPEND CUDA_NVCC_FLAGS "--allow-unsupported-compiler")
endif()

if (NOT MSVC)
	list(APPEND CUDA_NVCC_FLAGS "--disable-warnings")
endif()

list(APPEND CUDA_NVCC_FLAGS_RELEASE -O3)
list(APPEND CUDA_NVCC_FLAGS_DEBUG -G)

set(ETHMINER_CUDA_ARCHES "")

# Baseline architectures we always ship (bounded by toolkit support).
if (CUDA_VERSION VERSION_LESS 11.0)
	list(APPEND ETHMINER_CUDA_ARCHES 30)
endif()
if(CUDA_VERSION VERSION_LESS 12.0)
	list(APPEND ETHMINER_CUDA_ARCHES 35)
endif()
list(APPEND ETHMINER_CUDA_ARCHES 50 52 53 60 61 62)
if(NOT CUDA_VERSION VERSION_LESS 9.0)
	list(APPEND ETHMINER_CUDA_ARCHES 70)
endif()
if(NOT CUDA_VERSION VERSION_LESS 10.0)
	list(APPEND ETHMINER_CUDA_ARCHES 75)
endif()
if(NOT CUDA_VERSION VERSION_LESS 11.0)
	# NVIDIA A100 and NVIDIA DGX-A100
	list(APPEND ETHMINER_CUDA_ARCHES 80)
endif()
if(NOT CUDA_VERSION VERSION_LESS 11.1)
	# Tesla GA10x cards, RTX Ampere – RTX 3080/3090, RTX A6000, RTX A40
	list(APPEND ETHMINER_CUDA_ARCHES 86)
endif()
if(NOT CUDA_VERSION VERSION_LESS 11.8)
	# NVIDIA Ada Lovelace – RTX 40xx
	list(APPEND ETHMINER_CUDA_ARCHES 89)
endif()
if(NOT CUDA_VERSION VERSION_LESS 11.1)
	# Hopper/Blackwell & newer families (covers RTX 50xx / H100+)
	list(APPEND ETHMINER_CUDA_ARCHES 90)
endif()

# Allow COMPUTE to add extra archs instead of replacing the defaults (comma/semicolon separated).
if(COMPUTE)
	string(REPLACE "," ";" ETHMINER_USER_ARCHES "${COMPUTE}")
	foreach(_arch ${ETHMINER_USER_ARCHES})
		if(NOT _arch STREQUAL "")
			list(APPEND ETHMINER_CUDA_ARCHES ${_arch})
		endif()
	endforeach()
endif()

list(REMOVE_DUPLICATES ETHMINER_CUDA_ARCHES)

set(ETHMINER_MAX_ARCH 0)
foreach(_arch ${ETHMINER_CUDA_ARCHES})
	list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_${_arch},code=sm_${_arch}")
	if(_arch GREATER ETHMINER_MAX_ARCH)
		set(ETHMINER_MAX_ARCH ${_arch})
	endif()
endforeach()

# Emit PTX for forward compatibility with the highest architecture in the binary.
if(ETHMINER_MAX_ARCH GREATER 0)
	list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_${ETHMINER_MAX_ARCH},code=compute_${ETHMINER_MAX_ARCH}")
endif()

list(JOIN ETHMINER_CUDA_ARCHES "," ETHMINER_CUDA_ARCH_LIST_STR)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuda_arches.h.in
	${CMAKE_CURRENT_BINARY_DIR}/cuda_arches.h @ONLY)

file(GLOB sources "*.cpp" "*.cu")
file(GLOB headers "*.h" "*.cuh")
list(APPEND headers ${CMAKE_CURRENT_BINARY_DIR}/cuda_arches.h)

cuda_add_library(ethash-cuda STATIC ${sources} ${headers})
target_link_libraries(ethash-cuda ethcore ethash::ethash Boost::thread)
target_include_directories(ethash-cuda PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(ethash-cuda PRIVATE .. ${CMAKE_CURRENT_BINARY_DIR})
