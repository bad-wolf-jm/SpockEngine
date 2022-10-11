#pragma once

#include "CudaBuffer.h"

#include "Core/EntityRegistry/Registry.h"

namespace LTSE::Cuda
{

template<typename _Ty>
void CopyArray(GPUMemory<_Ty> a_Destination, GPUMemory<_Ty> a_Source) {
    size_t l_BytesToCopy = std::min(a_Destination.ByteSize(), a_Source.ByteSize());
    CUDA_ASSERT(cudaMemcpy((void*) a_Destination.device_data(), a_Source.device_data(), l_BytesToCopy, cudaMemcpyDeviceToDevice));
}

template<typename _Tx>
void CopyArray(GPUMemoryView<_Tx> a_Destination, GPUMemory<_Tx> a_Source) {
    size_t l_BytesToCopy = std::min(a_Destination.ByteSize(), a_Source.ByteSize());
    CUDA_ASSERT(cudaMemcpy((void*) a_Destination.device_data(), a_Source.device_data(), l_BytesToCopy, cudaMemcpyDeviceToDevice));
}

template<typename _Tx>
void CopyArray(GPUMemory<_Tx> a_Destination, GPUMemoryView<_Tx> a_Source) {
    size_t l_BytesToCopy = std::min(a_Destination.ByteSize(), a_Source.ByteSize());
    CUDA_ASSERT(cudaMemcpy((void*) a_Destination.device_data(), a_Source.device_data(), l_BytesToCopy, cudaMemcpyDeviceToDevice));
}

template<typename _Tx>
void CopyArray(GPUMemoryView<_Tx> a_Destination, GPUMemoryView<_Tx> a_Source) {
    size_t l_BytesToCopy = std::min(a_Destination.ByteSize(), a_Source.ByteSize());
    CUDA_ASSERT(cudaMemcpy((void*) a_Destination.device_data(), a_Source.device_data(), l_BytesToCopy, cudaMemcpyDeviceToDevice));
}

}