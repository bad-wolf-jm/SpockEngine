/** @file */

#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>

#include <fmt/core.h>

#include "Core/Logging.h"
#include "Core/Memory.h"

#include "CudaAssert.h"
#include "PointerView.h"

namespace LTSE::Cuda
{

#pragma once

    template <typename _Ty> struct GPUArray
    {
        _Ty *DevicePointer  = 0;
        size_t ElementCount = 0;
    };

    class GPUMemoryView : public Internal::sGPUDevicePointerView
    {
      public:
        GPUMemoryView() = default;

        GPUMemoryView( const GPUMemoryView & ) = default;

        GPUMemoryView( size_t a_Size, size_t a_Offset, Internal::sGPUDevicePointerView &a_DevicePointer )
            : Internal::sGPUDevicePointerView( a_Size, a_Offset, a_DevicePointer )
        {
        }

        GPUMemoryView( size_t a_Size, void *a_DevicePointer )
            : Internal::sGPUDevicePointerView( a_Size, a_DevicePointer )
        {
        }

        GPUMemoryView View( size_t a_Size, size_t a_Offset ) { return GPUMemoryView( a_Size, a_Offset, *this ); }
    };

    class GPUMemory : public Internal::sGPUDevicePointer
    {
      public:
        GPUMemory() = default;

        GPUMemory( size_t a_Size )
            : Internal::sGPUDevicePointer( a_Size ){};

        ~GPUMemory() = default;

        template <typename _Ty> static GPUMemory Create( uint32_t aSize ) { return GPUMemory( aSize * sizeof( _Ty ) ); }
        template <typename _Ty> static GPUMemory Create( std::vector<_Ty> aVec )
        {
            GPUMemory lOut = GPUMemory::Create<_Ty>( aVec.size() );
            lOut.Upload( aVec );
            return lOut;
        }

        CUdeviceptr RawDevicePtr() { return mDevicePointer; }

        CUdeviceptr *RawDevicePtrP() { return &( mDevicePointer ); }
    };

} // namespace LTSE::Cuda
