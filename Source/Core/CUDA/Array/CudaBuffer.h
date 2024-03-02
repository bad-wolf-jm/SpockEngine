/** @file */

#pragma once

#include "Core/CUDA/Cuda.h"
#include <vector>

#include <fmt/core.h>

#include "Core/Logging.h"
#include "Core/Memory.h"

#include "Core/CUDA/CudaAssert.h"
#include "PointerView.h"

namespace SE::Cuda
{

#pragma once

    template <typename _Ty>
    struct gpu_array_t
    {
        _Ty   *DevicePointer = 0;
        size_t ElementCount  = 0;
    };

    class gpu_memory_view_t : public Internal::gpu_device_pointer_view_t
    {
      public:
        gpu_memory_view_t() = default;

        gpu_memory_view_t( const gpu_memory_view_t & ) = default;

        gpu_memory_view_t( size_t a_Size, size_t a_Offset, Internal::sGPUDevicePointerView &a_DevicePointer )
            : Internal::sGPUDevicePointerView( a_Size, a_Offset, a_DevicePointer )
        {
        }

        gpu_memory_view_t( size_t a_Size, void *a_DevicePointer )
            : Internal::sGPUDevicePointerView( a_Size, a_DevicePointer )
        {
        }

        gpu_memory_view_t View( size_t a_Size, size_t a_Offset ) { return gpu_memory_view_t( a_Size, a_Offset, *this ); }
    };

    class gpu_memory_t : public Internal::gpu_device_pointer_t
    {
      public:
        gpu_memory_t() = default;

        gpu_memory_t( size_t a_Size )
            : Internal::gpu_device_pointer_t( a_Size ){};

        ~gpu_memory_t() = default;

        template <typename _Ty>
        static gpu_memory_t Create( uint32_t aSize )
        {
            return gpu_memory_t( aSize * sizeof( _Ty ) );
        }

        template <typename _Ty>
        static gpu_memory_t Create( vector_t<_Ty> aVec )
        {
            gpu_memory_t lOut = gpu_memory_t::Create<_Ty>( aVec.size() );
            lOut.Upload( aVec );
            return lOut;
        }

        raw_pointer_t RawDevicePtr() { return mDevicePointer; }

        raw_pointer_t *RawDevicePtrP() { return &( mDevicePointer ); }
    };

} // namespace SE::Cuda
