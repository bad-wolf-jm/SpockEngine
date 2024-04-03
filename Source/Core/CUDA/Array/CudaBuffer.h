/** @file */

#pragma once

#include "Core/CUDA/Cuda.h"
#include <vector>

#include <fmt/core.h>

#include "Core/Logging.h"
#include "Core/Memory.h"

#include "Core/CUDA/CudaAssert.h"
#include "PointerView.h"

namespace numlua::cuda
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

        gpu_memory_view_t( size_t size, size_t offset, Internal::gpu_device_pointer_view_t &devicePointer )
            : Internal::gpu_device_pointer_view_t( size, offset, devicePointer )
        {
        }

        gpu_memory_view_t( size_t size, void *devicePointer )
            : Internal::gpu_device_pointer_view_t( size, devicePointer )
        {
        }

        gpu_memory_view_t View( size_t size, size_t offset )
        {
            return gpu_memory_view_t( size, offset, *this );
        }
    };

    class gpu_memory_t : public Internal::gpu_device_pointer_t
    {
      public:
        gpu_memory_t() = default;

        gpu_memory_t( size_t size )
            : Internal::gpu_device_pointer_t( size ){};

        ~gpu_memory_t() = default;

        template <typename _Ty>
        static gpu_memory_t Create( uint32_t size )
        {
            return gpu_memory_t( size * sizeof( _Ty ) );
        }

        template <typename _Ty>
        static gpu_memory_t Create( vector_t<_Ty> vec )
        {
            gpu_memory_t out = gpu_memory_t::Create<_Ty>( vec.size() );
            out.Upload( vec );
            return out;
        }

        raw_pointer_t RawDevicePtr()
        {
            return DevicePointer;
        }

        raw_pointer_t *RawDevicePtrP()
        {
            return &( DevicePointer );
        }
    };

} // namespace SE::Cuda
