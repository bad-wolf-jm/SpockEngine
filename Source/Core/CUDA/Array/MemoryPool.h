/// @file   MemoryPool.h
///
/// @brief  Simple memory pool
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert. All rights reserved.

#pragma once

#include <array>

#include "PointerView.h"

/** \namespace SE::Cuda
 *
 * @brief
 *
 */
namespace SE::Cuda
{

    class memory_pool_t;

    /// \struct MemoryBuffer
    ///
    /// @brief Buffer allocated from a memory pool
    ///
    /// Allocates a fixed amount of GPU memory for use by kernels. The memory pool allocates buffers on request. The purpose
    /// of the memory pool is to avoid repeated memory allocations during complex calculations. The memory pool frees the memory
    /// allocated to it upon destruction. As of now there is no link between the memory pool and the various buffers it allocates.
    ///
    ///
    class memory_buffer_t : public Internal::gpu_device_pointer_view_t
    {
        friend class memory_pool_t;

      public:
        memory_buffer_t()  = default;
        ~memory_buffer_t() = default;

        memory_buffer_t( const memory_buffer_t & ) = default;

        memory_buffer_t View( size_t aSize, size_t aOffset )
        {
            return memory_buffer_t( aSize, aOffset, *this );
        }
        
        memory_buffer_t View( size_t aSize, size_t aOffset ) const
        {
            return memory_buffer_t( aSize, aOffset, *this );
        }

        memory_buffer_t( size_t aSize, void *aDevicePointer )
            : Internal::gpu_device_pointer_view_t( aSize, aDevicePointer )
        {
        }

        memory_buffer_t( size_t aSize, size_t aOffset, Internal::gpu_device_pointer_view_t const &aDevicePointer )
            : Internal::gpu_device_pointer_view_t( aSize, aOffset, aDevicePointer )
        {
        }
    };

    /// \struct MemoryPool
    ///
    /// @brief Simple memory manager
    ///
    /// Allocates a fixed amount of GPU memory for use by kernels. The memory pool allocates buffers on request. The purpose
    /// of the memory pool is to avoid repeated memory allocations during complex calculations. The memory pool frees the memory
    /// allocated to it upon destruction. As of now there is no link between the memory pool and the various buffers it allocates.
    ///
    ///
    class memory_pool_t : public Internal::gpu_device_pointer_t
    {
      public:
        memory_pool_t()  = default;
        ~memory_pool_t() = default;

        /// @brief Allocates `aTotalSize` bytes of memory on the GPU
        ///
        /// @param aTotalSize Size, in bytes, of the memory to allocate.
        ///
        memory_pool_t( size_t aTotalSize );

        /// @brief Allocates buffer from the pool.
        ///
        /// @exception  std::runtime_error If trying to allocate more memory than is available in the pool
        ///
        /// @param aBytes Size, in bytes, of the buffer to allocate.
        ///
        memory_buffer_t Allocate( size_t aBytes );

        /// @brief Resets the pool.
        void Reset();

      protected:
        size_t mFreePtr   = 0; //!< Pointer to the free area of the memory pool
        size_t mTotalSize = 0; //!< Total size of the memory pool, in bytes.
    };

} // namespace SE::Cuda