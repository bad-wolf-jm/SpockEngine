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

    class MemoryPool;

    /// \struct MemoryBuffer
    ///
    /// @brief Buffer allocated from a memory pool
    ///
    /// Allocates a fixed amount of GPU memory for use by kernels. The memory pool allocates buffers on request. The purpose
    /// of the memory pool is to avoid repeated memory allocations during complex calculations. The memory pool frees the memory
    /// allocated to it upon destruction. As of now there is no link between the memory pool and the various buffers it allocates.
    ///
    ///
    class MemoryBuffer : public Internal::sGPUDevicePointerView
    {
        friend class MemoryPool;

      public:
        MemoryBuffer()  = default;
        ~MemoryBuffer() = default;

        MemoryBuffer( const MemoryBuffer & ) = default;

        MemoryBuffer View( size_t aSize, size_t aOffset ) { return MemoryBuffer( aSize, aOffset, *this ); }
        MemoryBuffer View( size_t aSize, size_t aOffset ) const { return MemoryBuffer( aSize, aOffset, *this ); }

        MemoryBuffer( size_t aSize, void *aDevicePointer )
            : Internal::sGPUDevicePointerView( aSize, aDevicePointer )
        {
        }

        MemoryBuffer( size_t aSize, size_t aOffset, Internal::sGPUDevicePointerView const &aDevicePointer )
            : Internal::sGPUDevicePointerView( aSize, aOffset, aDevicePointer )
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
    class MemoryPool : public Internal::sGPUDevicePointer
    {
      public:
        MemoryPool()  = default;
        ~MemoryPool() = default;

        /// @brief Allocates `aTotalSize` bytes of memory on the GPU
        ///
        /// @param aTotalSize Size, in bytes, of the memory to allocate.
        ///
        MemoryPool( size_t aTotalSize );

        /// @brief Allocates buffer from the pool.
        ///
        /// @exception  std::runtime_error If trying to allocate more memory than is available in the pool
        ///
        /// @param aBytes Size, in bytes, of the buffer to allocate.
        ///
        MemoryBuffer Allocate( size_t aBytes );

        /// @brief Resets the pool.
        void Reset();

      protected:
        size_t mFreePtr   = 0; //!< Pointer to the free area of the memory pool
        size_t mTotalSize = 0; //!< Total size of the memory pool, in bytes.
    };

} // namespace SE::Cuda