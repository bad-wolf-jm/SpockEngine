/// @file   MemoryPool.cu
///
/// @brief  Memory pool implementation
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#include "MemoryPool.h"
#include <exception>
#include <stdexcept>

#include "Core/Logging.h"

namespace SE::Cuda
{

    memory_pool_t::memory_pool_t( size_t aTotalSize )
        : Internal::gpu_device_pointer_t( aTotalSize )
        , mTotalSize{ aTotalSize }
    {
        Reset();
    }

    memory_buffer_t memory_pool_t::Allocate( size_t aBytes )
    {
        size_t lAlignedBytes = ( ( aBytes >> 3 ) + 1 ) << 3;
        if( ( mFreePtr + lAlignedBytes ) > mTotalSize ) throw std::runtime_error( "MemoryPool is out of space!!" );

        size_t lStart = mFreePtr;
        mFreePtr += lAlignedBytes;
        return memory_buffer_t( aBytes, lStart, *this );
    }

    void memory_pool_t::Reset()
    {
        Zero();
        mFreePtr = 0;
    }

} // namespace SE::Cuda