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

    memory_pool_t::memory_pool_t( size_t totalSize )
        : Internal::gpu_device_pointer_t( totalSize )
        , _totalSize{ totalSize }
    {
        Reset();
    }

    memory_buffer_t memory_pool_t::Allocate( size_t bytes )
    {
        size_t alignedBytes = ( ( bytes >> 3 ) + 1 ) << 3;
        if( ( _freePtr + alignedBytes ) > _totalSize )
            throw std::runtime_error( "MemoryPool is out of space!!" );

        size_t start = _freePtr;
        _freePtr += alignedBytes;
        return memory_buffer_t( bytes, start, *this );
    }

    void memory_pool_t::Reset()
    {
        Zero();
        _freePtr = 0;
    }

} // namespace SE::Cuda