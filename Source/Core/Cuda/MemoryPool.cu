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

namespace LTSE::Cuda
{

    MemoryPool::MemoryPool( size_t aTotalSize )
        : Internal::sGPUDevicePointer( aTotalSize )
        , mTotalSize{ aTotalSize }
    {
        Reset();
    }

    MemoryBuffer MemoryPool::Allocate( size_t aBytes )
    {
        size_t lAlignedBytes = ( ( aBytes >> 3 ) + 1 ) << 3;
        if( ( mFreePtr + lAlignedBytes ) > mTotalSize ) throw std::runtime_error( "MemoryPool is out of space!!" );

        size_t lStart = mFreePtr;
        mFreePtr += lAlignedBytes;
        return MemoryBuffer( aBytes, lStart, *this );
    }

    void MemoryPool::Reset()
    {
        Zero();
        mFreePtr = 0;
    }

} // namespace LTSE::Cuda