/// @file   MultiTensor.cu
///
/// @brief  MultiTensor class implementation
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert. All rights reserved.

#include "MultiTensor.h"
#include <stdexcept>

namespace SE::Cuda
{
    multi_tensor_t::multi_tensor_t( memory_pool_t &aMemoryPool, const tensor_shape_t &aShape )
        : mShape{ aShape }
    {
        mMemoryBuffer                         = aMemoryPool.Allocate( mShape.mByteSize );
        mShape.mDeviceSideData.mShape         = aMemoryPool.Allocate( mShape.mLayerCount * mShape.mRank * sizeof( uint32_t ) );
        mShape.mDeviceSideData.mMaxDimensions = aMemoryPool.Allocate( mShape.mRank * sizeof( uint32_t ) );
        mShape.mDeviceSideData.mBufferSizes   = aMemoryPool.Allocate( mShape.mLayerCount * sizeof( buffer_size_info_t ) );
        mShape.SyncDeviceData();
    }

    multi_tensor_t::multi_tensor_t( memory_pool_t &aMemoryPool, memory_buffer_t &aMemoryBuffer, const tensor_shape_t &aShape )
        : mShape{ aShape }
    {
        mMemoryBuffer                         = aMemoryBuffer;
        mShape.mDeviceSideData.mShape         = aMemoryPool.Allocate( mShape.mLayerCount * mShape.mRank * sizeof( uint32_t ) );
        mShape.mDeviceSideData.mMaxDimensions = aMemoryPool.Allocate( mShape.mRank * sizeof( uint32_t ) );
        mShape.mDeviceSideData.mBufferSizes   = aMemoryPool.Allocate( mShape.mLayerCount * sizeof( buffer_size_info_t ) );
        mShape.SyncDeviceData();
    }
} // namespace SE::Cuda