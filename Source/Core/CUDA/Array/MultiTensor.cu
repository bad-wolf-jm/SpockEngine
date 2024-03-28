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
    multi_tensor_t::multi_tensor_t( memory_pool_t &memoryPool, const tensor_shape_t &shape )
        : _shape{ shape }
    {
        _memoryBuffer                         = memoryPool.Allocate( _shape.mByteSize );
        _shape.mDeviceSideData.mShape         = memoryPool.Allocate( _shape.mLayerCount * _shape.mRank * sizeof( uint32_t ) );
        _shape.mDeviceSideData.mMaxDimensions = memoryPool.Allocate( _shape.mRank * sizeof( uint32_t ) );
        _shape.mDeviceSideData.mBufferSizes   = memoryPool.Allocate( _shape.mLayerCount * sizeof( buffer_size_info_t ) );
        _shape.SyncDeviceData();
    }

    multi_tensor_t::multi_tensor_t( memory_pool_t &memoryPool, memory_buffer_t &aMemoryBuffer, const tensor_shape_t &shape )
        : _shape{ shape }
    {
        _memoryBuffer                         = aMemoryBuffer;
        _shape.mDeviceSideData.mShape         = memoryPool.Allocate( _shape.mLayerCount * _shape.mRank * sizeof( uint32_t ) );
        _shape.mDeviceSideData.mMaxDimensions = memoryPool.Allocate( _shape.mRank * sizeof( uint32_t ) );
        _shape.mDeviceSideData.mBufferSizes   = memoryPool.Allocate( _shape.mLayerCount * sizeof( buffer_size_info_t ) );
        _shape.SyncDeviceData();
    }
} // namespace SE::Cuda