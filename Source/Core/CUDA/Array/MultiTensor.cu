/// @file   MultiTensor.cu
///
/// @brief  MultiTensor class implementation
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert. All rights reserved.

#include "MultiTensor.h"
#include <stdexcept>

namespace numlua::cuda
{
    multi_tensor_t::multi_tensor_t( memory_pool_t &memoryPool, const tensor_shape_t &shape )
        : _shape{ shape }
    {
        _memoryBuffer                         = memoryPool.Allocate( _shape.ByteSize );
        _shape.DeviceSideData.Shape         = memoryPool.Allocate( _shape.LayerCount * _shape.Rank * sizeof( uint32_t ) );
        _shape.DeviceSideData.MaxDimensions = memoryPool.Allocate( _shape.Rank * sizeof( uint32_t ) );
        _shape.DeviceSideData.BufferSizes   = memoryPool.Allocate( _shape.LayerCount * sizeof( buffer_size_info_t ) );
        _shape.SyncDeviceData();
    }

    multi_tensor_t::multi_tensor_t( memory_pool_t &memoryPool, memory_buffer_t &aMemoryBuffer, const tensor_shape_t &shape )
        : _shape{ shape }
    {
        _memoryBuffer                         = aMemoryBuffer;
        _shape.DeviceSideData.Shape         = memoryPool.Allocate( _shape.LayerCount * _shape.Rank * sizeof( uint32_t ) );
        _shape.DeviceSideData.MaxDimensions = memoryPool.Allocate( _shape.Rank * sizeof( uint32_t ) );
        _shape.DeviceSideData.BufferSizes   = memoryPool.Allocate( _shape.LayerCount * sizeof( buffer_size_info_t ) );
        _shape.SyncDeviceData();
    }
} // namespace SE::Cuda