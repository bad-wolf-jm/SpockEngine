/// @file   DeviceKernels.h
///
/// @brief  Template CUDA kernel definitions
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert. All rights reserved.

#pragma once

#include <chrono>

#include <cuda.h>
#include <curand.h>
#include <stdexcept>
#include <variant>

#include "Core/Logging.h"
#include "Core/Math/Types.h"

#include "Core/CUDA/Array/MemoryPool.h"
#include "Core/CUDA/Array/MultiTensor.h"
#include "Core/CUDA/CudaAssert.h"
#include "Core/CUDA/Texture/Texture2D.h"

#include "HelperMacros.h"

namespace SE::TensorOps::Kernels
{
    using namespace SE::Cuda;

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void ConstantFill( multi_tensor_t array, _Ty constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *out = array.DeviceBufferAt<_Ty>( layer );
        out[i]   = constant;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void ConstantFill( multi_tensor_t array, memory_buffer_t constants )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *out = array.DeviceBufferAt<_Ty>( layer );

        out[i] = constants.DataAs<_Ty>()[blockIdx.x];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void ARange( multi_tensor_t out, memory_buffer_t left, memory_buffer_t right, memory_buffer_t aDelta )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *lOutArray = out.DeviceBufferAt<_Ty>( layer );

        lOutArray[i] = left.DataAs<_Ty>()[blockIdx.x] + i * aDelta.DataAs<_Ty>()[blockIdx.x];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Add( multi_tensor_t out, multi_tensor_t left, multi_tensor_t right )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( left.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *lArray    = left.DeviceBufferAt<_Ty>( layer );
        _Ty *lConstant = right.DeviceBufferAt<_Ty>( layer );
        _Ty *lOut      = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = lArray[i] + lConstant[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Add( multi_tensor_t out, multi_tensor_t left, multi_tensor_t right, broadcast_hint_t broadcastHint,
                                     memory_buffer_t blockSizes, memory_buffer_t broadcastSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < blockSizes.DataAs<uint32_t>()[layer] );

        auto lBroadcastSize = broadcastSizes.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( i < lBroadcastSize ) );

        auto *lOut = out.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * lBroadcastSize;

        switch( broadcastHint )
        {
        case broadcast_hint_t::LEFT:
        {
            _Ty *lLeft  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;
            _Ty *lRight = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * lBroadcastSize;

            lOut[i] = lLeft[0] + lRight[i];
        }
        break;
        case broadcast_hint_t::RIGHT:
        {
            _Ty *lLeft  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * lBroadcastSize;
            _Ty *lRight = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;

            lOut[i] = lLeft[i] + lRight[0];
        }
        break;
        default:
            break;
        }
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Add( multi_tensor_t out, multi_tensor_t array, _Ty constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *lArray = array.DeviceBufferAt<_Ty>( layer );
        _Ty *lOut   = out.DeviceBufferAt<_Ty>( layer );
        lOut[i]     = lArray[i] + constant;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Add( multi_tensor_t out, multi_tensor_t array, memory_buffer_t constants )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *lIn  = array.DeviceBufferAt<_Ty>( layer );
        _Ty *lOut = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = lIn[i] + constants.DataAs<_Ty>()[blockIdx.x];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Multiply( multi_tensor_t out, multi_tensor_t array, multi_tensor_t constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *lArray    = array.DeviceBufferAt<_Ty>( layer );
        _Ty *lConstant = constant.DeviceBufferAt<_Ty>( layer );
        _Ty *lOut      = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = lArray[i] * lConstant[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Multiply( multi_tensor_t out, multi_tensor_t left, multi_tensor_t right,
                                          broadcast_hint_t broadcastHint, memory_buffer_t blockSizes, memory_buffer_t broadcastSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < blockSizes.DataAs<uint32_t>()[layer] );

        auto lBroadcastSize = broadcastSizes.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( i < lBroadcastSize ) );

        auto *lOut = out.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * lBroadcastSize;

        switch( broadcastHint )
        {
        case broadcast_hint_t::LEFT:
        {
            _Ty *lLeft  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;
            _Ty *lRight = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * lBroadcastSize;

            lOut[i] = lLeft[0] * lRight[i];
        }
        break;
        case broadcast_hint_t::RIGHT:
        {
            _Ty *lLeft  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * lBroadcastSize;
            _Ty *lRight = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;

            lOut[i] = lLeft[i] * lRight[0];
        }
        break;
        default:
            break;
        }
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Multiply( multi_tensor_t out, multi_tensor_t array, _Ty constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *lArray = array.DeviceBufferAt<_Ty>( layer );
        _Ty *lOut   = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = lArray[i] * constant;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Multiply( multi_tensor_t out, multi_tensor_t array, memory_buffer_t constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *lIn  = array.DeviceBufferAt<_Ty>( layer );
        _Ty *lOut = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = lIn[i] * constant.DataAs<_Ty>()[blockIdx.x];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Subtract( multi_tensor_t out, multi_tensor_t array, _Ty constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *lArray = array.DeviceBufferAt<_Ty>( layer );
        _Ty *lOut   = out.DeviceBufferAt<_Ty>( layer );
        lOut[i]     = lArray[i] - constant;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Subtract( multi_tensor_t out, _Ty constant, multi_tensor_t array )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *lArray = array.DeviceBufferAt<_Ty>( layer );
        _Ty *lOut   = out.DeviceBufferAt<_Ty>( layer );
        lOut[i]     = constant - lArray[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Subtract( multi_tensor_t out, multi_tensor_t array, multi_tensor_t constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *lArray    = array.DeviceBufferAt<_Ty>( layer );
        _Ty *lConstant = constant.DeviceBufferAt<_Ty>( layer );
        _Ty *lOut      = out.DeviceBufferAt<_Ty>( layer );
        lOut[i]        = lArray[i] - lConstant[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Subtract( multi_tensor_t out, multi_tensor_t left, multi_tensor_t right,
                                          broadcast_hint_t broadcastHint, memory_buffer_t blockSizes, memory_buffer_t broadcastSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < blockSizes.DataAs<uint32_t>()[layer] );

        auto lBroadcastSize = broadcastSizes.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( i < lBroadcastSize ) );

        auto *lOut = out.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * lBroadcastSize;

        switch( broadcastHint )
        {
        case broadcast_hint_t::LEFT:
        {
            _Ty *lLeft  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;
            _Ty *lRight = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * lBroadcastSize;

            lOut[i] = lLeft[0] - lRight[i];
        }
        break;
        case broadcast_hint_t::RIGHT:
        {
            _Ty *lLeft  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * lBroadcastSize;
            _Ty *lRight = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;

            lOut[i] = lLeft[i] - lRight[0];
        }
        break;
        default:
            break;
        }
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Subtract( multi_tensor_t out, multi_tensor_t array, memory_buffer_t constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *lIn  = array.DeviceBufferAt<_Ty>( layer );
        _Ty *lOut = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = lIn[i] - constant.DataAs<_Ty>()[blockIdx.x];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Subtract( multi_tensor_t out, memory_buffer_t constant, multi_tensor_t array )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *lIn  = array.DeviceBufferAt<_Ty>( layer );
        _Ty *lOut = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = constant.DataAs<_Ty>()[blockIdx.x] - lIn[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Divide( multi_tensor_t out, multi_tensor_t array, _Ty constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        auto *lArray = array.DeviceBufferAt<_Ty>( layer );
        auto *lOut   = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = lArray[i] / constant;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Divide( multi_tensor_t out, _Ty constant, multi_tensor_t array )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<_Ty>( layer, i ) );

        auto *lArray = array.DeviceBufferAt<_Ty>( layer );
        auto *lOut   = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = constant / lArray[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Divide( multi_tensor_t out, multi_tensor_t array, multi_tensor_t constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        auto *lArray    = array.DeviceBufferAt<_Ty>( layer );
        auto *lConstant = constant.DeviceBufferAt<_Ty>( layer );
        auto *lOut      = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = lArray[i] / lConstant[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Divide( multi_tensor_t out, multi_tensor_t left, multi_tensor_t right, broadcast_hint_t broadcastHint,
                                        memory_buffer_t blockSizes, memory_buffer_t broadcastSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < blockSizes.DataAs<uint32_t>()[layer] );

        auto lBroadcastSize = broadcastSizes.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( i < lBroadcastSize ) );

        auto *lOut = out.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * lBroadcastSize;

        switch( broadcastHint )
        {
        case broadcast_hint_t::LEFT:
        {
            _Ty *lLeft  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;
            _Ty *lRight = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * lBroadcastSize;

            lOut[i] = lLeft[0] / lRight[i];
        }
        break;
        case broadcast_hint_t::RIGHT:
        {
            _Ty *lLeft  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * lBroadcastSize;
            _Ty *lRight = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;

            lOut[i] = lLeft[i] / lRight[0];
        }
        break;
        default:
            break;
        }
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Divide( multi_tensor_t out, multi_tensor_t array, memory_buffer_t constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        auto *lIn  = array.DeviceBufferAt<_Ty>( layer );
        auto *lOut = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = lIn[i] / constant.DataAs<_Ty>()[blockIdx.x];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Divide( multi_tensor_t out, memory_buffer_t constant, multi_tensor_t array )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        auto *lIn  = array.DeviceBufferAt<_Ty>( layer );
        auto *lOut = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = constant.DataAs<_Ty>()[blockIdx.x] / lIn[i];
    }

    CUDA_KERNEL_DEFINITION void And( multi_tensor_t out, multi_tensor_t array, uint8_t constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<uint8_t>( layer, i ) );

        uint8_t *lArray = array.DeviceBufferAt<uint8_t>( layer );
        uint8_t *lOut   = out.DeviceBufferAt<uint8_t>( layer );

        lOut[i] = ( constant && lArray[i] );
    }

    CUDA_KERNEL_DEFINITION void And( multi_tensor_t out, multi_tensor_t array, multi_tensor_t constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<uint8_t>( layer, i ) );

        uint8_t *lArray    = array.DeviceBufferAt<uint8_t>( layer );
        uint8_t *lConstant = constant.DeviceBufferAt<uint8_t>( layer );
        uint8_t *lOut      = out.DeviceBufferAt<uint8_t>( layer );

        lOut[i] = ( lArray[i] && lConstant[i] );
    }

    CUDA_KERNEL_DEFINITION void And( multi_tensor_t out, multi_tensor_t left, multi_tensor_t right, broadcast_hint_t broadcastHint,
                                     memory_buffer_t blockSizes, memory_buffer_t broadcastSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < blockSizes.DataAs<uint32_t>()[layer] );

        auto lBroadcastSize = broadcastSizes.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( i < lBroadcastSize ) );

        uint8_t *lOut = out.DeviceBufferAt<uint8_t>( layer ) + blockIdx.y * lBroadcastSize;

        switch( broadcastHint )
        {
        case broadcast_hint_t::LEFT:
        {
            uint8_t *lLeft  = left.DeviceBufferAt<uint8_t>( layer ) + blockIdx.y;
            uint8_t *lRight = right.DeviceBufferAt<uint8_t>( layer ) + blockIdx.y * lBroadcastSize;

            lOut[i] = ( lLeft[0] && lRight[i] );
        }
        break;
        case broadcast_hint_t::RIGHT:
        {
            uint8_t *lLeft  = left.DeviceBufferAt<uint8_t>( layer ) + blockIdx.y * lBroadcastSize;
            uint8_t *lRight = right.DeviceBufferAt<uint8_t>( layer ) + blockIdx.y;

            lOut[i] = ( lLeft[i] && lRight[0] );
        }
        break;
        default:
            break;
        }
    }

    CUDA_KERNEL_DEFINITION void And( multi_tensor_t out, multi_tensor_t array, memory_buffer_t constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<uint8_t>( layer, i ) );

        uint8_t *lIn  = array.DeviceBufferAt<uint8_t>( layer );
        uint8_t *lOut = out.DeviceBufferAt<uint8_t>( layer );

        lOut[i] = ( lIn[i] && constant.DataAs<uint8_t>()[blockIdx.x] );
    }

    CUDA_KERNEL_DEFINITION void Or( multi_tensor_t out, multi_tensor_t array, uint8_t constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<uint8_t>( layer, i ) );

        uint8_t *lArray = array.DeviceBufferAt<uint8_t>( layer );
        uint8_t *lOut   = out.DeviceBufferAt<uint8_t>( layer );

        lOut[i] = ( constant || lArray[i] );
    }

    CUDA_KERNEL_DEFINITION void Or( multi_tensor_t out, multi_tensor_t array, multi_tensor_t constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<uint8_t>( layer, i ) );

        uint8_t *lArray    = array.DeviceBufferAt<uint8_t>( layer );
        uint8_t *lConstant = constant.DeviceBufferAt<uint8_t>( layer );
        uint8_t *lOut      = out.DeviceBufferAt<uint8_t>( layer );

        lOut[i] = ( lArray[i] || lConstant[i] );
    }

    CUDA_KERNEL_DEFINITION void Or( multi_tensor_t out, multi_tensor_t left, multi_tensor_t right, broadcast_hint_t broadcastHint,
                                    memory_buffer_t blockSizes, memory_buffer_t broadcastSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < blockSizes.DataAs<uint32_t>()[layer] );

        auto lBroadcastSize = broadcastSizes.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( i < lBroadcastSize ) );

        uint8_t *lOut = out.DeviceBufferAt<uint8_t>( layer ) + blockIdx.y * lBroadcastSize;

        switch( broadcastHint )
        {
        case broadcast_hint_t::LEFT:
        {
            uint8_t *lLeft  = left.DeviceBufferAt<uint8_t>( layer ) + blockIdx.y;
            uint8_t *lRight = right.DeviceBufferAt<uint8_t>( layer ) + blockIdx.y * lBroadcastSize;

            lOut[i] = ( lLeft[0] || lRight[i] );
        }
        break;
        case broadcast_hint_t::RIGHT:
        {
            uint8_t *lLeft  = left.DeviceBufferAt<uint8_t>( layer ) + blockIdx.y * lBroadcastSize;
            uint8_t *lRight = right.DeviceBufferAt<uint8_t>( layer ) + blockIdx.y;

            lOut[i] = ( lLeft[i] || lRight[0] );
        }
        break;
        default:
            break;
        }
    }

    CUDA_KERNEL_DEFINITION void Or( multi_tensor_t out, multi_tensor_t array, memory_buffer_t constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<uint8_t>( layer, i ) );

        uint8_t *lIn  = array.DeviceBufferAt<uint8_t>( layer );
        uint8_t *lOut = out.DeviceBufferAt<uint8_t>( layer );

        lOut[i] = ( lIn[i] || constant.DataAs<uint8_t>()[blockIdx.x] );
    }

    CUDA_KERNEL_DEFINITION void Not( multi_tensor_t out, multi_tensor_t array )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<uint8_t>( layer, i ) );

        uint8_t *lIn  = array.DeviceBufferAt<uint8_t>( layer );
        uint8_t *lOut = out.DeviceBufferAt<uint8_t>( layer );

        lOut[i] = !( lIn[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void BitwiseAnd( multi_tensor_t out, multi_tensor_t array, _Ty constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *lArray = array.DeviceBufferAt<_Ty>( layer );
        _Ty *lOut   = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = ( constant & lArray[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void BitwiseAnd( multi_tensor_t out, multi_tensor_t left, multi_tensor_t right,
                                            broadcast_hint_t broadcastHint, memory_buffer_t blockSizes,
                                            memory_buffer_t broadcastSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < blockSizes.DataAs<uint32_t>()[layer] );

        auto lBroadcastSize = broadcastSizes.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( i < lBroadcastSize ) );

        auto *lOut = out.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * lBroadcastSize;

        switch( broadcastHint )
        {
        case broadcast_hint_t::LEFT:
        {
            _Ty *lLeft  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;
            _Ty *lRight = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * lBroadcastSize;

            lOut[i] = lLeft[0] & lRight[i];
        }
        break;
        case broadcast_hint_t::RIGHT:
        {
            _Ty *lLeft  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * lBroadcastSize;
            _Ty *lRight = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;

            lOut[i] = lLeft[i] & lRight[0];
        }
        break;
        default:
            break;
        }
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void BitwiseAnd( multi_tensor_t out, multi_tensor_t array, multi_tensor_t constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *lArray    = array.DeviceBufferAt<_Ty>( layer );
        _Ty *lConstant = constant.DeviceBufferAt<_Ty>( layer );
        _Ty *lOut      = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = ( lArray[i] & lConstant[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void BitwiseAnd( multi_tensor_t out, multi_tensor_t array, memory_buffer_t constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *lIn  = array.DeviceBufferAt<_Ty>( layer );
        _Ty *lOut = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = ( lIn[i] & constant.DataAs<_Ty>()[blockIdx.x] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void BitwiseOr( multi_tensor_t out, multi_tensor_t array, _Ty constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *lArray = array.DeviceBufferAt<_Ty>( layer );
        _Ty *lOut   = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = ( constant | lArray[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void BitwiseOr( multi_tensor_t out, multi_tensor_t array, multi_tensor_t constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *lArray    = array.DeviceBufferAt<_Ty>( layer );
        _Ty *lConstant = constant.DeviceBufferAt<_Ty>( layer );
        _Ty *lOut      = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = ( lArray[i] | lConstant[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void BitwiseOr( multi_tensor_t out, multi_tensor_t left, multi_tensor_t right,
                                           broadcast_hint_t broadcastHint, memory_buffer_t blockSizes, memory_buffer_t broadcastSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < blockSizes.DataAs<uint32_t>()[layer] );

        auto lBroadcastSize = broadcastSizes.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( i < lBroadcastSize ) );

        auto *lOut = out.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * lBroadcastSize;

        switch( broadcastHint )
        {
        case broadcast_hint_t::LEFT:
        {
            _Ty *lLeft  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;
            _Ty *lRight = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * lBroadcastSize;

            lOut[i] = lLeft[0] | lRight[i];
        }
        break;
        case broadcast_hint_t::RIGHT:
        {
            _Ty *lLeft  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * lBroadcastSize;
            _Ty *lRight = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;

            lOut[i] = lLeft[i] | lRight[0];
        }
        break;
        default:
            break;
        }
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void BitwiseOr( multi_tensor_t out, multi_tensor_t array, memory_buffer_t constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *lIn  = array.DeviceBufferAt<_Ty>( layer );
        _Ty *lOut = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = ( lIn[i] | constant.DataAs<_Ty>()[blockIdx.x] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Bitwise( multi_tensor_t out, multi_tensor_t array )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *lIn  = array.DeviceBufferAt<_Ty>( layer );
        _Ty *lOut = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = ~( lIn[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void InInterval( multi_tensor_t out, multi_tensor_t x, multi_tensor_t lower, multi_tensor_t upper,
                                            bool strictLower, bool strictUpper )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<_Ty>( layer, i ) );

        _Ty     *lX     = x.DeviceBufferAt<_Ty>( layer );
        _Ty     *lLower = lower.DeviceBufferAt<_Ty>( layer );
        _Ty     *lUpper = upper.DeviceBufferAt<_Ty>( layer );
        uint8_t *lOut   = out.DeviceBufferAt<uint8_t>( layer );

        bool lComp0 = strictLower ? ( lLower[i] < lX[i] ) : ( lLower[i] <= lX[i] );
        bool lComp1 = strictUpper ? ( lUpper[i] > lX[i] ) : ( lUpper[i] >= lX[i] );

        lOut[i] = ( lComp0 && lComp1 );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void InInterval( multi_tensor_t out, multi_tensor_t x, multi_tensor_t lower, memory_buffer_t upper,
                                            bool strictLower, bool strictUpper )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<_Ty>( layer, i ) );

        _Ty     *lX     = x.DeviceBufferAt<_Ty>( layer );
        _Ty     *lLower = lower.DeviceBufferAt<_Ty>( layer );
        _Ty     *lUpper = upper.DataAs<_Ty>();
        uint8_t *lOut   = out.DeviceBufferAt<uint8_t>( layer );

        bool lComp0 = strictLower ? ( lLower[i] < lX[i] ) : ( lLower[i] <= lX[i] );
        bool lComp1 = strictUpper ? ( lUpper[layer] > lX[i] ) : ( lUpper[layer] >= lX[i] );

        lOut[i] = ( lComp0 && lComp1 );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void InInterval( multi_tensor_t out, multi_tensor_t x, multi_tensor_t lower, _Ty upper, bool strictLower,
                                            bool strictUpper )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<_Ty>( layer, i ) );

        _Ty     *lX     = x.DeviceBufferAt<_Ty>( layer );
        _Ty     *lLower = lower.DeviceBufferAt<_Ty>( layer );
        uint8_t *lOut   = out.DeviceBufferAt<uint8_t>( layer );

        bool lComp0 = strictLower ? ( lLower[i] < lX[i] ) : ( lLower[i] <= lX[i] );
        bool lComp1 = strictUpper ? ( upper > lX[i] ) : ( upper >= lX[i] );

        lOut[i] = ( lComp0 && lComp1 );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void InInterval( multi_tensor_t out, multi_tensor_t x, memory_buffer_t lower, multi_tensor_t upper,
                                            bool strictLower, bool strictUpper )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<_Ty>( layer, i ) );

        _Ty     *lX     = x.DeviceBufferAt<_Ty>( layer );
        _Ty     *lLower = lower.DataAs<_Ty>();
        _Ty     *lUpper = upper.DeviceBufferAt<_Ty>( layer );
        uint8_t *lOut   = out.DeviceBufferAt<uint8_t>( layer );

        bool lComp0 = strictLower ? ( lLower[layer] < lX[i] ) : ( lLower[layer] <= lX[i] );
        bool lComp1 = strictUpper ? ( lUpper[i] > lX[i] ) : ( lUpper[i] >= lX[i] );

        lOut[i] = ( lComp0 && lComp1 );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void InInterval( multi_tensor_t out, multi_tensor_t x, memory_buffer_t lower, memory_buffer_t upper,
                                            bool strictLower, bool strictUpper )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<_Ty>( layer, i ) );

        _Ty     *lX     = x.DeviceBufferAt<_Ty>( layer );
        _Ty     *lLower = lower.DataAs<_Ty>();
        _Ty     *lUpper = upper.DataAs<_Ty>();
        uint8_t *lOut   = out.DeviceBufferAt<uint8_t>( layer );

        bool lComp0 = strictLower ? ( lLower[layer] < lX[i] ) : ( lLower[layer] <= lX[i] );
        bool lComp1 = strictUpper ? ( lUpper[layer] > lX[i] ) : ( lUpper[layer] >= lX[i] );

        lOut[i] = ( lComp0 && lComp1 );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void InInterval( multi_tensor_t out, multi_tensor_t x, memory_buffer_t lower, _Ty upper, bool strictLower,
                                            bool strictUpper )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<_Ty>( layer, i ) );

        _Ty     *lX     = x.DeviceBufferAt<_Ty>( layer );
        _Ty     *lLower = lower.DataAs<_Ty>();
        uint8_t *lOut   = out.DeviceBufferAt<uint8_t>( layer );

        bool lComp0 = strictLower ? ( lLower[layer] < lX[i] ) : ( lLower[layer] <= lX[i] );
        bool lComp1 = strictUpper ? ( upper > lX[i] ) : ( upper >= lX[i] );

        lOut[i] = ( lComp0 && lComp1 );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void InInterval( multi_tensor_t out, multi_tensor_t x, _Ty lower, multi_tensor_t upper, bool strictLower,
                                            bool strictUpper )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<_Ty>( layer, i ) );

        _Ty     *lX     = x.DeviceBufferAt<_Ty>( layer );
        _Ty     *lUpper = upper.DeviceBufferAt<_Ty>( layer );
        uint8_t *lOut   = out.DeviceBufferAt<uint8_t>( layer );

        bool lComp0 = strictLower ? ( lower < lX[i] ) : ( lower <= lX[i] );
        bool lComp1 = strictUpper ? ( lUpper[i] > lX[i] ) : ( lUpper[i] >= lX[i] );

        lOut[i] = ( lComp0 && lComp1 );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void InInterval( multi_tensor_t out, multi_tensor_t x, _Ty lower, memory_buffer_t upper, bool strictLower,
                                            bool strictUpper )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<_Ty>( layer, i ) );

        _Ty     *lX     = x.DeviceBufferAt<_Ty>( layer );
        _Ty     *lUpper = upper.DataAs<_Ty>();
        uint8_t *lOut   = out.DeviceBufferAt<uint8_t>( layer );

        bool lComp0 = strictLower ? ( lower < lX[i] ) : ( lower <= lX[i] );
        bool lComp1 = strictUpper ? ( lUpper[layer] > lX[i] ) : ( lUpper[layer] >= lX[i] );

        lOut[i] = ( lComp0 && lComp1 );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void InInterval( multi_tensor_t out, multi_tensor_t x, _Ty lower, _Ty upper, bool strictLower,
                                            bool strictUpper )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<_Ty>( layer, i ) );

        _Ty     *lX   = x.DeviceBufferAt<_Ty>( layer );
        uint8_t *lOut = out.DeviceBufferAt<uint8_t>( layer );

        bool lComp0 = strictLower ? ( lower < lX[i] ) : ( lower <= lX[i] );
        bool lComp1 = strictUpper ? ( upper > lX[i] ) : ( upper >= lX[i] );

        lOut[i] = ( lComp0 && lComp1 );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void EqualOp( multi_tensor_t out, multi_tensor_t x, multi_tensor_t y )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<uint8_t>( layer, i ) );

        _Ty     *lX   = x.DeviceBufferAt<_Ty>( layer );
        _Ty     *lY   = y.DeviceBufferAt<_Ty>( layer );
        uint8_t *lOut = out.DeviceBufferAt<uint8_t>( layer );

        lOut[i] = ( lX[i] == lY[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void EqualOp( multi_tensor_t out, multi_tensor_t x, memory_buffer_t y )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<uint8_t>( layer, i ) );

        _Ty     *lX   = x.DeviceBufferAt<_Ty>( layer );
        _Ty     *lY   = y.DataAs<_Ty>();
        uint8_t *lOut = out.DeviceBufferAt<uint8_t>( layer );

        lOut[i] = ( lX[i] == lY[layer] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void EqualOp( multi_tensor_t out, multi_tensor_t x, _Ty y )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<uint8_t>( layer, i ) );

        _Ty     *lX   = x.DeviceBufferAt<_Ty>( layer );
        uint8_t *lOut = out.DeviceBufferAt<uint8_t>( layer );

        lOut[i] = ( lX[i] == y );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void EqualOp( multi_tensor_t out, multi_tensor_t left, multi_tensor_t right, broadcast_hint_t broadcastHint,
                                         memory_buffer_t blockSizes, memory_buffer_t broadcastSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < blockSizes.DataAs<uint32_t>()[layer] );

        auto lBroadcastSize = broadcastSizes.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( i < lBroadcastSize ) );

        auto *lOut = out.DeviceBufferAt<uint8_t>( layer ) + blockIdx.y * lBroadcastSize;

        switch( broadcastHint )
        {
        case broadcast_hint_t::LEFT:
        {
            _Ty *lLeft  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;
            _Ty *lRight = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * lBroadcastSize;

            lOut[i] = lLeft[0] == lRight[i];
        }
        break;
        case broadcast_hint_t::RIGHT:
        {
            _Ty *lLeft  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * lBroadcastSize;
            _Ty *lRight = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;

            lOut[i] = lLeft[i] == lRight[0];
        }
        break;
        default:
            break;
        }
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void EqualOp( multi_tensor_t out, memory_buffer_t x, multi_tensor_t y )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<uint8_t>( layer, i ) );

        _Ty     *lX   = x.DataAs<_Ty>();
        _Ty     *lY   = y.DeviceBufferAt<_Ty>( layer );
        uint8_t *lOut = out.DeviceBufferAt<uint8_t>( layer );

        lOut[i] = ( lX[layer] == lY[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void EqualOp( multi_tensor_t out, _Ty x, multi_tensor_t y )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<uint8_t>( layer, i ) );

        _Ty     *lY   = y.DeviceBufferAt<_Ty>( layer );
        uint8_t *lOut = out.DeviceBufferAt<uint8_t>( layer );

        lOut[i] = ( x == lY[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOp( multi_tensor_t out, multi_tensor_t x, multi_tensor_t y )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<uint8_t>( layer, i ) );

        _Ty     *lX   = x.DeviceBufferAt<_Ty>( layer );
        _Ty     *lY   = y.DeviceBufferAt<_Ty>( layer );
        uint8_t *lOut = out.DeviceBufferAt<uint8_t>( layer );

        lOut[i] = ( lX[i] < lY[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOp( multi_tensor_t out, multi_tensor_t left, multi_tensor_t right,
                                            broadcast_hint_t broadcastHint, memory_buffer_t blockSizes,
                                            memory_buffer_t broadcastSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < blockSizes.DataAs<uint32_t>()[layer] );

        auto lBroadcastSize = broadcastSizes.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( i < lBroadcastSize ) );

        auto *lOut = out.DeviceBufferAt<uint8_t>( layer ) + blockIdx.y * lBroadcastSize;

        switch( broadcastHint )
        {
        case broadcast_hint_t::LEFT:
        {
            _Ty *lLeft  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;
            _Ty *lRight = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * lBroadcastSize;

            lOut[i] = lLeft[0] < lRight[i];
        }
        break;
        case broadcast_hint_t::RIGHT:
        {
            _Ty *lLeft  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * lBroadcastSize;
            _Ty *lRight = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;

            lOut[i] = lLeft[i] < lRight[0];
        }
        break;
        default:
            break;
        }
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOp( multi_tensor_t out, multi_tensor_t x, memory_buffer_t y )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<uint8_t>( layer, i ) );

        _Ty     *lX   = x.DeviceBufferAt<_Ty>( layer );
        _Ty     *lY   = y.DataAs<_Ty>();
        uint8_t *lOut = out.DeviceBufferAt<uint8_t>( layer );

        lOut[i] = ( lX[i] < lY[layer] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOp( multi_tensor_t out, multi_tensor_t x, _Ty y )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<uint8_t>( layer, i ) );

        _Ty     *lX   = x.DeviceBufferAt<_Ty>( layer );
        uint8_t *lOut = out.DeviceBufferAt<uint8_t>( layer );

        lOut[i] = ( lX[i] < y );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOp( multi_tensor_t out, memory_buffer_t x, multi_tensor_t y )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<uint8_t>( layer, i ) );

        _Ty     *lX   = x.DataAs<_Ty>();
        _Ty     *lY   = y.DeviceBufferAt<_Ty>( layer );
        uint8_t *lOut = out.DeviceBufferAt<uint8_t>( layer );

        lOut[i] = ( lX[layer] < lY[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOp( multi_tensor_t out, _Ty x, multi_tensor_t y )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<uint8_t>( layer, i ) );

        _Ty     *lY   = y.DeviceBufferAt<_Ty>( layer );
        uint8_t *lOut = out.DeviceBufferAt<uint8_t>( layer );

        lOut[i] = ( x < lY[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOrEqualOp( multi_tensor_t out, multi_tensor_t x, multi_tensor_t y )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<uint8_t>( layer, i ) );

        _Ty     *lX   = x.DeviceBufferAt<_Ty>( layer );
        _Ty     *lY   = y.DeviceBufferAt<_Ty>( layer );
        uint8_t *lOut = out.DeviceBufferAt<uint8_t>( layer );

        lOut[i] = ( lX[i] <= lY[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOrEqualOp( multi_tensor_t out, multi_tensor_t left, multi_tensor_t right,
                                                   broadcast_hint_t broadcastHint, memory_buffer_t blockSizes,
                                                   memory_buffer_t broadcastSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < blockSizes.DataAs<uint32_t>()[layer] );

        auto lBroadcastSize = broadcastSizes.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( i < lBroadcastSize ) );

        auto *lOut = out.DeviceBufferAt<uint8_t>( layer ) + blockIdx.y * lBroadcastSize;

        switch( broadcastHint )
        {
        case broadcast_hint_t::LEFT:
        {
            _Ty *lLeft  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;
            _Ty *lRight = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * lBroadcastSize;

            lOut[i] = lLeft[0] <= lRight[i];
        }
        break;
        case broadcast_hint_t::RIGHT:
        {
            _Ty *lLeft  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * lBroadcastSize;
            _Ty *lRight = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;

            lOut[i] = lLeft[i] <= lRight[0];
        }
        break;
        default:
            break;
        }
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOrEqualOp( multi_tensor_t out, multi_tensor_t x, memory_buffer_t y )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<uint8_t>( layer, i ) );

        _Ty     *lX   = x.DeviceBufferAt<_Ty>( layer );
        _Ty     *lY   = y.DataAs<_Ty>();
        uint8_t *lOut = out.DeviceBufferAt<uint8_t>( layer );

        lOut[i] = ( lX[i] <= lY[layer] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOrEqualOp( multi_tensor_t out, multi_tensor_t x, _Ty y )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<uint8_t>( layer, i ) );

        _Ty     *lX   = x.DeviceBufferAt<_Ty>( layer );
        uint8_t *lOut = out.DeviceBufferAt<uint8_t>( layer );

        lOut[i] = ( lX[i] <= y );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOrEqualOp( multi_tensor_t out, memory_buffer_t x, multi_tensor_t y )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<uint8_t>( layer, i ) );

        _Ty     *lX   = x.DataAs<_Ty>();
        _Ty     *lY   = y.DeviceBufferAt<_Ty>( layer );
        uint8_t *lOut = out.DeviceBufferAt<uint8_t>( layer );

        lOut[i] = ( lX[layer] <= lY[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOrEqualOp( multi_tensor_t out, _Ty x, multi_tensor_t y )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<uint8_t>( layer, i ) );

        _Ty     *lY   = y.DeviceBufferAt<_Ty>( layer );
        uint8_t *lOut = out.DeviceBufferAt<uint8_t>( layer );

        lOut[i] = ( x <= lY[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void WhereTensorTensor( multi_tensor_t out, multi_tensor_t condition, multi_tensor_t valueIfTrue,
                                                   multi_tensor_t valueIfFalse )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( condition.Shape().InBounds<uint8_t>( layer, i ) );

        uint8_t *lCondition    = condition.DeviceBufferAt<uint8_t>( layer );
        _Ty     *lValueIfTrue  = valueIfTrue.DeviceBufferAt<_Ty>( layer );
        _Ty     *lValueIfFalse = valueIfFalse.DeviceBufferAt<_Ty>( layer );
        _Ty     *lOut          = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = lCondition[i] ? lValueIfTrue[i] : lValueIfFalse[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void WhereTensorVector( multi_tensor_t out, multi_tensor_t condition, multi_tensor_t valueIfTrue,
                                                   memory_buffer_t valueIfFalse )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( condition.Shape().InBounds<uint8_t>( layer, i ) );

        uint8_t *lCondition    = condition.DeviceBufferAt<uint8_t>( layer );
        _Ty     *lValueIfTrue  = valueIfTrue.DeviceBufferAt<_Ty>( layer );
        _Ty     *lValueIfFalse = valueIfFalse.DataAs<_Ty>();
        _Ty     *lOut          = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = lCondition[i] ? lValueIfTrue[i] : lValueIfFalse[layer];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void WhereTensorScalar( multi_tensor_t out, multi_tensor_t condition, multi_tensor_t valueIfTrue,
                                                   _Ty valueIfFalse )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( condition.Shape().InBounds<uint8_t>( layer, i ) );

        uint8_t *lCondition   = condition.DeviceBufferAt<uint8_t>( layer );
        _Ty     *lValueIfTrue = valueIfTrue.DeviceBufferAt<_Ty>( layer );
        _Ty     *lOut         = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = lCondition[i] ? lValueIfTrue[i] : valueIfFalse;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void WhereVectorTensor( multi_tensor_t out, multi_tensor_t condition, memory_buffer_t valueIfTrue,
                                                   multi_tensor_t valueIfFalse )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( condition.Shape().InBounds<uint8_t>( layer, i ) );

        uint8_t *lCondition    = condition.DeviceBufferAt<uint8_t>( layer );
        _Ty     *lValueIfTrue  = valueIfTrue.DataAs<_Ty>();
        _Ty     *lValueIfFalse = valueIfFalse.DeviceBufferAt<_Ty>( layer );
        _Ty     *lOut          = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = lCondition[i] ? lValueIfTrue[layer] : lValueIfFalse[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void WhereVectorVector( multi_tensor_t out, multi_tensor_t condition, memory_buffer_t valueIfTrue,
                                                   memory_buffer_t valueIfFalse )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( condition.Shape().InBounds<uint8_t>( layer, i ) );

        uint8_t *lCondition    = condition.DeviceBufferAt<uint8_t>( layer );
        _Ty     *lValueIfTrue  = valueIfTrue.DataAs<_Ty>();
        _Ty     *lValueIfFalse = valueIfFalse.DataAs<_Ty>();
        _Ty     *lOut          = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = lCondition[i] ? lValueIfTrue[layer] : lValueIfFalse[layer];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void WhereVectorScalar( multi_tensor_t out, multi_tensor_t condition, memory_buffer_t valueIfTrue,
                                                   _Ty valueIfFalse )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( condition.Shape().InBounds<uint8_t>( layer, i ) );

        uint8_t *lCondition   = condition.DeviceBufferAt<uint8_t>( layer );
        _Ty     *lValueIfTrue = valueIfTrue.DataAs<_Ty>();
        _Ty     *lOut         = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = lCondition[i] ? lValueIfTrue[layer] : valueIfFalse;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void WhereScalarTensor( multi_tensor_t out, multi_tensor_t condition, _Ty valueIfTrue,
                                                   multi_tensor_t valueIfFalse )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( condition.Shape().InBounds<uint8_t>( layer, i ) );

        uint8_t *lCondition    = condition.DeviceBufferAt<uint8_t>( layer );
        _Ty     *lValueIfFalse = valueIfFalse.DeviceBufferAt<_Ty>( layer );
        _Ty     *lOut          = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = lCondition[i] ? valueIfTrue : lValueIfFalse[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void WhereScalarVector( multi_tensor_t out, multi_tensor_t condition, _Ty valueIfTrue,
                                                   memory_buffer_t valueIfFalse )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( condition.Shape().InBounds<uint8_t>( layer, i ) );

        uint8_t *lCondition    = condition.DeviceBufferAt<uint8_t>( layer );
        _Ty     *lValueIfFalse = valueIfFalse.DataAs<_Ty>();
        _Ty     *lOut          = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = lCondition[i] ? valueIfTrue : lValueIfFalse[layer];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void WhereScalarScalar( multi_tensor_t out, multi_tensor_t condition, _Ty valueIfTrue, _Ty valueIfFalse )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( condition.Shape().InBounds<uint8_t>( layer, i ) );

        uint8_t *lCondition = condition.DeviceBufferAt<uint8_t>( layer );
        _Ty     *lOut       = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = lCondition[i] ? valueIfTrue : valueIfFalse;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Repeat( multi_tensor_t out, multi_tensor_t array, memory_buffer_t repetitions )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        uint32_t N     = repetitions.DataAs<uint32_t>()[blockIdx.x];

        RETURN_UNLESS( array.Shape().InBounds<uint8_t>( layer, blockIdx.y ) );

        int i = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;
        int j = blockIdx.y;

        RETURN_UNLESS( ( i < N ) && ( array.Shape().InBounds<_Ty>( layer, j ) ) );

        _Ty *lInArray  = array.DeviceBufferAt<_Ty>( layer );
        _Ty *lOutArray = out.DeviceBufferAt<_Ty>( layer );

        lOutArray[blockIdx.y * N + i] = lInArray[j];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Tile( multi_tensor_t out, multi_tensor_t array, memory_buffer_t repetitions )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        uint32_t N     = repetitions.DataAs<uint32_t>()[blockIdx.x];

        RETURN_UNLESS( blockIdx.y < N );

        int i = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;
        int j = blockIdx.y * array.Shape().GetBufferSizeAs<_Ty>( layer ).mSize + i;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) && out.Shape().InBounds<_Ty>( layer, j ) );

        _Ty *lInArray  = array.DeviceBufferAt<_Ty>( layer );
        _Ty *lOutArray = out.DeviceBufferAt<_Ty>( layer );
        lOutArray[j]   = lInArray[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LinearSpace( multi_tensor_t out, multi_tensor_t left, multi_tensor_t right,
                                             memory_buffer_t subdivisions )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        uint32_t N     = subdivisions.DataAs<uint32_t>()[blockIdx.x];

        RETURN_UNLESS( left.Shape().InBounds<_Ty>( layer, blockIdx.y ) );

        int i = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;
        int j = blockIdx.y;
        int k = blockIdx.y * N + i;

        RETURN_UNLESS( i < N );

        _Ty *lInArrayA = left.DeviceBufferAt<_Ty>( layer );
        _Ty *lInArrayB = right.DeviceBufferAt<_Ty>( layer );
        _Ty *lOutArray = out.DeviceBufferAt<_Ty>( layer );

        float aDelta = ( lInArrayB[j] - lInArrayA[j] ) / static_cast<float>( N );
        lOutArray[k] = lInArrayA[j] + i * aDelta;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Mix( multi_tensor_t out, multi_tensor_t A, multi_tensor_t B, multi_tensor_t t )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( A.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *lA   = A.DeviceBufferAt<_Ty>( layer );
        _Ty *lB   = B.DeviceBufferAt<_Ty>( layer );
        _Ty *lT   = t.DeviceBufferAt<_Ty>( layer );
        _Ty *lOut = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = ( 1 - lT[i] ) * lA[i] + lT[i] * lB[i];
    }

    CUDA_KERNEL_DEFINITION void Sample2D( multi_tensor_t out, multi_tensor_t x, multi_tensor_t y, memory_buffer_t textures )
    {
        uint32_t                              layer = static_cast<uint32_t>( blockIdx.x );
        Cuda::texture_sampler2d_t::DeviceData lTex  = textures.DataAs<Cuda::texture_sampler2d_t::DeviceData>()[blockIdx.x];

        int i = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<float>( layer, i ) );

        auto *lXArray = x.DeviceBufferAt<float>( layer );
        auto *lYArray = y.DeviceBufferAt<float>( layer );
        auto *lOut    = out.DeviceBufferAt<float>( layer );

        lOut[i] = lTex.Fetch<float>( lXArray[i], lYArray[i] );
    }

    CUDA_KERNEL_DEFINITION void Sample2D( multi_tensor_t out, multi_tensor_t x, memory_buffer_t y, memory_buffer_t textures )
    {
        uint32_t                              layer = static_cast<uint32_t>( blockIdx.x );
        Cuda::texture_sampler2d_t::DeviceData lTex  = textures.DataAs<Cuda::texture_sampler2d_t::DeviceData>()[blockIdx.x];

        int i = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<float>( layer, i ) );

        auto *lXArray = x.DeviceBufferAt<float>( layer );
        auto *lOut    = out.DeviceBufferAt<float>( layer );

        lOut[i] = lTex.Fetch<float>( lXArray[i], y.DataAs<float>()[layer] );
    }

    CUDA_KERNEL_DEFINITION void Sample2D( multi_tensor_t out, multi_tensor_t x, float y, memory_buffer_t textures )
    {
        uint32_t                              layer = static_cast<uint32_t>( blockIdx.x );
        Cuda::texture_sampler2d_t::DeviceData lTex  = textures.DataAs<Cuda::texture_sampler2d_t::DeviceData>()[blockIdx.x];

        int i = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<float>( layer, i ) );

        auto *lXArray = x.DeviceBufferAt<float>( layer );
        auto *lOut    = out.DeviceBufferAt<float>( layer );

        lOut[i] = lTex.Fetch<float>( lXArray[i], y );
    }

    CUDA_KERNEL_DEFINITION void Sample2D( multi_tensor_t out, memory_buffer_t x, multi_tensor_t y, memory_buffer_t textures )
    {
        uint32_t                              layer = static_cast<uint32_t>( blockIdx.x );
        Cuda::texture_sampler2d_t::DeviceData lTex  = textures.DataAs<Cuda::texture_sampler2d_t::DeviceData>()[blockIdx.x];

        int i = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( y.Shape().InBounds<float>( layer, i ) );

        auto *lYArray = y.DeviceBufferAt<float>( layer );
        auto *lOut    = out.DeviceBufferAt<float>( layer );

        lOut[i] = lTex.Fetch<float>( x.DataAs<float>()[layer], lYArray[i] );
    }

    CUDA_KERNEL_DEFINITION void Sample2D( multi_tensor_t out, float x, multi_tensor_t y, memory_buffer_t textures )
    {
        uint32_t                              layer = static_cast<uint32_t>( blockIdx.x );
        Cuda::texture_sampler2d_t::DeviceData lTex  = textures.DataAs<Cuda::texture_sampler2d_t::DeviceData>()[blockIdx.x];

        int i = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( y.Shape().InBounds<float>( layer, i ) );

        auto *lYArray = y.DeviceBufferAt<float>( layer );
        auto *lOut    = out.DeviceBufferAt<float>( layer );

        lOut[i] = lTex.Fetch<float>( x, lYArray[i] );
    }

    template <typename _Ty, typename _OutTy>
    CUDA_KERNEL_DEFINITION void ToFixedPoint( multi_tensor_t out, multi_tensor_t array, _Ty aScaling )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        _Ty    *lInBuffer  = array.DeviceBufferAt<_Ty>( layer );
        _OutTy *lOutBuffer = out.DeviceBufferAt<_OutTy>( layer );
        lOutBuffer[i]      = static_cast<_OutTy>( lInBuffer[i] * aScaling );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AffineTransform( multi_tensor_t out, multi_tensor_t A, multi_tensor_t X, multi_tensor_t B )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( X.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *lA   = A.DeviceBufferAt<_Ty>( layer );
        _Ty *lX   = X.DeviceBufferAt<_Ty>( layer );
        _Ty *lB   = B.DeviceBufferAt<_Ty>( layer );
        _Ty *lOut = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = lA[i] * lX[i] + lB[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AffineTransform( multi_tensor_t out, multi_tensor_t A, multi_tensor_t X, memory_buffer_t B )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( X.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *lA   = A.DeviceBufferAt<_Ty>( layer );
        _Ty *lX   = X.DeviceBufferAt<_Ty>( layer );
        _Ty *lOut = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = lA[i] * lX[i] + B.DataAs<_Ty>()[layer];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AffineTransform( multi_tensor_t out, multi_tensor_t A, multi_tensor_t X, _Ty B )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( X.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *lA   = A.DeviceBufferAt<_Ty>( layer );
        _Ty *lX   = X.DeviceBufferAt<_Ty>( layer );
        _Ty *lOut = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = lA[i] * lX[i] + B;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AffineTransform( multi_tensor_t out, memory_buffer_t A, multi_tensor_t X, multi_tensor_t B )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( X.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *lX   = X.DeviceBufferAt<_Ty>( layer );
        _Ty *lB   = B.DeviceBufferAt<_Ty>( layer );
        _Ty *lOut = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = A.DataAs<_Ty>()[layer] * lX[i] + lB[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AffineTransform( multi_tensor_t out, memory_buffer_t A, multi_tensor_t X, memory_buffer_t B )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( X.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *lX   = X.DeviceBufferAt<_Ty>( layer );
        _Ty *lOut = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = A.DataAs<_Ty>()[blockIdx.x] * lX[i] + B.DataAs<_Ty>()[blockIdx.x];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AffineTransform( multi_tensor_t out, memory_buffer_t A, multi_tensor_t X, _Ty B )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( X.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *lX   = X.DeviceBufferAt<_Ty>( layer );
        _Ty *lOut = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = A.DataAs<_Ty>()[blockIdx.x] * lX[i] + B;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AffineTransform( multi_tensor_t out, _Ty A, multi_tensor_t X, multi_tensor_t B )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( X.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *lX   = X.DeviceBufferAt<_Ty>( layer );
        _Ty *lB   = B.DeviceBufferAt<_Ty>( layer );
        _Ty *lOut = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = A * lX[i] + lB[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AffineTransform( multi_tensor_t out, _Ty A, multi_tensor_t X, memory_buffer_t B )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( X.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *lX   = X.DeviceBufferAt<_Ty>( layer );
        _Ty *lOut = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = A * lX[i] + B.DataAs<_Ty>()[layer];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AffineTransform( multi_tensor_t out, _Ty A, multi_tensor_t X, _Ty B )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( X.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *lX   = X.DeviceBufferAt<_Ty>( layer );
        _Ty *lOut = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = A * lX[i] + B;
    }

    CUDA_KERNEL_DEFINITION void Floor( multi_tensor_t out, multi_tensor_t x )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<float>( layer, i ) );

        auto *lX   = x.DeviceBufferAt<float>( layer );
        auto *lOut = out.DeviceBufferAt<float>( layer );

        lOut[i] = floor( lX[i] );
    }

    CUDA_KERNEL_DEFINITION void Ceil( multi_tensor_t out, multi_tensor_t x )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<float>( layer, i ) );

        auto *lX   = x.DeviceBufferAt<float>( layer );
        auto *lOut = out.DeviceBufferAt<float>( layer );

        lOut[i] = ceil( lX[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Sqrt( multi_tensor_t out, multi_tensor_t x )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<_Ty>( layer, i ) );

        auto *lX   = x.DeviceBufferAt<_Ty>( layer );
        auto *lOut = out.DeviceBufferAt<float>( layer );

        lOut[i] = static_cast<_Ty>( sqrt( static_cast<float>( lX[i] ) ) );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Round( multi_tensor_t out, multi_tensor_t x )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<_Ty>( layer, i ) );

        auto *lX   = x.DeviceBufferAt<_Ty>( layer );
        auto *lOut = out.DeviceBufferAt<float>( layer );

        if constexpr( std::is_integral<_Ty>::value )
            lOut[i] = lX[i];
        else
            lOut[i] = __int2float_rd( __float2int_rn( lX[i] ) );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Abs( multi_tensor_t out, multi_tensor_t x )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<float>( layer, i ) );

        auto *lX   = x.DeviceBufferAt<_Ty>( layer );
        auto *lOut = out.DeviceBufferAt<_Ty>( layer );

        lOut[i] = lX[i] * ( lX[i] >= 0 ? 1.0f : -1.0f );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void CountZero( multi_tensor_t out, multi_tensor_t x, memory_buffer_t blockSizes,
                                           memory_buffer_t elementCount )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( i < blockSizes.DataAs<float>()[layer] );

        auto lElementCount = elementCount.DataAs<uint32_t>()[layer];

        auto *lX   = x.DeviceBufferAt<_Ty>( layer ) + i * lElementCount;
        auto *lOut = out.DeviceBufferAt<uint32_t>( layer ) + i;

        uint32_t lCount = 0;
        for( uint32_t k = 0; k < lElementCount; k++ )
        {
            if( lX[k] == static_cast<_Ty>( 0 ) )
                lCount++;
        }

        *lOut = lCount;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void CountNonZero( multi_tensor_t out, multi_tensor_t x, memory_buffer_t blockSizes,
                                              memory_buffer_t elementCount )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( i < blockSizes.DataAs<uint32_t>()[layer] );

        auto lElementCount = elementCount.DataAs<uint32_t>()[layer];

        auto *lX   = x.DeviceBufferAt<_Ty>( layer ) + i * lElementCount;
        auto *lOut = out.DeviceBufferAt<uint32_t>( layer ) + i;

        uint32_t lCount = 0;
        for( uint32_t k = 0; k < lElementCount; k++ )
        {
            if( lX[k] != static_cast<_Ty>( 0 ) )
                lCount++;
        }

        *lOut = lCount;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void ArraySummation( multi_tensor_t out, multi_tensor_t x, memory_buffer_t begin, memory_buffer_t end,
                                                memory_buffer_t elementCount, memory_buffer_t blockSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( i < blockSizes.DataAs<uint32_t>()[layer] );

        auto lBegin        = begin.DataAs<uint32_t>()[layer];
        auto lEnd          = end.DataAs<uint32_t>()[layer];
        auto lElementCount = elementCount.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( lBegin <= lEnd ) && ( lEnd < lElementCount ) );

        auto *lX   = x.DeviceBufferAt<_Ty>( layer ) + i * lElementCount;
        auto *lOut = out.DeviceBufferAt<_Ty>( layer ) + i;

        _Ty lAccumulator = 0;
        for( uint32_t k = lBegin; k <= lEnd; k++ )
            lAccumulator += lX[k];

        *lOut = lAccumulator;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void ArraySlice( multi_tensor_t out, multi_tensor_t x, memory_buffer_t begin, memory_buffer_t end,
                                            memory_buffer_t elementCount, memory_buffer_t blockSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( i < blockSizes.DataAs<uint32_t>()[layer] );

        auto lBegin        = begin.DataAs<uint32_t>()[layer];
        auto lEnd          = end.DataAs<uint32_t>()[layer];
        auto lElementCount = elementCount.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( lBegin <= lEnd ) && ( lEnd < lElementCount ) );

        auto *lX   = x.DeviceBufferAt<_Ty>( layer ) + i * lElementCount;
        auto *lOut = out.DeviceBufferAt<_Ty>( layer ) + i * ( lEnd - lBegin + 1 );

        for( uint32_t k = lBegin; k <= lEnd; k++ )
            lOut[k - lBegin] = lX[k];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Diff( multi_tensor_t out, multi_tensor_t x, uint32_t count, memory_buffer_t elementCount,
                                      memory_buffer_t blockSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( i < blockSizes.DataAs<uint32_t>()[layer] );

        auto lElementCount = elementCount.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( count < lElementCount ) );

        auto *lX   = x.DeviceBufferAt<_Ty>( layer ) + i * lElementCount;
        auto *lOut = out.DeviceBufferAt<_Ty>( layer ) + i * lElementCount;

        for( uint32_t l = 0; l < lElementCount; l++ )
        {
            lOut[l] = lX[l];
        }

        for( uint32_t k = 0; k < count; k++ )
        {
            for( uint32_t l = 0; l < lElementCount - k; l++ )
            {
                lOut[l] = lOut[l + 1] - lOut[l];
            }
        }

        for( uint32_t k = lElementCount - count; k < lElementCount; k++ )
        {
            lOut[k] = static_cast<_Ty>( 0 );
        }
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void ShiftLeft( multi_tensor_t out, multi_tensor_t x, uint32_t count, _Ty aFillValue,
                                           memory_buffer_t elementCount, memory_buffer_t blockSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( i < blockSizes.DataAs<uint32_t>()[layer] );

        auto lElementCount = elementCount.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( count < lElementCount ) );

        auto *lX   = x.DeviceBufferAt<_Ty>( layer ) + i * lElementCount;
        auto *lOut = out.DeviceBufferAt<_Ty>( layer ) + i * lElementCount;

        for( uint32_t k = 0; k < lElementCount - count; k++ )
            lOut[k] = lX[k + count];

        for( uint32_t k = lElementCount - count; k < lElementCount; k++ )
            lOut[k] = aFillValue;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void ShiftRight( multi_tensor_t out, multi_tensor_t x, uint32_t count, _Ty aFillValue,
                                            memory_buffer_t elementCount, memory_buffer_t blockSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( i < blockSizes.DataAs<uint32_t>()[layer] );

        auto lElementCount = elementCount.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( count < lElementCount ) );

        auto *lX   = x.DeviceBufferAt<_Ty>( layer ) + i * lElementCount;
        auto *lOut = out.DeviceBufferAt<_Ty>( layer ) + i * lElementCount;

        for( uint32_t k = count; k < lElementCount; k++ )
            lOut[k] = lX[k - count];

        for( uint32_t k = 0; k < count; k++ )
            lOut[k] = aFillValue;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Conv1D( multi_tensor_t out, multi_tensor_t array0, memory_buffer_t elementCount0,
                                        memory_buffer_t blockSizes0, multi_tensor_t array1, memory_buffer_t elementCount1,
                                        memory_buffer_t blockSizes1 )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < blockSizes0.DataAs<uint32_t>()[layer] );

        auto lElementCount0 = elementCount0.DataAs<uint32_t>()[layer];
        auto lElementCount1 = elementCount1.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( i < lElementCount0 ) );

        auto *lX   = array0.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * lElementCount0;
        auto *lK   = array1.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * lElementCount1;
        auto *lOut = out.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * lElementCount0;

        _Ty lConvolutionValue = static_cast<_Ty>( 0 );
        for( uint32_t j = 0; j < lElementCount1; j++ )
        {
            if( i >= j )
                lConvolutionValue += ( lX[i - j] * lK[j] );
        }

        lOut[i] = lConvolutionValue;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void HCat( multi_tensor_t out, multi_tensor_t x, memory_buffer_t elementCountX, multi_tensor_t y,
                                      memory_buffer_t elementCountY, memory_buffer_t blockSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( ( i < blockSizes.DataAs<uint32_t>()[layer] ) );

        auto lElementCountX = elementCountX.DataAs<uint32_t>()[layer];
        auto lElementCountY = elementCountY.DataAs<uint32_t>()[layer];

        auto *lX   = x.DeviceBufferAt<_Ty>( layer ) + i * lElementCountX;
        auto *lY   = y.DeviceBufferAt<_Ty>( layer ) + i * lElementCountY;
        auto *lOut = out.DeviceBufferAt<_Ty>( layer ) + i * ( lElementCountX + lElementCountY );

        uint32_t k = 0;
        for( uint32_t j = 0; j < lElementCountX; j++ )
            lOut[k++] = lX[j];
        for( uint32_t j = 0; j < lElementCountY; j++ )
            lOut[k++] = lY[j];
    }
} // namespace SE::TensorOps::Kernels