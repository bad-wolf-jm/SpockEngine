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
#include "Core/CUDA/Texture/Texture2D.h"

#include "HelperMacros.h"

namespace numlua::mtops::Kernels
{
    using namespace numlua::cuda;

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

        _Ty *_out = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = left.DataAs<_Ty>()[blockIdx.x] + i * aDelta.DataAs<_Ty>()[blockIdx.x];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Add( multi_tensor_t out, multi_tensor_t left, multi_tensor_t right )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( left.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *_array    = left.DeviceBufferAt<_Ty>( layer );
        _Ty *_constant = right.DeviceBufferAt<_Ty>( layer );
        _Ty *_out      = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = _array[i] + _constant[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Add( multi_tensor_t out, multi_tensor_t left, multi_tensor_t right, broadcast_hint_t broadcastHint,
                                     memory_buffer_t blockSizes, memory_buffer_t broadcastSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < blockSizes.DataAs<uint32_t>()[layer] );

        auto broadcast_size = broadcastSizes.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( i < broadcast_size ) );

        auto *_out = out.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * broadcast_size;

        switch( broadcastHint )
        {
        case broadcast_hint_t::LEFT:
        {
            _Ty *_left  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;
            _Ty *_right = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * broadcast_size;

            _out[i] = _left[0] + _right[i];
        }
        break;
        case broadcast_hint_t::RIGHT:
        {
            _Ty *_left  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * broadcast_size;
            _Ty *_right = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;

            _out[i] = _left[i] + _right[0];
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

        _Ty *_array = array.DeviceBufferAt<_Ty>( layer );
        _Ty *_out   = out.DeviceBufferAt<_Ty>( layer );
        _out[i]     = _array[i] + constant;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Add( multi_tensor_t out, multi_tensor_t array, memory_buffer_t constants )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *_in  = array.DeviceBufferAt<_Ty>( layer );
        _Ty *_out = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = _in[i] + constants.DataAs<_Ty>()[blockIdx.x];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Multiply( multi_tensor_t out, multi_tensor_t array, multi_tensor_t constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *_array    = array.DeviceBufferAt<_Ty>( layer );
        _Ty *_constant = constant.DeviceBufferAt<_Ty>( layer );
        _Ty *_out      = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = _array[i] * _constant[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Multiply( multi_tensor_t out, multi_tensor_t left, multi_tensor_t right,
                                          broadcast_hint_t broadcastHint, memory_buffer_t blockSizes, memory_buffer_t broadcastSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < blockSizes.DataAs<uint32_t>()[layer] );

        auto broadcast_size = broadcastSizes.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( i < broadcast_size ) );

        auto *_out = out.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * broadcast_size;

        switch( broadcastHint )
        {
        case broadcast_hint_t::LEFT:
        {
            _Ty *_left  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;
            _Ty *_right = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * broadcast_size;

            _out[i] = _left[0] * _right[i];
        }
        break;
        case broadcast_hint_t::RIGHT:
        {
            _Ty *_left  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * broadcast_size;
            _Ty *_right = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;

            _out[i] = _left[i] * _right[0];
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

        _Ty *_array = array.DeviceBufferAt<_Ty>( layer );
        _Ty *_out   = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = _array[i] * constant;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Multiply( multi_tensor_t out, multi_tensor_t array, memory_buffer_t constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *_in  = array.DeviceBufferAt<_Ty>( layer );
        _Ty *_out = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = _in[i] * constant.DataAs<_Ty>()[blockIdx.x];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Subtract( multi_tensor_t out, multi_tensor_t array, _Ty constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *_array = array.DeviceBufferAt<_Ty>( layer );
        _Ty *_out   = out.DeviceBufferAt<_Ty>( layer );
        _out[i]     = _array[i] - constant;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Subtract( multi_tensor_t out, _Ty constant, multi_tensor_t array )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *_array = array.DeviceBufferAt<_Ty>( layer );
        _Ty *_out   = out.DeviceBufferAt<_Ty>( layer );
        _out[i]     = constant - _array[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Subtract( multi_tensor_t out, multi_tensor_t array, multi_tensor_t constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *_array    = array.DeviceBufferAt<_Ty>( layer );
        _Ty *_constant = constant.DeviceBufferAt<_Ty>( layer );
        _Ty *_out      = out.DeviceBufferAt<_Ty>( layer );
        _out[i]        = _array[i] - _constant[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Subtract( multi_tensor_t out, multi_tensor_t left, multi_tensor_t right,
                                          broadcast_hint_t broadcastHint, memory_buffer_t blockSizes, memory_buffer_t broadcastSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < blockSizes.DataAs<uint32_t>()[layer] );

        auto broadcast_size = broadcastSizes.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( i < broadcast_size ) );

        auto *_out = out.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * broadcast_size;

        switch( broadcastHint )
        {
        case broadcast_hint_t::LEFT:
        {
            _Ty *_left  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;
            _Ty *_right = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * broadcast_size;

            _out[i] = _left[0] - _right[i];
        }
        break;
        case broadcast_hint_t::RIGHT:
        {
            _Ty *_left  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * broadcast_size;
            _Ty *_right = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;

            _out[i] = _left[i] - _right[0];
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

        _Ty *_in  = array.DeviceBufferAt<_Ty>( layer );
        _Ty *_out = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = _in[i] - constant.DataAs<_Ty>()[blockIdx.x];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Subtract( multi_tensor_t out, memory_buffer_t constant, multi_tensor_t array )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *_in  = array.DeviceBufferAt<_Ty>( layer );
        _Ty *_out = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = constant.DataAs<_Ty>()[blockIdx.x] - _in[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Divide( multi_tensor_t out, multi_tensor_t array, _Ty constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        auto *_array = array.DeviceBufferAt<_Ty>( layer );
        auto *_out   = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = _array[i] / constant;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Divide( multi_tensor_t out, _Ty constant, multi_tensor_t array )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<_Ty>( layer, i ) );

        auto *_array = array.DeviceBufferAt<_Ty>( layer );
        auto *_out   = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = constant / _array[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Divide( multi_tensor_t out, multi_tensor_t array, multi_tensor_t constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        auto *_array    = array.DeviceBufferAt<_Ty>( layer );
        auto *_constant = constant.DeviceBufferAt<_Ty>( layer );
        auto *_out      = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = _array[i] / _constant[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Divide( multi_tensor_t out, multi_tensor_t left, multi_tensor_t right, broadcast_hint_t broadcastHint,
                                        memory_buffer_t blockSizes, memory_buffer_t broadcastSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < blockSizes.DataAs<uint32_t>()[layer] );

        auto broadcast_size = broadcastSizes.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( i < broadcast_size ) );

        auto *_out = out.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * broadcast_size;

        switch( broadcastHint )
        {
        case broadcast_hint_t::LEFT:
        {
            _Ty *_left  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;
            _Ty *_right = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * broadcast_size;

            _out[i] = _left[0] / _right[i];
        }
        break;
        case broadcast_hint_t::RIGHT:
        {
            _Ty *_left  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * broadcast_size;
            _Ty *_right = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;

            _out[i] = _left[i] / _right[0];
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

        auto *_in  = array.DeviceBufferAt<_Ty>( layer );
        auto *_out = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = _in[i] / constant.DataAs<_Ty>()[blockIdx.x];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Divide( multi_tensor_t out, memory_buffer_t constant, multi_tensor_t array )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        auto *_in  = array.DeviceBufferAt<_Ty>( layer );
        auto *_out = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = constant.DataAs<_Ty>()[blockIdx.x] / _in[i];
    }

    CUDA_KERNEL_DEFINITION void And( multi_tensor_t out, multi_tensor_t array, uint8_t constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<uint8_t>( layer, i ) );

        uint8_t *_array = array.DeviceBufferAt<uint8_t>( layer );
        uint8_t *_out   = out.DeviceBufferAt<uint8_t>( layer );

        _out[i] = ( constant && _array[i] );
    }

    CUDA_KERNEL_DEFINITION void And( multi_tensor_t out, multi_tensor_t array, multi_tensor_t constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<uint8_t>( layer, i ) );

        uint8_t *_array    = array.DeviceBufferAt<uint8_t>( layer );
        uint8_t *_constant = constant.DeviceBufferAt<uint8_t>( layer );
        uint8_t *_out      = out.DeviceBufferAt<uint8_t>( layer );

        _out[i] = ( _array[i] && _constant[i] );
    }

    CUDA_KERNEL_DEFINITION void And( multi_tensor_t out, multi_tensor_t left, multi_tensor_t right, broadcast_hint_t broadcastHint,
                                     memory_buffer_t blockSizes, memory_buffer_t broadcastSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < blockSizes.DataAs<uint32_t>()[layer] );

        auto broadcast_size = broadcastSizes.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( i < broadcast_size ) );

        uint8_t *_out = out.DeviceBufferAt<uint8_t>( layer ) + blockIdx.y * broadcast_size;

        switch( broadcastHint )
        {
        case broadcast_hint_t::LEFT:
        {
            uint8_t *_left  = left.DeviceBufferAt<uint8_t>( layer ) + blockIdx.y;
            uint8_t *_right = right.DeviceBufferAt<uint8_t>( layer ) + blockIdx.y * broadcast_size;

            _out[i] = ( _left[0] && _right[i] );
        }
        break;
        case broadcast_hint_t::RIGHT:
        {
            uint8_t *_left  = left.DeviceBufferAt<uint8_t>( layer ) + blockIdx.y * broadcast_size;
            uint8_t *_right = right.DeviceBufferAt<uint8_t>( layer ) + blockIdx.y;

            _out[i] = ( _left[i] && _right[0] );
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

        uint8_t *_in  = array.DeviceBufferAt<uint8_t>( layer );
        uint8_t *_out = out.DeviceBufferAt<uint8_t>( layer );

        _out[i] = ( _in[i] && constant.DataAs<uint8_t>()[blockIdx.x] );
    }

    CUDA_KERNEL_DEFINITION void Or( multi_tensor_t out, multi_tensor_t array, uint8_t constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<uint8_t>( layer, i ) );

        uint8_t *_array = array.DeviceBufferAt<uint8_t>( layer );
        uint8_t *_out   = out.DeviceBufferAt<uint8_t>( layer );

        _out[i] = ( constant || _array[i] );
    }

    CUDA_KERNEL_DEFINITION void Or( multi_tensor_t out, multi_tensor_t array, multi_tensor_t constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<uint8_t>( layer, i ) );

        uint8_t *_array    = array.DeviceBufferAt<uint8_t>( layer );
        uint8_t *_constant = constant.DeviceBufferAt<uint8_t>( layer );
        uint8_t *_out      = out.DeviceBufferAt<uint8_t>( layer );

        _out[i] = ( _array[i] || _constant[i] );
    }

    CUDA_KERNEL_DEFINITION void Or( multi_tensor_t out, multi_tensor_t left, multi_tensor_t right, broadcast_hint_t broadcastHint,
                                    memory_buffer_t blockSizes, memory_buffer_t broadcastSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < blockSizes.DataAs<uint32_t>()[layer] );

        auto broadcast_size = broadcastSizes.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( i < broadcast_size ) );

        uint8_t *_out = out.DeviceBufferAt<uint8_t>( layer ) + blockIdx.y * broadcast_size;

        switch( broadcastHint )
        {
        case broadcast_hint_t::LEFT:
        {
            uint8_t *_left  = left.DeviceBufferAt<uint8_t>( layer ) + blockIdx.y;
            uint8_t *_right = right.DeviceBufferAt<uint8_t>( layer ) + blockIdx.y * broadcast_size;

            _out[i] = ( _left[0] || _right[i] );
        }
        break;
        case broadcast_hint_t::RIGHT:
        {
            uint8_t *_left  = left.DeviceBufferAt<uint8_t>( layer ) + blockIdx.y * broadcast_size;
            uint8_t *_right = right.DeviceBufferAt<uint8_t>( layer ) + blockIdx.y;

            _out[i] = ( _left[i] || _right[0] );
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

        uint8_t *_in  = array.DeviceBufferAt<uint8_t>( layer );
        uint8_t *_out = out.DeviceBufferAt<uint8_t>( layer );

        _out[i] = ( _in[i] || constant.DataAs<uint8_t>()[blockIdx.x] );
    }

    CUDA_KERNEL_DEFINITION void Not( multi_tensor_t out, multi_tensor_t array )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<uint8_t>( layer, i ) );

        uint8_t *_in  = array.DeviceBufferAt<uint8_t>( layer );
        uint8_t *_out = out.DeviceBufferAt<uint8_t>( layer );

        _out[i] = !( _in[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void BitwiseAnd( multi_tensor_t out, multi_tensor_t array, _Ty constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *_array = array.DeviceBufferAt<_Ty>( layer );
        _Ty *_out   = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = ( constant & _array[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void BitwiseAnd( multi_tensor_t out, multi_tensor_t left, multi_tensor_t right,
                                            broadcast_hint_t broadcastHint, memory_buffer_t blockSizes,
                                            memory_buffer_t broadcastSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < blockSizes.DataAs<uint32_t>()[layer] );

        auto broadcast_size = broadcastSizes.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( i < broadcast_size ) );

        auto *_out = out.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * broadcast_size;

        switch( broadcastHint )
        {
        case broadcast_hint_t::LEFT:
        {
            _Ty *_left  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;
            _Ty *_right = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * broadcast_size;

            _out[i] = _left[0] & _right[i];
        }
        break;
        case broadcast_hint_t::RIGHT:
        {
            _Ty *_left  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * broadcast_size;
            _Ty *_right = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;

            _out[i] = _left[i] & _right[0];
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

        _Ty *_array    = array.DeviceBufferAt<_Ty>( layer );
        _Ty *_constant = constant.DeviceBufferAt<_Ty>( layer );
        _Ty *_out      = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = ( _array[i] & _constant[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void BitwiseAnd( multi_tensor_t out, multi_tensor_t array, memory_buffer_t constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *_in  = array.DeviceBufferAt<_Ty>( layer );
        _Ty *_out = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = ( _in[i] & constant.DataAs<_Ty>()[blockIdx.x] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void BitwiseOr( multi_tensor_t out, multi_tensor_t array, _Ty constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *_array = array.DeviceBufferAt<_Ty>( layer );
        _Ty *_out   = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = ( constant | _array[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void BitwiseOr( multi_tensor_t out, multi_tensor_t array, multi_tensor_t constant )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *_array    = array.DeviceBufferAt<_Ty>( layer );
        _Ty *_constant = constant.DeviceBufferAt<_Ty>( layer );
        _Ty *_out      = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = ( _array[i] | _constant[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void BitwiseOr( multi_tensor_t out, multi_tensor_t left, multi_tensor_t right,
                                           broadcast_hint_t broadcastHint, memory_buffer_t blockSizes, memory_buffer_t broadcastSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < blockSizes.DataAs<uint32_t>()[layer] );

        auto broadcast_size = broadcastSizes.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( i < broadcast_size ) );

        auto *_out = out.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * broadcast_size;

        switch( broadcastHint )
        {
        case broadcast_hint_t::LEFT:
        {
            _Ty *_left  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;
            _Ty *_right = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * broadcast_size;

            _out[i] = _left[0] | _right[i];
        }
        break;
        case broadcast_hint_t::RIGHT:
        {
            _Ty *_left  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * broadcast_size;
            _Ty *_right = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;

            _out[i] = _left[i] | _right[0];
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

        _Ty *_in  = array.DeviceBufferAt<_Ty>( layer );
        _Ty *_out = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = ( _in[i] | constant.DataAs<_Ty>()[blockIdx.x] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Bitwise( multi_tensor_t out, multi_tensor_t array )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *_in  = array.DeviceBufferAt<_Ty>( layer );
        _Ty *_out = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = ~( _in[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void InInterval( multi_tensor_t out, multi_tensor_t x, multi_tensor_t lower, multi_tensor_t upper,
                                            bool strictLower, bool strictUpper )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<_Ty>( layer, i ) );

        _Ty     *_x     = x.DeviceBufferAt<_Ty>( layer );
        _Ty     *_lower = lower.DeviceBufferAt<_Ty>( layer );
        _Ty     *_upper = upper.DeviceBufferAt<_Ty>( layer );
        uint8_t *_out   = out.DeviceBufferAt<uint8_t>( layer );

        bool comp_0 = strictLower ? ( _lower[i] < _x[i] ) : ( _lower[i] <= _x[i] );
        bool comp_1 = strictUpper ? ( _upper[i] > _x[i] ) : ( _upper[i] >= _x[i] );

        _out[i] = ( comp_0 && comp_1 );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void InInterval( multi_tensor_t out, multi_tensor_t x, multi_tensor_t lower, memory_buffer_t upper,
                                            bool strictLower, bool strictUpper )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<_Ty>( layer, i ) );

        _Ty     *_x     = x.DeviceBufferAt<_Ty>( layer );
        _Ty     *_lower = lower.DeviceBufferAt<_Ty>( layer );
        _Ty     *_upper = upper.DataAs<_Ty>();
        uint8_t *_out   = out.DeviceBufferAt<uint8_t>( layer );

        bool comp_0 = strictLower ? ( _lower[i] < _x[i] ) : ( _lower[i] <= _x[i] );
        bool comp_1 = strictUpper ? ( _upper[layer] > _x[i] ) : ( _upper[layer] >= _x[i] );

        _out[i] = ( comp_0 && comp_1 );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void InInterval( multi_tensor_t out, multi_tensor_t x, multi_tensor_t lower, _Ty upper, bool strictLower,
                                            bool strictUpper )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<_Ty>( layer, i ) );

        _Ty     *_x     = x.DeviceBufferAt<_Ty>( layer );
        _Ty     *_lower = lower.DeviceBufferAt<_Ty>( layer );
        uint8_t *_out   = out.DeviceBufferAt<uint8_t>( layer );

        bool comp_0 = strictLower ? ( _lower[i] < _x[i] ) : ( _lower[i] <= _x[i] );
        bool comp_1 = strictUpper ? ( upper > _x[i] ) : ( upper >= _x[i] );

        _out[i] = ( comp_0 && comp_1 );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void InInterval( multi_tensor_t out, multi_tensor_t x, memory_buffer_t lower, multi_tensor_t upper,
                                            bool strictLower, bool strictUpper )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<_Ty>( layer, i ) );

        _Ty     *_x     = x.DeviceBufferAt<_Ty>( layer );
        _Ty     *_lower = lower.DataAs<_Ty>();
        _Ty     *_upper = upper.DeviceBufferAt<_Ty>( layer );
        uint8_t *_out   = out.DeviceBufferAt<uint8_t>( layer );

        bool comp_0 = strictLower ? ( _lower[layer] < _x[i] ) : ( _lower[layer] <= _x[i] );
        bool comp_1 = strictUpper ? ( _upper[i] > _x[i] ) : ( _upper[i] >= _x[i] );

        _out[i] = ( comp_0 && comp_1 );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void InInterval( multi_tensor_t out, multi_tensor_t x, memory_buffer_t lower, memory_buffer_t upper,
                                            bool strictLower, bool strictUpper )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<_Ty>( layer, i ) );

        _Ty     *_x     = x.DeviceBufferAt<_Ty>( layer );
        _Ty     *_lower = lower.DataAs<_Ty>();
        _Ty     *_upper = upper.DataAs<_Ty>();
        uint8_t *_out   = out.DeviceBufferAt<uint8_t>( layer );

        bool comp_0 = strictLower ? ( _lower[layer] < _x[i] ) : ( _lower[layer] <= _x[i] );
        bool comp_1 = strictUpper ? ( _upper[layer] > _x[i] ) : ( _upper[layer] >= _x[i] );

        _out[i] = ( comp_0 && comp_1 );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void InInterval( multi_tensor_t out, multi_tensor_t x, memory_buffer_t lower, _Ty upper, bool strictLower,
                                            bool strictUpper )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<_Ty>( layer, i ) );

        _Ty     *_x     = x.DeviceBufferAt<_Ty>( layer );
        _Ty     *_lower = lower.DataAs<_Ty>();
        uint8_t *_out   = out.DeviceBufferAt<uint8_t>( layer );

        bool comp_0 = strictLower ? ( _lower[layer] < _x[i] ) : ( _lower[layer] <= _x[i] );
        bool comp_1 = strictUpper ? ( upper > _x[i] ) : ( upper >= _x[i] );

        _out[i] = ( comp_0 && comp_1 );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void InInterval( multi_tensor_t out, multi_tensor_t x, _Ty lower, multi_tensor_t upper, bool strictLower,
                                            bool strictUpper )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<_Ty>( layer, i ) );

        _Ty     *_x     = x.DeviceBufferAt<_Ty>( layer );
        _Ty     *_upper = upper.DeviceBufferAt<_Ty>( layer );
        uint8_t *_out   = out.DeviceBufferAt<uint8_t>( layer );

        bool comp_0 = strictLower ? ( lower < _x[i] ) : ( lower <= _x[i] );
        bool comp_1 = strictUpper ? ( _upper[i] > _x[i] ) : ( _upper[i] >= _x[i] );

        _out[i] = ( comp_0 && comp_1 );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void InInterval( multi_tensor_t out, multi_tensor_t x, _Ty lower, memory_buffer_t upper, bool strictLower,
                                            bool strictUpper )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<_Ty>( layer, i ) );

        _Ty     *_x     = x.DeviceBufferAt<_Ty>( layer );
        _Ty     *_upper = upper.DataAs<_Ty>();
        uint8_t *_out   = out.DeviceBufferAt<uint8_t>( layer );

        bool comp_0 = strictLower ? ( lower < _x[i] ) : ( lower <= _x[i] );
        bool comp_1 = strictUpper ? ( _upper[layer] > _x[i] ) : ( _upper[layer] >= _x[i] );

        _out[i] = ( comp_0 && comp_1 );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void InInterval( multi_tensor_t out, multi_tensor_t x, _Ty lower, _Ty upper, bool strictLower,
                                            bool strictUpper )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<_Ty>( layer, i ) );

        _Ty     *_x   = x.DeviceBufferAt<_Ty>( layer );
        uint8_t *_out = out.DeviceBufferAt<uint8_t>( layer );

        bool comp_0 = strictLower ? ( lower < _x[i] ) : ( lower <= _x[i] );
        bool comp_1 = strictUpper ? ( upper > _x[i] ) : ( upper >= _x[i] );

        _out[i] = ( comp_0 && comp_1 );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void EqualOp( multi_tensor_t out, multi_tensor_t x, multi_tensor_t y )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<uint8_t>( layer, i ) );

        _Ty     *_x   = x.DeviceBufferAt<_Ty>( layer );
        _Ty     *_l   = y.DeviceBufferAt<_Ty>( layer );
        uint8_t *_out = out.DeviceBufferAt<uint8_t>( layer );

        _out[i] = ( _x[i] == _l[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void EqualOp( multi_tensor_t out, multi_tensor_t x, memory_buffer_t y )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<uint8_t>( layer, i ) );

        _Ty     *_x   = x.DeviceBufferAt<_Ty>( layer );
        _Ty     *_l   = y.DataAs<_Ty>();
        uint8_t *_out = out.DeviceBufferAt<uint8_t>( layer );

        _out[i] = ( _x[i] == _l[layer] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void EqualOp( multi_tensor_t out, multi_tensor_t x, _Ty y )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<uint8_t>( layer, i ) );

        _Ty     *_x   = x.DeviceBufferAt<_Ty>( layer );
        uint8_t *_out = out.DeviceBufferAt<uint8_t>( layer );

        _out[i] = ( _x[i] == y );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void EqualOp( multi_tensor_t out, multi_tensor_t left, multi_tensor_t right, broadcast_hint_t broadcastHint,
                                         memory_buffer_t blockSizes, memory_buffer_t broadcastSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < blockSizes.DataAs<uint32_t>()[layer] );

        auto broadcast_size = broadcastSizes.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( i < broadcast_size ) );

        auto *_out = out.DeviceBufferAt<uint8_t>( layer ) + blockIdx.y * broadcast_size;

        switch( broadcastHint )
        {
        case broadcast_hint_t::LEFT:
        {
            _Ty *_left  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;
            _Ty *_right = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * broadcast_size;

            _out[i] = _left[0] == _right[i];
        }
        break;
        case broadcast_hint_t::RIGHT:
        {
            _Ty *_left  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * broadcast_size;
            _Ty *_right = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;

            _out[i] = _left[i] == _right[0];
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

        _Ty     *_x   = x.DataAs<_Ty>();
        _Ty     *_l   = y.DeviceBufferAt<_Ty>( layer );
        uint8_t *_out = out.DeviceBufferAt<uint8_t>( layer );

        _out[i] = ( _x[layer] == _l[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void EqualOp( multi_tensor_t out, _Ty x, multi_tensor_t y )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<uint8_t>( layer, i ) );

        _Ty     *_l   = y.DeviceBufferAt<_Ty>( layer );
        uint8_t *_out = out.DeviceBufferAt<uint8_t>( layer );

        _out[i] = ( x == _l[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOp( multi_tensor_t out, multi_tensor_t x, multi_tensor_t y )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<uint8_t>( layer, i ) );

        _Ty     *_x   = x.DeviceBufferAt<_Ty>( layer );
        _Ty     *_l   = y.DeviceBufferAt<_Ty>( layer );
        uint8_t *_out = out.DeviceBufferAt<uint8_t>( layer );

        _out[i] = ( _x[i] < _l[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOp( multi_tensor_t out, multi_tensor_t left, multi_tensor_t right,
                                            broadcast_hint_t broadcastHint, memory_buffer_t blockSizes,
                                            memory_buffer_t broadcastSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < blockSizes.DataAs<uint32_t>()[layer] );

        auto broadcast_size = broadcastSizes.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( i < broadcast_size ) );

        auto *_out = out.DeviceBufferAt<uint8_t>( layer ) + blockIdx.y * broadcast_size;

        switch( broadcastHint )
        {
        case broadcast_hint_t::LEFT:
        {
            _Ty *_left  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;
            _Ty *_right = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * broadcast_size;

            _out[i] = _left[0] < _right[i];
        }
        break;
        case broadcast_hint_t::RIGHT:
        {
            _Ty *_left  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * broadcast_size;
            _Ty *_right = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;

            _out[i] = _left[i] < _right[0];
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

        _Ty     *_x   = x.DeviceBufferAt<_Ty>( layer );
        _Ty     *_l   = y.DataAs<_Ty>();
        uint8_t *_out = out.DeviceBufferAt<uint8_t>( layer );

        _out[i] = ( _x[i] < _l[layer] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOp( multi_tensor_t out, multi_tensor_t x, _Ty y )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<uint8_t>( layer, i ) );

        _Ty     *_x   = x.DeviceBufferAt<_Ty>( layer );
        uint8_t *_out = out.DeviceBufferAt<uint8_t>( layer );

        _out[i] = ( _x[i] < y );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOp( multi_tensor_t out, memory_buffer_t x, multi_tensor_t y )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<uint8_t>( layer, i ) );

        _Ty     *_x   = x.DataAs<_Ty>();
        _Ty     *_l   = y.DeviceBufferAt<_Ty>( layer );
        uint8_t *_out = out.DeviceBufferAt<uint8_t>( layer );

        _out[i] = ( _x[layer] < _l[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOp( multi_tensor_t out, _Ty x, multi_tensor_t y )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<uint8_t>( layer, i ) );

        _Ty     *_l   = y.DeviceBufferAt<_Ty>( layer );
        uint8_t *_out = out.DeviceBufferAt<uint8_t>( layer );

        _out[i] = ( x < _l[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOrEqualOp( multi_tensor_t out, multi_tensor_t x, multi_tensor_t y )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<uint8_t>( layer, i ) );

        _Ty     *_x   = x.DeviceBufferAt<_Ty>( layer );
        _Ty     *_l   = y.DeviceBufferAt<_Ty>( layer );
        uint8_t *_out = out.DeviceBufferAt<uint8_t>( layer );

        _out[i] = ( _x[i] <= _l[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOrEqualOp( multi_tensor_t out, multi_tensor_t left, multi_tensor_t right,
                                                   broadcast_hint_t broadcastHint, memory_buffer_t blockSizes,
                                                   memory_buffer_t broadcastSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < blockSizes.DataAs<uint32_t>()[layer] );

        auto broadcast_size = broadcastSizes.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( i < broadcast_size ) );

        auto *_out = out.DeviceBufferAt<uint8_t>( layer ) + blockIdx.y * broadcast_size;

        switch( broadcastHint )
        {
        case broadcast_hint_t::LEFT:
        {
            _Ty *_left  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;
            _Ty *_right = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * broadcast_size;

            _out[i] = _left[0] <= _right[i];
        }
        break;
        case broadcast_hint_t::RIGHT:
        {
            _Ty *_left  = left.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * broadcast_size;
            _Ty *_right = right.DeviceBufferAt<_Ty>( layer ) + blockIdx.y;

            _out[i] = _left[i] <= _right[0];
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

        _Ty     *_x   = x.DeviceBufferAt<_Ty>( layer );
        _Ty     *_l   = y.DataAs<_Ty>();
        uint8_t *_out = out.DeviceBufferAt<uint8_t>( layer );

        _out[i] = ( _x[i] <= _l[layer] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOrEqualOp( multi_tensor_t out, multi_tensor_t x, _Ty y )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<uint8_t>( layer, i ) );

        _Ty     *_x   = x.DeviceBufferAt<_Ty>( layer );
        uint8_t *_out = out.DeviceBufferAt<uint8_t>( layer );

        _out[i] = ( _x[i] <= y );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOrEqualOp( multi_tensor_t out, memory_buffer_t x, multi_tensor_t y )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<uint8_t>( layer, i ) );

        _Ty     *_x   = x.DataAs<_Ty>();
        _Ty     *_l   = y.DeviceBufferAt<_Ty>( layer );
        uint8_t *_out = out.DeviceBufferAt<uint8_t>( layer );

        _out[i] = ( _x[layer] <= _l[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOrEqualOp( multi_tensor_t out, _Ty x, multi_tensor_t y )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( out.Shape().InBounds<uint8_t>( layer, i ) );

        _Ty     *_l   = y.DeviceBufferAt<_Ty>( layer );
        uint8_t *_out = out.DeviceBufferAt<uint8_t>( layer );

        _out[i] = ( x <= _l[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void WhereTensorTensor( multi_tensor_t out, multi_tensor_t condition, multi_tensor_t valueIfTrue,
                                                   multi_tensor_t valueIfFalse )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( condition.Shape().InBounds<uint8_t>( layer, i ) );

        uint8_t *_consition      = condition.DeviceBufferAt<uint8_t>( layer );
        _Ty     *_value_if_true  = valueIfTrue.DeviceBufferAt<_Ty>( layer );
        _Ty     *_value_if_false = valueIfFalse.DeviceBufferAt<_Ty>( layer );
        _Ty     *_out            = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = _consition[i] ? _value_if_true[i] : _value_if_false[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void WhereTensorVector( multi_tensor_t out, multi_tensor_t condition, multi_tensor_t valueIfTrue,
                                                   memory_buffer_t valueIfFalse )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( condition.Shape().InBounds<uint8_t>( layer, i ) );

        uint8_t *_consition      = condition.DeviceBufferAt<uint8_t>( layer );
        _Ty     *_value_if_true  = valueIfTrue.DeviceBufferAt<_Ty>( layer );
        _Ty     *_value_if_false = valueIfFalse.DataAs<_Ty>();
        _Ty     *_out            = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = _consition[i] ? _value_if_true[i] : _value_if_false[layer];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void WhereTensorScalar( multi_tensor_t out, multi_tensor_t condition, multi_tensor_t valueIfTrue,
                                                   _Ty valueIfFalse )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( condition.Shape().InBounds<uint8_t>( layer, i ) );

        uint8_t *_consition     = condition.DeviceBufferAt<uint8_t>( layer );
        _Ty     *_value_if_true = valueIfTrue.DeviceBufferAt<_Ty>( layer );
        _Ty     *_out           = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = _consition[i] ? _value_if_true[i] : valueIfFalse;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void WhereVectorTensor( multi_tensor_t out, multi_tensor_t condition, memory_buffer_t valueIfTrue,
                                                   multi_tensor_t valueIfFalse )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( condition.Shape().InBounds<uint8_t>( layer, i ) );

        uint8_t *_consition      = condition.DeviceBufferAt<uint8_t>( layer );
        _Ty     *_value_if_true  = valueIfTrue.DataAs<_Ty>();
        _Ty     *_value_if_false = valueIfFalse.DeviceBufferAt<_Ty>( layer );
        _Ty     *_out            = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = _consition[i] ? _value_if_true[layer] : _value_if_false[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void WhereVectorVector( multi_tensor_t out, multi_tensor_t condition, memory_buffer_t valueIfTrue,
                                                   memory_buffer_t valueIfFalse )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( condition.Shape().InBounds<uint8_t>( layer, i ) );

        uint8_t *_consition      = condition.DeviceBufferAt<uint8_t>( layer );
        _Ty     *_value_if_true  = valueIfTrue.DataAs<_Ty>();
        _Ty     *_value_if_false = valueIfFalse.DataAs<_Ty>();
        _Ty     *_out            = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = _consition[i] ? _value_if_true[layer] : _value_if_false[layer];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void WhereVectorScalar( multi_tensor_t out, multi_tensor_t condition, memory_buffer_t valueIfTrue,
                                                   _Ty valueIfFalse )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( condition.Shape().InBounds<uint8_t>( layer, i ) );

        uint8_t *_consition     = condition.DeviceBufferAt<uint8_t>( layer );
        _Ty     *_value_if_true = valueIfTrue.DataAs<_Ty>();
        _Ty     *_out           = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = _consition[i] ? _value_if_true[layer] : valueIfFalse;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void WhereScalarTensor( multi_tensor_t out, multi_tensor_t condition, _Ty valueIfTrue,
                                                   multi_tensor_t valueIfFalse )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( condition.Shape().InBounds<uint8_t>( layer, i ) );

        uint8_t *_consition      = condition.DeviceBufferAt<uint8_t>( layer );
        _Ty     *_value_if_false = valueIfFalse.DeviceBufferAt<_Ty>( layer );
        _Ty     *_out            = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = _consition[i] ? valueIfTrue : _value_if_false[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void WhereScalarVector( multi_tensor_t out, multi_tensor_t condition, _Ty valueIfTrue,
                                                   memory_buffer_t valueIfFalse )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( condition.Shape().InBounds<uint8_t>( layer, i ) );

        uint8_t *_consition      = condition.DeviceBufferAt<uint8_t>( layer );
        _Ty     *_value_if_false = valueIfFalse.DataAs<_Ty>();
        _Ty     *_out            = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = _consition[i] ? valueIfTrue : _value_if_false[layer];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void WhereScalarScalar( multi_tensor_t out, multi_tensor_t condition, _Ty valueIfTrue, _Ty valueIfFalse )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( condition.Shape().InBounds<uint8_t>( layer, i ) );

        uint8_t *_consition = condition.DeviceBufferAt<uint8_t>( layer );
        _Ty     *_out       = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = _consition[i] ? valueIfTrue : valueIfFalse;
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

        _Ty *_array = array.DeviceBufferAt<_Ty>( layer );
        _Ty *_out   = out.DeviceBufferAt<_Ty>( layer );

        _out[blockIdx.y * N + i] = _array[j];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Tile( multi_tensor_t out, multi_tensor_t array, memory_buffer_t repetitions )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        uint32_t N     = repetitions.DataAs<uint32_t>()[blockIdx.x];

        RETURN_UNLESS( blockIdx.y < N );

        int i = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;
        int j = blockIdx.y * array.Shape().GetBufferSizeAs<_Ty>( layer ).Size + i;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) && out.Shape().InBounds<_Ty>( layer, j ) );

        _Ty *_array = array.DeviceBufferAt<_Ty>( layer );
        _Ty *_out   = out.DeviceBufferAt<_Ty>( layer );
        _out[j]     = _array[i];
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
        _Ty *_out      = out.DeviceBufferAt<_Ty>( layer );

        float aDelta = ( lInArrayB[j] - lInArrayA[j] ) / static_cast<float>( N );
        _out[k]      = lInArrayA[j] + i * aDelta;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Mix( multi_tensor_t out, multi_tensor_t A, multi_tensor_t B, multi_tensor_t t )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( A.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *_A   = A.DeviceBufferAt<_Ty>( layer );
        _Ty *_B   = B.DeviceBufferAt<_Ty>( layer );
        _Ty *lT   = t.DeviceBufferAt<_Ty>( layer );
        _Ty *_out = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = ( 1 - lT[i] ) * _A[i] + lT[i] * _B[i];
    }

    CUDA_KERNEL_DEFINITION void Sample2D( multi_tensor_t out, multi_tensor_t x, multi_tensor_t y, memory_buffer_t textures )
    {
        uint32_t                              layer = static_cast<uint32_t>( blockIdx.x );
        cuda::texture_sampler2d_t::DeviceData lTex  = textures.DataAs<cuda::texture_sampler2d_t::DeviceData>()[blockIdx.x];

        int i = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<float>( layer, i ) );

        auto *_x   = x.DeviceBufferAt<float>( layer );
        auto *_y   = y.DeviceBufferAt<float>( layer );
        auto *_out = out.DeviceBufferAt<float>( layer );

        _out[i] = lTex.Fetch<float>( _x[i], _y[i] );
    }

    CUDA_KERNEL_DEFINITION void Sample2D( multi_tensor_t out, multi_tensor_t x, memory_buffer_t y, memory_buffer_t textures )
    {
        uint32_t                              layer = static_cast<uint32_t>( blockIdx.x );
        cuda::texture_sampler2d_t::DeviceData lTex  = textures.DataAs<cuda::texture_sampler2d_t::DeviceData>()[blockIdx.x];

        int i = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<float>( layer, i ) );

        auto *_x   = x.DeviceBufferAt<float>( layer );
        auto *_out = out.DeviceBufferAt<float>( layer );

        _out[i] = lTex.Fetch<float>( _x[i], y.DataAs<float>()[layer] );
    }

    CUDA_KERNEL_DEFINITION void Sample2D( multi_tensor_t out, multi_tensor_t x, float y, memory_buffer_t textures )
    {
        uint32_t                              layer = static_cast<uint32_t>( blockIdx.x );
        cuda::texture_sampler2d_t::DeviceData lTex  = textures.DataAs<cuda::texture_sampler2d_t::DeviceData>()[blockIdx.x];

        int i = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<float>( layer, i ) );

        auto *_x   = x.DeviceBufferAt<float>( layer );
        auto *_out = out.DeviceBufferAt<float>( layer );

        _out[i] = lTex.Fetch<float>( _x[i], y );
    }

    CUDA_KERNEL_DEFINITION void Sample2D( multi_tensor_t out, memory_buffer_t x, multi_tensor_t y, memory_buffer_t textures )
    {
        uint32_t                              layer = static_cast<uint32_t>( blockIdx.x );
        cuda::texture_sampler2d_t::DeviceData lTex  = textures.DataAs<cuda::texture_sampler2d_t::DeviceData>()[blockIdx.x];

        int i = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( y.Shape().InBounds<float>( layer, i ) );

        auto *_y   = y.DeviceBufferAt<float>( layer );
        auto *_out = out.DeviceBufferAt<float>( layer );

        _out[i] = lTex.Fetch<float>( x.DataAs<float>()[layer], _y[i] );
    }

    CUDA_KERNEL_DEFINITION void Sample2D( multi_tensor_t out, float x, multi_tensor_t y, memory_buffer_t textures )
    {
        uint32_t                              layer = static_cast<uint32_t>( blockIdx.x );
        cuda::texture_sampler2d_t::DeviceData lTex  = textures.DataAs<cuda::texture_sampler2d_t::DeviceData>()[blockIdx.x];

        int i = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( y.Shape().InBounds<float>( layer, i ) );

        auto *_y   = y.DeviceBufferAt<float>( layer );
        auto *_out = out.DeviceBufferAt<float>( layer );

        _out[i] = lTex.Fetch<float>( x, _y[i] );
    }

    template <typename _Ty, typename _OutTy>
    CUDA_KERNEL_DEFINITION void ToFixedPoint( multi_tensor_t out, multi_tensor_t array, _Ty scaling )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( array.Shape().InBounds<_Ty>( layer, i ) );

        _Ty    *_array = array.DeviceBufferAt<_Ty>( layer );
        _OutTy *_out   = out.DeviceBufferAt<_OutTy>( layer );
        _out[i]        = static_cast<_OutTy>( _array[i] * scaling );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AffineTransform( multi_tensor_t out, multi_tensor_t A, multi_tensor_t X, multi_tensor_t B )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( X.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *_A   = A.DeviceBufferAt<_Ty>( layer );
        _Ty *_x   = X.DeviceBufferAt<_Ty>( layer );
        _Ty *_B   = B.DeviceBufferAt<_Ty>( layer );
        _Ty *_out = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = _A[i] * _x[i] + _B[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AffineTransform( multi_tensor_t out, multi_tensor_t A, multi_tensor_t X, memory_buffer_t B )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( X.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *_A   = A.DeviceBufferAt<_Ty>( layer );
        _Ty *_x   = X.DeviceBufferAt<_Ty>( layer );
        _Ty *_out = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = _A[i] * _x[i] + B.DataAs<_Ty>()[layer];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AffineTransform( multi_tensor_t out, multi_tensor_t A, multi_tensor_t X, _Ty B )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( X.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *_A   = A.DeviceBufferAt<_Ty>( layer );
        _Ty *_x   = X.DeviceBufferAt<_Ty>( layer );
        _Ty *_out = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = _A[i] * _x[i] + B;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AffineTransform( multi_tensor_t out, memory_buffer_t A, multi_tensor_t X, multi_tensor_t B )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( X.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *_x   = X.DeviceBufferAt<_Ty>( layer );
        _Ty *_B   = B.DeviceBufferAt<_Ty>( layer );
        _Ty *_out = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = A.DataAs<_Ty>()[layer] * _x[i] + _B[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AffineTransform( multi_tensor_t out, memory_buffer_t A, multi_tensor_t X, memory_buffer_t B )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( X.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *_x   = X.DeviceBufferAt<_Ty>( layer );
        _Ty *_out = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = A.DataAs<_Ty>()[blockIdx.x] * _x[i] + B.DataAs<_Ty>()[blockIdx.x];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AffineTransform( multi_tensor_t out, memory_buffer_t A, multi_tensor_t X, _Ty B )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( X.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *_x   = X.DeviceBufferAt<_Ty>( layer );
        _Ty *_out = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = A.DataAs<_Ty>()[blockIdx.x] * _x[i] + B;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AffineTransform( multi_tensor_t out, _Ty A, multi_tensor_t X, multi_tensor_t B )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( X.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *_x   = X.DeviceBufferAt<_Ty>( layer );
        _Ty *_B   = B.DeviceBufferAt<_Ty>( layer );
        _Ty *_out = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = A * _x[i] + _B[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AffineTransform( multi_tensor_t out, _Ty A, multi_tensor_t X, memory_buffer_t B )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( X.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *_x   = X.DeviceBufferAt<_Ty>( layer );
        _Ty *_out = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = A * _x[i] + B.DataAs<_Ty>()[layer];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AffineTransform( multi_tensor_t out, _Ty A, multi_tensor_t X, _Ty B )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( X.Shape().InBounds<_Ty>( layer, i ) );

        _Ty *_x   = X.DeviceBufferAt<_Ty>( layer );
        _Ty *_out = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = A * _x[i] + B;
    }

    CUDA_KERNEL_DEFINITION void Floor( multi_tensor_t out, multi_tensor_t x )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<float>( layer, i ) );

        auto *_x   = x.DeviceBufferAt<float>( layer );
        auto *_out = out.DeviceBufferAt<float>( layer );

        _out[i] = floor( _x[i] );
    }

    CUDA_KERNEL_DEFINITION void Ceil( multi_tensor_t out, multi_tensor_t x )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<float>( layer, i ) );

        auto *_x   = x.DeviceBufferAt<float>( layer );
        auto *_out = out.DeviceBufferAt<float>( layer );

        _out[i] = ceil( _x[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Sqrt( multi_tensor_t out, multi_tensor_t x )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<_Ty>( layer, i ) );

        auto *_x   = x.DeviceBufferAt<_Ty>( layer );
        auto *_out = out.DeviceBufferAt<float>( layer );

        _out[i] = static_cast<_Ty>( sqrt( static_cast<float>( _x[i] ) ) );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Round( multi_tensor_t out, multi_tensor_t x )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<_Ty>( layer, i ) );

        auto *_x   = x.DeviceBufferAt<_Ty>( layer );
        auto *_out = out.DeviceBufferAt<float>( layer );

        if constexpr( std::is_integral<_Ty>::value )
            _out[i] = _x[i];
        else
            _out[i] = __int2float_rd( __float2int_rn( _x[i] ) );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Abs( multi_tensor_t out, multi_tensor_t x )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( x.Shape().InBounds<float>( layer, i ) );

        auto *_x   = x.DeviceBufferAt<_Ty>( layer );
        auto *_out = out.DeviceBufferAt<_Ty>( layer );

        _out[i] = _x[i] * ( _x[i] >= 0 ? 1.0f : -1.0f );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void CountZero( multi_tensor_t out, multi_tensor_t x, memory_buffer_t blockSizes,
                                           memory_buffer_t elementCount )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( i < blockSizes.DataAs<float>()[layer] );

        auto _element_count = elementCount.DataAs<uint32_t>()[layer];

        auto *_x   = x.DeviceBufferAt<_Ty>( layer ) + i * _element_count;
        auto *_out = out.DeviceBufferAt<uint32_t>( layer ) + i;

        uint32_t count = 0;
        for( uint32_t k = 0; k < _element_count; k++ )
        {
            if( _x[k] == static_cast<_Ty>( 0 ) )
                count++;
        }

        *_out = count;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void CountNonZero( multi_tensor_t out, multi_tensor_t x, memory_buffer_t blockSizes,
                                              memory_buffer_t elementCount )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( i < blockSizes.DataAs<uint32_t>()[layer] );

        auto _element_count = elementCount.DataAs<uint32_t>()[layer];

        auto *_x   = x.DeviceBufferAt<_Ty>( layer ) + i * _element_count;
        auto *_out = out.DeviceBufferAt<uint32_t>( layer ) + i;

        uint32_t count = 0;
        for( uint32_t k = 0; k < _element_count; k++ )
        {
            if( _x[k] != static_cast<_Ty>( 0 ) )
                count++;
        }

        *_out = count;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void ArraySummation( multi_tensor_t out, multi_tensor_t x, memory_buffer_t begin, memory_buffer_t end,
                                                memory_buffer_t elementCount, memory_buffer_t blockSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( i < blockSizes.DataAs<uint32_t>()[layer] );

        auto _begin         = begin.DataAs<uint32_t>()[layer];
        auto _end           = end.DataAs<uint32_t>()[layer];
        auto _element_count = elementCount.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( _begin <= _end ) && ( _end < _element_count ) );

        auto *_x   = x.DeviceBufferAt<_Ty>( layer ) + i * _element_count;
        auto *_out = out.DeviceBufferAt<_Ty>( layer ) + i;

        _Ty _accumulator = 0;
        for( uint32_t k = _begin; k <= _end; k++ )
            _accumulator += _x[k];

        *_out = _accumulator;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void ArraySlice( multi_tensor_t out, multi_tensor_t x, memory_buffer_t begin, memory_buffer_t end,
                                            memory_buffer_t elementCount, memory_buffer_t blockSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( i < blockSizes.DataAs<uint32_t>()[layer] );

        auto _begin         = begin.DataAs<uint32_t>()[layer];
        auto _end           = end.DataAs<uint32_t>()[layer];
        auto _element_count = elementCount.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( _begin <= _end ) && ( _end < _element_count ) );

        auto *_x   = x.DeviceBufferAt<_Ty>( layer ) + i * _element_count;
        auto *_out = out.DeviceBufferAt<_Ty>( layer ) + i * ( _end - _begin + 1 );

        for( uint32_t k = _begin; k <= _end; k++ )
            _out[k - _begin] = _x[k];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Diff( multi_tensor_t out, multi_tensor_t x, uint32_t count, memory_buffer_t elementCount,
                                      memory_buffer_t blockSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( i < blockSizes.DataAs<uint32_t>()[layer] );

        auto _element_count = elementCount.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( count < _element_count ) );

        auto *_x   = x.DeviceBufferAt<_Ty>( layer ) + i * _element_count;
        auto *_out = out.DeviceBufferAt<_Ty>( layer ) + i * _element_count;

        for( uint32_t l = 0; l < _element_count; l++ )
        {
            _out[l] = _x[l];
        }

        for( uint32_t k = 0; k < count; k++ )
        {
            for( uint32_t l = 0; l < _element_count - k; l++ )
            {
                _out[l] = _out[l + 1] - _out[l];
            }
        }

        for( uint32_t k = _element_count - count; k < _element_count; k++ )
        {
            _out[k] = static_cast<_Ty>( 0 );
        }
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void ShiftLeft( multi_tensor_t out, multi_tensor_t x, uint32_t count, _Ty fill_value,
                                           memory_buffer_t elementCount, memory_buffer_t blockSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( i < blockSizes.DataAs<uint32_t>()[layer] );

        auto _element_count = elementCount.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( count < _element_count ) );

        auto *_x   = x.DeviceBufferAt<_Ty>( layer ) + i * _element_count;
        auto *_out = out.DeviceBufferAt<_Ty>( layer ) + i * _element_count;

        for( uint32_t k = 0; k < _element_count - count; k++ )
            _out[k] = _x[k + count];

        for( uint32_t k = _element_count - count; k < _element_count; k++ )
            _out[k] = fill_value;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void ShiftRight( multi_tensor_t out, multi_tensor_t x, uint32_t count, _Ty fill_value,
                                            memory_buffer_t elementCount, memory_buffer_t blockSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( i < blockSizes.DataAs<uint32_t>()[layer] );

        auto _element_count = elementCount.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( count < _element_count ) );

        auto *_x   = x.DeviceBufferAt<_Ty>( layer ) + i * _element_count;
        auto *_out = out.DeviceBufferAt<_Ty>( layer ) + i * _element_count;

        for( uint32_t k = count; k < _element_count; k++ )
            _out[k] = _x[k - count];

        for( uint32_t k = 0; k < count; k++ )
            _out[k] = fill_value;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Conv1D( multi_tensor_t out, multi_tensor_t array0, memory_buffer_t elementCount0,
                                        memory_buffer_t blockSizes0, multi_tensor_t array1, memory_buffer_t elementCount1,
                                        memory_buffer_t blockSizes1 )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < blockSizes0.DataAs<uint32_t>()[layer] );

        auto _element_count_0 = elementCount0.DataAs<uint32_t>()[layer];
        auto _element_count_1 = elementCount1.DataAs<uint32_t>()[layer];

        RETURN_UNLESS( ( i < _element_count_0 ) );

        auto *_x   = array0.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * _element_count_0;
        auto *_K   = array1.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * _element_count_1;
        auto *_out = out.DeviceBufferAt<_Ty>( layer ) + blockIdx.y * _element_count_0;

        _Ty lConvolutionValue = static_cast<_Ty>( 0 );
        for( uint32_t j = 0; j < _element_count_1; j++ )
        {
            if( i >= j )
                lConvolutionValue += ( _x[i - j] * _K[j] );
        }

        _out[i] = lConvolutionValue;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void HCat( multi_tensor_t out, multi_tensor_t x, memory_buffer_t elementCountX, multi_tensor_t y,
                                      memory_buffer_t elementCountY, memory_buffer_t blockSizes )
    {
        uint32_t layer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i     = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( ( i < blockSizes.DataAs<uint32_t>()[layer] ) );

        auto _element_count_x = elementCountX.DataAs<uint32_t>()[layer];
        auto _element_count_y = elementCountY.DataAs<uint32_t>()[layer];

        auto *_x   = x.DeviceBufferAt<_Ty>( layer ) + i * _element_count_x;
        auto *_l   = y.DeviceBufferAt<_Ty>( layer ) + i * _element_count_y;
        auto *_out = out.DeviceBufferAt<_Ty>( layer ) + i * ( _element_count_x + _element_count_y );

        uint32_t k = 0;
        for( uint32_t j = 0; j < _element_count_x; j++ )
            _out[k++] = _x[j];
        for( uint32_t j = 0; j < _element_count_y; j++ )
            _out[k++] = _l[j];
    }
} // namespace numlua::mtops::Kernels