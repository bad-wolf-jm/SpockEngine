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
    CUDA_KERNEL_DEFINITION void ConstantFill( multi_tensor_t aArray, _Ty aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *aOut = aArray.DeviceBufferAt<_Ty>( lLayer );
        aOut[i]   = aConstant;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void ConstantFill( multi_tensor_t aArray, memory_buffer_t aConstants )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *aOut = aArray.DeviceBufferAt<_Ty>( lLayer );

        aOut[i] = aConstants.DataAs<_Ty>()[blockIdx.x];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void ARange( multi_tensor_t aOut, memory_buffer_t aLeft, memory_buffer_t aRight, memory_buffer_t aDelta )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lOutArray = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOutArray[i] = aLeft.DataAs<_Ty>()[blockIdx.x] + i * aDelta.DataAs<_Ty>()[blockIdx.x];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AddArrayToArray( multi_tensor_t aOut, multi_tensor_t aLeft, multi_tensor_t aRight )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aLeft.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lArray    = aLeft.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lConstant = aRight.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut      = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = lArray[i] + lConstant[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AddArrayToArray( multi_tensor_t aOut, multi_tensor_t aLeft, multi_tensor_t aRight,
                                                 eBroadcastHint aBroadcastHint, memory_buffer_t aBlockSizes,
                                                 memory_buffer_t aBroadcastSizes )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < aBlockSizes.DataAs<uint32_t>()[lLayer] );

        auto lBroadcastSize = aBroadcastSizes.DataAs<uint32_t>()[lLayer];

        RETURN_UNLESS( ( i < lBroadcastSize ) );

        auto *lOut = aOut.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y * lBroadcastSize;

        switch( aBroadcastHint )
        {
        case eBroadcastHint::LEFT:
        {
            _Ty *lLeft  = aLeft.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y;
            _Ty *lRight = aRight.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y * lBroadcastSize;

            lOut[i] = lLeft[0] + lRight[i];
        }
        break;
        case eBroadcastHint::RIGHT:
        {
            _Ty *lLeft  = aLeft.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y * lBroadcastSize;
            _Ty *lRight = aRight.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y;

            lOut[i] = lLeft[i] + lRight[0];
        }
        break;
        default:
            break;
        }
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AddScalarToArray( multi_tensor_t aOut, multi_tensor_t aArray, _Ty aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lArray = aArray.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut   = aOut.DeviceBufferAt<_Ty>( lLayer );
        lOut[i]     = lArray[i] + aConstant;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AddArrayToVector( multi_tensor_t aOut, multi_tensor_t aArray, memory_buffer_t aConstants )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lIn  = aArray.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = lIn[i] + aConstants.DataAs<_Ty>()[blockIdx.x];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void MultiplyArrayByArray( multi_tensor_t aOut, multi_tensor_t aArray, multi_tensor_t aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lArray    = aArray.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lConstant = aConstant.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut      = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = lArray[i] * lConstant[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void MultiplyArrayByArray( multi_tensor_t aOut, multi_tensor_t aLeft, multi_tensor_t aRight,
                                                      eBroadcastHint aBroadcastHint, memory_buffer_t aBlockSizes,
                                                      memory_buffer_t aBroadcastSizes )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < aBlockSizes.DataAs<uint32_t>()[lLayer] );

        auto lBroadcastSize = aBroadcastSizes.DataAs<uint32_t>()[lLayer];

        RETURN_UNLESS( ( i < lBroadcastSize ) );

        auto *lOut = aOut.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y * lBroadcastSize;

        switch( aBroadcastHint )
        {
        case eBroadcastHint::LEFT:
        {
            _Ty *lLeft  = aLeft.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y;
            _Ty *lRight = aRight.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y * lBroadcastSize;

            lOut[i] = lLeft[0] * lRight[i];
        }
        break;
        case eBroadcastHint::RIGHT:
        {
            _Ty *lLeft  = aLeft.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y * lBroadcastSize;
            _Ty *lRight = aRight.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y;

            lOut[i] = lLeft[i] * lRight[0];
        }
        break;
        default:
            break;
        }
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void MultiplyScalarByArray( multi_tensor_t aOut, multi_tensor_t aArray, _Ty aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lArray = aArray.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut   = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = lArray[i] * aConstant;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void MultiplyArrayByVector( multi_tensor_t aOut, multi_tensor_t aArray, memory_buffer_t aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lIn  = aArray.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = lIn[i] * aConstant.DataAs<_Ty>()[blockIdx.x];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void SubtractArrayFromScalar( multi_tensor_t aOut, multi_tensor_t aArray, _Ty aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lArray = aArray.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut   = aOut.DeviceBufferAt<_Ty>( lLayer );
        lOut[i]     = aConstant - lArray[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void SubtractScalarFromArray( multi_tensor_t aOut, multi_tensor_t aArray, _Ty aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lArray = aArray.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut   = aOut.DeviceBufferAt<_Ty>( lLayer );
        lOut[i]     = lArray[i] - aConstant;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void SubtractArrayFromArray( multi_tensor_t aOut, multi_tensor_t aArray, multi_tensor_t aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lArray    = aArray.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lConstant = aConstant.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut      = aOut.DeviceBufferAt<_Ty>( lLayer );
        lOut[i]        = lArray[i] - lConstant[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void SubtractArrayFromArray( multi_tensor_t aOut, multi_tensor_t aLeft, multi_tensor_t aRight,
                                                        eBroadcastHint aBroadcastHint, memory_buffer_t aBlockSizes,
                                                        memory_buffer_t aBroadcastSizes )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < aBlockSizes.DataAs<uint32_t>()[lLayer] );

        auto lBroadcastSize = aBroadcastSizes.DataAs<uint32_t>()[lLayer];

        RETURN_UNLESS( ( i < lBroadcastSize ) );

        auto *lOut = aOut.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y * lBroadcastSize;

        switch( aBroadcastHint )
        {
        case eBroadcastHint::LEFT:
        {
            _Ty *lLeft  = aLeft.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y;
            _Ty *lRight = aRight.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y * lBroadcastSize;

            lOut[i] = lLeft[0] - lRight[i];
        }
        break;
        case eBroadcastHint::RIGHT:
        {
            _Ty *lLeft  = aLeft.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y * lBroadcastSize;
            _Ty *lRight = aRight.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y;

            lOut[i] = lLeft[i] - lRight[0];
        }
        break;
        default:
            break;
        }
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void SubtractVectorFromArray( multi_tensor_t aOut, multi_tensor_t aArray, memory_buffer_t aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lIn  = aArray.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = lIn[i] - aConstant.DataAs<_Ty>()[blockIdx.x];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void SubtractArrayFromVector( multi_tensor_t aOut, multi_tensor_t aArray, memory_buffer_t aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lIn  = aArray.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = aConstant.DataAs<_Ty>()[blockIdx.x] - lIn[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void DivideArrayByScalar( multi_tensor_t aOut, multi_tensor_t aArray, _Ty aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        auto *lArray = aArray.DeviceBufferAt<_Ty>( lLayer );
        auto *lOut   = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = lArray[i] / aConstant;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void DivideScalarByArray( multi_tensor_t aOut, multi_tensor_t aArray, _Ty aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<_Ty>( lLayer, i ) );

        auto *lArray = aArray.DeviceBufferAt<_Ty>( lLayer );
        auto *lOut   = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = aConstant / lArray[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void DivideArrayFromArray( multi_tensor_t aOut, multi_tensor_t aArray, multi_tensor_t aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        auto *lArray    = aArray.DeviceBufferAt<_Ty>( lLayer );
        auto *lConstant = aConstant.DeviceBufferAt<_Ty>( lLayer );
        auto *lOut      = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = lArray[i] / lConstant[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void DivideArrayFromArray( multi_tensor_t aOut, multi_tensor_t aLeft, multi_tensor_t aRight,
                                                      eBroadcastHint aBroadcastHint, memory_buffer_t aBlockSizes,
                                                      memory_buffer_t aBroadcastSizes )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < aBlockSizes.DataAs<uint32_t>()[lLayer] );

        auto lBroadcastSize = aBroadcastSizes.DataAs<uint32_t>()[lLayer];

        RETURN_UNLESS( ( i < lBroadcastSize ) );

        auto *lOut = aOut.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y * lBroadcastSize;

        switch( aBroadcastHint )
        {
        case eBroadcastHint::LEFT:
        {
            _Ty *lLeft  = aLeft.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y;
            _Ty *lRight = aRight.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y * lBroadcastSize;

            lOut[i] = lLeft[0] / lRight[i];
        }
        break;
        case eBroadcastHint::RIGHT:
        {
            _Ty *lLeft  = aLeft.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y * lBroadcastSize;
            _Ty *lRight = aRight.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y;

            lOut[i] = lLeft[i] / lRight[0];
        }
        break;
        default:
            break;
        }
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void DivideArrayByVector( multi_tensor_t aOut, multi_tensor_t aArray, memory_buffer_t aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        auto *lIn  = aArray.DeviceBufferAt<_Ty>( lLayer );
        auto *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = lIn[i] / aConstant.DataAs<_Ty>()[blockIdx.x];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void DivideVectorByArray( multi_tensor_t aOut, multi_tensor_t aArray, memory_buffer_t aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        auto *lIn  = aArray.DeviceBufferAt<_Ty>( lLayer );
        auto *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = aConstant.DataAs<_Ty>()[blockIdx.x] / lIn[i];
    }

    CUDA_KERNEL_DEFINITION void AndTensorScalar( multi_tensor_t aOut, multi_tensor_t aArray, uint8_t aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<uint8_t>( lLayer, i ) );

        uint8_t *lArray = aArray.DeviceBufferAt<uint8_t>( lLayer );
        uint8_t *lOut   = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( aConstant && lArray[i] );
    }

    CUDA_KERNEL_DEFINITION void AndTensorTensor( multi_tensor_t aOut, multi_tensor_t aArray, multi_tensor_t aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<uint8_t>( lLayer, i ) );

        uint8_t *lArray    = aArray.DeviceBufferAt<uint8_t>( lLayer );
        uint8_t *lConstant = aConstant.DeviceBufferAt<uint8_t>( lLayer );
        uint8_t *lOut      = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( lArray[i] && lConstant[i] );
    }

    CUDA_KERNEL_DEFINITION void AndTensorTensor( multi_tensor_t aOut, multi_tensor_t aLeft, multi_tensor_t aRight,
                                                 eBroadcastHint aBroadcastHint, memory_buffer_t aBlockSizes,
                                                 memory_buffer_t aBroadcastSizes )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < aBlockSizes.DataAs<uint32_t>()[lLayer] );

        auto lBroadcastSize = aBroadcastSizes.DataAs<uint32_t>()[lLayer];

        RETURN_UNLESS( ( i < lBroadcastSize ) );

        uint8_t *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer ) + blockIdx.y * lBroadcastSize;

        switch( aBroadcastHint )
        {
        case eBroadcastHint::LEFT:
        {
            uint8_t *lLeft  = aLeft.DeviceBufferAt<uint8_t>( lLayer ) + blockIdx.y;
            uint8_t *lRight = aRight.DeviceBufferAt<uint8_t>( lLayer ) + blockIdx.y * lBroadcastSize;

            lOut[i] = ( lLeft[0] && lRight[i] );
        }
        break;
        case eBroadcastHint::RIGHT:
        {
            uint8_t *lLeft  = aLeft.DeviceBufferAt<uint8_t>( lLayer ) + blockIdx.y * lBroadcastSize;
            uint8_t *lRight = aRight.DeviceBufferAt<uint8_t>( lLayer ) + blockIdx.y;

            lOut[i] = ( lLeft[i] && lRight[0] );
        }
        break;
        default:
            break;
        }
    }

    CUDA_KERNEL_DEFINITION void AndTensorVector( multi_tensor_t aOut, multi_tensor_t aArray, memory_buffer_t aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<uint8_t>( lLayer, i ) );

        uint8_t *lIn  = aArray.DeviceBufferAt<uint8_t>( lLayer );
        uint8_t *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( lIn[i] && aConstant.DataAs<uint8_t>()[blockIdx.x] );
    }

    CUDA_KERNEL_DEFINITION void OrTensorScalar( multi_tensor_t aOut, multi_tensor_t aArray, uint8_t aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<uint8_t>( lLayer, i ) );

        uint8_t *lArray = aArray.DeviceBufferAt<uint8_t>( lLayer );
        uint8_t *lOut   = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( aConstant || lArray[i] );
    }

    CUDA_KERNEL_DEFINITION void OrTensorTensor( multi_tensor_t aOut, multi_tensor_t aArray, multi_tensor_t aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<uint8_t>( lLayer, i ) );

        uint8_t *lArray    = aArray.DeviceBufferAt<uint8_t>( lLayer );
        uint8_t *lConstant = aConstant.DeviceBufferAt<uint8_t>( lLayer );
        uint8_t *lOut      = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( lArray[i] || lConstant[i] );
    }

    CUDA_KERNEL_DEFINITION void OrTensorTensor( multi_tensor_t aOut, multi_tensor_t aLeft, multi_tensor_t aRight, eBroadcastHint aBroadcastHint,
                                                memory_buffer_t aBlockSizes, memory_buffer_t aBroadcastSizes )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < aBlockSizes.DataAs<uint32_t>()[lLayer] );

        auto lBroadcastSize = aBroadcastSizes.DataAs<uint32_t>()[lLayer];

        RETURN_UNLESS( ( i < lBroadcastSize ) );

        uint8_t *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer ) + blockIdx.y * lBroadcastSize;

        switch( aBroadcastHint )
        {
        case eBroadcastHint::LEFT:
        {
            uint8_t *lLeft  = aLeft.DeviceBufferAt<uint8_t>( lLayer ) + blockIdx.y;
            uint8_t *lRight = aRight.DeviceBufferAt<uint8_t>( lLayer ) + blockIdx.y * lBroadcastSize;

            lOut[i] = ( lLeft[0] || lRight[i] );
        }
        break;
        case eBroadcastHint::RIGHT:
        {
            uint8_t *lLeft  = aLeft.DeviceBufferAt<uint8_t>( lLayer ) + blockIdx.y * lBroadcastSize;
            uint8_t *lRight = aRight.DeviceBufferAt<uint8_t>( lLayer ) + blockIdx.y;

            lOut[i] = ( lLeft[i] || lRight[0] );
        }
        break;
        default:
            break;
        }
    }

    CUDA_KERNEL_DEFINITION void OrTensorVector( multi_tensor_t aOut, multi_tensor_t aArray, memory_buffer_t aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<uint8_t>( lLayer, i ) );

        uint8_t *lIn  = aArray.DeviceBufferAt<uint8_t>( lLayer );
        uint8_t *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( lIn[i] || aConstant.DataAs<uint8_t>()[blockIdx.x] );
    }

    CUDA_KERNEL_DEFINITION void NotTensor( multi_tensor_t aOut, multi_tensor_t aArray )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<uint8_t>( lLayer, i ) );

        uint8_t *lIn  = aArray.DeviceBufferAt<uint8_t>( lLayer );
        uint8_t *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = !( lIn[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void BitwiseAndTensorScalar( multi_tensor_t aOut, multi_tensor_t aArray, _Ty aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lArray = aArray.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut   = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = ( aConstant & lArray[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void BitwiseAndTensorTensor( multi_tensor_t aOut, multi_tensor_t aLeft, multi_tensor_t aRight,
                                                        eBroadcastHint aBroadcastHint, memory_buffer_t aBlockSizes,
                                                        memory_buffer_t aBroadcastSizes )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < aBlockSizes.DataAs<uint32_t>()[lLayer] );

        auto lBroadcastSize = aBroadcastSizes.DataAs<uint32_t>()[lLayer];

        RETURN_UNLESS( ( i < lBroadcastSize ) );

        auto *lOut = aOut.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y * lBroadcastSize;

        switch( aBroadcastHint )
        {
        case eBroadcastHint::LEFT:
        {
            _Ty *lLeft  = aLeft.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y;
            _Ty *lRight = aRight.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y * lBroadcastSize;

            lOut[i] = lLeft[0] & lRight[i];
        }
        break;
        case eBroadcastHint::RIGHT:
        {
            _Ty *lLeft  = aLeft.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y * lBroadcastSize;
            _Ty *lRight = aRight.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y;

            lOut[i] = lLeft[i] & lRight[0];
        }
        break;
        default:
            break;
        }
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void BitwiseAndTensorTensor( multi_tensor_t aOut, multi_tensor_t aArray, multi_tensor_t aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lArray    = aArray.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lConstant = aConstant.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut      = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = ( lArray[i] & lConstant[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void BitwiseAnd_Tensor_Vector( multi_tensor_t aOut, multi_tensor_t aArray, memory_buffer_t aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lIn  = aArray.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = ( lIn[i] & aConstant.DataAs<_Ty>()[blockIdx.x] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void BitwiseOrTensorScalar( multi_tensor_t aOut, multi_tensor_t aArray, _Ty aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lArray = aArray.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut   = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = ( aConstant | lArray[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void BitwiseOrTensorTensor( multi_tensor_t aOut, multi_tensor_t aArray, multi_tensor_t aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lArray    = aArray.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lConstant = aConstant.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut      = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = ( lArray[i] | lConstant[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void BitwiseOrTensorTensor( multi_tensor_t aOut, multi_tensor_t aLeft, multi_tensor_t aRight,
                                                       eBroadcastHint aBroadcastHint, memory_buffer_t aBlockSizes,
                                                       memory_buffer_t aBroadcastSizes )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < aBlockSizes.DataAs<uint32_t>()[lLayer] );

        auto lBroadcastSize = aBroadcastSizes.DataAs<uint32_t>()[lLayer];

        RETURN_UNLESS( ( i < lBroadcastSize ) );

        auto *lOut = aOut.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y * lBroadcastSize;

        switch( aBroadcastHint )
        {
        case eBroadcastHint::LEFT:
        {
            _Ty *lLeft  = aLeft.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y;
            _Ty *lRight = aRight.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y * lBroadcastSize;

            lOut[i] = lLeft[0] | lRight[i];
        }
        break;
        case eBroadcastHint::RIGHT:
        {
            _Ty *lLeft  = aLeft.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y * lBroadcastSize;
            _Ty *lRight = aRight.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y;

            lOut[i] = lLeft[i] | lRight[0];
        }
        break;
        default:
            break;
        }
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void BitwiseOrTensorVector( multi_tensor_t aOut, multi_tensor_t aArray, memory_buffer_t aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lIn  = aArray.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = ( lIn[i] | aConstant.DataAs<_Ty>()[blockIdx.x] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void BitwiseNotTensor( multi_tensor_t aOut, multi_tensor_t aArray )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lIn  = aArray.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = ~( lIn[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void InIntervalTensorTensor( multi_tensor_t aOut, multi_tensor_t aX, multi_tensor_t aLower, multi_tensor_t aUpper,
                                                        bool aStrictLower, bool aStrictUpper )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aX.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty     *lX     = aX.DeviceBufferAt<_Ty>( lLayer );
        _Ty     *lLower = aLower.DeviceBufferAt<_Ty>( lLayer );
        _Ty     *lUpper = aUpper.DeviceBufferAt<_Ty>( lLayer );
        uint8_t *lOut   = aOut.DeviceBufferAt<uint8_t>( lLayer );

        bool lComp0 = aStrictLower ? ( lLower[i] < lX[i] ) : ( lLower[i] <= lX[i] );
        bool lComp1 = aStrictUpper ? ( lUpper[i] > lX[i] ) : ( lUpper[i] >= lX[i] );

        lOut[i] = ( lComp0 && lComp1 );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void InIntervalTensorVector( multi_tensor_t aOut, multi_tensor_t aX, multi_tensor_t aLower, memory_buffer_t aUpper,
                                                        bool aStrictLower, bool aStrictUpper )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aX.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty     *lX     = aX.DeviceBufferAt<_Ty>( lLayer );
        _Ty     *lLower = aLower.DeviceBufferAt<_Ty>( lLayer );
        _Ty     *lUpper = aUpper.DataAs<_Ty>();
        uint8_t *lOut   = aOut.DeviceBufferAt<uint8_t>( lLayer );

        bool lComp0 = aStrictLower ? ( lLower[i] < lX[i] ) : ( lLower[i] <= lX[i] );
        bool lComp1 = aStrictUpper ? ( lUpper[lLayer] > lX[i] ) : ( lUpper[lLayer] >= lX[i] );

        lOut[i] = ( lComp0 && lComp1 );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void InIntervalTensorScalar( multi_tensor_t aOut, multi_tensor_t aX, multi_tensor_t aLower, _Ty aUpper,
                                                        bool aStrictLower, bool aStrictUpper )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aX.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty     *lX     = aX.DeviceBufferAt<_Ty>( lLayer );
        _Ty     *lLower = aLower.DeviceBufferAt<_Ty>( lLayer );
        uint8_t *lOut   = aOut.DeviceBufferAt<uint8_t>( lLayer );

        bool lComp0 = aStrictLower ? ( lLower[i] < lX[i] ) : ( lLower[i] <= lX[i] );
        bool lComp1 = aStrictUpper ? ( aUpper > lX[i] ) : ( aUpper >= lX[i] );

        lOut[i] = ( lComp0 && lComp1 );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void InIntervalVectorTensor( multi_tensor_t aOut, multi_tensor_t aX, memory_buffer_t aLower, multi_tensor_t aUpper,
                                                        bool aStrictLower, bool aStrictUpper )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aX.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty     *lX     = aX.DeviceBufferAt<_Ty>( lLayer );
        _Ty     *lLower = aLower.DataAs<_Ty>();
        _Ty     *lUpper = aUpper.DeviceBufferAt<_Ty>( lLayer );
        uint8_t *lOut   = aOut.DeviceBufferAt<uint8_t>( lLayer );

        bool lComp0 = aStrictLower ? ( lLower[lLayer] < lX[i] ) : ( lLower[lLayer] <= lX[i] );
        bool lComp1 = aStrictUpper ? ( lUpper[i] > lX[i] ) : ( lUpper[i] >= lX[i] );

        lOut[i] = ( lComp0 && lComp1 );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void InIntervalVectorVector( multi_tensor_t aOut, multi_tensor_t aX, memory_buffer_t aLower, memory_buffer_t aUpper,
                                                        bool aStrictLower, bool aStrictUpper )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aX.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty     *lX     = aX.DeviceBufferAt<_Ty>( lLayer );
        _Ty     *lLower = aLower.DataAs<_Ty>();
        _Ty     *lUpper = aUpper.DataAs<_Ty>();
        uint8_t *lOut   = aOut.DeviceBufferAt<uint8_t>( lLayer );

        bool lComp0 = aStrictLower ? ( lLower[lLayer] < lX[i] ) : ( lLower[lLayer] <= lX[i] );
        bool lComp1 = aStrictUpper ? ( lUpper[lLayer] > lX[i] ) : ( lUpper[lLayer] >= lX[i] );

        lOut[i] = ( lComp0 && lComp1 );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void InIntervalVectorScalar( multi_tensor_t aOut, multi_tensor_t aX, memory_buffer_t aLower, _Ty aUpper,
                                                        bool aStrictLower, bool aStrictUpper )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aX.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty     *lX     = aX.DeviceBufferAt<_Ty>( lLayer );
        _Ty     *lLower = aLower.DataAs<_Ty>();
        uint8_t *lOut   = aOut.DeviceBufferAt<uint8_t>( lLayer );

        bool lComp0 = aStrictLower ? ( lLower[lLayer] < lX[i] ) : ( lLower[lLayer] <= lX[i] );
        bool lComp1 = aStrictUpper ? ( aUpper > lX[i] ) : ( aUpper >= lX[i] );

        lOut[i] = ( lComp0 && lComp1 );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void InIntervalScalarTensor( multi_tensor_t aOut, multi_tensor_t aX, _Ty aLower, multi_tensor_t aUpper,
                                                        bool aStrictLower, bool aStrictUpper )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aX.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty     *lX     = aX.DeviceBufferAt<_Ty>( lLayer );
        _Ty     *lUpper = aUpper.DeviceBufferAt<_Ty>( lLayer );
        uint8_t *lOut   = aOut.DeviceBufferAt<uint8_t>( lLayer );

        bool lComp0 = aStrictLower ? ( aLower < lX[i] ) : ( aLower <= lX[i] );
        bool lComp1 = aStrictUpper ? ( lUpper[i] > lX[i] ) : ( lUpper[i] >= lX[i] );

        lOut[i] = ( lComp0 && lComp1 );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void InIntervalScalarVector( multi_tensor_t aOut, multi_tensor_t aX, _Ty aLower, memory_buffer_t aUpper,
                                                        bool aStrictLower, bool aStrictUpper )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aX.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty     *lX     = aX.DeviceBufferAt<_Ty>( lLayer );
        _Ty     *lUpper = aUpper.DataAs<_Ty>();
        uint8_t *lOut   = aOut.DeviceBufferAt<uint8_t>( lLayer );

        bool lComp0 = aStrictLower ? ( aLower < lX[i] ) : ( aLower <= lX[i] );
        bool lComp1 = aStrictUpper ? ( lUpper[lLayer] > lX[i] ) : ( lUpper[lLayer] >= lX[i] );

        lOut[i] = ( lComp0 && lComp1 );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void InIntervalScalarScalar( multi_tensor_t aOut, multi_tensor_t aX, _Ty aLower, _Ty aUpper, bool aStrictLower,
                                                        bool aStrictUpper )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aX.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty     *lX   = aX.DeviceBufferAt<_Ty>( lLayer );
        uint8_t *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer );

        bool lComp0 = aStrictLower ? ( aLower < lX[i] ) : ( aLower <= lX[i] );
        bool lComp1 = aStrictUpper ? ( aUpper > lX[i] ) : ( aUpper >= lX[i] );

        lOut[i] = ( lComp0 && lComp1 );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void EqualOp( multi_tensor_t aOut, multi_tensor_t aX, multi_tensor_t aY )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<uint8_t>( lLayer, i ) );

        _Ty     *lX   = aX.DeviceBufferAt<_Ty>( lLayer );
        _Ty     *lY   = aY.DeviceBufferAt<_Ty>( lLayer );
        uint8_t *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( lX[i] == lY[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void EqualOp( multi_tensor_t aOut, multi_tensor_t aX, memory_buffer_t aY )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<uint8_t>( lLayer, i ) );

        _Ty     *lX   = aX.DeviceBufferAt<_Ty>( lLayer );
        _Ty     *lY   = aY.DataAs<_Ty>();
        uint8_t *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( lX[i] == lY[lLayer] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void EqualOp( multi_tensor_t aOut, multi_tensor_t aX, _Ty aY )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<uint8_t>( lLayer, i ) );

        _Ty     *lX   = aX.DeviceBufferAt<_Ty>( lLayer );
        uint8_t *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( lX[i] == aY );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void EqualOp( multi_tensor_t aOut, multi_tensor_t aLeft, multi_tensor_t aRight, eBroadcastHint aBroadcastHint,
                                         memory_buffer_t aBlockSizes, memory_buffer_t aBroadcastSizes )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < aBlockSizes.DataAs<uint32_t>()[lLayer] );

        auto lBroadcastSize = aBroadcastSizes.DataAs<uint32_t>()[lLayer];

        RETURN_UNLESS( ( i < lBroadcastSize ) );

        auto *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer ) + blockIdx.y * lBroadcastSize;

        switch( aBroadcastHint )
        {
        case eBroadcastHint::LEFT:
        {
            _Ty *lLeft  = aLeft.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y;
            _Ty *lRight = aRight.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y * lBroadcastSize;

            lOut[i] = lLeft[0] == lRight[i];
        }
        break;
        case eBroadcastHint::RIGHT:
        {
            _Ty *lLeft  = aLeft.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y * lBroadcastSize;
            _Ty *lRight = aRight.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y;

            lOut[i] = lLeft[i] == lRight[0];
        }
        break;
        default:
            break;
        }
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void EqualOp( multi_tensor_t aOut, memory_buffer_t aX, multi_tensor_t aY )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<uint8_t>( lLayer, i ) );

        _Ty     *lX   = aX.DataAs<_Ty>();
        _Ty     *lY   = aY.DeviceBufferAt<_Ty>( lLayer );
        uint8_t *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( lX[lLayer] == lY[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void EqualOp( multi_tensor_t aOut, _Ty aX, multi_tensor_t aY )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<uint8_t>( lLayer, i ) );

        _Ty     *lY   = aY.DeviceBufferAt<_Ty>( lLayer );
        uint8_t *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( aX == lY[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOp( multi_tensor_t aOut, multi_tensor_t aX, multi_tensor_t aY )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<uint8_t>( lLayer, i ) );

        _Ty     *lX   = aX.DeviceBufferAt<_Ty>( lLayer );
        _Ty     *lY   = aY.DeviceBufferAt<_Ty>( lLayer );
        uint8_t *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( lX[i] < lY[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOp( multi_tensor_t aOut, multi_tensor_t aLeft, multi_tensor_t aRight, eBroadcastHint aBroadcastHint,
                                            memory_buffer_t aBlockSizes, memory_buffer_t aBroadcastSizes )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < aBlockSizes.DataAs<uint32_t>()[lLayer] );

        auto lBroadcastSize = aBroadcastSizes.DataAs<uint32_t>()[lLayer];

        RETURN_UNLESS( ( i < lBroadcastSize ) );

        auto *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer ) + blockIdx.y * lBroadcastSize;

        switch( aBroadcastHint )
        {
        case eBroadcastHint::LEFT:
        {
            _Ty *lLeft  = aLeft.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y;
            _Ty *lRight = aRight.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y * lBroadcastSize;

            lOut[i] = lLeft[0] < lRight[i];
        }
        break;
        case eBroadcastHint::RIGHT:
        {
            _Ty *lLeft  = aLeft.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y * lBroadcastSize;
            _Ty *lRight = aRight.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y;

            lOut[i] = lLeft[i] < lRight[0];
        }
        break;
        default:
            break;
        }
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOp( multi_tensor_t aOut, multi_tensor_t aX, memory_buffer_t aY )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<uint8_t>( lLayer, i ) );

        _Ty     *lX   = aX.DeviceBufferAt<_Ty>( lLayer );
        _Ty     *lY   = aY.DataAs<_Ty>();
        uint8_t *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( lX[i] < lY[lLayer] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOp( multi_tensor_t aOut, multi_tensor_t aX, _Ty aY )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<uint8_t>( lLayer, i ) );

        _Ty     *lX   = aX.DeviceBufferAt<_Ty>( lLayer );
        uint8_t *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( lX[i] < aY );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOp( multi_tensor_t aOut, memory_buffer_t aX, multi_tensor_t aY )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<uint8_t>( lLayer, i ) );

        _Ty     *lX   = aX.DataAs<_Ty>();
        _Ty     *lY   = aY.DeviceBufferAt<_Ty>( lLayer );
        uint8_t *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( lX[lLayer] < lY[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOp( multi_tensor_t aOut, _Ty aX, multi_tensor_t aY )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<uint8_t>( lLayer, i ) );

        _Ty     *lY   = aY.DeviceBufferAt<_Ty>( lLayer );
        uint8_t *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( aX < lY[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOrEqualOp( multi_tensor_t aOut, multi_tensor_t aX, multi_tensor_t aY )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<uint8_t>( lLayer, i ) );

        _Ty     *lX   = aX.DeviceBufferAt<_Ty>( lLayer );
        _Ty     *lY   = aY.DeviceBufferAt<_Ty>( lLayer );
        uint8_t *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( lX[i] <= lY[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOrEqualOp( multi_tensor_t aOut, multi_tensor_t aLeft, multi_tensor_t aRight,
                                                   eBroadcastHint aBroadcastHint, memory_buffer_t aBlockSizes,
                                                   memory_buffer_t aBroadcastSizes )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < aBlockSizes.DataAs<uint32_t>()[lLayer] );

        auto lBroadcastSize = aBroadcastSizes.DataAs<uint32_t>()[lLayer];

        RETURN_UNLESS( ( i < lBroadcastSize ) );

        auto *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer ) + blockIdx.y * lBroadcastSize;

        switch( aBroadcastHint )
        {
        case eBroadcastHint::LEFT:
        {
            _Ty *lLeft  = aLeft.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y;
            _Ty *lRight = aRight.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y * lBroadcastSize;

            lOut[i] = lLeft[0] <= lRight[i];
        }
        break;
        case eBroadcastHint::RIGHT:
        {
            _Ty *lLeft  = aLeft.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y * lBroadcastSize;
            _Ty *lRight = aRight.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y;

            lOut[i] = lLeft[i] <= lRight[0];
        }
        break;
        default:
            break;
        }
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOrEqualOp( multi_tensor_t aOut, multi_tensor_t aX, memory_buffer_t aY )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<uint8_t>( lLayer, i ) );

        _Ty     *lX   = aX.DeviceBufferAt<_Ty>( lLayer );
        _Ty     *lY   = aY.DataAs<_Ty>();
        uint8_t *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( lX[i] <= lY[lLayer] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOrEqualOp( multi_tensor_t aOut, multi_tensor_t aX, _Ty aY )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<uint8_t>( lLayer, i ) );

        _Ty     *lX   = aX.DeviceBufferAt<_Ty>( lLayer );
        uint8_t *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( lX[i] <= aY );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOrEqualOp( multi_tensor_t aOut, memory_buffer_t aX, multi_tensor_t aY )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<uint8_t>( lLayer, i ) );

        _Ty     *lX   = aX.DataAs<_Ty>();
        _Ty     *lY   = aY.DeviceBufferAt<_Ty>( lLayer );
        uint8_t *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( lX[lLayer] <= lY[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOrEqualOp( multi_tensor_t aOut, _Ty aX, multi_tensor_t aY )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<uint8_t>( lLayer, i ) );

        _Ty     *lY   = aY.DeviceBufferAt<_Ty>( lLayer );
        uint8_t *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( aX <= lY[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void WhereTensorTensor( multi_tensor_t aOut, multi_tensor_t aCondition, multi_tensor_t aValueIfTrue,
                                                   multi_tensor_t aValueIfFalse )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aCondition.Shape().InBounds<uint8_t>( lLayer, i ) );

        uint8_t *lCondition    = aCondition.DeviceBufferAt<uint8_t>( lLayer );
        _Ty     *lValueIfTrue  = aValueIfTrue.DeviceBufferAt<_Ty>( lLayer );
        _Ty     *lValueIfFalse = aValueIfFalse.DeviceBufferAt<_Ty>( lLayer );
        _Ty     *lOut          = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = lCondition[i] ? lValueIfTrue[i] : lValueIfFalse[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void WhereTensorVector( multi_tensor_t aOut, multi_tensor_t aCondition, multi_tensor_t aValueIfTrue,
                                                   memory_buffer_t aValueIfFalse )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aCondition.Shape().InBounds<uint8_t>( lLayer, i ) );

        uint8_t *lCondition    = aCondition.DeviceBufferAt<uint8_t>( lLayer );
        _Ty     *lValueIfTrue  = aValueIfTrue.DeviceBufferAt<_Ty>( lLayer );
        _Ty     *lValueIfFalse = aValueIfFalse.DataAs<_Ty>();
        _Ty     *lOut          = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = lCondition[i] ? lValueIfTrue[i] : lValueIfFalse[lLayer];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void WhereTensorScalar( multi_tensor_t aOut, multi_tensor_t aCondition, multi_tensor_t aValueIfTrue,
                                                   _Ty aValueIfFalse )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aCondition.Shape().InBounds<uint8_t>( lLayer, i ) );

        uint8_t *lCondition   = aCondition.DeviceBufferAt<uint8_t>( lLayer );
        _Ty     *lValueIfTrue = aValueIfTrue.DeviceBufferAt<_Ty>( lLayer );
        _Ty     *lOut         = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = lCondition[i] ? lValueIfTrue[i] : aValueIfFalse;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void WhereVectorTensor( multi_tensor_t aOut, multi_tensor_t aCondition, memory_buffer_t aValueIfTrue,
                                                   multi_tensor_t aValueIfFalse )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aCondition.Shape().InBounds<uint8_t>( lLayer, i ) );

        uint8_t *lCondition    = aCondition.DeviceBufferAt<uint8_t>( lLayer );
        _Ty     *lValueIfTrue  = aValueIfTrue.DataAs<_Ty>();
        _Ty     *lValueIfFalse = aValueIfFalse.DeviceBufferAt<_Ty>( lLayer );
        _Ty     *lOut          = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = lCondition[i] ? lValueIfTrue[lLayer] : lValueIfFalse[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void WhereVectorVector( multi_tensor_t aOut, multi_tensor_t aCondition, memory_buffer_t aValueIfTrue,
                                                   memory_buffer_t aValueIfFalse )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aCondition.Shape().InBounds<uint8_t>( lLayer, i ) );

        uint8_t *lCondition    = aCondition.DeviceBufferAt<uint8_t>( lLayer );
        _Ty     *lValueIfTrue  = aValueIfTrue.DataAs<_Ty>();
        _Ty     *lValueIfFalse = aValueIfFalse.DataAs<_Ty>();
        _Ty     *lOut          = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = lCondition[i] ? lValueIfTrue[lLayer] : lValueIfFalse[lLayer];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void WhereVectorScalar( multi_tensor_t aOut, multi_tensor_t aCondition, memory_buffer_t aValueIfTrue,
                                                   _Ty aValueIfFalse )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aCondition.Shape().InBounds<uint8_t>( lLayer, i ) );

        uint8_t *lCondition   = aCondition.DeviceBufferAt<uint8_t>( lLayer );
        _Ty     *lValueIfTrue = aValueIfTrue.DataAs<_Ty>();
        _Ty     *lOut         = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = lCondition[i] ? lValueIfTrue[lLayer] : aValueIfFalse;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void WhereScalarTensor( multi_tensor_t aOut, multi_tensor_t aCondition, _Ty aValueIfTrue,
                                                   multi_tensor_t aValueIfFalse )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aCondition.Shape().InBounds<uint8_t>( lLayer, i ) );

        uint8_t *lCondition    = aCondition.DeviceBufferAt<uint8_t>( lLayer );
        _Ty     *lValueIfFalse = aValueIfFalse.DeviceBufferAt<_Ty>( lLayer );
        _Ty     *lOut          = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = lCondition[i] ? aValueIfTrue : lValueIfFalse[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void WhereScalarVector( multi_tensor_t aOut, multi_tensor_t aCondition, _Ty aValueIfTrue,
                                                   memory_buffer_t aValueIfFalse )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aCondition.Shape().InBounds<uint8_t>( lLayer, i ) );

        uint8_t *lCondition    = aCondition.DeviceBufferAt<uint8_t>( lLayer );
        _Ty     *lValueIfFalse = aValueIfFalse.DataAs<_Ty>();
        _Ty     *lOut          = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = lCondition[i] ? aValueIfTrue : lValueIfFalse[lLayer];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void WhereScalarScalar( multi_tensor_t aOut, multi_tensor_t aCondition, _Ty aValueIfTrue, _Ty aValueIfFalse )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aCondition.Shape().InBounds<uint8_t>( lLayer, i ) );

        uint8_t *lCondition = aCondition.DeviceBufferAt<uint8_t>( lLayer );
        _Ty     *lOut       = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = lCondition[i] ? aValueIfTrue : aValueIfFalse;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Repeat( multi_tensor_t aOut, multi_tensor_t aArray, memory_buffer_t aRepetitions )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        uint32_t N      = aRepetitions.DataAs<uint32_t>()[blockIdx.x];

        RETURN_UNLESS( aArray.Shape().InBounds<uint8_t>( lLayer, blockIdx.y ) );

        int i = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;
        int j = blockIdx.y;

        RETURN_UNLESS( ( i < N ) && ( aArray.Shape().InBounds<_Ty>( lLayer, j ) ) );

        _Ty *lInArray  = aArray.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOutArray = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOutArray[blockIdx.y * N + i] = lInArray[j];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Tile( multi_tensor_t aOut, multi_tensor_t aArray, memory_buffer_t aRepetitions )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        uint32_t N      = aRepetitions.DataAs<uint32_t>()[blockIdx.x];

        RETURN_UNLESS( blockIdx.y < N );

        int i = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;
        int j = blockIdx.y * aArray.Shape().GetBufferSizeAs<_Ty>( lLayer ).mSize + i;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) && aOut.Shape().InBounds<_Ty>( lLayer, j ) );

        _Ty *lInArray  = aArray.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOutArray = aOut.DeviceBufferAt<_Ty>( lLayer );
        lOutArray[j]   = lInArray[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LinearSpace( multi_tensor_t aOut, multi_tensor_t aLeft, multi_tensor_t aRight, memory_buffer_t aSubdivisions )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        uint32_t N      = aSubdivisions.DataAs<uint32_t>()[blockIdx.x];

        RETURN_UNLESS( aLeft.Shape().InBounds<_Ty>( lLayer, blockIdx.y ) );

        int i = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;
        int j = blockIdx.y;
        int k = blockIdx.y * N + i;

        RETURN_UNLESS( i < N );

        _Ty *lInArrayA = aLeft.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lInArrayB = aRight.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOutArray = aOut.DeviceBufferAt<_Ty>( lLayer );

        float aDelta = ( lInArrayB[j] - lInArrayA[j] ) / static_cast<float>( N );
        lOutArray[k] = lInArrayA[j] + i * aDelta;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Mix( multi_tensor_t aOut, multi_tensor_t A, multi_tensor_t B, multi_tensor_t t )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( A.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lA   = A.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lB   = B.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lT   = t.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = ( 1 - lT[i] ) * lA[i] + lT[i] * lB[i];
    }

    CUDA_KERNEL_DEFINITION void Sample2D( multi_tensor_t aOut, multi_tensor_t aX, multi_tensor_t aY, memory_buffer_t aTextures )
    {
        uint32_t                           lLayer = static_cast<uint32_t>( blockIdx.x );
        Cuda::TextureSampler2D::DeviceData lTex   = aTextures.DataAs<Cuda::TextureSampler2D::DeviceData>()[blockIdx.x];

        int i = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aX.Shape().InBounds<float>( lLayer, i ) );

        auto *lXArray = aX.DeviceBufferAt<float>( lLayer );
        auto *lYArray = aY.DeviceBufferAt<float>( lLayer );
        auto *lOut    = aOut.DeviceBufferAt<float>( lLayer );

        lOut[i] = lTex.Fetch<float>( lXArray[i], lYArray[i] );
    }

    CUDA_KERNEL_DEFINITION void Sample2D( multi_tensor_t aOut, multi_tensor_t aX, memory_buffer_t aY, memory_buffer_t aTextures )
    {
        uint32_t                           lLayer = static_cast<uint32_t>( blockIdx.x );
        Cuda::TextureSampler2D::DeviceData lTex   = aTextures.DataAs<Cuda::TextureSampler2D::DeviceData>()[blockIdx.x];

        int i = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aX.Shape().InBounds<float>( lLayer, i ) );

        auto *lXArray = aX.DeviceBufferAt<float>( lLayer );
        auto *lOut    = aOut.DeviceBufferAt<float>( lLayer );

        lOut[i] = lTex.Fetch<float>( lXArray[i], aY.DataAs<float>()[lLayer] );
    }

    CUDA_KERNEL_DEFINITION void Sample2D( multi_tensor_t aOut, multi_tensor_t aX, float aY, memory_buffer_t aTextures )
    {
        uint32_t                           lLayer = static_cast<uint32_t>( blockIdx.x );
        Cuda::TextureSampler2D::DeviceData lTex   = aTextures.DataAs<Cuda::TextureSampler2D::DeviceData>()[blockIdx.x];

        int i = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aX.Shape().InBounds<float>( lLayer, i ) );

        auto *lXArray = aX.DeviceBufferAt<float>( lLayer );
        auto *lOut    = aOut.DeviceBufferAt<float>( lLayer );

        lOut[i] = lTex.Fetch<float>( lXArray[i], aY );
    }

    CUDA_KERNEL_DEFINITION void Sample2D( multi_tensor_t aOut, memory_buffer_t aX, multi_tensor_t aY, memory_buffer_t aTextures )
    {
        uint32_t                           lLayer = static_cast<uint32_t>( blockIdx.x );
        Cuda::TextureSampler2D::DeviceData lTex   = aTextures.DataAs<Cuda::TextureSampler2D::DeviceData>()[blockIdx.x];

        int i = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aY.Shape().InBounds<float>( lLayer, i ) );

        auto *lYArray = aY.DeviceBufferAt<float>( lLayer );
        auto *lOut    = aOut.DeviceBufferAt<float>( lLayer );

        lOut[i] = lTex.Fetch<float>( aX.DataAs<float>()[lLayer], lYArray[i] );
    }

    CUDA_KERNEL_DEFINITION void Sample2D( multi_tensor_t aOut, float aX, multi_tensor_t aY, memory_buffer_t aTextures )
    {
        uint32_t                           lLayer = static_cast<uint32_t>( blockIdx.x );
        Cuda::TextureSampler2D::DeviceData lTex   = aTextures.DataAs<Cuda::TextureSampler2D::DeviceData>()[blockIdx.x];

        int i = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aY.Shape().InBounds<float>( lLayer, i ) );

        auto *lYArray = aY.DeviceBufferAt<float>( lLayer );
        auto *lOut    = aOut.DeviceBufferAt<float>( lLayer );

        lOut[i] = lTex.Fetch<float>( aX, lYArray[i] );
    }

    template <typename _Ty, typename _OutTy>
    CUDA_KERNEL_DEFINITION void ToFixedPoint( multi_tensor_t aOut, multi_tensor_t aArray, _Ty aScaling )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty    *lInBuffer  = aArray.DeviceBufferAt<_Ty>( lLayer );
        _OutTy *lOutBuffer = aOut.DeviceBufferAt<_OutTy>( lLayer );
        lOutBuffer[i]      = static_cast<_OutTy>( lInBuffer[i] * aScaling );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AffineTransform( multi_tensor_t aOut, multi_tensor_t A, multi_tensor_t X, multi_tensor_t B )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( X.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lA   = A.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lX   = X.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lB   = B.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = lA[i] * lX[i] + lB[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AffineTransform( multi_tensor_t aOut, multi_tensor_t A, multi_tensor_t X, memory_buffer_t B )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( X.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lA   = A.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lX   = X.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = lA[i] * lX[i] + B.DataAs<_Ty>()[lLayer];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AffineTransform( multi_tensor_t aOut, multi_tensor_t A, multi_tensor_t X, _Ty B )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( X.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lA   = A.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lX   = X.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = lA[i] * lX[i] + B;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AffineTransform( multi_tensor_t aOut, memory_buffer_t A, multi_tensor_t X, multi_tensor_t B )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( X.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lX   = X.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lB   = B.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = A.DataAs<_Ty>()[lLayer] * lX[i] + lB[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AffineTransform( multi_tensor_t aOut, memory_buffer_t A, multi_tensor_t X, memory_buffer_t B )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( X.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lX   = X.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = A.DataAs<_Ty>()[blockIdx.x] * lX[i] + B.DataAs<_Ty>()[blockIdx.x];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AffineTransform( multi_tensor_t aOut, memory_buffer_t A, multi_tensor_t X, _Ty B )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( X.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lX   = X.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = A.DataAs<_Ty>()[blockIdx.x] * lX[i] + B;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AffineTransform( multi_tensor_t aOut, _Ty A, multi_tensor_t X, multi_tensor_t B )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( X.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lX   = X.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lB   = B.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = A * lX[i] + lB[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AffineTransform( multi_tensor_t aOut, _Ty A, multi_tensor_t X, memory_buffer_t B )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( X.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lX   = X.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = A * lX[i] + B.DataAs<_Ty>()[lLayer];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AffineTransform( multi_tensor_t aOut, _Ty A, multi_tensor_t X, _Ty B )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( X.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lX   = X.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = A * lX[i] + B;
    }

    CUDA_KERNEL_DEFINITION void Floor( multi_tensor_t aOut, multi_tensor_t aX )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aX.Shape().InBounds<float>( lLayer, i ) );

        auto *lX   = aX.DeviceBufferAt<float>( lLayer );
        auto *lOut = aOut.DeviceBufferAt<float>( lLayer );

        lOut[i] = floor( lX[i] );
    }

    CUDA_KERNEL_DEFINITION void Ceil( multi_tensor_t aOut, multi_tensor_t aX )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aX.Shape().InBounds<float>( lLayer, i ) );

        auto *lX   = aX.DeviceBufferAt<float>( lLayer );
        auto *lOut = aOut.DeviceBufferAt<float>( lLayer );

        lOut[i] = ceil( lX[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Sqrt( multi_tensor_t aOut, multi_tensor_t aX )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aX.Shape().InBounds<_Ty>( lLayer, i ) );

        auto *lX   = aX.DeviceBufferAt<_Ty>( lLayer );
        auto *lOut = aOut.DeviceBufferAt<float>( lLayer );

        lOut[i] = static_cast<_Ty>( sqrt( static_cast<float>( lX[i] ) ) );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Round( multi_tensor_t aOut, multi_tensor_t aX )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aX.Shape().InBounds<_Ty>( lLayer, i ) );

        auto *lX   = aX.DeviceBufferAt<_Ty>( lLayer );
        auto *lOut = aOut.DeviceBufferAt<float>( lLayer );

        if constexpr( std::is_integral<_Ty>::value )
            lOut[i] = lX[i];
        else
            lOut[i] = __int2float_rd( __float2int_rn( lX[i] ) );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Abs( multi_tensor_t aOut, multi_tensor_t aX )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aX.Shape().InBounds<float>( lLayer, i ) );

        auto *lX   = aX.DeviceBufferAt<_Ty>( lLayer );
        auto *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = lX[i] * ( lX[i] >= 0 ? 1.0f : -1.0f );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void CountZero( multi_tensor_t aOut, multi_tensor_t aX, memory_buffer_t aBlockSizes, memory_buffer_t aElementCount )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( i < aBlockSizes.DataAs<float>()[lLayer] );

        auto lElementCount = aElementCount.DataAs<uint32_t>()[lLayer];

        auto *lX   = aX.DeviceBufferAt<_Ty>( lLayer ) + i * lElementCount;
        auto *lOut = aOut.DeviceBufferAt<uint32_t>( lLayer ) + i;

        uint32_t lCount = 0;
        for( uint32_t k = 0; k < lElementCount; k++ )
        {
            if( lX[k] == static_cast<_Ty>( 0 ) )
                lCount++;
        }

        *lOut = lCount;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void CountNonZero( multi_tensor_t aOut, multi_tensor_t aX, memory_buffer_t aBlockSizes, memory_buffer_t aElementCount )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( i < aBlockSizes.DataAs<uint32_t>()[lLayer] );

        auto lElementCount = aElementCount.DataAs<uint32_t>()[lLayer];

        auto *lX   = aX.DeviceBufferAt<_Ty>( lLayer ) + i * lElementCount;
        auto *lOut = aOut.DeviceBufferAt<uint32_t>( lLayer ) + i;

        uint32_t lCount = 0;
        for( uint32_t k = 0; k < lElementCount; k++ )
        {
            if( lX[k] != static_cast<_Ty>( 0 ) )
                lCount++;
        }

        *lOut = lCount;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void ArraySummation( multi_tensor_t aOut, multi_tensor_t aX, memory_buffer_t aBegin, memory_buffer_t aEnd,
                                                memory_buffer_t aElementCount, memory_buffer_t aBlockSizes )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( i < aBlockSizes.DataAs<uint32_t>()[lLayer] );

        auto lBegin        = aBegin.DataAs<uint32_t>()[lLayer];
        auto lEnd          = aEnd.DataAs<uint32_t>()[lLayer];
        auto lElementCount = aElementCount.DataAs<uint32_t>()[lLayer];

        RETURN_UNLESS( ( lBegin <= lEnd ) && ( lEnd < lElementCount ) );

        auto *lX   = aX.DeviceBufferAt<_Ty>( lLayer ) + i * lElementCount;
        auto *lOut = aOut.DeviceBufferAt<_Ty>( lLayer ) + i;

        _Ty lAccumulator = 0;
        for( uint32_t k = lBegin; k <= lEnd; k++ )
            lAccumulator += lX[k];

        *lOut = lAccumulator;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void ArraySlice( multi_tensor_t aOut, multi_tensor_t aX, memory_buffer_t aBegin, memory_buffer_t aEnd,
                                            memory_buffer_t aElementCount, memory_buffer_t aBlockSizes )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( i < aBlockSizes.DataAs<uint32_t>()[lLayer] );

        auto lBegin        = aBegin.DataAs<uint32_t>()[lLayer];
        auto lEnd          = aEnd.DataAs<uint32_t>()[lLayer];
        auto lElementCount = aElementCount.DataAs<uint32_t>()[lLayer];

        RETURN_UNLESS( ( lBegin <= lEnd ) && ( lEnd < lElementCount ) );

        auto *lX   = aX.DeviceBufferAt<_Ty>( lLayer ) + i * lElementCount;
        auto *lOut = aOut.DeviceBufferAt<_Ty>( lLayer ) + i * ( lEnd - lBegin + 1 );

        for( uint32_t k = lBegin; k <= lEnd; k++ )
            lOut[k - lBegin] = lX[k];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Diff( multi_tensor_t aOut, multi_tensor_t aX, uint32_t aCount, memory_buffer_t aElementCount,
                                      memory_buffer_t aBlockSizes )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( i < aBlockSizes.DataAs<uint32_t>()[lLayer] );

        auto lElementCount = aElementCount.DataAs<uint32_t>()[lLayer];

        RETURN_UNLESS( ( aCount < lElementCount ) );

        auto *lX   = aX.DeviceBufferAt<_Ty>( lLayer ) + i * lElementCount;
        auto *lOut = aOut.DeviceBufferAt<_Ty>( lLayer ) + i * lElementCount;

        for( uint32_t l = 0; l < lElementCount; l++ )
        {
            lOut[l] = lX[l];
        }

        for( uint32_t k = 0; k < aCount; k++ )
        {
            for( uint32_t l = 0; l < lElementCount - k; l++ )
            {
                lOut[l] = lOut[l + 1] - lOut[l];
            }
        }

        for( uint32_t k = lElementCount - aCount; k < lElementCount; k++ )
        {
            lOut[k] = static_cast<_Ty>( 0 );
        }
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void ShiftLeft( multi_tensor_t aOut, multi_tensor_t aX, uint32_t aCount, _Ty aFillValue,
                                           memory_buffer_t aElementCount, memory_buffer_t aBlockSizes )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( i < aBlockSizes.DataAs<uint32_t>()[lLayer] );

        auto lElementCount = aElementCount.DataAs<uint32_t>()[lLayer];

        RETURN_UNLESS( ( aCount < lElementCount ) );

        auto *lX   = aX.DeviceBufferAt<_Ty>( lLayer ) + i * lElementCount;
        auto *lOut = aOut.DeviceBufferAt<_Ty>( lLayer ) + i * lElementCount;

        for( uint32_t k = 0; k < lElementCount - aCount; k++ )
            lOut[k] = lX[k + aCount];

        for( uint32_t k = lElementCount - aCount; k < lElementCount; k++ )
            lOut[k] = aFillValue;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void ShiftRight( multi_tensor_t aOut, multi_tensor_t aX, uint32_t aCount, _Ty aFillValue,
                                            memory_buffer_t aElementCount, memory_buffer_t aBlockSizes )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( i < aBlockSizes.DataAs<uint32_t>()[lLayer] );

        auto lElementCount = aElementCount.DataAs<uint32_t>()[lLayer];

        RETURN_UNLESS( ( aCount < lElementCount ) );

        auto *lX   = aX.DeviceBufferAt<_Ty>( lLayer ) + i * lElementCount;
        auto *lOut = aOut.DeviceBufferAt<_Ty>( lLayer ) + i * lElementCount;

        for( uint32_t k = aCount; k < lElementCount; k++ )
            lOut[k] = lX[k - aCount];

        for( uint32_t k = 0; k < aCount; k++ )
            lOut[k] = aFillValue;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Conv1D( multi_tensor_t aOut, multi_tensor_t aArray0, memory_buffer_t aElementCount0, memory_buffer_t aBlockSizes0,
                                        multi_tensor_t aArray1, memory_buffer_t aElementCount1, memory_buffer_t aBlockSizes1 )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < aBlockSizes0.DataAs<uint32_t>()[lLayer] );

        auto lElementCount0 = aElementCount0.DataAs<uint32_t>()[lLayer];
        auto lElementCount1 = aElementCount1.DataAs<uint32_t>()[lLayer];

        RETURN_UNLESS( ( i < lElementCount0 ) );

        auto *lX   = aArray0.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y * lElementCount0;
        auto *lK   = aArray1.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y * lElementCount1;
        auto *lOut = aOut.DeviceBufferAt<_Ty>( lLayer ) + blockIdx.y * lElementCount0;

        _Ty lConvolutionValue = static_cast<_Ty>( 0 );
        for( uint32_t j = 0; j < lElementCount1; j++ )
        {
            if( i >= j )
                lConvolutionValue += ( lX[i - j] * lK[j] );
        }

        lOut[i] = lConvolutionValue;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void HCat( multi_tensor_t aOut, multi_tensor_t aX, memory_buffer_t aElementCountX, multi_tensor_t aY,
                                      memory_buffer_t aElementCountY, memory_buffer_t aBlockSizes )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( ( i < aBlockSizes.DataAs<uint32_t>()[lLayer] ) );

        auto lElementCountX = aElementCountX.DataAs<uint32_t>()[lLayer];
        auto lElementCountY = aElementCountY.DataAs<uint32_t>()[lLayer];

        auto *lX   = aX.DeviceBufferAt<_Ty>( lLayer ) + i * lElementCountX;
        auto *lY   = aY.DeviceBufferAt<_Ty>( lLayer ) + i * lElementCountY;
        auto *lOut = aOut.DeviceBufferAt<_Ty>( lLayer ) + i * ( lElementCountX + lElementCountY );

        uint32_t k = 0;
        for( uint32_t j = 0; j < lElementCountX; j++ )
            lOut[k++] = lX[j];
        for( uint32_t j = 0; j < lElementCountY; j++ )
            lOut[k++] = lY[j];
    }

} // namespace SE::TensorOps::Kernels