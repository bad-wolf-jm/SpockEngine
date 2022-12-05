/// @file   DeviceKernels.h
///
/// @brief  Template CUDA kernel definitions
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#pragma once

#include <chrono>

#include <cuda.h>
#include <curand.h>
#include <stdexcept>
#include <variant>

#include "Core/Logging.h"
#include "Core/Math/Types.h"

#include "Cuda/CudaAssert.h"
#include "Cuda/MemoryPool.h"
#include "Cuda/MultiTensor.h"
#include "Cuda/Texture2D.h"

#include "HelperMacros.h"

namespace SE::TensorOps::Kernels
{

    using namespace SE::Cuda;

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void ConstantFill( MultiTensor aArray, _Ty aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *aOut = aArray.DeviceBufferAt<_Ty>( lLayer );
        aOut[i]   = aConstant;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void ConstantFill( MultiTensor aArray, MemoryBuffer aConstants )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *aOut = aArray.DeviceBufferAt<_Ty>( lLayer );

        aOut[i] = aConstants.DataAs<_Ty>()[blockIdx.x];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void ARange( MultiTensor aOut, MemoryBuffer aLeft, MemoryBuffer aRight, MemoryBuffer aDelta )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lOutArray = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOutArray[i] = aLeft.DataAs<_Ty>()[blockIdx.x] + i * aDelta.DataAs<_Ty>()[blockIdx.x];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AddArrayToArray( MultiTensor aOut, MultiTensor aLeft, MultiTensor aRight )
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
    CUDA_KERNEL_DEFINITION void AddArrayToArray( MultiTensor aOut, MultiTensor aLeft, MultiTensor aRight,
                                                 eBroadcastHint aBroadcastHint, MemoryBuffer aBlockSizes,
                                                 MemoryBuffer aBroadcastSizes )
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
        default: break;
        }
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AddScalarToArray( MultiTensor aOut, MultiTensor aArray, _Ty aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lArray = aArray.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut   = aOut.DeviceBufferAt<_Ty>( lLayer );
        lOut[i]     = lArray[i] + aConstant;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AddArrayToVector( MultiTensor aOut, MultiTensor aArray, MemoryBuffer aConstants )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lIn  = aArray.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = lIn[i] + aConstants.DataAs<_Ty>()[blockIdx.x];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void MultiplyArrayByArray( MultiTensor aOut, MultiTensor aArray, MultiTensor aConstant )
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
    CUDA_KERNEL_DEFINITION void MultiplyArrayByArray( MultiTensor aOut, MultiTensor aLeft, MultiTensor aRight,
                                                      eBroadcastHint aBroadcastHint, MemoryBuffer aBlockSizes,
                                                      MemoryBuffer aBroadcastSizes )
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
        default: break;
        }
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void MultiplyScalarByArray( MultiTensor aOut, MultiTensor aArray, _Ty aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lArray = aArray.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut   = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = lArray[i] * aConstant;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void MultiplyArrayByVector( MultiTensor aOut, MultiTensor aArray, MemoryBuffer aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lIn  = aArray.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = lIn[i] * aConstant.DataAs<_Ty>()[blockIdx.x];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void SubtractArrayFromScalar( MultiTensor aOut, MultiTensor aArray, _Ty aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lArray = aArray.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut   = aOut.DeviceBufferAt<_Ty>( lLayer );
        lOut[i]     = aConstant - lArray[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void SubtractScalarFromArray( MultiTensor aOut, MultiTensor aArray, _Ty aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lArray = aArray.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut   = aOut.DeviceBufferAt<_Ty>( lLayer );
        lOut[i]     = lArray[i] - aConstant;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void SubtractArrayFromArray( MultiTensor aOut, MultiTensor aArray, MultiTensor aConstant )
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
    CUDA_KERNEL_DEFINITION void SubtractArrayFromArray( MultiTensor aOut, MultiTensor aLeft, MultiTensor aRight,
                                                        eBroadcastHint aBroadcastHint, MemoryBuffer aBlockSizes,
                                                        MemoryBuffer aBroadcastSizes )
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
        default: break;
        }
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void SubtractVectorFromArray( MultiTensor aOut, MultiTensor aArray, MemoryBuffer aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lIn  = aArray.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = lIn[i] - aConstant.DataAs<_Ty>()[blockIdx.x];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void SubtractArrayFromVector( MultiTensor aOut, MultiTensor aArray, MemoryBuffer aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lIn  = aArray.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = aConstant.DataAs<_Ty>()[blockIdx.x] - lIn[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void DivideArrayByScalar( MultiTensor aOut, MultiTensor aArray, _Ty aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        auto *lArray = aArray.DeviceBufferAt<_Ty>( lLayer );
        auto *lOut   = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = lArray[i] / aConstant;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void DivideScalarByArray( MultiTensor aOut, MultiTensor aArray, _Ty aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<_Ty>( lLayer, i ) );

        auto *lArray = aArray.DeviceBufferAt<_Ty>( lLayer );
        auto *lOut   = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = aConstant / lArray[i];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void DivideArrayFromArray( MultiTensor aOut, MultiTensor aArray, MultiTensor aConstant )
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
    CUDA_KERNEL_DEFINITION void DivideArrayFromArray( MultiTensor aOut, MultiTensor aLeft, MultiTensor aRight,
                                                      eBroadcastHint aBroadcastHint, MemoryBuffer aBlockSizes,
                                                      MemoryBuffer aBroadcastSizes )
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
        default: break;
        }
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void DivideArrayByVector( MultiTensor aOut, MultiTensor aArray, MemoryBuffer aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        auto *lIn  = aArray.DeviceBufferAt<_Ty>( lLayer );
        auto *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = lIn[i] / aConstant.DataAs<_Ty>()[blockIdx.x];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void DivideVectorByArray( MultiTensor aOut, MultiTensor aArray, MemoryBuffer aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        auto *lIn  = aArray.DeviceBufferAt<_Ty>( lLayer );
        auto *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = aConstant.DataAs<_Ty>()[blockIdx.x] / lIn[i];
    }

    CUDA_KERNEL_DEFINITION void AndTensorScalar( MultiTensor aOut, MultiTensor aArray, uint8_t aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<uint8_t>( lLayer, i ) );

        uint8_t *lArray = aArray.DeviceBufferAt<uint8_t>( lLayer );
        uint8_t *lOut   = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( aConstant && lArray[i] );
    }

    CUDA_KERNEL_DEFINITION void AndTensorTensor( MultiTensor aOut, MultiTensor aArray, MultiTensor aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<uint8_t>( lLayer, i ) );

        uint8_t *lArray    = aArray.DeviceBufferAt<uint8_t>( lLayer );
        uint8_t *lConstant = aConstant.DeviceBufferAt<uint8_t>( lLayer );
        uint8_t *lOut      = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( lArray[i] && lConstant[i] );
    }

    CUDA_KERNEL_DEFINITION void AndTensorTensor( MultiTensor aOut, MultiTensor aLeft, MultiTensor aRight,
                                                 eBroadcastHint aBroadcastHint, MemoryBuffer aBlockSizes,
                                                 MemoryBuffer aBroadcastSizes )
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
        default: break;
        }
    }

    CUDA_KERNEL_DEFINITION void AndTensorVector( MultiTensor aOut, MultiTensor aArray, MemoryBuffer aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<uint8_t>( lLayer, i ) );

        uint8_t *lIn  = aArray.DeviceBufferAt<uint8_t>( lLayer );
        uint8_t *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( lIn[i] && aConstant.DataAs<uint8_t>()[blockIdx.x] );
    }

    CUDA_KERNEL_DEFINITION void OrTensorScalar( MultiTensor aOut, MultiTensor aArray, uint8_t aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<uint8_t>( lLayer, i ) );

        uint8_t *lArray = aArray.DeviceBufferAt<uint8_t>( lLayer );
        uint8_t *lOut   = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( aConstant || lArray[i] );
    }

    CUDA_KERNEL_DEFINITION void OrTensorTensor( MultiTensor aOut, MultiTensor aArray, MultiTensor aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<uint8_t>( lLayer, i ) );

        uint8_t *lArray    = aArray.DeviceBufferAt<uint8_t>( lLayer );
        uint8_t *lConstant = aConstant.DeviceBufferAt<uint8_t>( lLayer );
        uint8_t *lOut      = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( lArray[i] || lConstant[i] );
    }

    CUDA_KERNEL_DEFINITION void OrTensorTensor( MultiTensor aOut, MultiTensor aLeft, MultiTensor aRight, eBroadcastHint aBroadcastHint,
                                                MemoryBuffer aBlockSizes, MemoryBuffer aBroadcastSizes )
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
        default: break;
        }
    }

    CUDA_KERNEL_DEFINITION void OrTensorVector( MultiTensor aOut, MultiTensor aArray, MemoryBuffer aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<uint8_t>( lLayer, i ) );

        uint8_t *lIn  = aArray.DeviceBufferAt<uint8_t>( lLayer );
        uint8_t *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( lIn[i] || aConstant.DataAs<uint8_t>()[blockIdx.x] );
    }

    CUDA_KERNEL_DEFINITION void NotTensor( MultiTensor aOut, MultiTensor aArray )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<uint8_t>( lLayer, i ) );

        uint8_t *lIn  = aArray.DeviceBufferAt<uint8_t>( lLayer );
        uint8_t *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = !( lIn[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void BitwiseAndTensorScalar( MultiTensor aOut, MultiTensor aArray, _Ty aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lArray = aArray.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut   = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = ( aConstant & lArray[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void BitwiseAndTensorTensor( MultiTensor aOut, MultiTensor aLeft, MultiTensor aRight,
                                                        eBroadcastHint aBroadcastHint, MemoryBuffer aBlockSizes,
                                                        MemoryBuffer aBroadcastSizes )
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
        default: break;
        }
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void BitwiseAndTensorTensor( MultiTensor aOut, MultiTensor aArray, MultiTensor aConstant )
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
    CUDA_KERNEL_DEFINITION void BitwiseAnd_Tensor_Vector( MultiTensor aOut, MultiTensor aArray, MemoryBuffer aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lIn  = aArray.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = ( lIn[i] & aConstant.DataAs<_Ty>()[blockIdx.x] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void BitwiseOrTensorScalar( MultiTensor aOut, MultiTensor aArray, _Ty aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lArray = aArray.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut   = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = ( aConstant | lArray[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void BitwiseOrTensorTensor( MultiTensor aOut, MultiTensor aArray, MultiTensor aConstant )
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
    CUDA_KERNEL_DEFINITION void BitwiseOrTensorTensor( MultiTensor aOut, MultiTensor aLeft, MultiTensor aRight,
                                                       eBroadcastHint aBroadcastHint, MemoryBuffer aBlockSizes,
                                                       MemoryBuffer aBroadcastSizes )
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
        default: break;
        }
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void BitwiseOrTensorVector( MultiTensor aOut, MultiTensor aArray, MemoryBuffer aConstant )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lIn  = aArray.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = ( lIn[i] | aConstant.DataAs<_Ty>()[blockIdx.x] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void BitwiseNotTensor( MultiTensor aOut, MultiTensor aArray )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lIn  = aArray.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = ~( lIn[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void InIntervalTensorTensor( MultiTensor aOut, MultiTensor aX, MultiTensor aLower, MultiTensor aUpper,
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
    CUDA_KERNEL_DEFINITION void InIntervalTensorVector( MultiTensor aOut, MultiTensor aX, MultiTensor aLower, MemoryBuffer aUpper,
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
    CUDA_KERNEL_DEFINITION void InIntervalTensorScalar( MultiTensor aOut, MultiTensor aX, MultiTensor aLower, _Ty aUpper,
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
    CUDA_KERNEL_DEFINITION void InIntervalVectorTensor( MultiTensor aOut, MultiTensor aX, MemoryBuffer aLower, MultiTensor aUpper,
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
    CUDA_KERNEL_DEFINITION void InIntervalVectorVector( MultiTensor aOut, MultiTensor aX, MemoryBuffer aLower, MemoryBuffer aUpper,
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
    CUDA_KERNEL_DEFINITION void InIntervalVectorScalar( MultiTensor aOut, MultiTensor aX, MemoryBuffer aLower, _Ty aUpper,
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
    CUDA_KERNEL_DEFINITION void InIntervalScalarTensor( MultiTensor aOut, MultiTensor aX, _Ty aLower, MultiTensor aUpper,
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
    CUDA_KERNEL_DEFINITION void InIntervalScalarVector( MultiTensor aOut, MultiTensor aX, _Ty aLower, MemoryBuffer aUpper,
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
    CUDA_KERNEL_DEFINITION void InIntervalScalarScalar( MultiTensor aOut, MultiTensor aX, _Ty aLower, _Ty aUpper, bool aStrictLower,
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
    CUDA_KERNEL_DEFINITION void EqualOp( MultiTensor aOut, MultiTensor aX, MultiTensor aY )
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
    CUDA_KERNEL_DEFINITION void EqualOp( MultiTensor aOut, MultiTensor aX, MemoryBuffer aY )
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
    CUDA_KERNEL_DEFINITION void EqualOp( MultiTensor aOut, MultiTensor aX, _Ty aY )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<uint8_t>( lLayer, i ) );

        _Ty     *lX   = aX.DeviceBufferAt<_Ty>( lLayer );
        uint8_t *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( lX[i] == aY );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void EqualOp( MultiTensor aOut, MultiTensor aLeft, MultiTensor aRight, eBroadcastHint aBroadcastHint,
                                         MemoryBuffer aBlockSizes, MemoryBuffer aBroadcastSizes )
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
        default: break;
        }
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void EqualOp( MultiTensor aOut, MemoryBuffer aX, MultiTensor aY )
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
    CUDA_KERNEL_DEFINITION void EqualOp( MultiTensor aOut, _Ty aX, MultiTensor aY )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<uint8_t>( lLayer, i ) );

        _Ty     *lY   = aY.DeviceBufferAt<_Ty>( lLayer );
        uint8_t *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( aX == lY[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOp( MultiTensor aOut, MultiTensor aX, MultiTensor aY )
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
    CUDA_KERNEL_DEFINITION void LessThanOp( MultiTensor aOut, MultiTensor aLeft, MultiTensor aRight, eBroadcastHint aBroadcastHint,
                                            MemoryBuffer aBlockSizes, MemoryBuffer aBroadcastSizes )
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
        default: break;
        }
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOp( MultiTensor aOut, MultiTensor aX, MemoryBuffer aY )
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
    CUDA_KERNEL_DEFINITION void LessThanOp( MultiTensor aOut, MultiTensor aX, _Ty aY )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<uint8_t>( lLayer, i ) );

        _Ty     *lX   = aX.DeviceBufferAt<_Ty>( lLayer );
        uint8_t *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( lX[i] < aY );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOp( MultiTensor aOut, MemoryBuffer aX, MultiTensor aY )
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
    CUDA_KERNEL_DEFINITION void LessThanOp( MultiTensor aOut, _Ty aX, MultiTensor aY )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<uint8_t>( lLayer, i ) );

        _Ty     *lY   = aY.DeviceBufferAt<_Ty>( lLayer );
        uint8_t *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( aX < lY[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOrEqualOp( MultiTensor aOut, MultiTensor aX, MultiTensor aY )
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
    CUDA_KERNEL_DEFINITION void LessThanOrEqualOp( MultiTensor aOut, MultiTensor aLeft, MultiTensor aRight,
                                                   eBroadcastHint aBroadcastHint, MemoryBuffer aBlockSizes,
                                                   MemoryBuffer aBroadcastSizes )
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
        default: break;
        }
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOrEqualOp( MultiTensor aOut, MultiTensor aX, MemoryBuffer aY )
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
    CUDA_KERNEL_DEFINITION void LessThanOrEqualOp( MultiTensor aOut, MultiTensor aX, _Ty aY )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<uint8_t>( lLayer, i ) );

        _Ty     *lX   = aX.DeviceBufferAt<_Ty>( lLayer );
        uint8_t *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( lX[i] <= aY );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void LessThanOrEqualOp( MultiTensor aOut, MemoryBuffer aX, MultiTensor aY )
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
    CUDA_KERNEL_DEFINITION void LessThanOrEqualOp( MultiTensor aOut, _Ty aX, MultiTensor aY )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aOut.Shape().InBounds<uint8_t>( lLayer, i ) );

        _Ty     *lY   = aY.DeviceBufferAt<_Ty>( lLayer );
        uint8_t *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer );

        lOut[i] = ( aX <= lY[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void WhereTensorTensor( MultiTensor aOut, MultiTensor aCondition, MultiTensor aValueIfTrue,
                                                   MultiTensor aValueIfFalse )
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
    CUDA_KERNEL_DEFINITION void WhereTensorVector( MultiTensor aOut, MultiTensor aCondition, MultiTensor aValueIfTrue,
                                                   MemoryBuffer aValueIfFalse )
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
    CUDA_KERNEL_DEFINITION void WhereTensorScalar( MultiTensor aOut, MultiTensor aCondition, MultiTensor aValueIfTrue,
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
    CUDA_KERNEL_DEFINITION void WhereVectorTensor( MultiTensor aOut, MultiTensor aCondition, MemoryBuffer aValueIfTrue,
                                                   MultiTensor aValueIfFalse )
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
    CUDA_KERNEL_DEFINITION void WhereVectorVector( MultiTensor aOut, MultiTensor aCondition, MemoryBuffer aValueIfTrue,
                                                   MemoryBuffer aValueIfFalse )
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
    CUDA_KERNEL_DEFINITION void WhereVectorScalar( MultiTensor aOut, MultiTensor aCondition, MemoryBuffer aValueIfTrue,
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
    CUDA_KERNEL_DEFINITION void WhereScalarTensor( MultiTensor aOut, MultiTensor aCondition, _Ty aValueIfTrue,
                                                   MultiTensor aValueIfFalse )
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
    CUDA_KERNEL_DEFINITION void WhereScalarVector( MultiTensor aOut, MultiTensor aCondition, _Ty aValueIfTrue,
                                                   MemoryBuffer aValueIfFalse )
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
    CUDA_KERNEL_DEFINITION void WhereScalarScalar( MultiTensor aOut, MultiTensor aCondition, _Ty aValueIfTrue, _Ty aValueIfFalse )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aCondition.Shape().InBounds<uint8_t>( lLayer, i ) );

        uint8_t *lCondition = aCondition.DeviceBufferAt<uint8_t>( lLayer );
        _Ty     *lOut       = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = lCondition[i] ? aValueIfTrue : aValueIfFalse;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Repeat( MultiTensor aOut, MultiTensor aArray, MemoryBuffer aRepetitions )
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
    CUDA_KERNEL_DEFINITION void Tile( MultiTensor aOut, MultiTensor aArray, MemoryBuffer aRepetitions )
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
    CUDA_KERNEL_DEFINITION void LinearSpace( MultiTensor aOut, MultiTensor aLeft, MultiTensor aRight, MemoryBuffer aSubdivisions )
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
    CUDA_KERNEL_DEFINITION void Mix( MultiTensor aOut, MultiTensor A, MultiTensor B, MultiTensor t )
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

    CUDA_KERNEL_DEFINITION void Sample2D( MultiTensor aOut, MultiTensor aX, MultiTensor aY, MemoryBuffer aTextures )
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

    CUDA_KERNEL_DEFINITION void Sample2D( MultiTensor aOut, MultiTensor aX, MemoryBuffer aY, MemoryBuffer aTextures )
    {
        uint32_t                           lLayer = static_cast<uint32_t>( blockIdx.x );
        Cuda::TextureSampler2D::DeviceData lTex   = aTextures.DataAs<Cuda::TextureSampler2D::DeviceData>()[blockIdx.x];

        int i = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aX.Shape().InBounds<float>( lLayer, i ) );

        auto *lXArray = aX.DeviceBufferAt<float>( lLayer );
        auto *lOut    = aOut.DeviceBufferAt<float>( lLayer );

        lOut[i] = lTex.Fetch<float>( lXArray[i], aY.DataAs<float>()[lLayer] );
    }

    CUDA_KERNEL_DEFINITION void Sample2D( MultiTensor aOut, MultiTensor aX, float aY, MemoryBuffer aTextures )
    {
        uint32_t                           lLayer = static_cast<uint32_t>( blockIdx.x );
        Cuda::TextureSampler2D::DeviceData lTex   = aTextures.DataAs<Cuda::TextureSampler2D::DeviceData>()[blockIdx.x];

        int i = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aX.Shape().InBounds<float>( lLayer, i ) );

        auto *lXArray = aX.DeviceBufferAt<float>( lLayer );
        auto *lOut    = aOut.DeviceBufferAt<float>( lLayer );

        lOut[i] = lTex.Fetch<float>( lXArray[i], aY );
    }

    CUDA_KERNEL_DEFINITION void Sample2D( MultiTensor aOut, MemoryBuffer aX, MultiTensor aY, MemoryBuffer aTextures )
    {
        uint32_t                           lLayer = static_cast<uint32_t>( blockIdx.x );
        Cuda::TextureSampler2D::DeviceData lTex   = aTextures.DataAs<Cuda::TextureSampler2D::DeviceData>()[blockIdx.x];

        int i = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aY.Shape().InBounds<float>( lLayer, i ) );

        auto *lYArray = aY.DeviceBufferAt<float>( lLayer );
        auto *lOut    = aOut.DeviceBufferAt<float>( lLayer );

        lOut[i] = lTex.Fetch<float>( aX.DataAs<float>()[lLayer], lYArray[i] );
    }

    CUDA_KERNEL_DEFINITION void Sample2D( MultiTensor aOut, float aX, MultiTensor aY, MemoryBuffer aTextures )
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
    CUDA_KERNEL_DEFINITION void ToFixedPoint( MultiTensor aOut, MultiTensor aArray, _Ty aScaling )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aArray.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty    *lInBuffer  = aArray.DeviceBufferAt<_Ty>( lLayer );
        _OutTy *lOutBuffer = aOut.DeviceBufferAt<_OutTy>( lLayer );
        lOutBuffer[i]      = static_cast<_OutTy>( lInBuffer[i] * aScaling );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AffineTransform( MultiTensor aOut, MultiTensor A, MultiTensor X, MultiTensor B )
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
    CUDA_KERNEL_DEFINITION void AffineTransform( MultiTensor aOut, MultiTensor A, MultiTensor X, MemoryBuffer B )
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
    CUDA_KERNEL_DEFINITION void AffineTransform( MultiTensor aOut, MultiTensor A, MultiTensor X, _Ty B )
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
    CUDA_KERNEL_DEFINITION void AffineTransform( MultiTensor aOut, MemoryBuffer A, MultiTensor X, MultiTensor B )
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
    CUDA_KERNEL_DEFINITION void AffineTransform( MultiTensor aOut, MemoryBuffer A, MultiTensor X, MemoryBuffer B )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( X.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lX   = X.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = A.DataAs<_Ty>()[blockIdx.x] * lX[i] + B.DataAs<_Ty>()[blockIdx.x];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AffineTransform( MultiTensor aOut, MemoryBuffer A, MultiTensor X, _Ty B )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( X.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lX   = X.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = A.DataAs<_Ty>()[blockIdx.x] * lX[i] + B;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AffineTransform( MultiTensor aOut, _Ty A, MultiTensor X, MultiTensor B )
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
    CUDA_KERNEL_DEFINITION void AffineTransform( MultiTensor aOut, _Ty A, MultiTensor X, MemoryBuffer B )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( X.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lX   = X.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = A * lX[i] + B.DataAs<_Ty>()[lLayer];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void AffineTransform( MultiTensor aOut, _Ty A, MultiTensor X, _Ty B )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( X.Shape().InBounds<_Ty>( lLayer, i ) );

        _Ty *lX   = X.DeviceBufferAt<_Ty>( lLayer );
        _Ty *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = A * lX[i] + B;
    }

    CUDA_KERNEL_DEFINITION void Floor( MultiTensor aOut, MultiTensor aX )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aX.Shape().InBounds<float>( lLayer, i ) );

        auto *lX   = aX.DeviceBufferAt<float>( lLayer );
        auto *lOut = aOut.DeviceBufferAt<float>( lLayer );

        lOut[i] = floor( lX[i] );
    }

    CUDA_KERNEL_DEFINITION void Ceil( MultiTensor aOut, MultiTensor aX )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aX.Shape().InBounds<float>( lLayer, i ) );

        auto *lX   = aX.DeviceBufferAt<float>( lLayer );
        auto *lOut = aOut.DeviceBufferAt<float>( lLayer );

        lOut[i] = ceil( lX[i] );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Sqrt( MultiTensor aOut, MultiTensor aX )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aX.Shape().InBounds<_Ty>( lLayer, i ) );

        auto *lX   = aX.DeviceBufferAt<_Ty>( lLayer );
        auto *lOut = aOut.DeviceBufferAt<float>( lLayer );

        lOut[i] = static_cast<_Ty>( sqrt( static_cast<float>( lX[i] ) ) );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Round( MultiTensor aOut, MultiTensor aX )
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
    CUDA_KERNEL_DEFINITION void Abs( MultiTensor aOut, MultiTensor aX )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( aX.Shape().InBounds<float>( lLayer, i ) );

        auto *lX   = aX.DeviceBufferAt<_Ty>( lLayer );
        auto *lOut = aOut.DeviceBufferAt<_Ty>( lLayer );

        lOut[i] = lX[i] * ( lX[i] >= 0 ? 1.0f : -1.0f );
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void CountZero( MultiTensor aOut, MultiTensor aX, MemoryBuffer aBlockSizes, MemoryBuffer aElementCount )
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
            if( lX[k] == static_cast<_Ty>( 0 ) ) lCount++;
        }

        *lOut = lCount;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void CountNonZero( MultiTensor aOut, MultiTensor aX, MemoryBuffer aBlockSizes, MemoryBuffer aElementCount )
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
            if( lX[k] != static_cast<_Ty>( 0 ) ) lCount++;
        }

        *lOut = lCount;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void ArraySummation( MultiTensor aOut, MultiTensor aX, MemoryBuffer aBegin, MemoryBuffer aEnd,
                                                MemoryBuffer aElementCount, MemoryBuffer aBlockSizes )
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
        for( uint32_t k = lBegin; k <= lEnd; k++ ) lAccumulator += lX[k];

        *lOut = lAccumulator;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void ArraySlice( MultiTensor aOut, MultiTensor aX, MemoryBuffer aBegin, MemoryBuffer aEnd,
                                            MemoryBuffer aElementCount, MemoryBuffer aBlockSizes )
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

        for( uint32_t k = lBegin; k <= lEnd; k++ ) lOut[k - lBegin] = lX[k];
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Diff( MultiTensor aOut, MultiTensor aX, uint32_t aCount, MemoryBuffer aElementCount,
                                      MemoryBuffer aBlockSizes )
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
    CUDA_KERNEL_DEFINITION void ShiftLeft( MultiTensor aOut, MultiTensor aX, uint32_t aCount, _Ty aFillValue,
                                           MemoryBuffer aElementCount, MemoryBuffer aBlockSizes )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( i < aBlockSizes.DataAs<uint32_t>()[lLayer] );

        auto lElementCount = aElementCount.DataAs<uint32_t>()[lLayer];

        RETURN_UNLESS( ( aCount < lElementCount ) );

        auto *lX   = aX.DeviceBufferAt<_Ty>( lLayer ) + i * lElementCount;
        auto *lOut = aOut.DeviceBufferAt<_Ty>( lLayer ) + i * lElementCount;

        for( uint32_t k = 0; k < lElementCount - aCount; k++ ) lOut[k] = lX[k + aCount];

        for( uint32_t k = lElementCount - aCount; k < lElementCount; k++ ) lOut[k] = aFillValue;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void ShiftRight( MultiTensor aOut, MultiTensor aX, uint32_t aCount, _Ty aFillValue,
                                            MemoryBuffer aElementCount, MemoryBuffer aBlockSizes )
    {
        uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
        int32_t  i      = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( i < aBlockSizes.DataAs<uint32_t>()[lLayer] );

        auto lElementCount = aElementCount.DataAs<uint32_t>()[lLayer];

        RETURN_UNLESS( ( aCount < lElementCount ) );

        auto *lX   = aX.DeviceBufferAt<_Ty>( lLayer ) + i * lElementCount;
        auto *lOut = aOut.DeviceBufferAt<_Ty>( lLayer ) + i * lElementCount;

        for( uint32_t k = aCount; k < lElementCount; k++ ) lOut[k] = lX[k - aCount];

        for( uint32_t k = 0; k < aCount; k++ ) lOut[k] = aFillValue;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void Conv1D( MultiTensor aOut, MultiTensor aArray0, MemoryBuffer aElementCount0, MemoryBuffer aBlockSizes0,
                                        MultiTensor aArray1, MemoryBuffer aElementCount1, MemoryBuffer aBlockSizes1 )
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
            if( i >= j ) lConvolutionValue += ( lX[i - j] * lK[j] );
        }

        lOut[i] = lConvolutionValue;
    }

    template <typename _Ty>
    CUDA_KERNEL_DEFINITION void HCat( MultiTensor aOut, MultiTensor aX, MemoryBuffer aElementCountX, MultiTensor aY,
                                      MemoryBuffer aElementCountY, MemoryBuffer aBlockSizes )
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
        for( uint32_t j = 0; j < lElementCountX; j++ ) lOut[k++] = lX[j];
        for( uint32_t j = 0; j < lElementCountY; j++ ) lOut[k++] = lY[j];
    }

} // namespace SE::TensorOps::Kernels