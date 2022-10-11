/// @file   KernelLaunchers.cu
///
/// @brief  C++ API for Cuda computation launchers
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#include "KernelLaunchers.h"

#include <chrono>

#include <cuda.h>
#include <curand.h>
#include <stdexcept>
#include <variant>

#include "Core/Logging.h"

#include "HelperMacros.h"

#include "DeviceKernels.inl"

namespace LTSE::TensorOps
{

    struct RandomNumberGenerator
    {
        curandGenerator_t Generator = nullptr;

        RandomNumberGenerator()
        {
            auto lNow   = std::chrono::system_clock::now();
            auto lNowNS = std::chrono::time_point_cast<std::chrono::nanoseconds>( lNow );
            auto lValue = lNowNS.time_since_epoch();
            CURAND_ASSERT( curandCreateGenerator( &Generator, CURAND_RNG_PSEUDO_DEFAULT ) );
            CURAND_ASSERT( curandSetPseudoRandomGeneratorSeed( Generator, lValue.count() ) );
        }

        ~RandomNumberGenerator() { curandDestroyGenerator( Generator ); }
    };

    template <typename _Ty> static void ConstantFillImpl( MultiTensor &aArray, ScalarValue &aConstant )
    {
        int lBlockCount = ( aArray.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aArray.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::ConstantFill<_Ty><<<lGridDim, lBlockDim>>>( aArray, std::get<_Ty>( aConstant ) );
    }

    template <typename _Ty> static void ConstantFillImpl( MultiTensor &aArray, MemoryBuffer &aInitialValues )
    {
        int lBlockCount = ( aArray.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aArray.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::ConstantFill<_Ty><<<lGridDim, lBlockDim>>>( aArray, aInitialValues );
    }

    void ConstantFill( eScalarType aTensorElementType, MultiTensor &aArray, MemoryBuffer &aInitialValues )
    {
        DISPATCH_BY_TYPE( aTensorElementType, ConstantFillImpl, ( aArray, aInitialValues ) );
    }

    void ConstantFill( eScalarType aTensorElementType, MultiTensor &aArray, ScalarValue &aInitialValues )
    {
        DISPATCH_BY_TYPE( aTensorElementType, ConstantFillImpl, ( aArray, aInitialValues ) );
    }

    void RandomUniformFill( eScalarType aTensorElementType, MultiTensor &aArray )
    {
        switch( aTensorElementType )
        {
        case eScalarType::FLOAT32:
        {
            RandomNumberGenerator lGenerator{};
            CURAND_ASSERT( curandGenerateUniform( lGenerator.Generator, aArray.DataAs<float>(), aArray.SizeAs<float>() ) );
        }
        break;
        case eScalarType::FLOAT64:
        {
            RandomNumberGenerator lGenerator{};
            CURAND_ASSERT( curandGenerateUniformDouble( lGenerator.Generator, aArray.DataAs<double>(), aArray.SizeAs<double>() ) );
        }
        break;
        default:
            std::runtime_error( "Random number type can only be float or double" );
        }
    }

    void RandomNormalFill( eScalarType aTensorElementType, MultiTensor &aArray, ScalarValue &aMu, ScalarValue &aSigma )
    {
        switch( aTensorElementType )
        {
        case eScalarType::FLOAT32:
        {
            float lMean = std::get<float>( aMu );
            float lStd  = std::get<float>( aSigma );
            if( lStd <= 0.0f )
                std::runtime_error( "Variance parameter should be strictly positive" );
            RandomNumberGenerator lGenerator{};
            CURAND_ASSERT( curandGenerateNormal( lGenerator.Generator, aArray.DataAs<float>(), aArray.SizeAs<float>(), lMean, lStd ) );
        }
        break;
        case eScalarType::FLOAT64:
        {
            double lMean = std::get<double>( aMu );
            double lStd  = std::get<double>( aSigma );
            if( lStd <= 0.0f )
                std::runtime_error( "Variance parameter should be strictly positive" );
            RandomNumberGenerator lGenerator{};
            CURAND_ASSERT( curandGenerateNormalDouble( lGenerator.Generator, aArray.DataAs<double>(), aArray.SizeAs<double>(), lMean, lStd ) );
        }
        break;
        default:
            std::runtime_error( "Random number type can only be float or double" );
        }
    }

    template <typename _Ty> static void ARangeOpImpl( MultiTensor &aOut, MemoryBuffer &aLeft, MemoryBuffer &aRight, MemoryBuffer &aDelta, uint32_t aMaxSubdivisions )
    {
        int lBlockCount = ( aMaxSubdivisions / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aOut.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::ARange<_Ty><<<lGridDim, lBlockDim>>>( aOut, aLeft, aRight, aDelta );
    }

    void ARangeOp( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MemoryBuffer &aRight, MemoryBuffer &aDelta, uint32_t aMaxSubdivisions )
    {
        switch( aTensorElementType )
        {
        case eScalarType::FLOAT32:
        {
            ARangeOpImpl<float>( aOut, aLeft, aRight, aDelta, aMaxSubdivisions );
            break;
        }
        case eScalarType::FLOAT64:
        {
            ARangeOpImpl<double>( aOut, aLeft, aRight, aDelta, aMaxSubdivisions );
            break;
        }
        default:
            throw std::runtime_error( "Linear space only supports float and double values" );
        }
    }

    template <typename _Ty> static void AddArrayToArrayImpl( MultiTensor &aOut, MultiTensor &aIn, MultiTensor &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::AddArrayToArray<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant );
    }

    template <typename _Ty>
    static void AddArrayToArrayImpl( MultiTensor &aOut, MultiTensor &aIn, MultiTensor &aConstant, eBroadcastHint aBroadcastHint, MemoryBuffer &aBlockSizes, uint32_t aMaxBlockSize,
                                     MemoryBuffer &aBroadcastSizes, uint32_t aMaxBroadcastSizes )
    {
        int lBlockCount = ( aMaxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), aMaxBlockSize, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::AddArrayToArray<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant, aBroadcastHint, aBlockSizes, aBroadcastSizes );
    }

    template <typename _ScalarType> static void AddScalarToArrayImpl( MultiTensor &aOut, MultiTensor &aArray, ScalarValue &aConstant )
    {
        int lBlockCount = ( aArray.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aArray.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::AddScalarToArray<_ScalarType><<<lGridDim, lBlockDim>>>( aOut, aArray, std::get<_ScalarType>( aConstant ) );
    }

    template <typename _Ty> static void AddArrayToVectorImpl( MultiTensor &aOut, MultiTensor &aIn, MemoryBuffer &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::AddArrayToVector<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant );
    }

    void AddOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, AddArrayToArrayImpl, ( aOut, aLeft, aRight ) );
    }

    void AddOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, eBroadcastHint aBroadcastHint, MemoryBuffer &aBlockSizes,
                uint32_t aMaxBlockSize, MemoryBuffer &aBroadcastSizes, uint32_t aMaxBroadcastSizes )
    {
        DISPATCH_BY_TYPE( aTensorElementType, AddArrayToArrayImpl, ( aOut, aLeft, aRight, aBroadcastHint, aBlockSizes, aMaxBlockSize, aBroadcastSizes, aMaxBroadcastSizes ) );
    }

    void AddOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, AddScalarToArrayImpl, ( aOut, aLeft, aRight ) );
    }

    void AddOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, AddArrayToVectorImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _ScalarType> void MultiplyArrayByScalarImpl( MultiTensor &aOut, MultiTensor &aIn, ScalarValue &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::MultiplyScalarByArray<_ScalarType><<<lGridDim, lBlockDim>>>( aOut, aIn, std::get<_ScalarType>( aConstant ) );
    }

    template <typename _Ty> static void MultiplyArrayByArrayImpl( MultiTensor &aOut, MultiTensor &aIn, MultiTensor &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::MultiplyArrayByArray<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant );
    }

    template <typename _Ty>
    static void MultiplyArrayByArrayImpl( MultiTensor &aOut, MultiTensor &aIn, MultiTensor &aConstant, eBroadcastHint aBroadcastHint, MemoryBuffer &aBlockSizes,
                                          uint32_t aMaxBlockSize, MemoryBuffer &aBroadcastSizes, uint32_t aMaxBroadcastSizes )
    {
        int lBlockCount = ( aMaxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), aMaxBlockSize, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::MultiplyArrayByArray<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant, aBroadcastHint, aBlockSizes, aBroadcastSizes );
    }

    template <typename _Ty> static void MultiplyArrayByVectorImpl( MultiTensor &aOut, MultiTensor &aIn, MemoryBuffer &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::MultiplyArrayByVector<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant );
    }

    void MultiplyOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, MultiplyArrayByScalarImpl, ( aOut, aLeft, aRight ) );
    }

    void MultiplyOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, MultiplyArrayByArrayImpl, ( aOut, aLeft, aRight ) );
    }

    void MultiplyOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, eBroadcastHint aBroadcastHint, MemoryBuffer &aBlockSizes,
                     uint32_t aMaxBlockSize, MemoryBuffer &aBroadcastSizes, uint32_t aMaxBroadcastSizes )
    {
        DISPATCH_BY_TYPE( aTensorElementType, MultiplyArrayByArrayImpl, ( aOut, aLeft, aRight, aBroadcastHint, aBlockSizes, aMaxBlockSize, aBroadcastSizes, aMaxBroadcastSizes ) );
    }

    void MultiplyOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, MultiplyArrayByVectorImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _Ty> void SubtractArrayFromScalarImpl( MultiTensor &aOut, ScalarValue &aConstant, MultiTensor &aIn )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::SubtractArrayFromScalar<<<lGridDim, lBlockDim>>>( aOut, aIn, std::get<_Ty>( aConstant ) );
    }

    template <typename _Ty> void SubtractScalarFromArrayImpl( MultiTensor &aOut, MultiTensor &aIn, ScalarValue &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::SubtractScalarFromArray<<<lGridDim, lBlockDim>>>( aOut, aIn, std::get<_Ty>( aConstant ) );
    }

    template <typename _Ty> static void SubtractVectorFromArrayImpl( MultiTensor &aOut, MultiTensor &aIn, MemoryBuffer &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::SubtractVectorFromArray<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant );
    }

    template <typename _Ty> static void SubtractArrayfromArrayImpl( MultiTensor &aOut, MultiTensor &aIn, MultiTensor &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::SubtractArrayFromArray<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant );
    }

    template <typename _Ty>
    static void SubtractArrayfromArrayImpl( MultiTensor &aOut, MultiTensor &aIn, MultiTensor &aConstant, eBroadcastHint aBroadcastHint, MemoryBuffer &aBlockSizes,
                                            uint32_t aMaxBlockSize, MemoryBuffer &aBroadcastSizes, uint32_t aMaxBroadcastSizes )
    {
        int lBlockCount = ( aMaxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), aMaxBlockSize, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::SubtractArrayFromArray<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant, aBroadcastHint, aBlockSizes, aBroadcastSizes );
    }

    template <typename _Ty> static void SubtractArrayFromVectorImpl( MultiTensor &aOut, MemoryBuffer &aConstant, MultiTensor &aIn )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::SubtractArrayFromVector<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant );
    }

    void SubtractOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, SubtractScalarFromArrayImpl, ( aOut, aLeft, aRight ) );
    }

    void SubtractOp( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aLeft, MultiTensor &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, SubtractArrayFromScalarImpl, ( aOut, aLeft, aRight ) );
    }

    void SubtractOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, SubtractArrayfromArrayImpl, ( aOut, aLeft, aRight ) );
    }

    void SubtractOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, eBroadcastHint aBroadcastHint, MemoryBuffer &aBlockSizes,
                     uint32_t aMaxBlockSize, MemoryBuffer &aBroadcastSizes, uint32_t aMaxBroadcastSizes )
    {
        DISPATCH_BY_TYPE( aTensorElementType, SubtractArrayfromArrayImpl,
                          ( aOut, aLeft, aRight, aBroadcastHint, aBlockSizes, aMaxBlockSize, aBroadcastSizes, aMaxBroadcastSizes ) );
    }

    void SubtractOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, SubtractVectorFromArrayImpl, ( aOut, aLeft, aRight ) );
    }

    void SubtractOp( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, SubtractArrayFromVectorImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _Ty> static void DivideArrayByScalarImpl( MultiTensor &aOut, MultiTensor &aIn, ScalarValue &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::DivideArrayByScalar<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, std::get<_Ty>( aConstant ) );
    }

    template <typename _Ty> static void DivideScalarByArrayImpl( MultiTensor &aOut, ScalarValue &aConstant, MultiTensor &aIn )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::DivideScalarByArray<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, std::get<_Ty>( aConstant ) );
    }

    template <typename _Ty> static void DivideArrayfromArrayImpl( MultiTensor &aOut, MultiTensor &aIn, MultiTensor &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::DivideArrayFromArray<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant );
    }

    template <typename _Ty>
    static void DivideArrayfromArrayImpl( MultiTensor &aOut, MultiTensor &aIn, MultiTensor &aConstant, eBroadcastHint aBroadcastHint, MemoryBuffer &aBlockSizes,
                                          uint32_t aMaxBlockSize, MemoryBuffer &aBroadcastSizes, uint32_t aMaxBroadcastSizes )
    {
        int lBlockCount = ( aMaxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), aMaxBlockSize, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::DivideArrayFromArray<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant, aBroadcastHint, aBlockSizes, aBroadcastSizes );
    }

    template <typename _Ty> static void DivideArrayByVectorImpl( MultiTensor &aOut, MultiTensor &aIn, MemoryBuffer &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::DivideArrayByVector<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant );
    }

    template <typename _Ty> static void DivideVectorByArrayImpl( MultiTensor &aOut, MemoryBuffer &aConstant, MultiTensor &aIn )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::DivideVectorByArray<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant );
    }

    void DivideOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, DivideArrayByScalarImpl, ( aOut, aLeft, aRight ) );
    }

    void DivideOp( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aLeft, MultiTensor &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, DivideScalarByArrayImpl, ( aOut, aLeft, aRight ) );
    }

    void DivideOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, DivideArrayfromArrayImpl, ( aOut, aLeft, aRight ) );
    }

    void DivideOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, eBroadcastHint aBroadcastHint, MemoryBuffer &aBlockSizes,
                   uint32_t aMaxBlockSize, MemoryBuffer &aBroadcastSizes, uint32_t aMaxBroadcastSizes )
    {
        DISPATCH_BY_TYPE( aTensorElementType, DivideArrayfromArrayImpl, ( aOut, aLeft, aRight, aBroadcastHint, aBlockSizes, aMaxBlockSize, aBroadcastSizes, aMaxBroadcastSizes ) );
    }

    void DivideOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, DivideArrayByVectorImpl, ( aOut, aLeft, aRight ) );
    }

    void DivideOp( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, DivideVectorByArrayImpl, ( aOut, aLeft, aRight ) );
    }

    void AndOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight )
    {
        int lBlockCount = ( aLeft.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aLeft.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::AndTensorScalar<<<lGridDim, lBlockDim>>>( aOut, aLeft, std::get<uint8_t>( aRight ) );
    }

    void AndOp( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aLeft, MultiTensor &aRight ) { AndOp( aTensorElementType, aOut, aRight, aLeft ); }

    void AndOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aIn, MultiTensor &aConstant, eBroadcastHint aBroadcastHint, MemoryBuffer &aBlockSizes,
                uint32_t aMaxBlockSize, MemoryBuffer &aBroadcastSizes, uint32_t aMaxBroadcastSizes )
    {
        int lBlockCount = ( aMaxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), aMaxBlockSize, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::AndTensorTensor<<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant, aBroadcastHint, aBlockSizes, aBroadcastSizes );
    }

    void AndOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight )
    {
        int lBlockCount = ( aLeft.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aLeft.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::AndTensorTensor<<<lGridDim, lBlockDim>>>( aOut, aLeft, aRight );
    }

    void AndOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight )
    {
        int lBlockCount = ( aLeft.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aLeft.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::AndTensorVector<<<lGridDim, lBlockDim>>>( aOut, aLeft, aRight );
    }

    void AndOp( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight ) { AndOp( aTensorElementType, aOut, aRight, aLeft ); }

    void OrOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight )
    {
        int lBlockCount = ( aLeft.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aLeft.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::OrTensorScalar<<<lGridDim, lBlockDim>>>( aOut, aLeft, std::get<uint8_t>( aRight ) );
    }

    void OrOp( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aLeft, MultiTensor &aRight ) { OrOp( aTensorElementType, aOut, aRight, aLeft ); }

    void OrOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight )
    {
        int lBlockCount = ( aLeft.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aLeft.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::OrTensorTensor<<<lGridDim, lBlockDim>>>( aOut, aLeft, aRight );
    }

    void OrOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aIn, MultiTensor &aConstant, eBroadcastHint aBroadcastHint, MemoryBuffer &aBlockSizes,
               uint32_t aMaxBlockSize, MemoryBuffer &aBroadcastSizes, uint32_t aMaxBroadcastSizes )
    {
        int lBlockCount = ( aMaxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), aMaxBlockSize, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::OrTensorTensor<<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant, aBroadcastHint, aBlockSizes, aBroadcastSizes );
    }

    void OrOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight )
    {
        int lBlockCount = ( aLeft.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aLeft.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::OrTensorVector<<<lGridDim, lBlockDim>>>( aOut, aLeft, aRight );
    }

    void OrOp( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight ) { OrOp( aTensorElementType, aOut, aRight, aLeft ); }

    void NotOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aOperand )
    {
        int lBlockCount = ( aOperand.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aOperand.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::NotTensor<<<lGridDim, lBlockDim>>>( aOut, aOperand );
    }

    template <typename _Ty> void BitwiseAnd_Tensor_Scalar_Impl( MultiTensor &aOut, MultiTensor &aIn, ScalarValue &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::BitwiseAndTensorScalar<<<lGridDim, lBlockDim>>>( aOut, aIn, std::get<_Ty>( aConstant ) );
    }

    void BitwiseAndOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight )
    {
        DISPATCH_BY_INTEGRAL_TYPE( aTensorElementType, BitwiseAnd_Tensor_Scalar_Impl, ( aOut, aLeft, aRight ) );
    }

    void BitwiseAndOp( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aLeft, MultiTensor &aRight ) { BitwiseAndOp( aTensorElementType, aOut, aRight, aLeft ); }

    template <typename _Ty>
    static void BitwiseAnd_Tensor_Tensor_Impl( MultiTensor &aOut, MultiTensor &aIn, MultiTensor &aConstant, eBroadcastHint aBroadcastHint, MemoryBuffer &aBlockSizes,
                                               uint32_t aMaxBlockSize, MemoryBuffer &aBroadcastSizes, uint32_t aMaxBroadcastSizes )
    {
        int lBlockCount = ( aMaxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), aMaxBlockSize, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::BitwiseAndTensorTensor<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant, aBroadcastHint, aBlockSizes, aBroadcastSizes );
    }

    void BitwiseAndOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, eBroadcastHint aBroadcastHint, MemoryBuffer &aBlockSizes,
                       uint32_t aMaxBlockSize, MemoryBuffer &aBroadcastSizes, uint32_t aMaxBroadcastSizes )
    {
        DISPATCH_BY_INTEGRAL_TYPE( aTensorElementType, BitwiseAnd_Tensor_Tensor_Impl,
                                   ( aOut, aLeft, aRight, aBroadcastHint, aBlockSizes, aMaxBlockSize, aBroadcastSizes, aMaxBroadcastSizes ) );
    }

    template <typename _Ty> void BitwiseAnd_Tensor_Tensor_Impl( MultiTensor &aOut, MultiTensor &aIn, MultiTensor &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::BitwiseAndTensorTensor<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant );
    }

    void BitwiseAndOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight )
    {
        DISPATCH_BY_INTEGRAL_TYPE( aTensorElementType, BitwiseAnd_Tensor_Tensor_Impl, ( aOut, aLeft, aRight ) );
    }

    template <typename _Ty> void BitwiseAnd_Tensor_Vector_Impl( MultiTensor &aOut, MultiTensor &aIn, MemoryBuffer &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::BitwiseAnd_Tensor_Vector<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant );
    }

    void BitwiseAndOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight )
    {
        DISPATCH_BY_INTEGRAL_TYPE( aTensorElementType, BitwiseAnd_Tensor_Vector_Impl, ( aOut, aLeft, aRight ) );
    }

    void BitwiseAndOp( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight ) { BitwiseAndOp( aTensorElementType, aOut, aRight, aLeft ); }

    template <typename _Ty> void BitwiseOr_Tensor_Scalar_Impl( MultiTensor &aOut, MultiTensor &aIn, ScalarValue &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::BitwiseOrTensorScalar<<<lGridDim, lBlockDim>>>( aOut, aIn, std::get<_Ty>( aConstant ) );
    }

    void BitwiseOrOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight )
    {
        DISPATCH_BY_INTEGRAL_TYPE( aTensorElementType, BitwiseOr_Tensor_Scalar_Impl, ( aOut, aLeft, aRight ) );
    }

    void BitwiseOrOp( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aLeft, MultiTensor &aRight ) { BitwiseOrOp( aTensorElementType, aOut, aRight, aLeft ); }

    template <typename _Ty> void BitwiseOr_Tensor_Tensor_Impl( MultiTensor &aOut, MultiTensor &aIn, MultiTensor &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::BitwiseOrTensorTensor<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant );
    }

    void BitwiseOrOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight )
    {
        DISPATCH_BY_INTEGRAL_TYPE( aTensorElementType, BitwiseOr_Tensor_Tensor_Impl, ( aOut, aLeft, aRight ) );
    }

    template <typename _Ty>
    static void BitwiseOr_Tensor_Tensor_Impl( MultiTensor &aOut, MultiTensor &aIn, MultiTensor &aConstant, eBroadcastHint aBroadcastHint, MemoryBuffer &aBlockSizes,
                                              uint32_t aMaxBlockSize, MemoryBuffer &aBroadcastSizes, uint32_t aMaxBroadcastSizes )
    {
        int lBlockCount = ( aMaxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), aMaxBlockSize, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::BitwiseOrTensorTensor<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant, aBroadcastHint, aBlockSizes, aBroadcastSizes );
    }

    void BitwiseOrOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, eBroadcastHint aBroadcastHint, MemoryBuffer &aBlockSizes,
                      uint32_t aMaxBlockSize, MemoryBuffer &aBroadcastSizes, uint32_t aMaxBroadcastSizes )
    {
        DISPATCH_BY_INTEGRAL_TYPE( aTensorElementType, BitwiseOr_Tensor_Tensor_Impl,
                                   ( aOut, aLeft, aRight, aBroadcastHint, aBlockSizes, aMaxBlockSize, aBroadcastSizes, aMaxBroadcastSizes ) );
    }

    template <typename _Ty> void BitwiseOrTensorVectorImpl( MultiTensor &aOut, MultiTensor &aIn, MemoryBuffer &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::BitwiseOrTensorVector<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant );
    }

    void BitwiseOrOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight )
    {
        DISPATCH_BY_INTEGRAL_TYPE( aTensorElementType, BitwiseOrTensorVectorImpl, ( aOut, aLeft, aRight ) );
    }

    void BitwiseOrOp( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight ) { BitwiseOrOp( aTensorElementType, aOut, aRight, aLeft ); }

    template <typename _Ty> void BitwiseNotTensorImpl( MultiTensor &aOut, MultiTensor &aIn )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::BitwiseNotTensor<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn );
    }

    void BitwiseNotOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aOperand )
    {
        DISPATCH_BY_INTEGRAL_TYPE( aTensorElementType, BitwiseNotTensorImpl, ( aOut, aOperand ) );
    }

    template <typename _Ty> static void EqualOpImpl( MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight )
    {
        int lBlockCount = ( aLeft.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aLeft.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::EqualOp<_Ty><<<lGridDim, lBlockDim>>>( aOut, aLeft, aRight );
    }

    void EqualOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, EqualOpImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _Ty>
    static void EqualOpImpl( MultiTensor &aOut, MultiTensor &aIn, MultiTensor &aConstant, eBroadcastHint aBroadcastHint, MemoryBuffer &aBlockSizes, uint32_t aMaxBlockSize,
                             MemoryBuffer &aBroadcastSizes, uint32_t aMaxBroadcastSizes )
    {
        int lBlockCount = ( aMaxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), aMaxBlockSize, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::EqualOp<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant, aBroadcastHint, aBlockSizes, aBroadcastSizes );
    }

    void EqualOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, eBroadcastHint aBroadcastHint, MemoryBuffer &aBlockSizes,
                  uint32_t aMaxBlockSize, MemoryBuffer &aBroadcastSizes, uint32_t aMaxBroadcastSizes )
    {
        DISPATCH_BY_TYPE( aTensorElementType, EqualOpImpl, ( aOut, aLeft, aRight, aBroadcastHint, aBlockSizes, aMaxBlockSize, aBroadcastSizes, aMaxBroadcastSizes ) );
    }

    template <typename _Ty> static void EqualOpImpl( MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight )
    {
        int lBlockCount = ( aLeft.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aLeft.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::EqualOp<_Ty><<<lGridDim, lBlockDim>>>( aOut, aLeft, aRight );
    }

    void EqualOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, EqualOpImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _ScalarType> static void EqualOpImpl( MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight )
    {
        int lBlockCount = ( aLeft.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aLeft.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::EqualOp<_ScalarType><<<lGridDim, lBlockDim>>>( aOut, aLeft, std::get<_ScalarType>( aRight ) );
    }

    void EqualOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, EqualOpImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _Ty> static void EqualOpImpl( MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight )
    {
        int lBlockCount = ( aRight.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aRight.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::EqualOp<_Ty><<<lGridDim, lBlockDim>>>( aOut, aLeft, aRight );
    }

    void EqualOp( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, EqualOpImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _Ty> static void EqualOpImpl( MultiTensor &aOut, ScalarValue &aLeft, MultiTensor &aRight )
    {
        int lBlockCount = ( aRight.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aRight.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::EqualOp<_Ty><<<lGridDim, lBlockDim>>>( aOut, std::get<_Ty>( aLeft ), aRight );
    }

    void EqualOp( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aLeft, MultiTensor &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, EqualOpImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _Ty> static void LessThanOpImpl( MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight )
    {
        int lBlockCount = ( aLeft.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aLeft.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOp<_Ty><<<lGridDim, lBlockDim>>>( aOut, aLeft, aRight );
    }

    void LessThanOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, LessThanOpImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _Ty>
    static void LessThanOpImpl( MultiTensor &aOut, MultiTensor &aIn, MultiTensor &aConstant, eBroadcastHint aBroadcastHint, MemoryBuffer &aBlockSizes, uint32_t aMaxBlockSize,
                                MemoryBuffer &aBroadcastSizes, uint32_t aMaxBroadcastSizes )
    {
        int lBlockCount = ( aMaxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), aMaxBlockSize, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOp<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant, aBroadcastHint, aBlockSizes, aBroadcastSizes );
    }

    void LessThanOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, eBroadcastHint aBroadcastHint, MemoryBuffer &aBlockSizes,
                     uint32_t aMaxBlockSize, MemoryBuffer &aBroadcastSizes, uint32_t aMaxBroadcastSizes )
    {
        DISPATCH_BY_TYPE( aTensorElementType, LessThanOpImpl, ( aOut, aLeft, aRight, aBroadcastHint, aBlockSizes, aMaxBlockSize, aBroadcastSizes, aMaxBroadcastSizes ) );
    }

    template <typename _Ty> static void LessThanOpImpl( MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight )
    {
        int lBlockCount = ( aLeft.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aLeft.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOp<_Ty><<<lGridDim, lBlockDim>>>( aOut, aLeft, aRight );
    }

    void LessThanOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, LessThanOpImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _ScalarType> static void LessThanOpImpl( MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight )
    {
        int lBlockCount = ( aLeft.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aLeft.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOp<_ScalarType><<<lGridDim, lBlockDim>>>( aOut, aLeft, std::get<_ScalarType>( aRight ) );
    }

    void LessThanOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, LessThanOpImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _Ty> static void LessThanOpImpl( MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight )
    {
        int lBlockCount = ( aRight.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aRight.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOp<_Ty><<<lGridDim, lBlockDim>>>( aOut, aLeft, aRight );
    }

    void LessThanOp( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, LessThanOpImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _Ty> static void LessThanOpImpl( MultiTensor &aOut, ScalarValue &aLeft, MultiTensor &aRight )
    {
        int lBlockCount = ( aRight.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aRight.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOp<_Ty><<<lGridDim, lBlockDim>>>( aOut, std::get<_Ty>( aLeft ), aRight );
    }

    void LessThanOp( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aLeft, MultiTensor &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, LessThanOpImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _Ty> static void LessThanOrEqualOpImpl( MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight )
    {
        int lBlockCount = ( aLeft.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aLeft.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOrEqualOp<_Ty><<<lGridDim, lBlockDim>>>( aOut, aLeft, aRight );
    }

    void LessThanOrEqualOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, LessThanOrEqualOpImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _Ty>
    static void LessThanOrEqualOpImpl( MultiTensor &aOut, MultiTensor &aIn, MultiTensor &aConstant, eBroadcastHint aBroadcastHint, MemoryBuffer &aBlockSizes,
                                       uint32_t aMaxBlockSize, MemoryBuffer &aBroadcastSizes, uint32_t aMaxBroadcastSizes )
    {
        int lBlockCount = ( aMaxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), aMaxBlockSize, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOrEqualOp<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant, aBroadcastHint, aBlockSizes, aBroadcastSizes );
    }

    void LessThanOrEqualOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, eBroadcastHint aBroadcastHint, MemoryBuffer &aBlockSizes,
                            uint32_t aMaxBlockSize, MemoryBuffer &aBroadcastSizes, uint32_t aMaxBroadcastSizes )
    {
        DISPATCH_BY_TYPE( aTensorElementType, LessThanOrEqualOpImpl, ( aOut, aLeft, aRight, aBroadcastHint, aBlockSizes, aMaxBlockSize, aBroadcastSizes, aMaxBroadcastSizes ) );
    }

    template <typename _Ty> static void LessThanOrEqualOpImpl( MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight )
    {
        int lBlockCount = ( aLeft.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aLeft.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOrEqualOp<_Ty><<<lGridDim, lBlockDim>>>( aOut, aLeft, aRight );
    }

    void LessThanOrEqualOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, LessThanOrEqualOpImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _ScalarType> static void LessThanOrEqualOpImpl( MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight )
    {
        int lBlockCount = ( aLeft.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aLeft.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOrEqualOp<_ScalarType><<<lGridDim, lBlockDim>>>( aOut, aLeft, std::get<_ScalarType>( aRight ) );
    }

    void LessThanOrEqualOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, LessThanOrEqualOpImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _Ty> static void LessThanOrEqualOpImpl( MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight )
    {
        int lBlockCount = ( aRight.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aRight.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOrEqualOp<_Ty><<<lGridDim, lBlockDim>>>( aOut, aLeft, aRight );
    }

    void LessThanOrEqualOp( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, LessThanOrEqualOpImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _Ty> static void LessThanOrEqualOpImpl( MultiTensor &aOut, ScalarValue &aLeft, MultiTensor &aRight )
    {
        int lBlockCount = ( aRight.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aRight.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOrEqualOp<_Ty><<<lGridDim, lBlockDim>>>( aOut, std::get<_Ty>( aLeft ), aRight );
    }

    void LessThanOrEqualOp( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aLeft, MultiTensor &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, LessThanOrEqualOpImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _Ty>
    static void InIntervalTensorTensorImpl( MultiTensor &aOut, MultiTensor &aX, MultiTensor &aLower, MultiTensor &aUpper, bool aStrictLower, bool aStrictUpper )
    {
        int lBlockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::InIntervalTensorTensor<_Ty><<<lGridDim, lBlockDim>>>( aOut, aX, aLower, aUpper, aStrictLower, aStrictUpper );
    }

    void InIntervalOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aX, MultiTensor &aLower, MultiTensor &aUpper, bool aStrictLower, bool aStrictUpper )
    {
        DISPATCH_BY_TYPE( aTensorElementType, InIntervalTensorTensorImpl, ( aOut, aX, aLower, aUpper, aStrictLower, aStrictUpper ) );
    }

    template <typename _Ty>
    static void InIntervalTensorVectorImpl( MultiTensor &aOut, MultiTensor &aX, MultiTensor &aLower, MemoryBuffer &aUpper, bool aStrictLower, bool aStrictUpper )
    {
        int lBlockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::InIntervalTensorVector<_Ty><<<lGridDim, lBlockDim>>>( aOut, aX, aLower, aUpper, aStrictLower, aStrictUpper );
    }

    void InIntervalOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aX, MultiTensor &aLower, MemoryBuffer &aUpper, bool aStrictLower, bool aStrictUpper )
    {
        DISPATCH_BY_TYPE( aTensorElementType, InIntervalTensorVectorImpl, ( aOut, aX, aLower, aUpper, aStrictLower, aStrictUpper ) );
    }

    template <typename _Ty>
    static void InIntervalTensorScalarImpl( MultiTensor &aOut, MultiTensor &aX, MultiTensor &aLower, ScalarValue &aUpper, bool aStrictLower, bool aStrictUpper )
    {
        int lBlockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::InIntervalTensorScalar<_Ty><<<lGridDim, lBlockDim>>>( aOut, aX, aLower, std::get<_Ty>( aUpper ), aStrictLower, aStrictUpper );
    }

    void InIntervalOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aX, MultiTensor &aLower, ScalarValue &aUpper, bool aStrictLower, bool aStrictUpper )
    {
        DISPATCH_BY_TYPE( aTensorElementType, InIntervalTensorScalarImpl, ( aOut, aX, aLower, aUpper, aStrictLower, aStrictUpper ) );
    }

    template <typename _Ty>
    static void InIntervalVectorTensorImpl( MultiTensor &aOut, MultiTensor &aX, MemoryBuffer &aLower, MultiTensor &aUpper, bool aStrictLower, bool aStrictUpper )
    {
        int lBlockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::InIntervalVectorTensor<_Ty><<<lGridDim, lBlockDim>>>( aOut, aX, aLower, aUpper, aStrictLower, aStrictUpper );
    }

    void InIntervalOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aX, MemoryBuffer &aLower, MultiTensor &aUpper, bool aStrictLower, bool aStrictUpper )
    {
        DISPATCH_BY_TYPE( aTensorElementType, InIntervalVectorTensorImpl, ( aOut, aX, aLower, aUpper, aStrictLower, aStrictUpper ) );
    }

    template <typename _Ty>
    static void InIntervalVectorVectorImpl( MultiTensor &aOut, MultiTensor &aX, MemoryBuffer &aLower, MemoryBuffer &aUpper, bool aStrictLower, bool aStrictUpper )
    {
        int lBlockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::InIntervalVectorVector<_Ty><<<lGridDim, lBlockDim>>>( aOut, aX, aLower, aUpper, aStrictLower, aStrictUpper );
    }

    void InIntervalOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aX, MemoryBuffer &aLower, MemoryBuffer &aUpper, bool aStrictLower, bool aStrictUpper )
    {
        DISPATCH_BY_TYPE( aTensorElementType, InIntervalVectorVectorImpl, ( aOut, aX, aLower, aUpper, aStrictLower, aStrictUpper ) );
    }

    template <typename _Ty>
    static void InIntervalVectorScalarImpl( MultiTensor &aOut, MultiTensor &aX, MemoryBuffer &aLower, ScalarValue &aUpper, bool aStrictLower, bool aStrictUpper )
    {
        int lBlockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::InIntervalVectorScalar<_Ty><<<lGridDim, lBlockDim>>>( aOut, aX, aLower, std::get<_Ty>( aUpper ), aStrictLower, aStrictUpper );
    }

    void InIntervalOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aX, MemoryBuffer &aLower, ScalarValue &aUpper, bool aStrictLower, bool aStrictUpper )
    {
        DISPATCH_BY_TYPE( aTensorElementType, InIntervalVectorScalarImpl, ( aOut, aX, aLower, aUpper, aStrictLower, aStrictUpper ) );
    }

    template <typename _Ty>
    static void InIntervalScalarTensorImpl( MultiTensor &aOut, MultiTensor &aX, ScalarValue &aLower, MultiTensor &aUpper, bool aStrictLower, bool aStrictUpper )
    {
        int lBlockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::InIntervalScalarTensor<_Ty><<<lGridDim, lBlockDim>>>( aOut, aX, std::get<_Ty>( aLower ), aUpper, aStrictLower, aStrictUpper );
    }

    void InIntervalOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aX, ScalarValue &aLower, MultiTensor &aUpper, bool aStrictLower, bool aStrictUpper )
    {
        DISPATCH_BY_TYPE( aTensorElementType, InIntervalScalarTensorImpl, ( aOut, aX, aLower, aUpper, aStrictLower, aStrictUpper ) );
    }

    template <typename _Ty>
    static void InIntervalScalarVectorImpl( MultiTensor &aOut, MultiTensor &aX, ScalarValue &aLower, MemoryBuffer &aUpper, bool aStrictLower, bool aStrictUpper )
    {
        int lBlockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::InIntervalScalarVector<_Ty><<<lGridDim, lBlockDim>>>( aOut, aX, std::get<_Ty>( aLower ), aUpper, aStrictLower, aStrictUpper );
    }

    void InIntervalOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aX, ScalarValue &aLower, MemoryBuffer &aUpper, bool aStrictLower, bool aStrictUpper )
    {
        DISPATCH_BY_TYPE( aTensorElementType, InIntervalScalarVectorImpl, ( aOut, aX, aLower, aUpper, aStrictLower, aStrictUpper ) );
    }

    template <typename _Ty>
    static void InIntervalScalarScalarImpl( MultiTensor &aOut, MultiTensor &aX, ScalarValue &aLower, ScalarValue &aUpper, bool aStrictLower, bool aStrictUpper )
    {
        int lBlockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::InIntervalScalarScalar<_Ty><<<lGridDim, lBlockDim>>>( aOut, aX, std::get<_Ty>( aLower ), std::get<_Ty>( aUpper ), aStrictLower, aStrictUpper );
    }

    void InIntervalOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aX, ScalarValue &aLower, ScalarValue &aUpper, bool aStrictLower, bool aStrictUpper )
    {
        DISPATCH_BY_TYPE( aTensorElementType, InIntervalScalarScalarImpl, ( aOut, aX, aLower, aUpper, aStrictLower, aStrictUpper ) );
    }

    template <typename _Ty> static void WhereOpTensorTensorImpl( MultiTensor &aOut, MultiTensor &aCondition, MultiTensor &aValueIfTrue, MultiTensor &aValueIfFalse )
    {
        int lBlockCount = ( aCondition.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aCondition.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::WhereTensorTensor<_Ty><<<lGridDim, lBlockDim>>>( aOut, aCondition, aValueIfTrue, aValueIfFalse );
    }

    void WhereOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aCondition, MultiTensor &aValueIfTrue, MultiTensor &aValueIfFalse )
    {
        DISPATCH_BY_TYPE( aTensorElementType, WhereOpTensorTensorImpl, ( aOut, aCondition, aValueIfTrue, aValueIfFalse ) );
    }

    template <typename _Ty> static void WhereTensorVectorImpl( MultiTensor &aOut, MultiTensor &aCondition, MultiTensor &aValueIfTrue, MemoryBuffer &aValueIfFalse )
    {
        int lBlockCount = ( aCondition.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aCondition.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::WhereTensorVector<_Ty><<<lGridDim, lBlockDim>>>( aOut, aCondition, aValueIfTrue, aValueIfFalse );
    }

    void WhereOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aCondition, MultiTensor &aValueIfTrue, MemoryBuffer &aValueIfFalse )
    {
        DISPATCH_BY_TYPE( aTensorElementType, WhereTensorVectorImpl, ( aOut, aCondition, aValueIfTrue, aValueIfFalse ) );
    }

    template <typename _Ty> static void WhereTensorScalarImpl( MultiTensor &aOut, MultiTensor &aCondition, MultiTensor &aValueIfTrue, ScalarValue &aValueIfFalse )
    {
        int lBlockCount = ( aCondition.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aCondition.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::WhereTensorScalar<_Ty><<<lGridDim, lBlockDim>>>( aOut, aCondition, aValueIfTrue, std::get<_Ty>( aValueIfFalse ) );
    }

    void WhereOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aCondition, MultiTensor &aValueIfTrue, ScalarValue &aValueIfFalse )
    {
        DISPATCH_BY_TYPE( aTensorElementType, WhereTensorScalarImpl, ( aOut, aCondition, aValueIfTrue, aValueIfFalse ) );
    }

    template <typename _Ty> static void WhereVectorTensorImpl( MultiTensor &aOut, MultiTensor &aCondition, MemoryBuffer &aValueIfTrue, MultiTensor &aValueIfFalse )
    {
        int lBlockCount = ( aCondition.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aCondition.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::WhereVectorTensor<_Ty><<<lGridDim, lBlockDim>>>( aOut, aCondition, aValueIfTrue, aValueIfFalse );
    }
    void WhereOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aCondition, MemoryBuffer &aValueIfTrue, MultiTensor &aValueIfFalse )
    {
        DISPATCH_BY_TYPE( aTensorElementType, WhereVectorTensorImpl, ( aOut, aCondition, aValueIfTrue, aValueIfFalse ) );
    }

    template <typename _Ty> static void WhereVectorVectorImpl( MultiTensor &aOut, MultiTensor &aCondition, MemoryBuffer &aValueIfTrue, MemoryBuffer &aValueIfFalse )
    {
        int lBlockCount = ( aCondition.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aCondition.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::WhereVectorVector<_Ty><<<lGridDim, lBlockDim>>>( aOut, aCondition, aValueIfTrue, aValueIfFalse );
    }

    void WhereOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aCondition, MemoryBuffer &aValueIfTrue, MemoryBuffer &aValueIfFalse )
    {
        DISPATCH_BY_TYPE( aTensorElementType, WhereVectorVectorImpl, ( aOut, aCondition, aValueIfTrue, aValueIfFalse ) );
    }

    template <typename _Ty> static void WhereVectorScalarImpl( MultiTensor &aOut, MultiTensor &aCondition, MemoryBuffer &aValueIfTrue, ScalarValue &aValueIfFalse )
    {
        int lBlockCount = ( aCondition.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aCondition.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::WhereVectorScalar<_Ty><<<lGridDim, lBlockDim>>>( aOut, aCondition, aValueIfTrue, std::get<_Ty>( aValueIfFalse ) );
    }

    void WhereOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aCondition, MemoryBuffer &aValueIfTrue, ScalarValue &aValueIfFalse )
    {
        DISPATCH_BY_TYPE( aTensorElementType, WhereVectorScalarImpl, ( aOut, aCondition, aValueIfTrue, aValueIfFalse ) );
    }

    template <typename _Ty> static void WhereScalarTensorImpl( MultiTensor &aOut, MultiTensor &aCondition, ScalarValue &aValueIfTrue, MultiTensor &aValueIfFalse )
    {
        int lBlockCount = ( aCondition.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aCondition.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::WhereScalarTensor<_Ty><<<lGridDim, lBlockDim>>>( aOut, aCondition, std::get<_Ty>( aValueIfTrue ), aValueIfFalse );
    }

    void WhereOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aCondition, ScalarValue &aValueIfTrue, MultiTensor &aValueIfFalse )
    {
        DISPATCH_BY_TYPE( aTensorElementType, WhereScalarTensorImpl, ( aOut, aCondition, aValueIfTrue, aValueIfFalse ) );
    }

    template <typename _Ty> static void WhereScalarVectorImpl( MultiTensor &aOut, MultiTensor &aCondition, ScalarValue &aValueIfTrue, MemoryBuffer &aValueIfFalse )
    {
        int lBlockCount = ( aCondition.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aCondition.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::WhereScalarVector<_Ty><<<lGridDim, lBlockDim>>>( aOut, aCondition, std::get<_Ty>( aValueIfTrue ), aValueIfFalse );
    }

    void WhereOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aCondition, ScalarValue &aValueIfTrue, MemoryBuffer &aValueIfFalse )
    {
        DISPATCH_BY_TYPE( aTensorElementType, WhereScalarVectorImpl, ( aOut, aCondition, aValueIfTrue, aValueIfFalse ) );
    }

    template <typename _Ty> static void WhereScalarScalarImpl( MultiTensor &aOut, MultiTensor &aCondition, ScalarValue &aValueIfTrue, ScalarValue &aValueIfFalse )
    {
        int lBlockCount = ( aCondition.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aCondition.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::WhereScalarScalar<_Ty><<<lGridDim, lBlockDim>>>( aOut, aCondition, std::get<_Ty>( aValueIfTrue ), std::get<_Ty>( aValueIfFalse ) );
    }

    void WhereOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aCondition, ScalarValue &aValueIfTrue, ScalarValue &aValueIfFalse )
    {
        DISPATCH_BY_TYPE( aTensorElementType, WhereScalarScalarImpl, ( aOut, aCondition, aValueIfTrue, aValueIfFalse ) );
    }

    template <typename _Ty> static void RepeatOpImpl( MultiTensor &aOut, MultiTensor &aArray, MemoryBuffer &aRepetitions, uint32_t lMaxRepetitions )
    {
        int lBlockCount = ( lMaxRepetitions / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aArray.Shape().CountLayers(), aArray.Shape().mMaxBufferSize, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Repeat<_Ty><<<lGridDim, lBlockDim>>>( aOut, aArray, aRepetitions );
    }

    void RepeatOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aArray, MemoryBuffer &aRepetitions, uint32_t lMaxRepetitions )
    {
        DISPATCH_BY_TYPE( aTensorElementType, RepeatOpImpl, ( aOut, aArray, aRepetitions, lMaxRepetitions ) );
    }

    template <typename _Ty> static void TileOpImpl( MultiTensor &aOut, MultiTensor &aArray, MemoryBuffer &aRepetitions, uint32_t lMaxRepetitions )
    {
        int lBlockCount = ( aArray.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aArray.Shape().CountLayers(), lMaxRepetitions, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Tile<_Ty><<<lGridDim, lBlockDim>>>( aOut, aArray, aRepetitions );
    }

    void TileOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aArray, MemoryBuffer &aRepetitions, uint32_t lMaxRepetitions )
    {
        DISPATCH_BY_TYPE( aTensorElementType, TileOpImpl, ( aOut, aArray, aRepetitions, lMaxRepetitions ) );
    }

    template <typename _Ty> static void LinearSpaceOpImpl( MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, MemoryBuffer &aSubdivisions, uint32_t aMaxSubdivisions )
    {
        int lBlockCount = ( aMaxSubdivisions / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aLeft.Shape().CountLayers(), aLeft.Shape().mMaxBufferSize, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::LinearSpace<_Ty><<<lGridDim, lBlockDim>>>( aOut, aLeft, aRight, aSubdivisions );
    }

    void LinearSpaceOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, MemoryBuffer &aSubdivisions, uint32_t aMaxSubdivisions )
    {
        switch( aTensorElementType )
        {
        case eScalarType::FLOAT32:
        {
            LinearSpaceOpImpl<float>( aOut, aLeft, aRight, aSubdivisions, aMaxSubdivisions );
            break;
        }
        case eScalarType::FLOAT64:
        {
            LinearSpaceOpImpl<double>( aOut, aLeft, aRight, aSubdivisions, aMaxSubdivisions );
            break;
        }
        default:
            throw std::runtime_error( "Linear space only supports float and double values" );
        }
    }

    template <typename _Ty> static void MixImpl( MultiTensor &aOut, MultiTensor &A, MultiTensor &B, MultiTensor &t )
    {
        int lBlockCount = ( A.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( A.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Mix<_Ty><<<lGridDim, lBlockDim>>>( aOut, A, B, t );
    }

    void MixOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &A, MultiTensor &B, MultiTensor &t )
    {
        DISPATCH_BY_TYPE( aTensorElementType, MixImpl, ( aOut, A, B, t ) );
    }

    void Sample2DOp( MultiTensor &aOut, MultiTensor &X, MultiTensor &Y, MemoryBuffer &aTextures )
    {
        int lBlockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( X.Shape().CountLayers(), lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Sample2D<<<lGridDim, lBlockDim>>>( aOut, X, Y, aTextures );
    }

    void Sample2DOp( MultiTensor &aOut, MultiTensor &X, MemoryBuffer &Y, MemoryBuffer &aTextures )
    {
        int lBlockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( X.Shape().CountLayers(), lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Sample2D<<<lGridDim, lBlockDim>>>( aOut, X, Y, aTextures );
    }

    void Sample2DOp( MultiTensor &aOut, MultiTensor &X, ScalarValue &Y, MemoryBuffer &aTextures )
    {
        int lBlockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( X.Shape().CountLayers(), lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Sample2D<<<lGridDim, lBlockDim>>>( aOut, X, std::get<float>( Y ), aTextures );
    }

    void Sample2DOp( MultiTensor &aOut, MemoryBuffer &X, MultiTensor &Y, MemoryBuffer &aTextures )
    {
        int lBlockCount = ( Y.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( Y.Shape().CountLayers(), lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Sample2D<<<lGridDim, lBlockDim>>>( aOut, X, Y, aTextures );
    }

    void Sample2DOp( MultiTensor &aOut, ScalarValue &X, MultiTensor &Y, MemoryBuffer &aTextures )
    {
        int lBlockCount = ( Y.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( Y.Shape().CountLayers(), lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Sample2D<<<lGridDim, lBlockDim>>>( aOut, std::get<float>( X ), Y, aTextures );
    }


    template <typename _Ty> static void ToFixedPointOpImpl( MultiTensor &aOut, eScalarType aOutputElementType, MultiTensor &aArray, _Ty aScaling )
    {
        int lBlockCount = ( aArray.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aArray.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        switch( aOutputElementType )
        {
        case eScalarType::UINT8:
        {
            Kernels::ToFixedPoint<_Ty, uint8_t><<<lGridDim, lBlockDim>>>( aOut, aArray, aScaling );
            break;
        }
        case eScalarType::UINT16:
        {
            Kernels::ToFixedPoint<_Ty, uint16_t><<<lGridDim, lBlockDim>>>( aOut, aArray, aScaling );
            break;
        }
        case eScalarType::UINT32:
        {
            Kernels::ToFixedPoint<_Ty, uint32_t><<<lGridDim, lBlockDim>>>( aOut, aArray, aScaling );
            break;
        }
        case eScalarType::UINT64:
        {
            Kernels::ToFixedPoint<_Ty, uint64_t><<<lGridDim, lBlockDim>>>( aOut, aArray, aScaling );
            break;
        }
        case eScalarType::INT8:
        {
            Kernels::ToFixedPoint<_Ty, int8_t><<<lGridDim, lBlockDim>>>( aOut, aArray, aScaling );
            break;
        }
        case eScalarType::INT16:
        {
            Kernels::ToFixedPoint<_Ty, int16_t><<<lGridDim, lBlockDim>>>( aOut, aArray, aScaling );
            break;
        }
        case eScalarType::INT32:
        {
            Kernels::ToFixedPoint<_Ty, int32_t><<<lGridDim, lBlockDim>>>( aOut, aArray, aScaling );
            break;
        }
        case eScalarType::INT64:
        {
            Kernels::ToFixedPoint<_Ty, int64_t><<<lGridDim, lBlockDim>>>( aOut, aArray, aScaling );
            break;
        }
        default:
            throw std::runtime_error( "Linear space only supports float and double values" );
        }
    }

    void ToFixedPointOp( eScalarType aTensorElementType, MultiTensor &aOut, eScalarType aOutputElementType, MultiTensor &aArray, ScalarValue &aScaling )
    {
        switch( aTensorElementType )
        {
        case eScalarType::FLOAT32:
        {
            ToFixedPointOpImpl<float>( aOut, aOutputElementType, aArray, std::get<float>( aScaling ) );
            break;
        }
        case eScalarType::FLOAT64:
        {
            ToFixedPointOpImpl<double>( aOut, aOutputElementType, aArray, std::get<double>( aScaling ) );
            break;
        }
        default:
            throw std::runtime_error( "Linear space only supports float and double values" );
        }
    }

    template <typename _Ty> static void AffineTransformImpl( MultiTensor &aOut, MultiTensor &A, MultiTensor &X, MultiTensor &B )
    {
        int lBlockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( X.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::AffineTransform<_Ty><<<lGridDim, lBlockDim>>>( aOut, A, X, B );
    }

    template <typename _Ty> static void AffineTransformImpl( MultiTensor &aOut, MultiTensor &A, MultiTensor &X, MemoryBuffer &B )
    {
        int lBlockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( X.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::AffineTransform<_Ty><<<lGridDim, lBlockDim>>>( aOut, A, X, B );
    }

    template <typename _Ty> static void AffineTransformImpl( MultiTensor &aOut, MultiTensor &A, MultiTensor &X, ScalarValue &B )
    {
        int lBlockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( X.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::AffineTransform<_Ty><<<lGridDim, lBlockDim>>>( aOut, A, X, std::get<_Ty>( B ) );
    }

    template <typename _Ty> static void AffineTransformImpl( MultiTensor &aOut, MemoryBuffer &A, MultiTensor &X, MultiTensor &B )
    {
        int lBlockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( X.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::AffineTransform<_Ty><<<lGridDim, lBlockDim>>>( aOut, A, X, B );
    }

    template <typename _Ty> static void AffineTransformImpl( MultiTensor &aOut, MemoryBuffer &A, MultiTensor &X, MemoryBuffer &B )
    {
        int lBlockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( X.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::AffineTransform<_Ty><<<lGridDim, lBlockDim>>>( aOut, A, X, B );
    }

    template <typename _Ty> static void AffineTransformImpl( MultiTensor &aOut, MemoryBuffer &A, MultiTensor &X, ScalarValue &B )
    {
        int lBlockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( X.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::AffineTransform<_Ty><<<lGridDim, lBlockDim>>>( aOut, A, X, std::get<_Ty>( B ) );
    }

    template <typename _Ty> static void AffineTransformImpl( MultiTensor &aOut, ScalarValue &A, MultiTensor &X, MultiTensor &B )
    {
        int lBlockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( X.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::AffineTransform<_Ty><<<lGridDim, lBlockDim>>>( aOut, std::get<_Ty>( A ), X, B );
    }

    template <typename _Ty> static void AffineTransformImpl( MultiTensor &aOut, ScalarValue &A, MultiTensor &X, MemoryBuffer &B )
    {
        int lBlockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( X.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::AffineTransform<_Ty><<<lGridDim, lBlockDim>>>( aOut, std::get<_Ty>( A ), X, B );
    }

    template <typename _Ty> static void AffineTransformImpl( MultiTensor &aOut, ScalarValue &A, MultiTensor &X, ScalarValue &B )
    {
        int lBlockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( X.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::AffineTransform<_Ty><<<lGridDim, lBlockDim>>>( aOut, std::get<_Ty>( A ), X, std::get<_Ty>( B ) );
    }

    void AffineTransformOp( eScalarType aOutputElementType, MultiTensor &aOut, MultiTensor &A, MultiTensor &X, MultiTensor &B )
    {
        DISPATCH_BY_TYPE( aOutputElementType, AffineTransformImpl, ( aOut, A, X, B ) );
    }

    void AffineTransformOp( eScalarType aOutputElementType, MultiTensor &aOut, MultiTensor &A, MultiTensor &X, MemoryBuffer &B )
    {
        DISPATCH_BY_TYPE( aOutputElementType, AffineTransformImpl, ( aOut, A, X, B ) );
    }

    void AffineTransformOp( eScalarType aOutputElementType, MultiTensor &aOut, MultiTensor &A, MultiTensor &X, ScalarValue &B )
    {
        DISPATCH_BY_TYPE( aOutputElementType, AffineTransformImpl, ( aOut, A, X, B ) );
    }

    void AffineTransformOp( eScalarType aOutputElementType, MultiTensor &aOut, MemoryBuffer &A, MultiTensor &X, MultiTensor &B )
    {
        DISPATCH_BY_TYPE( aOutputElementType, AffineTransformImpl, ( aOut, A, X, B ) );
    }

    void AffineTransformOp( eScalarType aOutputElementType, MultiTensor &aOut, MemoryBuffer &A, MultiTensor &X, MemoryBuffer &B )
    {
        DISPATCH_BY_TYPE( aOutputElementType, AffineTransformImpl, ( aOut, A, X, B ) );
    }

    void AffineTransformOp( eScalarType aOutputElementType, MultiTensor &aOut, MemoryBuffer &A, MultiTensor &X, ScalarValue &B )
    {
        DISPATCH_BY_TYPE( aOutputElementType, AffineTransformImpl, ( aOut, A, X, B ) );
    }

    void AffineTransformOp( eScalarType aOutputElementType, MultiTensor &aOut, ScalarValue &A, MultiTensor &X, MultiTensor &B )
    {
        DISPATCH_BY_TYPE( aOutputElementType, AffineTransformImpl, ( aOut, A, X, B ) );
    }

    void AffineTransformOp( eScalarType aOutputElementType, MultiTensor &aOut, ScalarValue &A, MultiTensor &X, MemoryBuffer &B )
    {
        DISPATCH_BY_TYPE( aOutputElementType, AffineTransformImpl, ( aOut, A, X, B ) );
    }

    void AffineTransformOp( eScalarType aOutputElementType, MultiTensor &aOut, ScalarValue &A, MultiTensor &X, ScalarValue &B )
    {
        DISPATCH_BY_TYPE( aOutputElementType, AffineTransformImpl, ( aOut, A, X, B ) );
    }

    void FloorOp( MultiTensor &aOut, MultiTensor &aX )
    {
        int lBlockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Floor<<<lGridDim, lBlockDim>>>( aOut, aX );
    }

    void CeilOp( MultiTensor &aOut, MultiTensor &aX )
    {
        int lBlockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Ceil<<<lGridDim, lBlockDim>>>( aOut, aX );
    }

    template <typename _Ty> void AbsImpl( MultiTensor &aOut, MultiTensor &aX )
    {
        int lBlockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Abs<_Ty><<<lGridDim, lBlockDim>>>( aOut, aX );
    }

    void AbsOp( eScalarType aOutputElementType, MultiTensor &aOut, MultiTensor &aX ) { DISPATCH_BY_SIGNED_TYPE( aOutputElementType, AbsImpl, ( aOut, aX ) ); }

    template <typename _Ty> void SqrtImpl( MultiTensor &aOut, MultiTensor &aX )
    {
        int lBlockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Sqrt<_Ty><<<lGridDim, lBlockDim>>>( aOut, aX );
    }

    void SqrtOp( eScalarType aOutputElementType, MultiTensor &aOut, MultiTensor &aX ) { DISPATCH_BY_TYPE( aOutputElementType, SqrtImpl, ( aOut, aX ) ); }

    template <typename _Ty> void RoundImpl( MultiTensor &aOut, MultiTensor &aX )
    {
        int lBlockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Round<_Ty><<<lGridDim, lBlockDim>>>( aOut, aX );
    }

    void RoundOp( eScalarType aOutputElementType, MultiTensor &aOut, MultiTensor &aX ) { DISPATCH_BY_TYPE( aOutputElementType, RoundImpl, ( aOut, aX ) ); }

    void CountTrueOp( MultiTensor &aOut, MultiTensor &aX, MemoryBuffer &aBlockSizes, MemoryBuffer &aElementCount, uint32_t aMaxBlockSize )
    {
        int lBlockCount = ( aMaxBlockSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::CountNonZero<uint8_t><<<lGridDim, lBlockDim>>>( aOut, aX, aBlockSizes, aElementCount );
    }

    template <typename _Ty> void CountNonZeroImpl( MultiTensor &aOut, MultiTensor &aX, MemoryBuffer &aBlockSizes, MemoryBuffer &aElementCount, uint32_t aMaxBlockSize )
    {
        int lBlockCount = ( aMaxBlockSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::CountNonZero<_Ty><<<lGridDim, lBlockDim>>>( aOut, aX, aBlockSizes, aElementCount );
    }

    void CountNonZeroOp( eScalarType aOutputElementType, MultiTensor &aOut, MultiTensor &aX, MemoryBuffer &aBlockSizes, MemoryBuffer &aElementCount, uint32_t aMaxBlockSize )
    {
        DISPATCH_BY_TYPE( aOutputElementType, CountNonZeroImpl, ( aOut, aX, aBlockSizes, aElementCount, aMaxBlockSize ) );
    }

    template <typename _Ty> void CountZeroImpl( MultiTensor &aOut, MultiTensor &aX, MemoryBuffer &aBlockSizes, MemoryBuffer &aElementCount, uint32_t aMaxBlockSize )
    {
        int lBlockCount = ( aMaxBlockSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::CountZero<_Ty><<<lGridDim, lBlockDim>>>( aOut, aX, aBlockSizes, aElementCount );
    }

    void CountZeroOp( eScalarType aOutputElementType, MultiTensor &aOut, MultiTensor &aX, MemoryBuffer &aBlockSizes, MemoryBuffer &aElementCount, uint32_t aMaxBlockSize )
    {
        DISPATCH_BY_TYPE( aOutputElementType, CountZeroImpl, ( aOut, aX, aBlockSizes, aElementCount, aMaxBlockSize ) );
    }

    template <typename _Ty>
    void ArraySummationImpl( MultiTensor &aOut, MultiTensor &aX, MemoryBuffer &aBegin, MemoryBuffer &aEnd, MemoryBuffer &aElementCount, MemoryBuffer &aBlockSizes,
                             uint32_t aMaxBlockSize )
    {
        int lBlockCount = ( aMaxBlockSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::ArraySummation<_Ty><<<lGridDim, lBlockDim>>>( aOut, aX, aBegin, aEnd, aElementCount, aBlockSizes );
    }

    void ArraySummationOp( eScalarType aOutputElementType, MultiTensor &aOut, MultiTensor &aX, MemoryBuffer &aBegin, MemoryBuffer &aEnd, MemoryBuffer &aElementCount,
                           MemoryBuffer &aBlockSizes, uint32_t aMaxBlockSize )
    {
        DISPATCH_BY_TYPE( aOutputElementType, ArraySummationImpl, ( aOut, aX, aBegin, aEnd, aElementCount, aBlockSizes, aMaxBlockSize ) );
    }

    template <typename _Ty>
    void ArraySliceImpl( MultiTensor &aOut, MultiTensor &aX, MemoryBuffer &aBegin, MemoryBuffer &aEnd, MemoryBuffer &aElementCount, MemoryBuffer &aBlockSizes,
                         uint32_t aMaxBlockSize )
    {
        int lBlockCount = ( aMaxBlockSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::ArraySlice<_Ty><<<lGridDim, lBlockDim>>>( aOut, aX, aBegin, aEnd, aElementCount, aBlockSizes );
    }

    void ArraySliceOp( eScalarType aOutputElementType, MultiTensor &aOut, MultiTensor &aX, MemoryBuffer &aBegin, MemoryBuffer &aEnd, MemoryBuffer &aElementCount,
                       MemoryBuffer &aBlockSizes, uint32_t aMaxBlockSize )
    {
        DISPATCH_BY_TYPE( aOutputElementType, ArraySliceImpl, ( aOut, aX, aBegin, aEnd, aElementCount, aBlockSizes, aMaxBlockSize ) );
    }

    template <typename _Ty> void DiffImpl( MultiTensor &aOut, MultiTensor &aX, uint32_t aCount, MemoryBuffer &aElementCount, MemoryBuffer &aBlockSizes, uint32_t aMaxBlockSize )
    {
        int lBlockCount = ( aMaxBlockSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Diff<_Ty><<<lGridDim, lBlockDim>>>( aOut, aX, aCount, aElementCount, aBlockSizes );
    }

    void DiffOp( eScalarType aOutputElementType, MultiTensor &aOut, MultiTensor &aX, uint32_t aCount, MemoryBuffer &aElementCount, MemoryBuffer &aBlockSizes,
                 uint32_t aMaxBlockSize )
    {
        DISPATCH_BY_TYPE( aOutputElementType, DiffImpl, ( aOut, aX, aCount, aElementCount, aBlockSizes, aMaxBlockSize ) );
    }

    template <typename _Ty>
    void ShiftImpl( MultiTensor &aOut, MultiTensor &aX, int32_t aCount, ScalarValue &aFillValue, MemoryBuffer &aElementCount, MemoryBuffer &aBlockSizes, uint32_t aMaxBlockSize )
    {
        int lBlockCount = ( aMaxBlockSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        if (aCount < 0)
            Kernels::ShiftLeft<_Ty><<<lGridDim, lBlockDim>>>( aOut, aX, -aCount, std::get<_Ty>( aFillValue ), aElementCount, aBlockSizes );
        else
            Kernels::ShiftRight<_Ty><<<lGridDim, lBlockDim>>>( aOut, aX, aCount, std::get<_Ty>( aFillValue ), aElementCount, aBlockSizes );
    }

    void ShiftOp( eScalarType aOutputElementType, MultiTensor &aOut, MultiTensor &aX, int32_t aCount, ScalarValue &aFillValue, MemoryBuffer &aElementCount,
                  MemoryBuffer &aBlockSizes, uint32_t aMaxBlockSize )
    {
        DISPATCH_BY_TYPE( aOutputElementType, ShiftImpl, ( aOut, aX, aCount, aFillValue, aElementCount, aBlockSizes, aMaxBlockSize ) );
    }

    template <typename _Ty>
    void Conv1DImpl( MultiTensor &aOut, MultiTensor &aArray0, MemoryBuffer &aElementCount0, MemoryBuffer &aBlockSizes0, uint32_t aMaxElementCount0, uint32_t aMaxBlockSize0,
                     MultiTensor &aArray1, MemoryBuffer &aElementCount1, MemoryBuffer aBlockSizes1, uint32_t aMaxBlockSize1 )
    {
        int lBlockCount = ( aMaxElementCount0 / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aArray0.Shape().CountLayers(), aMaxBlockSize0, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Conv1D<_Ty><<<lGridDim, lBlockDim>>>( aOut, aArray0, aElementCount0, aBlockSizes0, aArray1, aElementCount1, aBlockSizes1 );
    }

    void Conv1DOp( eScalarType aOutputElementType, MultiTensor &aOut, MultiTensor &aArray0, MemoryBuffer &aElementCount0, MemoryBuffer &aBlockSizes0, uint32_t aMaxElementCount0,
                   uint32_t aMaxBlockSize0, MultiTensor &aArray1, MemoryBuffer &aElementCount1, MemoryBuffer &aBlockSizes1, uint32_t aMaxBlockSize1 )
    {
        DISPATCH_BY_TYPE( aOutputElementType, Conv1DImpl,
                          ( aOut, aArray0, aElementCount0, aBlockSizes0, aMaxElementCount0, aMaxBlockSize0, aArray1, aElementCount1, aBlockSizes1, aMaxBlockSize1 ) );
    }

    template <typename _Ty>
    void HCatImpl( MultiTensor &aOut, MultiTensor &aArray0, MemoryBuffer &aElementCount0, MultiTensor &aArray1, MemoryBuffer &aElementCount1, MemoryBuffer &aBlockSizes,
                   uint32_t aMaxBlockSize )
    {
        int lBlockCount = ( aMaxBlockSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aArray0.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::HCat<_Ty><<<lGridDim, lBlockDim>>>( aOut, aArray0, aElementCount0, aArray1, aElementCount1, aBlockSizes );
    }

    void HCatOp( eScalarType aOutputElementType, MultiTensor &aOut, MultiTensor &aArray0, MemoryBuffer &aElementCount0, MultiTensor &aArray1, MemoryBuffer &aElementCount1,
                 MemoryBuffer &aBlockSizes0, uint32_t aMaxBlockSize0 )
    {
        DISPATCH_BY_TYPE( aOutputElementType, HCatImpl, ( aOut, aArray0, aElementCount0, aArray1, aElementCount1, aBlockSizes0, aMaxBlockSize0 ) );
    }

} // namespace LTSE::TensorOps