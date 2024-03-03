/// @file   KernelLaunchers.cu
///
/// @brief  C++ API for Cuda computation launchers
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert. All rights reserved.

#include "KernelLaunchers.h"

#include <chrono>

#include <cuda.h>
#include <curand.h>
#include <stdexcept>
#include <variant>

#include "Core/Logging.h"

#include "HelperMacros.h"

#include "DeviceKernels.inl"

namespace SE::TensorOps
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

        ~RandomNumberGenerator()
        {
            curandDestroyGenerator( Generator );
        }
    };

    template <typename _Ty>
    static void ConstantFillImpl( multi_tensor_t &aArray, scalar_value_t &aConstant )
    {
        int lBlockCount = ( aArray.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aArray.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::ConstantFill<_Ty><<<lGridDim, lBlockDim>>>( aArray, std::get<_Ty>( aConstant ) );
    }

    template <typename _Ty>
    static void ConstantFillImpl( multi_tensor_t &aArray, memory_buffer_t &aInitialValues )
    {
        int lBlockCount = ( aArray.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aArray.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::ConstantFill<_Ty><<<lGridDim, lBlockDim>>>( aArray, aInitialValues );
    }

    void ConstantFill( scalar_type_t aTensorElementType, multi_tensor_t &aArray, memory_buffer_t &aInitialValues )
    {
        DISPATCH_BY_TYPE( aTensorElementType, ConstantFillImpl, ( aArray, aInitialValues ) );
    }

    void ConstantFill( scalar_type_t aTensorElementType, multi_tensor_t &aArray, scalar_value_t &aInitialValues )
    {
        DISPATCH_BY_TYPE( aTensorElementType, ConstantFillImpl, ( aArray, aInitialValues ) );
    }

    void RandomUniformFill( scalar_type_t aTensorElementType, multi_tensor_t &aArray )
    {
        switch( aTensorElementType )
        {
        case scalar_type_t::FLOAT32:
        {
            RandomNumberGenerator lGenerator{};
            CURAND_ASSERT( curandGenerateUniform( lGenerator.Generator, aArray.DataAs<float>(), aArray.SizeAs<float>() ) );
        }
        break;
        case scalar_type_t::FLOAT64:
        {
            RandomNumberGenerator lGenerator{};
            CURAND_ASSERT( curandGenerateUniformDouble( lGenerator.Generator, aArray.DataAs<double>(), aArray.SizeAs<double>() ) );
        }
        break;
        default:
            std::runtime_error( "Random number type can only be float or double" );
        }
    }

    void RandomNormalFill( scalar_type_t aTensorElementType, multi_tensor_t &aArray, scalar_value_t &aMu, scalar_value_t &aSigma )
    {
        switch( aTensorElementType )
        {
        case scalar_type_t::FLOAT32:
        {
            float lMean = std::get<float>( aMu );
            float lStd  = std::get<float>( aSigma );
            if( lStd <= 0.0f )
                std::runtime_error( "Variance parameter should be strictly positive" );
            RandomNumberGenerator lGenerator{};
            CURAND_ASSERT( curandGenerateNormal( lGenerator.Generator, aArray.DataAs<float>(), aArray.SizeAs<float>(), lMean, lStd ) );
        }
        break;
        case scalar_type_t::FLOAT64:
        {
            double lMean = std::get<double>( aMu );
            double lStd  = std::get<double>( aSigma );
            if( lStd <= 0.0f )
                std::runtime_error( "Variance parameter should be strictly positive" );
            RandomNumberGenerator lGenerator{};
            CURAND_ASSERT(
                curandGenerateNormalDouble( lGenerator.Generator, aArray.DataAs<double>(), aArray.SizeAs<double>(), lMean, lStd ) );
        }
        break;
        default:
            std::runtime_error( "Random number type can only be float or double" );
        }
    }

    template <typename _Ty>
    static void ARangeOpImpl( multi_tensor_t &aOut, memory_buffer_t &aLeft, memory_buffer_t &aRight, memory_buffer_t &aDelta,
                              uint32_t aMaxSubdivisions )
    {
        int lBlockCount = ( aMaxSubdivisions / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aOut.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::ARange<_Ty><<<lGridDim, lBlockDim>>>( aOut, aLeft, aRight, aDelta );
    }

    void ARangeOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, memory_buffer_t &aRight, memory_buffer_t &aDelta,
                   uint32_t aMaxSubdivisions )
    {
        switch( aTensorElementType )
        {
        case scalar_type_t::FLOAT32:
        {
            ARangeOpImpl<float>( aOut, aLeft, aRight, aDelta, aMaxSubdivisions );
            break;
        }
        case scalar_type_t::FLOAT64:
        {
            ARangeOpImpl<double>( aOut, aLeft, aRight, aDelta, aMaxSubdivisions );
            break;
        }
        default:
            throw std::runtime_error( "Linear space only supports float and double values" );
        }
    }

    template <typename _Ty>
    static void AddArrayToArrayImpl( multi_tensor_t &aOut, multi_tensor_t &aIn, multi_tensor_t &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Add<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant );
    }

    template <typename _Ty>
    static void AddArrayToArrayImpl( multi_tensor_t &aOut, multi_tensor_t &aIn, multi_tensor_t &aConstant, broadcast_hint_t aBroadcastHint,
                                     memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize, memory_buffer_t &aBroadcastSizes,
                                     uint32_t aMaxBroadcastSizes )
    {
        int lBlockCount = ( aMaxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), aMaxBlockSize, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Add<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant, aBroadcastHint, aBlockSizes, aBroadcastSizes );
    }

    template <typename _ScalarType>
    static void AddScalarToArrayImpl( multi_tensor_t &aOut, multi_tensor_t &aArray, scalar_value_t &aConstant )
    {
        int lBlockCount = ( aArray.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aArray.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Add<_ScalarType><<<lGridDim, lBlockDim>>>( aOut, aArray, std::get<_ScalarType>( aConstant ) );
    }

    template <typename _Ty>
    static void AddArrayToVectorImpl( multi_tensor_t &aOut, multi_tensor_t &aIn, memory_buffer_t &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Add<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant );
    }

    void AddOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, AddArrayToArrayImpl, ( aOut, aLeft, aRight ) );
    }

    void AddOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                broadcast_hint_t aBroadcastHint, memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize, memory_buffer_t &aBroadcastSizes,
                uint32_t aMaxBroadcastSizes )
    {
        DISPATCH_BY_TYPE( aTensorElementType, AddArrayToArrayImpl,
                          ( aOut, aLeft, aRight, aBroadcastHint, aBlockSizes, aMaxBlockSize, aBroadcastSizes, aMaxBroadcastSizes ) );
    }

    void AddOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, scalar_value_t &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, AddScalarToArrayImpl, ( aOut, aLeft, aRight ) );
    }

    void AddOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, AddArrayToVectorImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _ScalarType>
    void MultiplyArrayByScalarImpl( multi_tensor_t &aOut, multi_tensor_t &aIn, scalar_value_t &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Multiply<_ScalarType><<<lGridDim, lBlockDim>>>( aOut, aIn, std::get<_ScalarType>( aConstant ) );
    }

    template <typename _Ty>
    static void MultiplyArrayByArrayImpl( multi_tensor_t &aOut, multi_tensor_t &aIn, multi_tensor_t &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Multiply<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant );
    }

    template <typename _Ty>
    static void MultiplyArrayByArrayImpl( multi_tensor_t &aOut, multi_tensor_t &aIn, multi_tensor_t &aConstant, broadcast_hint_t aBroadcastHint,
                                          memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize, memory_buffer_t &aBroadcastSizes,
                                          uint32_t aMaxBroadcastSizes )
    {
        int lBlockCount = ( aMaxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), aMaxBlockSize, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Multiply<_Ty>
            <<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant, aBroadcastHint, aBlockSizes, aBroadcastSizes );
    }

    template <typename _Ty>
    static void MultiplyArrayByVectorImpl( multi_tensor_t &aOut, multi_tensor_t &aIn, memory_buffer_t &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Multiply<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant );
    }

    void MultiplyOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, scalar_value_t &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, MultiplyArrayByScalarImpl, ( aOut, aLeft, aRight ) );
    }

    void MultiplyOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, MultiplyArrayByArrayImpl, ( aOut, aLeft, aRight ) );
    }

    void MultiplyOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                     broadcast_hint_t aBroadcastHint, memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize, memory_buffer_t &aBroadcastSizes,
                     uint32_t aMaxBroadcastSizes )
    {
        DISPATCH_BY_TYPE( aTensorElementType, MultiplyArrayByArrayImpl,
                          ( aOut, aLeft, aRight, aBroadcastHint, aBlockSizes, aMaxBlockSize, aBroadcastSizes, aMaxBroadcastSizes ) );
    }

    void MultiplyOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, MultiplyArrayByVectorImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _Ty>
    void SubtractArrayFromScalarImpl( multi_tensor_t &aOut, scalar_value_t &aConstant, multi_tensor_t &aIn )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Subtract<<<lGridDim, lBlockDim>>>( aOut, std::get<_Ty>( aConstant ), aIn );
    }

    template <typename _Ty>
    void SubtractScalarFromArrayImpl( multi_tensor_t &aOut, multi_tensor_t &aIn, scalar_value_t &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Subtract<<<lGridDim, lBlockDim>>>( aOut, aIn, std::get<_Ty>( aConstant ) );
    }

    template <typename _Ty>
    static void SubtractVectorFromArrayImpl( multi_tensor_t &aOut, multi_tensor_t &aIn, memory_buffer_t &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Subtract<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant );
    }

    template <typename _Ty>
    static void SubtractArrayfromArrayImpl( multi_tensor_t &aOut, multi_tensor_t &aIn, multi_tensor_t &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Subtract<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant );
    }

    template <typename _Ty>
    static void SubtractArrayfromArrayImpl( multi_tensor_t &aOut, multi_tensor_t &aIn, multi_tensor_t &aConstant, broadcast_hint_t aBroadcastHint,
                                            memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize, memory_buffer_t &aBroadcastSizes,
                                            uint32_t aMaxBroadcastSizes )
    {
        int lBlockCount = ( aMaxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), aMaxBlockSize, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Subtract<_Ty>
            <<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant, aBroadcastHint, aBlockSizes, aBroadcastSizes );
    }

    template <typename _Ty>
    static void SubtractArrayFromVectorImpl( multi_tensor_t &aOut, memory_buffer_t &aConstant, multi_tensor_t &aIn )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Subtract<_Ty><<<lGridDim, lBlockDim>>>( aOut, aConstant, aIn );
    }

    void SubtractOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, scalar_value_t &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, SubtractScalarFromArrayImpl, ( aOut, aLeft, aRight ) );
    }

    void SubtractOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aLeft, multi_tensor_t &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, SubtractArrayFromScalarImpl, ( aOut, aLeft, aRight ) );
    }

    void SubtractOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, SubtractArrayfromArrayImpl, ( aOut, aLeft, aRight ) );
    }

    void SubtractOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                     broadcast_hint_t aBroadcastHint, memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize, memory_buffer_t &aBroadcastSizes,
                     uint32_t aMaxBroadcastSizes )
    {
        DISPATCH_BY_TYPE( aTensorElementType, SubtractArrayfromArrayImpl,
                          ( aOut, aLeft, aRight, aBroadcastHint, aBlockSizes, aMaxBlockSize, aBroadcastSizes, aMaxBroadcastSizes ) );
    }

    void SubtractOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, SubtractVectorFromArrayImpl, ( aOut, aLeft, aRight ) );
    }

    void SubtractOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, SubtractArrayFromVectorImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _Ty>
    static void DivideArrayByScalarImpl( multi_tensor_t &aOut, multi_tensor_t &aIn, scalar_value_t &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Divide<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, std::get<_Ty>( aConstant ) );
    }

    template <typename _Ty>
    static void DivideScalarByArrayImpl( multi_tensor_t &aOut, scalar_value_t &aConstant, multi_tensor_t &aIn )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Divide<_Ty><<<lGridDim, lBlockDim>>>( aOut, std::get<_Ty>( aConstant ), aIn );
    }

    template <typename _Ty>
    static void DivideArrayfromArrayImpl( multi_tensor_t &aOut, multi_tensor_t &aIn, multi_tensor_t &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Divide<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant );
    }

    template <typename _Ty>
    static void DivideArrayfromArrayImpl( multi_tensor_t &aOut, multi_tensor_t &aIn, multi_tensor_t &aConstant, broadcast_hint_t aBroadcastHint,
                                          memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize, memory_buffer_t &aBroadcastSizes,
                                          uint32_t aMaxBroadcastSizes )
    {
        int lBlockCount = ( aMaxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), aMaxBlockSize, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Divide<_Ty>
            <<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant, aBroadcastHint, aBlockSizes, aBroadcastSizes );
    }

    template <typename _Ty>
    static void DivideArrayByVectorImpl( multi_tensor_t &aOut, multi_tensor_t &aIn, memory_buffer_t &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Divide<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant );
    }

    template <typename _Ty>
    static void DivideVectorByArrayImpl( multi_tensor_t &aOut, memory_buffer_t &aConstant, multi_tensor_t &aIn )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Divide<_Ty><<<lGridDim, lBlockDim>>>( aOut, aConstant, aIn );
    }

    void DivideOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, scalar_value_t &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, DivideArrayByScalarImpl, ( aOut, aLeft, aRight ) );
    }

    void DivideOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aLeft, multi_tensor_t &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, DivideScalarByArrayImpl, ( aOut, aLeft, aRight ) );
    }

    void DivideOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, DivideArrayfromArrayImpl, ( aOut, aLeft, aRight ) );
    }

    void DivideOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                   broadcast_hint_t aBroadcastHint, memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize, memory_buffer_t &aBroadcastSizes,
                   uint32_t aMaxBroadcastSizes )
    {
        DISPATCH_BY_TYPE( aTensorElementType, DivideArrayfromArrayImpl,
                          ( aOut, aLeft, aRight, aBroadcastHint, aBlockSizes, aMaxBlockSize, aBroadcastSizes, aMaxBroadcastSizes ) );
    }

    void DivideOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, DivideArrayByVectorImpl, ( aOut, aLeft, aRight ) );
    }

    void DivideOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, DivideVectorByArrayImpl, ( aOut, aLeft, aRight ) );
    }

    void AndOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, scalar_value_t &aRight )
    {
        int lBlockCount = ( aLeft.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aLeft.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::And<<<lGridDim, lBlockDim>>>( aOut, aLeft, std::get<uint8_t>( aRight ) );
    }

    void AndOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aLeft, multi_tensor_t &aRight )
    {
        AndOp( aTensorElementType, aOut, aRight, aLeft );
    }

    void AndOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aIn, multi_tensor_t &aConstant,
                broadcast_hint_t aBroadcastHint, memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize, memory_buffer_t &aBroadcastSizes,
                uint32_t aMaxBroadcastSizes )
    {
        int lBlockCount = ( aMaxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), aMaxBlockSize, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::And<<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant, aBroadcastHint, aBlockSizes, aBroadcastSizes );
    }

    void AndOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight )
    {
        int lBlockCount = ( aLeft.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aLeft.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::And<<<lGridDim, lBlockDim>>>( aOut, aLeft, aRight );
    }

    void AndOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight )
    {
        int lBlockCount = ( aLeft.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aLeft.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::And<<<lGridDim, lBlockDim>>>( aOut, aLeft, aRight );
    }

    void AndOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight )
    {
        AndOp( aTensorElementType, aOut, aRight, aLeft );
    }

    void OrOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, scalar_value_t &aRight )
    {
        int lBlockCount = ( aLeft.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aLeft.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Or<<<lGridDim, lBlockDim>>>( aOut, aLeft, std::get<uint8_t>( aRight ) );
    }

    void OrOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aLeft, multi_tensor_t &aRight )
    {
        OrOp( aTensorElementType, aOut, aRight, aLeft );
    }

    void OrOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight )
    {
        int lBlockCount = ( aLeft.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aLeft.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Or<<<lGridDim, lBlockDim>>>( aOut, aLeft, aRight );
    }

    void OrOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aIn, multi_tensor_t &aConstant,
               broadcast_hint_t aBroadcastHint, memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize, memory_buffer_t &aBroadcastSizes,
               uint32_t aMaxBroadcastSizes )
    {
        int lBlockCount = ( aMaxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), aMaxBlockSize, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Or<<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant, aBroadcastHint, aBlockSizes, aBroadcastSizes );
    }

    void OrOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight )
    {
        int lBlockCount = ( aLeft.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aLeft.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Or<<<lGridDim, lBlockDim>>>( aOut, aLeft, aRight );
    }

    void OrOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight )
    {
        OrOp( aTensorElementType, aOut, aRight, aLeft );
    }

    void NotOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aOperand )
    {
        int lBlockCount = ( aOperand.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aOperand.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Not<<<lGridDim, lBlockDim>>>( aOut, aOperand );
    }

    template <typename _Ty>
    void BitwiseAnd_Tensor_Scalar_Impl( multi_tensor_t &aOut, multi_tensor_t &aIn, scalar_value_t &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::BitwiseAnd<<<lGridDim, lBlockDim>>>( aOut, aIn, std::get<_Ty>( aConstant ) );
    }

    void BitwiseAndOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, scalar_value_t &aRight )
    {
        DISPATCH_BY_INTEGRAL_TYPE( aTensorElementType, BitwiseAnd_Tensor_Scalar_Impl, ( aOut, aLeft, aRight ) );
    }

    void BitwiseAndOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aLeft, multi_tensor_t &aRight )
    {
        BitwiseAndOp( aTensorElementType, aOut, aRight, aLeft );
    }

    template <typename _Ty>
    static void BitwiseAnd_Tensor_Tensor_Impl( multi_tensor_t &aOut, multi_tensor_t &aIn, multi_tensor_t &aConstant,
                                               broadcast_hint_t aBroadcastHint, memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize,
                                               memory_buffer_t &aBroadcastSizes, uint32_t aMaxBroadcastSizes )
    {
        int lBlockCount = ( aMaxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), aMaxBlockSize, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::BitwiseAnd<_Ty>
            <<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant, aBroadcastHint, aBlockSizes, aBroadcastSizes );
    }

    void BitwiseAndOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                       broadcast_hint_t aBroadcastHint, memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize, memory_buffer_t &aBroadcastSizes,
                       uint32_t aMaxBroadcastSizes )
    {
        DISPATCH_BY_INTEGRAL_TYPE(
            aTensorElementType, BitwiseAnd_Tensor_Tensor_Impl,
            ( aOut, aLeft, aRight, aBroadcastHint, aBlockSizes, aMaxBlockSize, aBroadcastSizes, aMaxBroadcastSizes ) );
    }

    template <typename _Ty>
    void BitwiseAnd_Tensor_Tensor_Impl( multi_tensor_t &aOut, multi_tensor_t &aIn, multi_tensor_t &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::BitwiseAnd<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant );
    }

    void BitwiseAndOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight )
    {
        DISPATCH_BY_INTEGRAL_TYPE( aTensorElementType, BitwiseAnd_Tensor_Tensor_Impl, ( aOut, aLeft, aRight ) );
    }

    template <typename _Ty>
    void BitwiseAnd_Tensor_Vector_Impl( multi_tensor_t &aOut, multi_tensor_t &aIn, memory_buffer_t &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::BitwiseAnd<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant );
    }

    void BitwiseAndOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight )
    {
        DISPATCH_BY_INTEGRAL_TYPE( aTensorElementType, BitwiseAnd_Tensor_Vector_Impl, ( aOut, aLeft, aRight ) );
    }

    void BitwiseAndOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight )
    {
        BitwiseAndOp( aTensorElementType, aOut, aRight, aLeft );
    }

    template <typename _Ty>
    void BitwiseOr_Tensor_Scalar_Impl( multi_tensor_t &aOut, multi_tensor_t &aIn, scalar_value_t &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::BitwiseOr<<<lGridDim, lBlockDim>>>( aOut, aIn, std::get<_Ty>( aConstant ) );
    }

    void BitwiseOrOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, scalar_value_t &aRight )
    {
        DISPATCH_BY_INTEGRAL_TYPE( aTensorElementType, BitwiseOr_Tensor_Scalar_Impl, ( aOut, aLeft, aRight ) );
    }

    void BitwiseOrOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aLeft, multi_tensor_t &aRight )
    {
        BitwiseOrOp( aTensorElementType, aOut, aRight, aLeft );
    }

    template <typename _Ty>
    void BitwiseOr_Tensor_Tensor_Impl( multi_tensor_t &aOut, multi_tensor_t &aIn, multi_tensor_t &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::BitwiseOr<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant );
    }

    void BitwiseOrOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight )
    {
        DISPATCH_BY_INTEGRAL_TYPE( aTensorElementType, BitwiseOr_Tensor_Tensor_Impl, ( aOut, aLeft, aRight ) );
    }

    template <typename _Ty>
    static void BitwiseOr_Tensor_Tensor_Impl( multi_tensor_t &aOut, multi_tensor_t &aIn, multi_tensor_t &aConstant,
                                              broadcast_hint_t aBroadcastHint, memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize,
                                              memory_buffer_t &aBroadcastSizes, uint32_t aMaxBroadcastSizes )
    {
        int lBlockCount = ( aMaxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), aMaxBlockSize, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::BitwiseOr<_Ty>
            <<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant, aBroadcastHint, aBlockSizes, aBroadcastSizes );
    }

    void BitwiseOrOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                      broadcast_hint_t aBroadcastHint, memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize, memory_buffer_t &aBroadcastSizes,
                      uint32_t aMaxBroadcastSizes )
    {
        DISPATCH_BY_INTEGRAL_TYPE(
            aTensorElementType, BitwiseOr_Tensor_Tensor_Impl,
            ( aOut, aLeft, aRight, aBroadcastHint, aBlockSizes, aMaxBlockSize, aBroadcastSizes, aMaxBroadcastSizes ) );
    }

    template <typename _Ty>
    void BitwiseOrTensorVectorImpl( multi_tensor_t &aOut, multi_tensor_t &aIn, memory_buffer_t &aConstant )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::BitwiseOr<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant );
    }

    void BitwiseOrOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight )
    {
        DISPATCH_BY_INTEGRAL_TYPE( aTensorElementType, BitwiseOrTensorVectorImpl, ( aOut, aLeft, aRight ) );
    }

    void BitwiseOrOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight )
    {
        BitwiseOrOp( aTensorElementType, aOut, aRight, aLeft );
    }

    template <typename _Ty>
    void BitwiseNotTensorImpl( multi_tensor_t &aOut, multi_tensor_t &aIn )
    {
        int lBlockCount = ( aIn.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Bitwise<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn );
    }

    void BitwiseNotOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aOperand )
    {
        DISPATCH_BY_INTEGRAL_TYPE( aTensorElementType, BitwiseNotTensorImpl, ( aOut, aOperand ) );
    }

    template <typename _Ty>
    static void EqualOpImpl( multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight )
    {
        int lBlockCount = ( aLeft.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aLeft.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::EqualOp<_Ty><<<lGridDim, lBlockDim>>>( aOut, aLeft, aRight );
    }

    void EqualOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, EqualOpImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _Ty>
    static void EqualOpImpl( multi_tensor_t &aOut, multi_tensor_t &aIn, multi_tensor_t &aConstant, broadcast_hint_t aBroadcastHint,
                             memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize, memory_buffer_t &aBroadcastSizes,
                             uint32_t aMaxBroadcastSizes )
    {
        int lBlockCount = ( aMaxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), aMaxBlockSize, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::EqualOp<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant, aBroadcastHint, aBlockSizes, aBroadcastSizes );
    }

    void EqualOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                  broadcast_hint_t aBroadcastHint, memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize, memory_buffer_t &aBroadcastSizes,
                  uint32_t aMaxBroadcastSizes )
    {
        DISPATCH_BY_TYPE( aTensorElementType, EqualOpImpl,
                          ( aOut, aLeft, aRight, aBroadcastHint, aBlockSizes, aMaxBlockSize, aBroadcastSizes, aMaxBroadcastSizes ) );
    }

    template <typename _Ty>
    static void EqualOpImpl( multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight )
    {
        int lBlockCount = ( aLeft.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aLeft.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::EqualOp<_Ty><<<lGridDim, lBlockDim>>>( aOut, aLeft, aRight );
    }

    void EqualOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, EqualOpImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _ScalarType>
    static void EqualOpImpl( multi_tensor_t &aOut, multi_tensor_t &aLeft, scalar_value_t &aRight )
    {
        int lBlockCount = ( aLeft.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aLeft.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::EqualOp<_ScalarType><<<lGridDim, lBlockDim>>>( aOut, aLeft, std::get<_ScalarType>( aRight ) );
    }

    void EqualOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, scalar_value_t &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, EqualOpImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _Ty>
    static void EqualOpImpl( multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight )
    {
        int lBlockCount = ( aRight.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aRight.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::EqualOp<_Ty><<<lGridDim, lBlockDim>>>( aOut, aLeft, aRight );
    }

    void EqualOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, EqualOpImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _Ty>
    static void EqualOpImpl( multi_tensor_t &aOut, scalar_value_t &aLeft, multi_tensor_t &aRight )
    {
        int lBlockCount = ( aRight.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aRight.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::EqualOp<_Ty><<<lGridDim, lBlockDim>>>( aOut, std::get<_Ty>( aLeft ), aRight );
    }

    void EqualOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aLeft, multi_tensor_t &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, EqualOpImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _Ty>
    static void LessThanOpImpl( multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight )
    {
        int lBlockCount = ( aLeft.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aLeft.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOp<_Ty><<<lGridDim, lBlockDim>>>( aOut, aLeft, aRight );
    }

    void LessThanOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, LessThanOpImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _Ty>
    static void LessThanOpImpl( multi_tensor_t &aOut, multi_tensor_t &aIn, multi_tensor_t &aConstant, broadcast_hint_t aBroadcastHint,
                                memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize, memory_buffer_t &aBroadcastSizes,
                                uint32_t aMaxBroadcastSizes )
    {
        int lBlockCount = ( aMaxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), aMaxBlockSize, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOp<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant, aBroadcastHint, aBlockSizes, aBroadcastSizes );
    }

    void LessThanOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                     broadcast_hint_t aBroadcastHint, memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize, memory_buffer_t &aBroadcastSizes,
                     uint32_t aMaxBroadcastSizes )
    {
        DISPATCH_BY_TYPE( aTensorElementType, LessThanOpImpl,
                          ( aOut, aLeft, aRight, aBroadcastHint, aBlockSizes, aMaxBlockSize, aBroadcastSizes, aMaxBroadcastSizes ) );
    }

    template <typename _Ty>
    static void LessThanOpImpl( multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight )
    {
        int lBlockCount = ( aLeft.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aLeft.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOp<_Ty><<<lGridDim, lBlockDim>>>( aOut, aLeft, aRight );
    }

    void LessThanOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, LessThanOpImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _ScalarType>
    static void LessThanOpImpl( multi_tensor_t &aOut, multi_tensor_t &aLeft, scalar_value_t &aRight )
    {
        int lBlockCount = ( aLeft.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aLeft.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOp<_ScalarType><<<lGridDim, lBlockDim>>>( aOut, aLeft, std::get<_ScalarType>( aRight ) );
    }

    void LessThanOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, scalar_value_t &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, LessThanOpImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _Ty>
    static void LessThanOpImpl( multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight )
    {
        int lBlockCount = ( aRight.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aRight.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOp<_Ty><<<lGridDim, lBlockDim>>>( aOut, aLeft, aRight );
    }

    void LessThanOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, LessThanOpImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _Ty>
    static void LessThanOpImpl( multi_tensor_t &aOut, scalar_value_t &aLeft, multi_tensor_t &aRight )
    {
        int lBlockCount = ( aRight.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aRight.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOp<_Ty><<<lGridDim, lBlockDim>>>( aOut, std::get<_Ty>( aLeft ), aRight );
    }

    void LessThanOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aLeft, multi_tensor_t &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, LessThanOpImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _Ty>
    static void LessThanOrEqualOpImpl( multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight )
    {
        int lBlockCount = ( aLeft.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aLeft.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOrEqualOp<_Ty><<<lGridDim, lBlockDim>>>( aOut, aLeft, aRight );
    }

    void LessThanOrEqualOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, LessThanOrEqualOpImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _Ty>
    static void LessThanOrEqualOpImpl( multi_tensor_t &aOut, multi_tensor_t &aIn, multi_tensor_t &aConstant, broadcast_hint_t aBroadcastHint,
                                       memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize, memory_buffer_t &aBroadcastSizes,
                                       uint32_t aMaxBroadcastSizes )
    {
        int lBlockCount = ( aMaxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aIn.Shape().CountLayers(), aMaxBlockSize, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOrEqualOp<_Ty><<<lGridDim, lBlockDim>>>( aOut, aIn, aConstant, aBroadcastHint, aBlockSizes, aBroadcastSizes );
    }

    void LessThanOrEqualOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                            broadcast_hint_t aBroadcastHint, memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize,
                            memory_buffer_t &aBroadcastSizes, uint32_t aMaxBroadcastSizes )
    {
        DISPATCH_BY_TYPE( aTensorElementType, LessThanOrEqualOpImpl,
                          ( aOut, aLeft, aRight, aBroadcastHint, aBlockSizes, aMaxBlockSize, aBroadcastSizes, aMaxBroadcastSizes ) );
    }

    template <typename _Ty>
    static void LessThanOrEqualOpImpl( multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight )
    {
        int lBlockCount = ( aLeft.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aLeft.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOrEqualOp<_Ty><<<lGridDim, lBlockDim>>>( aOut, aLeft, aRight );
    }

    void LessThanOrEqualOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, LessThanOrEqualOpImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _ScalarType>
    static void LessThanOrEqualOpImpl( multi_tensor_t &aOut, multi_tensor_t &aLeft, scalar_value_t &aRight )
    {
        int lBlockCount = ( aLeft.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aLeft.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOrEqualOp<_ScalarType><<<lGridDim, lBlockDim>>>( aOut, aLeft, std::get<_ScalarType>( aRight ) );
    }

    void LessThanOrEqualOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, scalar_value_t &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, LessThanOrEqualOpImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _Ty>
    static void LessThanOrEqualOpImpl( multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight )
    {
        int lBlockCount = ( aRight.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aRight.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOrEqualOp<_Ty><<<lGridDim, lBlockDim>>>( aOut, aLeft, aRight );
    }

    void LessThanOrEqualOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, LessThanOrEqualOpImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _Ty>
    static void LessThanOrEqualOpImpl( multi_tensor_t &aOut, scalar_value_t &aLeft, multi_tensor_t &aRight )
    {
        int lBlockCount = ( aRight.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aRight.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOrEqualOp<_Ty><<<lGridDim, lBlockDim>>>( aOut, std::get<_Ty>( aLeft ), aRight );
    }

    void LessThanOrEqualOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aLeft, multi_tensor_t &aRight )
    {
        DISPATCH_BY_TYPE( aTensorElementType, LessThanOrEqualOpImpl, ( aOut, aLeft, aRight ) );
    }

    template <typename _Ty>
    static void InIntervalTensorTensorImpl( multi_tensor_t &aOut, multi_tensor_t &aX, multi_tensor_t &aLower, multi_tensor_t &aUpper,
                                            bool aStrictLower, bool aStrictUpper )
    {
        int lBlockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::InInterval<_Ty><<<lGridDim, lBlockDim>>>( aOut, aX, aLower, aUpper, aStrictLower, aStrictUpper );
    }

    void InIntervalOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aX, multi_tensor_t &aLower, multi_tensor_t &aUpper,
                       bool aStrictLower, bool aStrictUpper )
    {
        DISPATCH_BY_TYPE( aTensorElementType, InIntervalTensorTensorImpl, ( aOut, aX, aLower, aUpper, aStrictLower, aStrictUpper ) );
    }

    template <typename _Ty>
    static void InIntervalTensorVectorImpl( multi_tensor_t &aOut, multi_tensor_t &aX, multi_tensor_t &aLower, memory_buffer_t &aUpper,
                                            bool aStrictLower, bool aStrictUpper )
    {
        int lBlockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::InInterval<_Ty><<<lGridDim, lBlockDim>>>( aOut, aX, aLower, aUpper, aStrictLower, aStrictUpper );
    }

    void InIntervalOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aX, multi_tensor_t &aLower, memory_buffer_t &aUpper,
                       bool aStrictLower, bool aStrictUpper )
    {
        DISPATCH_BY_TYPE( aTensorElementType, InIntervalTensorVectorImpl, ( aOut, aX, aLower, aUpper, aStrictLower, aStrictUpper ) );
    }

    template <typename _Ty>
    static void InIntervalTensorScalarImpl( multi_tensor_t &aOut, multi_tensor_t &aX, multi_tensor_t &aLower, scalar_value_t &aUpper,
                                            bool aStrictLower, bool aStrictUpper )
    {
        int lBlockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::InInterval<_Ty>
            <<<lGridDim, lBlockDim>>>( aOut, aX, aLower, std::get<_Ty>( aUpper ), aStrictLower, aStrictUpper );
    }

    void InIntervalOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aX, multi_tensor_t &aLower, scalar_value_t &aUpper,
                       bool aStrictLower, bool aStrictUpper )
    {
        DISPATCH_BY_TYPE( aTensorElementType, InIntervalTensorScalarImpl, ( aOut, aX, aLower, aUpper, aStrictLower, aStrictUpper ) );
    }

    template <typename _Ty>
    static void InIntervalVectorTensorImpl( multi_tensor_t &aOut, multi_tensor_t &aX, memory_buffer_t &aLower, multi_tensor_t &aUpper,
                                            bool aStrictLower, bool aStrictUpper )
    {
        int lBlockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::InInterval<_Ty><<<lGridDim, lBlockDim>>>( aOut, aX, aLower, aUpper, aStrictLower, aStrictUpper );
    }

    void InIntervalOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aX, memory_buffer_t &aLower, multi_tensor_t &aUpper,
                       bool aStrictLower, bool aStrictUpper )
    {
        DISPATCH_BY_TYPE( aTensorElementType, InIntervalVectorTensorImpl, ( aOut, aX, aLower, aUpper, aStrictLower, aStrictUpper ) );
    }

    template <typename _Ty>
    static void InIntervalVectorVectorImpl( multi_tensor_t &aOut, multi_tensor_t &aX, memory_buffer_t &aLower, memory_buffer_t &aUpper,
                                            bool aStrictLower, bool aStrictUpper )
    {
        int lBlockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::InInterval<_Ty><<<lGridDim, lBlockDim>>>( aOut, aX, aLower, aUpper, aStrictLower, aStrictUpper );
    }

    void InIntervalOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aX, memory_buffer_t &aLower, memory_buffer_t &aUpper,
                       bool aStrictLower, bool aStrictUpper )
    {
        DISPATCH_BY_TYPE( aTensorElementType, InIntervalVectorVectorImpl, ( aOut, aX, aLower, aUpper, aStrictLower, aStrictUpper ) );
    }

    template <typename _Ty>
    static void InIntervalVectorScalarImpl( multi_tensor_t &aOut, multi_tensor_t &aX, memory_buffer_t &aLower, scalar_value_t &aUpper,
                                            bool aStrictLower, bool aStrictUpper )
    {
        int lBlockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::InInterval<_Ty>
            <<<lGridDim, lBlockDim>>>( aOut, aX, aLower, std::get<_Ty>( aUpper ), aStrictLower, aStrictUpper );
    }

    void InIntervalOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aX, memory_buffer_t &aLower, scalar_value_t &aUpper,
                       bool aStrictLower, bool aStrictUpper )
    {
        DISPATCH_BY_TYPE( aTensorElementType, InIntervalVectorScalarImpl, ( aOut, aX, aLower, aUpper, aStrictLower, aStrictUpper ) );
    }

    template <typename _Ty>
    static void InIntervalScalarTensorImpl( multi_tensor_t &aOut, multi_tensor_t &aX, scalar_value_t &aLower, multi_tensor_t &aUpper,
                                            bool aStrictLower, bool aStrictUpper )
    {
        int lBlockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::InInterval<_Ty>
            <<<lGridDim, lBlockDim>>>( aOut, aX, std::get<_Ty>( aLower ), aUpper, aStrictLower, aStrictUpper );
    }

    void InIntervalOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aX, scalar_value_t &aLower, multi_tensor_t &aUpper,
                       bool aStrictLower, bool aStrictUpper )
    {
        DISPATCH_BY_TYPE( aTensorElementType, InIntervalScalarTensorImpl, ( aOut, aX, aLower, aUpper, aStrictLower, aStrictUpper ) );
    }

    template <typename _Ty>
    static void InIntervalScalarVectorImpl( multi_tensor_t &aOut, multi_tensor_t &aX, scalar_value_t &aLower, memory_buffer_t &aUpper,
                                            bool aStrictLower, bool aStrictUpper )
    {
        int lBlockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::InInterval<_Ty>
            <<<lGridDim, lBlockDim>>>( aOut, aX, std::get<_Ty>( aLower ), aUpper, aStrictLower, aStrictUpper );
    }

    void InIntervalOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aX, scalar_value_t &aLower, memory_buffer_t &aUpper,
                       bool aStrictLower, bool aStrictUpper )
    {
        DISPATCH_BY_TYPE( aTensorElementType, InIntervalScalarVectorImpl, ( aOut, aX, aLower, aUpper, aStrictLower, aStrictUpper ) );
    }

    template <typename _Ty>
    static void InIntervalScalarScalarImpl( multi_tensor_t &aOut, multi_tensor_t &aX, scalar_value_t &aLower, scalar_value_t &aUpper,
                                            bool aStrictLower, bool aStrictUpper )
    {
        int lBlockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::InInterval<_Ty>
            <<<lGridDim, lBlockDim>>>( aOut, aX, std::get<_Ty>( aLower ), std::get<_Ty>( aUpper ), aStrictLower, aStrictUpper );
    }

    void InIntervalOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aX, scalar_value_t &aLower, scalar_value_t &aUpper,
                       bool aStrictLower, bool aStrictUpper )
    {
        DISPATCH_BY_TYPE( aTensorElementType, InIntervalScalarScalarImpl, ( aOut, aX, aLower, aUpper, aStrictLower, aStrictUpper ) );
    }

    template <typename _Ty>
    static void WhereOpTensorTensorImpl( multi_tensor_t &aOut, multi_tensor_t &aCondition, multi_tensor_t &aValueIfTrue,
                                         multi_tensor_t &aValueIfFalse )
    {
        int lBlockCount = ( aCondition.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aCondition.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::WhereTensorTensor<_Ty><<<lGridDim, lBlockDim>>>( aOut, aCondition, aValueIfTrue, aValueIfFalse );
    }

    void WhereOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aCondition, multi_tensor_t &aValueIfTrue,
                  multi_tensor_t &aValueIfFalse )
    {
        DISPATCH_BY_TYPE( aTensorElementType, WhereOpTensorTensorImpl, ( aOut, aCondition, aValueIfTrue, aValueIfFalse ) );
    }

    template <typename _Ty>
    static void WhereTensorVectorImpl( multi_tensor_t &aOut, multi_tensor_t &aCondition, multi_tensor_t &aValueIfTrue,
                                       memory_buffer_t &aValueIfFalse )
    {
        int lBlockCount = ( aCondition.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aCondition.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::WhereTensorVector<_Ty><<<lGridDim, lBlockDim>>>( aOut, aCondition, aValueIfTrue, aValueIfFalse );
    }

    void WhereOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aCondition, multi_tensor_t &aValueIfTrue,
                  memory_buffer_t &aValueIfFalse )
    {
        DISPATCH_BY_TYPE( aTensorElementType, WhereTensorVectorImpl, ( aOut, aCondition, aValueIfTrue, aValueIfFalse ) );
    }

    template <typename _Ty>
    static void WhereTensorScalarImpl( multi_tensor_t &aOut, multi_tensor_t &aCondition, multi_tensor_t &aValueIfTrue,
                                       scalar_value_t &aValueIfFalse )
    {
        int lBlockCount = ( aCondition.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aCondition.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::WhereTensorScalar<_Ty><<<lGridDim, lBlockDim>>>( aOut, aCondition, aValueIfTrue, std::get<_Ty>( aValueIfFalse ) );
    }

    void WhereOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aCondition, multi_tensor_t &aValueIfTrue,
                  scalar_value_t &aValueIfFalse )
    {
        DISPATCH_BY_TYPE( aTensorElementType, WhereTensorScalarImpl, ( aOut, aCondition, aValueIfTrue, aValueIfFalse ) );
    }

    template <typename _Ty>
    static void WhereVectorTensorImpl( multi_tensor_t &aOut, multi_tensor_t &aCondition, memory_buffer_t &aValueIfTrue,
                                       multi_tensor_t &aValueIfFalse )
    {
        int lBlockCount = ( aCondition.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aCondition.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::WhereVectorTensor<_Ty><<<lGridDim, lBlockDim>>>( aOut, aCondition, aValueIfTrue, aValueIfFalse );
    }
    void WhereOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aCondition, memory_buffer_t &aValueIfTrue,
                  multi_tensor_t &aValueIfFalse )
    {
        DISPATCH_BY_TYPE( aTensorElementType, WhereVectorTensorImpl, ( aOut, aCondition, aValueIfTrue, aValueIfFalse ) );
    }

    template <typename _Ty>
    static void WhereVectorVectorImpl( multi_tensor_t &aOut, multi_tensor_t &aCondition, memory_buffer_t &aValueIfTrue,
                                       memory_buffer_t &aValueIfFalse )
    {
        int lBlockCount = ( aCondition.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aCondition.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::WhereVectorVector<_Ty><<<lGridDim, lBlockDim>>>( aOut, aCondition, aValueIfTrue, aValueIfFalse );
    }

    void WhereOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aCondition, memory_buffer_t &aValueIfTrue,
                  memory_buffer_t &aValueIfFalse )
    {
        DISPATCH_BY_TYPE( aTensorElementType, WhereVectorVectorImpl, ( aOut, aCondition, aValueIfTrue, aValueIfFalse ) );
    }

    template <typename _Ty>
    static void WhereVectorScalarImpl( multi_tensor_t &aOut, multi_tensor_t &aCondition, memory_buffer_t &aValueIfTrue,
                                       scalar_value_t &aValueIfFalse )
    {
        int lBlockCount = ( aCondition.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aCondition.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::WhereVectorScalar<_Ty><<<lGridDim, lBlockDim>>>( aOut, aCondition, aValueIfTrue, std::get<_Ty>( aValueIfFalse ) );
    }

    void WhereOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aCondition, memory_buffer_t &aValueIfTrue,
                  scalar_value_t &aValueIfFalse )
    {
        DISPATCH_BY_TYPE( aTensorElementType, WhereVectorScalarImpl, ( aOut, aCondition, aValueIfTrue, aValueIfFalse ) );
    }

    template <typename _Ty>
    static void WhereScalarTensorImpl( multi_tensor_t &aOut, multi_tensor_t &aCondition, scalar_value_t &aValueIfTrue,
                                       multi_tensor_t &aValueIfFalse )
    {
        int lBlockCount = ( aCondition.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aCondition.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::WhereScalarTensor<_Ty><<<lGridDim, lBlockDim>>>( aOut, aCondition, std::get<_Ty>( aValueIfTrue ), aValueIfFalse );
    }

    void WhereOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aCondition, scalar_value_t &aValueIfTrue,
                  multi_tensor_t &aValueIfFalse )
    {
        DISPATCH_BY_TYPE( aTensorElementType, WhereScalarTensorImpl, ( aOut, aCondition, aValueIfTrue, aValueIfFalse ) );
    }

    template <typename _Ty>
    static void WhereScalarVectorImpl( multi_tensor_t &aOut, multi_tensor_t &aCondition, scalar_value_t &aValueIfTrue,
                                       memory_buffer_t &aValueIfFalse )
    {
        int lBlockCount = ( aCondition.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aCondition.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::WhereScalarVector<_Ty><<<lGridDim, lBlockDim>>>( aOut, aCondition, std::get<_Ty>( aValueIfTrue ), aValueIfFalse );
    }

    void WhereOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aCondition, scalar_value_t &aValueIfTrue,
                  memory_buffer_t &aValueIfFalse )
    {
        DISPATCH_BY_TYPE( aTensorElementType, WhereScalarVectorImpl, ( aOut, aCondition, aValueIfTrue, aValueIfFalse ) );
    }

    template <typename _Ty>
    static void WhereScalarScalarImpl( multi_tensor_t &aOut, multi_tensor_t &aCondition, scalar_value_t &aValueIfTrue,
                                       scalar_value_t &aValueIfFalse )
    {
        int lBlockCount = ( aCondition.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aCondition.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::WhereScalarScalar<_Ty>
            <<<lGridDim, lBlockDim>>>( aOut, aCondition, std::get<_Ty>( aValueIfTrue ), std::get<_Ty>( aValueIfFalse ) );
    }

    void WhereOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aCondition, scalar_value_t &aValueIfTrue,
                  scalar_value_t &aValueIfFalse )
    {
        DISPATCH_BY_TYPE( aTensorElementType, WhereScalarScalarImpl, ( aOut, aCondition, aValueIfTrue, aValueIfFalse ) );
    }

    template <typename _Ty>
    static void RepeatOpImpl( multi_tensor_t &aOut, multi_tensor_t &aArray, memory_buffer_t &aRepetitions, uint32_t lMaxRepetitions )
    {
        int lBlockCount = ( lMaxRepetitions / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aArray.Shape().CountLayers(), aArray.Shape().mMaxBufferSize, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Repeat<_Ty><<<lGridDim, lBlockDim>>>( aOut, aArray, aRepetitions );
    }

    void RepeatOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aArray, memory_buffer_t &aRepetitions,
                   uint32_t lMaxRepetitions )
    {
        DISPATCH_BY_TYPE( aTensorElementType, RepeatOpImpl, ( aOut, aArray, aRepetitions, lMaxRepetitions ) );
    }

    template <typename _Ty>
    static void TileOpImpl( multi_tensor_t &aOut, multi_tensor_t &aArray, memory_buffer_t &aRepetitions, uint32_t lMaxRepetitions )
    {
        int lBlockCount = ( aArray.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aArray.Shape().CountLayers(), lMaxRepetitions, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Tile<_Ty><<<lGridDim, lBlockDim>>>( aOut, aArray, aRepetitions );
    }

    void TileOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aArray, memory_buffer_t &aRepetitions,
                 uint32_t lMaxRepetitions )
    {
        DISPATCH_BY_TYPE( aTensorElementType, TileOpImpl, ( aOut, aArray, aRepetitions, lMaxRepetitions ) );
    }

    template <typename _Ty>
    static void LinearSpaceOpImpl( multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight, memory_buffer_t &aSubdivisions,
                                   uint32_t aMaxSubdivisions )
    {
        int lBlockCount = ( aMaxSubdivisions / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aLeft.Shape().CountLayers(), aLeft.Shape().mMaxBufferSize, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::LinearSpace<_Ty><<<lGridDim, lBlockDim>>>( aOut, aLeft, aRight, aSubdivisions );
    }

    void LinearSpaceOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                        memory_buffer_t &aSubdivisions, uint32_t aMaxSubdivisions )
    {
        switch( aTensorElementType )
        {
        case scalar_type_t::FLOAT32:
        {
            LinearSpaceOpImpl<float>( aOut, aLeft, aRight, aSubdivisions, aMaxSubdivisions );
            break;
        }
        case scalar_type_t::FLOAT64:
        {
            LinearSpaceOpImpl<double>( aOut, aLeft, aRight, aSubdivisions, aMaxSubdivisions );
            break;
        }
        default:
            throw std::runtime_error( "Linear space only supports float and double values" );
        }
    }

    template <typename _Ty>
    static void MixImpl( multi_tensor_t &aOut, multi_tensor_t &A, multi_tensor_t &B, multi_tensor_t &t )
    {
        int lBlockCount = ( A.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( A.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Mix<_Ty><<<lGridDim, lBlockDim>>>( aOut, A, B, t );
    }

    void MixOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &A, multi_tensor_t &B, multi_tensor_t &t )
    {
        DISPATCH_BY_TYPE( aTensorElementType, MixImpl, ( aOut, A, B, t ) );
    }

    void Sample2DOp( multi_tensor_t &aOut, multi_tensor_t &X, multi_tensor_t &Y, memory_buffer_t &aTextures )
    {
        int lBlockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( X.Shape().CountLayers(), lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Sample2D<<<lGridDim, lBlockDim>>>( aOut, X, Y, aTextures );
    }

    void Sample2DOp( multi_tensor_t &aOut, multi_tensor_t &X, memory_buffer_t &Y, memory_buffer_t &aTextures )
    {
        int lBlockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( X.Shape().CountLayers(), lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Sample2D<<<lGridDim, lBlockDim>>>( aOut, X, Y, aTextures );
    }

    void Sample2DOp( multi_tensor_t &aOut, multi_tensor_t &X, scalar_value_t &Y, memory_buffer_t &aTextures )
    {
        int lBlockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( X.Shape().CountLayers(), lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Sample2D<<<lGridDim, lBlockDim>>>( aOut, X, std::get<float>( Y ), aTextures );
    }

    void Sample2DOp( multi_tensor_t &aOut, memory_buffer_t &X, multi_tensor_t &Y, memory_buffer_t &aTextures )
    {
        int lBlockCount = ( Y.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( Y.Shape().CountLayers(), lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Sample2D<<<lGridDim, lBlockDim>>>( aOut, X, Y, aTextures );
    }

    void Sample2DOp( multi_tensor_t &aOut, scalar_value_t &X, multi_tensor_t &Y, memory_buffer_t &aTextures )
    {
        int lBlockCount = ( Y.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( Y.Shape().CountLayers(), lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Sample2D<<<lGridDim, lBlockDim>>>( aOut, std::get<float>( X ), Y, aTextures );
    }

    template <typename _Ty>
    static void ToFixedPointOpImpl( multi_tensor_t &aOut, scalar_type_t aOutputElementType, multi_tensor_t &aArray, _Ty aScaling )
    {
        int lBlockCount = ( aArray.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aArray.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        switch( aOutputElementType )
        {
        case scalar_type_t::UINT8:
        {
            Kernels::ToFixedPoint<_Ty, uint8_t><<<lGridDim, lBlockDim>>>( aOut, aArray, aScaling );
            break;
        }
        case scalar_type_t::UINT16:
        {
            Kernels::ToFixedPoint<_Ty, uint16_t><<<lGridDim, lBlockDim>>>( aOut, aArray, aScaling );
            break;
        }
        case scalar_type_t::UINT32:
        {
            Kernels::ToFixedPoint<_Ty, uint32_t><<<lGridDim, lBlockDim>>>( aOut, aArray, aScaling );
            break;
        }
        case scalar_type_t::UINT64:
        {
            Kernels::ToFixedPoint<_Ty, uint64_t><<<lGridDim, lBlockDim>>>( aOut, aArray, aScaling );
            break;
        }
        case scalar_type_t::INT8:
        {
            Kernels::ToFixedPoint<_Ty, int8_t><<<lGridDim, lBlockDim>>>( aOut, aArray, aScaling );
            break;
        }
        case scalar_type_t::INT16:
        {
            Kernels::ToFixedPoint<_Ty, int16_t><<<lGridDim, lBlockDim>>>( aOut, aArray, aScaling );
            break;
        }
        case scalar_type_t::INT32:
        {
            Kernels::ToFixedPoint<_Ty, int32_t><<<lGridDim, lBlockDim>>>( aOut, aArray, aScaling );
            break;
        }
        case scalar_type_t::INT64:
        {
            Kernels::ToFixedPoint<_Ty, int64_t><<<lGridDim, lBlockDim>>>( aOut, aArray, aScaling );
            break;
        }
        default:
            throw std::runtime_error( "Linear space only supports float and double values" );
        }
    }

    void ToFixedPointOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_type_t aOutputElementType, multi_tensor_t &aArray,
                         scalar_value_t &aScaling )
    {
        switch( aTensorElementType )
        {
        case scalar_type_t::FLOAT32:
        {
            ToFixedPointOpImpl<float>( aOut, aOutputElementType, aArray, std::get<float>( aScaling ) );
            break;
        }
        case scalar_type_t::FLOAT64:
        {
            ToFixedPointOpImpl<double>( aOut, aOutputElementType, aArray, std::get<double>( aScaling ) );
            break;
        }
        default:
            throw std::runtime_error( "Linear space only supports float and double values" );
        }
    }

    template <typename _Ty>
    static void AffineTransformImpl( multi_tensor_t &aOut, multi_tensor_t &A, multi_tensor_t &X, multi_tensor_t &B )
    {
        int lBlockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( X.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::AffineTransform<_Ty><<<lGridDim, lBlockDim>>>( aOut, A, X, B );
    }

    template <typename _Ty>
    static void AffineTransformImpl( multi_tensor_t &aOut, multi_tensor_t &A, multi_tensor_t &X, memory_buffer_t &B )
    {
        int lBlockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( X.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::AffineTransform<_Ty><<<lGridDim, lBlockDim>>>( aOut, A, X, B );
    }

    template <typename _Ty>
    static void AffineTransformImpl( multi_tensor_t &aOut, multi_tensor_t &A, multi_tensor_t &X, scalar_value_t &B )
    {
        int lBlockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( X.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::AffineTransform<_Ty><<<lGridDim, lBlockDim>>>( aOut, A, X, std::get<_Ty>( B ) );
    }

    template <typename _Ty>
    static void AffineTransformImpl( multi_tensor_t &aOut, memory_buffer_t &A, multi_tensor_t &X, multi_tensor_t &B )
    {
        int lBlockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( X.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::AffineTransform<_Ty><<<lGridDim, lBlockDim>>>( aOut, A, X, B );
    }

    template <typename _Ty>
    static void AffineTransformImpl( multi_tensor_t &aOut, memory_buffer_t &A, multi_tensor_t &X, memory_buffer_t &B )
    {
        int lBlockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( X.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::AffineTransform<_Ty><<<lGridDim, lBlockDim>>>( aOut, A, X, B );
    }

    template <typename _Ty>
    static void AffineTransformImpl( multi_tensor_t &aOut, memory_buffer_t &A, multi_tensor_t &X, scalar_value_t &B )
    {
        int lBlockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( X.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::AffineTransform<_Ty><<<lGridDim, lBlockDim>>>( aOut, A, X, std::get<_Ty>( B ) );
    }

    template <typename _Ty>
    static void AffineTransformImpl( multi_tensor_t &aOut, scalar_value_t &A, multi_tensor_t &X, multi_tensor_t &B )
    {
        int lBlockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( X.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::AffineTransform<_Ty><<<lGridDim, lBlockDim>>>( aOut, std::get<_Ty>( A ), X, B );
    }

    template <typename _Ty>
    static void AffineTransformImpl( multi_tensor_t &aOut, scalar_value_t &A, multi_tensor_t &X, memory_buffer_t &B )
    {
        int lBlockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( X.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::AffineTransform<_Ty><<<lGridDim, lBlockDim>>>( aOut, std::get<_Ty>( A ), X, B );
    }

    template <typename _Ty>
    static void AffineTransformImpl( multi_tensor_t &aOut, scalar_value_t &A, multi_tensor_t &X, scalar_value_t &B )
    {
        int lBlockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( X.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::AffineTransform<_Ty><<<lGridDim, lBlockDim>>>( aOut, std::get<_Ty>( A ), X, std::get<_Ty>( B ) );
    }

    void AffineTransformOp( scalar_type_t aOutputElementType, multi_tensor_t &aOut, multi_tensor_t &A, multi_tensor_t &X, multi_tensor_t &B )
    {
        DISPATCH_BY_TYPE( aOutputElementType, AffineTransformImpl, ( aOut, A, X, B ) );
    }

    void AffineTransformOp( scalar_type_t aOutputElementType, multi_tensor_t &aOut, multi_tensor_t &A, multi_tensor_t &X, memory_buffer_t &B )
    {
        DISPATCH_BY_TYPE( aOutputElementType, AffineTransformImpl, ( aOut, A, X, B ) );
    }

    void AffineTransformOp( scalar_type_t aOutputElementType, multi_tensor_t &aOut, multi_tensor_t &A, multi_tensor_t &X, scalar_value_t &B )
    {
        DISPATCH_BY_TYPE( aOutputElementType, AffineTransformImpl, ( aOut, A, X, B ) );
    }

    void AffineTransformOp( scalar_type_t aOutputElementType, multi_tensor_t &aOut, memory_buffer_t &A, multi_tensor_t &X, multi_tensor_t &B )
    {
        DISPATCH_BY_TYPE( aOutputElementType, AffineTransformImpl, ( aOut, A, X, B ) );
    }

    void AffineTransformOp( scalar_type_t aOutputElementType, multi_tensor_t &aOut, memory_buffer_t &A, multi_tensor_t &X, memory_buffer_t &B )
    {
        DISPATCH_BY_TYPE( aOutputElementType, AffineTransformImpl, ( aOut, A, X, B ) );
    }

    void AffineTransformOp( scalar_type_t aOutputElementType, multi_tensor_t &aOut, memory_buffer_t &A, multi_tensor_t &X, scalar_value_t &B )
    {
        DISPATCH_BY_TYPE( aOutputElementType, AffineTransformImpl, ( aOut, A, X, B ) );
    }

    void AffineTransformOp( scalar_type_t aOutputElementType, multi_tensor_t &aOut, scalar_value_t &A, multi_tensor_t &X, multi_tensor_t &B )
    {
        DISPATCH_BY_TYPE( aOutputElementType, AffineTransformImpl, ( aOut, A, X, B ) );
    }

    void AffineTransformOp( scalar_type_t aOutputElementType, multi_tensor_t &aOut, scalar_value_t &A, multi_tensor_t &X, memory_buffer_t &B )
    {
        DISPATCH_BY_TYPE( aOutputElementType, AffineTransformImpl, ( aOut, A, X, B ) );
    }

    void AffineTransformOp( scalar_type_t aOutputElementType, multi_tensor_t &aOut, scalar_value_t &A, multi_tensor_t &X, scalar_value_t &B )
    {
        DISPATCH_BY_TYPE( aOutputElementType, AffineTransformImpl, ( aOut, A, X, B ) );
    }

    void FloorOp( multi_tensor_t &aOut, multi_tensor_t &aX )
    {
        int lBlockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Floor<<<lGridDim, lBlockDim>>>( aOut, aX );
    }

    void CeilOp( multi_tensor_t &aOut, multi_tensor_t &aX )
    {
        int lBlockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Ceil<<<lGridDim, lBlockDim>>>( aOut, aX );
    }

    template <typename _Ty>
    void AbsImpl( multi_tensor_t &aOut, multi_tensor_t &aX )
    {
        int lBlockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Abs<_Ty><<<lGridDim, lBlockDim>>>( aOut, aX );
    }

    void AbsOp( scalar_type_t aOutputElementType, multi_tensor_t &aOut, multi_tensor_t &aX )
    {
        DISPATCH_BY_SIGNED_TYPE( aOutputElementType, AbsImpl, ( aOut, aX ) );
    }

    template <typename _Ty>
    void SqrtImpl( multi_tensor_t &aOut, multi_tensor_t &aX )
    {
        int lBlockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Sqrt<_Ty><<<lGridDim, lBlockDim>>>( aOut, aX );
    }

    void SqrtOp( scalar_type_t aOutputElementType, multi_tensor_t &aOut, multi_tensor_t &aX )
    {
        DISPATCH_BY_TYPE( aOutputElementType, SqrtImpl, ( aOut, aX ) );
    }

    template <typename _Ty>
    void RoundImpl( multi_tensor_t &aOut, multi_tensor_t &aX )
    {
        int lBlockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Round<_Ty><<<lGridDim, lBlockDim>>>( aOut, aX );
    }

    void RoundOp( scalar_type_t aOutputElementType, multi_tensor_t &aOut, multi_tensor_t &aX )
    {
        DISPATCH_BY_TYPE( aOutputElementType, RoundImpl, ( aOut, aX ) );
    }

    void CountTrueOp( multi_tensor_t &aOut, multi_tensor_t &aX, memory_buffer_t &aBlockSizes, memory_buffer_t &aElementCount,
                      uint32_t aMaxBlockSize )
    {
        int lBlockCount = ( aMaxBlockSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::CountNonZero<uint8_t><<<lGridDim, lBlockDim>>>( aOut, aX, aBlockSizes, aElementCount );
    }

    template <typename _Ty>
    void CountNonZeroImpl( multi_tensor_t &aOut, multi_tensor_t &aX, memory_buffer_t &aBlockSizes, memory_buffer_t &aElementCount,
                           uint32_t aMaxBlockSize )
    {
        int lBlockCount = ( aMaxBlockSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::CountNonZero<_Ty><<<lGridDim, lBlockDim>>>( aOut, aX, aBlockSizes, aElementCount );
    }

    void CountNonZeroOp( scalar_type_t aOutputElementType, multi_tensor_t &aOut, multi_tensor_t &aX, memory_buffer_t &aBlockSizes,
                         memory_buffer_t &aElementCount, uint32_t aMaxBlockSize )
    {
        DISPATCH_BY_TYPE( aOutputElementType, CountNonZeroImpl, ( aOut, aX, aBlockSizes, aElementCount, aMaxBlockSize ) );
    }

    template <typename _Ty>
    void CountZeroImpl( multi_tensor_t &aOut, multi_tensor_t &aX, memory_buffer_t &aBlockSizes, memory_buffer_t &aElementCount,
                        uint32_t aMaxBlockSize )
    {
        int lBlockCount = ( aMaxBlockSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::CountZero<_Ty><<<lGridDim, lBlockDim>>>( aOut, aX, aBlockSizes, aElementCount );
    }

    void CountZeroOp( scalar_type_t aOutputElementType, multi_tensor_t &aOut, multi_tensor_t &aX, memory_buffer_t &aBlockSizes,
                      memory_buffer_t &aElementCount, uint32_t aMaxBlockSize )
    {
        DISPATCH_BY_TYPE( aOutputElementType, CountZeroImpl, ( aOut, aX, aBlockSizes, aElementCount, aMaxBlockSize ) );
    }

    template <typename _Ty>
    void ArraySummationImpl( multi_tensor_t &aOut, multi_tensor_t &aX, memory_buffer_t &aBegin, memory_buffer_t &aEnd, memory_buffer_t &aElementCount,
                             memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize )
    {
        int lBlockCount = ( aMaxBlockSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::ArraySummation<_Ty><<<lGridDim, lBlockDim>>>( aOut, aX, aBegin, aEnd, aElementCount, aBlockSizes );
    }

    void ArraySummationOp( scalar_type_t aOutputElementType, multi_tensor_t &aOut, multi_tensor_t &aX, memory_buffer_t &aBegin,
                           memory_buffer_t &aEnd, memory_buffer_t &aElementCount, memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize )
    {
        DISPATCH_BY_TYPE( aOutputElementType, ArraySummationImpl,
                          ( aOut, aX, aBegin, aEnd, aElementCount, aBlockSizes, aMaxBlockSize ) );
    }

    template <typename _Ty>
    void ArraySliceImpl( multi_tensor_t &aOut, multi_tensor_t &aX, memory_buffer_t &aBegin, memory_buffer_t &aEnd, memory_buffer_t &aElementCount,
                         memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize )
    {
        int lBlockCount = ( aMaxBlockSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::ArraySlice<_Ty><<<lGridDim, lBlockDim>>>( aOut, aX, aBegin, aEnd, aElementCount, aBlockSizes );
    }

    void ArraySliceOp( scalar_type_t aOutputElementType, multi_tensor_t &aOut, multi_tensor_t &aX, memory_buffer_t &aBegin, memory_buffer_t &aEnd,
                       memory_buffer_t &aElementCount, memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize )
    {
        DISPATCH_BY_TYPE( aOutputElementType, ArraySliceImpl, ( aOut, aX, aBegin, aEnd, aElementCount, aBlockSizes, aMaxBlockSize ) );
    }

    template <typename _Ty>
    void DiffImpl( multi_tensor_t &aOut, multi_tensor_t &aX, uint32_t aCount, memory_buffer_t &aElementCount, memory_buffer_t &aBlockSizes,
                   uint32_t aMaxBlockSize )
    {
        int lBlockCount = ( aMaxBlockSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Diff<_Ty><<<lGridDim, lBlockDim>>>( aOut, aX, aCount, aElementCount, aBlockSizes );
    }

    void DiffOp( scalar_type_t aOutputElementType, multi_tensor_t &aOut, multi_tensor_t &aX, uint32_t aCount, memory_buffer_t &aElementCount,
                 memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize )
    {
        DISPATCH_BY_TYPE( aOutputElementType, DiffImpl, ( aOut, aX, aCount, aElementCount, aBlockSizes, aMaxBlockSize ) );
    }

    template <typename _Ty>
    void ShiftImpl( multi_tensor_t &aOut, multi_tensor_t &aX, int32_t aCount, scalar_value_t &aFillValue, memory_buffer_t &aElementCount,
                    memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize )
    {
        int lBlockCount = ( aMaxBlockSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aX.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        if( aCount < 0 )
            Kernels::ShiftLeft<_Ty>
                <<<lGridDim, lBlockDim>>>( aOut, aX, -aCount, std::get<_Ty>( aFillValue ), aElementCount, aBlockSizes );
        else
            Kernels::ShiftRight<_Ty>
                <<<lGridDim, lBlockDim>>>( aOut, aX, aCount, std::get<_Ty>( aFillValue ), aElementCount, aBlockSizes );
    }

    void ShiftOp( scalar_type_t aOutputElementType, multi_tensor_t &aOut, multi_tensor_t &aX, int32_t aCount, scalar_value_t &aFillValue,
                  memory_buffer_t &aElementCount, memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize )
    {
        DISPATCH_BY_TYPE( aOutputElementType, ShiftImpl, ( aOut, aX, aCount, aFillValue, aElementCount, aBlockSizes, aMaxBlockSize ) );
    }

    template <typename _Ty>
    void Conv1DImpl( multi_tensor_t &aOut, multi_tensor_t &aArray0, memory_buffer_t &aElementCount0, memory_buffer_t &aBlockSizes0,
                     uint32_t aMaxElementCount0, uint32_t aMaxBlockSize0, multi_tensor_t &aArray1, memory_buffer_t &aElementCount1,
                     memory_buffer_t aBlockSizes1, uint32_t aMaxBlockSize1 )
    {
        int lBlockCount = ( aMaxElementCount0 / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aArray0.Shape().CountLayers(), aMaxBlockSize0, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Conv1D<_Ty>
            <<<lGridDim, lBlockDim>>>( aOut, aArray0, aElementCount0, aBlockSizes0, aArray1, aElementCount1, aBlockSizes1 );
    }

    void Conv1DOp( scalar_type_t aOutputElementType, multi_tensor_t &aOut, multi_tensor_t &aArray0, memory_buffer_t &aElementCount0,
                   memory_buffer_t &aBlockSizes0, uint32_t aMaxElementCount0, uint32_t aMaxBlockSize0, multi_tensor_t &aArray1,
                   memory_buffer_t &aElementCount1, memory_buffer_t &aBlockSizes1, uint32_t aMaxBlockSize1 )
    {
        DISPATCH_BY_TYPE( aOutputElementType, Conv1DImpl,
                          ( aOut, aArray0, aElementCount0, aBlockSizes0, aMaxElementCount0, aMaxBlockSize0, aArray1, aElementCount1,
                            aBlockSizes1, aMaxBlockSize1 ) );
    }

    template <typename _Ty>
    void HCatImpl( multi_tensor_t &aOut, multi_tensor_t &aArray0, memory_buffer_t &aElementCount0, multi_tensor_t &aArray1,
                   memory_buffer_t &aElementCount1, memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize )
    {
        int lBlockCount = ( aMaxBlockSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aArray0.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::HCat<_Ty><<<lGridDim, lBlockDim>>>( aOut, aArray0, aElementCount0, aArray1, aElementCount1, aBlockSizes );
    }

    void HCatOp( scalar_type_t aOutputElementType, multi_tensor_t &aOut, multi_tensor_t &aArray0, memory_buffer_t &aElementCount0,
                 multi_tensor_t &aArray1, memory_buffer_t &aElementCount1, memory_buffer_t &aBlockSizes0, uint32_t aMaxBlockSize0 )
    {
        DISPATCH_BY_TYPE( aOutputElementType, HCatImpl,
                          ( aOut, aArray0, aElementCount0, aArray1, aElementCount1, aBlockSizes0, aMaxBlockSize0 ) );
    }

} // namespace SE::TensorOps