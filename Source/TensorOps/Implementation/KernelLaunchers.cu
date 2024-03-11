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
    static void ConstantFillImpl( multi_tensor_t &aArray, scalar_value_t &constant )
    {
        int blockCount = ( aArray.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aArray.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::ConstantFill<_Ty><<<gridDim, blockDim>>>( aArray, std::get<_Ty>( constant ) );
    }

    template <typename _Ty>
    static void ConstantFillImpl( multi_tensor_t &aArray, memory_buffer_t &initialValues )
    {
        int blockCount = ( aArray.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aArray.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::ConstantFill<_Ty><<<gridDim, blockDim>>>( aArray, initialValues );
    }

    void ConstantFill( scalar_type_t tensorElementType, multi_tensor_t &aArray, memory_buffer_t &initialValues )
    {
        DISPATCH_BY_TYPE( tensorElementType, ConstantFillImpl, ( aArray, initialValues ) );
    }

    void ConstantFill( scalar_type_t tensorElementType, multi_tensor_t &aArray, scalar_value_t &initialValues )
    {
        DISPATCH_BY_TYPE( tensorElementType, ConstantFillImpl, ( aArray, initialValues ) );
    }

    void RandomUniformFill( scalar_type_t tensorElementType, multi_tensor_t &aArray )
    {
        switch( tensorElementType )
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

    void RandomNormalFill( scalar_type_t tensorElementType, multi_tensor_t &aArray, scalar_value_t &aMu, scalar_value_t &aSigma )
    {
        switch( tensorElementType )
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
    static void ARangeOpImpl( multi_tensor_t &out, memory_buffer_t &left, memory_buffer_t &right, memory_buffer_t &aDelta,
                              uint32_t aMaxSubdivisions )
    {
        int blockCount = ( aMaxSubdivisions / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( out.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::ARange<_Ty><<<gridDim, blockDim>>>( out, left, right, aDelta );
    }

    void ARangeOp( scalar_type_t tensorElementType, multi_tensor_t &out, memory_buffer_t &left, memory_buffer_t &right,
                   memory_buffer_t &aDelta, uint32_t aMaxSubdivisions )
    {
        switch( tensorElementType )
        {
        case scalar_type_t::FLOAT32:
        {
            ARangeOpImpl<float>( out, left, right, aDelta, aMaxSubdivisions );
            break;
        }
        case scalar_type_t::FLOAT64:
        {
            ARangeOpImpl<double>( out, left, right, aDelta, aMaxSubdivisions );
            break;
        }
        default:
            throw std::runtime_error( "Linear space only supports float and double values" );
        }
    }

    template <typename _Ty>
    static void AddArrayToArrayImpl( multi_tensor_t &out, multi_tensor_t &in, multi_tensor_t &constant )
    {
        int blockCount = ( in.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Add<_Ty><<<gridDim, blockDim>>>( out, in, constant );
    }

    template <typename _Ty>
    static void AddArrayToArrayImpl( multi_tensor_t &out, multi_tensor_t &in, multi_tensor_t &constant,
                                     broadcast_hint_t aBroadcastHint, memory_buffer_t &blockSizes, uint32_t maxBlockSize,
                                     memory_buffer_t &broadcastSizes, uint32_t maxBroadcastSizes )
    {
        int blockCount = ( maxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), maxBlockSize, blockCount );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Add<_Ty><<<gridDim, blockDim>>>( out, in, constant, aBroadcastHint, blockSizes, broadcastSizes );
    }

    template <typename _ScalarType>
    static void AddScalarToArrayImpl( multi_tensor_t &out, multi_tensor_t &aArray, scalar_value_t &constant )
    {
        int blockCount = ( aArray.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aArray.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Add<_ScalarType><<<gridDim, blockDim>>>( out, aArray, std::get<_ScalarType>( constant ) );
    }

    template <typename _Ty>
    static void AddArrayToVectorImpl( multi_tensor_t &out, multi_tensor_t &in, memory_buffer_t &constant )
    {
        int blockCount = ( in.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Add<_Ty><<<gridDim, blockDim>>>( out, in, constant );
    }

    void AddOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right )
    {
        DISPATCH_BY_TYPE( tensorElementType, AddArrayToArrayImpl, ( out, left, right ) );
    }

    void AddOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right,
                broadcast_hint_t aBroadcastHint, memory_buffer_t &blockSizes, uint32_t maxBlockSize, memory_buffer_t &broadcastSizes,
                uint32_t maxBroadcastSizes )
    {
        DISPATCH_BY_TYPE( tensorElementType, AddArrayToArrayImpl,
                          ( out, left, right, aBroadcastHint, blockSizes, maxBlockSize, broadcastSizes, maxBroadcastSizes ) );
    }

    void AddOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, scalar_value_t &right )
    {
        DISPATCH_BY_TYPE( tensorElementType, AddScalarToArrayImpl, ( out, left, right ) );
    }

    void AddOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, memory_buffer_t &right )
    {
        DISPATCH_BY_TYPE( tensorElementType, AddArrayToVectorImpl, ( out, left, right ) );
    }

    template <typename _ScalarType>
    void MultiplyArrayByScalarImpl( multi_tensor_t &out, multi_tensor_t &in, scalar_value_t &constant )
    {
        int blockCount = ( in.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Multiply<_ScalarType><<<gridDim, blockDim>>>( out, in, std::get<_ScalarType>( constant ) );
    }

    template <typename _Ty>
    static void MultiplyArrayByArrayImpl( multi_tensor_t &out, multi_tensor_t &in, multi_tensor_t &constant )
    {
        int blockCount = ( in.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Multiply<_Ty><<<gridDim, blockDim>>>( out, in, constant );
    }

    template <typename _Ty>
    static void MultiplyArrayByArrayImpl( multi_tensor_t &out, multi_tensor_t &in, multi_tensor_t &constant,
                                          broadcast_hint_t aBroadcastHint, memory_buffer_t &blockSizes, uint32_t maxBlockSize,
                                          memory_buffer_t &broadcastSizes, uint32_t maxBroadcastSizes )
    {
        int blockCount = ( maxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), maxBlockSize, blockCount );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Multiply<_Ty><<<gridDim, blockDim>>>( out, in, constant, aBroadcastHint, blockSizes, broadcastSizes );
    }

    template <typename _Ty>
    static void MultiplyArrayByVectorImpl( multi_tensor_t &out, multi_tensor_t &in, memory_buffer_t &constant )
    {
        int blockCount = ( in.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Multiply<_Ty><<<gridDim, blockDim>>>( out, in, constant );
    }

    void MultiplyOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, scalar_value_t &right )
    {
        DISPATCH_BY_TYPE( tensorElementType, MultiplyArrayByScalarImpl, ( out, left, right ) );
    }

    void MultiplyOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right )
    {
        DISPATCH_BY_TYPE( tensorElementType, MultiplyArrayByArrayImpl, ( out, left, right ) );
    }

    void MultiplyOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right,
                     broadcast_hint_t aBroadcastHint, memory_buffer_t &blockSizes, uint32_t maxBlockSize,
                     memory_buffer_t &broadcastSizes, uint32_t maxBroadcastSizes )
    {
        DISPATCH_BY_TYPE( tensorElementType, MultiplyArrayByArrayImpl,
                          ( out, left, right, aBroadcastHint, blockSizes, maxBlockSize, broadcastSizes, maxBroadcastSizes ) );
    }

    void MultiplyOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, memory_buffer_t &right )
    {
        DISPATCH_BY_TYPE( tensorElementType, MultiplyArrayByVectorImpl, ( out, left, right ) );
    }

    template <typename _Ty>
    void SubtractArrayFromScalarImpl( multi_tensor_t &out, scalar_value_t &constant, multi_tensor_t &in )
    {
        int blockCount = ( in.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Subtract<<<gridDim, blockDim>>>( out, std::get<_Ty>( constant ), in );
    }

    template <typename _Ty>
    void SubtractScalarFromArrayImpl( multi_tensor_t &out, multi_tensor_t &in, scalar_value_t &constant )
    {
        int blockCount = ( in.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Subtract<<<gridDim, blockDim>>>( out, in, std::get<_Ty>( constant ) );
    }

    template <typename _Ty>
    static void SubtractVectorFromArrayImpl( multi_tensor_t &out, multi_tensor_t &in, memory_buffer_t &constant )
    {
        int blockCount = ( in.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Subtract<_Ty><<<gridDim, blockDim>>>( out, in, constant );
    }

    template <typename _Ty>
    static void SubtractArrayfromArrayImpl( multi_tensor_t &out, multi_tensor_t &in, multi_tensor_t &constant )
    {
        int blockCount = ( in.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Subtract<_Ty><<<gridDim, blockDim>>>( out, in, constant );
    }

    template <typename _Ty>
    static void SubtractArrayfromArrayImpl( multi_tensor_t &out, multi_tensor_t &in, multi_tensor_t &constant,
                                            broadcast_hint_t aBroadcastHint, memory_buffer_t &blockSizes, uint32_t maxBlockSize,
                                            memory_buffer_t &broadcastSizes, uint32_t maxBroadcastSizes )
    {
        int blockCount = ( maxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), maxBlockSize, blockCount );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Subtract<_Ty><<<gridDim, blockDim>>>( out, in, constant, aBroadcastHint, blockSizes, broadcastSizes );
    }

    template <typename _Ty>
    static void SubtractArrayFromVectorImpl( multi_tensor_t &out, memory_buffer_t &constant, multi_tensor_t &in )
    {
        int blockCount = ( in.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Subtract<_Ty><<<gridDim, blockDim>>>( out, constant, in );
    }

    void SubtractOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, scalar_value_t &right )
    {
        DISPATCH_BY_TYPE( tensorElementType, SubtractScalarFromArrayImpl, ( out, left, right ) );
    }

    void SubtractOp( scalar_type_t tensorElementType, multi_tensor_t &out, scalar_value_t &left, multi_tensor_t &right )
    {
        DISPATCH_BY_TYPE( tensorElementType, SubtractArrayFromScalarImpl, ( out, left, right ) );
    }

    void SubtractOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right )
    {
        DISPATCH_BY_TYPE( tensorElementType, SubtractArrayfromArrayImpl, ( out, left, right ) );
    }

    void SubtractOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right,
                     broadcast_hint_t aBroadcastHint, memory_buffer_t &blockSizes, uint32_t maxBlockSize,
                     memory_buffer_t &broadcastSizes, uint32_t maxBroadcastSizes )
    {
        DISPATCH_BY_TYPE( tensorElementType, SubtractArrayfromArrayImpl,
                          ( out, left, right, aBroadcastHint, blockSizes, maxBlockSize, broadcastSizes, maxBroadcastSizes ) );
    }

    void SubtractOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, memory_buffer_t &right )
    {
        DISPATCH_BY_TYPE( tensorElementType, SubtractVectorFromArrayImpl, ( out, left, right ) );
    }

    void SubtractOp( scalar_type_t tensorElementType, multi_tensor_t &out, memory_buffer_t &left, multi_tensor_t &right )
    {
        DISPATCH_BY_TYPE( tensorElementType, SubtractArrayFromVectorImpl, ( out, left, right ) );
    }

    template <typename _Ty>
    static void DivideArrayByScalarImpl( multi_tensor_t &out, multi_tensor_t &in, scalar_value_t &constant )
    {
        int blockCount = ( in.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Divide<_Ty><<<gridDim, blockDim>>>( out, in, std::get<_Ty>( constant ) );
    }

    template <typename _Ty>
    static void DivideScalarByArrayImpl( multi_tensor_t &out, scalar_value_t &constant, multi_tensor_t &in )
    {
        int blockCount = ( in.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Divide<_Ty><<<gridDim, blockDim>>>( out, std::get<_Ty>( constant ), in );
    }

    template <typename _Ty>
    static void DivideArrayfromArrayImpl( multi_tensor_t &out, multi_tensor_t &in, multi_tensor_t &constant )
    {
        int blockCount = ( in.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Divide<_Ty><<<gridDim, blockDim>>>( out, in, constant );
    }

    template <typename _Ty>
    static void DivideArrayfromArrayImpl( multi_tensor_t &out, multi_tensor_t &in, multi_tensor_t &constant,
                                          broadcast_hint_t aBroadcastHint, memory_buffer_t &blockSizes, uint32_t maxBlockSize,
                                          memory_buffer_t &broadcastSizes, uint32_t maxBroadcastSizes )
    {
        int blockCount = ( maxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), maxBlockSize, blockCount );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Divide<_Ty><<<gridDim, blockDim>>>( out, in, constant, aBroadcastHint, blockSizes, broadcastSizes );
    }

    template <typename _Ty>
    static void DivideArrayByVectorImpl( multi_tensor_t &out, multi_tensor_t &in, memory_buffer_t &constant )
    {
        int blockCount = ( in.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Divide<_Ty><<<gridDim, blockDim>>>( out, in, constant );
    }

    template <typename _Ty>
    static void DivideVectorByArrayImpl( multi_tensor_t &out, memory_buffer_t &constant, multi_tensor_t &in )
    {
        int blockCount = ( in.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Divide<_Ty><<<gridDim, blockDim>>>( out, constant, in );
    }

    void DivideOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, scalar_value_t &right )
    {
        DISPATCH_BY_TYPE( tensorElementType, DivideArrayByScalarImpl, ( out, left, right ) );
    }

    void DivideOp( scalar_type_t tensorElementType, multi_tensor_t &out, scalar_value_t &left, multi_tensor_t &right )
    {
        DISPATCH_BY_TYPE( tensorElementType, DivideScalarByArrayImpl, ( out, left, right ) );
    }

    void DivideOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right )
    {
        DISPATCH_BY_TYPE( tensorElementType, DivideArrayfromArrayImpl, ( out, left, right ) );
    }

    void DivideOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right,
                   broadcast_hint_t aBroadcastHint, memory_buffer_t &blockSizes, uint32_t maxBlockSize,
                   memory_buffer_t &broadcastSizes, uint32_t maxBroadcastSizes )
    {
        DISPATCH_BY_TYPE( tensorElementType, DivideArrayfromArrayImpl,
                          ( out, left, right, aBroadcastHint, blockSizes, maxBlockSize, broadcastSizes, maxBroadcastSizes ) );
    }

    void DivideOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, memory_buffer_t &right )
    {
        DISPATCH_BY_TYPE( tensorElementType, DivideArrayByVectorImpl, ( out, left, right ) );
    }

    void DivideOp( scalar_type_t tensorElementType, multi_tensor_t &out, memory_buffer_t &left, multi_tensor_t &right )
    {
        DISPATCH_BY_TYPE( tensorElementType, DivideVectorByArrayImpl, ( out, left, right ) );
    }

    void AndOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, scalar_value_t &right )
    {
        int blockCount = ( left.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( left.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::And<<<gridDim, blockDim>>>( out, left, std::get<uint8_t>( right ) );
    }

    void AndOp( scalar_type_t tensorElementType, multi_tensor_t &out, scalar_value_t &left, multi_tensor_t &right )
    {
        AndOp( tensorElementType, out, right, left );
    }

    void AndOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &in, multi_tensor_t &constant,
                broadcast_hint_t aBroadcastHint, memory_buffer_t &blockSizes, uint32_t maxBlockSize, memory_buffer_t &broadcastSizes,
                uint32_t maxBroadcastSizes )
    {
        int blockCount = ( maxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), maxBlockSize, blockCount );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::And<<<gridDim, blockDim>>>( out, in, constant, aBroadcastHint, blockSizes, broadcastSizes );
    }

    void AndOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right )
    {
        int blockCount = ( left.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( left.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::And<<<gridDim, blockDim>>>( out, left, right );
    }

    void AndOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, memory_buffer_t &right )
    {
        int blockCount = ( left.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( left.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::And<<<gridDim, blockDim>>>( out, left, right );
    }

    void AndOp( scalar_type_t tensorElementType, multi_tensor_t &out, memory_buffer_t &left, multi_tensor_t &right )
    {
        AndOp( tensorElementType, out, right, left );
    }

    void OrOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, scalar_value_t &right )
    {
        int blockCount = ( left.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( left.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Or<<<gridDim, blockDim>>>( out, left, std::get<uint8_t>( right ) );
    }

    void OrOp( scalar_type_t tensorElementType, multi_tensor_t &out, scalar_value_t &left, multi_tensor_t &right )
    {
        OrOp( tensorElementType, out, right, left );
    }

    void OrOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right )
    {
        int blockCount = ( left.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( left.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Or<<<gridDim, blockDim>>>( out, left, right );
    }

    void OrOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &in, multi_tensor_t &constant,
               broadcast_hint_t aBroadcastHint, memory_buffer_t &blockSizes, uint32_t maxBlockSize, memory_buffer_t &broadcastSizes,
               uint32_t maxBroadcastSizes )
    {
        int blockCount = ( maxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), maxBlockSize, blockCount );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Or<<<gridDim, blockDim>>>( out, in, constant, aBroadcastHint, blockSizes, broadcastSizes );
    }

    void OrOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, memory_buffer_t &right )
    {
        int blockCount = ( left.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( left.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Or<<<gridDim, blockDim>>>( out, left, right );
    }

    void OrOp( scalar_type_t tensorElementType, multi_tensor_t &out, memory_buffer_t &left, multi_tensor_t &right )
    {
        OrOp( tensorElementType, out, right, left );
    }

    void NotOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &aOperand )
    {
        int blockCount = ( aOperand.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aOperand.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Not<<<gridDim, blockDim>>>( out, aOperand );
    }

    template <typename _Ty>
    void BitwiseAnd_Tensor_Scalar_Impl( multi_tensor_t &out, multi_tensor_t &in, scalar_value_t &constant )
    {
        int blockCount = ( in.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::BitwiseAnd<<<gridDim, blockDim>>>( out, in, std::get<_Ty>( constant ) );
    }

    void BitwiseAndOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, scalar_value_t &right )
    {
        DISPATCH_BY_INTEGRAL_TYPE( tensorElementType, BitwiseAnd_Tensor_Scalar_Impl, ( out, left, right ) );
    }

    void BitwiseAndOp( scalar_type_t tensorElementType, multi_tensor_t &out, scalar_value_t &left, multi_tensor_t &right )
    {
        BitwiseAndOp( tensorElementType, out, right, left );
    }

    template <typename _Ty>
    static void BitwiseAnd_Tensor_Tensor_Impl( multi_tensor_t &out, multi_tensor_t &in, multi_tensor_t &constant,
                                               broadcast_hint_t aBroadcastHint, memory_buffer_t &blockSizes, uint32_t maxBlockSize,
                                               memory_buffer_t &broadcastSizes, uint32_t maxBroadcastSizes )
    {
        int blockCount = ( maxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), maxBlockSize, blockCount );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::BitwiseAnd<_Ty><<<gridDim, blockDim>>>( out, in, constant, aBroadcastHint, blockSizes, broadcastSizes );
    }

    void BitwiseAndOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right,
                       broadcast_hint_t aBroadcastHint, memory_buffer_t &blockSizes, uint32_t maxBlockSize,
                       memory_buffer_t &broadcastSizes, uint32_t maxBroadcastSizes )
    {
        DISPATCH_BY_INTEGRAL_TYPE( tensorElementType, BitwiseAnd_Tensor_Tensor_Impl,
                                   ( out, left, right, aBroadcastHint, blockSizes, maxBlockSize, broadcastSizes, maxBroadcastSizes ) );
    }

    template <typename _Ty>
    void BitwiseAnd_Tensor_Tensor_Impl( multi_tensor_t &out, multi_tensor_t &in, multi_tensor_t &constant )
    {
        int blockCount = ( in.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::BitwiseAnd<_Ty><<<gridDim, blockDim>>>( out, in, constant );
    }

    void BitwiseAndOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right )
    {
        DISPATCH_BY_INTEGRAL_TYPE( tensorElementType, BitwiseAnd_Tensor_Tensor_Impl, ( out, left, right ) );
    }

    template <typename _Ty>
    void BitwiseAnd_Tensor_Vector_Impl( multi_tensor_t &out, multi_tensor_t &in, memory_buffer_t &constant )
    {
        int blockCount = ( in.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::BitwiseAnd<_Ty><<<gridDim, blockDim>>>( out, in, constant );
    }

    void BitwiseAndOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, memory_buffer_t &right )
    {
        DISPATCH_BY_INTEGRAL_TYPE( tensorElementType, BitwiseAnd_Tensor_Vector_Impl, ( out, left, right ) );
    }

    void BitwiseAndOp( scalar_type_t tensorElementType, multi_tensor_t &out, memory_buffer_t &left, multi_tensor_t &right )
    {
        BitwiseAndOp( tensorElementType, out, right, left );
    }

    template <typename _Ty>
    void BitwiseOr_Tensor_Scalar_Impl( multi_tensor_t &out, multi_tensor_t &in, scalar_value_t &constant )
    {
        int blockCount = ( in.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::BitwiseOr<<<gridDim, blockDim>>>( out, in, std::get<_Ty>( constant ) );
    }

    void BitwiseOrOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, scalar_value_t &right )
    {
        DISPATCH_BY_INTEGRAL_TYPE( tensorElementType, BitwiseOr_Tensor_Scalar_Impl, ( out, left, right ) );
    }

    void BitwiseOrOp( scalar_type_t tensorElementType, multi_tensor_t &out, scalar_value_t &left, multi_tensor_t &right )
    {
        BitwiseOrOp( tensorElementType, out, right, left );
    }

    template <typename _Ty>
    void BitwiseOr_Tensor_Tensor_Impl( multi_tensor_t &out, multi_tensor_t &in, multi_tensor_t &constant )
    {
        int blockCount = ( in.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::BitwiseOr<_Ty><<<gridDim, blockDim>>>( out, in, constant );
    }

    void BitwiseOrOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right )
    {
        DISPATCH_BY_INTEGRAL_TYPE( tensorElementType, BitwiseOr_Tensor_Tensor_Impl, ( out, left, right ) );
    }

    template <typename _Ty>
    static void BitwiseOr_Tensor_Tensor_Impl( multi_tensor_t &out, multi_tensor_t &in, multi_tensor_t &constant,
                                              broadcast_hint_t aBroadcastHint, memory_buffer_t &blockSizes, uint32_t maxBlockSize,
                                              memory_buffer_t &broadcastSizes, uint32_t maxBroadcastSizes )
    {
        int blockCount = ( maxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), maxBlockSize, blockCount );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::BitwiseOr<_Ty><<<gridDim, blockDim>>>( out, in, constant, aBroadcastHint, blockSizes, broadcastSizes );
    }

    void BitwiseOrOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right,
                      broadcast_hint_t aBroadcastHint, memory_buffer_t &blockSizes, uint32_t maxBlockSize,
                      memory_buffer_t &broadcastSizes, uint32_t maxBroadcastSizes )
    {
        DISPATCH_BY_INTEGRAL_TYPE( tensorElementType, BitwiseOr_Tensor_Tensor_Impl,
                                   ( out, left, right, aBroadcastHint, blockSizes, maxBlockSize, broadcastSizes, maxBroadcastSizes ) );
    }

    template <typename _Ty>
    void BitwiseOrTensorVectorImpl( multi_tensor_t &out, multi_tensor_t &in, memory_buffer_t &constant )
    {
        int blockCount = ( in.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::BitwiseOr<_Ty><<<gridDim, blockDim>>>( out, in, constant );
    }

    void BitwiseOrOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, memory_buffer_t &right )
    {
        DISPATCH_BY_INTEGRAL_TYPE( tensorElementType, BitwiseOrTensorVectorImpl, ( out, left, right ) );
    }

    void BitwiseOrOp( scalar_type_t tensorElementType, multi_tensor_t &out, memory_buffer_t &left, multi_tensor_t &right )
    {
        BitwiseOrOp( tensorElementType, out, right, left );
    }

    template <typename _Ty>
    void BitwiseNotTensorImpl( multi_tensor_t &out, multi_tensor_t &in )
    {
        int blockCount = ( in.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Bitwise<_Ty><<<gridDim, blockDim>>>( out, in );
    }

    void BitwiseNotOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &aOperand )
    {
        DISPATCH_BY_INTEGRAL_TYPE( tensorElementType, BitwiseNotTensorImpl, ( out, aOperand ) );
    }

    template <typename _Ty>
    static void EqualOpImpl( multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right )
    {
        int blockCount = ( left.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( left.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::EqualOp<_Ty><<<gridDim, blockDim>>>( out, left, right );
    }

    void EqualOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right )
    {
        DISPATCH_BY_TYPE( tensorElementType, EqualOpImpl, ( out, left, right ) );
    }

    template <typename _Ty>
    static void EqualOpImpl( multi_tensor_t &out, multi_tensor_t &in, multi_tensor_t &constant, broadcast_hint_t aBroadcastHint,
                             memory_buffer_t &blockSizes, uint32_t maxBlockSize, memory_buffer_t &broadcastSizes,
                             uint32_t maxBroadcastSizes )
    {
        int blockCount = ( maxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), maxBlockSize, blockCount );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::EqualOp<_Ty><<<gridDim, blockDim>>>( out, in, constant, aBroadcastHint, blockSizes, broadcastSizes );
    }

    void EqualOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right,
                  broadcast_hint_t aBroadcastHint, memory_buffer_t &blockSizes, uint32_t maxBlockSize, memory_buffer_t &broadcastSizes,
                  uint32_t maxBroadcastSizes )
    {
        DISPATCH_BY_TYPE( tensorElementType, EqualOpImpl,
                          ( out, left, right, aBroadcastHint, blockSizes, maxBlockSize, broadcastSizes, maxBroadcastSizes ) );
    }

    template <typename _Ty>
    static void EqualOpImpl( multi_tensor_t &out, multi_tensor_t &left, memory_buffer_t &right )
    {
        int blockCount = ( left.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( left.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::EqualOp<_Ty><<<gridDim, blockDim>>>( out, left, right );
    }

    void EqualOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, memory_buffer_t &right )
    {
        DISPATCH_BY_TYPE( tensorElementType, EqualOpImpl, ( out, left, right ) );
    }

    template <typename _ScalarType>
    static void EqualOpImpl( multi_tensor_t &out, multi_tensor_t &left, scalar_value_t &right )
    {
        int blockCount = ( left.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( left.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::EqualOp<_ScalarType><<<gridDim, blockDim>>>( out, left, std::get<_ScalarType>( right ) );
    }

    void EqualOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, scalar_value_t &right )
    {
        DISPATCH_BY_TYPE( tensorElementType, EqualOpImpl, ( out, left, right ) );
    }

    template <typename _Ty>
    static void EqualOpImpl( multi_tensor_t &out, memory_buffer_t &left, multi_tensor_t &right )
    {
        int blockCount = ( right.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( right.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::EqualOp<_Ty><<<gridDim, blockDim>>>( out, left, right );
    }

    void EqualOp( scalar_type_t tensorElementType, multi_tensor_t &out, memory_buffer_t &left, multi_tensor_t &right )
    {
        DISPATCH_BY_TYPE( tensorElementType, EqualOpImpl, ( out, left, right ) );
    }

    template <typename _Ty>
    static void EqualOpImpl( multi_tensor_t &out, scalar_value_t &left, multi_tensor_t &right )
    {
        int blockCount = ( right.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( right.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::EqualOp<_Ty><<<gridDim, blockDim>>>( out, std::get<_Ty>( left ), right );
    }

    void EqualOp( scalar_type_t tensorElementType, multi_tensor_t &out, scalar_value_t &left, multi_tensor_t &right )
    {
        DISPATCH_BY_TYPE( tensorElementType, EqualOpImpl, ( out, left, right ) );
    }

    template <typename _Ty>
    static void LessThanOpImpl( multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right )
    {
        int blockCount = ( left.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( left.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOp<_Ty><<<gridDim, blockDim>>>( out, left, right );
    }

    void LessThanOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right )
    {
        DISPATCH_BY_TYPE( tensorElementType, LessThanOpImpl, ( out, left, right ) );
    }

    template <typename _Ty>
    static void LessThanOpImpl( multi_tensor_t &out, multi_tensor_t &in, multi_tensor_t &constant, broadcast_hint_t aBroadcastHint,
                                memory_buffer_t &blockSizes, uint32_t maxBlockSize, memory_buffer_t &broadcastSizes,
                                uint32_t maxBroadcastSizes )
    {
        int blockCount = ( maxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), maxBlockSize, blockCount );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOp<_Ty><<<gridDim, blockDim>>>( out, in, constant, aBroadcastHint, blockSizes, broadcastSizes );
    }

    void LessThanOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right,
                     broadcast_hint_t aBroadcastHint, memory_buffer_t &blockSizes, uint32_t maxBlockSize,
                     memory_buffer_t &broadcastSizes, uint32_t maxBroadcastSizes )
    {
        DISPATCH_BY_TYPE( tensorElementType, LessThanOpImpl,
                          ( out, left, right, aBroadcastHint, blockSizes, maxBlockSize, broadcastSizes, maxBroadcastSizes ) );
    }

    template <typename _Ty>
    static void LessThanOpImpl( multi_tensor_t &out, multi_tensor_t &left, memory_buffer_t &right )
    {
        int blockCount = ( left.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( left.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOp<_Ty><<<gridDim, blockDim>>>( out, left, right );
    }

    void LessThanOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, memory_buffer_t &right )
    {
        DISPATCH_BY_TYPE( tensorElementType, LessThanOpImpl, ( out, left, right ) );
    }

    template <typename _ScalarType>
    static void LessThanOpImpl( multi_tensor_t &out, multi_tensor_t &left, scalar_value_t &right )
    {
        int blockCount = ( left.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( left.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOp<_ScalarType><<<gridDim, blockDim>>>( out, left, std::get<_ScalarType>( right ) );
    }

    void LessThanOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, scalar_value_t &right )
    {
        DISPATCH_BY_TYPE( tensorElementType, LessThanOpImpl, ( out, left, right ) );
    }

    template <typename _Ty>
    static void LessThanOpImpl( multi_tensor_t &out, memory_buffer_t &left, multi_tensor_t &right )
    {
        int blockCount = ( right.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( right.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOp<_Ty><<<gridDim, blockDim>>>( out, left, right );
    }

    void LessThanOp( scalar_type_t tensorElementType, multi_tensor_t &out, memory_buffer_t &left, multi_tensor_t &right )
    {
        DISPATCH_BY_TYPE( tensorElementType, LessThanOpImpl, ( out, left, right ) );
    }

    template <typename _Ty>
    static void LessThanOpImpl( multi_tensor_t &out, scalar_value_t &left, multi_tensor_t &right )
    {
        int blockCount = ( right.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( right.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOp<_Ty><<<gridDim, blockDim>>>( out, std::get<_Ty>( left ), right );
    }

    void LessThanOp( scalar_type_t tensorElementType, multi_tensor_t &out, scalar_value_t &left, multi_tensor_t &right )
    {
        DISPATCH_BY_TYPE( tensorElementType, LessThanOpImpl, ( out, left, right ) );
    }

    template <typename _Ty>
    static void LessThanOrEqualOpImpl( multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right )
    {
        int blockCount = ( left.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( left.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOrEqualOp<_Ty><<<gridDim, blockDim>>>( out, left, right );
    }

    void LessThanOrEqualOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right )
    {
        DISPATCH_BY_TYPE( tensorElementType, LessThanOrEqualOpImpl, ( out, left, right ) );
    }

    template <typename _Ty>
    static void LessThanOrEqualOpImpl( multi_tensor_t &out, multi_tensor_t &in, multi_tensor_t &constant,
                                       broadcast_hint_t aBroadcastHint, memory_buffer_t &blockSizes, uint32_t maxBlockSize,
                                       memory_buffer_t &broadcastSizes, uint32_t maxBroadcastSizes )
    {
        int blockCount = ( maxBroadcastSizes / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( in.Shape().CountLayers(), maxBlockSize, blockCount );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOrEqualOp<_Ty><<<gridDim, blockDim>>>( out, in, constant, aBroadcastHint, blockSizes, broadcastSizes );
    }

    void LessThanOrEqualOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right,
                            broadcast_hint_t aBroadcastHint, memory_buffer_t &blockSizes, uint32_t maxBlockSize,
                            memory_buffer_t &broadcastSizes, uint32_t maxBroadcastSizes )
    {
        DISPATCH_BY_TYPE( tensorElementType, LessThanOrEqualOpImpl,
                          ( out, left, right, aBroadcastHint, blockSizes, maxBlockSize, broadcastSizes, maxBroadcastSizes ) );
    }

    template <typename _Ty>
    static void LessThanOrEqualOpImpl( multi_tensor_t &out, multi_tensor_t &left, memory_buffer_t &right )
    {
        int blockCount = ( left.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( left.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOrEqualOp<_Ty><<<gridDim, blockDim>>>( out, left, right );
    }

    void LessThanOrEqualOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, memory_buffer_t &right )
    {
        DISPATCH_BY_TYPE( tensorElementType, LessThanOrEqualOpImpl, ( out, left, right ) );
    }

    template <typename _ScalarType>
    static void LessThanOrEqualOpImpl( multi_tensor_t &out, multi_tensor_t &left, scalar_value_t &right )
    {
        int blockCount = ( left.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( left.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOrEqualOp<_ScalarType><<<gridDim, blockDim>>>( out, left, std::get<_ScalarType>( right ) );
    }

    void LessThanOrEqualOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, scalar_value_t &right )
    {
        DISPATCH_BY_TYPE( tensorElementType, LessThanOrEqualOpImpl, ( out, left, right ) );
    }

    template <typename _Ty>
    static void LessThanOrEqualOpImpl( multi_tensor_t &out, memory_buffer_t &left, multi_tensor_t &right )
    {
        int blockCount = ( right.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( right.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOrEqualOp<_Ty><<<gridDim, blockDim>>>( out, left, right );
    }

    void LessThanOrEqualOp( scalar_type_t tensorElementType, multi_tensor_t &out, memory_buffer_t &left, multi_tensor_t &right )
    {
        DISPATCH_BY_TYPE( tensorElementType, LessThanOrEqualOpImpl, ( out, left, right ) );
    }

    template <typename _Ty>
    static void LessThanOrEqualOpImpl( multi_tensor_t &out, scalar_value_t &left, multi_tensor_t &right )
    {
        int blockCount = ( right.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( right.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::LessThanOrEqualOp<_Ty><<<gridDim, blockDim>>>( out, std::get<_Ty>( left ), right );
    }

    void LessThanOrEqualOp( scalar_type_t tensorElementType, multi_tensor_t &out, scalar_value_t &left, multi_tensor_t &right )
    {
        DISPATCH_BY_TYPE( tensorElementType, LessThanOrEqualOpImpl, ( out, left, right ) );
    }

    template <typename _Ty>
    static void InIntervalTensorTensorImpl( multi_tensor_t &out, multi_tensor_t &aX, multi_tensor_t &aLower, multi_tensor_t &aUpper,
                                            bool aStrictLower, bool aStrictUpper )
    {
        int blockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aX.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::InInterval<_Ty><<<gridDim, blockDim>>>( out, aX, aLower, aUpper, aStrictLower, aStrictUpper );
    }

    void InIntervalOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &aX, multi_tensor_t &aLower,
                       multi_tensor_t &aUpper, bool aStrictLower, bool aStrictUpper )
    {
        DISPATCH_BY_TYPE( tensorElementType, InIntervalTensorTensorImpl, ( out, aX, aLower, aUpper, aStrictLower, aStrictUpper ) );
    }

    template <typename _Ty>
    static void InIntervalTensorVectorImpl( multi_tensor_t &out, multi_tensor_t &aX, multi_tensor_t &aLower, memory_buffer_t &aUpper,
                                            bool aStrictLower, bool aStrictUpper )
    {
        int blockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aX.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::InInterval<_Ty><<<gridDim, blockDim>>>( out, aX, aLower, aUpper, aStrictLower, aStrictUpper );
    }

    void InIntervalOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &aX, multi_tensor_t &aLower,
                       memory_buffer_t &aUpper, bool aStrictLower, bool aStrictUpper )
    {
        DISPATCH_BY_TYPE( tensorElementType, InIntervalTensorVectorImpl, ( out, aX, aLower, aUpper, aStrictLower, aStrictUpper ) );
    }

    template <typename _Ty>
    static void InIntervalTensorScalarImpl( multi_tensor_t &out, multi_tensor_t &aX, multi_tensor_t &aLower, scalar_value_t &aUpper,
                                            bool aStrictLower, bool aStrictUpper )
    {
        int blockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aX.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::InInterval<_Ty><<<gridDim, blockDim>>>( out, aX, aLower, std::get<_Ty>( aUpper ), aStrictLower, aStrictUpper );
    }

    void InIntervalOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &aX, multi_tensor_t &aLower,
                       scalar_value_t &aUpper, bool aStrictLower, bool aStrictUpper )
    {
        DISPATCH_BY_TYPE( tensorElementType, InIntervalTensorScalarImpl, ( out, aX, aLower, aUpper, aStrictLower, aStrictUpper ) );
    }

    template <typename _Ty>
    static void InIntervalVectorTensorImpl( multi_tensor_t &out, multi_tensor_t &aX, memory_buffer_t &aLower, multi_tensor_t &aUpper,
                                            bool aStrictLower, bool aStrictUpper )
    {
        int blockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aX.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::InInterval<_Ty><<<gridDim, blockDim>>>( out, aX, aLower, aUpper, aStrictLower, aStrictUpper );
    }

    void InIntervalOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &aX, memory_buffer_t &aLower,
                       multi_tensor_t &aUpper, bool aStrictLower, bool aStrictUpper )
    {
        DISPATCH_BY_TYPE( tensorElementType, InIntervalVectorTensorImpl, ( out, aX, aLower, aUpper, aStrictLower, aStrictUpper ) );
    }

    template <typename _Ty>
    static void InIntervalVectorVectorImpl( multi_tensor_t &out, multi_tensor_t &aX, memory_buffer_t &aLower, memory_buffer_t &aUpper,
                                            bool aStrictLower, bool aStrictUpper )
    {
        int blockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aX.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::InInterval<_Ty><<<gridDim, blockDim>>>( out, aX, aLower, aUpper, aStrictLower, aStrictUpper );
    }

    void InIntervalOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &aX, memory_buffer_t &aLower,
                       memory_buffer_t &aUpper, bool aStrictLower, bool aStrictUpper )
    {
        DISPATCH_BY_TYPE( tensorElementType, InIntervalVectorVectorImpl, ( out, aX, aLower, aUpper, aStrictLower, aStrictUpper ) );
    }

    template <typename _Ty>
    static void InIntervalVectorScalarImpl( multi_tensor_t &out, multi_tensor_t &aX, memory_buffer_t &aLower, scalar_value_t &aUpper,
                                            bool aStrictLower, bool aStrictUpper )
    {
        int blockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aX.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::InInterval<_Ty><<<gridDim, blockDim>>>( out, aX, aLower, std::get<_Ty>( aUpper ), aStrictLower, aStrictUpper );
    }

    void InIntervalOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &aX, memory_buffer_t &aLower,
                       scalar_value_t &aUpper, bool aStrictLower, bool aStrictUpper )
    {
        DISPATCH_BY_TYPE( tensorElementType, InIntervalVectorScalarImpl, ( out, aX, aLower, aUpper, aStrictLower, aStrictUpper ) );
    }

    template <typename _Ty>
    static void InIntervalScalarTensorImpl( multi_tensor_t &out, multi_tensor_t &aX, scalar_value_t &aLower, multi_tensor_t &aUpper,
                                            bool aStrictLower, bool aStrictUpper )
    {
        int blockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aX.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::InInterval<_Ty><<<gridDim, blockDim>>>( out, aX, std::get<_Ty>( aLower ), aUpper, aStrictLower, aStrictUpper );
    }

    void InIntervalOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &aX, scalar_value_t &aLower,
                       multi_tensor_t &aUpper, bool aStrictLower, bool aStrictUpper )
    {
        DISPATCH_BY_TYPE( tensorElementType, InIntervalScalarTensorImpl, ( out, aX, aLower, aUpper, aStrictLower, aStrictUpper ) );
    }

    template <typename _Ty>
    static void InIntervalScalarVectorImpl( multi_tensor_t &out, multi_tensor_t &aX, scalar_value_t &aLower, memory_buffer_t &aUpper,
                                            bool aStrictLower, bool aStrictUpper )
    {
        int blockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aX.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::InInterval<_Ty><<<gridDim, blockDim>>>( out, aX, std::get<_Ty>( aLower ), aUpper, aStrictLower, aStrictUpper );
    }

    void InIntervalOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &aX, scalar_value_t &aLower,
                       memory_buffer_t &aUpper, bool aStrictLower, bool aStrictUpper )
    {
        DISPATCH_BY_TYPE( tensorElementType, InIntervalScalarVectorImpl, ( out, aX, aLower, aUpper, aStrictLower, aStrictUpper ) );
    }

    template <typename _Ty>
    static void InIntervalScalarScalarImpl( multi_tensor_t &out, multi_tensor_t &aX, scalar_value_t &aLower, scalar_value_t &aUpper,
                                            bool aStrictLower, bool aStrictUpper )
    {
        int blockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aX.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::InInterval<_Ty>
            <<<gridDim, blockDim>>>( out, aX, std::get<_Ty>( aLower ), std::get<_Ty>( aUpper ), aStrictLower, aStrictUpper );
    }

    void InIntervalOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &aX, scalar_value_t &aLower,
                       scalar_value_t &aUpper, bool aStrictLower, bool aStrictUpper )
    {
        DISPATCH_BY_TYPE( tensorElementType, InIntervalScalarScalarImpl, ( out, aX, aLower, aUpper, aStrictLower, aStrictUpper ) );
    }

    template <typename _Ty>
    static void WhereOpTensorTensorImpl( multi_tensor_t &out, multi_tensor_t &aCondition, multi_tensor_t &aValueIfTrue,
                                         multi_tensor_t &aValueIfFalse )
    {
        int blockCount = ( aCondition.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aCondition.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::WhereTensorTensor<_Ty><<<gridDim, blockDim>>>( out, aCondition, aValueIfTrue, aValueIfFalse );
    }

    void WhereOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &aCondition, multi_tensor_t &aValueIfTrue,
                  multi_tensor_t &aValueIfFalse )
    {
        DISPATCH_BY_TYPE( tensorElementType, WhereOpTensorTensorImpl, ( out, aCondition, aValueIfTrue, aValueIfFalse ) );
    }

    template <typename _Ty>
    static void WhereTensorVectorImpl( multi_tensor_t &out, multi_tensor_t &aCondition, multi_tensor_t &aValueIfTrue,
                                       memory_buffer_t &aValueIfFalse )
    {
        int blockCount = ( aCondition.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aCondition.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::WhereTensorVector<_Ty><<<gridDim, blockDim>>>( out, aCondition, aValueIfTrue, aValueIfFalse );
    }

    void WhereOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &aCondition, multi_tensor_t &aValueIfTrue,
                  memory_buffer_t &aValueIfFalse )
    {
        DISPATCH_BY_TYPE( tensorElementType, WhereTensorVectorImpl, ( out, aCondition, aValueIfTrue, aValueIfFalse ) );
    }

    template <typename _Ty>
    static void WhereTensorScalarImpl( multi_tensor_t &out, multi_tensor_t &aCondition, multi_tensor_t &aValueIfTrue,
                                       scalar_value_t &aValueIfFalse )
    {
        int blockCount = ( aCondition.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aCondition.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::WhereTensorScalar<_Ty><<<gridDim, blockDim>>>( out, aCondition, aValueIfTrue, std::get<_Ty>( aValueIfFalse ) );
    }

    void WhereOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &aCondition, multi_tensor_t &aValueIfTrue,
                  scalar_value_t &aValueIfFalse )
    {
        DISPATCH_BY_TYPE( tensorElementType, WhereTensorScalarImpl, ( out, aCondition, aValueIfTrue, aValueIfFalse ) );
    }

    template <typename _Ty>
    static void WhereVectorTensorImpl( multi_tensor_t &out, multi_tensor_t &aCondition, memory_buffer_t &aValueIfTrue,
                                       multi_tensor_t &aValueIfFalse )
    {
        int blockCount = ( aCondition.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aCondition.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::WhereVectorTensor<_Ty><<<gridDim, blockDim>>>( out, aCondition, aValueIfTrue, aValueIfFalse );
    }
    void WhereOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &aCondition, memory_buffer_t &aValueIfTrue,
                  multi_tensor_t &aValueIfFalse )
    {
        DISPATCH_BY_TYPE( tensorElementType, WhereVectorTensorImpl, ( out, aCondition, aValueIfTrue, aValueIfFalse ) );
    }

    template <typename _Ty>
    static void WhereVectorVectorImpl( multi_tensor_t &out, multi_tensor_t &aCondition, memory_buffer_t &aValueIfTrue,
                                       memory_buffer_t &aValueIfFalse )
    {
        int blockCount = ( aCondition.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aCondition.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::WhereVectorVector<_Ty><<<gridDim, blockDim>>>( out, aCondition, aValueIfTrue, aValueIfFalse );
    }

    void WhereOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &aCondition, memory_buffer_t &aValueIfTrue,
                  memory_buffer_t &aValueIfFalse )
    {
        DISPATCH_BY_TYPE( tensorElementType, WhereVectorVectorImpl, ( out, aCondition, aValueIfTrue, aValueIfFalse ) );
    }

    template <typename _Ty>
    static void WhereVectorScalarImpl( multi_tensor_t &out, multi_tensor_t &aCondition, memory_buffer_t &aValueIfTrue,
                                       scalar_value_t &aValueIfFalse )
    {
        int blockCount = ( aCondition.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aCondition.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::WhereVectorScalar<_Ty><<<gridDim, blockDim>>>( out, aCondition, aValueIfTrue, std::get<_Ty>( aValueIfFalse ) );
    }

    void WhereOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &aCondition, memory_buffer_t &aValueIfTrue,
                  scalar_value_t &aValueIfFalse )
    {
        DISPATCH_BY_TYPE( tensorElementType, WhereVectorScalarImpl, ( out, aCondition, aValueIfTrue, aValueIfFalse ) );
    }

    template <typename _Ty>
    static void WhereScalarTensorImpl( multi_tensor_t &out, multi_tensor_t &aCondition, scalar_value_t &aValueIfTrue,
                                       multi_tensor_t &aValueIfFalse )
    {
        int blockCount = ( aCondition.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aCondition.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::WhereScalarTensor<_Ty><<<gridDim, blockDim>>>( out, aCondition, std::get<_Ty>( aValueIfTrue ), aValueIfFalse );
    }

    void WhereOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &aCondition, scalar_value_t &aValueIfTrue,
                  multi_tensor_t &aValueIfFalse )
    {
        DISPATCH_BY_TYPE( tensorElementType, WhereScalarTensorImpl, ( out, aCondition, aValueIfTrue, aValueIfFalse ) );
    }

    template <typename _Ty>
    static void WhereScalarVectorImpl( multi_tensor_t &out, multi_tensor_t &aCondition, scalar_value_t &aValueIfTrue,
                                       memory_buffer_t &aValueIfFalse )
    {
        int blockCount = ( aCondition.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aCondition.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::WhereScalarVector<_Ty><<<gridDim, blockDim>>>( out, aCondition, std::get<_Ty>( aValueIfTrue ), aValueIfFalse );
    }

    void WhereOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &aCondition, scalar_value_t &aValueIfTrue,
                  memory_buffer_t &aValueIfFalse )
    {
        DISPATCH_BY_TYPE( tensorElementType, WhereScalarVectorImpl, ( out, aCondition, aValueIfTrue, aValueIfFalse ) );
    }

    template <typename _Ty>
    static void WhereScalarScalarImpl( multi_tensor_t &out, multi_tensor_t &aCondition, scalar_value_t &aValueIfTrue,
                                       scalar_value_t &aValueIfFalse )
    {
        int blockCount = ( aCondition.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aCondition.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::WhereScalarScalar<_Ty>
            <<<gridDim, blockDim>>>( out, aCondition, std::get<_Ty>( aValueIfTrue ), std::get<_Ty>( aValueIfFalse ) );
    }

    void WhereOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &aCondition, scalar_value_t &aValueIfTrue,
                  scalar_value_t &aValueIfFalse )
    {
        DISPATCH_BY_TYPE( tensorElementType, WhereScalarScalarImpl, ( out, aCondition, aValueIfTrue, aValueIfFalse ) );
    }

    template <typename _Ty>
    static void RepeatOpImpl( multi_tensor_t &out, multi_tensor_t &aArray, memory_buffer_t &aRepetitions, uint32_t lMaxRepetitions )
    {
        int blockCount = ( lMaxRepetitions / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aArray.Shape().CountLayers(), aArray.Shape().mMaxBufferSize, blockCount );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Repeat<_Ty><<<gridDim, blockDim>>>( out, aArray, aRepetitions );
    }

    void RepeatOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &aArray, memory_buffer_t &aRepetitions,
                   uint32_t lMaxRepetitions )
    {
        DISPATCH_BY_TYPE( tensorElementType, RepeatOpImpl, ( out, aArray, aRepetitions, lMaxRepetitions ) );
    }

    template <typename _Ty>
    static void TileOpImpl( multi_tensor_t &out, multi_tensor_t &aArray, memory_buffer_t &aRepetitions, uint32_t lMaxRepetitions )
    {
        int blockCount = ( aArray.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aArray.Shape().CountLayers(), lMaxRepetitions, blockCount );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Tile<_Ty><<<gridDim, blockDim>>>( out, aArray, aRepetitions );
    }

    void TileOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &aArray, memory_buffer_t &aRepetitions,
                 uint32_t lMaxRepetitions )
    {
        DISPATCH_BY_TYPE( tensorElementType, TileOpImpl, ( out, aArray, aRepetitions, lMaxRepetitions ) );
    }

    template <typename _Ty>
    static void LinearSpaceOpImpl( multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right, memory_buffer_t &aSubdivisions,
                                   uint32_t aMaxSubdivisions )
    {
        int blockCount = ( aMaxSubdivisions / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( left.Shape().CountLayers(), left.Shape().mMaxBufferSize, blockCount );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::LinearSpace<_Ty><<<gridDim, blockDim>>>( out, left, right, aSubdivisions );
    }

    void LinearSpaceOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right,
                        memory_buffer_t &aSubdivisions, uint32_t aMaxSubdivisions )
    {
        switch( tensorElementType )
        {
        case scalar_type_t::FLOAT32:
        {
            LinearSpaceOpImpl<float>( out, left, right, aSubdivisions, aMaxSubdivisions );
            break;
        }
        case scalar_type_t::FLOAT64:
        {
            LinearSpaceOpImpl<double>( out, left, right, aSubdivisions, aMaxSubdivisions );
            break;
        }
        default:
            throw std::runtime_error( "Linear space only supports float and double values" );
        }
    }

    template <typename _Ty>
    static void MixImpl( multi_tensor_t &out, multi_tensor_t &A, multi_tensor_t &B, multi_tensor_t &t )
    {
        int blockCount = ( A.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( A.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Mix<_Ty><<<gridDim, blockDim>>>( out, A, B, t );
    }

    void MixOp( scalar_type_t tensorElementType, multi_tensor_t &out, multi_tensor_t &A, multi_tensor_t &B, multi_tensor_t &t )
    {
        DISPATCH_BY_TYPE( tensorElementType, MixImpl, ( out, A, B, t ) );
    }

    void Sample2DOp( multi_tensor_t &out, multi_tensor_t &X, multi_tensor_t &Y, memory_buffer_t &aTextures )
    {
        int blockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( X.Shape().CountLayers(), blockCount );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Sample2D<<<gridDim, blockDim>>>( out, X, Y, aTextures );
    }

    void Sample2DOp( multi_tensor_t &out, multi_tensor_t &X, memory_buffer_t &Y, memory_buffer_t &aTextures )
    {
        int blockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( X.Shape().CountLayers(), blockCount );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Sample2D<<<gridDim, blockDim>>>( out, X, Y, aTextures );
    }

    void Sample2DOp( multi_tensor_t &out, multi_tensor_t &X, scalar_value_t &Y, memory_buffer_t &aTextures )
    {
        int blockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( X.Shape().CountLayers(), blockCount );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Sample2D<<<gridDim, blockDim>>>( out, X, std::get<float>( Y ), aTextures );
    }

    void Sample2DOp( multi_tensor_t &out, memory_buffer_t &X, multi_tensor_t &Y, memory_buffer_t &aTextures )
    {
        int blockCount = ( Y.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( Y.Shape().CountLayers(), blockCount );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Sample2D<<<gridDim, blockDim>>>( out, X, Y, aTextures );
    }

    void Sample2DOp( multi_tensor_t &out, scalar_value_t &X, multi_tensor_t &Y, memory_buffer_t &aTextures )
    {
        int blockCount = ( Y.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( Y.Shape().CountLayers(), blockCount );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Sample2D<<<gridDim, blockDim>>>( out, std::get<float>( X ), Y, aTextures );
    }

    template <typename _Ty>
    static void ToFixedPointOpImpl( multi_tensor_t &out, scalar_type_t outputElementType, multi_tensor_t &aArray, _Ty aScaling )
    {
        int blockCount = ( aArray.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aArray.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        switch( outputElementType )
        {
        case scalar_type_t::UINT8:
        {
            Kernels::ToFixedPoint<_Ty, uint8_t><<<gridDim, blockDim>>>( out, aArray, aScaling );
            break;
        }
        case scalar_type_t::UINT16:
        {
            Kernels::ToFixedPoint<_Ty, uint16_t><<<gridDim, blockDim>>>( out, aArray, aScaling );
            break;
        }
        case scalar_type_t::UINT32:
        {
            Kernels::ToFixedPoint<_Ty, uint32_t><<<gridDim, blockDim>>>( out, aArray, aScaling );
            break;
        }
        case scalar_type_t::UINT64:
        {
            Kernels::ToFixedPoint<_Ty, uint64_t><<<gridDim, blockDim>>>( out, aArray, aScaling );
            break;
        }
        case scalar_type_t::INT8:
        {
            Kernels::ToFixedPoint<_Ty, int8_t><<<gridDim, blockDim>>>( out, aArray, aScaling );
            break;
        }
        case scalar_type_t::INT16:
        {
            Kernels::ToFixedPoint<_Ty, int16_t><<<gridDim, blockDim>>>( out, aArray, aScaling );
            break;
        }
        case scalar_type_t::INT32:
        {
            Kernels::ToFixedPoint<_Ty, int32_t><<<gridDim, blockDim>>>( out, aArray, aScaling );
            break;
        }
        case scalar_type_t::INT64:
        {
            Kernels::ToFixedPoint<_Ty, int64_t><<<gridDim, blockDim>>>( out, aArray, aScaling );
            break;
        }
        default:
            throw std::runtime_error( "Linear space only supports float and double values" );
        }
    }

    void ToFixedPointOp( scalar_type_t tensorElementType, multi_tensor_t &out, scalar_type_t outputElementType, multi_tensor_t &aArray,
                         scalar_value_t &aScaling )
    {
        switch( tensorElementType )
        {
        case scalar_type_t::FLOAT32:
        {
            ToFixedPointOpImpl<float>( out, outputElementType, aArray, std::get<float>( aScaling ) );
            break;
        }
        case scalar_type_t::FLOAT64:
        {
            ToFixedPointOpImpl<double>( out, outputElementType, aArray, std::get<double>( aScaling ) );
            break;
        }
        default:
            throw std::runtime_error( "Linear space only supports float and double values" );
        }
    }

    template <typename _Ty>
    static void AffineTransformImpl( multi_tensor_t &out, multi_tensor_t &A, multi_tensor_t &X, multi_tensor_t &B )
    {
        int blockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( X.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::AffineTransform<_Ty><<<gridDim, blockDim>>>( out, A, X, B );
    }

    template <typename _Ty>
    static void AffineTransformImpl( multi_tensor_t &out, multi_tensor_t &A, multi_tensor_t &X, memory_buffer_t &B )
    {
        int blockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( X.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::AffineTransform<_Ty><<<gridDim, blockDim>>>( out, A, X, B );
    }

    template <typename _Ty>
    static void AffineTransformImpl( multi_tensor_t &out, multi_tensor_t &A, multi_tensor_t &X, scalar_value_t &B )
    {
        int blockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( X.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::AffineTransform<_Ty><<<gridDim, blockDim>>>( out, A, X, std::get<_Ty>( B ) );
    }

    template <typename _Ty>
    static void AffineTransformImpl( multi_tensor_t &out, memory_buffer_t &A, multi_tensor_t &X, multi_tensor_t &B )
    {
        int blockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( X.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::AffineTransform<_Ty><<<gridDim, blockDim>>>( out, A, X, B );
    }

    template <typename _Ty>
    static void AffineTransformImpl( multi_tensor_t &out, memory_buffer_t &A, multi_tensor_t &X, memory_buffer_t &B )
    {
        int blockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( X.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::AffineTransform<_Ty><<<gridDim, blockDim>>>( out, A, X, B );
    }

    template <typename _Ty>
    static void AffineTransformImpl( multi_tensor_t &out, memory_buffer_t &A, multi_tensor_t &X, scalar_value_t &B )
    {
        int blockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( X.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::AffineTransform<_Ty><<<gridDim, blockDim>>>( out, A, X, std::get<_Ty>( B ) );
    }

    template <typename _Ty>
    static void AffineTransformImpl( multi_tensor_t &out, scalar_value_t &A, multi_tensor_t &X, multi_tensor_t &B )
    {
        int blockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( X.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::AffineTransform<_Ty><<<gridDim, blockDim>>>( out, std::get<_Ty>( A ), X, B );
    }

    template <typename _Ty>
    static void AffineTransformImpl( multi_tensor_t &out, scalar_value_t &A, multi_tensor_t &X, memory_buffer_t &B )
    {
        int blockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( X.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::AffineTransform<_Ty><<<gridDim, blockDim>>>( out, std::get<_Ty>( A ), X, B );
    }

    template <typename _Ty>
    static void AffineTransformImpl( multi_tensor_t &out, scalar_value_t &A, multi_tensor_t &X, scalar_value_t &B )
    {
        int blockCount = ( X.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( X.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::AffineTransform<_Ty><<<gridDim, blockDim>>>( out, std::get<_Ty>( A ), X, std::get<_Ty>( B ) );
    }

    void AffineTransformOp( scalar_type_t outputElementType, multi_tensor_t &out, multi_tensor_t &A, multi_tensor_t &X,
                            multi_tensor_t &B )
    {
        DISPATCH_BY_TYPE( outputElementType, AffineTransformImpl, ( out, A, X, B ) );
    }

    void AffineTransformOp( scalar_type_t outputElementType, multi_tensor_t &out, multi_tensor_t &A, multi_tensor_t &X,
                            memory_buffer_t &B )
    {
        DISPATCH_BY_TYPE( outputElementType, AffineTransformImpl, ( out, A, X, B ) );
    }

    void AffineTransformOp( scalar_type_t outputElementType, multi_tensor_t &out, multi_tensor_t &A, multi_tensor_t &X,
                            scalar_value_t &B )
    {
        DISPATCH_BY_TYPE( outputElementType, AffineTransformImpl, ( out, A, X, B ) );
    }

    void AffineTransformOp( scalar_type_t outputElementType, multi_tensor_t &out, memory_buffer_t &A, multi_tensor_t &X,
                            multi_tensor_t &B )
    {
        DISPATCH_BY_TYPE( outputElementType, AffineTransformImpl, ( out, A, X, B ) );
    }

    void AffineTransformOp( scalar_type_t outputElementType, multi_tensor_t &out, memory_buffer_t &A, multi_tensor_t &X,
                            memory_buffer_t &B )
    {
        DISPATCH_BY_TYPE( outputElementType, AffineTransformImpl, ( out, A, X, B ) );
    }

    void AffineTransformOp( scalar_type_t outputElementType, multi_tensor_t &out, memory_buffer_t &A, multi_tensor_t &X,
                            scalar_value_t &B )
    {
        DISPATCH_BY_TYPE( outputElementType, AffineTransformImpl, ( out, A, X, B ) );
    }

    void AffineTransformOp( scalar_type_t outputElementType, multi_tensor_t &out, scalar_value_t &A, multi_tensor_t &X,
                            multi_tensor_t &B )
    {
        DISPATCH_BY_TYPE( outputElementType, AffineTransformImpl, ( out, A, X, B ) );
    }

    void AffineTransformOp( scalar_type_t outputElementType, multi_tensor_t &out, scalar_value_t &A, multi_tensor_t &X,
                            memory_buffer_t &B )
    {
        DISPATCH_BY_TYPE( outputElementType, AffineTransformImpl, ( out, A, X, B ) );
    }

    void AffineTransformOp( scalar_type_t outputElementType, multi_tensor_t &out, scalar_value_t &A, multi_tensor_t &X,
                            scalar_value_t &B )
    {
        DISPATCH_BY_TYPE( outputElementType, AffineTransformImpl, ( out, A, X, B ) );
    }

    void FloorOp( multi_tensor_t &out, multi_tensor_t &aX )
    {
        int blockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aX.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Floor<<<gridDim, blockDim>>>( out, aX );
    }

    void CeilOp( multi_tensor_t &out, multi_tensor_t &aX )
    {
        int blockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aX.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Ceil<<<gridDim, blockDim>>>( out, aX );
    }

    template <typename _Ty>
    void AbsImpl( multi_tensor_t &out, multi_tensor_t &aX )
    {
        int blockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aX.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Abs<_Ty><<<gridDim, blockDim>>>( out, aX );
    }

    void AbsOp( scalar_type_t outputElementType, multi_tensor_t &out, multi_tensor_t &aX )
    {
        DISPATCH_BY_SIGNED_TYPE( outputElementType, AbsImpl, ( out, aX ) );
    }

    template <typename _Ty>
    void SqrtImpl( multi_tensor_t &out, multi_tensor_t &aX )
    {
        int blockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aX.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Sqrt<_Ty><<<gridDim, blockDim>>>( out, aX );
    }

    void SqrtOp( scalar_type_t outputElementType, multi_tensor_t &out, multi_tensor_t &aX )
    {
        DISPATCH_BY_TYPE( outputElementType, SqrtImpl, ( out, aX ) );
    }

    template <typename _Ty>
    void RoundImpl( multi_tensor_t &out, multi_tensor_t &aX )
    {
        int blockCount = ( aX.Shape().mMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aX.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Round<_Ty><<<gridDim, blockDim>>>( out, aX );
    }

    void RoundOp( scalar_type_t outputElementType, multi_tensor_t &out, multi_tensor_t &aX )
    {
        DISPATCH_BY_TYPE( outputElementType, RoundImpl, ( out, aX ) );
    }

    void CountTrueOp( multi_tensor_t &out, multi_tensor_t &aX, memory_buffer_t &blockSizes, memory_buffer_t &aElementCount,
                      uint32_t maxBlockSize )
    {
        int blockCount = ( maxBlockSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aX.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::CountNonZero<uint8_t><<<gridDim, blockDim>>>( out, aX, blockSizes, aElementCount );
    }

    template <typename _Ty>
    void CountNonZeroImpl( multi_tensor_t &out, multi_tensor_t &aX, memory_buffer_t &blockSizes, memory_buffer_t &aElementCount,
                           uint32_t maxBlockSize )
    {
        int blockCount = ( maxBlockSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aX.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::CountNonZero<_Ty><<<gridDim, blockDim>>>( out, aX, blockSizes, aElementCount );
    }

    void CountNonZeroOp( scalar_type_t outputElementType, multi_tensor_t &out, multi_tensor_t &aX, memory_buffer_t &blockSizes,
                         memory_buffer_t &aElementCount, uint32_t maxBlockSize )
    {
        DISPATCH_BY_TYPE( outputElementType, CountNonZeroImpl, ( out, aX, blockSizes, aElementCount, maxBlockSize ) );
    }

    template <typename _Ty>
    void CountZeroImpl( multi_tensor_t &out, multi_tensor_t &aX, memory_buffer_t &blockSizes, memory_buffer_t &aElementCount,
                        uint32_t maxBlockSize )
    {
        int blockCount = ( maxBlockSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aX.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::CountZero<_Ty><<<gridDim, blockDim>>>( out, aX, blockSizes, aElementCount );
    }

    void CountZeroOp( scalar_type_t outputElementType, multi_tensor_t &out, multi_tensor_t &aX, memory_buffer_t &blockSizes,
                      memory_buffer_t &aElementCount, uint32_t maxBlockSize )
    {
        DISPATCH_BY_TYPE( outputElementType, CountZeroImpl, ( out, aX, blockSizes, aElementCount, maxBlockSize ) );
    }

    template <typename _Ty>
    void ArraySummationImpl( multi_tensor_t &out, multi_tensor_t &aX, memory_buffer_t &aBegin, memory_buffer_t &aEnd,
                             memory_buffer_t &aElementCount, memory_buffer_t &blockSizes, uint32_t maxBlockSize )
    {
        int blockCount = ( maxBlockSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aX.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::ArraySummation<_Ty><<<gridDim, blockDim>>>( out, aX, aBegin, aEnd, aElementCount, blockSizes );
    }

    void ArraySummationOp( scalar_type_t outputElementType, multi_tensor_t &out, multi_tensor_t &aX, memory_buffer_t &aBegin,
                           memory_buffer_t &aEnd, memory_buffer_t &aElementCount, memory_buffer_t &blockSizes, uint32_t maxBlockSize )
    {
        DISPATCH_BY_TYPE( outputElementType, ArraySummationImpl, ( out, aX, aBegin, aEnd, aElementCount, blockSizes, maxBlockSize ) );
    }

    template <typename _Ty>
    void ArraySliceImpl( multi_tensor_t &out, multi_tensor_t &aX, memory_buffer_t &aBegin, memory_buffer_t &aEnd,
                         memory_buffer_t &aElementCount, memory_buffer_t &blockSizes, uint32_t maxBlockSize )
    {
        int blockCount = ( maxBlockSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aX.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::ArraySlice<_Ty><<<gridDim, blockDim>>>( out, aX, aBegin, aEnd, aElementCount, blockSizes );
    }

    void ArraySliceOp( scalar_type_t outputElementType, multi_tensor_t &out, multi_tensor_t &aX, memory_buffer_t &aBegin,
                       memory_buffer_t &aEnd, memory_buffer_t &aElementCount, memory_buffer_t &blockSizes, uint32_t maxBlockSize )
    {
        DISPATCH_BY_TYPE( outputElementType, ArraySliceImpl, ( out, aX, aBegin, aEnd, aElementCount, blockSizes, maxBlockSize ) );
    }

    template <typename _Ty>
    void DiffImpl( multi_tensor_t &out, multi_tensor_t &aX, uint32_t aCount, memory_buffer_t &aElementCount,
                   memory_buffer_t &blockSizes, uint32_t maxBlockSize )
    {
        int blockCount = ( maxBlockSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aX.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Diff<_Ty><<<gridDim, blockDim>>>( out, aX, aCount, aElementCount, blockSizes );
    }

    void DiffOp( scalar_type_t outputElementType, multi_tensor_t &out, multi_tensor_t &aX, uint32_t aCount,
                 memory_buffer_t &aElementCount, memory_buffer_t &blockSizes, uint32_t maxBlockSize )
    {
        DISPATCH_BY_TYPE( outputElementType, DiffImpl, ( out, aX, aCount, aElementCount, blockSizes, maxBlockSize ) );
    }

    template <typename _Ty>
    void ShiftImpl( multi_tensor_t &out, multi_tensor_t &aX, int32_t aCount, scalar_value_t &aFillValue,
                    memory_buffer_t &aElementCount, memory_buffer_t &blockSizes, uint32_t maxBlockSize )
    {
        int blockCount = ( maxBlockSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aX.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        if( aCount < 0 )
            Kernels::ShiftLeft<_Ty><<<gridDim, blockDim>>>( out, aX, -aCount, std::get<_Ty>( aFillValue ), aElementCount, blockSizes );
        else
            Kernels::ShiftRight<_Ty><<<gridDim, blockDim>>>( out, aX, aCount, std::get<_Ty>( aFillValue ), aElementCount, blockSizes );
    }

    void ShiftOp( scalar_type_t outputElementType, multi_tensor_t &out, multi_tensor_t &aX, int32_t aCount, scalar_value_t &aFillValue,
                  memory_buffer_t &aElementCount, memory_buffer_t &blockSizes, uint32_t maxBlockSize )
    {
        DISPATCH_BY_TYPE( outputElementType, ShiftImpl, ( out, aX, aCount, aFillValue, aElementCount, blockSizes, maxBlockSize ) );
    }

    template <typename _Ty>
    void Conv1DImpl( multi_tensor_t &out, multi_tensor_t &aArray0, memory_buffer_t &aElementCount0, memory_buffer_t &blockSizes0,
                     uint32_t aMaxElementCount0, uint32_t maxBlockSize0, multi_tensor_t &aArray1, memory_buffer_t &aElementCount1,
                     memory_buffer_t blockSizes1, uint32_t maxBlockSize1 )
    {
        int blockCount = ( aMaxElementCount0 / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aArray0.Shape().CountLayers(), maxBlockSize0, blockCount );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::Conv1D<_Ty><<<gridDim, blockDim>>>( out, aArray0, aElementCount0, blockSizes0, aArray1, aElementCount1, blockSizes1 );
    }

    void Conv1DOp( scalar_type_t outputElementType, multi_tensor_t &out, multi_tensor_t &aArray0, memory_buffer_t &aElementCount0,
                   memory_buffer_t &blockSizes0, uint32_t aMaxElementCount0, uint32_t maxBlockSize0, multi_tensor_t &aArray1,
                   memory_buffer_t &aElementCount1, memory_buffer_t &blockSizes1, uint32_t maxBlockSize1 )
    {
        DISPATCH_BY_TYPE( outputElementType, Conv1DImpl,
                          ( out, aArray0, aElementCount0, blockSizes0, aMaxElementCount0, maxBlockSize0, aArray1, aElementCount1,
                            blockSizes1, maxBlockSize1 ) );
    }

    template <typename _Ty>
    void HCatImpl( multi_tensor_t &out, multi_tensor_t &aArray0, memory_buffer_t &aElementCount0, multi_tensor_t &aArray1,
                   memory_buffer_t &aElementCount1, memory_buffer_t &blockSizes, uint32_t maxBlockSize )
    {
        int blockCount = ( maxBlockSize / Private::ThreadsPerBlock ) + 1;

        dim3 gridDim( aArray0.Shape().CountLayers(), blockCount, 1 );
        dim3 blockDim( Private::ThreadsPerBlock );

        Kernels::HCat<_Ty><<<gridDim, blockDim>>>( out, aArray0, aElementCount0, aArray1, aElementCount1, blockSizes );
    }

    void HCatOp( scalar_type_t outputElementType, multi_tensor_t &out, multi_tensor_t &aArray0, memory_buffer_t &aElementCount0,
                 multi_tensor_t &aArray1, memory_buffer_t &aElementCount1, memory_buffer_t &blockSizes0, uint32_t maxBlockSize0 )
    {
        DISPATCH_BY_TYPE( outputElementType, HCatImpl,
                          ( out, aArray0, aElementCount0, aArray1, aElementCount1, blockSizes0, maxBlockSize0 ) );
    }

} // namespace SE::TensorOps