/// @file   Test_CudaBuffer.h
///
/// @brief  Cuda buffer abstraction unit tests
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <numeric>

#include "TestUtils.h"

#include "Core/Math/Types.h"
#include "Core/Cuda/MemoryPool.h"
#include "Core/Cuda/MultiTensor.h"
#include "Core/Cuda/PointerView.h"

using namespace LTSE::Cuda;
using namespace TestUtils;

struct TestStruct
{
    float a, b, c, d;
};

TEST_CASE( "GPU buffers have the appropriate sizes", "[CORE_CUDA]" )
{
    size_t lBufferSize = 128;

    SECTION( "Initial size is 0" )
    {
        Internal::sGPUDevicePointer lBuffer;
        REQUIRE( lBuffer.Size() == 0 );
    }

    SECTION( "Buffers are created with the appropriate size" )
    {
        auto lBuffer = Internal::sGPUDevicePointer( lBufferSize );
        REQUIRE( lBuffer.Size() == lBufferSize );
    }

    SECTION( "Vectors fetched from GPU have the correct length" )
    {
        auto lBuffer = Internal::sGPUDevicePointer( lBufferSize * sizeof( float ) );
        auto lData2  = lBuffer.Fetch<float>();
        REQUIRE( lData2.size() == lBufferSize );
    }
}

TEST_CASE( "GPU Transfer data between host and GPU", "[CORE_CUDA]" )
{
    size_t lBufferSize = 128;

    SECTION( "Upload vector to GPU buffer" )
    {
        auto lBuffer = Internal::sGPUDevicePointer( lBufferSize * sizeof( float ) );

        std::vector<float> lData( lBufferSize );
        float i = 1.0f;
        for( auto &x : lData )
        {
            x = i * 2.0f;
            i += 1.0f;
        }
        lBuffer.Upload( lData );

        auto lData2 = lBuffer.Fetch<float>();
        REQUIRE( lData == lData2 );
    }

    SECTION( "Setting GPU buffer to zero" )
    {
        auto lBuffer = Internal::sGPUDevicePointer( lBufferSize * sizeof( float ) );

        std::vector<float> lData( lBufferSize );
        std::vector<float> lZero( lBufferSize );
        std::fill( lZero.begin(), lZero.end(), 0.0f );
        float i = 1.0f;
        for( auto &x : lData )
        {
            x = i * 2.0f;
        }
        lBuffer.Upload( lData );
        lBuffer.Zero();
        auto lData2 = lBuffer.Fetch<float>();
        REQUIRE( lZero == lData2 );
    }

    SECTION( "Upload value to GPU buffer" )
    {
        auto lBuffer = Internal::sGPUDevicePointer( lBufferSize * sizeof( float ) );

        float lValue = 34.0f;
        lBuffer.Zero();
        lBuffer.Upload( lValue );

        auto lData2 = lBuffer.Fetch<float>();
        REQUIRE( lValue == lData2[0] );
        REQUIRE( 0.0f == lData2[1] );
    }

    SECTION( "Upload value to GPU buffer at a specific offset" )
    {
        auto lBuffer = Internal::sGPUDevicePointer( lBufferSize * sizeof( float ) );

        float lValue = 34.0f;
        lBuffer.Zero();
        lBuffer.Upload( lValue, 3 );

        std::vector<float> lData( lBufferSize );
        std::fill( lData.begin(), lData.end(), 0.0f );
        lData[3]    = lValue;
        auto lData2 = lBuffer.Fetch<float>();
        REQUIRE( lData == lData2 );
    }

    SECTION( "Fetch initial segment of buffer" )
    {
        auto lBuffer = Internal::sGPUDevicePointer( lBufferSize * sizeof( float ) );

        std::vector<float> lData( lBufferSize );
        float i = 1.0f;
        for( auto &x : lData )
        {
            x = i * 2.0f;
            i += 1.0f;
        }
        lBuffer.Upload( lData );

        auto lData2 = lBuffer.Fetch<float>( 32 );
        auto lData3 = std::vector<float>( lData.begin(), lData.begin() + 32 );
        REQUIRE( lData3 == lData2 );
        REQUIRE( lData2.size() == 32 );
    }

    SECTION( "Fetch part of buffer starting at a gicen offset" )
    {
        auto lBuffer = Internal::sGPUDevicePointer( lBufferSize * sizeof( float ) );

        std::vector<float> lData( lBufferSize );
        float i = 1.0f;
        for( auto &x : lData )
        {
            x = i * 2.0f;
            i += 1.0f;
        }
        lBuffer.Upload( lData );

        auto lData2 = lBuffer.Fetch<float>( 32, 32 );
        auto lData3 = std::vector<float>( lData.begin() + 32, lData.begin() + 64 );
        REQUIRE( lData3 == lData2 );
        REQUIRE( lData2.size() == 32 );
    }
}

TEST_CASE( "Memory Views", "[CORE_CUDA]" )
{
    size_t lBufferSize = 128;

    SECTION( "View into allocated buffer at given offset has the requested size" )
    {
        auto lBuffer     = Internal::sGPUDevicePointer( lBufferSize );
        auto lBufferView = Internal::sGPUDevicePointerView( 32, 32, lBuffer );
        REQUIRE( lBufferView.Size() == 32 );
    }

    SECTION( "Setting a view to 0 only changes the view area" )
    {
        auto lBuffer     = Internal::sGPUDevicePointer( lBufferSize * sizeof( float ) );
        auto lBufferView = Internal::sGPUDevicePointerView( 32 * sizeof( float ), 32 * sizeof( float ), lBuffer );

        std::vector<float> lData( 32 );
        std::vector<float> lZero( 32 );
        float i = 1.0f;
        for( auto &x : lData )
        {
            x = i * 2.0f;
        }
        for( auto &x : lZero )
        {
            x = 0.0f;
        }
        lBufferView.Upload( lData );

        lBufferView.Zero();
        auto lData2 = lBufferView.Fetch<float>();
        REQUIRE( lZero == lData2 );
    }
}

TEST_CASE( "Buffer pool", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t a_TotalSize = 128 * 32;

    SECTION( "Allocate memory pool creates buffer of the appropriate size" )
    {
        MemoryPool lMemoryPool( a_TotalSize );
        REQUIRE( lMemoryPool.Size() == a_TotalSize );
    }

    SECTION( "Allocated buffer from pool has the requested size" )
    {
        MemoryPool lMemoryPool( a_TotalSize );
        MemoryBuffer lBuffer = lMemoryPool.Allocate( 32 );

        REQUIRE( lBuffer.SizeAs<uint8_t>() == 32 );
        REQUIRE( lBuffer.Size() == 32 );
    }

    SECTION( "Allocate buffer larger than pool raises exception" )
    {
        MemoryPool lMemoryPool( a_TotalSize );

        try
        {
            MemoryBuffer lBuffer = lMemoryPool.Allocate( a_TotalSize + 5 );
            REQUIRE( false );
        }
        catch( ... )
        {
            REQUIRE( true );
        }
    }
}

TEST_CASE( "MultiTensors", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    MemoryPool lMemoryPool( lPoolSize );

    MultiTensor lTestTensor;

    SECTION( "Initial size is 0" ) { REQUIRE( lTestTensor.Size() == 0 ); }

    SECTION( "Multi tensors are created with the appropriate shape" )
    {
        MultiTensor lTestTensor( lMemoryPool, sTensorShape( { { 1, 2, 3 }, { 4, 5, 6 } }, sizeof( math::vec3 ) ) );

        REQUIRE( lTestTensor.Shape().CountLayers() == 2 );
    }

    SECTION( "MultiTensors are created with the appropriate byte size" )
    {
        std::vector<uint32_t> lDim1{ 3, 5, 8 };
        std::vector<uint32_t> lDim2{ 5, 3, 6 };
        MultiTensor lTestTensor( lMemoryPool, sTensorShape( { lDim1, lDim2 }, sizeof( math::vec3 ) ) );

        REQUIRE( lTestTensor.Size() >= Prod( lDim1 ) + Prod( lDim2 ) );
    }

    SECTION( "Type-specific size accessor returns the appropriate value" )
    {
        std::vector<uint32_t> lDim1{ 3, 5, 8, sizeof( float ) };
        std::vector<uint32_t> lDim2{ 5, 3, 6, sizeof( float ) };
        MultiTensor lTestTensor( lMemoryPool, sTensorShape( { lDim1, lDim2 }, sizeof( uint8_t ) ) );

        REQUIRE( lTestTensor.SizeAs<float>() == ( Prod( lDim1 ) + Prod( lDim2 ) ) / sizeof( float ) );
    }

    SECTION( "MultiTensor is created with the appropriate dimensions" )
    {
        std::vector<uint32_t> lDim1{ 2, 3, 7 };
        std::vector<uint32_t> lDim2{ 2, 11, 3 };
        std::vector<uint32_t> lDim3{ 5, 1, 2 };
        MultiTensor lTestTensor( lMemoryPool, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( uint8_t ) ) );

        REQUIRE( lTestTensor.Shape().GetShapeForLayer( 0 ) == lDim1 );
        REQUIRE( lTestTensor.Shape().GetShapeForLayer( 1 ) == lDim2 );
        REQUIRE( lTestTensor.Shape().GetShapeForLayer( 2 ) == lDim3 );

        REQUIRE( lTestTensor.Shape().GetDimension( 0 ) == std::vector<uint32_t>{ 2, 2, 5 } );
        REQUIRE( lTestTensor.Shape().GetDimension( 1 ) == std::vector<uint32_t>{ 3, 11, 1 } );
        REQUIRE( lTestTensor.Shape().GetDimension( 2 ) == std::vector<uint32_t>{ 7, 3, 2 } );

        REQUIRE( lTestTensor.Shape().GetDimension( -3 ) == std::vector<uint32_t>{ 2, 2, 5 } );
        REQUIRE( lTestTensor.Shape().GetDimension( -2 ) == std::vector<uint32_t>{ 3, 11, 1 } );
        REQUIRE( lTestTensor.Shape().GetDimension( -1 ) == std::vector<uint32_t>{ 7, 3, 2 } );
    }

    SECTION( "MultiTensor is created with the appropriate strides" )
    {
        std::vector<uint32_t> lDim1{ 2, 3, 7 };
        std::vector<uint32_t> lDim2{ 2, 11, 3 };
        std::vector<uint32_t> lDim3{ 5, 1, 2 };
        MultiTensor lTestTensor( lMemoryPool, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( uint8_t ) ) );

        std::vector<uint32_t> lStride1{ 21, 7, 1 };
        std::vector<uint32_t> lStride2{ 33, 3, 1 };
        std::vector<uint32_t> lStride3{ 2, 2, 1 };

        REQUIRE( lTestTensor.Shape().GetStridesForLayer( 0 ) == lStride1 );
        REQUIRE( lTestTensor.Shape().GetStridesForLayer( 1 ) == lStride2 );
        REQUIRE( lTestTensor.Shape().GetStridesForLayer( 2 ) == lStride3 );
    }

    SECTION( "Each layer in the MultiTensor has the correct size" )
    {
        std::vector<uint32_t> lDim1{ 2, 3, 7 };
        std::vector<uint32_t> lDim2{ 2, 11, 3 };
        std::vector<uint32_t> lDim3{ 5, 1, 2 };
        MultiTensor lTestTensor( lMemoryPool, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( math::vec3 ) ) );

        uint32_t lDim1Size = Prod( lDim1 );
        uint32_t lDim2Size = Prod( lDim2 );
        uint32_t lDim3Size = std::accumulate( lDim3.begin(), lDim3.end(), 1, std::multiplies<uint32_t>() );

        uint32_t lDim1Offset = 0;
        uint32_t lDim2Offset = lDim1Size;
        uint32_t lDim3Offset = lDim1Size + lDim2Size;

        REQUIRE( lTestTensor.Shape().GetBufferSize( 0 ) ==
                 sBufferSizeInfo{ lDim1Size * static_cast<uint32_t>( sizeof( math::vec3 ) ), lDim1Offset * static_cast<uint32_t>( sizeof( math::vec3 ) ) } );
        REQUIRE( lTestTensor.Shape().GetBufferSize( 1 ) ==
                 sBufferSizeInfo{ lDim2Size * static_cast<uint32_t>( sizeof( math::vec3 ) ), lDim2Offset * static_cast<uint32_t>( sizeof( math::vec3 ) ) } );
        REQUIRE( lTestTensor.Shape().GetBufferSize( 2 ) ==
                 sBufferSizeInfo{ lDim3Size * static_cast<uint32_t>( sizeof( math::vec3 ) ), lDim3Offset * static_cast<uint32_t>( sizeof( math::vec3 ) ) } );
    }

    SECTION( "The largest dimension is calculated correctly" )
    {
        std::vector<uint32_t> lDim1{ 2, 3, 7 };
        std::vector<uint32_t> lDim2{ 2, 11, 3 };
        std::vector<uint32_t> lDim3{ 5, 1, 2 };
        MultiTensor lTestTensor( lMemoryPool, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( math::vec3 ) ) );

        auto lMaxDimension = lTestTensor.Shape().mMaxDimensions;
        REQUIRE( lMaxDimension == std::vector<uint32_t>{ 5, 11, 7 } );
    }

    SECTION( "Flattening works as expected" )
    {
        std::vector<uint32_t> lDim1{ 2, 3, 7, 4, 8, 3 };
        std::vector<uint32_t> lDim2{ 2, 11, 3, 8, 2, 3 };
        std::vector<uint32_t> lDim3{ 5, 1, 2, 8, 7, 6 };

        {
            sTensorShape lTensorShape( { lDim1, lDim2, lDim3 }, sizeof( math::vec3 ) );
            lTensorShape.Flatten( 1 );

            std::vector<uint32_t> lExpectedDim1{ 2, 3, 7, 4, 8, 3 };
            std::vector<uint32_t> lExpectedDim2{ 2, 11, 3, 8, 2, 3 };
            std::vector<uint32_t> lExpectedDim3{ 5, 1, 2, 8, 7, 6 };

            REQUIRE( lTensorShape.mRank == lExpectedDim1.size() );
            REQUIRE( lTensorShape.GetShapeForLayer( 0 ) == lExpectedDim1 );
            REQUIRE( lTensorShape.GetShapeForLayer( 1 ) == lExpectedDim2 );
            REQUIRE( lTensorShape.GetShapeForLayer( 2 ) == lExpectedDim3 );
        }

        {
            sTensorShape lTensorShape( { lDim1, lDim2, lDim3 }, sizeof( math::vec3 ) );
            lTensorShape.Flatten( 0 );

            std::vector<uint32_t> lExpectedDim1{ 2 * 3 * 7 * 4 * 8 * 3 };
            std::vector<uint32_t> lExpectedDim2{ 2 * 11 * 3 * 8 * 2 * 3 };
            std::vector<uint32_t> lExpectedDim3{ 5 * 1 * 2 * 8 * 7 * 6 };

            REQUIRE( lTensorShape.mRank == lExpectedDim1.size() );
            REQUIRE( lTensorShape.GetShapeForLayer( 0 ) == lExpectedDim1 );
            REQUIRE( lTensorShape.GetShapeForLayer( 1 ) == lExpectedDim2 );
            REQUIRE( lTensorShape.GetShapeForLayer( 2 ) == lExpectedDim3 );
        }

        {
            sTensorShape lTensorShape( { lDim1, lDim2, lDim3 }, sizeof( math::vec3 ) );
            lTensorShape.Flatten( lTensorShape.mRank );

            std::vector<uint32_t> lExpectedDim1{ 2 * 3 * 7 * 4 * 8 * 3 };
            std::vector<uint32_t> lExpectedDim2{ 2 * 11 * 3 * 8 * 2 * 3 };
            std::vector<uint32_t> lExpectedDim3{ 5 * 1 * 2 * 8 * 7 * 6 };

            REQUIRE( lTensorShape.mRank == lExpectedDim1.size() );
            REQUIRE( lTensorShape.GetShapeForLayer( 0 ) == lExpectedDim1 );
            REQUIRE( lTensorShape.GetShapeForLayer( 1 ) == lExpectedDim2 );
            REQUIRE( lTensorShape.GetShapeForLayer( 2 ) == lExpectedDim3 );
        }

        {
            sTensorShape lTensorShape( { lDim1, lDim2, lDim3 }, sizeof( math::vec3 ) );
            lTensorShape.Flatten( 2 );

            std::vector<uint32_t> lExpectedDim1{ 2 * 3, 7, 4, 8, 3 };
            std::vector<uint32_t> lExpectedDim2{ 2 * 11, 3, 8, 2, 3 };
            std::vector<uint32_t> lExpectedDim3{ 5 * 1, 2, 8, 7, 6 };

            REQUIRE( lTensorShape.mRank == lExpectedDim1.size() );
            REQUIRE( lTensorShape.GetShapeForLayer( 0 ) == lExpectedDim1 );
            REQUIRE( lTensorShape.GetShapeForLayer( 1 ) == lExpectedDim2 );
            REQUIRE( lTensorShape.GetShapeForLayer( 2 ) == lExpectedDim3 );
        }

        {
            sTensorShape lTensorShape( { lDim1, lDim2, lDim3 }, sizeof( math::vec3 ) );
            lTensorShape.Flatten( 3 );

            std::vector<uint32_t> lExpectedDim1{ 2 * 3 * 7, 4, 8, 3 };
            std::vector<uint32_t> lExpectedDim2{ 2 * 11 * 3, 8, 2, 3 };
            std::vector<uint32_t> lExpectedDim3{ 5 * 1 * 2, 8, 7, 6 };

            REQUIRE( lTensorShape.mRank == lExpectedDim1.size() );
            REQUIRE( lTensorShape.GetShapeForLayer( 0 ) == lExpectedDim1 );
            REQUIRE( lTensorShape.GetShapeForLayer( 1 ) == lExpectedDim2 );
            REQUIRE( lTensorShape.GetShapeForLayer( 2 ) == lExpectedDim3 );
        }

        {
            sTensorShape lTensorShape( { lDim1, lDim2, lDim3 }, sizeof( math::vec3 ) );
            lTensorShape.Flatten( -1 );

            std::vector<uint32_t> lExpectedDim1{ 2 * 3 * 7 * 4 * 8, 3 };
            std::vector<uint32_t> lExpectedDim2{ 2 * 11 * 3 * 8 * 2, 3 };
            std::vector<uint32_t> lExpectedDim3{ 5 * 1 * 2 * 8 * 7, 6 };

            REQUIRE( lTensorShape.mRank == lExpectedDim1.size() );
            REQUIRE( lTensorShape.GetShapeForLayer( 0 ) == lExpectedDim1 );
            REQUIRE( lTensorShape.GetShapeForLayer( 1 ) == lExpectedDim2 );
            REQUIRE( lTensorShape.GetShapeForLayer( 2 ) == lExpectedDim3 );
        }

        {
            sTensorShape lTensorShape( { lDim1, lDim2, lDim3 }, sizeof( math::vec3 ) );
            lTensorShape.Flatten( -2 );

            std::vector<uint32_t> lExpectedDim1{ 2 * 3 * 7 * 4, 8, 3 };
            std::vector<uint32_t> lExpectedDim2{ 2 * 11 * 3 * 8, 2, 3 };
            std::vector<uint32_t> lExpectedDim3{ 5 * 1 * 2 * 8, 7, 6 };

            REQUIRE( lTensorShape.mRank == lExpectedDim1.size() );
            REQUIRE( lTensorShape.GetShapeForLayer( 0 ) == lExpectedDim1 );
            REQUIRE( lTensorShape.GetShapeForLayer( 1 ) == lExpectedDim2 );
            REQUIRE( lTensorShape.GetShapeForLayer( 2 ) == lExpectedDim3 );
        }
    }

    SECTION( "Trimming works as expected" )
    {
        std::vector<uint32_t> lDim1{ 2, 3, 7, 4, 8, 3 };
        std::vector<uint32_t> lDim2{ 2, 11, 3, 8, 2, 3 };
        std::vector<uint32_t> lDim3{ 5, 1, 2, 8, 7, 6 };

        {
            sTensorShape lTensorShape( { lDim1, lDim2, lDim3 }, sizeof( math::vec3 ) );
            lTensorShape.Trim( 1 );

            std::vector<uint32_t> lExpectedDim1{ 2 };
            std::vector<uint32_t> lExpectedDim2{ 2 };
            std::vector<uint32_t> lExpectedDim3{ 5 };

            REQUIRE( lTensorShape.mRank == lExpectedDim1.size() );
            REQUIRE( lTensorShape.GetShapeForLayer( 0 ) == lExpectedDim1 );
            REQUIRE( lTensorShape.GetShapeForLayer( 1 ) == lExpectedDim2 );
            REQUIRE( lTensorShape.GetShapeForLayer( 2 ) == lExpectedDim3 );
        }

        {
            sTensorShape lTensorShape( { lDim1, lDim2, lDim3 }, sizeof( math::vec3 ) );
            lTensorShape.Trim( 0 );

            std::vector<uint32_t> lExpectedDim1{ 2, 3, 7, 4, 8, 3 };
            std::vector<uint32_t> lExpectedDim2{ 2, 11, 3, 8, 2, 3 };
            std::vector<uint32_t> lExpectedDim3{ 5, 1, 2, 8, 7, 6 };

            REQUIRE( lTensorShape.mRank == lExpectedDim1.size() );
            REQUIRE( lTensorShape.GetShapeForLayer( 0 ) == lExpectedDim1 );
            REQUIRE( lTensorShape.GetShapeForLayer( 1 ) == lExpectedDim2 );
            REQUIRE( lTensorShape.GetShapeForLayer( 2 ) == lExpectedDim3 );
        }

        {
            sTensorShape lTensorShape( { lDim1, lDim2, lDim3 }, sizeof( math::vec3 ) );
            lTensorShape.Trim( lTensorShape.mRank );

            std::vector<uint32_t> lExpectedDim1{ 2, 3, 7, 4, 8, 3 };
            std::vector<uint32_t> lExpectedDim2{ 2, 11, 3, 8, 2, 3 };
            std::vector<uint32_t> lExpectedDim3{ 5, 1, 2, 8, 7, 6 };

            REQUIRE( lTensorShape.mRank == lExpectedDim1.size() );
            REQUIRE( lTensorShape.GetShapeForLayer( 0 ) == lExpectedDim1 );
            REQUIRE( lTensorShape.GetShapeForLayer( 1 ) == lExpectedDim2 );
            REQUIRE( lTensorShape.GetShapeForLayer( 2 ) == lExpectedDim3 );
        }

        {
            sTensorShape lTensorShape( { lDim1, lDim2, lDim3 }, sizeof( math::vec3 ) );
            lTensorShape.Trim( 2 );

            std::vector<uint32_t> lExpectedDim1{ 2, 3 };
            std::vector<uint32_t> lExpectedDim2{ 2, 11 };
            std::vector<uint32_t> lExpectedDim3{ 5, 1 };

            REQUIRE( lTensorShape.mRank == lExpectedDim1.size() );
            REQUIRE( lTensorShape.GetShapeForLayer( 0 ) == lExpectedDim1 );
            REQUIRE( lTensorShape.GetShapeForLayer( 1 ) == lExpectedDim2 );
            REQUIRE( lTensorShape.GetShapeForLayer( 2 ) == lExpectedDim3 );
        }

        {
            sTensorShape lTensorShape( { lDim1, lDim2, lDim3 }, sizeof( math::vec3 ) );
            lTensorShape.Trim( 3 );

            std::vector<uint32_t> lExpectedDim1{ 2, 3, 7 };
            std::vector<uint32_t> lExpectedDim2{ 2, 11, 3 };
            std::vector<uint32_t> lExpectedDim3{ 5, 1, 2 };

            REQUIRE( lTensorShape.mRank == lExpectedDim1.size() );
            REQUIRE( lTensorShape.GetShapeForLayer( 0 ) == lExpectedDim1 );
            REQUIRE( lTensorShape.GetShapeForLayer( 1 ) == lExpectedDim2 );
            REQUIRE( lTensorShape.GetShapeForLayer( 2 ) == lExpectedDim3 );
        }

        {
            sTensorShape lTensorShape( { lDim1, lDim2, lDim3 }, sizeof( math::vec3 ) );
            lTensorShape.Trim( -1 );

            std::vector<uint32_t> lExpectedDim1{ 2, 3, 7, 4, 8 };
            std::vector<uint32_t> lExpectedDim2{ 2, 11, 3, 8, 2 };
            std::vector<uint32_t> lExpectedDim3{ 5, 1, 2, 8, 7 };

            REQUIRE( lTensorShape.mRank == lExpectedDim1.size() );
            REQUIRE( lTensorShape.GetShapeForLayer( 0 ) == lExpectedDim1 );
            REQUIRE( lTensorShape.GetShapeForLayer( 1 ) == lExpectedDim2 );
            REQUIRE( lTensorShape.GetShapeForLayer( 2 ) == lExpectedDim3 );
        }

        {
            sTensorShape lTensorShape( { lDim1, lDim2, lDim3 }, sizeof( math::vec3 ) );
            lTensorShape.Trim( -2 );

            std::vector<uint32_t> lExpectedDim1{ 2, 3, 7, 4 };
            std::vector<uint32_t> lExpectedDim2{ 2, 11, 3, 8 };
            std::vector<uint32_t> lExpectedDim3{ 5, 1, 2, 8 };

            REQUIRE( lTensorShape.mRank == lExpectedDim1.size() );
            REQUIRE( lTensorShape.GetShapeForLayer( 0 ) == lExpectedDim1 );
            REQUIRE( lTensorShape.GetShapeForLayer( 1 ) == lExpectedDim2 );
            REQUIRE( lTensorShape.GetShapeForLayer( 2 ) == lExpectedDim3 );
        }
    }

    SECTION( "Inserting dimensions works as expected" )
    {
        std::vector<uint32_t> lDim1{ 2, 3 };
        std::vector<uint32_t> lDim2{ 2, 11 };
        std::vector<uint32_t> lDim3{ 5, 1 };

        {
            sTensorShape lTensorShape( { lDim1, lDim2, lDim3 }, sizeof( math::vec3 ) );

            lTensorShape.InsertDimension( 0, std::vector<uint32_t>{ 7, 3, 2 } );
            std::vector<uint32_t> lExpectedDim1{ 7, 2, 3 };
            std::vector<uint32_t> lExpectedDim2{ 3, 2, 11 };
            std::vector<uint32_t> lExpectedDim3{ 2, 5, 1 };

            REQUIRE( lTensorShape.GetShapeForLayer( 0 ) == lExpectedDim1 );
            REQUIRE( lTensorShape.GetShapeForLayer( 1 ) == lExpectedDim2 );
            REQUIRE( lTensorShape.GetShapeForLayer( 2 ) == lExpectedDim3 );

            REQUIRE( lTensorShape.GetDimension( 0 ) == std::vector<uint32_t>{ 7, 3, 2 } );
            REQUIRE( lTensorShape.GetDimension( 1 ) == std::vector<uint32_t>{ 2, 2, 5 } );
            REQUIRE( lTensorShape.GetDimension( 2 ) == std::vector<uint32_t>{ 3, 11, 1 } );

            std::vector<uint32_t> lExpectedStride1{ 6, 3, 1 };
            std::vector<uint32_t> lExpectedStride2{ 22, 11, 1 };
            std::vector<uint32_t> lExpectedStride3{ 5, 1, 1 };

            REQUIRE( lTensorShape.GetStridesForLayer( 0 ) == lExpectedStride1 );
            REQUIRE( lTensorShape.GetStridesForLayer( 1 ) == lExpectedStride2 );
            REQUIRE( lTensorShape.GetStridesForLayer( 2 ) == lExpectedStride3 );
        }

        {
            sTensorShape lTensorShape( { lDim1, lDim2, lDim3 }, sizeof( math::vec3 ) );

            lTensorShape.InsertDimension( 1, std::vector<uint32_t>{ 7, 3, 2 } );
            std::vector<uint32_t> lExpectedDim1{ 2, 7, 3 };
            std::vector<uint32_t> lExpectedDim2{ 2, 3, 11 };
            std::vector<uint32_t> lExpectedDim3{ 5, 2, 1 };

            REQUIRE( lTensorShape.GetShapeForLayer( 0 ) == lExpectedDim1 );
            REQUIRE( lTensorShape.GetShapeForLayer( 1 ) == lExpectedDim2 );
            REQUIRE( lTensorShape.GetShapeForLayer( 2 ) == lExpectedDim3 );

            REQUIRE( lTensorShape.GetDimension( 0 ) == std::vector<uint32_t>{ 2, 2, 5 } );
            REQUIRE( lTensorShape.GetDimension( 1 ) == std::vector<uint32_t>{ 7, 3, 2 } );
            REQUIRE( lTensorShape.GetDimension( 2 ) == std::vector<uint32_t>{ 3, 11, 1 } );

            std::vector<uint32_t> lExpectedStride1{ 21, 3, 1 };
            std::vector<uint32_t> lExpectedStride2{ 33, 11, 1 };
            std::vector<uint32_t> lExpectedStride3{ 2, 1, 1 };

            REQUIRE( lTensorShape.GetStridesForLayer( 0 ) == lExpectedStride1 );
            REQUIRE( lTensorShape.GetStridesForLayer( 1 ) == lExpectedStride2 );
            REQUIRE( lTensorShape.GetStridesForLayer( 2 ) == lExpectedStride3 );
        }

        {
            sTensorShape lTensorShape( { lDim1, lDim2, lDim3 }, sizeof( math::vec3 ) );

            lTensorShape.InsertDimension( 2, std::vector<uint32_t>{ 7, 3, 2 } );
            std::vector<uint32_t> lExpectedDim1{ 2, 3, 7 };
            std::vector<uint32_t> lExpectedDim2{ 2, 11, 3 };
            std::vector<uint32_t> lExpectedDim3{ 5, 1, 2 };

            REQUIRE( lTensorShape.GetShapeForLayer( 0 ) == lExpectedDim1 );
            REQUIRE( lTensorShape.GetShapeForLayer( 1 ) == lExpectedDim2 );
            REQUIRE( lTensorShape.GetShapeForLayer( 2 ) == lExpectedDim3 );

            REQUIRE( lTensorShape.GetDimension( 0 ) == std::vector<uint32_t>{ 2, 2, 5 } );
            REQUIRE( lTensorShape.GetDimension( 1 ) == std::vector<uint32_t>{ 3, 11, 1 } );
            REQUIRE( lTensorShape.GetDimension( 2 ) == std::vector<uint32_t>{ 7, 3, 2 } );

            std::vector<uint32_t> lExpectedStride1{ 21, 7, 1 };
            std::vector<uint32_t> lExpectedStride2{ 33, 3, 1 };
            std::vector<uint32_t> lExpectedStride3{ 2, 2, 1 };

            REQUIRE( lTensorShape.GetStridesForLayer( 0 ) == lExpectedStride1 );
            REQUIRE( lTensorShape.GetStridesForLayer( 1 ) == lExpectedStride2 );
            REQUIRE( lTensorShape.GetStridesForLayer( 2 ) == lExpectedStride3 );
        }

        {
            sTensorShape lTensorShape( { lDim1, lDim2, lDim3 }, sizeof( math::vec3 ) );

            lTensorShape.InsertDimension( -1, std::vector<uint32_t>{ 7, 3, 2 } );
            std::vector<uint32_t> lExpectedDim1{ 2, 3, 7 };
            std::vector<uint32_t> lExpectedDim2{ 2, 11, 3 };
            std::vector<uint32_t> lExpectedDim3{ 5, 1, 2 };

            REQUIRE( lTensorShape.GetShapeForLayer( 0 ) == lExpectedDim1 );
            REQUIRE( lTensorShape.GetShapeForLayer( 1 ) == lExpectedDim2 );
            REQUIRE( lTensorShape.GetShapeForLayer( 2 ) == lExpectedDim3 );

            REQUIRE( lTensorShape.GetDimension( 0 ) == std::vector<uint32_t>{ 2, 2, 5 } );
            REQUIRE( lTensorShape.GetDimension( 1 ) == std::vector<uint32_t>{ 3, 11, 1 } );
            REQUIRE( lTensorShape.GetDimension( 2 ) == std::vector<uint32_t>{ 7, 3, 2 } );

            std::vector<uint32_t> lExpectedStride1{ 21, 7, 1 };
            std::vector<uint32_t> lExpectedStride2{ 33, 3, 1 };
            std::vector<uint32_t> lExpectedStride3{ 2, 2, 1 };

            REQUIRE( lTensorShape.GetStridesForLayer( 0 ) == lExpectedStride1 );
            REQUIRE( lTensorShape.GetStridesForLayer( 1 ) == lExpectedStride2 );
            REQUIRE( lTensorShape.GetStridesForLayer( 2 ) == lExpectedStride3 );
        }

        {
            sTensorShape lTensorShape( { lDim1, lDim2, lDim3 }, sizeof( math::vec3 ) );

            lTensorShape.InsertDimension( -2, std::vector<uint32_t>{ 7, 3, 2 } );
            std::vector<uint32_t> lExpectedDim1{ 2, 7, 3 };
            std::vector<uint32_t> lExpectedDim2{ 2, 3, 11 };
            std::vector<uint32_t> lExpectedDim3{ 5, 2, 1 };

            REQUIRE( lTensorShape.GetShapeForLayer( 0 ) == lExpectedDim1 );
            REQUIRE( lTensorShape.GetShapeForLayer( 1 ) == lExpectedDim2 );
            REQUIRE( lTensorShape.GetShapeForLayer( 2 ) == lExpectedDim3 );

            REQUIRE( lTensorShape.GetDimension( 0 ) == std::vector<uint32_t>{ 2, 2, 5 } );
            REQUIRE( lTensorShape.GetDimension( 1 ) == std::vector<uint32_t>{ 7, 3, 2 } );
            REQUIRE( lTensorShape.GetDimension( 2 ) == std::vector<uint32_t>{ 3, 11, 1 } );

            std::vector<uint32_t> lExpectedStride1{ 21, 3, 1 };
            std::vector<uint32_t> lExpectedStride2{ 33, 11, 1 };
            std::vector<uint32_t> lExpectedStride3{ 2, 1, 1 };

            REQUIRE( lTensorShape.GetStridesForLayer( 0 ) == lExpectedStride1 );
            REQUIRE( lTensorShape.GetStridesForLayer( 1 ) == lExpectedStride2 );
            REQUIRE( lTensorShape.GetStridesForLayer( 2 ) == lExpectedStride3 );
        }

        {
            sTensorShape lTensorShape( { lDim1, lDim2, lDim3 }, sizeof( math::vec3 ) );

            lTensorShape.InsertDimension( -3, std::vector<uint32_t>{ 7, 3, 2 } );
            std::vector<uint32_t> lExpectedDim1{ 7, 2, 3 };
            std::vector<uint32_t> lExpectedDim2{ 3, 2, 11 };
            std::vector<uint32_t> lExpectedDim3{ 2, 5, 1 };

            REQUIRE( lTensorShape.GetShapeForLayer( 0 ) == lExpectedDim1 );
            REQUIRE( lTensorShape.GetShapeForLayer( 1 ) == lExpectedDim2 );
            REQUIRE( lTensorShape.GetShapeForLayer( 2 ) == lExpectedDim3 );

            REQUIRE( lTensorShape.GetDimension( 0 ) == std::vector<uint32_t>{ 7, 3, 2 } );
            REQUIRE( lTensorShape.GetDimension( 1 ) == std::vector<uint32_t>{ 2, 2, 5 } );
            REQUIRE( lTensorShape.GetDimension( 2 ) == std::vector<uint32_t>{ 3, 11, 1 } );

            std::vector<uint32_t> lExpectedStride1{ 6, 3, 1 };
            std::vector<uint32_t> lExpectedStride2{ 22, 11, 1 };
            std::vector<uint32_t> lExpectedStride3{ 5, 1, 1 };

            REQUIRE( lTensorShape.GetStridesForLayer( 0 ) == lExpectedStride1 );
            REQUIRE( lTensorShape.GetStridesForLayer( 1 ) == lExpectedStride2 );
            REQUIRE( lTensorShape.GetStridesForLayer( 2 ) == lExpectedStride3 );
        }
    }

    SECTION( "View into MultiTensor layers have the appropriate size" )
    {
        std::vector<uint32_t> lDim1{ 2, 3, 7 };
        std::vector<uint32_t> lDim2{ 5, 11, 13 };
        std::vector<uint32_t> lDim3{ 5, 3, 2 };
        MultiTensor lTestTensor( lMemoryPool, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( math::vec3 ) ) );

        auto lBuffer1 = lTestTensor.BufferAt( 1 );
        auto lBuffer2 = lTestTensor.BufferAt( 2 );
        REQUIRE( lBuffer1.Size() == Prod( lDim2 ) * sizeof( math::vec3 ) );
        REQUIRE( lBuffer2.Size() == Prod( lDim3 ) * sizeof( math::vec3 ) );
    }

    SECTION( "Fetch stack layer" )
    {
        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };
        MultiTensor lTestTensor2( lMemoryPool, sTensorShape( { lDim1, lDim2 }, sizeof( int32_t ) ) );

        std::vector<int32_t> lValues( lTestTensor2.SizeAs<int32_t>() );
        int32_t i = 0;
        for( auto &v : lValues )
        {
            v = i++;
        }
        lTestTensor2.Upload( lValues );

        std::vector<int32_t> lExpectedValues1( Prod( lDim1 ) );
        i = 0;
        for( auto &v : lExpectedValues1 )
        {
            v = i++;
        }
        std::vector<int32_t> lB1 = lTestTensor2.FetchBufferAt<int32_t>( 0 );
        REQUIRE( lB1.size() == Prod( lDim1 ) );
        REQUIRE( lB1 == lExpectedValues1 );

        std::vector<int32_t> lExpectedValues2( Prod( lDim2 ) );
        for( auto &v : lExpectedValues2 )
        {
            v = i++;
        }
        std::vector<int32_t> lB2 = lTestTensor2.FetchBufferAt<int32_t>( 1 );
        REQUIRE( lB2.size() == Prod( lDim2 ) );
        REQUIRE( lB2 == lExpectedValues2 );
    }

    SECTION( "Upload arrays to layers" )
    {
        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };
        MultiTensor lTestTensor2( lMemoryPool, sTensorShape( { lDim1, lDim2 }, sizeof( int32_t ) ) );

        std::vector<int32_t> lValues0( Prod( lDim1 ) );
        std::vector<int32_t> lValues1( Prod( lDim2 ) );
        int32_t i = 0;
        for( auto &v : lValues0 )
        {
            v = i++;
        }
        for( auto &v : lValues1 )
        {
            v = i++;
        }
        lTestTensor2.Upload( lValues0, 0 );
        lTestTensor2.Upload( lValues1, 1 );

        std::vector<int32_t> lB1 = lTestTensor2.FetchBufferAt<int32_t>( 0 );
        std::vector<int32_t> lB2 = lTestTensor2.FetchBufferAt<int32_t>( 1 );
        REQUIRE( lB1 == lValues0 );
        REQUIRE( lB2 == lValues1 );
    }
}
