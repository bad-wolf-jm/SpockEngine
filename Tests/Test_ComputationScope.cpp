#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "TestUtils.h"

#include "Core/Math/Types.h"

#include "Core/Cuda/MemoryPool.h"
#include "Core/Cuda/MultiTensor.h"

#include "TensorOps/NodeComponents.h"
#include "TensorOps/Scope.h"

using namespace LTSE::Core;
using namespace LTSE::TensorOps;
using namespace TestUtils;

std::vector<uint8_t> RandomBooleanValues( std::vector<uint32_t> aDim )
{
    uint32_t             lSize = std::accumulate( aDim.begin(), aDim.end(), 1, std::multiplies<uint32_t>() );
    std::vector<uint8_t> lResult{};

    for( uint32_t j = 0; j < lSize; j++ )
    {
        auto lY = RandomBool();
        lResult.push_back( lY );
    }

    return lResult;
}

template <typename _Ty>
std::vector<_Ty> RandomValues( std::vector<uint32_t> aDim, _Ty aMin, _Ty aMax )
{
    uint32_t         lSize = std::accumulate( aDim.begin(), aDim.end(), 1, std::multiplies<uint32_t>() );
    std::vector<_Ty> lResult{};

    for( uint32_t j = 0; j < lSize; j++ )
    {
        auto lY = RandomNumber<_Ty>( aMin, aMax );
        lResult.push_back( lY );
    }

    return lResult;
}

std::vector<std::vector<uint8_t>> RandomBooleanVector( std::vector<uint32_t> aDim )
{
    uint32_t                          lSize   = std::accumulate( aDim.begin(), aDim.end() - 1, 1, std::multiplies<uint32_t>() );
    uint32_t                          lLength = aDim.back();
    std::vector<std::vector<uint8_t>> lResult{};

    for( uint32_t j = 0; j < lSize; j++ )
    {
        auto lY = RandomBool( lLength );
        lResult.push_back( lY );
    }

    return lResult;
}

template <typename _Ty>
std::vector<std::vector<_Ty>> RandomVector( std::vector<uint32_t> aDim, _Ty aMin, _Ty aMax )
{
    uint32_t                      lSize   = std::accumulate( aDim.begin(), aDim.end() - 1, 1, std::multiplies<uint32_t>() );
    uint32_t                      lLength = aDim.back();
    std::vector<std::vector<_Ty>> lResult{};

    for( uint32_t j = 0; j < lSize; j++ )
    {
        auto lY = RandomNumber<_Ty>( lLength, aMin, aMax );
        lResult.push_back( lY );
    }

    return lResult;
}

template <typename _Ty>
std::vector<_Ty> BroadcastMap( std::vector<_Ty> aVec1, _Ty aValue, std::function<_Ty( _Ty, _Ty )> aFunction )
{
    std::vector<_Ty> lResult{};

    for( uint32_t i = 0; i < aVec1.size(); i++ ) lResult.push_back( aFunction( aVec1[i], aValue ) );

    return lResult;
}

template <typename _Ty>
std::vector<_Ty> BroadcastMap( _Ty aValue, std::vector<_Ty> aVec1, std::function<_Ty( _Ty, _Ty )> aFunction )
{
    std::vector<_Ty> lResult{};

    for( uint32_t i = 0; i < aVec1.size(); i++ ) lResult.push_back( aFunction( aValue, aVec1[i] ) );

    return lResult;
}

template <typename _Ty>
std::vector<std::vector<_Ty>> BroadcastMap(
    std::vector<std::vector<_Ty>> aVec1, std::vector<_Ty> aVec2, std::function<_Ty( _Ty, _Ty )> aFunction )
{
    std::vector<std::vector<_Ty>> lResult{};

    for( uint32_t i = 0; i < aVec1.size(); i++ ) lResult.push_back( BroadcastMap( aVec1[i], aVec2[i], aFunction ) );

    return lResult;
}

template <typename _Ty>
std::vector<std::vector<_Ty>> BroadcastMap( std::vector<std::vector<_Ty>> aVec1, _Ty aVec2, std::function<_Ty( _Ty, _Ty )> aFunction )
{
    std::vector<std::vector<_Ty>> lResult{};

    for( uint32_t i = 0; i < aVec1.size(); i++ ) lResult.push_back( BroadcastMap( aVec1[i], aVec2, aFunction ) );

    return lResult;
}

template <typename _Ty>
std::vector<std::vector<_Ty>> BroadcastMap(
    std::vector<_Ty> aVec1, std::vector<std::vector<_Ty>> aVec2, std::function<_Ty( _Ty, _Ty )> aFunction )
{
    std::vector<std::vector<_Ty>> lResult{};

    for( uint32_t i = 0; i < aVec1.size(); i++ ) lResult.push_back( BroadcastMap( aVec1[i], aVec2[i], aFunction ) );

    return lResult;
}

TEST_CASE( "VectorNode", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    SECTION( "Node allocation" )
    {
        std::vector<uint32_t> lValue = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

        auto lNode = VectorValue<uint32_t>( lScope, lValue );
        lScope.Run( lNode );

        REQUIRE( lNode.Get<sVectorComponent<uint32_t>>().mData.SizeAs<uint32_t>() == lValue.size() );
    }

    SECTION( "Node initialization" )
    {
        std::vector<uint32_t> lValue = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

        auto lNode = VectorValue<uint32_t>( lScope, lValue );
        lScope.Run( lNode );

        auto lBuffer2 = lNode.Get<sVectorComponent<uint32_t>>().mData.Fetch<uint32_t>();
        REQUIRE( lBuffer2 == lValue );
    }
}

TEST_CASE( "TensorNode", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    SECTION( "Node allocation" )
    {
        sConstantValueInitializerComponent lInitializer{};
        lInitializer.mValue = (uint8_t)3;

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lNode    = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( int8_t ) ) );
        auto lBuffer1 = lNode.Get<sMultiTensorComponent>().mValue.BufferAt( 0 );
        auto lBuffer2 = lNode.Get<sMultiTensorComponent>().mValue.BufferAt( 1 );
        REQUIRE( lBuffer1.Size() == Prod( lDim1 ) );
        REQUIRE( lBuffer2.Size() == Prod( lDim2 ) );
    }

    SECTION( "Constant initializer (float)" )
    {
        sConstantValueInitializerComponent lInitializer{};
        lInitializer.mValue = 3.0f;

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
        lScope.Run( lNode );

        std::vector<float> lExpectedValues( lNode.Get<sMultiTensorComponent>().mValue.SizeAs<float>() );
        std::vector<float> lTensorValues = lNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        for( auto &v : lExpectedValues )
        {
            v = 3.0f;
        }
        REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
    }

    SECTION( "Constant initializer (double)" )
    {
        sConstantValueInitializerComponent lInitializer{};
        lInitializer.mValue = (double)3.0f;

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( double ) ) );
        lScope.Run( lNode );

        std::vector<double> lExpectedValues( lNode.Get<sMultiTensorComponent>().mValue.SizeAs<double>() );
        std::vector<double> lTensorValues = lNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<double>();
        for( auto &v : lExpectedValues )
        {
            v = 3.0f;
        }
        REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
    }

    SECTION( "Constant initializer (uint8_t)" )
    {
        sConstantValueInitializerComponent lInitializer{};
        lInitializer.mValue = (uint8_t)3;

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( uint8_t ) ) );
        lScope.Run( lNode );

        std::vector<uint8_t> lExpectedValues( lNode.Get<sMultiTensorComponent>().mValue.SizeAs<uint8_t>() );
        std::vector<uint8_t> lTensorValues = lNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
        for( auto &v : lExpectedValues )
        {
            v = 3;
        }
        REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
    }

    SECTION( "Constant initializer (uint16_t)" )
    {
        sConstantValueInitializerComponent lInitializer{};
        lInitializer.mValue = (uint16_t)256;

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( uint16_t ) ) );
        lScope.Run( lNode );

        std::vector<uint16_t> lExpectedValues( lNode.Get<sMultiTensorComponent>().mValue.SizeAs<uint16_t>() );
        std::vector<uint16_t> lTensorValues = lNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint16_t>();
        for( auto &v : lExpectedValues )
        {
            v = 256;
        }
        REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
    }

    SECTION( "Constant initializer (uint32_t)" )
    {
        sConstantValueInitializerComponent lInitializer{};
        lInitializer.mValue = (uint32_t)1000000;

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( uint32_t ) ) );
        lScope.Run( lNode );

        std::vector<uint32_t> lExpectedValues( lNode.Get<sMultiTensorComponent>().mValue.SizeAs<uint32_t>() );
        std::vector<uint32_t> lTensorValues = lNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint32_t>();
        for( auto &v : lExpectedValues )
        {
            v = 1000000;
        }
        REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
    }

    SECTION( "Constant initializer (uint64_t)" )
    {
        sConstantValueInitializerComponent lInitializer{};
        lInitializer.mValue = (uint64_t)10000000000;

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( uint64_t ) ) );
        lScope.Run( lNode );

        std::vector<uint64_t> lExpectedValues( lNode.Get<sMultiTensorComponent>().mValue.SizeAs<uint64_t>() );
        std::vector<uint64_t> lTensorValues = lNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint64_t>();
        for( auto &v : lExpectedValues )
        {
            v = 10000000000;
        }
        REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
    }

    SECTION( "Vector initializer (float)" )
    {
        sVectorInitializerComponent lInitializer( std::vector<float>{ 4.0f, 5.0f } );

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
        lScope.Run( lNode );

        std::vector<float> lExpectedValues0( Prod( lDim1 ) );
        std::vector<float> lExpectedValues1( Prod( lDim2 ) );
        std::fill( lExpectedValues0.begin(), lExpectedValues0.end(), 4.0f );
        std::fill( lExpectedValues1.begin(), lExpectedValues1.end(), 5.0f );
        lExpectedValues0.insert( lExpectedValues0.end(), lExpectedValues1.begin(), lExpectedValues1.end() );

        std::vector<float> lTensorValues = lNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        REQUIRE( VectorEqual( lTensorValues, lExpectedValues0 ) );
    }

    SECTION( "Vector initializer (double)" )
    {
        sVectorInitializerComponent lInitializer( std::vector<double>{ 4.0, 5.0 } );

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( double ) ) );
        lScope.Run( lNode );

        std::vector<double> lExpectedValues0( Prod( lDim1 ) );
        std::vector<double> lExpectedValues1( Prod( lDim2 ) );
        std::fill( lExpectedValues0.begin(), lExpectedValues0.end(), (double)4.0 );
        std::fill( lExpectedValues1.begin(), lExpectedValues1.end(), (double)5.0 );
        lExpectedValues0.insert( lExpectedValues0.end(), lExpectedValues1.begin(), lExpectedValues1.end() );
        std::vector<double> lTensorValues = lNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<double>();

        REQUIRE( VectorEqual( lTensorValues, lExpectedValues0 ) );
    }

    SECTION( "Vector initializer (uint8_t)" )
    {
        sVectorInitializerComponent lInitializer( std::vector<uint8_t>{ 4, 5 } );

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( uint8_t ) ) );
        lScope.Run( lNode );

        std::vector<uint8_t> lExpectedValues0( Prod( lDim1 ) );
        std::vector<uint8_t> lExpectedValues1( Prod( lDim2 ) );
        std::fill( lExpectedValues0.begin(), lExpectedValues0.end(), (uint8_t)4 );
        std::fill( lExpectedValues1.begin(), lExpectedValues1.end(), (uint8_t)5 );
        lExpectedValues0.insert( lExpectedValues0.end(), lExpectedValues1.begin(), lExpectedValues1.end() );

        std::vector<uint8_t> lTensorValues = lNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
        REQUIRE( VectorEqual( lTensorValues, lExpectedValues0 ) );
    }

    SECTION( "Vector initializer (uint16_t)" )
    {
        sVectorInitializerComponent lInitializer( std::vector<uint16_t>{ 256, 512 } );

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( uint16_t ) ) );
        lScope.Run( lNode );

        std::vector<uint16_t> lExpectedValues0( Prod( lDim1 ) );
        std::vector<uint16_t> lExpectedValues1( Prod( lDim2 ) );
        std::fill( lExpectedValues0.begin(), lExpectedValues0.end(), (uint16_t)256 );
        std::fill( lExpectedValues1.begin(), lExpectedValues1.end(), (uint16_t)512 );
        lExpectedValues0.insert( lExpectedValues0.end(), lExpectedValues1.begin(), lExpectedValues1.end() );
        std::vector<uint16_t> lTensorValues = lNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint16_t>();
        REQUIRE( VectorEqual( lTensorValues, lExpectedValues0 ) );
    }

    SECTION( "Vector initializer (uint32_t)" )
    {
        sVectorInitializerComponent lInitializer( std::vector<uint32_t>{ 1234567, 7654321 } );

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( uint32_t ) ) );
        lScope.Run( lNode );

        std::vector<uint32_t> lExpectedValues0( Prod( lDim1 ) );
        std::vector<uint32_t> lExpectedValues1( Prod( lDim2 ) );
        std::fill( lExpectedValues0.begin(), lExpectedValues0.end(), (uint32_t)1234567 );
        std::fill( lExpectedValues1.begin(), lExpectedValues1.end(), (uint32_t)7654321 );
        lExpectedValues0.insert( lExpectedValues0.end(), lExpectedValues1.begin(), lExpectedValues1.end() );
        std::vector<uint32_t> lTensorValues = lNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint32_t>();
        REQUIRE( VectorEqual( lTensorValues, lExpectedValues0 ) );
    }

    SECTION( "Vector initializer (uint64_t)" )
    {
        sVectorInitializerComponent lInitializer( std::vector<uint64_t>{ 1234567890, 987654321 } );

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( uint64_t ) ) );
        lScope.Run( lNode );

        std::vector<uint64_t> lExpectedValues0( Prod( lDim1 ) );
        std::vector<uint64_t> lExpectedValues1( Prod( lDim2 ) );
        std::fill( lExpectedValues0.begin(), lExpectedValues0.end(), (uint64_t)1234567890 );
        std::fill( lExpectedValues1.begin(), lExpectedValues1.end(), (uint64_t)987654321 );
        lExpectedValues0.insert( lExpectedValues0.end(), lExpectedValues1.begin(), lExpectedValues1.end() );
        std::vector<uint64_t> lTensorValues = lNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint64_t>();
        REQUIRE( VectorEqual( lTensorValues, lExpectedValues0 ) );
    }

    SECTION( "Data initializer (float)" )
    {
        std::vector<float> lExpectedValues{
            4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f };
        sDataInitializerComponent lInitializer( lExpectedValues );

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
        lScope.Run( lNode );

        std::vector<float> lTensorValues = lNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        REQUIRE( VectorEqual( lExpectedValues, lTensorValues ) );
    }

    SECTION( "Data initializer (double)" )
    {
        std::vector<double> lExpectedValues{
            4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0 };
        sDataInitializerComponent lInitializer( lExpectedValues );

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( double ) ) );
        lScope.Run( lNode );

        std::vector<double> lTensorValues = lNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<double>();
        REQUIRE( VectorEqual( lExpectedValues, lTensorValues ) );
    }

    SECTION( "Data initializer (uint8_t)" )
    {
        std::vector<uint8_t>      lExpectedValues{ 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
        sDataInitializerComponent lInitializer( lExpectedValues );

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( uint8_t ) ) );
        lScope.Run( lNode );

        std::vector<uint8_t> lTensorValues = lNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
        REQUIRE( VectorEqual( lExpectedValues, lTensorValues ) );
    }

    SECTION( "Data initializer (uint16_t)" )
    {
        std::vector<uint16_t>     lExpectedValues{ 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
        sDataInitializerComponent lInitializer( lExpectedValues );

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( uint16_t ) ) );
        lScope.Run( lNode );

        std::vector<uint16_t> lTensorValues = lNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint16_t>();
        REQUIRE( VectorEqual( lExpectedValues, lTensorValues ) );
    }

    SECTION( "Data initializer (uint32_t)" )
    {
        std::vector<uint32_t>     lExpectedValues{ 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
        sDataInitializerComponent lInitializer( lExpectedValues );

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( uint32_t ) ) );
        lScope.Run( lNode );

        std::vector<uint32_t> lTensorValues = lNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint32_t>();
        REQUIRE( VectorEqual( lExpectedValues, lTensorValues ) );
    }

    SECTION( "Data initializer (uint64_t)" )
    {
        std::vector<uint64_t>     lExpectedValues{ 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
        sDataInitializerComponent lInitializer( lExpectedValues );

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( uint64_t ) ) );
        lScope.Run( lNode );

        std::vector<uint64_t> lTensorValues = lNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint64_t>();
        REQUIRE( VectorEqual( lExpectedValues, lTensorValues ) );
    }

    SECTION( "Random uniform initializer (float)" )
    {
        sRandomUniformInitializerComponent lInitializer{};
        lInitializer.mType = eScalarType::FLOAT32;

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
        lScope.Run( lNode );

        std::vector<float> lExpectedValues( lNode.Get<sMultiTensorComponent>().mValue.SizeAs<float>() );
        std::vector<float> lTensorValues = lNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        for( auto &v : lExpectedValues )
        {
            v = 0.0f;
        }
        REQUIRE( lTensorValues != lExpectedValues );
    }

    SECTION( "Random uniform initializer (double)" )
    {
        sRandomUniformInitializerComponent lInitializer{};
        lInitializer.mType = eScalarType::FLOAT64;

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( double ) ) );
        lScope.Run( lNode );

        std::vector<double> lExpectedValues( lNode.Get<sMultiTensorComponent>().mValue.SizeAs<double>() );
        std::vector<double> lTensorValues = lNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<double>();
        for( auto &v : lExpectedValues )
        {
            v = 0.0;
        }
        REQUIRE( lTensorValues != lExpectedValues );
    }

    SECTION( "Random normal initializer (float)" )
    {
        sRandomNormalInitializerComponent lInitializer{};
        lInitializer.mType = eScalarType::FLOAT32;

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
        lScope.Run( lNode );

        std::vector<float> lExpectedValues( lNode.Get<sMultiTensorComponent>().mValue.SizeAs<float>() );
        std::vector<float> lTensorValues = lNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        for( auto &v : lExpectedValues )
        {
            v = 0.0f;
        }
        REQUIRE( lTensorValues != lExpectedValues );
    }

    SECTION( "Random normal initializer (double)" )
    {
        sRandomNormalInitializerComponent lInitializer{};
        lInitializer.mType = eScalarType::FLOAT64;
        lInitializer.mMean = (double)0.0;
        lInitializer.mStd  = (double)1.0;

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( double ) ) );
        lScope.Run( lNode );

        std::vector<double> lExpectedValues( lNode.Get<sMultiTensorComponent>().mValue.SizeAs<double>() );
        std::vector<double> lTensorValues = lNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<double>();
        for( auto &v : lExpectedValues )
        {
            v = 0.0;
        }
        REQUIRE( lTensorValues != lExpectedValues );
    }
}

TEST_CASE( "Arithmetic nodes", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    SECTION( "Add scalar to array (float)" )
    {
        sRandomNormalInitializerComponent lInitializer{};
        lInitializer.mType = eScalarType::FLOAT32;

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lOpNode  = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
        auto lOpSNode = ConstantScalarValue( lScope, 1.234f );

        auto lResult0 = Add( lScope, lOpNode, lOpSNode );
        auto lResult1 = Add( lScope, lOpSNode, lOpNode );

        lScope.Run( { lResult0, lResult1 } );

        std::vector<float> lLeftTensorValues = lOpNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        std::vector<float> lExpectedValues( lOpNode.Get<sMultiTensorComponent>().mValue.SizeAs<float>() );
        std::vector<float> lTensorValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        std::vector<float> lTensorValues1 = lResult1.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        for( uint32_t i = 0; i < lLeftTensorValues.size(); i++ )
        {
            lExpectedValues[i] = lLeftTensorValues[i] + 1.234f;
        }
        REQUIRE( VectorEqual( lTensorValues0, lExpectedValues ) );
        REQUIRE( VectorEqual( lTensorValues1, lExpectedValues ) );
    }

    SECTION( "Add array to array (float)" )
    {
        sRandomNormalInitializerComponent lInitializer{};
        lInitializer.mType = eScalarType::FLOAT32;

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lOpNode  = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
        auto lOpSNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
        auto lResult0 = Add( lScope, lOpNode, lOpSNode );

        lScope.Run( lResult0 );

        std::vector<float> lLeftTensorValues  = lOpNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        std::vector<float> lRightTensorValues = lOpSNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        std::vector<float> lExpectedValues( lOpNode.Get<sMultiTensorComponent>().mValue.SizeAs<float>() );
        std::vector<float> lTensorValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        for( uint32_t i = 0; i < lLeftTensorValues.size(); i++ )
        {
            lExpectedValues[i] = lLeftTensorValues[i] + lRightTensorValues[i];
        }
        REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
    }

    SECTION( "Add array to vector (float)" )
    {
        sRandomNormalInitializerComponent lInitializer{};
        lInitializer.mType = eScalarType::FLOAT32;

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto                     lOpNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
        std::vector<ScalarValue> lConstants{ 3.123f, 4.345f };
        auto                     lOpSNode = VectorValue( lScope, lConstants );
        auto                     lResult0 = Add( lScope, lOpNode, lOpSNode );
        auto                     lResult1 = Add( lScope, lOpSNode, lOpNode );
        lScope.Run( { lResult0, lResult1 } );

        {
            std::vector<float> lLeftTensorValues  = lOpNode.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
            float              lRightTensorValues = std::get<float>( lConstants[0] );
            std::vector<float> lExpectedValues( lLeftTensorValues.size() );
            std::vector<float> lTensorValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
            for( uint32_t i = 0; i < lLeftTensorValues.size(); i++ )
            {
                lExpectedValues[i] = lLeftTensorValues[i] + lRightTensorValues;
            }
            REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
        }

        {
            std::vector<float> lLeftTensorValues  = lOpNode.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
            float              lRightTensorValues = std::get<float>( lConstants[1] );
            std::vector<float> lExpectedValues( lLeftTensorValues.size() );
            std::vector<float> lTensorValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
            for( uint32_t i = 0; i < lLeftTensorValues.size(); i++ )
            {
                lExpectedValues[i] = lLeftTensorValues[i] + lRightTensorValues;
            }
            REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
        }

        {
            std::vector<float> lLeftTensorValues  = lOpNode.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
            float              lRightTensorValues = std::get<float>( lConstants[0] );
            std::vector<float> lExpectedValues( lLeftTensorValues.size() );
            std::vector<float> lTensorValues = lResult1.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
            for( uint32_t i = 0; i < lLeftTensorValues.size(); i++ )
            {
                lExpectedValues[i] = lLeftTensorValues[i] + lRightTensorValues;
            }
            REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
        }

        {
            std::vector<float> lLeftTensorValues  = lOpNode.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
            float              lRightTensorValues = std::get<float>( lConstants[1] );
            std::vector<float> lExpectedValues( lLeftTensorValues.size() );
            std::vector<float> lTensorValues = lResult1.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
            for( uint32_t i = 0; i < lLeftTensorValues.size(); i++ )
            {
                lExpectedValues[i] = lLeftTensorValues[i] + lRightTensorValues;
            }
            REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
        }
    }

    SECTION( "Multiply scalar by array (float)" )
    {
        sRandomNormalInitializerComponent lInitializer{};
        lInitializer.mType = eScalarType::FLOAT32;

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lOpNode  = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
        auto lOpSNode = ConstantScalarValue( lScope, 1.234f );
        auto lResult0 = Multiply( lScope, lOpNode, lOpSNode );
        auto lResult1 = Multiply( lScope, lOpSNode, lOpNode );
        lScope.Run( { lResult0, lResult1 } );

        std::vector<float> lLeftTensorValues = lOpNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        std::vector<float> lExpectedValues( lOpNode.Get<sMultiTensorComponent>().mValue.SizeAs<float>() );
        std::vector<float> lTensorValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        std::vector<float> lTensorValues1 = lResult1.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        for( uint32_t i = 0; i < lLeftTensorValues.size(); i++ )
        {
            lExpectedValues[i] = lLeftTensorValues[i] * 1.234f;
        }
        REQUIRE( VectorEqual( lTensorValues0, lExpectedValues ) );
        REQUIRE( VectorEqual( lTensorValues1, lExpectedValues ) );
    }

    SECTION( "Multiply array by array (float)" )
    {
        sRandomNormalInitializerComponent lInitializer{};
        lInitializer.mType = eScalarType::FLOAT32;

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lOpNode  = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
        auto lOpSNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
        auto lResult0 = Multiply( lScope, lOpNode, lOpSNode );
        lScope.Run( lResult0 );

        std::vector<float> lLeftTensorValues  = lOpNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        std::vector<float> lRightTensorValues = lOpSNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        std::vector<float> lExpectedValues( lOpNode.Get<sMultiTensorComponent>().mValue.SizeAs<float>() );
        std::vector<float> lTensorValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        for( uint32_t i = 0; i < lLeftTensorValues.size(); i++ )
        {
            lExpectedValues[i] = lLeftTensorValues[i] * lRightTensorValues[i];
        }
        REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
    }

    SECTION( "Multiply array by vector (float)" )
    {
        sRandomNormalInitializerComponent lInitializer{};
        lInitializer.mType = eScalarType::FLOAT32;

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto                     lOpNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
        std::vector<ScalarValue> lConstants{ 3.123f, 4.345f };
        auto                     lOpSNode = VectorValue( lScope, lConstants );
        auto                     lResult0 = Multiply( lScope, lOpNode, lOpSNode );
        auto                     lResult1 = Multiply( lScope, lOpSNode, lOpNode );
        lScope.Run( { lResult0, lResult1 } );

        {
            std::vector<float> lLeftTensorValues  = lOpNode.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
            float              lRightTensorValues = std::get<float>( lConstants[0] );
            std::vector<float> lExpectedValues( lLeftTensorValues.size() );
            std::vector<float> lTensorValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
            for( uint32_t i = 0; i < lLeftTensorValues.size(); i++ )
            {
                lExpectedValues[i] = lLeftTensorValues[i] * lRightTensorValues;
            }
            REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
        }

        {
            std::vector<float> lLeftTensorValues  = lOpNode.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
            float              lRightTensorValues = std::get<float>( lConstants[1] );
            std::vector<float> lExpectedValues( lLeftTensorValues.size() );
            std::vector<float> lTensorValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
            for( uint32_t i = 0; i < lLeftTensorValues.size(); i++ )
            {
                lExpectedValues[i] = lLeftTensorValues[i] * lRightTensorValues;
            }
            REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
        }

        {
            std::vector<float> lLeftTensorValues  = lOpNode.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
            float              lRightTensorValues = std::get<float>( lConstants[0] );
            std::vector<float> lExpectedValues( lLeftTensorValues.size() );
            std::vector<float> lTensorValues = lResult1.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
            for( uint32_t i = 0; i < lLeftTensorValues.size(); i++ )
            {
                lExpectedValues[i] = lLeftTensorValues[i] * lRightTensorValues;
            }
            REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
        }

        {
            std::vector<float> lLeftTensorValues  = lOpNode.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
            float              lRightTensorValues = std::get<float>( lConstants[1] );
            std::vector<float> lExpectedValues( lLeftTensorValues.size() );
            std::vector<float> lTensorValues = lResult1.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
            for( uint32_t i = 0; i < lLeftTensorValues.size(); i++ )
            {
                lExpectedValues[i] = lLeftTensorValues[i] * lRightTensorValues;
            }
            REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
        }
    }

    SECTION( "Subtract scalar from array (float)" )
    {
        sRandomNormalInitializerComponent lInitializer{};
        lInitializer.mType = eScalarType::FLOAT32;

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lOpNode  = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
        auto lOpSNode = ConstantScalarValue( lScope, 1.234f );
        auto lResult0 = Subtract( lScope, lOpNode, lOpSNode );
        auto lResult1 = Subtract( lScope, lOpSNode, lOpNode );
        lScope.Run( { lResult0, lResult1 } );

        std::vector<float> lLeftTensorValues = lOpNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        std::vector<float> lExpectedValues0( lOpNode.Get<sMultiTensorComponent>().mValue.SizeAs<float>() );
        std::vector<float> lExpectedValues1( lOpNode.Get<sMultiTensorComponent>().mValue.SizeAs<float>() );
        std::vector<float> lTensorValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        std::vector<float> lTensorValues1 = lResult1.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        for( uint32_t i = 0; i < lLeftTensorValues.size(); i++ )
        {
            lExpectedValues0[i] = lLeftTensorValues[i] - 1.234f;
            lExpectedValues1[i] = 1.234f - lLeftTensorValues[i];
        }
        REQUIRE( VectorEqual( lTensorValues0, lExpectedValues0 ) );
        REQUIRE( VectorEqual( lTensorValues1, lExpectedValues1 ) );
    }

    SECTION( "Subtract vector from array (float)" )
    {
        sRandomNormalInitializerComponent lInitializer{};
        lInitializer.mType = eScalarType::FLOAT32;

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto                     lOpNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
        std::vector<ScalarValue> lConstants{ 3.123f, 4.345f };
        auto                     lOpSNode = VectorValue( lScope, lConstants );
        auto                     lResult0 = Subtract( lScope, lOpNode, lOpSNode );
        auto                     lResult1 = Subtract( lScope, lOpSNode, lOpNode );
        lScope.Run( { lResult0, lResult1 } );

        {
            std::vector<float> lLeftTensorValues  = lOpNode.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
            float              lRightTensorValues = std::get<float>( lConstants[0] );
            std::vector<float> lExpectedValues( lLeftTensorValues.size() );
            std::vector<float> lTensorValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
            for( uint32_t i = 0; i < lLeftTensorValues.size(); i++ )
            {
                lExpectedValues[i] = lLeftTensorValues[i] - lRightTensorValues;
            }
            REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
        }

        {
            std::vector<float> lLeftTensorValues  = lOpNode.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
            float              lRightTensorValues = std::get<float>( lConstants[1] );
            std::vector<float> lExpectedValues( lLeftTensorValues.size() );
            std::vector<float> lTensorValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
            for( uint32_t i = 0; i < lLeftTensorValues.size(); i++ )
            {
                lExpectedValues[i] = lLeftTensorValues[i] - lRightTensorValues;
            }
            REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
        }

        {
            std::vector<float> lLeftTensorValues  = lOpNode.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
            float              lRightTensorValues = std::get<float>( lConstants[0] );
            std::vector<float> lExpectedValues( lLeftTensorValues.size() );
            std::vector<float> lTensorValues = lResult1.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
            for( uint32_t i = 0; i < lLeftTensorValues.size(); i++ )
            {
                lExpectedValues[i] = lRightTensorValues - lLeftTensorValues[i];
            }
            REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
        }

        {
            std::vector<float> lLeftTensorValues  = lOpNode.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
            float              lRightTensorValues = std::get<float>( lConstants[1] );
            std::vector<float> lExpectedValues( lLeftTensorValues.size() );
            std::vector<float> lTensorValues = lResult1.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
            for( uint32_t i = 0; i < lLeftTensorValues.size(); i++ )
            {
                lExpectedValues[i] = lRightTensorValues - lLeftTensorValues[i];
            }
            REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
        }
    }

    SECTION( "Subtract array from array (float)" )
    {
        sRandomNormalInitializerComponent lInitializer{};
        lInitializer.mType = eScalarType::FLOAT32;

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lOpNode  = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
        auto lOpSNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
        auto lResult0 = Subtract( lScope, lOpNode, lOpSNode );
        lScope.Run( lResult0 );

        std::vector<float> lLeftTensorValues  = lOpNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        std::vector<float> lRightTensorValues = lOpSNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        std::vector<float> lExpectedValues( lOpNode.Get<sMultiTensorComponent>().mValue.SizeAs<float>() );
        std::vector<float> lTensorValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        for( uint32_t i = 0; i < lLeftTensorValues.size(); i++ )
        {
            lExpectedValues[i] = lLeftTensorValues[i] - lRightTensorValues[i];
        }
        REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
    }

    SECTION( "Divide vector by array (float)" )
    {
        sRandomNormalInitializerComponent lInitializer{};
        lInitializer.mType = eScalarType::FLOAT32;

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto                     lOpNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
        std::vector<ScalarValue> lConstants{ 3.123f, 4.345f };
        auto                     lOpSNode = VectorValue( lScope, lConstants );
        auto                     lResult0 = Divide( lScope, lOpNode, lOpSNode );
        auto                     lResult1 = Divide( lScope, lOpSNode, lOpNode );
        lScope.Run( { lResult0, lResult1 } );

        {
            std::vector<float> lLeftTensorValues  = lOpNode.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
            float              lRightTensorValues = std::get<float>( lConstants[0] );
            std::vector<float> lExpectedValues( lLeftTensorValues.size() );
            std::vector<float> lTensorValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
            for( uint32_t i = 0; i < lLeftTensorValues.size(); i++ )
            {
                lExpectedValues[i] = lLeftTensorValues[i] / lRightTensorValues;
            }
            REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
        }

        {
            std::vector<float> lLeftTensorValues  = lOpNode.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
            float              lRightTensorValues = std::get<float>( lConstants[1] );
            std::vector<float> lExpectedValues( lLeftTensorValues.size() );
            std::vector<float> lTensorValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
            for( uint32_t i = 0; i < lLeftTensorValues.size(); i++ )
            {
                lExpectedValues[i] = lLeftTensorValues[i] / lRightTensorValues;
            }
            REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
        }

        {
            std::vector<float> lLeftTensorValues  = lOpNode.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
            float              lRightTensorValues = std::get<float>( lConstants[0] );
            std::vector<float> lExpectedValues( lLeftTensorValues.size() );
            std::vector<float> lTensorValues = lResult1.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
            for( uint32_t i = 0; i < lLeftTensorValues.size(); i++ )
            {
                lExpectedValues[i] = lRightTensorValues / lLeftTensorValues[i];
            }
            REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
        }

        {
            std::vector<float> lLeftTensorValues  = lOpNode.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
            float              lRightTensorValues = std::get<float>( lConstants[1] );
            std::vector<float> lExpectedValues( lLeftTensorValues.size() );
            std::vector<float> lTensorValues = lResult1.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
            for( uint32_t i = 0; i < lLeftTensorValues.size(); i++ )
            {
                lExpectedValues[i] = lRightTensorValues / lLeftTensorValues[i];
            }
            REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
        }
    }
}

TEMPLATE_TEST_CASE(
    "DIVIDE Array_Scalar", "[CORE_COMPUTATION_GRAPH]", uint16_t, uint32_t, uint64_t, int16_t, int32_t, int64_t, float, double )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    std::vector<uint32_t>              lDim1{ 7, 3, 1400 };
    std::vector<std::vector<TestType>> lValues1 = RandomVector<TestType>( lDim1, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t>              lDim2{ 2, 7, 700 };
    std::vector<std::vector<TestType>> lValues2 = RandomVector<TestType>( lDim2, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t>              lDim3{ 3, 5, 200 };
    std::vector<std::vector<TestType>> lValues3 = RandomVector<TestType>( lDim3, 0, std::numeric_limits<TestType>::max() );

    auto lScalarNode = ConstantScalarValue( lScope, static_cast<TestType>( 13 ) );

    std::vector<TestType> lExpectedValues;
    auto                  lExpectedValues1 =
        BroadcastMap<TestType>( lValues1, static_cast<TestType>( 13 ), []( TestType x, TestType y ) { return x / y; } );
    auto lExpectedValues2 =
        BroadcastMap<TestType>( lValues2, static_cast<TestType>( 13 ), []( TestType x, TestType y ) { return x / y; } );
    auto lExpectedValues3 =
        BroadcastMap<TestType>( lValues3, static_cast<TestType>( 13 ), []( TestType x, TestType y ) { return x / y; } );

    lExpectedValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lExpectedValues1 ), ConcatenateVectors( lExpectedValues2 ), ConcatenateVectors( lExpectedValues3 ) } );

    std::vector<TestType> lInputValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lValues1 ), ConcatenateVectors( lValues2 ), ConcatenateVectors( lValues3 ) } );

    sDataInitializerComponent lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( lScope, lInputInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( TestType ) ) );

    auto lResult0 = Divide( lScope, lInputTensor, lScalarNode );
    lScope.Run( lResult0 );

    auto lOutputShape = lResult0.Get<sMultiTensorComponent>().mValue.Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.mRank == 3 );
    REQUIRE( lOutputShape.mShape[0] == std::vector<uint32_t>{ 7, 3, 1400 } );
    REQUIRE( lOutputShape.mShape[1] == std::vector<uint32_t>{ 2, 7, 700 } );
    REQUIRE( lOutputShape.mShape[2] == std::vector<uint32_t>{ 3, 5, 200 } );

    std::vector<TestType> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<TestType>();
    REQUIRE( lResultValues0 == lExpectedValues );
}

TEST_CASE( "Tensor AND Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 1024 * 1024;
    Scope  lScope( lPoolSize );

    std::vector<uint32_t> lDim1{ 29, 12, 23 };
    std::vector<uint32_t> lDim2{ 33, 14, 13 };

    std::vector<uint8_t> lValues0  = RandomBool( 29 * 12 * 23 );
    std::vector<uint8_t> lValues01 = RandomBool( 33 * 14 * 13 );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    sDataInitializerComponent lInitializer0( lValues0 );
    auto lOpNodeLeft = MultiTensorValue( lScope, lInitializer0, sTensorShape( { lDim1, lDim2 }, sizeof( uint8_t ) ) );

    std::vector<uint8_t> lValues1  = RandomBool( 29 * 12 * 23 );
    std::vector<uint8_t> lValues11 = RandomBool( 33 * 14 * 13 );
    lValues1.insert( lValues1.end(), lValues11.begin(), lValues11.end() );
    sDataInitializerComponent lInitializer1( lValues1 );
    auto lOpNodeRight = MultiTensorValue( lScope, lInitializer1, sTensorShape( { lDim1, lDim2 }, sizeof( uint8_t ) ) );

    auto lResult0 = And( lScope, lOpNodeLeft, lOpNodeRight );
    auto lResult1 = And( lScope, lOpNodeRight, lOpNodeLeft );
    lScope.Run( { lResult0, lResult1 } );

    std::vector<uint8_t> lExpectedValues( lValues0.size() );
    for( uint32_t i = 0; i < lValues0.size(); i++ )
    {
        lExpectedValues[i] = ( lValues0[i] && lValues1[i] );
    }
    std::vector<uint8_t> lTensorValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( lTensorValues0 == lExpectedValues );

    std::vector<uint8_t> lTensorValues1 = lResult1.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( lTensorValues1 == lExpectedValues );
}

TEST_CASE( "Tensor AND Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 1024 * 1024;
    Scope  lScope( lPoolSize );

    std::vector<uint32_t> lDim1{ 2, 12, 23 };
    std::vector<uint32_t> lDim2{ 3, 14, 13 };

    std::vector<uint8_t> Values00  = RandomBool( 2 * 12 * 23 );
    std::vector<uint8_t> lValues01 = RandomBool( 3 * 14 * 13 );
    std::vector<uint8_t> lValues0;
    lValues0.insert( lValues0.end(), Values00.begin(), Values00.end() );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    sDataInitializerComponent lInitializer0( lValues0 );
    auto lOpNodeLeft = MultiTensorValue( lScope, lInitializer0, sTensorShape( { lDim1, lDim2 }, sizeof( uint8_t ) ) );

    std::vector<ScalarValue> lConstants{ static_cast<uint8_t>( 0 ), static_cast<uint8_t>( 1 ) };
    auto                     lOpNodeRight = VectorValue( lScope, lConstants );

    auto lResult0 = And( lScope, lOpNodeLeft, lOpNodeRight );
    auto lResult1 = And( lScope, lOpNodeRight, lOpNodeLeft );
    lScope.Run( { lResult0, lResult1 } );

    std::vector<uint8_t> lExpectedValues0( Values00.size() );
    for( uint32_t i = 0; i < Values00.size(); i++ )
    {
        lExpectedValues0[i] = ( Values00[i] && std::get<uint8_t>( lConstants[0] ) );
    }

    std::vector<uint8_t> ExpectedValues1( lValues01.size() );
    for( uint32_t i = 0; i < lValues01.size(); i++ )
    {
        ExpectedValues1[i] = ( lValues01[i] && std::get<uint8_t>( lConstants[1] ) );
    }

    lExpectedValues0.insert( lExpectedValues0.end(), ExpectedValues1.begin(), ExpectedValues1.end() );

    std::vector<uint8_t> lTensorValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( lTensorValues0 == lExpectedValues0 );

    std::vector<uint8_t> lTensorValues1 = lResult1.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( lTensorValues1 == lExpectedValues0 );
}

TEST_CASE( "Tensor AND Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 1024 * 1024;
    Scope  lScope( lPoolSize );

    std::vector<uint32_t> lDim1{ 2, 12, 23 };
    std::vector<uint32_t> lDim2{ 3, 14, 13 };

    std::vector<uint8_t> Values00  = RandomBool( 2 * 12 * 23 );
    std::vector<uint8_t> lValues01 = RandomBool( 3 * 14 * 13 );
    std::vector<uint8_t> lValues0;
    lValues0.insert( lValues0.end(), Values00.begin(), Values00.end() );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    sDataInitializerComponent lInitializer0( lValues0 );
    auto lOpNodeLeft = MultiTensorValue( lScope, lInitializer0, sTensorShape( { lDim1, lDim2 }, sizeof( uint8_t ) ) );

    {
        auto lOpNodeRight = ConstantScalarValue( lScope, static_cast<uint8_t>( 0 ) );

        auto lResult0 = And( lScope, lOpNodeLeft, lOpNodeRight );
        auto lResult1 = And( lScope, lOpNodeRight, lOpNodeLeft );
        lScope.Run( { lResult0, lResult1 } );

        std::vector<uint8_t> lExpectedValues( lValues0.size() );
        std::fill( lExpectedValues.begin(), lExpectedValues.end(), static_cast<uint8_t>( 0 ) );

        std::vector<uint8_t> lTensorValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
        REQUIRE( lTensorValues0 == lExpectedValues );

        std::vector<uint8_t> lTensorValues1 = lResult1.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
        REQUIRE( lTensorValues1 == lExpectedValues );
    }

    {
        auto lOpNodeRight = ConstantScalarValue( lScope, static_cast<uint8_t>( 1 ) );

        auto lResult0 = And( lScope, lOpNodeLeft, lOpNodeRight );
        auto lResult1 = And( lScope, lOpNodeRight, lOpNodeLeft );
        lScope.Run( { lResult0, lResult1 } );

        std::vector<uint8_t> lExpectedValues( lValues0.size() );

        for( uint32_t i = 0; i < lValues0.size(); i++ )
        {
            lExpectedValues[i] = lValues0[i];
        }

        std::vector<uint8_t> lTensorValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
        REQUIRE( lTensorValues0 == lExpectedValues );

        std::vector<uint8_t> lTensorValues1 = lResult1.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
        REQUIRE( lTensorValues1 == lExpectedValues );
    }
}

TEST_CASE( "Tensor OR Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 1024 * 1024;
    Scope  lScope( lPoolSize );

    std::vector<uint32_t> lDim1{ 29, 12, 23 };
    std::vector<uint32_t> lDim2{ 33, 14, 13 };

    std::vector<uint8_t> lValues0  = RandomBool( 29 * 12 * 23 );
    std::vector<uint8_t> lValues01 = RandomBool( 33 * 14 * 13 );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    sDataInitializerComponent lInitializer0( lValues0 );
    auto lOpNodeLeft = MultiTensorValue( lScope, lInitializer0, sTensorShape( { lDim1, lDim2 }, sizeof( uint8_t ) ) );

    std::vector<uint8_t> lValues1  = RandomBool( 29 * 12 * 23 );
    std::vector<uint8_t> lValues11 = RandomBool( 33 * 14 * 13 );
    lValues1.insert( lValues1.end(), lValues11.begin(), lValues11.end() );
    sDataInitializerComponent lInitializer1( lValues1 );
    auto lOpNodeRight = MultiTensorValue( lScope, lInitializer1, sTensorShape( { lDim1, lDim2 }, sizeof( uint8_t ) ) );

    auto lResult0 = Or( lScope, lOpNodeLeft, lOpNodeRight );
    auto lResult1 = Or( lScope, lOpNodeRight, lOpNodeLeft );
    lScope.Run( { lResult0, lResult1 } );

    std::vector<uint8_t> lExpectedValues( lValues0.size() );
    for( uint32_t i = 0; i < lValues0.size(); i++ )
    {
        lExpectedValues[i] = ( lValues0[i] || lValues1[i] );
    }

    std::vector<uint8_t> lTensorValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( lTensorValues0 == lExpectedValues );

    std::vector<uint8_t> lTensorValues1 = lResult1.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( lTensorValues1 == lExpectedValues );
}

TEST_CASE( "Tensor OR Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 1024 * 1024;
    Scope  lScope( lPoolSize );

    std::vector<uint32_t> lDim1{ 2, 12, 23 };
    std::vector<uint32_t> lDim2{ 3, 14, 13 };

    std::vector<uint8_t> Values00  = RandomBool( 2 * 12 * 23 );
    std::vector<uint8_t> lValues01 = RandomBool( 3 * 14 * 13 );
    std::vector<uint8_t> lValues0;
    lValues0.insert( lValues0.end(), Values00.begin(), Values00.end() );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    sDataInitializerComponent lInitializer0( lValues0 );
    auto lOpNodeLeft = MultiTensorValue( lScope, lInitializer0, sTensorShape( { lDim1, lDim2 }, sizeof( uint8_t ) ) );

    std::vector<ScalarValue> lConstants{ static_cast<uint8_t>( 0 ), static_cast<uint8_t>( 1 ) };
    auto                     lOpNodeRight = VectorValue( lScope, lConstants );

    auto lResult0 = Or( lScope, lOpNodeLeft, lOpNodeRight );
    auto lResult1 = Or( lScope, lOpNodeRight, lOpNodeLeft );
    lScope.Run( { lResult0, lResult1 } );

    std::vector<uint8_t> lExpectedValues0( Values00.size() );
    for( uint32_t i = 0; i < Values00.size(); i++ )
    {
        lExpectedValues0[i] = ( Values00[i] || std::get<uint8_t>( lConstants[0] ) );
    }

    std::vector<uint8_t> ExpectedValues1( lValues01.size() );
    for( uint32_t i = 0; i < lValues01.size(); i++ )
    {
        ExpectedValues1[i] = ( lValues01[i] || std::get<uint8_t>( lConstants[1] ) );
    }

    lExpectedValues0.insert( lExpectedValues0.end(), ExpectedValues1.begin(), ExpectedValues1.end() );

    std::vector<uint8_t> lTensorValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( lTensorValues0 == lExpectedValues0 );

    std::vector<uint8_t> lTensorValues1 = lResult1.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( lTensorValues1 == lExpectedValues0 );
}

TEST_CASE( "Tensor OR Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 1024 * 1024;
    Scope  lScope( lPoolSize );

    std::vector<uint32_t> lDim1{ 2, 12, 23 };
    std::vector<uint32_t> lDim2{ 3, 14, 13 };

    std::vector<uint8_t> Values00  = RandomBool( 2 * 12 * 23 );
    std::vector<uint8_t> lValues01 = RandomBool( 3 * 14 * 13 );
    std::vector<uint8_t> lValues0;
    lValues0.insert( lValues0.end(), Values00.begin(), Values00.end() );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    sDataInitializerComponent lInitializer0( lValues0 );
    auto lOpNodeLeft = MultiTensorValue( lScope, lInitializer0, sTensorShape( { lDim1, lDim2 }, sizeof( uint8_t ) ) );

    {
        auto lOpNodeRight = ConstantScalarValue( lScope, static_cast<uint8_t>( 1 ) );

        auto lResult0 = Or( lScope, lOpNodeLeft, lOpNodeRight );
        auto lResult1 = Or( lScope, lOpNodeRight, lOpNodeLeft );
        lScope.Run( { lResult0, lResult1 } );

        std::vector<uint8_t> lExpectedValues( lValues0.size() );
        std::fill( lExpectedValues.begin(), lExpectedValues.end(), static_cast<uint8_t>( 1 ) );

        std::vector<uint8_t> lTensorValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
        REQUIRE( lTensorValues0 == lExpectedValues );

        std::vector<uint8_t> lTensorValues1 = lResult1.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
        REQUIRE( lTensorValues1 == lExpectedValues );
    }

    {
        auto lOpNodeRight = ConstantScalarValue( lScope, static_cast<uint8_t>( 0 ) );

        auto lResult0 = Or( lScope, lOpNodeLeft, lOpNodeRight );
        auto lResult1 = Or( lScope, lOpNodeRight, lOpNodeLeft );
        lScope.Run( { lResult0, lResult1 } );

        std::vector<uint8_t> lExpectedValues( lValues0.size() );

        for( uint32_t i = 0; i < lValues0.size(); i++ )
        {
            lExpectedValues[i] = lValues0[i];
        }

        std::vector<uint8_t> lTensorValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
        REQUIRE( lTensorValues0 == lExpectedValues );

        std::vector<uint8_t> lTensorValues1 = lResult1.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
        REQUIRE( lTensorValues1 == lExpectedValues );
    }
}

TEST_CASE( "NOT Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 1024 * 1024;
    Scope  lScope( lPoolSize );

    std::vector<uint32_t> lDim1{ 29, 12, 23 };
    std::vector<uint32_t> lDim2{ 33, 14, 13 };

    std::vector<uint8_t> lValues0  = RandomBool( 29 * 12 * 23 );
    std::vector<uint8_t> lValues01 = RandomBool( 33 * 14 * 13 );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    sDataInitializerComponent lInitializer0( lValues0 );
    auto lOpNodeLeft = MultiTensorValue( lScope, lInitializer0, sTensorShape( { lDim1, lDim2 }, sizeof( uint8_t ) ) );

    auto lResult0 = Not( lScope, lOpNodeLeft );
    lScope.Run( lResult0 );

    std::vector<uint8_t> lTensorValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> lExpectedValues( lValues0.size() );
    for( uint32_t i = 0; i < lValues0.size(); i++ )
    {
        lExpectedValues[i] = !( lValues0[i] );
    }
    REQUIRE( lTensorValues == lExpectedValues );
}

TEST_CASE( "Tensor BITWISE_AND Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 1024 * 1024;
    Scope  lScope( lPoolSize );

    std::vector<uint32_t> lDim1{ 29, 12, 23 };
    std::vector<uint32_t> lDim2{ 33, 14, 13 };

    auto lValues0  = RandomNumber<uint64_t>( 29 * 12 * 23 );
    auto lValues01 = RandomNumber<uint64_t>( 33 * 14 * 13 );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    sDataInitializerComponent lInitializer0( lValues0 );
    auto lOpNodeLeft = MultiTensorValue( lScope, lInitializer0, sTensorShape( { lDim1, lDim2 }, sizeof( uint64_t ) ) );

    auto lValues1  = RandomNumber<uint64_t>( 29 * 12 * 23 );
    auto lValues11 = RandomNumber<uint64_t>( 33 * 14 * 13 );
    lValues1.insert( lValues1.end(), lValues11.begin(), lValues11.end() );
    sDataInitializerComponent lInitializer1( lValues1 );
    auto lOpNodeRight = MultiTensorValue( lScope, lInitializer1, sTensorShape( { lDim1, lDim2 }, sizeof( uint64_t ) ) );

    auto lResult0 = BitwiseAnd( lScope, lOpNodeLeft, lOpNodeRight );
    auto lResult1 = BitwiseAnd( lScope, lOpNodeRight, lOpNodeLeft );
    lScope.Run( { lResult0, lResult1 } );

    std::vector<uint64_t> lExpectedValues( lValues0.size() );
    for( uint32_t i = 0; i < lValues0.size(); i++ )
    {
        lExpectedValues[i] = ( lValues0[i] & lValues1[i] );
    }
    auto lTensorValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( lTensorValues0 == lExpectedValues );

    auto lTensorValues1 = lResult1.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( lTensorValues1 == lExpectedValues );
}

TEST_CASE( "Tensor BITWISE_AND Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 1024 * 1024;
    Scope  lScope( lPoolSize );

    std::vector<uint32_t> lDim1{ 2, 12, 23 };
    std::vector<uint32_t> lDim2{ 3, 14, 13 };

    auto                  Values00  = RandomNumber<uint64_t>( 2 * 12 * 23 );
    auto                  lValues01 = RandomNumber<uint64_t>( 3 * 14 * 13 );
    std::vector<uint64_t> lValues0;
    lValues0.insert( lValues0.end(), Values00.begin(), Values00.end() );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    sDataInitializerComponent lInitializer0( lValues0 );
    auto lOpNodeLeft = MultiTensorValue( lScope, lInitializer0, sTensorShape( { lDim1, lDim2 }, sizeof( uint64_t ) ) );

    std::vector<ScalarValue> lConstants{ static_cast<uint64_t>( 0x1b34d765ef12acac ), static_cast<uint64_t>( 0x1b34d065ef120cfc ) };
    auto                     lOpNodeRight = VectorValue( lScope, lConstants );

    auto lResult0 = BitwiseAnd( lScope, lOpNodeLeft, lOpNodeRight );
    auto lResult1 = BitwiseAnd( lScope, lOpNodeRight, lOpNodeLeft );
    lScope.Run( { lResult0, lResult1 } );

    std::vector<uint64_t> lExpectedValues0( Values00.size() );
    for( uint32_t i = 0; i < Values00.size(); i++ )
    {
        lExpectedValues0[i] = ( Values00[i] & std::get<uint64_t>( lConstants[0] ) );
    }

    std::vector<uint64_t> ExpectedValues1( lValues01.size() );
    for( uint32_t i = 0; i < lValues01.size(); i++ )
    {
        ExpectedValues1[i] = ( lValues01[i] & std::get<uint64_t>( lConstants[1] ) );
    }

    lExpectedValues0.insert( lExpectedValues0.end(), ExpectedValues1.begin(), ExpectedValues1.end() );

    auto lTensorValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( lTensorValues0 == lExpectedValues0 );

    auto lTensorValues1 = lResult1.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( lTensorValues1 == lExpectedValues0 );
}

TEST_CASE( "Tensor BITWISE_AND Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 1024 * 1024;
    Scope  lScope( lPoolSize );

    std::vector<uint32_t> lDim1{ 2, 12, 23 };
    std::vector<uint32_t> lDim2{ 3, 14, 13 };

    auto                  Values00  = RandomNumber<uint64_t>( 2 * 12 * 23 );
    auto                  lValues01 = RandomNumber<uint64_t>( 3 * 14 * 13 );
    std::vector<uint64_t> lValues0;
    lValues0.insert( lValues0.end(), Values00.begin(), Values00.end() );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    sDataInitializerComponent lInitializer0( lValues0 );
    auto lOpNodeLeft = MultiTensorValue( lScope, lInitializer0, sTensorShape( { lDim1, lDim2 }, sizeof( uint64_t ) ) );

    auto lOpNodeRight = ConstantScalarValue( lScope, static_cast<uint64_t>( 0x1b34d765ef12acac ) );

    auto lResult0 = BitwiseAnd( lScope, lOpNodeLeft, lOpNodeRight );
    auto lResult1 = BitwiseAnd( lScope, lOpNodeRight, lOpNodeLeft );
    lScope.Run( { lResult0, lResult1 } );

    std::vector<uint64_t> lExpectedValues( lValues0.size() );
    for( uint32_t i = 0; i < lValues0.size(); i++ )
    {
        lExpectedValues[i] = ( lValues0[i] & static_cast<uint64_t>( 0x1b34d765ef12acac ) );
    }

    auto lTensorValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( lTensorValues0 == lExpectedValues );

    auto lTensorValues1 = lResult1.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( lTensorValues1 == lExpectedValues );
}

TEST_CASE( "Tensor BITWISE_OR Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 1024 * 1024;
    Scope  lScope( lPoolSize );

    std::vector<uint32_t> lDim1{ 29, 12, 23 };
    std::vector<uint32_t> lDim2{ 33, 14, 13 };

    auto lValues0  = RandomNumber<uint64_t>( 29 * 12 * 23 );
    auto lValues01 = RandomNumber<uint64_t>( 33 * 14 * 13 );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    sDataInitializerComponent lInitializer0( lValues0 );
    auto lOpNodeLeft = MultiTensorValue( lScope, lInitializer0, sTensorShape( { lDim1, lDim2 }, sizeof( uint64_t ) ) );

    auto lValues1  = RandomNumber<uint64_t>( 29 * 12 * 23 );
    auto lValues11 = RandomNumber<uint64_t>( 33 * 14 * 13 );
    lValues1.insert( lValues1.end(), lValues11.begin(), lValues11.end() );
    sDataInitializerComponent lInitializer1( lValues1 );
    auto lOpNodeRight = MultiTensorValue( lScope, lInitializer1, sTensorShape( { lDim1, lDim2 }, sizeof( uint64_t ) ) );

    auto lResult0 = BitwiseOr( lScope, lOpNodeLeft, lOpNodeRight );
    auto lResult1 = BitwiseOr( lScope, lOpNodeRight, lOpNodeLeft );
    lScope.Run( { lResult0, lResult1 } );

    std::vector<uint64_t> lExpectedValues( lValues0.size() );
    for( uint32_t i = 0; i < lValues0.size(); i++ )
    {
        lExpectedValues[i] = ( lValues0[i] | lValues1[i] );
    }

    auto lTensorValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( lTensorValues0 == lExpectedValues );

    auto lTensorValues1 = lResult1.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( lTensorValues1 == lExpectedValues );
}

TEST_CASE( "Tensor BITWISE_OR Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 1024 * 1024;
    Scope  lScope( lPoolSize );

    std::vector<uint32_t> lDim1{ 2, 12, 23 };
    std::vector<uint32_t> lDim2{ 3, 14, 13 };

    auto                  Values00  = RandomNumber<uint64_t>( 2 * 12 * 23 );
    auto                  lValues01 = RandomNumber<uint64_t>( 3 * 14 * 13 );
    std::vector<uint64_t> lValues0;
    lValues0.insert( lValues0.end(), Values00.begin(), Values00.end() );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    sDataInitializerComponent lInitializer0( lValues0 );
    auto lOpNodeLeft = MultiTensorValue( lScope, lInitializer0, sTensorShape( { lDim1, lDim2 }, sizeof( uint64_t ) ) );

    std::vector<ScalarValue> lConstants{ static_cast<uint64_t>( 0x1b34d765ef12acac ), static_cast<uint64_t>( 0x1b34d065ef120cfc ) };
    auto                     lOpNodeRight = VectorValue( lScope, lConstants );

    auto lResult0 = BitwiseOr( lScope, lOpNodeLeft, lOpNodeRight );
    auto lResult1 = BitwiseOr( lScope, lOpNodeRight, lOpNodeLeft );
    lScope.Run( { lResult0, lResult1 } );

    std::vector<uint64_t> lExpectedValues0( Values00.size() );
    for( uint32_t i = 0; i < Values00.size(); i++ )
    {
        lExpectedValues0[i] = ( Values00[i] | std::get<uint64_t>( lConstants[0] ) );
    }

    std::vector<uint64_t> ExpectedValues1( lValues01.size() );
    for( uint32_t i = 0; i < lValues01.size(); i++ )
    {
        ExpectedValues1[i] = ( lValues01[i] | std::get<uint64_t>( lConstants[1] ) );
    }

    lExpectedValues0.insert( lExpectedValues0.end(), ExpectedValues1.begin(), ExpectedValues1.end() );

    auto lTensorValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( lTensorValues0 == lExpectedValues0 );

    auto lTensorValues1 = lResult1.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( lTensorValues1 == lExpectedValues0 );
}

TEST_CASE( "Tensor BITWISE_OR Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 1024 * 1024;
    Scope  lScope( lPoolSize );

    std::vector<uint32_t> lDim1{ 2, 12, 23 };
    std::vector<uint32_t> lDim2{ 3, 14, 13 };

    auto                  Values00  = RandomNumber<uint64_t>( 2 * 12 * 23 );
    auto                  lValues01 = RandomNumber<uint64_t>( 3 * 14 * 13 );
    std::vector<uint64_t> lValues0;
    lValues0.insert( lValues0.end(), Values00.begin(), Values00.end() );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    sDataInitializerComponent lInitializer0( lValues0 );
    auto lOpNodeLeft = MultiTensorValue( lScope, lInitializer0, sTensorShape( { lDim1, lDim2 }, sizeof( uint64_t ) ) );

    auto lOpNodeRight = ConstantScalarValue( lScope, static_cast<uint64_t>( 0x1b34d765ef12acac ) );

    auto lResult0 = BitwiseOr( lScope, lOpNodeLeft, lOpNodeRight );
    auto lResult1 = BitwiseOr( lScope, lOpNodeRight, lOpNodeLeft );
    lScope.Run( { lResult0, lResult1 } );

    std::vector<uint64_t> lExpectedValues( lValues0.size() );
    for( uint32_t i = 0; i < lValues0.size(); i++ )
    {
        lExpectedValues[i] = ( lValues0[i] | static_cast<uint64_t>( 0x1b34d765ef12acac ) );
    }

    auto lTensorValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( lTensorValues0 == lExpectedValues );

    auto lTensorValues1 = lResult1.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( lTensorValues1 == lExpectedValues );
}

TEST_CASE( "BITWISE_NOT Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 1024 * 1024;
    Scope  lScope( lPoolSize );

    std::vector<uint32_t> lDim1{ 29, 12, 23 };
    std::vector<uint32_t> lDim2{ 33, 14, 13 };

    auto lValues0  = RandomNumber<uint64_t>( 29 * 12 * 23 );
    auto lValues01 = RandomNumber<uint64_t>( 33 * 14 * 13 );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    sDataInitializerComponent lInitializer0( lValues0 );
    auto lOpNodeLeft = MultiTensorValue( lScope, lInitializer0, sTensorShape( { lDim1, lDim2 }, sizeof( uint64_t ) ) );

    auto lResult0 = BitwiseNot( lScope, lOpNodeLeft );
    lScope.Run( lResult0 );

    auto                  lTensorValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint64_t>();
    std::vector<uint64_t> lExpectedValues( lValues0.size() );
    for( uint32_t i = 0; i < lValues0.size(); i++ )
    {
        lExpectedValues[i] = ~( lValues0[i] );
    }
    REQUIRE( lTensorValues == lExpectedValues );
}

TEST_CASE( "Affine transform node", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    SECTION( "Affine transform tensor/tensor/tensor (float)" )
    {
        sRandomUniformInitializerComponent lInitializer{};
        lInitializer.mType = eScalarType::FLOAT32;

        std::vector<uint32_t> lDim1{ 8, 12 };
        std::vector<uint32_t> lDim2{ 8, 16 };

        auto lNodeA = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
        auto lNodeX = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
        auto lNodeB = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

        auto lResult0 = AffineTransform( lScope, lNodeA, lNodeX, lNodeB );
        lScope.Run( lResult0 );

        std::vector<float> lValues_A = lNodeA.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        std::vector<float> lValues_X = lNodeX.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        std::vector<float> lValues_B = lNodeB.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        std::vector<float> lExpectedValues( lNodeX.Get<sMultiTensorComponent>().mValue.SizeAs<float>() );
        for( uint32_t i = 0; i < lValues_X.size(); i++ )
        {
            lExpectedValues[i] = lValues_X[i] * lValues_A[i] + lValues_B[i];
        }
        std::vector<float> lTensorValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();

        REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
    }

    SECTION( "Affine transform tensor/tensor/vector (float)" )
    {
        sRandomUniformInitializerComponent lInitializer{};
        lInitializer.mType = eScalarType::FLOAT32;

        std::vector<uint32_t> lDim1{ 8, 12 };
        std::vector<uint32_t> lDim2{ 8, 16 };

        std::vector<float>       lBValues{ 2.142983764918237649f, 3.234987659834765f };
        std::vector<ScalarValue> lB( 5 );
        lB[0] = lBValues[0];
        lB[1] = lBValues[1];

        auto lNodeA = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
        auto lNodeX = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
        auto lNodeB = VectorValue( lScope, lB );

        auto lResult0 = AffineTransform( lScope, lNodeA, lNodeX, lNodeB );
        lScope.Run( lResult0 );

        std::vector<float> lExpectedValues = {};
        for( uint32_t i = 0; i < lBValues.size(); i++ )
        {
            uint32_t           lSize = lNodeX.Get<sMultiTensorComponent>().mValue.Shape().GetBufferSizeAs<float>( i ).mSize;
            std::vector<float> lA    = lNodeA.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( i );
            std::vector<float> lX    = lNodeX.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( i );
            std::vector<float> lValues( lSize );
            for( uint32_t j = 0; j < lSize; j++ )
            {
                lValues[j] = lA[j] * lX[j] + lBValues[i];
            }
            lExpectedValues.insert( lExpectedValues.end(), lValues.begin(), lValues.end() );
        }

        std::vector<float> lTensorValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
    }

    SECTION( "Affine transform tensor/tensor/scalar (float)" )
    {
        sRandomUniformInitializerComponent lInitializer{};
        lInitializer.mType = eScalarType::FLOAT32;

        std::vector<uint32_t> lDim1{ 8, 12 };
        std::vector<uint32_t> lDim2{ 8, 16 };

        float lScalarB = 2.142983764918237649f;
        auto  lNodeA   = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
        auto  lNodeX   = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
        auto  lNodeB   = ConstantScalarValue( lScope, lScalarB );

        auto lResult0 = AffineTransform( lScope, lNodeA, lNodeX, lNodeB );
        lScope.Run( lResult0 );

        std::vector<float> lExpectedValues = {};
        for( uint32_t i = 0; i < lNodeX.Get<sMultiTensorComponent>().mValue.Shape().CountLayers(); i++ )
        {
            uint32_t           lSize = lNodeX.Get<sMultiTensorComponent>().mValue.Shape().GetBufferSizeAs<float>( i ).mSize;
            std::vector<float> lA    = lNodeA.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( i );
            std::vector<float> lX    = lNodeX.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( i );
            std::vector<float> lValues( lSize );
            for( uint32_t j = 0; j < lSize; j++ )
            {
                lValues[j] = lA[j] * lX[j] + lScalarB;
            }
            lExpectedValues.insert( lExpectedValues.end(), lValues.begin(), lValues.end() );
        }

        std::vector<float> lTensorValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
    }

    SECTION( "Affine transform vector/tensor/tensor (float)" )
    {
        sRandomUniformInitializerComponent lInitializer{};
        lInitializer.mType = eScalarType::FLOAT32;

        std::vector<uint32_t> lDim1{ 8, 22 };
        std::vector<uint32_t> lDim2{ 8, 64 };

        std::vector<float> lAValues{ 1.109238740928374f, 2.12398471982364f };
        std::vector<float> lBValues{ 2.142983764918237649f, 3.234987659834765f };

        std::vector<ScalarValue> lA( 5 );
        lA[0] = lAValues[0];
        lA[1] = lAValues[1];

        std::vector<ScalarValue> lB( 5 );
        lB[0] = lBValues[0];
        lB[1] = lBValues[1];

        auto lNodeA = VectorValue( lScope, lA );
        auto lNodeB = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
        auto lNodeX = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

        auto lResult0 = AffineTransform( lScope, lNodeA, lNodeX, lNodeB );
        lScope.Run( lResult0 );

        std::vector<float> lExpectedValues = {};
        for( uint32_t i = 0; i < lAValues.size(); i++ )
        {
            uint32_t           lSize = lNodeX.Get<sMultiTensorComponent>().mValue.Shape().GetBufferSizeAs<float>( i ).mSize;
            std::vector<float> lX    = lNodeX.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( i );
            std::vector<float> lB    = lNodeB.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( i );
            std::vector<float> lValues( lSize );
            for( uint32_t j = 0; j < lSize; j++ )
            {
                lValues[j] = lAValues[i] * lX[j] + lB[j];
            }
            lExpectedValues.insert( lExpectedValues.end(), lValues.begin(), lValues.end() );
        }

        std::vector<float> lTensorValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
    }

    SECTION( "Affine transform vector/tensor/vector (float)" )
    {
        sRandomUniformInitializerComponent lInitializer{};
        lInitializer.mType = eScalarType::FLOAT32;

        std::vector<uint32_t> lDim1{ 8, 22 };
        std::vector<uint32_t> lDim2{ 8, 64 };

        std::vector<float> lAValues{ 1.109238740928374f, 2.12398471982364f };
        std::vector<float> lBValues{ 2.142983764918237649f, 3.234987659834765f };

        std::vector<ScalarValue> lA( 5 );
        lA[0] = lAValues[0];
        lA[1] = lAValues[1];

        std::vector<ScalarValue> lB( 5 );
        lB[0] = lBValues[0];
        lB[1] = lBValues[1];

        auto lNodeA = VectorValue( lScope, lA );
        auto lNodeB = VectorValue( lScope, lB );
        auto lNodeX = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

        auto lResult0 = AffineTransform( lScope, lNodeA, lNodeX, lNodeB );
        lScope.Run( lResult0 );

        std::vector<float> lExpectedValues = {};
        for( uint32_t i = 0; i < lAValues.size(); i++ )
        {
            uint32_t           lSize = lNodeX.Get<sMultiTensorComponent>().mValue.Shape().GetBufferSizeAs<float>( i ).mSize;
            std::vector<float> lX    = lNodeX.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( i );
            std::vector<float> lValues( lSize );
            for( uint32_t j = 0; j < lSize; j++ )
            {
                lValues[j] = lAValues[i] * lX[j] + lBValues[i];
            }
            lExpectedValues.insert( lExpectedValues.end(), lValues.begin(), lValues.end() );
        }

        std::vector<float> lTensorValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
    }

    SECTION( "Affine transform vector/tensor/scalar (float)" )
    {
        sRandomUniformInitializerComponent lInitializer{};
        lInitializer.mType = eScalarType::FLOAT32;

        std::vector<uint32_t> lDim1{ 8, 22 };
        std::vector<uint32_t> lDim2{ 8, 64 };

        std::vector<float> lAValues{ 1.109238740928374f, 2.12398471982364f };
        std::vector<float> lBValues{ 2.142983764918237649f, 3.234987659834765f };

        std::vector<ScalarValue> lA( 5 );
        lA[0] = lAValues[0];
        lA[1] = lAValues[1];

        std::vector<ScalarValue> lB( 5 );
        lB[0] = lBValues[0];
        lB[1] = lBValues[1];

        float lScalarB = 2.142983764918237649f;
        auto  lNodeA   = VectorValue( lScope, lA );
        auto  lNodeB   = ConstantScalarValue( lScope, lScalarB );
        auto  lNodeX   = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

        auto lResult0 = AffineTransform( lScope, lNodeA, lNodeX, lNodeB );
        lScope.Run( lResult0 );

        std::vector<float> lExpectedValues = {};
        for( uint32_t i = 0; i < lAValues.size(); i++ )
        {
            uint32_t           lSize = lNodeX.Get<sMultiTensorComponent>().mValue.Shape().GetBufferSizeAs<float>( i ).mSize;
            std::vector<float> lX    = lNodeX.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( i );
            std::vector<float> lValues( lSize );
            for( uint32_t j = 0; j < lSize; j++ )
            {
                lValues[j] = lAValues[i] * lX[j] + lScalarB;
            }
            lExpectedValues.insert( lExpectedValues.end(), lValues.begin(), lValues.end() );
        }

        std::vector<float> lTensorValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
    }

    SECTION( "Affine transform scalar/tensor/tensor (float)" )
    {
        sRandomUniformInitializerComponent lInitializer{};
        lInitializer.mType = eScalarType::FLOAT32;

        std::vector<uint32_t> lDim1{ 8, 22 };
        std::vector<uint32_t> lDim2{ 8, 64 };

        std::vector<float> lAValues{ 1.109238740928374f, 2.12398471982364f };
        std::vector<float> lBValues{ 2.142983764918237649f, 3.234987659834765f };

        std::vector<ScalarValue> lA( 5 );
        lA[0] = lAValues[0];
        lA[1] = lAValues[1];

        std::vector<ScalarValue> lB( 5 );
        lB[0] = lBValues[0];
        lB[1] = lBValues[1];

        float lScalarA = 2.142983764918237649f;
        auto  lNodeA   = ConstantScalarValue( lScope, lScalarA );
        auto  lNodeB   = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
        auto  lNodeX   = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

        auto lResult0 = AffineTransform( lScope, lNodeA, lNodeX, lNodeB );
        lScope.Run( lResult0 );

        std::vector<float> lExpectedValues = {};
        for( uint32_t i = 0; i < lAValues.size(); i++ )
        {
            uint32_t           lSize = lNodeX.Get<sMultiTensorComponent>().mValue.Shape().GetBufferSizeAs<float>( i ).mSize;
            std::vector<float> lX    = lNodeX.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( i );
            std::vector<float> lB    = lNodeB.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( i );
            std::vector<float> lValues( lSize );
            for( uint32_t j = 0; j < lSize; j++ )
            {
                lValues[j] = lScalarA * lX[j] + lB[j];
            }
            lExpectedValues.insert( lExpectedValues.end(), lValues.begin(), lValues.end() );
        }

        std::vector<float> lTensorValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
    }

    SECTION( "Affine transform scalar/tensor/vector (float)" )
    {
        sRandomUniformInitializerComponent lInitializer{};
        lInitializer.mType = eScalarType::FLOAT32;

        std::vector<uint32_t> lDim1{ 8, 2 };
        std::vector<uint32_t> lDim2{ 8, 6 };

        std::vector<float> lAValues{ 1.109238740928374f, 2.12398471982364f };
        std::vector<float> lBValues{ 2.142983764918237649f, 3.234987659834765f };

        std::vector<ScalarValue> lA( 5 );
        lA[0] = lAValues[0];
        lA[1] = lAValues[1];

        std::vector<ScalarValue> lB( 5 );
        lB[0] = lBValues[0];
        lB[1] = lBValues[1];

        float lScalarA = 2.142983764918237649f;
        auto  lNodeA   = ConstantScalarValue( lScope, lScalarA );
        auto  lNodeB   = VectorValue( lScope, lB );
        auto  lNodeX   = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

        auto lResult0 = AffineTransform( lScope, lNodeA, lNodeX, lNodeB );
        lScope.Run( lResult0 );

        std::vector<float> lExpectedValues = {};
        for( uint32_t i = 0; i < lAValues.size(); i++ )
        {
            uint32_t           lSize = lNodeX.Get<sMultiTensorComponent>().mValue.Shape().GetBufferSizeAs<float>( i ).mSize;
            std::vector<float> lX    = lNodeX.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( i );
            std::vector<float> lValues( lSize );
            for( uint32_t j = 0; j < lSize; j++ )
            {
                lValues[j] = lScalarA * lX[j] + lBValues[i];
            }
            lExpectedValues.insert( lExpectedValues.end(), lValues.begin(), lValues.end() );
        }

        std::vector<float> lTensorValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
    }

    SECTION( "Affine transform scalar/tensor/scalar (float)" )
    {
        sRandomUniformInitializerComponent lInitializer{};
        lInitializer.mType = eScalarType::FLOAT32;

        std::vector<uint32_t> lDim1{ 8, 22 };
        std::vector<uint32_t> lDim2{ 8, 64 };

        std::vector<float> lAValues{ 1.109238740928374f, 2.12398471982364f };
        std::vector<float> lBValues{ 2.142983764918237649f, 3.234987659834765f };

        std::vector<ScalarValue> lA( 5 );
        lA[0] = lAValues[0];
        lA[1] = lAValues[1];

        std::vector<ScalarValue> lB( 5 );
        lB[0] = lBValues[0];
        lB[1] = lBValues[1];

        float lScalarA = 2.142983764918237649f;
        auto  lNodeA   = ConstantScalarValue( lScope, lScalarA );
        float lScalarB = 2.142983764918237649f;
        auto  lNodeB   = ConstantScalarValue( lScope, lScalarB );
        auto  lNodeX   = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

        auto lResult0 = AffineTransform( lScope, lNodeA, lNodeX, lNodeB );
        lScope.Run( lResult0 );

        std::vector<float> lExpectedValues = {};
        for( uint32_t i = 0; i < lAValues.size(); i++ )
        {
            uint32_t           lSize = lNodeX.Get<sMultiTensorComponent>().mValue.Shape().GetBufferSizeAs<float>( i ).mSize;
            std::vector<float> lX    = lNodeX.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( i );
            std::vector<float> lValues( lSize );
            for( uint32_t j = 0; j < lSize; j++ )
            {
                lValues[j] = lScalarA * lX[j] + lScalarB;
            }
            lExpectedValues.insert( lExpectedValues.end(), lValues.begin(), lValues.end() );
        }

        std::vector<float> lTensorValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
    }
}

TEST_CASE( "Mix node", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    SECTION( "Mix tensors (float)" )
    {
        sRandomNormalInitializerComponent lInitializer{};
        lInitializer.mType = eScalarType::FLOAT32;

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lNodeA  = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
        auto lNodeB  = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
        auto lNode_T = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

        auto lResult0 = Mix( lScope, lNodeA, lNodeB, lNode_T );
        lScope.Run( lResult0 );

        std::vector<float> lValues_A = lNodeA.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        std::vector<float> lValues_B = lNodeB.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        std::vector<float> lValues_T = lNode_T.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        std::vector<float> lExpectedValues( lNodeA.Get<sMultiTensorComponent>().mValue.SizeAs<float>() );
        for( uint32_t i = 0; i < lValues_A.size(); i++ )
        {
            lExpectedValues[i] = ( 1.0f - lValues_T[i] ) * lValues_A[i] + lValues_T[i] * lValues_B[i];
        }
        std::vector<float> lTensorValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();

        REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
    }
}

TEST_CASE( "Linear space node", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    SECTION( "Linear space allocation (float)" )
    {
        sRandomNormalInitializerComponent lInitializer{};
        lInitializer.mType = eScalarType::FLOAT32;

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lNodeA = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
        auto lNodeB = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

        std::vector<uint32_t> lSubdivisions{ 32, 64 };
        auto                  lNode_S = VectorValue( lScope, lSubdivisions );

        auto &lResult0 = LinearSpace( lScope, lNodeA, lNodeB, lNode_S );
        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().mRank == 3 );
        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().mShape[0] == std::vector<uint32_t>{ 2, 2, 32 } );
        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().mShape[1] == std::vector<uint32_t>{ 3, 4, 64 } );
    }

    SECTION( "Linear space (float)" )
    {

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        sConstantValueInitializerComponent lInitializer0{};
        lInitializer0.mValue = 0.5f;

        auto lNodeA = MultiTensorValue( lScope, lInitializer0, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

        sConstantValueInitializerComponent lInitializer1{};
        lInitializer1.mValue = 1.5f;
        auto lNodeB          = MultiTensorValue( lScope, lInitializer1, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

        std::vector<uint32_t> lSubdivisions{ 32, 64 };
        auto                  lNode_S = VectorValue( lScope, lSubdivisions );

        auto &lResult0 = LinearSpace( lScope, lNodeA, lNodeB, lNode_S );
        lScope.Run( lResult0 );

        {
            constexpr uint32_t lSubdivisions = 32;
            std::vector<float> lValues_A     = lNodeA.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
            std::vector<float> lValues_B     = lNodeB.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
            std::vector<float> lExpectedValues1( Prod( lDim1 ) * lSubdivisions );
            uint32_t           x = 0;
            uint32_t           y = 0;
            for( uint32_t i = 0; i < lDim1[0]; i++ )
            {
                for( uint32_t j = 0; j < lDim1[1]; j++ )
                {
                    float Delta = ( lValues_B[x] - lValues_A[x] ) / static_cast<float>( lSubdivisions );
                    for( uint32_t k = 0; k < lSubdivisions; k++ )
                    {
                        lExpectedValues1[y] = lValues_A[x] + static_cast<float>( k ) * Delta;
                        y++;
                    }
                    x++;
                }
            }
            std::vector<float> lB1 = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
            REQUIRE( VectorEqual( lB1, lExpectedValues1 ) );
        }

        {
            constexpr uint32_t lSubdivisions = 64;
            std::vector<float> lValues_A     = lNodeA.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
            std::vector<float> lValues_B     = lNodeB.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
            std::vector<float> lExpectedValues1( Prod( lDim2 ) * lSubdivisions );
            uint32_t           x = 0;
            uint32_t           y = 0;
            for( uint32_t i = 0; i < lDim2[0]; i++ )
            {
                for( uint32_t j = 0; j < lDim2[1]; j++ )
                {
                    float Delta = ( lValues_B[x] - lValues_A[x] ) / static_cast<float>( lSubdivisions );
                    for( uint32_t k = 0; k < lSubdivisions; k++ )
                    {
                        lExpectedValues1[y] = lValues_A[x] + static_cast<float>( k ) * Delta;
                        y++;
                    }
                    x++;
                }
            }
            std::vector<float> lB1 = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
            REQUIRE( VectorEqual( lB1, lExpectedValues1 ) );
        }
    }
}

TEST_CASE( "ARange node", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    SECTION( "ARange allocation (float)" )
    {
        std::vector<float> lAValues{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        std::vector<float> lBValues{ 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
        std::vector<float> lDValues{ 0.01f, .02f, .03f, .04f, .05f };

        auto lNodeA  = ScalarVectorValue( lScope, eScalarType::FLOAT32, lAValues );
        auto lNodeB  = ScalarVectorValue( lScope, eScalarType::FLOAT32, lBValues );
        auto lNode_D = ScalarVectorValue( lScope, eScalarType::FLOAT32, lDValues );

        auto lResult0 = ARange( lScope, lNodeA, lNodeB, lNode_D );
        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().CountLayers() == 5 );
        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().mRank == 1 );
        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().mShape[0] ==
                 std::vector<uint32_t>{ static_cast<uint32_t>( std::ceil( ( lBValues[0] - lAValues[0] ) / lDValues[0] ) ) } );
        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().mShape[1] ==
                 std::vector<uint32_t>{ static_cast<uint32_t>( std::ceil( ( lBValues[1] - lAValues[1] ) / lDValues[1] ) ) } );
        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().mShape[2] ==
                 std::vector<uint32_t>{ static_cast<uint32_t>( std::ceil( ( lBValues[2] - lAValues[2] ) / lDValues[2] ) ) } );
        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().mShape[3] ==
                 std::vector<uint32_t>{ static_cast<uint32_t>( std::ceil( ( lBValues[3] - lAValues[3] ) / lDValues[3] ) ) } );
        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().mShape[4] ==
                 std::vector<uint32_t>{ static_cast<uint32_t>( std::ceil( ( lBValues[4] - lAValues[4] ) / lDValues[4] ) ) } );
    }

    SECTION( "ARange (float)" )
    {
        std::vector<float> lAValues{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        std::vector<float> lBValues{ 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
        std::vector<float> lDValues{ 0.01f, .02f, .03f, .04f, .05f };

        auto lNodeA  = ScalarVectorValue( lScope, eScalarType::FLOAT32, lAValues );
        auto lNodeB  = ScalarVectorValue( lScope, eScalarType::FLOAT32, lBValues );
        auto lNode_D = ScalarVectorValue( lScope, eScalarType::FLOAT32, lDValues );

        auto lResult0 = ARange( lScope, lNodeA, lNodeB, lNode_D );

        lScope.Run( lResult0 );

        std::vector<float> lExpectedValues = {};

        for( uint32_t i = 0; i < lAValues.size(); i++ )
        {
            uint32_t           lSubdivisions = static_cast<uint32_t>( std::ceil( ( lBValues[i] - lAValues[i] ) / lDValues[i] ) );
            std::vector<float> lValues( lSubdivisions );
            for( uint32_t j = 0; j < lSubdivisions; j++ )
            {
                lValues[j] = lAValues[i] + j * lDValues[i];
            }
            lExpectedValues.insert( lExpectedValues.end(), lValues.begin(), lValues.end() );
        }

        std::vector<float> lTensorValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
    }

    SECTION( "ARange (float)" )
    {
        std::vector<float> lAValues{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        std::vector<float> lBValues{ 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
        std::vector<float> lDValues{ 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };

        auto lNodeA  = ScalarVectorValue( lScope, eScalarType::FLOAT32, lAValues );
        auto lNodeB  = ScalarVectorValue( lScope, eScalarType::FLOAT32, lBValues );
        auto lNode_D = ScalarVectorValue( lScope, eScalarType::FLOAT32, lDValues );

        auto lResult0 = ARange( lScope, lNodeA, lNodeB, lNode_D );

        lScope.Run( lResult0 );

        std::vector<float> lExpectedValues = {};

        for( uint32_t i = 0; i < lAValues.size(); i++ )
        {
            uint32_t           lSubdivisions = static_cast<uint32_t>( std::ceil( ( lBValues[i] - lAValues[i] ) / lDValues[i] ) );
            std::vector<float> lValues( lSubdivisions );
            for( uint32_t j = 0; j < lSubdivisions; j++ )
            {
                lValues[j] = lAValues[i] + j * lDValues[i];
            }
            lExpectedValues.insert( lExpectedValues.end(), lValues.begin(), lValues.end() );
        }

        std::vector<float> lTensorValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        REQUIRE( VectorEqual( lTensorValues, lExpectedValues ) );
    }
}

TEST_CASE( "Repeat node", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    SECTION( "Repeat node allocation (float)" )
    {
        sRandomNormalInitializerComponent lInitializer{};
        lInitializer.mType = eScalarType::FLOAT32;

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lNodeA = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

        std::vector<uint32_t> lSubdivisions{ 3, 5 };
        auto                  lNode_S = VectorValue( lScope, lSubdivisions );

        auto lResult0 = Repeat( lScope, lNodeA, lNode_S );
        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().mRank == 3 );
        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().mShape[0] == std::vector<uint32_t>{ 2, 2, 3 } );
        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().mShape[1] == std::vector<uint32_t>{ 3, 4, 5 } );
    }

    SECTION( "Repeat (float)" )
    {
        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        sRandomNormalInitializerComponent lInitializer0{};
        lInitializer0.mType = eScalarType::FLOAT32;

        auto lNodeA = MultiTensorValue( lScope, lInitializer0, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

        std::vector<uint32_t> lRepetitions{ 3, 5 };
        auto                  lNode_R = VectorValue( lScope, lRepetitions );

        auto lResult0 = Repeat( lScope, lNodeA, lNode_R );
        lScope.Run( lResult0 );

        {
            constexpr uint32_t lSubdivisions = 3;
            std::vector<float> lValues_A     = lNodeA.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
            std::vector<float> lExpectedValues1( Prod( lDim1 ) * lSubdivisions );
            uint32_t           x = 0;
            uint32_t           y = 0;
            for( uint32_t i = 0; i < lDim1[0]; i++ )
            {
                for( uint32_t j = 0; j < lDim1[1]; j++ )
                {
                    for( uint32_t k = 0; k < lSubdivisions; k++ )
                    {
                        lExpectedValues1[y] = lValues_A[x];
                        y++;
                    }
                    x++;
                }
            }
            std::vector<float> lB1 = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
            REQUIRE( VectorEqual( lB1, lExpectedValues1 ) );
        }

        {
            constexpr uint32_t lSubdivisions = 5;
            std::vector<float> lValues_A     = lNodeA.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
            std::vector<float> lExpectedValues1( Prod( lDim2 ) * lSubdivisions );
            uint32_t           x = 0;
            uint32_t           y = 0;
            for( uint32_t i = 0; i < lDim2[0]; i++ )
            {
                for( uint32_t j = 0; j < lDim2[1]; j++ )
                {
                    for( uint32_t k = 0; k < lSubdivisions; k++ )
                    {
                        lExpectedValues1[y] = lValues_A[x];
                        y++;
                    }
                    x++;
                }
            }
            std::vector<float> lB1 = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
            REQUIRE( VectorEqual( lB1, lExpectedValues1 ) );
        }
    }
}

TEST_CASE( "Tile node", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    SECTION( "Tile node allocation (float)" )
    {
        sRandomNormalInitializerComponent lInitializer{};
        lInitializer.mType = eScalarType::FLOAT32;

        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        auto lNodeA = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

        std::vector<uint32_t> lSubdivisions{ 3, 5 };
        auto                  lNode_S = VectorValue( lScope, lSubdivisions );

        auto lResult0 = Tile( lScope, lNodeA, lNode_S );
        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().mRank == 3 );
        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().mShape[0] == std::vector<uint32_t>{ 3, 2, 2 } );
        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().mShape[1] == std::vector<uint32_t>{ 5, 3, 4 } );
    }

    SECTION( "Tile (float)" )
    {
        std::vector<uint32_t> lDim1{ 2, 2 };
        std::vector<uint32_t> lDim2{ 3, 4 };

        sRandomNormalInitializerComponent lInitializer0{};
        lInitializer0.mType = eScalarType::FLOAT32;

        auto lNodeA = MultiTensorValue( lScope, lInitializer0, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

        std::vector<uint32_t> lRepetitions{ 7, 11 };
        auto                  lNode_R = VectorValue( lScope, lRepetitions );

        auto lResult0 = Tile( lScope, lNodeA, lNode_R );
        lScope.Run( lResult0 );

        {
            constexpr uint32_t lSubdivisions = 7;
            std::vector<float> lValues_A     = lNodeA.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
            std::vector<float> lExpectedValues1( Prod( lDim1 ) * lSubdivisions );
            uint32_t           x = 0;
            uint32_t           y = 0;
            for( uint32_t k = 0; k < lSubdivisions; k++ )
            {
                for( uint32_t i = 0; i < lDim1[0] * lDim1[1]; i++ )
                {
                    lExpectedValues1[y] = lValues_A[i];
                    y++;
                }
            }
            std::vector<float> lB1 = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
            REQUIRE( VectorEqual( lB1, lExpectedValues1 ) );
        }

        {
            constexpr uint32_t lSubdivisions = 11;
            std::vector<float> lValues_A     = lNodeA.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
            std::vector<float> lExpectedValues1( Prod( lDim2 ) * lSubdivisions );
            uint32_t           x = 0;
            uint32_t           y = 0;
            for( uint32_t k = 0; k < lSubdivisions; k++ )
            {
                for( uint32_t i = 0; i < lDim2[0] * lDim2[1]; i++ )
                {
                    lExpectedValues1[y] = lValues_A[i];
                    y++;
                }
            }
            std::vector<float> lB1 = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
            REQUIRE( VectorEqual( lB1, lExpectedValues1 ) );
        }
    }
}

TEST_CASE( "Expand MultiTensors", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomUniformInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 5, 23, 42 };

    SECTION( "Expanding multi-tensors preserved types" )
    {
        auto lNodeA =
            MultiTensorValue( lScope, lInitializer, sTensorShape( std::vector<std::vector<uint32_t>>{ lDim1 }, sizeof( float ) ) );
        auto lResult0 = Expand( lScope, lNodeA );

        REQUIRE( lResult0.Get<sTypeComponent>().mValue == lNodeA.Get<sTypeComponent>().mValue );
    }

    SECTION( "Expanding multi-tensors gives the correct dimension" )
    {
        auto lNodeA =
            MultiTensorValue( lScope, lInitializer, sTensorShape( std::vector<std::vector<uint32_t>>{ lDim1 }, sizeof( float ) ) );
        auto lResult0 = Expand( lScope, lNodeA );

        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().mRank == ( lDim1.size() - 1 ) );
        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().CountLayers() == lDim1[0] );
        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().mShape[0] == std::vector<uint32_t>{ 23, 42 } );
        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().mShape[1] == std::vector<uint32_t>{ 23, 42 } );
        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().mShape[2] == std::vector<uint32_t>{ 23, 42 } );
        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().mShape[3] == std::vector<uint32_t>{ 23, 42 } );
    }

    SECTION( "Expanding multi-tensors does not change values" )
    {
        auto lNodeA =
            MultiTensorValue( lScope, lInitializer, sTensorShape( std::vector<std::vector<uint32_t>>{ lDim1 }, sizeof( float ) ) );
        auto lResult0 = Expand( lScope, lNodeA );

        lScope.Run( lResult0 );

        std::vector<float> lTensorValues0 = lNodeA.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        std::vector<float> lTensorValues1 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        REQUIRE( VectorEqual( lTensorValues0, lTensorValues0 ) );
    }
}

TEST_CASE( "Collapse MultiTensors", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomUniformInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 23, 42 };
    std::vector<uint32_t> lDim2{ 23, 42 };
    std::vector<uint32_t> lDim3{ 23, 42 };
    std::vector<uint32_t> lDim4{ 23, 42 };

    SECTION( "Collapsing multi-tensors preserved types" )
    {
        auto lNodeA = MultiTensorValue(
            lScope, lInitializer, sTensorShape( std::vector<std::vector<uint32_t>>{ lDim1, lDim2 }, sizeof( float ) ) );
        auto lResult0 = Collapse( lScope, lNodeA );

        REQUIRE( lResult0.Get<sTypeComponent>().mValue == lNodeA.Get<sTypeComponent>().mValue );
    }

    SECTION( "Collapsing multi-tensors gives the correct dimension" )
    {
        auto lNodeA = MultiTensorValue(
            lScope, lInitializer, sTensorShape( std::vector<std::vector<uint32_t>>{ lDim1, lDim2, lDim3, lDim4 }, sizeof( float ) ) );
        auto lResult0 = Collapse( lScope, lNodeA );

        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().mRank == 3 );
        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().CountLayers() == 1 );
        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().mShape[0] == std::vector<uint32_t>{ 4, 23, 42 } );
    }

    SECTION( "Collapsing multi-tensors does not change values" )
    {
        auto lNodeA = MultiTensorValue(
            lScope, lInitializer, sTensorShape( std::vector<std::vector<uint32_t>>{ lDim1, lDim2, lDim3, lDim4 }, sizeof( float ) ) );
        auto lResult0 = Collapse( lScope, lNodeA );

        lScope.Run( lResult0 );

        std::vector<float> lTensorValues0 = lNodeA.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        std::vector<float> lTensorValues1 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        REQUIRE( VectorEqual( lTensorValues0, lTensorValues0 ) );
    }
}

TEST_CASE( "Reshape MultiTensors", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomUniformInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 21, 42 };
    std::vector<uint32_t> lDim2{ 25, 40 };
    std::vector<uint32_t> lDim3{ 14, 4 };

    SECTION( "Reshaping multi-tensors preserved types" )
    {
        auto lNodeA = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( float ) ) );

        std::vector<uint32_t> lODim1{ 7, 3, 42 };
        std::vector<uint32_t> lODim2{ 25, 8, 5 };
        std::vector<uint32_t> lODim3{ 7, 2, 4 };
        auto                  lResult0 = Reshape( lScope, lNodeA, sTensorShape( { lODim1, lODim2, lODim3 }, sizeof( float ) ) );

        REQUIRE( lResult0.Get<sTypeComponent>().mValue == lNodeA.Get<sTypeComponent>().mValue );
    }

    SECTION( "Reshaping multi-tensors gives the correct dimension" )
    {
        auto lNodeA = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( float ) ) );

        std::vector<uint32_t> lODim1{ 7, 3, 42 };
        std::vector<uint32_t> lODim2{ 25, 8, 5 };
        std::vector<uint32_t> lODim3{ 7, 2, 4 };

        auto lResult0 = Reshape( lScope, lNodeA, sTensorShape( { lODim1, lODim2, lODim3 }, sizeof( float ) ) );

        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().mRank == 3 );
        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().CountLayers() == 3 );
        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().mShape[0] == lODim1 );
        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().mShape[1] == lODim2 );
        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().mShape[2] == lODim3 );
    }

    SECTION( "Reshaping multi-tensors does not change values" )
    {
        auto lNodeA = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( float ) ) );

        std::vector<uint32_t> lODim1{ 7, 3, 42 };
        std::vector<uint32_t> lODim2{ 25, 8, 5 };
        std::vector<uint32_t> lODim3{ 7, 2, 4 };

        auto lResult0 = Reshape( lScope, lNodeA, sTensorShape( { lODim1, lODim2, lODim3 }, sizeof( float ) ) );

        std::vector<float> lTensorValues0 = lNodeA.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        std::vector<float> lTensorValues1 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        REQUIRE( VectorEqual( lTensorValues0, lTensorValues0 ) );
    }
}

TEST_CASE( "Flatten MultiTensors", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomUniformInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 21, 42 };
    std::vector<uint32_t> lDim2{ 25, 40 };
    std::vector<uint32_t> lDim3{ 14, 4 };

    SECTION( "Reshaping multi-tensors preserved types" )
    {
        auto lNodeA   = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
        auto lResult0 = Flatten( lScope, lNodeA );

        REQUIRE( lResult0.Get<sTypeComponent>().mValue == lNodeA.Get<sTypeComponent>().mValue );
    }

    SECTION( "Flattening multi-tensors gives the correct dimension" )
    {
        auto lNodeA = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( float ) ) );

        std::vector<uint32_t> lODim1{ 7, 3, 42 };
        std::vector<uint32_t> lODim2{ 25, 8, 5 };
        std::vector<uint32_t> lODim3{ 7, 2, 4 };

        auto lResult0 = Flatten( lScope, lNodeA );

        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().mRank == 1 );
        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().CountLayers() == 3 );
        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().mShape[0][0] == Prod( lODim1 ) );
        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().mShape[1][0] == Prod( lODim2 ) );
        REQUIRE( lResult0.Get<sMultiTensorComponent>().mValue.Shape().mShape[2][0] == Prod( lODim3 ) );
    }

    SECTION( "Flattening multi-tensors does not change values" )
    {
        auto lNodeA = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( float ) ) );

        std::vector<uint32_t> lODim1{ 7, 3, 42 };
        std::vector<uint32_t> lODim2{ 25, 8, 5 };
        std::vector<uint32_t> lODim3{ 7, 2, 4 };

        auto lResult0 = Flatten( lScope, lNodeA );

        std::vector<float> lTensorValues0 = lNodeA.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        std::vector<float> lTensorValues1 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
        REQUIRE( VectorEqual( lTensorValues0, lTensorValues0 ) );
    }
}

TEST_CASE( "Scope operation names", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    std::vector<uint32_t> lValue = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    auto                  lNode  = VectorValue<uint32_t>( lScope.WithOpName( "Node_1" ), lValue );
    REQUIRE( lNode == lScope["Node_1"] );
}

TEST_CASE( "InInterval Tensor_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    auto lOpNode     = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
    auto lLowerBound = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
    auto lUpperBound = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
    auto lResult0    = InInterval( lScope, lOpNode, lLowerBound, lUpperBound, false, false );

    lScope.Run( lResult0 );

    std::vector<float> lXValues          = lOpNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    std::vector<float> lLowerBoundValues = lLowerBound.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    std::vector<float> lUpperBoundValues = lUpperBound.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();

    std::vector<uint8_t> lResultValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> lExpectedValues( lOpNode.Get<sMultiTensorComponent>().mValue.SizeAs<float>() );

    for( uint32_t i = 0; i < lExpectedValues.size(); i++ )
    {
        lExpectedValues[i] = ( lLowerBoundValues[i] <= lXValues[i] ) && ( lXValues[i] <= lUpperBoundValues[i] );
    }
    REQUIRE( lResultValues == lExpectedValues );
}

TEST_CASE( "InInterval Tensor_Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    auto lOpNode     = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
    auto lLowerBound = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

    std::vector<ScalarValue> lConstants{ 0.2345f, 0.345f };
    auto                     lUpperBound = VectorValue( lScope, lConstants );
    auto                     lResult0    = InInterval( lScope, lOpNode, lLowerBound, lUpperBound, false, false );

    lScope.Run( lResult0 );

    std::vector<float>   lXValues0          = lOpNode.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
    std::vector<float>   lLowerBoundValues0 = lLowerBound.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
    std::vector<uint8_t> lResultValues0     = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<uint8_t>( 0 );
    std::vector<uint8_t> lExpectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < lExpectedValues0.size(); i++ )
        lExpectedValues0[i] = ( lLowerBoundValues0[i] <= lXValues0[i] ) && ( lXValues0[i] <= std::get<float>( lConstants[0] ) );
    REQUIRE( lResultValues0 == lExpectedValues0 );

    std::vector<float>   lXValues1          = lOpNode.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
    std::vector<float>   lLowerBoundValues1 = lLowerBound.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
    std::vector<uint8_t> lResultValues1     = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<uint8_t>( 1 );
    std::vector<uint8_t> lExpectedValues1( lResultValues1.size() );
    for( uint32_t i = 0; i < lExpectedValues1.size(); i++ )
        lExpectedValues1[i] = ( lLowerBoundValues1[i] <= lXValues1[i] ) && ( lXValues1[i] <= std::get<float>( lConstants[1] ) );
    REQUIRE( lResultValues1 == lExpectedValues1 );
}

TEST_CASE( "InInterval Tensor_Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    auto lOpNode     = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
    auto lLowerBound = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

    auto lUpperBound = ConstantScalarValue( lScope, 0.245f );
    auto lResult0    = InInterval( lScope, lOpNode, lLowerBound, lUpperBound, false, false );

    lScope.Run( lResult0 );

    std::vector<float>   lXValues0          = lOpNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    std::vector<float>   lLowerBoundValues0 = lLowerBound.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    std::vector<uint8_t> lResultValues0     = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> lExpectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < lExpectedValues0.size(); i++ )
        lExpectedValues0[i] = ( lLowerBoundValues0[i] <= lXValues0[i] ) && ( lXValues0[i] <= 0.245f );
    REQUIRE( lResultValues0 == lExpectedValues0 );
}

TEST_CASE( "InInterval Vector_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    auto lOpNode     = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
    auto lUpperBound = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

    std::vector<ScalarValue> lConstants{ 0.2345f, 0.345f };
    auto                     lLowerBound = VectorValue( lScope, lConstants );
    auto                     lResult0    = InInterval( lScope, lOpNode, lLowerBound, lUpperBound, false, false );

    lScope.Run( lResult0 );

    std::vector<float>   lXValues0          = lOpNode.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
    std::vector<float>   lUpperBoundValues0 = lUpperBound.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
    std::vector<uint8_t> lResultValues0     = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<uint8_t>( 0 );
    std::vector<uint8_t> lExpectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < lExpectedValues0.size(); i++ )
        lExpectedValues0[i] = ( std::get<float>( lConstants[0] ) <= lXValues0[i] ) && ( lXValues0[i] <= lUpperBoundValues0[i] );
    REQUIRE( lResultValues0 == lExpectedValues0 );

    std::vector<float>   lXValues1          = lOpNode.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
    std::vector<float>   lUpperBoundValues1 = lUpperBound.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
    std::vector<uint8_t> lResultValues1     = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<uint8_t>( 1 );
    std::vector<uint8_t> lExpectedValues1( lResultValues1.size() );
    for( uint32_t i = 0; i < lExpectedValues1.size(); i++ )
        lExpectedValues1[i] = ( std::get<float>( lConstants[1] ) <= lXValues1[i] ) && ( lXValues1[i] <= lUpperBoundValues1[i] );
    REQUIRE( lResultValues1 == lExpectedValues1 );
}

TEST_CASE( "InInterval Scalar_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    auto lOpNode     = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
    auto lUpperBound = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

    auto lLowerBound = ConstantScalarValue( lScope, 0.245f );
    auto lResult0    = InInterval( lScope, lOpNode, lLowerBound, lUpperBound, false, false );

    lScope.Run( lResult0 );

    std::vector<float>   lXValues0          = lOpNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    std::vector<float>   lUpperBoundValues0 = lUpperBound.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    std::vector<uint8_t> lResultValues0     = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> lExpectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < lExpectedValues0.size(); i++ )
        lExpectedValues0[i] = ( 0.245f <= lXValues0[i] ) && ( lXValues0[i] <= lUpperBoundValues0[i] );
    REQUIRE( lResultValues0 == lExpectedValues0 );
}

TEST_CASE( "InInterval Scalar_Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    auto lOpNode     = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
    auto lLowerBound = ConstantScalarValue( lScope, 0.245f );
    auto lUpperBound = ConstantScalarValue( lScope, 0.75f );

    auto lResult0 = InInterval( lScope, lOpNode, lLowerBound, lUpperBound, false, false );

    lScope.Run( lResult0 );

    std::vector<float>   lXValues0      = lOpNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    std::vector<uint8_t> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> lExpectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < lExpectedValues0.size(); i++ )
        lExpectedValues0[i] = ( 0.245f <= lXValues0[i] ) && ( lXValues0[i] <= 0.75f );
    REQUIRE( lResultValues0 == lExpectedValues0 );
}

TEST_CASE( "LessThan Tensor_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    auto lX = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
    auto lY = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

    auto lResult0 = LessThan( lScope, lX, lY );
    lScope.Run( lResult0 );

    std::vector<float> lXValues = lX.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    std::vector<float> lYValues = lY.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();

    std::vector<uint8_t> lResultValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> lExpectedValues( lResultValues.size() );

    for( uint32_t i = 0; i < lExpectedValues.size(); i++ )
    {
        lExpectedValues[i] = ( lXValues[i] < lYValues[i] );
    }
    REQUIRE( lResultValues == lExpectedValues );
}

TEST_CASE( "LessThan Tensor_Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    auto lX = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

    std::vector<ScalarValue> lConstants{ 0.2345f, 0.345f };
    auto                     lY       = VectorValue( lScope, lConstants );
    auto                     lResult0 = LessThan( lScope, lX, lY );

    lScope.Run( lResult0 );

    std::vector<float>   lXValues0      = lX.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
    std::vector<uint8_t> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<uint8_t>( 0 );
    std::vector<uint8_t> lExpectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < lExpectedValues0.size(); i++ ) lExpectedValues0[i] = ( lXValues0[i] < std::get<float>( lConstants[0] ) );
    REQUIRE( lResultValues0 == lExpectedValues0 );

    std::vector<float>   lXValues1      = lX.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
    std::vector<uint8_t> lResultValues1 = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<uint8_t>( 1 );
    std::vector<uint8_t> lExpectedValues1( lResultValues1.size() );
    for( uint32_t i = 0; i < lExpectedValues1.size(); i++ ) lExpectedValues1[i] = ( lXValues1[i] < std::get<float>( lConstants[1] ) );
    REQUIRE( lResultValues1 == lExpectedValues1 );
}

TEST_CASE( "LessThan Tensor_Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    auto lX = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
    auto lY = ConstantScalarValue( lScope, 0.245f );

    auto lResult0 = LessThan( lScope, lX, lY );

    lScope.Run( lResult0 );

    std::vector<float>   lXValues0      = lX.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    std::vector<uint8_t> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> lExpectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < lExpectedValues0.size(); i++ ) lExpectedValues0[i] = ( lXValues0[i] < 0.245f );
    REQUIRE( lResultValues0 == lExpectedValues0 );
}

TEST_CASE( "LessThan Vector_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    std::vector<ScalarValue> lConstants{ 0.2345f, 0.345f };
    auto                     lX = VectorValue( lScope, lConstants );
    auto                     lY = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

    auto lResult0 = LessThan( lScope, lX, lY );

    lScope.Run( lResult0 );

    std::vector<float>   lYValues0      = lY.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
    std::vector<uint8_t> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<uint8_t>( 0 );
    std::vector<uint8_t> lExpectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < lExpectedValues0.size(); i++ ) lExpectedValues0[i] = ( std::get<float>( lConstants[0] ) < lYValues0[i] );
    REQUIRE( lResultValues0 == lExpectedValues0 );

    std::vector<float>   lYValues1      = lY.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
    std::vector<uint8_t> lResultValues1 = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<uint8_t>( 1 );
    std::vector<uint8_t> lExpectedValues1( lResultValues1.size() );
    for( uint32_t i = 0; i < lExpectedValues1.size(); i++ ) lExpectedValues1[i] = ( std::get<float>( lConstants[1] ) < lYValues1[i] );
    REQUIRE( lResultValues1 == lExpectedValues1 );
}

TEST_CASE( "LessThan Scalar_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    auto lX = ConstantScalarValue( lScope, 0.245f );
    auto lY = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

    auto lResult0 = LessThan( lScope, lX, lY );

    lScope.Run( lResult0 );

    std::vector<float>   lYValues0      = lY.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    std::vector<uint8_t> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> lExpectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < lExpectedValues0.size(); i++ ) lExpectedValues0[i] = ( 0.245f < lYValues0[i] );
    REQUIRE( lResultValues0 == lExpectedValues0 );
}

TEST_CASE( "LessThanOrEqual Tensor_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    auto lX = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
    auto lY = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

    auto lResult0 = LessThanOrEqual( lScope, lX, lY );
    lScope.Run( lResult0 );

    std::vector<float> lXValues = lX.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    std::vector<float> lYValues = lY.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();

    std::vector<uint8_t> lResultValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> lExpectedValues( lResultValues.size() );

    for( uint32_t i = 0; i < lExpectedValues.size(); i++ )
    {
        lExpectedValues[i] = ( lXValues[i] <= lYValues[i] );
    }
    REQUIRE( lResultValues == lExpectedValues );
}

TEST_CASE( "LessThanOrEqual Tensor_Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    auto lX = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

    std::vector<ScalarValue> lConstants{ 0.2345f, 0.345f };
    auto                     lY       = VectorValue( lScope, lConstants );
    auto                     lResult0 = LessThanOrEqual( lScope, lX, lY );

    lScope.Run( lResult0 );

    std::vector<float>   lXValues0      = lX.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
    std::vector<uint8_t> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<uint8_t>( 0 );
    std::vector<uint8_t> lExpectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < lExpectedValues0.size(); i++ ) lExpectedValues0[i] = ( lXValues0[i] <= std::get<float>( lConstants[0] ) );
    REQUIRE( lResultValues0 == lExpectedValues0 );

    std::vector<float>   lXValues1      = lX.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
    std::vector<uint8_t> lResultValues1 = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<uint8_t>( 1 );
    std::vector<uint8_t> lExpectedValues1( lResultValues1.size() );
    for( uint32_t i = 0; i < lExpectedValues1.size(); i++ ) lExpectedValues1[i] = ( lXValues1[i] <= std::get<float>( lConstants[1] ) );
    REQUIRE( lResultValues1 == lExpectedValues1 );
}

TEST_CASE( "LessThanOrEqual Tensor_Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    auto lX = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
    auto lY = ConstantScalarValue( lScope, 0.245f );

    auto lResult0 = LessThanOrEqual( lScope, lX, lY );

    lScope.Run( lResult0 );

    std::vector<float>   lXValues0      = lX.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    std::vector<uint8_t> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> lExpectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < lExpectedValues0.size(); i++ ) lExpectedValues0[i] = ( lXValues0[i] <= 0.245f );
    REQUIRE( lResultValues0 == lExpectedValues0 );
}

TEST_CASE( "LessThanOrEqual Vector_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    std::vector<ScalarValue> lConstants{ 0.2345f, 0.345f };
    auto                     lX = VectorValue( lScope, lConstants );
    auto                     lY = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

    auto lResult0 = LessThanOrEqual( lScope, lX, lY );

    lScope.Run( lResult0 );

    std::vector<float>   lYValues0      = lY.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
    std::vector<uint8_t> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<uint8_t>( 0 );
    std::vector<uint8_t> lExpectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < lExpectedValues0.size(); i++ ) lExpectedValues0[i] = ( std::get<float>( lConstants[0] ) <= lYValues0[i] );
    REQUIRE( lResultValues0 == lExpectedValues0 );

    std::vector<float>   lYValues1      = lY.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
    std::vector<uint8_t> lResultValues1 = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<uint8_t>( 1 );
    std::vector<uint8_t> lExpectedValues1( lResultValues1.size() );
    for( uint32_t i = 0; i < lExpectedValues1.size(); i++ ) lExpectedValues1[i] = ( std::get<float>( lConstants[1] ) <= lYValues1[i] );
    REQUIRE( lResultValues1 == lExpectedValues1 );
}

TEST_CASE( "LessThanOrEqual Scalar_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    auto lX = ConstantScalarValue( lScope, 0.245f );
    auto lY = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

    auto lResult0 = LessThanOrEqual( lScope, lX, lY );

    lScope.Run( lResult0 );

    std::vector<float>   lYValues0      = lY.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    std::vector<uint8_t> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> lExpectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < lExpectedValues0.size(); i++ ) lExpectedValues0[i] = ( 0.245f <= lYValues0[i] );
    REQUIRE( lResultValues0 == lExpectedValues0 );
}

TEST_CASE( "GreaterThan Tensor_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    auto lX = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
    auto lY = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

    auto lResult0 = GreaterThan( lScope, lX, lY );
    lScope.Run( lResult0 );

    std::vector<float> lXValues = lX.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    std::vector<float> lYValues = lY.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();

    std::vector<uint8_t> lResultValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> lExpectedValues( lResultValues.size() );

    for( uint32_t i = 0; i < lExpectedValues.size(); i++ )
    {
        lExpectedValues[i] = ( lXValues[i] > lYValues[i] );
    }
    REQUIRE( lResultValues == lExpectedValues );
}

TEST_CASE( "GreaterThan Tensor_Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    auto lX = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

    std::vector<ScalarValue> lConstants{ 0.2345f, 0.345f };
    auto                     lY       = VectorValue( lScope, lConstants );
    auto                     lResult0 = GreaterThan( lScope, lX, lY );

    lScope.Run( lResult0 );

    std::vector<float>   lXValues0      = lX.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
    std::vector<uint8_t> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<uint8_t>( 0 );
    std::vector<uint8_t> lExpectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < lExpectedValues0.size(); i++ ) lExpectedValues0[i] = ( lXValues0[i] > std::get<float>( lConstants[0] ) );
    REQUIRE( lResultValues0 == lExpectedValues0 );

    std::vector<float>   lXValues1      = lX.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
    std::vector<uint8_t> lResultValues1 = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<uint8_t>( 1 );
    std::vector<uint8_t> lExpectedValues1( lResultValues1.size() );
    for( uint32_t i = 0; i < lExpectedValues1.size(); i++ ) lExpectedValues1[i] = ( lXValues1[i] > std::get<float>( lConstants[1] ) );
    REQUIRE( lResultValues1 == lExpectedValues1 );
}

TEST_CASE( "GreaterThan Tensor_Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    auto lX = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
    auto lY = ConstantScalarValue( lScope, 0.245f );

    auto lResult0 = GreaterThan( lScope, lX, lY );

    lScope.Run( lResult0 );

    std::vector<float>   lXValues0      = lX.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    std::vector<uint8_t> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> lExpectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < lExpectedValues0.size(); i++ ) lExpectedValues0[i] = ( lXValues0[i] > 0.245f );
    REQUIRE( lResultValues0 == lExpectedValues0 );
}

TEST_CASE( "GreaterThan Vector_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    std::vector<ScalarValue> lConstants{ 0.2345f, 0.345f };
    auto                     lX = VectorValue( lScope, lConstants );
    auto                     lY = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

    auto lResult0 = GreaterThan( lScope, lX, lY );

    lScope.Run( lResult0 );

    std::vector<float>   lYValues0      = lY.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
    std::vector<uint8_t> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<uint8_t>( 0 );
    std::vector<uint8_t> lExpectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < lExpectedValues0.size(); i++ ) lExpectedValues0[i] = ( std::get<float>( lConstants[0] ) > lYValues0[i] );
    REQUIRE( lResultValues0 == lExpectedValues0 );

    std::vector<float>   lYValues1      = lY.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
    std::vector<uint8_t> lResultValues1 = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<uint8_t>( 1 );
    std::vector<uint8_t> lExpectedValues1( lResultValues1.size() );
    for( uint32_t i = 0; i < lExpectedValues1.size(); i++ ) lExpectedValues1[i] = ( std::get<float>( lConstants[1] ) > lYValues1[i] );
    REQUIRE( lResultValues1 == lExpectedValues1 );
}

TEST_CASE( "GreaterThan Scalar_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    auto lX = ConstantScalarValue( lScope, 0.245f );
    auto lY = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

    auto lResult0 = GreaterThan( lScope, lX, lY );

    lScope.Run( lResult0 );

    std::vector<float>   lYValues0      = lY.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    std::vector<uint8_t> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> lExpectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < lExpectedValues0.size(); i++ ) lExpectedValues0[i] = ( 0.245f > lYValues0[i] );
    REQUIRE( lResultValues0 == lExpectedValues0 );
}

TEST_CASE( "GreaterThanOrEqual Tensor_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    auto lX = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
    auto lY = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

    auto lResult0 = GreaterThanOrEqual( lScope, lX, lY );
    lScope.Run( lResult0 );

    std::vector<float> lXValues = lX.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    std::vector<float> lYValues = lY.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();

    std::vector<uint8_t> lResultValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> lExpectedValues( lResultValues.size() );

    for( uint32_t i = 0; i < lExpectedValues.size(); i++ )
    {
        lExpectedValues[i] = ( lXValues[i] >= lYValues[i] );
    }
    REQUIRE( lResultValues == lExpectedValues );
}

TEST_CASE( "GreaterThanOrEqual Tensor_Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    auto lX = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

    std::vector<ScalarValue> lConstants{ 0.2345f, 0.345f };
    auto                     lY       = VectorValue( lScope, lConstants );
    auto                     lResult0 = GreaterThanOrEqual( lScope, lX, lY );

    lScope.Run( lResult0 );

    std::vector<float>   lXValues0      = lX.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
    std::vector<uint8_t> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<uint8_t>( 0 );
    std::vector<uint8_t> lExpectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < lExpectedValues0.size(); i++ ) lExpectedValues0[i] = ( lXValues0[i] >= std::get<float>( lConstants[0] ) );
    REQUIRE( lResultValues0 == lExpectedValues0 );

    std::vector<float>   lXValues1      = lX.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
    std::vector<uint8_t> lResultValues1 = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<uint8_t>( 1 );
    std::vector<uint8_t> lExpectedValues1( lResultValues1.size() );
    for( uint32_t i = 0; i < lExpectedValues1.size(); i++ ) lExpectedValues1[i] = ( lXValues1[i] >= std::get<float>( lConstants[1] ) );
    REQUIRE( lResultValues1 == lExpectedValues1 );
}

TEST_CASE( "GreaterThanOrEqual Tensor_Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    auto lX = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
    auto lY = ConstantScalarValue( lScope, 0.245f );

    auto lResult0 = GreaterThanOrEqual( lScope, lX, lY );

    lScope.Run( lResult0 );

    std::vector<float>   lXValues0      = lX.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    std::vector<uint8_t> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> lExpectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < lExpectedValues0.size(); i++ ) lExpectedValues0[i] = ( lXValues0[i] >= 0.245f );
    REQUIRE( lResultValues0 == lExpectedValues0 );
}

TEST_CASE( "GreaterThanOrEqual Vector_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    std::vector<ScalarValue> lConstants{ 0.2345f, 0.345f };
    auto                     lX = VectorValue( lScope, lConstants );
    auto                     lY = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

    auto lResult0 = GreaterThanOrEqual( lScope, lX, lY );

    lScope.Run( lResult0 );

    std::vector<float>   lYValues0      = lY.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
    std::vector<uint8_t> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<uint8_t>( 0 );
    std::vector<uint8_t> lExpectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < lExpectedValues0.size(); i++ ) lExpectedValues0[i] = ( std::get<float>( lConstants[0] ) >= lYValues0[i] );
    REQUIRE( lResultValues0 == lExpectedValues0 );

    std::vector<float>   lYValues1      = lY.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
    std::vector<uint8_t> lResultValues1 = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<uint8_t>( 1 );
    std::vector<uint8_t> lExpectedValues1( lResultValues1.size() );
    for( uint32_t i = 0; i < lExpectedValues1.size(); i++ ) lExpectedValues1[i] = ( std::get<float>( lConstants[1] ) >= lYValues1[i] );
    REQUIRE( lResultValues1 == lExpectedValues1 );
}

TEST_CASE( "GreaterThanOrEqual Scalar_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    auto lX = ConstantScalarValue( lScope, 0.245f );
    auto lY = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

    auto lResult0 = GreaterThanOrEqual( lScope, lX, lY );

    lScope.Run( lResult0 );

    std::vector<float>   lYValues0      = lY.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    std::vector<uint8_t> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> lExpectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < lExpectedValues0.size(); i++ ) lExpectedValues0[i] = ( 0.245f >= lYValues0[i] );
    REQUIRE( lResultValues0 == lExpectedValues0 );
}

TEST_CASE( "Where Tensor_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    auto                      lValues0 = RandomBool( 12 * 23 + 13 * 24 );
    sDataInitializerComponent lInitializer0( lValues0 );
    auto lCondition = MultiTensorValue( lScope, lInitializer0, sTensorShape( { lDim1, lDim2 }, sizeof( uint8_t ) ) );

    auto lX = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
    auto lY = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

    auto lResult0 = Where( lScope, lCondition, lX, lY );
    lScope.Run( lResult0 );

    std::vector<float> lXValues = lX.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    std::vector<float> lYValues = lY.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();

    std::vector<float> lResultValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    std::vector<float> lExpectedValues( lResultValues.size() );

    for( uint32_t i = 0; i < lExpectedValues.size(); i++ )
    {
        lExpectedValues[i] = lValues0[i] ? lXValues[i] : lYValues[i];
    }
    REQUIRE( lResultValues == lExpectedValues );
}

TEST_CASE( "Where Tensor_Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    auto                 lValues00 = RandomBool( 12 * 23 );
    auto                 lValues01 = RandomBool( 13 * 24 );
    std::vector<uint8_t> lValues0;
    lValues0.insert( lValues0.end(), lValues00.begin(), lValues00.end() );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    sDataInitializerComponent lInitializer0( lValues0 );
    auto lCondition = MultiTensorValue( lScope, lInitializer0, sTensorShape( { lDim1, lDim2 }, sizeof( uint8_t ) ) );

    auto lX = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

    std::vector<ScalarValue> lConstants{ 0.27986534f, 0.31490728f };
    auto                     lY = VectorValue( lScope, lConstants );

    auto lResult0 = Where( lScope, lCondition, lX, lY );

    lScope.Run( lResult0 );

    std::vector<float> lXValues0      = lX.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
    std::vector<float> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
    std::vector<float> lExpectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < lExpectedValues0.size(); i++ )
        lExpectedValues0[i] = lValues00[i] ? lXValues0[i] : std::get<float>( lConstants[0] );
    REQUIRE( lResultValues0 == lExpectedValues0 );

    std::vector<float> lXValues1      = lX.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
    std::vector<float> lResultValues1 = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
    std::vector<float> lExpectedValues1( lResultValues1.size() );
    for( uint32_t i = 0; i < lExpectedValues1.size(); i++ )
        lExpectedValues1[i] = lValues01[i] ? lXValues1[i] : std::get<float>( lConstants[1] );
    REQUIRE( lResultValues1 == lExpectedValues1 );
}

TEST_CASE( "Where Tensor_Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    auto                      lValues0 = RandomBool( 12 * 23 + 13 * 24 );
    sDataInitializerComponent lInitializer0( lValues0 );
    auto lCondition = MultiTensorValue( lScope, lInitializer0, sTensorShape( { lDim1, lDim2 }, sizeof( uint8_t ) ) );

    auto lX = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );
    auto lY = ConstantScalarValue( lScope, 0.245f );

    auto lResult0 = Where( lScope, lCondition, lX, lY );

    lScope.Run( lResult0 );

    std::vector<float> lXValues0      = lX.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    std::vector<float> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    std::vector<float> lExpectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < lExpectedValues0.size(); i++ ) lExpectedValues0[i] = lValues0[i] ? lXValues0[i] : 0.245f;
    REQUIRE( lResultValues0 == lExpectedValues0 );
}

TEST_CASE( "Where Vector_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    auto lValues00 = RandomBool( 12 * 23 );
    auto lValues01 = RandomBool( 13 * 24 );

    std::vector<uint8_t> lValues0;
    lValues0.insert( lValues0.end(), lValues00.begin(), lValues00.end() );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    sDataInitializerComponent lInitializer0( lValues0 );

    auto lCondition = MultiTensorValue( lScope, lInitializer0, sTensorShape( { lDim1, lDim2 }, sizeof( uint8_t ) ) );

    std::vector<ScalarValue> lConstants{ 0.2345f, 0.345f };
    auto                     lX = VectorValue( lScope, lConstants );

    auto lY = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

    auto lResult0 = Where( lScope, lCondition, lX, lY );

    lScope.Run( lResult0 );

    std::vector<float> lXValues0      = lY.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
    std::vector<float> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
    std::vector<float> lExpectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < lExpectedValues0.size(); i++ )
        lExpectedValues0[i] = lValues00[i] ? std::get<float>( lConstants[0] ) : lXValues0[i];
    REQUIRE( lResultValues0 == lExpectedValues0 );

    std::vector<float> lXValues1      = lY.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
    std::vector<float> lResultValues1 = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
    std::vector<float> lExpectedValues1( lResultValues1.size() );
    for( uint32_t i = 0; i < lExpectedValues1.size(); i++ )
        lExpectedValues1[i] = lValues01[i] ? std::get<float>( lConstants[1] ) : lXValues1[i];
    REQUIRE( lResultValues1 == lExpectedValues1 );
}

TEST_CASE( "Where Vector_Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    auto                 lValues00 = RandomBool( 12 * 23 );
    auto                 lValues01 = RandomBool( 13 * 24 );
    std::vector<uint8_t> lValues0;
    lValues0.insert( lValues0.end(), lValues00.begin(), lValues00.end() );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    sDataInitializerComponent lInitializer0( lValues0 );

    auto lCondition = MultiTensorValue( lScope, lInitializer0, sTensorShape( { lDim1, lDim2 }, sizeof( uint8_t ) ) );

    std::vector<ScalarValue> lConstants0{ 0.2345f, 0.345f };
    auto                     lX = VectorValue( lScope, lConstants0 );

    std::vector<ScalarValue> lConstants1{ 0.26534f, 0.19048265f };
    auto                     lY = VectorValue( lScope, lConstants1 );

    auto lResult0 = Where( lScope, lCondition, lX, lY );

    lScope.Run( lResult0 );

    std::vector<float> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
    std::vector<float> lExpectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < lExpectedValues0.size(); i++ )
        lExpectedValues0[i] = lValues00[i] ? std::get<float>( lConstants0[0] ) : std::get<float>( lConstants1[0] );
    REQUIRE( lResultValues0 == lExpectedValues0 );

    std::vector<float> lResultValues1 = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
    std::vector<float> lExpectedValues1( lResultValues1.size() );
    for( uint32_t i = 0; i < lExpectedValues1.size(); i++ )
        lExpectedValues1[i] = lValues01[i] ? std::get<float>( lConstants0[1] ) : std::get<float>( lConstants1[1] );
    REQUIRE( lResultValues1 == lExpectedValues1 );
}

TEST_CASE( "Where Vector_Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    auto                 lValues00 = RandomBool( 12 * 23 );
    auto                 lValues01 = RandomBool( 13 * 24 );
    std::vector<uint8_t> lValues0;
    lValues0.insert( lValues0.end(), lValues00.begin(), lValues00.end() );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    sDataInitializerComponent lInitializer0( lValues0 );
    auto lCondition = MultiTensorValue( lScope, lInitializer0, sTensorShape( { lDim1, lDim2 }, sizeof( uint8_t ) ) );

    std::vector<ScalarValue> lConstants{ 0.2345f, 0.345f };
    auto                     lX = VectorValue( lScope, lConstants );
    auto                     lY = ConstantScalarValue( lScope, 0.245f );

    auto lResult0 = Where( lScope, lCondition, lX, lY );

    lScope.Run( lResult0 );

    std::vector<float> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
    std::vector<float> lExpectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < lExpectedValues0.size(); i++ )
        lExpectedValues0[i] = lValues00[i] ? std::get<float>( lConstants[0] ) : 0.245f;
    REQUIRE( lResultValues0 == lExpectedValues0 );

    std::vector<float> lResultValues1 = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
    std::vector<float> lExpectedValues1( lResultValues1.size() );
    for( uint32_t i = 0; i < lExpectedValues1.size(); i++ )
        lExpectedValues1[i] = lValues01[i] ? std::get<float>( lConstants[1] ) : 0.245f;
    REQUIRE( lResultValues1 == lExpectedValues1 );
}

TEST_CASE( "Where Scalar_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    auto                      lValues0 = RandomBool( 12 * 23 + 13 * 24 );
    sDataInitializerComponent lInitializer0( lValues0 );
    auto lCondition = MultiTensorValue( lScope, lInitializer0, sTensorShape( { lDim1, lDim2 }, sizeof( uint8_t ) ) );

    auto lX = ConstantScalarValue( lScope, 0.245f );
    auto lY = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( float ) ) );

    auto lResult0 = Where( lScope, lCondition, lX, lY );

    lScope.Run( lResult0 );

    std::vector<float> lYValues0      = lY.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    std::vector<float> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    std::vector<float> lExpectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < lExpectedValues0.size(); i++ ) lExpectedValues0[i] = lValues0[i] ? 0.245f : lYValues0[i];
    REQUIRE( lResultValues0 == lExpectedValues0 );
}

TEST_CASE( "Where Scalar_Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    auto                 lValues00 = RandomBool( 12 * 23 );
    auto                 lValues01 = RandomBool( 13 * 24 );
    std::vector<uint8_t> lValues0;
    lValues0.insert( lValues0.end(), lValues00.begin(), lValues00.end() );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    sDataInitializerComponent lInitializer0( lValues0 );
    auto lCondition = MultiTensorValue( lScope, lInitializer0, sTensorShape( { lDim1, lDim2 }, sizeof( uint8_t ) ) );

    auto lX = ConstantScalarValue( lScope, 0.245f );

    std::vector<ScalarValue> lConstants{ 0.2345f, 0.345f };
    auto                     lY = VectorValue( lScope, lConstants );

    auto lResult0 = Where( lScope, lCondition, lX, lY );

    lScope.Run( lResult0 );

    std::vector<float> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 0 );
    std::vector<float> lExpectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < lExpectedValues0.size(); i++ )
        lExpectedValues0[i] = lValues00[i] ? 0.245f : std::get<float>( lConstants[0] );
    REQUIRE( lResultValues0 == lExpectedValues0 );

    std::vector<float> lResultValues1 = lResult0.Get<sMultiTensorComponent>().mValue.FetchBufferAt<float>( 1 );
    std::vector<float> lExpectedValues1( lResultValues1.size() );
    for( uint32_t i = 0; i < lExpectedValues1.size(); i++ )
        lExpectedValues1[i] = lValues01[i] ? 0.245f : std::get<float>( lConstants[1] );
    REQUIRE( lResultValues1 == lExpectedValues1 );
}

TEST_CASE( "Where Scalar_Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 128 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 12, 23 };
    std::vector<uint32_t> lDim2{ 13, 24 };

    auto                      lValues0 = RandomBool( 12 * 23 + 13 * 24 );
    sDataInitializerComponent lInitializer0( lValues0 );
    auto lCondition = MultiTensorValue( lScope, lInitializer0, sTensorShape( { lDim1, lDim2 }, sizeof( uint8_t ) ) );

    auto lX = ConstantScalarValue( lScope, 0.9234587f );
    auto lY = ConstantScalarValue( lScope, 0.1324978f );

    auto lResult0 = Where( lScope, lCondition, lX, lY );

    lScope.Run( lResult0 );

    std::vector<float> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    std::vector<float> lExpectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < lExpectedValues0.size(); i++ ) lExpectedValues0[i] = lValues0[i] ? 0.9234587f : 0.1324978f;
    REQUIRE( lResultValues0 == lExpectedValues0 );
}

TEST_CASE( "ArraySlice VECTOR_VECTOR", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    auto lSliceStart = std::vector<uint32_t>{ 15, 27, 400 };
    auto lSliceEnd   = std::vector<uint32_t>{ 81, 59, 510 };

    std::vector<uint32_t> lDim1{ 7, 3, 1024 };
    std::vector<uint64_t> lValues1;
    std::vector<uint64_t> lExpectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 1024 );
            lValues1.insert( lValues1.end(), lY.begin(), lY.end() );
            lExpectedValues1.insert( lExpectedValues1.end(), lY.begin() + lSliceStart[0], lY.begin() + lSliceEnd[0] + 1 );
        }
    }

    std::vector<uint32_t> lDim2{ 2, 7, 256 };
    std::vector<uint64_t> lValues2;
    std::vector<uint64_t> lExpectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 256 );
            lValues2.insert( lValues2.end(), lY.begin(), lY.end() );
            lExpectedValues2.insert( lExpectedValues2.end(), lY.begin() + lSliceStart[1], lY.begin() + lSliceEnd[1] + 1 );
        }
    }

    std::vector<uint32_t> lDim3{ 3, 5, 512 };
    std::vector<uint64_t> lValues3;
    std::vector<uint64_t> lExpectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 512 );
            lValues3.insert( lValues3.end(), lY.begin(), lY.end() );
            lExpectedValues3.insert( lExpectedValues3.end(), lY.begin() + lSliceStart[2], lY.begin() + lSliceEnd[2] + 1 );
        }
    }

    std::vector<uint64_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    sDataInitializerComponent lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( lScope, lInputInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( uint64_t ) ) );

    auto lStart = VectorValue( lScope, lSliceStart );
    auto lEnd   = VectorValue( lScope, lSliceEnd );

    auto lResult0 = Slice( lScope, lInputTensor, lStart, lEnd );
    lScope.Run( lResult0 );

    auto lOutputShape = lResult0.Get<sMultiTensorComponent>().mValue.Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.mRank == 3 );
    REQUIRE( lOutputShape.mShape[0] == std::vector<uint32_t>{ 7, 3, lSliceEnd[0] - lSliceStart[0] + 1 } );
    REQUIRE( lOutputShape.mShape[1] == std::vector<uint32_t>{ 2, 7, lSliceEnd[1] - lSliceStart[1] + 1 } );
    REQUIRE( lOutputShape.mShape[2] == std::vector<uint32_t>{ 3, 5, lSliceEnd[2] - lSliceStart[2] + 1 } );

    std::vector<uint64_t> lExpectedValues =
        ConcatenateVectors( std::vector<std::vector<uint64_t>>{ lExpectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<uint64_t> lResultValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( lResultValues == lExpectedValues );
}

TEST_CASE( "ArraySlice SCALAR_VECTOR", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    uint32_t lSliceStart = 25;
    auto     lSliceEnd   = std::vector<uint32_t>{ 81, 59, 51 };

    std::vector<uint32_t> lDim1{ 7, 3, 1024 };
    std::vector<uint64_t> lValues1;
    std::vector<uint64_t> lExpectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 1024 );
            lValues1.insert( lValues1.end(), lY.begin(), lY.end() );
            lExpectedValues1.insert( lExpectedValues1.end(), lY.begin() + lSliceStart, lY.begin() + lSliceEnd[0] + 1 );
        }
    }

    std::vector<uint32_t> lDim2{ 2, 7, 256 };
    std::vector<uint64_t> lValues2;
    std::vector<uint64_t> lExpectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 256 );
            lValues2.insert( lValues2.end(), lY.begin(), lY.end() );
            lExpectedValues2.insert( lExpectedValues2.end(), lY.begin() + lSliceStart, lY.begin() + lSliceEnd[1] + 1 );
        }
    }

    std::vector<uint32_t> lDim3{ 3, 5, 512 };
    std::vector<uint64_t> lValues3;
    std::vector<uint64_t> lExpectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 512 );
            lValues3.insert( lValues3.end(), lY.begin(), lY.end() );
            lExpectedValues3.insert( lExpectedValues3.end(), lY.begin() + lSliceStart, lY.begin() + lSliceEnd[2] + 1 );
        }
    }

    std::vector<uint64_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    sDataInitializerComponent lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( lScope, lInputInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( uint64_t ) ) );

    auto lStart = ConstantScalarValue( lScope, lSliceStart );
    auto lEnd   = VectorValue( lScope, lSliceEnd );

    auto lResult0 = Slice( lScope, lInputTensor, lStart, lEnd );
    lScope.Run( lResult0 );

    auto lOutputShape = lResult0.Get<sMultiTensorComponent>().mValue.Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.mRank == 3 );
    REQUIRE( lOutputShape.mShape[0] == std::vector<uint32_t>{ 7, 3, lSliceEnd[0] - lSliceStart + 1 } );
    REQUIRE( lOutputShape.mShape[1] == std::vector<uint32_t>{ 2, 7, lSliceEnd[1] - lSliceStart + 1 } );
    REQUIRE( lOutputShape.mShape[2] == std::vector<uint32_t>{ 3, 5, lSliceEnd[2] - lSliceStart + 1 } );

    std::vector<uint64_t> lExpectedValues =
        ConcatenateVectors( std::vector<std::vector<uint64_t>>{ lExpectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<uint64_t> lResultValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( lResultValues == lExpectedValues );
}

TEST_CASE( "ArraySlice VECTOR_SCALAR", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    auto     lSliceStart = std::vector<uint32_t>{ 15, 27, 39 };
    uint32_t lSliceEnd   = 65;

    std::vector<uint32_t> lDim1{ 7, 3, 1024 };
    std::vector<uint64_t> lValues1;
    std::vector<uint64_t> lExpectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 1024 );
            lValues1.insert( lValues1.end(), lY.begin(), lY.end() );
            lExpectedValues1.insert( lExpectedValues1.end(), lY.begin() + lSliceStart[0], lY.begin() + lSliceEnd + 1 );
        }
    }

    std::vector<uint32_t> lDim2{ 2, 7, 256 };
    std::vector<uint64_t> lValues2;
    std::vector<uint64_t> lExpectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 256 );
            lValues2.insert( lValues2.end(), lY.begin(), lY.end() );
            lExpectedValues2.insert( lExpectedValues2.end(), lY.begin() + lSliceStart[1], lY.begin() + lSliceEnd + 1 );
        }
    }

    std::vector<uint32_t> lDim3{ 3, 5, 512 };
    std::vector<uint64_t> lValues3;
    std::vector<uint64_t> lExpectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 512 );
            lValues3.insert( lValues3.end(), lY.begin(), lY.end() );
            lExpectedValues3.insert( lExpectedValues3.end(), lY.begin() + lSliceStart[2], lY.begin() + lSliceEnd + 1 );
        }
    }

    std::vector<uint64_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    sDataInitializerComponent lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( lScope, lInputInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( uint64_t ) ) );

    auto lStart = VectorValue( lScope, lSliceStart );
    auto lEnd   = ConstantScalarValue( lScope, lSliceEnd );

    auto lResult0 = Slice( lScope, lInputTensor, lStart, lEnd );
    lScope.Run( lResult0 );

    auto lOutputShape = lResult0.Get<sMultiTensorComponent>().mValue.Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.mRank == 3 );
    REQUIRE( lOutputShape.mShape[0] == std::vector<uint32_t>{ 7, 3, lSliceEnd - lSliceStart[0] + 1 } );
    REQUIRE( lOutputShape.mShape[1] == std::vector<uint32_t>{ 2, 7, lSliceEnd - lSliceStart[1] + 1 } );
    REQUIRE( lOutputShape.mShape[2] == std::vector<uint32_t>{ 3, 5, lSliceEnd - lSliceStart[2] + 1 } );

    std::vector<uint64_t> lExpectedValues =
        ConcatenateVectors( std::vector<std::vector<uint64_t>>{ lExpectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<uint64_t> lResultValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( lResultValues == lExpectedValues );
}

TEST_CASE( "ArraySlice SCALAR_SCALAR", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    uint32_t lSliceStart = 15;
    uint32_t lSliceEnd   = 65;

    std::vector<uint32_t> lDim1{ 7, 3, 1024 };
    std::vector<uint64_t> lValues1;
    std::vector<uint64_t> lExpectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 1024 );
            lValues1.insert( lValues1.end(), lY.begin(), lY.end() );
            lExpectedValues1.insert( lExpectedValues1.end(), lY.begin() + lSliceStart, lY.begin() + lSliceEnd + 1 );
        }
    }

    std::vector<uint32_t> lDim2{ 2, 7, 256 };
    std::vector<uint64_t> lValues2;
    std::vector<uint64_t> lExpectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 256 );
            lValues2.insert( lValues2.end(), lY.begin(), lY.end() );
            lExpectedValues2.insert( lExpectedValues2.end(), lY.begin() + lSliceStart, lY.begin() + lSliceEnd + 1 );
        }
    }

    std::vector<uint32_t> lDim3{ 3, 5, 512 };
    std::vector<uint64_t> lValues3;
    std::vector<uint64_t> lExpectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 512 );
            lValues3.insert( lValues3.end(), lY.begin(), lY.end() );
            lExpectedValues3.insert( lExpectedValues3.end(), lY.begin() + lSliceStart, lY.begin() + lSliceEnd + 1 );
        }
    }

    std::vector<uint64_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    sDataInitializerComponent lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( lScope, lInputInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( uint64_t ) ) );

    auto lStart = ConstantScalarValue( lScope, lSliceStart );
    auto lEnd   = ConstantScalarValue( lScope, lSliceEnd );

    auto lResult0 = Slice( lScope, lInputTensor, lStart, lEnd );
    lScope.Run( lResult0 );

    auto lOutputShape = lResult0.Get<sMultiTensorComponent>().mValue.Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.mRank == 3 );
    REQUIRE( lOutputShape.mShape[0] == std::vector<uint32_t>{ 7, 3, lSliceEnd - lSliceStart + 1 } );
    REQUIRE( lOutputShape.mShape[1] == std::vector<uint32_t>{ 2, 7, lSliceEnd - lSliceStart + 1 } );
    REQUIRE( lOutputShape.mShape[2] == std::vector<uint32_t>{ 3, 5, lSliceEnd - lSliceStart + 1 } );

    std::vector<uint64_t> lExpectedValues =
        ConcatenateVectors( std::vector<std::vector<uint64_t>>{ lExpectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<uint64_t> lResultValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( lResultValues == lExpectedValues );
}

TEST_CASE( "ArraySummation VECTOR_VECTOR", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    auto lSliceStart = std::vector<uint32_t>{ 15, 27, 400 };
    auto lSliceEnd   = std::vector<uint32_t>{ 81, 59, 510 };

    std::vector<uint32_t> lDim1{ 7, 3, 1024 };
    std::vector<uint64_t> lValues1;
    std::vector<uint64_t> lExpectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 1024 );
            lValues1.insert( lValues1.end(), lY.begin(), lY.end() );
            lExpectedValues1.push_back( std::accumulate(
                lY.begin() + lSliceStart[0], lY.begin() + lSliceEnd[0] + 1, static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint32_t> lDim2{ 2, 7, 256 };
    std::vector<uint64_t> lValues2;
    std::vector<uint64_t> lExpectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 256 );
            lValues2.insert( lValues2.end(), lY.begin(), lY.end() );
            lExpectedValues2.push_back( std::accumulate(
                lY.begin() + lSliceStart[1], lY.begin() + lSliceEnd[1] + 1, static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint32_t> lDim3{ 3, 5, 512 };
    std::vector<uint64_t> lValues3;
    std::vector<uint64_t> lExpectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 512 );
            lValues3.insert( lValues3.end(), lY.begin(), lY.end() );
            lExpectedValues3.push_back( std::accumulate(
                lY.begin() + lSliceStart[2], lY.begin() + lSliceEnd[2] + 1, static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint64_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    sDataInitializerComponent lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( lScope, lInputInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( uint64_t ) ) );

    auto lStart = VectorValue( lScope, lSliceStart );
    auto lEnd   = VectorValue( lScope, lSliceEnd );

    auto lResult0 = Summation( lScope, lInputTensor, lStart, lEnd );
    lScope.Run( lResult0 );

    auto lOutputShape = lResult0.Get<sMultiTensorComponent>().mValue.Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.mRank == 2 );
    REQUIRE( lOutputShape.mShape[0] == std::vector<uint32_t>{ 7, 3 } );
    REQUIRE( lOutputShape.mShape[1] == std::vector<uint32_t>{ 2, 7 } );
    REQUIRE( lOutputShape.mShape[2] == std::vector<uint32_t>{ 3, 5 } );

    std::vector<uint64_t> lExpectedValues =
        ConcatenateVectors( std::vector<std::vector<uint64_t>>{ lExpectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<uint64_t> lResultValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( lResultValues == lExpectedValues );
}

TEST_CASE( "ArraySummation SCALAR_VECTOR", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    uint32_t lSliceStart = 25;
    auto     lSliceEnd   = std::vector<uint32_t>{ 81, 59, 51 };

    std::vector<uint32_t> lDim1{ 7, 3, 1024 };
    std::vector<uint64_t> lValues1;
    std::vector<uint64_t> lExpectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 1024 );
            lValues1.insert( lValues1.end(), lY.begin(), lY.end() );
            lExpectedValues1.push_back( std::accumulate(
                lY.begin() + lSliceStart, lY.begin() + lSliceEnd[0] + 1, static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint32_t> lDim2{ 2, 7, 256 };
    std::vector<uint64_t> lValues2;
    std::vector<uint64_t> lExpectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 256 );
            lValues2.insert( lValues2.end(), lY.begin(), lY.end() );
            lExpectedValues2.push_back( std::accumulate(
                lY.begin() + lSliceStart, lY.begin() + lSliceEnd[1] + 1, static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint32_t> lDim3{ 3, 5, 512 };
    std::vector<uint64_t> lValues3;
    std::vector<uint64_t> lExpectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 512 );
            lValues3.insert( lValues3.end(), lY.begin(), lY.end() );
            lExpectedValues3.push_back( std::accumulate(
                lY.begin() + lSliceStart, lY.begin() + lSliceEnd[2] + 1, static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint64_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    sDataInitializerComponent lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( lScope, lInputInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( uint64_t ) ) );

    auto lStart = ConstantScalarValue( lScope, lSliceStart );
    auto lEnd   = VectorValue( lScope, lSliceEnd );

    auto lResult0 = Summation( lScope, lInputTensor, lStart, lEnd );
    lScope.Run( lResult0 );

    auto lOutputShape = lResult0.Get<sMultiTensorComponent>().mValue.Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.mRank == 2 );
    REQUIRE( lOutputShape.mShape[0] == std::vector<uint32_t>{ 7, 3 } );
    REQUIRE( lOutputShape.mShape[1] == std::vector<uint32_t>{ 2, 7 } );
    REQUIRE( lOutputShape.mShape[2] == std::vector<uint32_t>{ 3, 5 } );

    std::vector<uint64_t> lExpectedValues =
        ConcatenateVectors( std::vector<std::vector<uint64_t>>{ lExpectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<uint64_t> lResultValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( lResultValues == lExpectedValues );
}

TEST_CASE( "ArraySummation VECTOR_SCALAR", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    auto     lSliceStart = std::vector<uint32_t>{ 15, 27, 39 };
    uint32_t lSliceEnd   = 65;

    std::vector<uint32_t> lDim1{ 7, 3, 1024 };
    std::vector<uint64_t> lValues1;
    std::vector<uint64_t> lExpectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 1024 );
            lValues1.insert( lValues1.end(), lY.begin(), lY.end() );
            lExpectedValues1.push_back( std::accumulate(
                lY.begin() + lSliceStart[0], lY.begin() + lSliceEnd + 1, static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint32_t> lDim2{ 2, 7, 256 };
    std::vector<uint64_t> lValues2;
    std::vector<uint64_t> lExpectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 256 );
            lValues2.insert( lValues2.end(), lY.begin(), lY.end() );
            lExpectedValues2.push_back( std::accumulate(
                lY.begin() + lSliceStart[1], lY.begin() + lSliceEnd + 1, static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint32_t> lDim3{ 3, 5, 512 };
    std::vector<uint64_t> lValues3;
    std::vector<uint64_t> lExpectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 512 );
            lValues3.insert( lValues3.end(), lY.begin(), lY.end() );
            lExpectedValues3.push_back( std::accumulate(
                lY.begin() + lSliceStart[2], lY.begin() + lSliceEnd + 1, static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint64_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    sDataInitializerComponent lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( lScope, lInputInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( uint64_t ) ) );

    auto lStart = VectorValue( lScope, lSliceStart );
    auto lEnd   = ConstantScalarValue( lScope, lSliceEnd );

    auto lResult0 = Summation( lScope, lInputTensor, lStart, lEnd );
    lScope.Run( lResult0 );

    auto lOutputShape = lResult0.Get<sMultiTensorComponent>().mValue.Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.mRank == 2 );
    REQUIRE( lOutputShape.mShape[0] == std::vector<uint32_t>{ 7, 3 } );
    REQUIRE( lOutputShape.mShape[1] == std::vector<uint32_t>{ 2, 7 } );
    REQUIRE( lOutputShape.mShape[2] == std::vector<uint32_t>{ 3, 5 } );

    std::vector<uint64_t> lExpectedValues =
        ConcatenateVectors( std::vector<std::vector<uint64_t>>{ lExpectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<uint64_t> lResultValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( lResultValues == lExpectedValues );
}

TEST_CASE( "ArraySummation SCALAR_SCALAR", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    uint32_t lSliceStart = 15;
    uint32_t lSliceEnd   = 65;

    std::vector<uint32_t> lDim1{ 7, 3, 1024 };
    std::vector<uint64_t> lValues1;
    std::vector<uint64_t> lExpectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 1024 );
            lValues1.insert( lValues1.end(), lY.begin(), lY.end() );
            lExpectedValues1.push_back( std::accumulate(
                lY.begin() + lSliceStart, lY.begin() + lSliceEnd + 1, static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint32_t> lDim2{ 2, 7, 256 };
    std::vector<uint64_t> lValues2;
    std::vector<uint64_t> lExpectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 256 );
            lValues2.insert( lValues2.end(), lY.begin(), lY.end() );
            lExpectedValues2.push_back( std::accumulate(
                lY.begin() + lSliceStart, lY.begin() + lSliceEnd + 1, static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint32_t> lDim3{ 3, 5, 512 };
    std::vector<uint64_t> lValues3;
    std::vector<uint64_t> lExpectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 512 );
            lValues3.insert( lValues3.end(), lY.begin(), lY.end() );
            lExpectedValues3.push_back( std::accumulate(
                lY.begin() + lSliceStart, lY.begin() + lSliceEnd + 1, static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint64_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    sDataInitializerComponent lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( lScope, lInputInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( uint64_t ) ) );

    auto lStart = ConstantScalarValue( lScope, lSliceStart );
    auto lEnd   = ConstantScalarValue( lScope, lSliceEnd );

    auto lResult0 = Summation( lScope, lInputTensor, lStart, lEnd );
    lScope.Run( lResult0 );

    auto lOutputShape = lResult0.Get<sMultiTensorComponent>().mValue.Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.mRank == 2 );
    REQUIRE( lOutputShape.mShape[0] == std::vector<uint32_t>{ 7, 3 } );
    REQUIRE( lOutputShape.mShape[1] == std::vector<uint32_t>{ 2, 7 } );
    REQUIRE( lOutputShape.mShape[2] == std::vector<uint32_t>{ 3, 5 } );

    std::vector<uint64_t> lExpectedValues =
        ConcatenateVectors( std::vector<std::vector<uint64_t>>{ lExpectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<uint64_t> lResultValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( lResultValues == lExpectedValues );
}

TEST_CASE( "ArraySummation full", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    uint32_t lSliceStart = 15;
    uint32_t lSliceEnd   = 65;

    std::vector<uint32_t> lDim1{ 7, 3, 1024 };
    std::vector<uint64_t> lValues1;
    std::vector<uint64_t> lExpectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 1024 );
            lValues1.insert( lValues1.end(), lY.begin(), lY.end() );
            lExpectedValues1.push_back( std::accumulate( lY.begin(), lY.end(), static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint32_t> lDim2{ 2, 7, 256 };
    std::vector<uint64_t> lValues2;
    std::vector<uint64_t> lExpectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 256 );
            lValues2.insert( lValues2.end(), lY.begin(), lY.end() );
            lExpectedValues2.push_back( std::accumulate( lY.begin(), lY.end(), static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint32_t> lDim3{ 3, 5, 512 };
    std::vector<uint64_t> lValues3;
    std::vector<uint64_t> lExpectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 512 );
            lValues3.insert( lValues3.end(), lY.begin(), lY.end() );
            lExpectedValues3.push_back( std::accumulate( lY.begin(), lY.end(), static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint64_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    sDataInitializerComponent lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( lScope, lInputInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( uint64_t ) ) );

    auto lStart = ConstantScalarValue( lScope, lSliceStart );
    auto lEnd   = ConstantScalarValue( lScope, lSliceEnd );

    auto lResult0 = Summation( lScope, lInputTensor );
    lScope.Run( lResult0 );

    auto lOutputShape = lResult0.Get<sMultiTensorComponent>().mValue.Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.mRank == 2 );
    REQUIRE( lOutputShape.mShape[0] == std::vector<uint32_t>{ 7, 3 } );
    REQUIRE( lOutputShape.mShape[1] == std::vector<uint32_t>{ 2, 7 } );
    REQUIRE( lOutputShape.mShape[2] == std::vector<uint32_t>{ 3, 5 } );

    std::vector<uint64_t> lExpectedValues =
        ConcatenateVectors( std::vector<std::vector<uint64_t>>{ lExpectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<uint64_t> lResultValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( lResultValues == lExpectedValues );
}

TEST_CASE( "CountTrue", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 7, 3, 1024 };
    std::vector<uint8_t>  lValues1;
    std::vector<uint32_t> lExpectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto lY = RandomBool( 1024 );
            lValues1.insert( lValues1.end(), lY.begin(), lY.end() );

            uint32_t lTrueCount = 0;
            for( auto &x : lY ) lTrueCount += x ? 1 : 0;
            lExpectedValues1.push_back( lTrueCount );
        }
    }

    std::vector<uint32_t> lDim2{ 2, 7, 256 };
    std::vector<uint8_t>  lValues2;
    std::vector<uint32_t> lExpectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto lY = RandomBool( 256 );
            lValues2.insert( lValues2.end(), lY.begin(), lY.end() );
            uint32_t lTrueCount = 0;
            for( auto &x : lY ) lTrueCount += x ? 1 : 0;
            lExpectedValues2.push_back( lTrueCount );
        }
    }

    std::vector<uint32_t> lDim3{ 3, 5, 512 };
    std::vector<uint8_t>  lValues3;
    std::vector<uint32_t> lExpectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto lY = RandomBool( 512 );
            lValues3.insert( lValues3.end(), lY.begin(), lY.end() );
            uint32_t lTrueCount = 0;
            for( auto &x : lY ) lTrueCount += x ? 1 : 0;
            lExpectedValues3.push_back( lTrueCount );
        }
    }

    std::vector<uint8_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    sDataInitializerComponent lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( lScope, lInputInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( uint8_t ) ) );

    auto lResult0 = CountTrue( lScope, lInputTensor );
    lScope.Run( lResult0 );

    auto lOutputShape = lResult0.Get<sMultiTensorComponent>().mValue.Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.mRank == 2 );
    REQUIRE( lOutputShape.mShape[0] == std::vector<uint32_t>{ 7, 3 } );
    REQUIRE( lOutputShape.mShape[1] == std::vector<uint32_t>{ 2, 7 } );
    REQUIRE( lOutputShape.mShape[2] == std::vector<uint32_t>{ 3, 5 } );

    std::vector<uint32_t> lExpectedValues =
        ConcatenateVectors( std::vector<std::vector<uint32_t>>{ lExpectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<uint32_t> lResultValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint32_t>();
    REQUIRE( lResultValues == lExpectedValues );
}

TEST_CASE( "CountNonZero", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 7, 3, 1024 };
    std::vector<uint64_t> lValues1;
    std::vector<uint32_t> lExpectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 1024 );
            lValues1.insert( lValues1.end(), lY.begin(), lY.end() );

            uint32_t lTrueCount = 0;
            for( auto &x : lY ) lTrueCount += ( x != 0 ) ? 1 : 0;
            lExpectedValues1.push_back( lTrueCount );
        }
    }

    std::vector<uint32_t> lDim2{ 2, 7, 256 };
    std::vector<uint64_t> lValues2;
    std::vector<uint32_t> lExpectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 256 );
            lValues2.insert( lValues2.end(), lY.begin(), lY.end() );
            uint32_t lTrueCount = 0;
            for( auto &x : lY ) lTrueCount += ( x != 0 ) ? 1 : 0;
            lExpectedValues2.push_back( lTrueCount );
        }
    }

    std::vector<uint32_t> lDim3{ 3, 5, 512 };
    std::vector<uint64_t> lValues3;
    std::vector<uint32_t> lExpectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 512 );
            lValues3.insert( lValues3.end(), lY.begin(), lY.end() );
            uint32_t lTrueCount = 0;
            for( auto &x : lY ) lTrueCount += ( x != 0 ) ? 1 : 0;
            lExpectedValues3.push_back( lTrueCount );
        }
    }

    std::vector<uint64_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    sDataInitializerComponent lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( lScope, lInputInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( uint64_t ) ) );

    auto lResult0 = CountNonZero( lScope, lInputTensor );
    lScope.Run( lResult0 );

    auto lOutputShape = lResult0.Get<sMultiTensorComponent>().mValue.Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.mRank == 2 );
    REQUIRE( lOutputShape.mShape[0] == std::vector<uint32_t>{ 7, 3 } );
    REQUIRE( lOutputShape.mShape[1] == std::vector<uint32_t>{ 2, 7 } );
    REQUIRE( lOutputShape.mShape[2] == std::vector<uint32_t>{ 3, 5 } );

    std::vector<uint32_t> lExpectedValues =
        ConcatenateVectors( std::vector<std::vector<uint32_t>>{ lExpectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<uint32_t> lResultValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint32_t>();
    REQUIRE( lResultValues == lExpectedValues );
}

TEST_CASE( "CountZero", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 7, 3, 1024 };
    std::vector<uint64_t> lValues1;
    std::vector<uint32_t> lExpectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 1024 );
            lValues1.insert( lValues1.end(), lY.begin(), lY.end() );

            uint32_t lTrueCount = 0;
            for( auto &x : lY ) lTrueCount += ( x == 0 ) ? 1 : 0;
            lExpectedValues1.push_back( lTrueCount );
        }
    }

    std::vector<uint32_t> lDim2{ 2, 7, 256 };
    std::vector<uint64_t> lValues2;
    std::vector<uint32_t> lExpectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 256 );
            lValues2.insert( lValues2.end(), lY.begin(), lY.end() );
            uint32_t lTrueCount = 0;
            for( auto &x : lY ) lTrueCount += ( x == 0 ) ? 1 : 0;
            lExpectedValues2.push_back( lTrueCount );
        }
    }

    std::vector<uint32_t> lDim3{ 3, 5, 512 };
    std::vector<uint64_t> lValues3;
    std::vector<uint32_t> lExpectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto lY = RandomNumber<uint64_t>( 512 );
            lValues3.insert( lValues3.end(), lY.begin(), lY.end() );
            uint32_t lTrueCount = 0;
            for( auto &x : lY ) lTrueCount += ( x == 0 ) ? 1 : 0;
            lExpectedValues3.push_back( lTrueCount );
        }
    }

    std::vector<uint64_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    sDataInitializerComponent lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( lScope, lInputInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( uint64_t ) ) );

    auto lResult0 = CountZero( lScope, lInputTensor );
    lScope.Run( lResult0 );

    auto lOutputShape = lResult0.Get<sMultiTensorComponent>().mValue.Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.mRank == 2 );
    REQUIRE( lOutputShape.mShape[0] == std::vector<uint32_t>{ 7, 3 } );
    REQUIRE( lOutputShape.mShape[1] == std::vector<uint32_t>{ 2, 7 } );
    REQUIRE( lOutputShape.mShape[2] == std::vector<uint32_t>{ 3, 5 } );

    std::vector<uint32_t> lExpectedValues =
        ConcatenateVectors( std::vector<std::vector<uint32_t>>{ lExpectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<uint32_t> lResultValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint32_t>();
    REQUIRE( lResultValues == lExpectedValues );
}

TEST_CASE( "Floor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 7, 3, 500 };
    std::vector<uint32_t> lDim2{ 2, 7, 256 };
    std::vector<uint32_t> lDim3{ 3, 5, 512 };

    auto lOpNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( float ) ) );

    auto lResult0 = Floor( lScope, lOpNode );
    lScope.Run( lResult0 );

    std::vector<float> lResultValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    std::vector<float> lExpectedValues( lOpNode.Get<sMultiTensorComponent>().mValue.SizeAs<float>() );
    std::vector<float> lOpNodeValues = lOpNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    for( uint32_t i = 0; i < lOpNodeValues.size(); i++ )
    {
        lExpectedValues[i] = std::floor( lOpNodeValues[i] );
    }

    REQUIRE( lResultValues == lExpectedValues );
}

TEST_CASE( "Ceiling", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 7, 3, 500 };
    std::vector<uint32_t> lDim2{ 2, 7, 256 };
    std::vector<uint32_t> lDim3{ 3, 5, 512 };

    auto lOpNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( float ) ) );

    auto lResult0 = Ceil( lScope, lOpNode );
    lScope.Run( lResult0 );

    std::vector<float> lResultValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    std::vector<float> lExpectedValues( lOpNode.Get<sMultiTensorComponent>().mValue.SizeAs<float>() );
    std::vector<float> lOpNodeValues = lOpNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    for( uint32_t i = 0; i < lOpNodeValues.size(); i++ )
    {
        lExpectedValues[i] = std::ceil( lOpNodeValues[i] );
    }

    REQUIRE( lResultValues == lExpectedValues );
}

TEST_CASE( "Absolute value", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 7, 3, 500 };
    std::vector<uint32_t> lDim2{ 2, 7, 256 };
    std::vector<uint32_t> lDim3{ 3, 5, 512 };

    auto lOpNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( float ) ) );

    auto lResult0 = Abs( lScope, lOpNode );
    lScope.Run( lResult0 );

    std::vector<float> lResultValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    std::vector<float> lExpectedValues( lOpNode.Get<sMultiTensorComponent>().mValue.SizeAs<float>() );
    std::vector<float> lOpNodeValues = lOpNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    for( uint32_t i = 0; i < lOpNodeValues.size(); i++ )
    {
        lExpectedValues[i] = std::abs( lOpNodeValues[i] );
    }

    REQUIRE( lResultValues == lExpectedValues );
}

TEST_CASE( "Square roots", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 17, 12, 51 };
    std::vector<uint32_t> lDim2{ 12, 17, 23 };
    std::vector<uint32_t> lDim3{ 13, 15, 52 };

    auto lOpNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( float ) ) );

    auto lResult0 = Sqrt( lScope, lOpNode );
    lScope.Run( lResult0 );

    std::vector<float> lResultValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    std::vector<float> lExpectedValues( lOpNode.Get<sMultiTensorComponent>().mValue.SizeAs<float>() );
    std::vector<float> lOpNodeValues = lOpNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    for( uint32_t i = 0; i < lOpNodeValues.size(); i++ )
    {
        lExpectedValues[i] = std::sqrt( lOpNodeValues[i] );
    }

    REQUIRE( lResultValues.size() == lExpectedValues.size() );
    std::vector<bool> lComparison( lResultValues.size() );
    for( uint32_t i = 0; i < lOpNodeValues.size(); i++ )
    {
        lComparison[i] =
            ( std::isnan( lResultValues[i] ) && std::isnan( lExpectedValues[i] ) ) || ( lResultValues[i] == lExpectedValues[i] );
    }

    REQUIRE( std::all_of( lComparison.begin(), lComparison.end(), []( auto x ) { return x; } ) );
}

TEST_CASE( "Rounding", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 17, 12, 51 };
    std::vector<uint32_t> lDim2{ 12, 17, 23 };
    std::vector<uint32_t> lDim3{ 13, 15, 52 };

    auto lOpNode = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( float ) ) );

    auto lResult0 = Round( lScope, lOpNode );
    lScope.Run( lResult0 );

    std::vector<float> lResultValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    std::vector<float> lExpectedValues( lOpNode.Get<sMultiTensorComponent>().mValue.SizeAs<float>() );
    std::vector<float> lOpNodeValues = lOpNode.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    for( uint32_t i = 0; i < lOpNodeValues.size(); i++ )
    {
        lExpectedValues[i] = std::round( lOpNodeValues[i] );
    }

    REQUIRE( lResultValues.size() == lExpectedValues.size() );
    std::vector<bool> lComparison( lResultValues.size() );
    for( uint32_t i = 0; i < lOpNodeValues.size(); i++ )
    {
        lComparison[i] = ( lResultValues[i] == lExpectedValues[i] );
    }

    REQUIRE( std::all_of( lComparison.begin(), lComparison.end(), []( auto x ) { return x; } ) );
}

TEST_CASE( "Finite differences", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 7, 3, 1024 };
    std::vector<int64_t>  lValues1;
    std::vector<int64_t>  lExpectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto lY = RandomNumber<int64_t>( 1024 );
            lValues1.insert( lValues1.end(), lY.begin(), lY.end() );
            for( uint32_t l = 0; l < lY.size() - 1; l++ ) lExpectedValues1.push_back( lY[l + 1] - lY[l] );
            lExpectedValues1.push_back( static_cast<int64_t>( 0 ) );
        }
    }

    std::vector<uint32_t> lDim2{ 2, 7, 256 };
    std::vector<int64_t>  lValues2;
    std::vector<int64_t>  lExpectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto lY = RandomNumber<int64_t>( 256 );
            lValues2.insert( lValues2.end(), lY.begin(), lY.end() );
            for( uint32_t l = 0; l < lY.size() - 1; l++ ) lExpectedValues2.push_back( lY[l + 1] - lY[l] );
            lExpectedValues2.push_back( static_cast<int64_t>( 0 ) );
        }
    }

    std::vector<uint32_t> lDim3{ 3, 5, 512 };
    std::vector<int64_t>  lValues3;
    std::vector<int64_t>  lExpectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto lY = RandomNumber<int64_t>( 512 );
            lValues3.insert( lValues3.end(), lY.begin(), lY.end() );
            for( uint32_t l = 0; l < lY.size() - 1; l++ ) lExpectedValues3.push_back( lY[l + 1] - lY[l] );
            lExpectedValues3.push_back( static_cast<int64_t>( 0 ) );
        }
    }

    std::vector<int64_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    sDataInitializerComponent lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( lScope, lInputInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( int64_t ) ) );

    auto lResult0 = Diff( lScope, lInputTensor, 1 );
    lScope.Run( lResult0 );

    auto lOutputShape = lResult0.Get<sMultiTensorComponent>().mValue.Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.mRank == 3 );
    REQUIRE( lOutputShape.mShape[0] == std::vector<uint32_t>{ 7, 3, 1024 } );
    REQUIRE( lOutputShape.mShape[1] == std::vector<uint32_t>{ 2, 7, 256 } );
    REQUIRE( lOutputShape.mShape[2] == std::vector<uint32_t>{ 3, 5, 512 } );

    std::vector<int64_t> lExpectedValues =
        ConcatenateVectors( std::vector<std::vector<int64_t>>{ lExpectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<int64_t> lResultValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<int64_t>();
    REQUIRE( lResultValues == lExpectedValues );
}

TEST_CASE( "Finite shift to the left  by 1", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 7, 3, 1024 };
    std::vector<int64_t>  lValues1;
    std::vector<int64_t>  lExpectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto lY = RandomNumber<int64_t>( 1024 );
            lValues1.insert( lValues1.end(), lY.begin(), lY.end() );
            for( uint32_t l = 0; l < lY.size() - 1; l++ ) lExpectedValues1.push_back( lY[l + 1] );
            lExpectedValues1.push_back( static_cast<int64_t>( 121212 ) );
        }
    }

    std::vector<uint32_t> lDim2{ 2, 7, 256 };
    std::vector<int64_t>  lValues2;
    std::vector<int64_t>  lExpectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto lY = RandomNumber<int64_t>( 256 );
            lValues2.insert( lValues2.end(), lY.begin(), lY.end() );
            for( uint32_t l = 0; l < lY.size() - 1; l++ ) lExpectedValues2.push_back( lY[l + 1] );
            lExpectedValues2.push_back( static_cast<int64_t>( 121212 ) );
        }
    }

    std::vector<uint32_t> lDim3{ 3, 5, 512 };
    std::vector<int64_t>  lValues3;
    std::vector<int64_t>  lExpectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto lY = RandomNumber<int64_t>( 512 );
            lValues3.insert( lValues3.end(), lY.begin(), lY.end() );
            for( uint32_t l = 0; l < lY.size() - 1; l++ ) lExpectedValues3.push_back( lY[l + 1] );
            lExpectedValues3.push_back( static_cast<int64_t>( 121212 ) );
        }
    }

    std::vector<int64_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    sDataInitializerComponent lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( lScope, lInputInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( int64_t ) ) );

    auto lFillValue = ConstantScalarValue( lScope, static_cast<int64_t>( 121212 ) );
    auto lResult0   = Shift( lScope, lInputTensor, -1, lFillValue );
    lScope.Run( lResult0 );

    auto lOutputShape = lResult0.Get<sMultiTensorComponent>().mValue.Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.mRank == 3 );
    REQUIRE( lOutputShape.mShape[0] == std::vector<uint32_t>{ 7, 3, 1024 } );
    REQUIRE( lOutputShape.mShape[1] == std::vector<uint32_t>{ 2, 7, 256 } );
    REQUIRE( lOutputShape.mShape[2] == std::vector<uint32_t>{ 3, 5, 512 } );

    std::vector<int64_t> lExpectedValues =
        ConcatenateVectors( std::vector<std::vector<int64_t>>{ lExpectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<int64_t> lResultValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<int64_t>();
    REQUIRE( lResultValues == lExpectedValues );
}

TEST_CASE( "Finite shift to the left by 3", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 7, 3, 1024 };
    std::vector<int64_t>  lValues1;
    std::vector<int64_t>  lExpectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto lY = RandomNumber<int64_t>( 1024 );
            lValues1.insert( lValues1.end(), lY.begin(), lY.end() );
            for( uint32_t l = 0; l < lY.size() - 3; l++ ) lExpectedValues1.push_back( lY[l + 3] );
            lExpectedValues1.push_back( static_cast<int64_t>( 121212 ) );
            lExpectedValues1.push_back( static_cast<int64_t>( 121212 ) );
            lExpectedValues1.push_back( static_cast<int64_t>( 121212 ) );
        }
    }

    std::vector<uint32_t> lDim2{ 2, 7, 256 };
    std::vector<int64_t>  lValues2;
    std::vector<int64_t>  lExpectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto lY = RandomNumber<int64_t>( 256 );
            lValues2.insert( lValues2.end(), lY.begin(), lY.end() );
            for( uint32_t l = 0; l < lY.size() - 3; l++ ) lExpectedValues2.push_back( lY[l + 3] );
            lExpectedValues2.push_back( static_cast<int64_t>( 121212 ) );
            lExpectedValues2.push_back( static_cast<int64_t>( 121212 ) );
            lExpectedValues2.push_back( static_cast<int64_t>( 121212 ) );
        }
    }

    std::vector<uint32_t> lDim3{ 3, 5, 512 };
    std::vector<int64_t>  lValues3;
    std::vector<int64_t>  lExpectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto lY = RandomNumber<int64_t>( 512 );
            lValues3.insert( lValues3.end(), lY.begin(), lY.end() );
            for( uint32_t l = 0; l < lY.size() - 3; l++ ) lExpectedValues3.push_back( lY[l + 3] );
            lExpectedValues3.push_back( static_cast<int64_t>( 121212 ) );
            lExpectedValues3.push_back( static_cast<int64_t>( 121212 ) );
            lExpectedValues3.push_back( static_cast<int64_t>( 121212 ) );
        }
    }

    std::vector<int64_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    sDataInitializerComponent lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( lScope, lInputInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( int64_t ) ) );

    auto lFillValue = ConstantScalarValue( lScope, static_cast<int64_t>( 121212 ) );
    auto lResult0   = Shift( lScope, lInputTensor, -3, lFillValue );
    lScope.Run( lResult0 );

    auto lOutputShape = lResult0.Get<sMultiTensorComponent>().mValue.Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.mRank == 3 );
    REQUIRE( lOutputShape.mShape[0] == std::vector<uint32_t>{ 7, 3, 1024 } );
    REQUIRE( lOutputShape.mShape[1] == std::vector<uint32_t>{ 2, 7, 256 } );
    REQUIRE( lOutputShape.mShape[2] == std::vector<uint32_t>{ 3, 5, 512 } );

    std::vector<int64_t> lExpectedValues =
        ConcatenateVectors( std::vector<std::vector<int64_t>>{ lExpectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<int64_t> lResultValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<int64_t>();
    REQUIRE( lResultValues == lExpectedValues );
}

TEST_CASE( "1D convolution", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    sRandomNormalInitializerComponent lInitializer{};
    lInitializer.mType = eScalarType::FLOAT32;

    std::vector<uint32_t> lDim1{ 7, 3, 124 };
    std::vector<int64_t>  lValues1;
    std::vector<int64_t>  lExpectedValues1;

    std::vector<uint32_t> lKDim1{ 7, 3, 34 };
    std::vector<int64_t>  lKernel1;

    auto lConv1D = []( std::vector<int64_t> aX, std::vector<int64_t> aY ) -> std::vector<int64_t>
    {
        auto lOutput = std::vector<int64_t>( aX.size() );

        for( uint32_t i = 0; i < aX.size(); i++ )
        {
            int32_t lAcc = 0;
            for( uint32_t k = 0; k < aY.size(); k++ )
            {
                if( k <= i ) lAcc += aX[i - k] * aY[k];
            }
            lOutput[i] = lAcc;
        }
        return lOutput;
    };

    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto lY = RandomNumber<int64_t>( 124, -10000, 10000 );
            lValues1.insert( lValues1.end(), lY.begin(), lY.end() );

            auto lZ = RandomNumber<int64_t>( 34, -10000, 10000 );
            lKernel1.insert( lKernel1.end(), lZ.begin(), lZ.end() );

            auto lC = lConv1D( lY, lZ );
            lExpectedValues1.insert( lExpectedValues1.end(), lC.begin(), lC.end() );
        }
    }

    std::vector<uint32_t> lDim2{ 2, 7, 75 };
    std::vector<int64_t>  lValues2;
    std::vector<int64_t>  lExpectedValues2;

    std::vector<uint32_t> lKDim2{ 2, 7, 42 };
    std::vector<int64_t>  lKernel2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto lY = RandomNumber<int64_t>( 75, -10000, 10000 );
            lValues2.insert( lValues2.end(), lY.begin(), lY.end() );

            auto lZ = RandomNumber<int64_t>( 42, -10000, 10000 );
            lKernel2.insert( lKernel2.end(), lZ.begin(), lZ.end() );

            auto lC = lConv1D( lY, lZ );
            lExpectedValues2.insert( lExpectedValues2.end(), lC.begin(), lC.end() );
        }
    }

    std::vector<uint32_t> lDim3{ 3, 5, 23 };
    std::vector<int64_t>  lValues3;
    std::vector<int64_t>  lExpectedValues3;

    std::vector<uint32_t> lKDim3{ 3, 5, 5 };
    std::vector<int64_t>  lKernel3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto lY = RandomNumber<int64_t>( 23, -10000, 10000 );
            lValues3.insert( lValues3.end(), lY.begin(), lY.end() );

            auto lZ = RandomNumber<int64_t>( 5, -10000, 10000 );
            lKernel3.insert( lKernel3.end(), lZ.begin(), lZ.end() );

            auto lC = lConv1D( lY, lZ );
            lExpectedValues3.insert( lExpectedValues3.end(), lC.begin(), lC.end() );
        }
    }

    std::vector<int64_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    std::vector<int64_t> lKernelValues;
    lKernelValues.insert( lKernelValues.end(), lKernel1.begin(), lKernel1.end() );
    lKernelValues.insert( lKernelValues.end(), lKernel2.begin(), lKernel2.end() );
    lKernelValues.insert( lKernelValues.end(), lKernel3.begin(), lKernel3.end() );

    sDataInitializerComponent lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( lScope, lInputInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( int64_t ) ) );

    sDataInitializerComponent lKernelInitializer( lKernelValues );
    auto lKernelensor = MultiTensorValue( lScope, lKernelInitializer, sTensorShape( { lKDim1, lKDim2, lKDim3 }, sizeof( int64_t ) ) );

    auto lResult0 = Conv1D( lScope, lInputTensor, lKernelensor );
    lScope.Run( lResult0 );

    auto lOutputShape = lResult0.Get<sMultiTensorComponent>().mValue.Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.mRank == 3 );
    REQUIRE( lOutputShape.mShape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( lOutputShape.mShape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( lOutputShape.mShape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<int64_t> lExpectedValues =
        ConcatenateVectors( std::vector<std::vector<int64_t>>{ lExpectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<int64_t> lResultValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<int64_t>();
    REQUIRE( lResultValues == lExpectedValues );
}

TEST_CASE( "1D convolution (uint32_t)", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    std::vector<uint32_t> lDim1{ 7, 3, 124 };
    std::vector<uint32_t> lValues1;
    std::vector<uint32_t> lExpectedValues1;

    std::vector<uint32_t> lKDim1{ 7, 3, 34 };
    std::vector<uint32_t> lKernel1;

    auto lConv1D = []( std::vector<uint32_t> aX, std::vector<uint32_t> aY ) -> std::vector<uint32_t>
    {
        auto lOutput = std::vector<uint32_t>( aX.size() );

        for( uint32_t i = 0; i < aX.size(); i++ )
        {
            uint32_t lAcc = 0;
            for( uint32_t k = 0; k < aY.size(); k++ )
            {
                if( k <= i ) lAcc += aX[i - k] * aY[k];
            }
            lOutput[i] = lAcc;
        }
        return lOutput;
    };

    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto lY = RandomNumber<uint32_t>( 124, 0, 10000 );
            lValues1.insert( lValues1.end(), lY.begin(), lY.end() );

            auto lZ = RandomNumber<uint32_t>( 34, 0, 10000 );
            lKernel1.insert( lKernel1.end(), lZ.begin(), lZ.end() );

            auto lC = lConv1D( lY, lZ );
            lExpectedValues1.insert( lExpectedValues1.end(), lC.begin(), lC.end() );
        }
    }

    std::vector<uint32_t> lDim2{ 2, 7, 75 };
    std::vector<uint32_t> lValues2;
    std::vector<uint32_t> lExpectedValues2;

    std::vector<uint32_t> lKDim2{ 2, 7, 42 };
    std::vector<uint32_t> lKernel2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto lY = RandomNumber<uint32_t>( 75, 0, 10000 );
            lValues2.insert( lValues2.end(), lY.begin(), lY.end() );

            auto lZ = RandomNumber<uint32_t>( 42, 0, 10000 );
            lKernel2.insert( lKernel2.end(), lZ.begin(), lZ.end() );

            auto lC = lConv1D( lY, lZ );
            lExpectedValues2.insert( lExpectedValues2.end(), lC.begin(), lC.end() );
        }
    }

    std::vector<uint32_t> lDim3{ 3, 5, 23 };
    std::vector<uint32_t> lValues3;
    std::vector<uint32_t> lExpectedValues3;

    std::vector<uint32_t> lKDim3{ 3, 5, 5 };
    std::vector<uint32_t> lKernel3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto lY = RandomNumber<uint32_t>( 23, 0, 10000 );
            lValues3.insert( lValues3.end(), lY.begin(), lY.end() );

            auto lZ = RandomNumber<uint32_t>( 5, 0, 10000 );
            lKernel3.insert( lKernel3.end(), lZ.begin(), lZ.end() );

            auto lC = lConv1D( lY, lZ );
            lExpectedValues3.insert( lExpectedValues3.end(), lC.begin(), lC.end() );
        }
    }

    std::vector<uint32_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    std::vector<uint32_t> lKernelValues;
    lKernelValues.insert( lKernelValues.end(), lKernel1.begin(), lKernel1.end() );
    lKernelValues.insert( lKernelValues.end(), lKernel2.begin(), lKernel2.end() );
    lKernelValues.insert( lKernelValues.end(), lKernel3.begin(), lKernel3.end() );

    sDataInitializerComponent lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( lScope, lInputInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( uint32_t ) ) );

    sDataInitializerComponent lKernelInitializer( lKernelValues );
    auto lKernelensor = MultiTensorValue( lScope, lKernelInitializer, sTensorShape( { lKDim1, lKDim2, lKDim3 }, sizeof( uint32_t ) ) );

    auto lResult0 = Conv1D( lScope, lInputTensor, lKernelensor );
    lScope.Run( lResult0 );

    auto lOutputShape = lResult0.Get<sMultiTensorComponent>().mValue.Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.mRank == 3 );
    REQUIRE( lOutputShape.mShape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( lOutputShape.mShape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( lOutputShape.mShape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<uint32_t> lExpectedValues =
        ConcatenateVectors( std::vector<std::vector<uint32_t>>{ lExpectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<uint32_t> lResultValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint32_t>();
    REQUIRE( lResultValues == lExpectedValues );
}

TEST_CASE( "HCat (uint32_t)", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    std::vector<uint32_t> lDim1{ 7, 3, 124 };
    std::vector<uint32_t> lValues1;
    std::vector<uint32_t> lExpectedValues1;

    std::vector<uint32_t> lKDim1{ 7, 3, 34 };
    std::vector<uint32_t> lKernel1;

    auto lHCat = []( std::vector<uint32_t> aX, std::vector<uint32_t> aY ) -> std::vector<uint32_t>
    {
        auto lOutput = std::vector<uint32_t>{};
        lOutput.insert( lOutput.end(), aX.begin(), aX.end() );
        lOutput.insert( lOutput.end(), aY.begin(), aY.end() );

        return lOutput;
    };

    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto lY = RandomNumber<uint32_t>( 124, 0, 10000 );
            lValues1.insert( lValues1.end(), lY.begin(), lY.end() );

            auto lZ = RandomNumber<uint32_t>( 34, 0, 10000 );
            lKernel1.insert( lKernel1.end(), lZ.begin(), lZ.end() );

            auto lC = lHCat( lY, lZ );
            lExpectedValues1.insert( lExpectedValues1.end(), lC.begin(), lC.end() );
        }
    }

    std::vector<uint32_t> lDim2{ 2, 7, 75 };
    std::vector<uint32_t> lValues2;
    std::vector<uint32_t> lExpectedValues2;

    std::vector<uint32_t> lKDim2{ 2, 7, 42 };
    std::vector<uint32_t> lKernel2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto lY = RandomNumber<uint32_t>( 75, 0, 10000 );
            lValues2.insert( lValues2.end(), lY.begin(), lY.end() );

            auto lZ = RandomNumber<uint32_t>( 42, 0, 10000 );
            lKernel2.insert( lKernel2.end(), lZ.begin(), lZ.end() );

            auto lC = lHCat( lY, lZ );
            lExpectedValues2.insert( lExpectedValues2.end(), lC.begin(), lC.end() );
        }
    }

    std::vector<uint32_t> lDim3{ 3, 5, 23 };
    std::vector<uint32_t> lValues3;
    std::vector<uint32_t> lExpectedValues3;

    std::vector<uint32_t> lKDim3{ 3, 5, 5 };
    std::vector<uint32_t> lKernel3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto lY = RandomNumber<uint32_t>( 23, 0, 10000 );
            lValues3.insert( lValues3.end(), lY.begin(), lY.end() );

            auto lZ = RandomNumber<uint32_t>( 5, 0, 10000 );
            lKernel3.insert( lKernel3.end(), lZ.begin(), lZ.end() );

            auto lC = lHCat( lY, lZ );
            lExpectedValues3.insert( lExpectedValues3.end(), lC.begin(), lC.end() );
        }
    }

    std::vector<uint32_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    std::vector<uint32_t> lKernelValues;
    lKernelValues.insert( lKernelValues.end(), lKernel1.begin(), lKernel1.end() );
    lKernelValues.insert( lKernelValues.end(), lKernel2.begin(), lKernel2.end() );
    lKernelValues.insert( lKernelValues.end(), lKernel3.begin(), lKernel3.end() );

    sDataInitializerComponent lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( lScope, lInputInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( uint32_t ) ) );

    sDataInitializerComponent lKernelInitializer( lKernelValues );
    auto lKernelensor = MultiTensorValue( lScope, lKernelInitializer, sTensorShape( { lKDim1, lKDim2, lKDim3 }, sizeof( uint32_t ) ) );

    auto lResult0 = HCat( lScope, lInputTensor, lKernelensor );
    lScope.Run( lResult0 );

    auto lOutputShape = lResult0.Get<sMultiTensorComponent>().mValue.Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.mRank == 3 );
    REQUIRE( lOutputShape.mShape[0] == std::vector<uint32_t>{ 7, 3, 124 + 34 } );
    REQUIRE( lOutputShape.mShape[1] == std::vector<uint32_t>{ 2, 7, 75 + 42 } );
    REQUIRE( lOutputShape.mShape[2] == std::vector<uint32_t>{ 3, 5, 23 + 5 } );

    std::vector<uint32_t> lExpectedValues =
        ConcatenateVectors( std::vector<std::vector<uint32_t>>{ lExpectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<uint32_t> lResultValues = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint32_t>();
    REQUIRE( lResultValues == lExpectedValues );
}

TEMPLATE_TEST_CASE( "Addition broadcast", "[CORE_COMPUTATION_GRAPH]", uint16_t, uint32_t, uint64_t, int16_t, int32_t, int64_t, float )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    std::vector<uint32_t>              lDim1{ 7, 3, 124 };
    std::vector<std::vector<TestType>> lValues1 = RandomVector<TestType>( lDim1, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              lDim2{ 2, 7, 75 };
    std::vector<std::vector<TestType>> lValues2 = RandomVector<TestType>( lDim2, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              lDim3{ 3, 5, 23 };
    std::vector<std::vector<TestType>> lValues3 = RandomVector<TestType>( lDim3, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim1( lDim1.begin(), lDim1.end() - 1 );
    std::vector<TestType> lKernel1 = RandomValues<TestType>( lKDim1, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim2( lDim2.begin(), lDim2.end() - 1 );
    std::vector<TestType> lKernel2 = RandomValues<TestType>( lKDim2, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim3( lDim3.begin(), lDim3.end() - 1 );
    std::vector<TestType> lKernel3 = RandomValues<TestType>( lKDim3, 0, std::numeric_limits<TestType>::max() / 2 );

    auto lExpectedValues1 = BroadcastMap<TestType>( lValues1, lKernel1, []( TestType x, TestType y ) { return x + y; } );
    auto lExpectedValues2 = BroadcastMap<TestType>( lValues2, lKernel2, []( TestType x, TestType y ) { return x + y; } );
    auto lExpectedValues3 = BroadcastMap<TestType>( lValues3, lKernel3, []( TestType x, TestType y ) { return x + y; } );

    std::vector<TestType> lInputValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lValues1 ), ConcatenateVectors( lValues2 ), ConcatenateVectors( lValues3 ) } );

    std::vector<TestType> lKernelValues = ConcatenateVectors( std::vector<std::vector<TestType>>{ lKernel1, lKernel2, lKernel3 } );

    sDataInitializerComponent lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( lScope, lInputInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( TestType ) ) );

    sDataInitializerComponent lKernelInitializer( lKernelValues );
    auto lKernelensor = MultiTensorValue( lScope, lKernelInitializer, sTensorShape( { lKDim1, lKDim2, lKDim3 }, sizeof( TestType ) ) );

    auto lResult0 = Add( lScope, lInputTensor, lKernelensor );
    auto lResult1 = Add( lScope, lKernelensor, lInputTensor );
    lScope.Run( { lResult0, lResult1 } );

    auto lOutputShape = lResult0.Get<sMultiTensorComponent>().mValue.Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.mRank == 3 );
    REQUIRE( lOutputShape.mShape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( lOutputShape.mShape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( lOutputShape.mShape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<TestType> lExpectedValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lExpectedValues1 ), ConcatenateVectors( lExpectedValues2 ), ConcatenateVectors( lExpectedValues3 ) } );

    std::vector<TestType> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<TestType>();
    REQUIRE( lResultValues0 == lExpectedValues );

    std::vector<TestType> lResultValues1 = lResult1.Get<sMultiTensorComponent>().mValue.FetchFlattened<TestType>();
    REQUIRE( lResultValues1 == lExpectedValues );
}

TEMPLATE_TEST_CASE(
    "Multiplication broadcast", "[CORE_COMPUTATION_GRAPH]", uint16_t, uint32_t, uint64_t, int16_t, int32_t, int64_t, float )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    std::vector<uint32_t>              lDim1{ 7, 3, 124 };
    std::vector<std::vector<TestType>> lValues1 = RandomVector<TestType>( lDim1, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              lDim2{ 2, 7, 75 };
    std::vector<std::vector<TestType>> lValues2 = RandomVector<TestType>( lDim2, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              lDim3{ 3, 5, 23 };
    std::vector<std::vector<TestType>> lValues3 = RandomVector<TestType>( lDim3, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim1( lDim1.begin(), lDim1.end() - 1 );
    std::vector<TestType> lKernel1 = RandomValues<TestType>( lKDim1, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim2( lDim2.begin(), lDim2.end() - 1 );
    std::vector<TestType> lKernel2 = RandomValues<TestType>( lKDim2, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim3( lDim3.begin(), lDim3.end() - 1 );
    std::vector<TestType> lKernel3 = RandomValues<TestType>( lKDim3, 0, std::numeric_limits<TestType>::max() / 2 );

    auto lExpectedValues1 = BroadcastMap<TestType>( lValues1, lKernel1, []( TestType x, TestType y ) { return x * y; } );
    auto lExpectedValues2 = BroadcastMap<TestType>( lValues2, lKernel2, []( TestType x, TestType y ) { return x * y; } );
    auto lExpectedValues3 = BroadcastMap<TestType>( lValues3, lKernel3, []( TestType x, TestType y ) { return x * y; } );

    std::vector<TestType> lInputValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lValues1 ), ConcatenateVectors( lValues2 ), ConcatenateVectors( lValues3 ) } );

    std::vector<TestType> lKernelValues = ConcatenateVectors( std::vector<std::vector<TestType>>{ lKernel1, lKernel2, lKernel3 } );

    sDataInitializerComponent lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( lScope, lInputInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( TestType ) ) );

    sDataInitializerComponent lKernelInitializer( lKernelValues );
    auto lKernelensor = MultiTensorValue( lScope, lKernelInitializer, sTensorShape( { lKDim1, lKDim2, lKDim3 }, sizeof( TestType ) ) );

    auto lResult0 = Multiply( lScope, lInputTensor, lKernelensor );
    auto lResult1 = Multiply( lScope, lKernelensor, lInputTensor );
    lScope.Run( { lResult0, lResult1 } );

    auto lOutputShape = lResult0.Get<sMultiTensorComponent>().mValue.Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.mRank == 3 );
    REQUIRE( lOutputShape.mShape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( lOutputShape.mShape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( lOutputShape.mShape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<TestType> lExpectedValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lExpectedValues1 ), ConcatenateVectors( lExpectedValues2 ), ConcatenateVectors( lExpectedValues3 ) } );

    std::vector<TestType> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<TestType>();
    REQUIRE( lResultValues0 == lExpectedValues );

    std::vector<TestType> lResultValues1 = lResult1.Get<sMultiTensorComponent>().mValue.FetchFlattened<TestType>();
    REQUIRE( lResultValues1 == lExpectedValues );
}

TEST_CASE( "Divison broadcast", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    std::vector<uint32_t>           lDim1{ 7, 3, 124 };
    std::vector<std::vector<float>> lValues1 = RandomVector<float>( lDim1, 0, std::numeric_limits<float>::max() / 2 );

    std::vector<uint32_t>           lDim2{ 2, 7, 75 };
    std::vector<std::vector<float>> lValues2 = RandomVector<float>( lDim2, 0, std::numeric_limits<float>::max() / 2 );

    std::vector<uint32_t>           lDim3{ 3, 5, 23 };
    std::vector<std::vector<float>> lValues3 = RandomVector<float>( lDim3, 0, std::numeric_limits<float>::max() / 2 );

    std::vector<uint32_t> lKDim1( lDim1.begin(), lDim1.end() - 1 );
    std::vector<float>    lKernel1 = RandomValues<float>( lKDim1, 0.001, std::numeric_limits<float>::max() / 2 );

    std::vector<uint32_t> lKDim2( lDim2.begin(), lDim2.end() - 1 );
    std::vector<float>    lKernel2 = RandomValues<float>( lKDim2, 0.001, std::numeric_limits<float>::max() / 2 );

    std::vector<uint32_t> lKDim3( lDim3.begin(), lDim3.end() - 1 );
    std::vector<float>    lKernel3 = RandomValues<float>( lKDim3, 0.001, std::numeric_limits<float>::max() / 2 );

    auto lExpectedValues01 = BroadcastMap<float>( lValues1, lKernel1, []( float x, float y ) { return x / y; } );
    auto lExpectedValues02 = BroadcastMap<float>( lValues2, lKernel2, []( float x, float y ) { return x / y; } );
    auto lExpectedValues03 = BroadcastMap<float>( lValues3, lKernel3, []( float x, float y ) { return x / y; } );

    auto lExpectedValues11 = BroadcastMap<float>( lKernel1, lValues1, []( float x, float y ) { return x / y; } );
    auto lExpectedValues12 = BroadcastMap<float>( lKernel2, lValues2, []( float x, float y ) { return x / y; } );
    auto lExpectedValues13 = BroadcastMap<float>( lKernel3, lValues3, []( float x, float y ) { return x / y; } );

    std::vector<float> lInputValues = ConcatenateVectors( std::vector<std::vector<float>>{
        ConcatenateVectors( lValues1 ), ConcatenateVectors( lValues2 ), ConcatenateVectors( lValues3 ) } );

    std::vector<float> lKernelValues = ConcatenateVectors( std::vector<std::vector<float>>{ lKernel1, lKernel2, lKernel3 } );

    sDataInitializerComponent lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( lScope, lInputInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( float ) ) );

    sDataInitializerComponent lKernelInitializer( lKernelValues );
    auto lKernelensor = MultiTensorValue( lScope, lKernelInitializer, sTensorShape( { lKDim1, lKDim2, lKDim3 }, sizeof( float ) ) );

    auto lResult0 = Divide( lScope, lInputTensor, lKernelensor );
    auto lResult1 = Divide( lScope, lKernelensor, lInputTensor );
    lScope.Run( { lResult0, lResult1 } );

    auto lOutputShape = lResult0.Get<sMultiTensorComponent>().mValue.Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.mRank == 3 );
    REQUIRE( lOutputShape.mShape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( lOutputShape.mShape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( lOutputShape.mShape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<float> lExpectedValues0 = ConcatenateVectors( std::vector<std::vector<float>>{
        ConcatenateVectors( lExpectedValues01 ), ConcatenateVectors( lExpectedValues02 ), ConcatenateVectors( lExpectedValues03 ) } );
    std::vector<float> lExpectedValues1 = ConcatenateVectors( std::vector<std::vector<float>>{
        ConcatenateVectors( lExpectedValues11 ), ConcatenateVectors( lExpectedValues12 ), ConcatenateVectors( lExpectedValues13 ) } );

    std::vector<float> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    REQUIRE( lResultValues0 == lExpectedValues0 );

    std::vector<float> lResultValues1 = lResult1.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    REQUIRE( lResultValues1 == lExpectedValues1 );
}

TEMPLATE_TEST_CASE( "Subtraction broadcast", "[CORE_COMPUTATION_GRAPH]", int16_t, int32_t, int64_t, float, double )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    std::vector<uint32_t>              lDim1{ 7, 3, 124 };
    std::vector<std::vector<TestType>> lValues1 = RandomVector<TestType>( lDim1, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              lDim2{ 2, 7, 75 };
    std::vector<std::vector<TestType>> lValues2 = RandomVector<TestType>( lDim2, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              lDim3{ 3, 5, 23 };
    std::vector<std::vector<TestType>> lValues3 = RandomVector<TestType>( lDim3, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim1( lDim1.begin(), lDim1.end() - 1 );
    std::vector<TestType> lKernel1 = RandomValues<TestType>( lKDim1, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim2( lDim2.begin(), lDim2.end() - 1 );
    std::vector<TestType> lKernel2 = RandomValues<TestType>( lKDim2, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim3( lDim3.begin(), lDim3.end() - 1 );
    std::vector<TestType> lKernel3 = RandomValues<TestType>( lKDim3, 0, std::numeric_limits<TestType>::max() / 2 );

    auto lExpectedValues01 = BroadcastMap<TestType>( lValues1, lKernel1, []( TestType x, TestType y ) { return x - y; } );
    auto lExpectedValues02 = BroadcastMap<TestType>( lValues2, lKernel2, []( TestType x, TestType y ) { return x - y; } );
    auto lExpectedValues03 = BroadcastMap<TestType>( lValues3, lKernel3, []( TestType x, TestType y ) { return x - y; } );

    auto lExpectedValues11 = BroadcastMap<TestType>( lKernel1, lValues1, []( TestType x, TestType y ) { return x - y; } );
    auto lExpectedValues12 = BroadcastMap<TestType>( lKernel2, lValues2, []( TestType x, TestType y ) { return x - y; } );
    auto lExpectedValues13 = BroadcastMap<TestType>( lKernel3, lValues3, []( TestType x, TestType y ) { return x - y; } );

    std::vector<TestType> lInputValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lValues1 ), ConcatenateVectors( lValues2 ), ConcatenateVectors( lValues3 ) } );

    std::vector<TestType> lKernelValues = ConcatenateVectors( std::vector<std::vector<TestType>>{ lKernel1, lKernel2, lKernel3 } );

    sDataInitializerComponent lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( lScope, lInputInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( TestType ) ) );

    sDataInitializerComponent lKernelInitializer( lKernelValues );
    auto lKernelensor = MultiTensorValue( lScope, lKernelInitializer, sTensorShape( { lKDim1, lKDim2, lKDim3 }, sizeof( TestType ) ) );

    auto lResult0 = Subtract( lScope, lInputTensor, lKernelensor );
    auto lResult1 = Subtract( lScope, lKernelensor, lInputTensor );
    lScope.Run( { lResult0, lResult1 } );

    auto lOutputShape = lResult0.Get<sMultiTensorComponent>().mValue.Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.mRank == 3 );
    REQUIRE( lOutputShape.mShape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( lOutputShape.mShape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( lOutputShape.mShape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<TestType> lExpectedValues0 = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lExpectedValues01 ), ConcatenateVectors( lExpectedValues02 ), ConcatenateVectors( lExpectedValues03 ) } );
    std::vector<TestType> lExpectedValues1 = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lExpectedValues11 ), ConcatenateVectors( lExpectedValues12 ), ConcatenateVectors( lExpectedValues13 ) } );

    std::vector<TestType> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<TestType>();
    REQUIRE( lResultValues0 == lExpectedValues0 );

    std::vector<TestType> lResultValues1 = lResult1.Get<sMultiTensorComponent>().mValue.FetchFlattened<TestType>();
    REQUIRE( lResultValues1 == lExpectedValues1 );
}

TEST_CASE( "AND broadcast", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    std::vector<uint32_t>             lDim1{ 7, 3, 124 };
    std::vector<std::vector<uint8_t>> lValues1 = RandomBooleanVector( lDim1 );

    std::vector<uint32_t>             lDim2{ 2, 7, 75 };
    std::vector<std::vector<uint8_t>> lValues2 = RandomBooleanVector( lDim2 );

    std::vector<uint32_t>             lDim3{ 3, 5, 23 };
    std::vector<std::vector<uint8_t>> lValues3 = RandomBooleanVector( lDim3 );

    std::vector<uint32_t> lKDim1( lDim1.begin(), lDim1.end() - 1 );
    std::vector<uint8_t>  lKernel1 = RandomBooleanValues( lKDim1 );

    std::vector<uint32_t> lKDim2( lDim2.begin(), lDim2.end() - 1 );
    std::vector<uint8_t>  lKernel2 = RandomBooleanValues( lKDim2 );

    std::vector<uint32_t> lKDim3( lDim3.begin(), lDim3.end() - 1 );
    std::vector<uint8_t>  lKernel3 = RandomBooleanValues( lKDim3 );

    auto lExpectedValues1 = BroadcastMap<uint8_t>( lValues1, lKernel1, []( uint8_t x, uint8_t y ) { return x && y; } );
    auto lExpectedValues2 = BroadcastMap<uint8_t>( lValues2, lKernel2, []( uint8_t x, uint8_t y ) { return x && y; } );
    auto lExpectedValues3 = BroadcastMap<uint8_t>( lValues3, lKernel3, []( uint8_t x, uint8_t y ) { return x && y; } );

    std::vector<uint8_t> lInputValues = ConcatenateVectors( std::vector<std::vector<uint8_t>>{
        ConcatenateVectors( lValues1 ), ConcatenateVectors( lValues2 ), ConcatenateVectors( lValues3 ) } );

    std::vector<uint8_t> lKernelValues = ConcatenateVectors( std::vector<std::vector<uint8_t>>{ lKernel1, lKernel2, lKernel3 } );

    sDataInitializerComponent lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( lScope, lInputInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( uint8_t ) ) );

    sDataInitializerComponent lKernelInitializer( lKernelValues );
    auto lKernelensor = MultiTensorValue( lScope, lKernelInitializer, sTensorShape( { lKDim1, lKDim2, lKDim3 }, sizeof( uint8_t ) ) );

    auto lResult0 = And( lScope, lInputTensor, lKernelensor );
    auto lResult1 = And( lScope, lKernelensor, lInputTensor );
    lScope.Run( { lResult0, lResult1 } );

    auto lOutputShape = lResult0.Get<sMultiTensorComponent>().mValue.Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.mRank == 3 );
    REQUIRE( lOutputShape.mShape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( lOutputShape.mShape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( lOutputShape.mShape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<uint8_t> lExpectedValues = ConcatenateVectors( std::vector<std::vector<uint8_t>>{
        ConcatenateVectors( lExpectedValues1 ), ConcatenateVectors( lExpectedValues2 ), ConcatenateVectors( lExpectedValues3 ) } );

    std::vector<uint8_t> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( lResultValues0 == lExpectedValues );

    std::vector<uint8_t> lResultValues1 = lResult1.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( lResultValues1 == lExpectedValues );
}

TEST_CASE( "OR broadcast", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    std::vector<uint32_t>             lDim1{ 7, 3, 124 };
    std::vector<std::vector<uint8_t>> lValues1 = RandomBooleanVector( lDim1 );

    std::vector<uint32_t>             lDim2{ 2, 7, 75 };
    std::vector<std::vector<uint8_t>> lValues2 = RandomBooleanVector( lDim2 );

    std::vector<uint32_t>             lDim3{ 3, 5, 23 };
    std::vector<std::vector<uint8_t>> lValues3 = RandomBooleanVector( lDim3 );

    std::vector<uint32_t> lKDim1( lDim1.begin(), lDim1.end() - 1 );
    std::vector<uint8_t>  lKernel1 = RandomBooleanValues( lKDim1 );

    std::vector<uint32_t> lKDim2( lDim2.begin(), lDim2.end() - 1 );
    std::vector<uint8_t>  lKernel2 = RandomBooleanValues( lKDim2 );

    std::vector<uint32_t> lKDim3( lDim3.begin(), lDim3.end() - 1 );
    std::vector<uint8_t>  lKernel3 = RandomBooleanValues( lKDim3 );

    auto lExpectedValues1 = BroadcastMap<uint8_t>( lValues1, lKernel1, []( uint8_t x, uint8_t y ) { return x || y; } );
    auto lExpectedValues2 = BroadcastMap<uint8_t>( lValues2, lKernel2, []( uint8_t x, uint8_t y ) { return x || y; } );
    auto lExpectedValues3 = BroadcastMap<uint8_t>( lValues3, lKernel3, []( uint8_t x, uint8_t y ) { return x || y; } );

    std::vector<uint8_t> lInputValues = ConcatenateVectors( std::vector<std::vector<uint8_t>>{
        ConcatenateVectors( lValues1 ), ConcatenateVectors( lValues2 ), ConcatenateVectors( lValues3 ) } );

    std::vector<uint8_t> lKernelValues = ConcatenateVectors( std::vector<std::vector<uint8_t>>{ lKernel1, lKernel2, lKernel3 } );

    sDataInitializerComponent lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( lScope, lInputInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( uint8_t ) ) );

    sDataInitializerComponent lKernelInitializer( lKernelValues );
    auto lKernelensor = MultiTensorValue( lScope, lKernelInitializer, sTensorShape( { lKDim1, lKDim2, lKDim3 }, sizeof( uint8_t ) ) );

    auto lResult0 = Or( lScope, lInputTensor, lKernelensor );
    auto lResult1 = Or( lScope, lKernelensor, lInputTensor );
    lScope.Run( { lResult0, lResult1 } );

    auto lOutputShape = lResult0.Get<sMultiTensorComponent>().mValue.Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.mRank == 3 );
    REQUIRE( lOutputShape.mShape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( lOutputShape.mShape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( lOutputShape.mShape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<uint8_t> lExpectedValues = ConcatenateVectors( std::vector<std::vector<uint8_t>>{
        ConcatenateVectors( lExpectedValues1 ), ConcatenateVectors( lExpectedValues2 ), ConcatenateVectors( lExpectedValues3 ) } );

    std::vector<uint8_t> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( lResultValues0 == lExpectedValues );

    std::vector<uint8_t> lResultValues1 = lResult1.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( lResultValues1 == lExpectedValues );
}

TEMPLATE_TEST_CASE( "Bitwise AND broadcast", "[CORE_COMPUTATION_GRAPH]", uint16_t, uint32_t, uint64_t, int16_t, int32_t, int64_t )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    std::vector<uint32_t>              lDim1{ 7, 3, 124 };
    std::vector<std::vector<TestType>> lValues1 = RandomVector<TestType>( lDim1, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t>              lDim2{ 2, 7, 75 };
    std::vector<std::vector<TestType>> lValues2 = RandomVector<TestType>( lDim2, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t>              lDim3{ 3, 5, 23 };
    std::vector<std::vector<TestType>> lValues3 = RandomVector<TestType>( lDim3, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t> lKDim1( lDim1.begin(), lDim1.end() - 1 );
    std::vector<TestType> lKernel1 = RandomValues<TestType>( lKDim1, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t> lKDim2( lDim2.begin(), lDim2.end() - 1 );
    std::vector<TestType> lKernel2 = RandomValues<TestType>( lKDim2, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t> lKDim3( lDim3.begin(), lDim3.end() - 1 );
    std::vector<TestType> lKernel3 = RandomValues<TestType>( lKDim3, 0, std::numeric_limits<TestType>::max() );

    auto lExpectedValues1 = BroadcastMap<TestType>( lValues1, lKernel1, []( TestType x, TestType y ) { return x & y; } );
    auto lExpectedValues2 = BroadcastMap<TestType>( lValues2, lKernel2, []( TestType x, TestType y ) { return x & y; } );
    auto lExpectedValues3 = BroadcastMap<TestType>( lValues3, lKernel3, []( TestType x, TestType y ) { return x & y; } );

    std::vector<TestType> lInputValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lValues1 ), ConcatenateVectors( lValues2 ), ConcatenateVectors( lValues3 ) } );

    std::vector<TestType> lKernelValues = ConcatenateVectors( std::vector<std::vector<TestType>>{ lKernel1, lKernel2, lKernel3 } );

    sDataInitializerComponent lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( lScope, lInputInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( TestType ) ) );

    sDataInitializerComponent lKernelInitializer( lKernelValues );
    auto lKernelensor = MultiTensorValue( lScope, lKernelInitializer, sTensorShape( { lKDim1, lKDim2, lKDim3 }, sizeof( TestType ) ) );

    auto lResult0 = BitwiseAnd( lScope, lInputTensor, lKernelensor );
    auto lResult1 = BitwiseAnd( lScope, lKernelensor, lInputTensor );
    lScope.Run( { lResult0, lResult1 } );

    auto lOutputShape = lResult0.Get<sMultiTensorComponent>().mValue.Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.mRank == 3 );
    REQUIRE( lOutputShape.mShape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( lOutputShape.mShape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( lOutputShape.mShape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<TestType> lExpectedValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lExpectedValues1 ), ConcatenateVectors( lExpectedValues2 ), ConcatenateVectors( lExpectedValues3 ) } );

    std::vector<TestType> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<TestType>();
    REQUIRE( lResultValues0 == lExpectedValues );

    std::vector<TestType> lResultValues1 = lResult1.Get<sMultiTensorComponent>().mValue.FetchFlattened<TestType>();
    REQUIRE( lResultValues1 == lExpectedValues );
}

TEMPLATE_TEST_CASE( "Bitwise OR broadcast", "[CORE_COMPUTATION_GRAPH]", uint16_t, uint32_t, uint64_t, int16_t, int32_t, int64_t )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    std::vector<uint32_t>              lDim1{ 7, 3, 124 };
    std::vector<std::vector<TestType>> lValues1 = RandomVector<TestType>( lDim1, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t>              lDim2{ 2, 7, 75 };
    std::vector<std::vector<TestType>> lValues2 = RandomVector<TestType>( lDim2, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t>              lDim3{ 3, 5, 23 };
    std::vector<std::vector<TestType>> lValues3 = RandomVector<TestType>( lDim3, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t> lKDim1( lDim1.begin(), lDim1.end() - 1 );
    std::vector<TestType> lKernel1 = RandomValues<TestType>( lKDim1, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t> lKDim2( lDim2.begin(), lDim2.end() - 1 );
    std::vector<TestType> lKernel2 = RandomValues<TestType>( lKDim2, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t> lKDim3( lDim3.begin(), lDim3.end() - 1 );
    std::vector<TestType> lKernel3 = RandomValues<TestType>( lKDim3, 0, std::numeric_limits<TestType>::max() );

    auto lExpectedValues1 = BroadcastMap<TestType>( lValues1, lKernel1, []( TestType x, TestType y ) { return x | y; } );
    auto lExpectedValues2 = BroadcastMap<TestType>( lValues2, lKernel2, []( TestType x, TestType y ) { return x | y; } );
    auto lExpectedValues3 = BroadcastMap<TestType>( lValues3, lKernel3, []( TestType x, TestType y ) { return x | y; } );

    std::vector<TestType> lInputValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lValues1 ), ConcatenateVectors( lValues2 ), ConcatenateVectors( lValues3 ) } );

    std::vector<TestType> lKernelValues = ConcatenateVectors( std::vector<std::vector<TestType>>{ lKernel1, lKernel2, lKernel3 } );

    sDataInitializerComponent lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( lScope, lInputInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( TestType ) ) );

    sDataInitializerComponent lKernelInitializer( lKernelValues );
    auto lKernelensor = MultiTensorValue( lScope, lKernelInitializer, sTensorShape( { lKDim1, lKDim2, lKDim3 }, sizeof( TestType ) ) );

    auto lResult0 = BitwiseOr( lScope, lInputTensor, lKernelensor );
    auto lResult1 = BitwiseOr( lScope, lKernelensor, lInputTensor );
    lScope.Run( { lResult0, lResult1 } );

    auto lOutputShape = lResult0.Get<sMultiTensorComponent>().mValue.Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.mRank == 3 );
    REQUIRE( lOutputShape.mShape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( lOutputShape.mShape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( lOutputShape.mShape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<TestType> lExpectedValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lExpectedValues1 ), ConcatenateVectors( lExpectedValues2 ), ConcatenateVectors( lExpectedValues3 ) } );

    std::vector<TestType> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<TestType>();
    REQUIRE( lResultValues0 == lExpectedValues );

    std::vector<TestType> lResultValues1 = lResult1.Get<sMultiTensorComponent>().mValue.FetchFlattened<TestType>();
    REQUIRE( lResultValues1 == lExpectedValues );
}

TEMPLATE_TEST_CASE(
    "Equal broadcast", "[CORE_COMPUTATION_GRAPH]", uint16_t, uint32_t, uint64_t, int16_t, int32_t, int64_t, float, double )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    std::vector<uint32_t>              lDim1{ 7, 3, 124 };
    std::vector<std::vector<TestType>> lValues1 = RandomVector<TestType>( lDim1, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              lDim2{ 2, 7, 75 };
    std::vector<std::vector<TestType>> lValues2 = RandomVector<TestType>( lDim2, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              lDim3{ 3, 5, 23 };
    std::vector<std::vector<TestType>> lValues3 = RandomVector<TestType>( lDim3, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim1( lDim1.begin(), lDim1.end() - 1 );
    std::vector<TestType> lKernel1 = RandomValues<TestType>( lKDim1, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim2( lDim2.begin(), lDim2.end() - 1 );
    std::vector<TestType> lKernel2 = RandomValues<TestType>( lKDim2, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim3( lDim3.begin(), lDim3.end() - 1 );
    std::vector<TestType> lKernel3 = RandomValues<TestType>( lKDim3, 0, std::numeric_limits<TestType>::max() / 2 );

    auto lExpectedValues1 = BroadcastMap<TestType>( lValues1, lKernel1, []( TestType x, TestType y ) { return x == y; } );
    auto lExpectedValues2 = BroadcastMap<TestType>( lValues2, lKernel2, []( TestType x, TestType y ) { return x == y; } );
    auto lExpectedValues3 = BroadcastMap<TestType>( lValues3, lKernel3, []( TestType x, TestType y ) { return x == y; } );

    std::vector<TestType> lInputValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lValues1 ), ConcatenateVectors( lValues2 ), ConcatenateVectors( lValues3 ) } );

    std::vector<TestType> lKernelValues = ConcatenateVectors( std::vector<std::vector<TestType>>{ lKernel1, lKernel2, lKernel3 } );

    sDataInitializerComponent lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( lScope, lInputInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( TestType ) ) );

    sDataInitializerComponent lKernelInitializer( lKernelValues );
    auto lKernelensor = MultiTensorValue( lScope, lKernelInitializer, sTensorShape( { lKDim1, lKDim2, lKDim3 }, sizeof( TestType ) ) );

    auto lResult0 = Equal( lScope, lInputTensor, lKernelensor );
    auto lResult1 = Equal( lScope, lKernelensor, lInputTensor );
    lScope.Run( { lResult0, lResult1 } );

    auto lOutputShape = lResult0.Get<sMultiTensorComponent>().mValue.Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.mRank == 3 );
    REQUIRE( lOutputShape.mShape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( lOutputShape.mShape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( lOutputShape.mShape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<TestType> lExpectedValues0 = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lExpectedValues1 ), ConcatenateVectors( lExpectedValues2 ), ConcatenateVectors( lExpectedValues3 ) } );

    std::vector<uint8_t> lExpectedValues{};
    for( auto x : lExpectedValues0 ) lExpectedValues.push_back( static_cast<uint8_t>( x != 0 ) );
    std::vector<uint8_t> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( lResultValues0 == lExpectedValues );

    std::vector<uint8_t> lResultValues1 = lResult1.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( lResultValues1 == lExpectedValues );
}

TEMPLATE_TEST_CASE(
    "LessThan broadcast", "[CORE_COMPUTATION_GRAPH]", uint16_t, uint32_t, uint64_t, int16_t, int32_t, int64_t, float, double )
{
    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    std::vector<uint32_t>              lDim1{ 7, 3, 124 };
    std::vector<std::vector<TestType>> lValues1 = RandomVector<TestType>( lDim1, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              lDim2{ 2, 7, 75 };
    std::vector<std::vector<TestType>> lValues2 = RandomVector<TestType>( lDim2, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              lDim3{ 3, 5, 23 };
    std::vector<std::vector<TestType>> lValues3 = RandomVector<TestType>( lDim3, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim1( lDim1.begin(), lDim1.end() - 1 );
    std::vector<TestType> lKernel1 = RandomValues<TestType>( lKDim1, 0.001, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim2( lDim2.begin(), lDim2.end() - 1 );
    std::vector<TestType> lKernel2 = RandomValues<TestType>( lKDim2, 0.001, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim3( lDim3.begin(), lDim3.end() - 1 );
    std::vector<TestType> lKernel3 = RandomValues<TestType>( lKDim3, 0.001, std::numeric_limits<TestType>::max() / 2 );

    auto lExpectedValues01 = BroadcastMap<TestType>( lValues1, lKernel1, []( TestType x, TestType y ) { return x < y; } );
    auto lExpectedValues02 = BroadcastMap<TestType>( lValues2, lKernel2, []( TestType x, TestType y ) { return x < y; } );
    auto lExpectedValues03 = BroadcastMap<TestType>( lValues3, lKernel3, []( TestType x, TestType y ) { return x < y; } );

    auto lExpectedValues11 = BroadcastMap<TestType>( lKernel1, lValues1, []( TestType x, TestType y ) { return x < y; } );
    auto lExpectedValues12 = BroadcastMap<TestType>( lKernel2, lValues2, []( TestType x, TestType y ) { return x < y; } );
    auto lExpectedValues13 = BroadcastMap<TestType>( lKernel3, lValues3, []( TestType x, TestType y ) { return x < y; } );

    std::vector<TestType> lInputValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lValues1 ), ConcatenateVectors( lValues2 ), ConcatenateVectors( lValues3 ) } );

    std::vector<TestType> lKernelValues = ConcatenateVectors( std::vector<std::vector<TestType>>{ lKernel1, lKernel2, lKernel3 } );

    sDataInitializerComponent lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( lScope, lInputInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( TestType ) ) );

    sDataInitializerComponent lKernelInitializer( lKernelValues );
    auto lKernelensor = MultiTensorValue( lScope, lKernelInitializer, sTensorShape( { lKDim1, lKDim2, lKDim3 }, sizeof( TestType ) ) );

    auto lResult0 = LessThan( lScope, lInputTensor, lKernelensor );
    auto lResult1 = LessThan( lScope, lKernelensor, lInputTensor );
    lScope.Run( { lResult0, lResult1 } );

    auto lOutputShape = lResult0.Get<sMultiTensorComponent>().mValue.Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.mRank == 3 );
    REQUIRE( lOutputShape.mShape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( lOutputShape.mShape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( lOutputShape.mShape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<TestType> lExpectedValues00 = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lExpectedValues01 ), ConcatenateVectors( lExpectedValues02 ), ConcatenateVectors( lExpectedValues03 ) } );
    std::vector<uint8_t>  lExpectedValues0{};
    for( auto x : lExpectedValues00 ) lExpectedValues0.push_back( static_cast<uint8_t>( x != 0 ) );

    std::vector<TestType> lExpectedValues10 = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lExpectedValues11 ), ConcatenateVectors( lExpectedValues12 ), ConcatenateVectors( lExpectedValues13 ) } );
    std::vector<uint8_t>  lExpectedValues1{};
    for( auto x : lExpectedValues10 ) lExpectedValues1.push_back( static_cast<uint8_t>( x != 0 ) );

    std::vector<uint8_t> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( lResultValues0 == lExpectedValues0 );

    std::vector<uint8_t> lResultValues1 = lResult1.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( lResultValues1 == lExpectedValues1 );
}

TEMPLATE_TEST_CASE(
    "LessThanOrEqual broadcast", "[CORE_COMPUTATION_GRAPH]", uint16_t, uint32_t, uint64_t, int16_t, int32_t, int64_t, float, double )
{

    size_t lPoolSize = 3 * 1024 * 1024;
    Scope  lScope( lPoolSize );

    std::vector<uint32_t>              lDim1{ 7, 3, 124 };
    std::vector<std::vector<TestType>> lValues1 = RandomVector<TestType>( lDim1, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              lDim2{ 2, 7, 75 };
    std::vector<std::vector<TestType>> lValues2 = RandomVector<TestType>( lDim2, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              lDim3{ 3, 5, 23 };
    std::vector<std::vector<TestType>> lValues3 = RandomVector<TestType>( lDim3, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim1( lDim1.begin(), lDim1.end() - 1 );
    std::vector<TestType> lKernel1 = RandomValues<TestType>( lKDim1, 0.001, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim2( lDim2.begin(), lDim2.end() - 1 );
    std::vector<TestType> lKernel2 = RandomValues<TestType>( lKDim2, 0.001, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim3( lDim3.begin(), lDim3.end() - 1 );
    std::vector<TestType> lKernel3 = RandomValues<TestType>( lKDim3, 0.001, std::numeric_limits<TestType>::max() / 2 );

    auto lExpectedValues01 = BroadcastMap<TestType>( lValues1, lKernel1, []( TestType x, TestType y ) { return x <= y; } );
    auto lExpectedValues02 = BroadcastMap<TestType>( lValues2, lKernel2, []( TestType x, TestType y ) { return x <= y; } );
    auto lExpectedValues03 = BroadcastMap<TestType>( lValues3, lKernel3, []( TestType x, TestType y ) { return x <= y; } );

    auto lExpectedValues11 = BroadcastMap<TestType>( lKernel1, lValues1, []( TestType x, TestType y ) { return x <= y; } );
    auto lExpectedValues12 = BroadcastMap<TestType>( lKernel2, lValues2, []( TestType x, TestType y ) { return x <= y; } );
    auto lExpectedValues13 = BroadcastMap<TestType>( lKernel3, lValues3, []( TestType x, TestType y ) { return x <= y; } );

    std::vector<TestType> lInputValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lValues1 ), ConcatenateVectors( lValues2 ), ConcatenateVectors( lValues3 ) } );

    std::vector<TestType> lKernelValues = ConcatenateVectors( std::vector<std::vector<TestType>>{ lKernel1, lKernel2, lKernel3 } );

    sDataInitializerComponent lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( lScope, lInputInitializer, sTensorShape( { lDim1, lDim2, lDim3 }, sizeof( TestType ) ) );

    sDataInitializerComponent lKernelInitializer( lKernelValues );
    auto lKernelensor = MultiTensorValue( lScope, lKernelInitializer, sTensorShape( { lKDim1, lKDim2, lKDim3 }, sizeof( TestType ) ) );

    auto lResult0 = LessThanOrEqual( lScope, lInputTensor, lKernelensor );
    auto lResult1 = LessThanOrEqual( lScope, lKernelensor, lInputTensor );
    lScope.Run( { lResult0, lResult1 } );

    auto lOutputShape = lResult0.Get<sMultiTensorComponent>().mValue.Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.mRank == 3 );
    REQUIRE( lOutputShape.mShape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( lOutputShape.mShape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( lOutputShape.mShape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<TestType> lExpectedValues00 = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lExpectedValues01 ), ConcatenateVectors( lExpectedValues02 ), ConcatenateVectors( lExpectedValues03 ) } );
    std::vector<uint8_t>  lExpectedValues0{};
    for( auto x : lExpectedValues00 ) lExpectedValues0.push_back( static_cast<uint8_t>( x != 0 ) );

    std::vector<TestType> lExpectedValues10 = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lExpectedValues11 ), ConcatenateVectors( lExpectedValues12 ), ConcatenateVectors( lExpectedValues13 ) } );
    std::vector<uint8_t>  lExpectedValues1{};
    for( auto x : lExpectedValues10 ) lExpectedValues1.push_back( static_cast<uint8_t>( x != 0 ) );

    std::vector<uint8_t> lResultValues0 = lResult0.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( lResultValues0 == lExpectedValues0 );

    std::vector<uint8_t> lResultValues1 = lResult1.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( lResultValues1 == lExpectedValues1 );
}
