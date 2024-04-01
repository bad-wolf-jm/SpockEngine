#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "TestUtils.h"

#include "Core/Math/Types.h"

#include "Core/CUDA/Array/MemoryPool.h"
#include "Core/CUDA/Array/MultiTensor.h"

#include "TensorOps/NodeComponents.h"
#include "TensorOps/Scope.h"

using namespace SE::Core;
using namespace SE::TensorOps;
using namespace TestUtils;

std::vector<uint8_t> RandomBooleanValues( std::vector<uint32_t> dim )
{
    uint32_t             size = std::accumulate( dim.begin(), dim.end(), 1, std::multiplies<uint32_t>() );
    std::vector<uint8_t> result{};

    for( uint32_t j = 0; j < size; j++ )
    {
        auto y = RandomBool();
        result.push_back( y );
    }

    return result;
}

template <typename _Ty>
std::vector<_Ty> RandomValues( std::vector<uint32_t> dim, _Ty min, _Ty max )
{
    uint32_t         size = std::accumulate( dim.begin(), dim.end(), 1, std::multiplies<uint32_t>() );
    std::vector<_Ty> result{};

    for( uint32_t j = 0; j < size; j++ )
    {
        auto y = RandomNumber<_Ty>( min, max );
        result.push_back( y );
    }

    return result;
}

std::vector<std::vector<uint8_t>> RandomBooleanVector( std::vector<uint32_t> dim )
{
    uint32_t                          size   = std::accumulate( dim.begin(), dim.end() - 1, 1, std::multiplies<uint32_t>() );
    uint32_t                          length = dim.back();
    std::vector<std::vector<uint8_t>> result{};

    for( uint32_t j = 0; j < size; j++ )
    {
        auto y = RandomBool( length );
        result.push_back( y );
    }

    return result;
}

template <typename _Ty>
std::vector<std::vector<_Ty>> RandomVector( std::vector<uint32_t> dim, _Ty min, _Ty max )
{
    uint32_t                      size   = std::accumulate( dim.begin(), dim.end() - 1, 1, std::multiplies<uint32_t>() );
    uint32_t                      length = dim.back();
    std::vector<std::vector<_Ty>> result{};

    for( uint32_t j = 0; j < size; j++ )
    {
        auto y = RandomNumber<_Ty>( length, min, max );
        result.push_back( y );
    }

    return result;
}

template <typename _Ty>
std::vector<_Ty> BroadcastMap( std::vector<_Ty> vec1, _Ty value, std::function<_Ty( _Ty, _Ty )> function )
{
    std::vector<_Ty> result{};

    for( uint32_t i = 0; i < vec1.size(); i++ )
        result.push_back( function( vec1[i], value ) );

    return result;
}

template <typename _Ty>
std::vector<_Ty> BroadcastMap( _Ty value, std::vector<_Ty> vec1, std::function<_Ty( _Ty, _Ty )> function )
{
    std::vector<_Ty> result{};

    for( uint32_t i = 0; i < vec1.size(); i++ )
        result.push_back( function( value, vec1[i] ) );

    return result;
}

template <typename _Ty>
std::vector<std::vector<_Ty>> BroadcastMap( std::vector<std::vector<_Ty>> vec1, std::vector<_Ty> vec2,
                                            std::function<_Ty( _Ty, _Ty )> function )
{
    std::vector<std::vector<_Ty>> result{};

    for( uint32_t i = 0; i < vec1.size(); i++ )
        result.push_back( BroadcastMap( vec1[i], vec2[i], function ) );

    return result;
}

template <typename _Ty>
std::vector<std::vector<_Ty>> BroadcastMap( std::vector<std::vector<_Ty>> vec1, _Ty vec2, std::function<_Ty( _Ty, _Ty )> function )
{
    std::vector<std::vector<_Ty>> result{};

    for( uint32_t i = 0; i < vec1.size(); i++ )
        result.push_back( BroadcastMap( vec1[i], vec2, function ) );

    return result;
}

template <typename _Ty>
std::vector<std::vector<_Ty>> BroadcastMap( std::vector<_Ty> vec1, std::vector<std::vector<_Ty>> vec2,
                                            std::function<_Ty( _Ty, _Ty )> function )
{
    std::vector<std::vector<_Ty>> result{};

    for( uint32_t i = 0; i < vec1.size(); i++ )
        result.push_back( BroadcastMap( vec1[i], vec2[i], function ) );

    return result;
}

TEST_CASE( "VectorNode", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    SECTION( "Node allocation" )
    {
        std::vector<uint32_t> value = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

        auto node = VectorValue<uint32_t>( scope, value );
        scope.Run( node );

        REQUIRE( node.Get<vector_buffer_t>().mValue.SizeAs<uint32_t>() == value.size() );
    }

    SECTION( "Node initialization" )
    {
        std::vector<uint32_t> value = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

        auto node = VectorValue<uint32_t>( scope, value );
        scope.Run( node );

        auto buffer2 = node.Get<vector_buffer_t>().mValue.Fetch<uint32_t>();
        REQUIRE( buffer2 == value );
    }
}

TEST_CASE( "TensorNode", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    SECTION( "Node allocation" )
    {
        constant_value_initializer_t initializer{};
        initializer.mValue = (uint8_t)3;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( int8_t ) ) );
        scope.Run( node );

        auto lBuffer1 = node.Get<multi_tensor_value_t>().mValue.BufferAt( 0 );
        auto buffer2  = node.Get<multi_tensor_value_t>().mValue.BufferAt( 1 );
        REQUIRE( lBuffer1.Size() == Prod( dim1 ) );
        REQUIRE( buffer2.Size() == Prod( dim2 ) );
    }

    SECTION( "Constant initializer (float)" )
    {
        constant_value_initializer_t initializer{};
        initializer.mValue = 3.0f;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        scope.Run( node );

        std::vector<float> expectedValues( node.Get<multi_tensor_value_t>().mValue.SizeAs<float>() );
        std::vector<float> tensorValues = node.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        for( auto &v : expectedValues )
        {
            v = 3.0f;
        }
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Constant initializer (double)" )
    {
        constant_value_initializer_t initializer{};
        initializer.mValue = (double)3.0f;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( double ) ) );
        scope.Run( node );

        std::vector<double> expectedValues( node.Get<multi_tensor_value_t>().mValue.SizeAs<double>() );
        std::vector<double> tensorValues = node.Get<multi_tensor_value_t>().mValue.FetchFlattened<double>();
        for( auto &v : expectedValues )
        {
            v = 3.0f;
        }
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Constant initializer (uint8_t)" )
    {
        constant_value_initializer_t initializer{};
        initializer.mValue = (uint8_t)3;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );
        scope.Run( node );

        std::vector<uint8_t> expectedValues( node.Get<multi_tensor_value_t>().mValue.SizeAs<uint8_t>() );
        std::vector<uint8_t> tensorValues = node.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
        for( auto &v : expectedValues )
        {
            v = 3;
        }
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Constant initializer (uint16_t)" )
    {
        constant_value_initializer_t initializer{};
        initializer.mValue = (uint16_t)256;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( uint16_t ) ) );
        scope.Run( node );

        std::vector<uint16_t> expectedValues( node.Get<multi_tensor_value_t>().mValue.SizeAs<uint16_t>() );
        std::vector<uint16_t> tensorValues = node.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint16_t>();
        for( auto &v : expectedValues )
        {
            v = 256;
        }
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Constant initializer (uint32_t)" )
    {
        constant_value_initializer_t initializer{};
        initializer.mValue = (uint32_t)1000000;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( uint32_t ) ) );
        scope.Run( node );

        std::vector<uint32_t> expectedValues( node.Get<multi_tensor_value_t>().mValue.SizeAs<uint32_t>() );
        std::vector<uint32_t> tensorValues = node.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint32_t>();
        for( auto &v : expectedValues )
        {
            v = 1000000;
        }
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Constant initializer (uint64_t)" )
    {
        constant_value_initializer_t initializer{};
        initializer.mValue = (uint64_t)10000000000;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( uint64_t ) ) );
        scope.Run( node );

        std::vector<uint64_t> expectedValues( node.Get<multi_tensor_value_t>().mValue.SizeAs<uint64_t>() );
        std::vector<uint64_t> tensorValues = node.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint64_t>();
        for( auto &v : expectedValues )
        {
            v = 10000000000;
        }
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Vector initializer (float)" )
    {
        vector_initializer_t initializer( std::vector<float>{ 4.0f, 5.0f } );

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        scope.Run( node );

        std::vector<float> expectedValues0( Prod( dim1 ) );
        std::vector<float> expectedValues1( Prod( dim2 ) );
        std::fill( expectedValues0.begin(), expectedValues0.end(), 4.0f );
        std::fill( expectedValues1.begin(), expectedValues1.end(), 5.0f );
        expectedValues0.insert( expectedValues0.end(), expectedValues1.begin(), expectedValues1.end() );

        std::vector<float> tensorValues = node.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        REQUIRE( VectorEqual( tensorValues, expectedValues0 ) );
    }

    SECTION( "Vector initializer (double)" )
    {
        vector_initializer_t initializer( std::vector<double>{ 4.0, 5.0 } );

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( double ) ) );
        scope.Run( node );

        std::vector<double> expectedValues0( Prod( dim1 ) );
        std::vector<double> expectedValues1( Prod( dim2 ) );
        std::fill( expectedValues0.begin(), expectedValues0.end(), (double)4.0 );
        std::fill( expectedValues1.begin(), expectedValues1.end(), (double)5.0 );
        expectedValues0.insert( expectedValues0.end(), expectedValues1.begin(), expectedValues1.end() );
        std::vector<double> tensorValues = node.Get<multi_tensor_value_t>().mValue.FetchFlattened<double>();

        REQUIRE( VectorEqual( tensorValues, expectedValues0 ) );
    }

    SECTION( "Vector initializer (uint8_t)" )
    {
        vector_initializer_t initializer( std::vector<uint8_t>{ 4, 5 } );

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );
        scope.Run( node );

        std::vector<uint8_t> expectedValues0( Prod( dim1 ) );
        std::vector<uint8_t> expectedValues1( Prod( dim2 ) );
        std::fill( expectedValues0.begin(), expectedValues0.end(), (uint8_t)4 );
        std::fill( expectedValues1.begin(), expectedValues1.end(), (uint8_t)5 );
        expectedValues0.insert( expectedValues0.end(), expectedValues1.begin(), expectedValues1.end() );

        std::vector<uint8_t> tensorValues = node.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
        REQUIRE( VectorEqual( tensorValues, expectedValues0 ) );
    }

    SECTION( "Vector initializer (uint16_t)" )
    {
        vector_initializer_t initializer( std::vector<uint16_t>{ 256, 512 } );

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( uint16_t ) ) );
        scope.Run( node );

        std::vector<uint16_t> expectedValues0( Prod( dim1 ) );
        std::vector<uint16_t> expectedValues1( Prod( dim2 ) );
        std::fill( expectedValues0.begin(), expectedValues0.end(), (uint16_t)256 );
        std::fill( expectedValues1.begin(), expectedValues1.end(), (uint16_t)512 );
        expectedValues0.insert( expectedValues0.end(), expectedValues1.begin(), expectedValues1.end() );
        std::vector<uint16_t> tensorValues = node.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint16_t>();
        REQUIRE( VectorEqual( tensorValues, expectedValues0 ) );
    }

    SECTION( "Vector initializer (uint32_t)" )
    {
        vector_initializer_t initializer( std::vector<uint32_t>{ 1234567, 7654321 } );

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( uint32_t ) ) );
        scope.Run( node );

        std::vector<uint32_t> expectedValues0( Prod( dim1 ) );
        std::vector<uint32_t> expectedValues1( Prod( dim2 ) );
        std::fill( expectedValues0.begin(), expectedValues0.end(), (uint32_t)1234567 );
        std::fill( expectedValues1.begin(), expectedValues1.end(), (uint32_t)7654321 );
        expectedValues0.insert( expectedValues0.end(), expectedValues1.begin(), expectedValues1.end() );
        std::vector<uint32_t> tensorValues = node.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint32_t>();
        REQUIRE( VectorEqual( tensorValues, expectedValues0 ) );
    }

    SECTION( "Vector initializer (uint64_t)" )
    {
        vector_initializer_t initializer( std::vector<uint64_t>{ 1234567890, 987654321 } );

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( uint64_t ) ) );
        scope.Run( node );

        std::vector<uint64_t> expectedValues0( Prod( dim1 ) );
        std::vector<uint64_t> expectedValues1( Prod( dim2 ) );
        std::fill( expectedValues0.begin(), expectedValues0.end(), (uint64_t)1234567890 );
        std::fill( expectedValues1.begin(), expectedValues1.end(), (uint64_t)987654321 );
        expectedValues0.insert( expectedValues0.end(), expectedValues1.begin(), expectedValues1.end() );
        std::vector<uint64_t> tensorValues = node.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint64_t>();
        REQUIRE( VectorEqual( tensorValues, expectedValues0 ) );
    }

    SECTION( "Data initializer (float)" )
    {
        std::vector<float> expectedValues{ 4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f,
                                           12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f };
        data_initializer_t initializer( expectedValues );

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        scope.Run( node );

        std::vector<float> tensorValues = node.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        REQUIRE( VectorEqual( expectedValues, tensorValues ) );
    }

    SECTION( "Data initializer (double)" )
    {
        std::vector<double> expectedValues{ 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0 };
        data_initializer_t  initializer( expectedValues );

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( double ) ) );
        scope.Run( node );

        std::vector<double> tensorValues = node.Get<multi_tensor_value_t>().mValue.FetchFlattened<double>();
        REQUIRE( VectorEqual( expectedValues, tensorValues ) );
    }

    SECTION( "Data initializer (uint8_t)" )
    {
        std::vector<uint8_t> expectedValues{ 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
        data_initializer_t   initializer( expectedValues );

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );
        scope.Run( node );

        std::vector<uint8_t> tensorValues = node.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
        REQUIRE( VectorEqual( expectedValues, tensorValues ) );
    }

    SECTION( "Data initializer (uint16_t)" )
    {
        std::vector<uint16_t> expectedValues{ 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
        data_initializer_t    initializer( expectedValues );

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( uint16_t ) ) );
        scope.Run( node );

        std::vector<uint16_t> tensorValues = node.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint16_t>();
        REQUIRE( VectorEqual( expectedValues, tensorValues ) );
    }

    SECTION( "Data initializer (uint32_t)" )
    {
        std::vector<uint32_t> expectedValues{ 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
        data_initializer_t    initializer( expectedValues );

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( uint32_t ) ) );
        scope.Run( node );

        std::vector<uint32_t> tensorValues = node.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint32_t>();
        REQUIRE( VectorEqual( expectedValues, tensorValues ) );
    }

    SECTION( "Data initializer (uint64_t)" )
    {
        std::vector<uint64_t> expectedValues{ 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
        data_initializer_t    initializer( expectedValues );

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( uint64_t ) ) );
        scope.Run( node );

        std::vector<uint64_t> tensorValues = node.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint64_t>();
        REQUIRE( VectorEqual( expectedValues, tensorValues ) );
    }

    SECTION( "Random uniform initializer (float)" )
    {
        random_uniform_initializer_t initializer{};
        initializer.mType = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        scope.Run( node );

        std::vector<float> expectedValues( node.Get<multi_tensor_value_t>().mValue.SizeAs<float>() );
        std::vector<float> tensorValues = node.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        for( auto &v : expectedValues )
        {
            v = 0.0f;
        }
        REQUIRE( tensorValues != expectedValues );
    }

    SECTION( "Random uniform initializer (double)" )
    {
        random_uniform_initializer_t initializer{};
        initializer.mType = scalar_type_t::FLOAT64;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( double ) ) );
        scope.Run( node );

        std::vector<double> expectedValues( node.Get<multi_tensor_value_t>().mValue.SizeAs<double>() );
        std::vector<double> tensorValues = node.Get<multi_tensor_value_t>().mValue.FetchFlattened<double>();
        for( auto &v : expectedValues )
        {
            v = 0.0;
        }
        REQUIRE( tensorValues != expectedValues );
    }

    SECTION( "Random normal initializer (float)" )
    {
        random_normal_initializer_t initializer{};
        initializer.mType = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        scope.Run( node );

        std::vector<float> expectedValues( node.Get<multi_tensor_value_t>().mValue.SizeAs<float>() );
        std::vector<float> tensorValues = node.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        for( auto &v : expectedValues )
        {
            v = 0.0f;
        }
        REQUIRE( tensorValues != expectedValues );
    }

    SECTION( "Random normal initializer (double)" )
    {
        random_normal_initializer_t initializer{};
        initializer.mType = scalar_type_t::FLOAT64;
        initializer.mMean = (double)0.0;
        initializer.mStd  = (double)1.0;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( double ) ) );
        scope.Run( node );

        std::vector<double> expectedValues( node.Get<multi_tensor_value_t>().mValue.SizeAs<double>() );
        std::vector<double> tensorValues = node.Get<multi_tensor_value_t>().mValue.FetchFlattened<double>();
        for( auto &v : expectedValues )
        {
            v = 0.0;
        }
        REQUIRE( tensorValues != expectedValues );
    }
}

TEST_CASE( "Arithmetic nodes", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    SECTION( "Add scalar to array (float)" )
    {
        random_normal_initializer_t initializer{};
        initializer.mType = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto opNode  = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto opSNode = ConstantScalarValue( scope, 1.234f );

        auto result0 = Add( scope, opNode, opSNode );
        auto result1 = Add( scope, opSNode, opNode );

        scope.Run( { result0, result1 } );

        std::vector<float> leftTensorValues = opNode.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        std::vector<float> expectedValues( opNode.Get<multi_tensor_value_t>().mValue.SizeAs<float>() );
        std::vector<float> tensorValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        std::vector<float> tensorValues1 = result1.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
        {
            expectedValues[i] = leftTensorValues[i] + 1.234f;
        }
        REQUIRE( VectorEqual( tensorValues0, expectedValues ) );
        REQUIRE( VectorEqual( tensorValues1, expectedValues ) );
    }

    SECTION( "Add array to array (float)" )
    {
        random_normal_initializer_t initializer{};
        initializer.mType = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto opNode  = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto opSNode = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto result0 = Add( scope, opNode, opSNode );

        scope.Run( result0 );

        std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        std::vector<float> rightTensorValues = opSNode.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        std::vector<float> expectedValues( opNode.Get<multi_tensor_value_t>().mValue.SizeAs<float>() );
        std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
        {
            expectedValues[i] = leftTensorValues[i] + rightTensorValues[i];
        }
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Add array to vector (float)" )
    {
        random_normal_initializer_t initializer{};
        initializer.mType = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto opNode = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        std::vector<scalar_value_t> constants{ 3.123f, 4.345f };
        auto                        opSNode = VectorValue( scope, constants );
        auto                        result0 = Add( scope, opNode, opSNode );
        auto                        result1 = Add( scope, opSNode, opNode );
        scope.Run( { result0, result1 } );

        {
            std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
            float              rightTensorValues = std::get<float>( constants[0] );
            std::vector<float> expectedValues( leftTensorValues.size() );
            std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
            for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
            {
                expectedValues[i] = leftTensorValues[i] + rightTensorValues;
            }
            REQUIRE( VectorEqual( tensorValues, expectedValues ) );
        }

        {
            std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
            float              rightTensorValues = std::get<float>( constants[1] );
            std::vector<float> expectedValues( leftTensorValues.size() );
            std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
            for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
            {
                expectedValues[i] = leftTensorValues[i] + rightTensorValues;
            }
            REQUIRE( VectorEqual( tensorValues, expectedValues ) );
        }

        {
            std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
            float              rightTensorValues = std::get<float>( constants[0] );
            std::vector<float> expectedValues( leftTensorValues.size() );
            std::vector<float> tensorValues = result1.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
            for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
            {
                expectedValues[i] = leftTensorValues[i] + rightTensorValues;
            }
            REQUIRE( VectorEqual( tensorValues, expectedValues ) );
        }

        {
            std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
            float              rightTensorValues = std::get<float>( constants[1] );
            std::vector<float> expectedValues( leftTensorValues.size() );
            std::vector<float> tensorValues = result1.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
            for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
            {
                expectedValues[i] = leftTensorValues[i] + rightTensorValues;
            }
            REQUIRE( VectorEqual( tensorValues, expectedValues ) );
        }
    }

    SECTION( "Multiply scalar by array (float)" )
    {
        random_normal_initializer_t initializer{};
        initializer.mType = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto opNode  = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto opSNode = ConstantScalarValue( scope, 1.234f );
        auto result0 = Multiply( scope, opNode, opSNode );
        auto result1 = Multiply( scope, opSNode, opNode );
        scope.Run( { result0, result1 } );

        std::vector<float> leftTensorValues = opNode.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        std::vector<float> expectedValues( opNode.Get<multi_tensor_value_t>().mValue.SizeAs<float>() );
        std::vector<float> tensorValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        std::vector<float> tensorValues1 = result1.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
        {
            expectedValues[i] = leftTensorValues[i] * 1.234f;
        }
        REQUIRE( VectorEqual( tensorValues0, expectedValues ) );
        REQUIRE( VectorEqual( tensorValues1, expectedValues ) );
    }

    SECTION( "Multiply array by array (float)" )
    {
        random_normal_initializer_t initializer{};
        initializer.mType = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto opNode  = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto opSNode = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto result0 = Multiply( scope, opNode, opSNode );
        scope.Run( result0 );

        std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        std::vector<float> rightTensorValues = opSNode.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        std::vector<float> expectedValues( opNode.Get<multi_tensor_value_t>().mValue.SizeAs<float>() );
        std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
        {
            expectedValues[i] = leftTensorValues[i] * rightTensorValues[i];
        }
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Multiply array by vector (float)" )
    {
        random_normal_initializer_t initializer{};
        initializer.mType = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto opNode = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        std::vector<scalar_value_t> constants{ 3.123f, 4.345f };
        auto                        opSNode = VectorValue( scope, constants );
        auto                        result0 = Multiply( scope, opNode, opSNode );
        auto                        result1 = Multiply( scope, opSNode, opNode );
        scope.Run( { result0, result1 } );

        {
            std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
            float              rightTensorValues = std::get<float>( constants[0] );
            std::vector<float> expectedValues( leftTensorValues.size() );
            std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
            for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
            {
                expectedValues[i] = leftTensorValues[i] * rightTensorValues;
            }
            REQUIRE( VectorEqual( tensorValues, expectedValues ) );
        }

        {
            std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
            float              rightTensorValues = std::get<float>( constants[1] );
            std::vector<float> expectedValues( leftTensorValues.size() );
            std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
            for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
            {
                expectedValues[i] = leftTensorValues[i] * rightTensorValues;
            }
            REQUIRE( VectorEqual( tensorValues, expectedValues ) );
        }

        {
            std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
            float              rightTensorValues = std::get<float>( constants[0] );
            std::vector<float> expectedValues( leftTensorValues.size() );
            std::vector<float> tensorValues = result1.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
            for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
            {
                expectedValues[i] = leftTensorValues[i] * rightTensorValues;
            }
            REQUIRE( VectorEqual( tensorValues, expectedValues ) );
        }

        {
            std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
            float              rightTensorValues = std::get<float>( constants[1] );
            std::vector<float> expectedValues( leftTensorValues.size() );
            std::vector<float> tensorValues = result1.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
            for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
            {
                expectedValues[i] = leftTensorValues[i] * rightTensorValues;
            }
            REQUIRE( VectorEqual( tensorValues, expectedValues ) );
        }
    }

    SECTION( "Subtract scalar from array (float)" )
    {
        random_normal_initializer_t initializer{};
        initializer.mType = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto opNode  = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto opSNode = ConstantScalarValue( scope, 1.234f );
        auto result0 = Subtract( scope, opNode, opSNode );
        auto result1 = Subtract( scope, opSNode, opNode );
        scope.Run( { result0, result1 } );

        std::vector<float> leftTensorValues = opNode.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        std::vector<float> expectedValues0( opNode.Get<multi_tensor_value_t>().mValue.SizeAs<float>() );
        std::vector<float> expectedValues1( opNode.Get<multi_tensor_value_t>().mValue.SizeAs<float>() );
        std::vector<float> tensorValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        std::vector<float> tensorValues1 = result1.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
        {
            expectedValues0[i] = leftTensorValues[i] - 1.234f;
            expectedValues1[i] = 1.234f - leftTensorValues[i];
        }
        REQUIRE( VectorEqual( tensorValues0, expectedValues0 ) );
        REQUIRE( VectorEqual( tensorValues1, expectedValues1 ) );
    }

    SECTION( "Subtract vector from array (float)" )
    {
        random_normal_initializer_t initializer{};
        initializer.mType = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto opNode = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        std::vector<scalar_value_t> constants{ 3.123f, 4.345f };
        auto                        opSNode = VectorValue( scope, constants );
        auto                        result0 = Subtract( scope, opNode, opSNode );
        auto                        result1 = Subtract( scope, opSNode, opNode );
        scope.Run( { result0, result1 } );

        {
            std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
            float              rightTensorValues = std::get<float>( constants[0] );
            std::vector<float> expectedValues( leftTensorValues.size() );
            std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
            for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
            {
                expectedValues[i] = leftTensorValues[i] - rightTensorValues;
            }
            REQUIRE( VectorEqual( tensorValues, expectedValues ) );
        }

        {
            std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
            float              rightTensorValues = std::get<float>( constants[1] );
            std::vector<float> expectedValues( leftTensorValues.size() );
            std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
            for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
            {
                expectedValues[i] = leftTensorValues[i] - rightTensorValues;
            }
            REQUIRE( VectorEqual( tensorValues, expectedValues ) );
        }

        {
            std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
            float              rightTensorValues = std::get<float>( constants[0] );
            std::vector<float> expectedValues( leftTensorValues.size() );
            std::vector<float> tensorValues = result1.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
            for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
            {
                expectedValues[i] = rightTensorValues - leftTensorValues[i];
            }
            REQUIRE( VectorEqual( tensorValues, expectedValues ) );
        }

        {
            std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
            float              rightTensorValues = std::get<float>( constants[1] );
            std::vector<float> expectedValues( leftTensorValues.size() );
            std::vector<float> tensorValues = result1.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
            for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
            {
                expectedValues[i] = rightTensorValues - leftTensorValues[i];
            }
            REQUIRE( VectorEqual( tensorValues, expectedValues ) );
        }
    }

    SECTION( "Subtract array from array (float)" )
    {
        random_normal_initializer_t initializer{};
        initializer.mType = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto opNode  = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto opSNode = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto result0 = Subtract( scope, opNode, opSNode );
        scope.Run( result0 );

        std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        std::vector<float> rightTensorValues = opSNode.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        std::vector<float> expectedValues( opNode.Get<multi_tensor_value_t>().mValue.SizeAs<float>() );
        std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
        {
            expectedValues[i] = leftTensorValues[i] - rightTensorValues[i];
        }
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Divide vector by array (float)" )
    {
        random_normal_initializer_t initializer{};
        initializer.mType = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto opNode = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        std::vector<scalar_value_t> constants{ 3.123f, 4.345f };
        auto                        opSNode = VectorValue( scope, constants );
        auto                        result0 = Divide( scope, opNode, opSNode );
        auto                        result1 = Divide( scope, opSNode, opNode );
        scope.Run( { result0, result1 } );

        {
            std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
            float              rightTensorValues = std::get<float>( constants[0] );
            std::vector<float> expectedValues( leftTensorValues.size() );
            std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
            for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
            {
                expectedValues[i] = leftTensorValues[i] / rightTensorValues;
            }
            REQUIRE( VectorEqual( tensorValues, expectedValues ) );
        }

        {
            std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
            float              rightTensorValues = std::get<float>( constants[1] );
            std::vector<float> expectedValues( leftTensorValues.size() );
            std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
            for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
            {
                expectedValues[i] = leftTensorValues[i] / rightTensorValues;
            }
            REQUIRE( VectorEqual( tensorValues, expectedValues ) );
        }

        {
            std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
            float              rightTensorValues = std::get<float>( constants[0] );
            std::vector<float> expectedValues( leftTensorValues.size() );
            std::vector<float> tensorValues = result1.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
            for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
            {
                expectedValues[i] = rightTensorValues / leftTensorValues[i];
            }
            REQUIRE( VectorEqual( tensorValues, expectedValues ) );
        }

        {
            std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
            float              rightTensorValues = std::get<float>( constants[1] );
            std::vector<float> expectedValues( leftTensorValues.size() );
            std::vector<float> tensorValues = result1.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
            for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
            {
                expectedValues[i] = rightTensorValues / leftTensorValues[i];
            }
            REQUIRE( VectorEqual( tensorValues, expectedValues ) );
        }
    }
}

TEMPLATE_TEST_CASE( "DIVIDE Array_Scalar", "[CORE_COMPUTATION_GRAPH]", uint16_t, uint32_t, uint64_t, int16_t, int32_t, int64_t, float,
                    double )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t>              dim1{ 7, 3, 1400 };
    std::vector<std::vector<TestType>> lValues1 = RandomVector<TestType>( dim1, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t>              dim2{ 2, 7, 700 };
    std::vector<std::vector<TestType>> lValues2 = RandomVector<TestType>( dim2, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t>              lDim3{ 3, 5, 200 };
    std::vector<std::vector<TestType>> lValues3 = RandomVector<TestType>( lDim3, 0, std::numeric_limits<TestType>::max() );

    auto lScalarNode = ConstantScalarValue( scope, static_cast<TestType>( 13 ) );

    std::vector<TestType> expectedValues;
    auto                  expectedValues1 =
        BroadcastMap<TestType>( lValues1, static_cast<TestType>( 13 ), []( TestType x, TestType y ) { return x / y; } );
    auto lExpectedValues2 =
        BroadcastMap<TestType>( lValues2, static_cast<TestType>( 13 ), []( TestType x, TestType y ) { return x / y; } );
    auto lExpectedValues3 =
        BroadcastMap<TestType>( lValues3, static_cast<TestType>( 13 ), []( TestType x, TestType y ) { return x / y; } );

    expectedValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( expectedValues1 ), ConcatenateVectors( lExpectedValues2 ), ConcatenateVectors( lExpectedValues3 ) } );

    std::vector<TestType> lInputValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lValues1 ), ConcatenateVectors( lValues2 ), ConcatenateVectors( lValues3 ) } );

    data_initializer_t lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( scope, lInputInitializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( TestType ) ) );

    auto result0 = Divide( scope, lInputTensor, lScalarNode );
    scope.Run( result0 );

    auto lOutputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.Rank == 3 );
    REQUIRE( lOutputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 1400 } );
    REQUIRE( lOutputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 700 } );
    REQUIRE( lOutputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 200 } );

    std::vector<TestType> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<TestType>();
    REQUIRE( lResultValues0 == expectedValues );
}

TEST_CASE( "Tensor AND Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> dim1{ 29, 12, 23 };
    std::vector<uint32_t> dim2{ 33, 14, 13 };

    std::vector<uint8_t> lValues0  = RandomBool( 29 * 12 * 23 );
    std::vector<uint8_t> lValues01 = RandomBool( 33 * 14 * 13 );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    data_initializer_t lInitializer0( lValues0 );
    auto               lOpNodeLeft = MultiTensorValue( scope, lInitializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    std::vector<uint8_t> lValues1  = RandomBool( 29 * 12 * 23 );
    std::vector<uint8_t> lValues11 = RandomBool( 33 * 14 * 13 );
    lValues1.insert( lValues1.end(), lValues11.begin(), lValues11.end() );
    data_initializer_t lInitializer1( lValues1 );
    auto               lOpNodeRight = MultiTensorValue( scope, lInitializer1, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    auto result0 = And( scope, lOpNodeLeft, lOpNodeRight );
    auto result1 = And( scope, lOpNodeRight, lOpNodeLeft );
    scope.Run( { result0, result1 } );

    std::vector<uint8_t> expectedValues( lValues0.size() );
    for( uint32_t i = 0; i < lValues0.size(); i++ )
    {
        expectedValues[i] = ( lValues0[i] && lValues1[i] );
    }
    std::vector<uint8_t> tensorValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( tensorValues0 == expectedValues );

    std::vector<uint8_t> tensorValues1 = result1.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( tensorValues1 == expectedValues );
}

TEST_CASE( "Tensor AND Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> dim1{ 2, 12, 23 };
    std::vector<uint32_t> dim2{ 3, 14, 13 };

    std::vector<uint8_t> Values00  = RandomBool( 2 * 12 * 23 );
    std::vector<uint8_t> lValues01 = RandomBool( 3 * 14 * 13 );
    std::vector<uint8_t> lValues0;
    lValues0.insert( lValues0.end(), Values00.begin(), Values00.end() );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    data_initializer_t lInitializer0( lValues0 );
    auto               lOpNodeLeft = MultiTensorValue( scope, lInitializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    std::vector<scalar_value_t> constants{ static_cast<uint8_t>( 0 ), static_cast<uint8_t>( 1 ) };
    auto                        lOpNodeRight = VectorValue( scope, constants );

    auto result0 = And( scope, lOpNodeLeft, lOpNodeRight );
    auto result1 = And( scope, lOpNodeRight, lOpNodeLeft );
    scope.Run( { result0, result1 } );

    std::vector<uint8_t> expectedValues0( Values00.size() );
    for( uint32_t i = 0; i < Values00.size(); i++ )
    {
        expectedValues0[i] = ( Values00[i] && std::get<uint8_t>( constants[0] ) );
    }

    std::vector<uint8_t> ExpectedValues1( lValues01.size() );
    for( uint32_t i = 0; i < lValues01.size(); i++ )
    {
        ExpectedValues1[i] = ( lValues01[i] && std::get<uint8_t>( constants[1] ) );
    }

    expectedValues0.insert( expectedValues0.end(), ExpectedValues1.begin(), ExpectedValues1.end() );

    std::vector<uint8_t> tensorValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( tensorValues0 == expectedValues0 );

    std::vector<uint8_t> tensorValues1 = result1.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( tensorValues1 == expectedValues0 );
}

TEST_CASE( "Tensor AND Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> dim1{ 2, 12, 23 };
    std::vector<uint32_t> dim2{ 3, 14, 13 };

    std::vector<uint8_t> Values00  = RandomBool( 2 * 12 * 23 );
    std::vector<uint8_t> lValues01 = RandomBool( 3 * 14 * 13 );
    std::vector<uint8_t> lValues0;
    lValues0.insert( lValues0.end(), Values00.begin(), Values00.end() );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    data_initializer_t lInitializer0( lValues0 );
    auto               lOpNodeLeft = MultiTensorValue( scope, lInitializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    {
        auto lOpNodeRight = ConstantScalarValue( scope, static_cast<uint8_t>( 0 ) );

        auto result0 = And( scope, lOpNodeLeft, lOpNodeRight );
        auto result1 = And( scope, lOpNodeRight, lOpNodeLeft );
        scope.Run( { result0, result1 } );

        std::vector<uint8_t> expectedValues( lValues0.size() );
        std::fill( expectedValues.begin(), expectedValues.end(), static_cast<uint8_t>( 0 ) );

        std::vector<uint8_t> tensorValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
        REQUIRE( tensorValues0 == expectedValues );

        std::vector<uint8_t> tensorValues1 = result1.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
        REQUIRE( tensorValues1 == expectedValues );
    }

    {
        auto lOpNodeRight = ConstantScalarValue( scope, static_cast<uint8_t>( 1 ) );

        auto result0 = And( scope, lOpNodeLeft, lOpNodeRight );
        auto result1 = And( scope, lOpNodeRight, lOpNodeLeft );
        scope.Run( { result0, result1 } );

        std::vector<uint8_t> expectedValues( lValues0.size() );

        for( uint32_t i = 0; i < lValues0.size(); i++ )
        {
            expectedValues[i] = lValues0[i];
        }

        std::vector<uint8_t> tensorValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
        REQUIRE( tensorValues0 == expectedValues );

        std::vector<uint8_t> tensorValues1 = result1.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
        REQUIRE( tensorValues1 == expectedValues );
    }
}

TEST_CASE( "Tensor OR Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> dim1{ 29, 12, 23 };
    std::vector<uint32_t> dim2{ 33, 14, 13 };

    std::vector<uint8_t> lValues0  = RandomBool( 29 * 12 * 23 );
    std::vector<uint8_t> lValues01 = RandomBool( 33 * 14 * 13 );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    data_initializer_t lInitializer0( lValues0 );
    auto               lOpNodeLeft = MultiTensorValue( scope, lInitializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    std::vector<uint8_t> lValues1  = RandomBool( 29 * 12 * 23 );
    std::vector<uint8_t> lValues11 = RandomBool( 33 * 14 * 13 );
    lValues1.insert( lValues1.end(), lValues11.begin(), lValues11.end() );
    data_initializer_t lInitializer1( lValues1 );
    auto               lOpNodeRight = MultiTensorValue( scope, lInitializer1, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    auto result0 = Or( scope, lOpNodeLeft, lOpNodeRight );
    auto result1 = Or( scope, lOpNodeRight, lOpNodeLeft );
    scope.Run( { result0, result1 } );

    std::vector<uint8_t> expectedValues( lValues0.size() );
    for( uint32_t i = 0; i < lValues0.size(); i++ )
    {
        expectedValues[i] = ( lValues0[i] || lValues1[i] );
    }

    std::vector<uint8_t> tensorValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( tensorValues0 == expectedValues );

    std::vector<uint8_t> tensorValues1 = result1.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( tensorValues1 == expectedValues );
}

TEST_CASE( "Tensor OR Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> dim1{ 2, 12, 23 };
    std::vector<uint32_t> dim2{ 3, 14, 13 };

    std::vector<uint8_t> Values00  = RandomBool( 2 * 12 * 23 );
    std::vector<uint8_t> lValues01 = RandomBool( 3 * 14 * 13 );
    std::vector<uint8_t> lValues0;
    lValues0.insert( lValues0.end(), Values00.begin(), Values00.end() );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    data_initializer_t lInitializer0( lValues0 );
    auto               lOpNodeLeft = MultiTensorValue( scope, lInitializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    std::vector<scalar_value_t> constants{ static_cast<uint8_t>( 0 ), static_cast<uint8_t>( 1 ) };
    auto                        lOpNodeRight = VectorValue( scope, constants );

    auto result0 = Or( scope, lOpNodeLeft, lOpNodeRight );
    auto result1 = Or( scope, lOpNodeRight, lOpNodeLeft );
    scope.Run( { result0, result1 } );

    std::vector<uint8_t> expectedValues0( Values00.size() );
    for( uint32_t i = 0; i < Values00.size(); i++ )
    {
        expectedValues0[i] = ( Values00[i] || std::get<uint8_t>( constants[0] ) );
    }

    std::vector<uint8_t> ExpectedValues1( lValues01.size() );
    for( uint32_t i = 0; i < lValues01.size(); i++ )
    {
        ExpectedValues1[i] = ( lValues01[i] || std::get<uint8_t>( constants[1] ) );
    }

    expectedValues0.insert( expectedValues0.end(), ExpectedValues1.begin(), ExpectedValues1.end() );

    std::vector<uint8_t> tensorValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( tensorValues0 == expectedValues0 );

    std::vector<uint8_t> tensorValues1 = result1.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( tensorValues1 == expectedValues0 );
}

TEST_CASE( "Tensor OR Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> dim1{ 2, 12, 23 };
    std::vector<uint32_t> dim2{ 3, 14, 13 };

    std::vector<uint8_t> Values00  = RandomBool( 2 * 12 * 23 );
    std::vector<uint8_t> lValues01 = RandomBool( 3 * 14 * 13 );
    std::vector<uint8_t> lValues0;
    lValues0.insert( lValues0.end(), Values00.begin(), Values00.end() );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    data_initializer_t lInitializer0( lValues0 );
    auto               lOpNodeLeft = MultiTensorValue( scope, lInitializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    {
        auto lOpNodeRight = ConstantScalarValue( scope, static_cast<uint8_t>( 1 ) );

        auto result0 = Or( scope, lOpNodeLeft, lOpNodeRight );
        auto result1 = Or( scope, lOpNodeRight, lOpNodeLeft );
        scope.Run( { result0, result1 } );

        std::vector<uint8_t> expectedValues( lValues0.size() );
        std::fill( expectedValues.begin(), expectedValues.end(), static_cast<uint8_t>( 1 ) );

        std::vector<uint8_t> tensorValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
        REQUIRE( tensorValues0 == expectedValues );

        std::vector<uint8_t> tensorValues1 = result1.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
        REQUIRE( tensorValues1 == expectedValues );
    }

    {
        auto lOpNodeRight = ConstantScalarValue( scope, static_cast<uint8_t>( 0 ) );

        auto result0 = Or( scope, lOpNodeLeft, lOpNodeRight );
        auto result1 = Or( scope, lOpNodeRight, lOpNodeLeft );
        scope.Run( { result0, result1 } );

        std::vector<uint8_t> expectedValues( lValues0.size() );

        for( uint32_t i = 0; i < lValues0.size(); i++ )
        {
            expectedValues[i] = lValues0[i];
        }

        std::vector<uint8_t> tensorValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
        REQUIRE( tensorValues0 == expectedValues );

        std::vector<uint8_t> tensorValues1 = result1.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
        REQUIRE( tensorValues1 == expectedValues );
    }
}

TEST_CASE( "NOT Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> dim1{ 29, 12, 23 };
    std::vector<uint32_t> dim2{ 33, 14, 13 };

    std::vector<uint8_t> lValues0  = RandomBool( 29 * 12 * 23 );
    std::vector<uint8_t> lValues01 = RandomBool( 33 * 14 * 13 );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    data_initializer_t lInitializer0( lValues0 );
    auto               lOpNodeLeft = MultiTensorValue( scope, lInitializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    auto result0 = Not( scope, lOpNodeLeft );
    scope.Run( result0 );

    std::vector<uint8_t> tensorValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues( lValues0.size() );
    for( uint32_t i = 0; i < lValues0.size(); i++ )
    {
        expectedValues[i] = !( lValues0[i] );
    }
    REQUIRE( tensorValues == expectedValues );
}

TEST_CASE( "Tensor BITWISE_AND Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> dim1{ 29, 12, 23 };
    std::vector<uint32_t> dim2{ 33, 14, 13 };

    auto lValues0  = RandomNumber<uint64_t>( 29 * 12 * 23 );
    auto lValues01 = RandomNumber<uint64_t>( 33 * 14 * 13 );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    data_initializer_t lInitializer0( lValues0 );
    auto               lOpNodeLeft = MultiTensorValue( scope, lInitializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint64_t ) ) );

    auto lValues1  = RandomNumber<uint64_t>( 29 * 12 * 23 );
    auto lValues11 = RandomNumber<uint64_t>( 33 * 14 * 13 );
    lValues1.insert( lValues1.end(), lValues11.begin(), lValues11.end() );
    data_initializer_t lInitializer1( lValues1 );
    auto               lOpNodeRight = MultiTensorValue( scope, lInitializer1, tensor_shape_t( { dim1, dim2 }, sizeof( uint64_t ) ) );

    auto result0 = BitwiseAnd( scope, lOpNodeLeft, lOpNodeRight );
    auto result1 = BitwiseAnd( scope, lOpNodeRight, lOpNodeLeft );
    scope.Run( { result0, result1 } );

    std::vector<uint64_t> expectedValues( lValues0.size() );
    for( uint32_t i = 0; i < lValues0.size(); i++ )
    {
        expectedValues[i] = ( lValues0[i] & lValues1[i] );
    }
    auto tensorValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( tensorValues0 == expectedValues );

    auto tensorValues1 = result1.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( tensorValues1 == expectedValues );
}

TEST_CASE( "Tensor BITWISE_AND Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> dim1{ 2, 12, 23 };
    std::vector<uint32_t> dim2{ 3, 14, 13 };

    auto                  Values00  = RandomNumber<uint64_t>( 2 * 12 * 23 );
    auto                  lValues01 = RandomNumber<uint64_t>( 3 * 14 * 13 );
    std::vector<uint64_t> lValues0;
    lValues0.insert( lValues0.end(), Values00.begin(), Values00.end() );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    data_initializer_t lInitializer0( lValues0 );
    auto               lOpNodeLeft = MultiTensorValue( scope, lInitializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint64_t ) ) );

    std::vector<scalar_value_t> constants{ static_cast<uint64_t>( 0x1b34d765ef12acac ), static_cast<uint64_t>( 0x1b34d065ef120cfc ) };
    auto                        lOpNodeRight = VectorValue( scope, constants );

    auto result0 = BitwiseAnd( scope, lOpNodeLeft, lOpNodeRight );
    auto result1 = BitwiseAnd( scope, lOpNodeRight, lOpNodeLeft );
    scope.Run( { result0, result1 } );

    std::vector<uint64_t> expectedValues0( Values00.size() );
    for( uint32_t i = 0; i < Values00.size(); i++ )
    {
        expectedValues0[i] = ( Values00[i] & std::get<uint64_t>( constants[0] ) );
    }

    std::vector<uint64_t> ExpectedValues1( lValues01.size() );
    for( uint32_t i = 0; i < lValues01.size(); i++ )
    {
        ExpectedValues1[i] = ( lValues01[i] & std::get<uint64_t>( constants[1] ) );
    }

    expectedValues0.insert( expectedValues0.end(), ExpectedValues1.begin(), ExpectedValues1.end() );

    auto tensorValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( tensorValues0 == expectedValues0 );

    auto tensorValues1 = result1.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( tensorValues1 == expectedValues0 );
}

TEST_CASE( "Tensor BITWISE_AND Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> dim1{ 2, 12, 23 };
    std::vector<uint32_t> dim2{ 3, 14, 13 };

    auto                  Values00  = RandomNumber<uint64_t>( 2 * 12 * 23 );
    auto                  lValues01 = RandomNumber<uint64_t>( 3 * 14 * 13 );
    std::vector<uint64_t> lValues0;
    lValues0.insert( lValues0.end(), Values00.begin(), Values00.end() );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    data_initializer_t lInitializer0( lValues0 );
    auto               lOpNodeLeft = MultiTensorValue( scope, lInitializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint64_t ) ) );

    auto lOpNodeRight = ConstantScalarValue( scope, static_cast<uint64_t>( 0x1b34d765ef12acac ) );

    auto result0 = BitwiseAnd( scope, lOpNodeLeft, lOpNodeRight );
    auto result1 = BitwiseAnd( scope, lOpNodeRight, lOpNodeLeft );
    scope.Run( { result0, result1 } );

    std::vector<uint64_t> expectedValues( lValues0.size() );
    for( uint32_t i = 0; i < lValues0.size(); i++ )
    {
        expectedValues[i] = ( lValues0[i] & static_cast<uint64_t>( 0x1b34d765ef12acac ) );
    }

    auto tensorValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( tensorValues0 == expectedValues );

    auto tensorValues1 = result1.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( tensorValues1 == expectedValues );
}

TEST_CASE( "Tensor BITWISE_OR Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> dim1{ 29, 12, 23 };
    std::vector<uint32_t> dim2{ 33, 14, 13 };

    auto lValues0  = RandomNumber<uint64_t>( 29 * 12 * 23 );
    auto lValues01 = RandomNumber<uint64_t>( 33 * 14 * 13 );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    data_initializer_t lInitializer0( lValues0 );
    auto               lOpNodeLeft = MultiTensorValue( scope, lInitializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint64_t ) ) );

    auto lValues1  = RandomNumber<uint64_t>( 29 * 12 * 23 );
    auto lValues11 = RandomNumber<uint64_t>( 33 * 14 * 13 );
    lValues1.insert( lValues1.end(), lValues11.begin(), lValues11.end() );
    data_initializer_t lInitializer1( lValues1 );
    auto               lOpNodeRight = MultiTensorValue( scope, lInitializer1, tensor_shape_t( { dim1, dim2 }, sizeof( uint64_t ) ) );

    auto result0 = BitwiseOr( scope, lOpNodeLeft, lOpNodeRight );
    auto result1 = BitwiseOr( scope, lOpNodeRight, lOpNodeLeft );
    scope.Run( { result0, result1 } );

    std::vector<uint64_t> expectedValues( lValues0.size() );
    for( uint32_t i = 0; i < lValues0.size(); i++ )
    {
        expectedValues[i] = ( lValues0[i] | lValues1[i] );
    }

    auto tensorValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( tensorValues0 == expectedValues );

    auto tensorValues1 = result1.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( tensorValues1 == expectedValues );
}

TEST_CASE( "Tensor BITWISE_OR Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> dim1{ 2, 12, 23 };
    std::vector<uint32_t> dim2{ 3, 14, 13 };

    auto                  Values00  = RandomNumber<uint64_t>( 2 * 12 * 23 );
    auto                  lValues01 = RandomNumber<uint64_t>( 3 * 14 * 13 );
    std::vector<uint64_t> lValues0;
    lValues0.insert( lValues0.end(), Values00.begin(), Values00.end() );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    data_initializer_t lInitializer0( lValues0 );
    auto               lOpNodeLeft = MultiTensorValue( scope, lInitializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint64_t ) ) );

    std::vector<scalar_value_t> constants{ static_cast<uint64_t>( 0x1b34d765ef12acac ), static_cast<uint64_t>( 0x1b34d065ef120cfc ) };
    auto                        lOpNodeRight = VectorValue( scope, constants );

    auto result0 = BitwiseOr( scope, lOpNodeLeft, lOpNodeRight );
    auto result1 = BitwiseOr( scope, lOpNodeRight, lOpNodeLeft );
    scope.Run( { result0, result1 } );

    std::vector<uint64_t> expectedValues0( Values00.size() );
    for( uint32_t i = 0; i < Values00.size(); i++ )
    {
        expectedValues0[i] = ( Values00[i] | std::get<uint64_t>( constants[0] ) );
    }

    std::vector<uint64_t> ExpectedValues1( lValues01.size() );
    for( uint32_t i = 0; i < lValues01.size(); i++ )
    {
        ExpectedValues1[i] = ( lValues01[i] | std::get<uint64_t>( constants[1] ) );
    }

    expectedValues0.insert( expectedValues0.end(), ExpectedValues1.begin(), ExpectedValues1.end() );

    auto tensorValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( tensorValues0 == expectedValues0 );

    auto tensorValues1 = result1.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( tensorValues1 == expectedValues0 );
}

TEST_CASE( "Tensor BITWISE_OR Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> dim1{ 2, 12, 23 };
    std::vector<uint32_t> dim2{ 3, 14, 13 };

    auto                  Values00  = RandomNumber<uint64_t>( 2 * 12 * 23 );
    auto                  lValues01 = RandomNumber<uint64_t>( 3 * 14 * 13 );
    std::vector<uint64_t> lValues0;
    lValues0.insert( lValues0.end(), Values00.begin(), Values00.end() );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    data_initializer_t lInitializer0( lValues0 );
    auto               lOpNodeLeft = MultiTensorValue( scope, lInitializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint64_t ) ) );

    auto lOpNodeRight = ConstantScalarValue( scope, static_cast<uint64_t>( 0x1b34d765ef12acac ) );

    auto result0 = BitwiseOr( scope, lOpNodeLeft, lOpNodeRight );
    auto result1 = BitwiseOr( scope, lOpNodeRight, lOpNodeLeft );
    scope.Run( { result0, result1 } );

    std::vector<uint64_t> expectedValues( lValues0.size() );
    for( uint32_t i = 0; i < lValues0.size(); i++ )
    {
        expectedValues[i] = ( lValues0[i] | static_cast<uint64_t>( 0x1b34d765ef12acac ) );
    }

    auto tensorValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( tensorValues0 == expectedValues );

    auto tensorValues1 = result1.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( tensorValues1 == expectedValues );
}

TEST_CASE( "BITWISE_NOT Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> dim1{ 29, 12, 23 };
    std::vector<uint32_t> dim2{ 33, 14, 13 };

    auto lValues0  = RandomNumber<uint64_t>( 29 * 12 * 23 );
    auto lValues01 = RandomNumber<uint64_t>( 33 * 14 * 13 );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    data_initializer_t lInitializer0( lValues0 );
    auto               lOpNodeLeft = MultiTensorValue( scope, lInitializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint64_t ) ) );

    auto result0 = BitwiseNot( scope, lOpNodeLeft );
    scope.Run( result0 );

    auto                  tensorValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint64_t>();
    std::vector<uint64_t> expectedValues( lValues0.size() );
    for( uint32_t i = 0; i < lValues0.size(); i++ )
    {
        expectedValues[i] = ~( lValues0[i] );
    }
    REQUIRE( tensorValues == expectedValues );
}

TEST_CASE( "Affine transform node", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    SECTION( "Affine transform tensor/tensor/tensor (float)" )
    {
        random_uniform_initializer_t initializer{};
        initializer.mType = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 8, 12 };
        std::vector<uint32_t> dim2{ 8, 16 };

        auto lNodeA = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto lNodeX = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto lNodeB = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

        auto result0 = AffineTransform( scope, lNodeA, lNodeX, lNodeB );
        scope.Run( result0 );

        std::vector<float> lValues_A = lNodeA.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        std::vector<float> lValues_X = lNodeX.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        std::vector<float> lValues_B = lNodeB.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        std::vector<float> expectedValues( lNodeX.Get<multi_tensor_value_t>().mValue.SizeAs<float>() );
        for( uint32_t i = 0; i < lValues_X.size(); i++ )
        {
            expectedValues[i] = lValues_X[i] * lValues_A[i] + lValues_B[i];
        }
        std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();

        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Affine transform tensor/tensor/vector (float)" )
    {
        random_uniform_initializer_t initializer{};
        initializer.mType = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 8, 12 };
        std::vector<uint32_t> dim2{ 8, 16 };

        std::vector<float>          lBValues{ 2.142983764918237649f, 3.234987659834765f };
        std::vector<scalar_value_t> lB( 5 );
        lB[0] = lBValues[0];
        lB[1] = lBValues[1];

        auto lNodeA = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto lNodeX = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto lNodeB = VectorValue( scope, lB );

        auto result0 = AffineTransform( scope, lNodeA, lNodeX, lNodeB );
        scope.Run( result0 );

        std::vector<float> expectedValues = {};
        for( uint32_t i = 0; i < lBValues.size(); i++ )
        {
            uint32_t           size = lNodeX.Get<multi_tensor_value_t>().Shape().GetBufferSizeAs<float>( i ).Size;
            std::vector<float> lA   = lNodeA.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( i );
            std::vector<float> lX   = lNodeX.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( i );
            std::vector<float> lValues( size );
            for( uint32_t j = 0; j < size; j++ )
            {
                lValues[j] = lA[j] * lX[j] + lBValues[i];
            }
            expectedValues.insert( expectedValues.end(), lValues.begin(), lValues.end() );
        }

        std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Affine transform tensor/tensor/scalar (float)" )
    {
        random_uniform_initializer_t initializer{};
        initializer.mType = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 8, 12 };
        std::vector<uint32_t> dim2{ 8, 16 };

        float lScalarB = 2.142983764918237649f;
        auto  lNodeA   = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto  lNodeX   = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto  lNodeB   = ConstantScalarValue( scope, lScalarB );

        auto result0 = AffineTransform( scope, lNodeA, lNodeX, lNodeB );
        scope.Run( result0 );

        std::vector<float> expectedValues = {};
        for( uint32_t i = 0; i < lNodeX.Get<multi_tensor_value_t>().Shape().CountLayers(); i++ )
        {
            uint32_t           size = lNodeX.Get<multi_tensor_value_t>().Shape().GetBufferSizeAs<float>( i ).Size;
            std::vector<float> lA   = lNodeA.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( i );
            std::vector<float> lX   = lNodeX.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( i );
            std::vector<float> lValues( size );
            for( uint32_t j = 0; j < size; j++ )
            {
                lValues[j] = lA[j] * lX[j] + lScalarB;
            }
            expectedValues.insert( expectedValues.end(), lValues.begin(), lValues.end() );
        }

        std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Affine transform vector/tensor/tensor (float)" )
    {
        random_uniform_initializer_t initializer{};
        initializer.mType = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 8, 22 };
        std::vector<uint32_t> dim2{ 8, 64 };

        std::vector<float> lAValues{ 1.109238740928374f, 2.12398471982364f };
        std::vector<float> lBValues{ 2.142983764918237649f, 3.234987659834765f };

        std::vector<scalar_value_t> lA( 5 );
        lA[0] = lAValues[0];
        lA[1] = lAValues[1];

        std::vector<scalar_value_t> lB( 5 );
        lB[0] = lBValues[0];
        lB[1] = lBValues[1];

        auto lNodeA = VectorValue( scope, lA );
        auto lNodeB = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto lNodeX = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

        auto result0 = AffineTransform( scope, lNodeA, lNodeX, lNodeB );
        scope.Run( result0 );

        std::vector<float> expectedValues = {};
        for( uint32_t i = 0; i < lAValues.size(); i++ )
        {
            uint32_t           size = lNodeX.Get<multi_tensor_value_t>().Shape().GetBufferSizeAs<float>( i ).Size;
            std::vector<float> lX   = lNodeX.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( i );
            std::vector<float> lB   = lNodeB.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( i );
            std::vector<float> lValues( size );
            for( uint32_t j = 0; j < size; j++ )
            {
                lValues[j] = lAValues[i] * lX[j] + lB[j];
            }
            expectedValues.insert( expectedValues.end(), lValues.begin(), lValues.end() );
        }

        std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Affine transform vector/tensor/vector (float)" )
    {
        random_uniform_initializer_t initializer{};
        initializer.mType = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 8, 22 };
        std::vector<uint32_t> dim2{ 8, 64 };

        std::vector<float> lAValues{ 1.109238740928374f, 2.12398471982364f };
        std::vector<float> lBValues{ 2.142983764918237649f, 3.234987659834765f };

        std::vector<scalar_value_t> lA( 5 );
        lA[0] = lAValues[0];
        lA[1] = lAValues[1];

        std::vector<scalar_value_t> lB( 5 );
        lB[0] = lBValues[0];
        lB[1] = lBValues[1];

        auto lNodeA = VectorValue( scope, lA );
        auto lNodeB = VectorValue( scope, lB );
        auto lNodeX = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

        auto result0 = AffineTransform( scope, lNodeA, lNodeX, lNodeB );
        scope.Run( result0 );

        std::vector<float> expectedValues = {};
        for( uint32_t i = 0; i < lAValues.size(); i++ )
        {
            uint32_t           size = lNodeX.Get<multi_tensor_value_t>().Shape().GetBufferSizeAs<float>( i ).Size;
            std::vector<float> lX   = lNodeX.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( i );
            std::vector<float> lValues( size );
            for( uint32_t j = 0; j < size; j++ )
            {
                lValues[j] = lAValues[i] * lX[j] + lBValues[i];
            }
            expectedValues.insert( expectedValues.end(), lValues.begin(), lValues.end() );
        }

        std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Affine transform vector/tensor/scalar (float)" )
    {
        random_uniform_initializer_t initializer{};
        initializer.mType = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 8, 22 };
        std::vector<uint32_t> dim2{ 8, 64 };

        std::vector<float> lAValues{ 1.109238740928374f, 2.12398471982364f };
        std::vector<float> lBValues{ 2.142983764918237649f, 3.234987659834765f };

        std::vector<scalar_value_t> lA( 5 );
        lA[0] = lAValues[0];
        lA[1] = lAValues[1];

        std::vector<scalar_value_t> lB( 5 );
        lB[0] = lBValues[0];
        lB[1] = lBValues[1];

        float lScalarB = 2.142983764918237649f;
        auto  lNodeA   = VectorValue( scope, lA );
        auto  lNodeB   = ConstantScalarValue( scope, lScalarB );
        auto  lNodeX   = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

        auto result0 = AffineTransform( scope, lNodeA, lNodeX, lNodeB );
        scope.Run( result0 );

        std::vector<float> expectedValues = {};
        for( uint32_t i = 0; i < lAValues.size(); i++ )
        {
            uint32_t           size = lNodeX.Get<multi_tensor_value_t>().Shape().GetBufferSizeAs<float>( i ).Size;
            std::vector<float> lX   = lNodeX.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( i );
            std::vector<float> lValues( size );
            for( uint32_t j = 0; j < size; j++ )
            {
                lValues[j] = lAValues[i] * lX[j] + lScalarB;
            }
            expectedValues.insert( expectedValues.end(), lValues.begin(), lValues.end() );
        }

        std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Affine transform scalar/tensor/tensor (float)" )
    {
        random_uniform_initializer_t initializer{};
        initializer.mType = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 8, 22 };
        std::vector<uint32_t> dim2{ 8, 64 };

        std::vector<float> lAValues{ 1.109238740928374f, 2.12398471982364f };
        std::vector<float> lBValues{ 2.142983764918237649f, 3.234987659834765f };

        std::vector<scalar_value_t> lA( 5 );
        lA[0] = lAValues[0];
        lA[1] = lAValues[1];

        std::vector<scalar_value_t> lB( 5 );
        lB[0] = lBValues[0];
        lB[1] = lBValues[1];

        float lScalarA = 2.142983764918237649f;
        auto  lNodeA   = ConstantScalarValue( scope, lScalarA );
        auto  lNodeB   = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto  lNodeX   = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

        auto result0 = AffineTransform( scope, lNodeA, lNodeX, lNodeB );
        scope.Run( result0 );

        std::vector<float> expectedValues = {};
        for( uint32_t i = 0; i < lAValues.size(); i++ )
        {
            uint32_t           size = lNodeX.Get<multi_tensor_value_t>().Shape().GetBufferSizeAs<float>( i ).Size;
            std::vector<float> lX   = lNodeX.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( i );
            std::vector<float> lB   = lNodeB.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( i );
            std::vector<float> lValues( size );
            for( uint32_t j = 0; j < size; j++ )
            {
                lValues[j] = lScalarA * lX[j] + lB[j];
            }
            expectedValues.insert( expectedValues.end(), lValues.begin(), lValues.end() );
        }

        std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Affine transform scalar/tensor/vector (float)" )
    {
        random_uniform_initializer_t initializer{};
        initializer.mType = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 8, 2 };
        std::vector<uint32_t> dim2{ 8, 6 };

        std::vector<float> lAValues{ 1.109238740928374f, 2.12398471982364f };
        std::vector<float> lBValues{ 2.142983764918237649f, 3.234987659834765f };

        std::vector<scalar_value_t> lA( 5 );
        lA[0] = lAValues[0];
        lA[1] = lAValues[1];

        std::vector<scalar_value_t> lB( 5 );
        lB[0] = lBValues[0];
        lB[1] = lBValues[1];

        float lScalarA = 2.142983764918237649f;
        auto  lNodeA   = ConstantScalarValue( scope, lScalarA );
        auto  lNodeB   = VectorValue( scope, lB );
        auto  lNodeX   = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

        auto result0 = AffineTransform( scope, lNodeA, lNodeX, lNodeB );
        scope.Run( result0 );

        std::vector<float> expectedValues = {};
        for( uint32_t i = 0; i < lAValues.size(); i++ )
        {
            uint32_t           size = lNodeX.Get<multi_tensor_value_t>().Shape().GetBufferSizeAs<float>( i ).Size;
            std::vector<float> lX   = lNodeX.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( i );
            std::vector<float> lValues( size );
            for( uint32_t j = 0; j < size; j++ )
            {
                lValues[j] = lScalarA * lX[j] + lBValues[i];
            }
            expectedValues.insert( expectedValues.end(), lValues.begin(), lValues.end() );
        }

        std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Affine transform scalar/tensor/scalar (float)" )
    {
        random_uniform_initializer_t initializer{};
        initializer.mType = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 8, 22 };
        std::vector<uint32_t> dim2{ 8, 64 };

        std::vector<float> lAValues{ 1.109238740928374f, 2.12398471982364f };
        std::vector<float> lBValues{ 2.142983764918237649f, 3.234987659834765f };

        std::vector<scalar_value_t> lA( 5 );
        lA[0] = lAValues[0];
        lA[1] = lAValues[1];

        std::vector<scalar_value_t> lB( 5 );
        lB[0] = lBValues[0];
        lB[1] = lBValues[1];

        float lScalarA = 2.142983764918237649f;
        auto  lNodeA   = ConstantScalarValue( scope, lScalarA );
        float lScalarB = 2.142983764918237649f;
        auto  lNodeB   = ConstantScalarValue( scope, lScalarB );
        auto  lNodeX   = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

        auto result0 = AffineTransform( scope, lNodeA, lNodeX, lNodeB );
        scope.Run( result0 );

        std::vector<float> expectedValues = {};
        for( uint32_t i = 0; i < lAValues.size(); i++ )
        {
            uint32_t           size = lNodeX.Get<multi_tensor_value_t>().Shape().GetBufferSizeAs<float>( i ).Size;
            std::vector<float> lX   = lNodeX.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( i );
            std::vector<float> lValues( size );
            for( uint32_t j = 0; j < size; j++ )
            {
                lValues[j] = lScalarA * lX[j] + lScalarB;
            }
            expectedValues.insert( expectedValues.end(), lValues.begin(), lValues.end() );
        }

        std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }
}

TEST_CASE( "Mix node", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    SECTION( "Mix tensors (float)" )
    {
        random_normal_initializer_t initializer{};
        initializer.mType = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto lNodeA  = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto lNodeB  = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto lNode_T = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

        auto result0 = Mix( scope, lNodeA, lNodeB, lNode_T );
        scope.Run( result0 );

        std::vector<float> lValues_A = lNodeA.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        std::vector<float> lValues_B = lNodeB.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        std::vector<float> lValues_T = lNode_T.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        std::vector<float> expectedValues( lNodeA.Get<multi_tensor_value_t>().mValue.SizeAs<float>() );
        for( uint32_t i = 0; i < lValues_A.size(); i++ )
        {
            expectedValues[i] = ( 1.0f - lValues_T[i] ) * lValues_A[i] + lValues_T[i] * lValues_B[i];
        }
        std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();

        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }
}

TEST_CASE( "Linear space node", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    SECTION( "Linear space allocation (float)" )
    {
        random_normal_initializer_t initializer{};
        initializer.mType = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto lNodeA = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto lNodeB = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

        std::vector<uint32_t> lSubdivisions{ 32, 64 };
        auto                  lNode_S = VectorValue( scope, lSubdivisions );

        auto &result0 = LinearSpace( scope, lNodeA, lNodeB, lNode_S );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Rank == 3 );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[0] == std::vector<uint32_t>{ 2, 2, 32 } );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[1] == std::vector<uint32_t>{ 3, 4, 64 } );
    }

    SECTION( "Linear space (float)" )
    {

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        constant_value_initializer_t lInitializer0{};
        lInitializer0.mValue = 0.5f;

        auto lNodeA = MultiTensorValue( scope, lInitializer0, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

        constant_value_initializer_t lInitializer1{};
        lInitializer1.mValue = 1.5f;
        auto lNodeB          = MultiTensorValue( scope, lInitializer1, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

        std::vector<uint32_t> lSubdivisions{ 32, 64 };
        auto                  lNode_S = VectorValue( scope, lSubdivisions );

        auto &result0 = LinearSpace( scope, lNodeA, lNodeB, lNode_S );
        scope.Run( result0 );

        {
            constexpr uint32_t lSubdivisions = 32;
            std::vector<float> lValues_A     = lNodeA.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
            std::vector<float> lValues_B     = lNodeB.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
            std::vector<float> expectedValues1( Prod( dim1 ) * lSubdivisions );
            uint32_t           x = 0;
            uint32_t           y = 0;
            for( uint32_t i = 0; i < dim1[0]; i++ )
            {
                for( uint32_t j = 0; j < dim1[1]; j++ )
                {
                    float Delta = ( lValues_B[x] - lValues_A[x] ) / static_cast<float>( lSubdivisions );
                    for( uint32_t k = 0; k < lSubdivisions; k++ )
                    {
                        expectedValues1[y] = lValues_A[x] + static_cast<float>( k ) * Delta;
                        y++;
                    }
                    x++;
                }
            }
            std::vector<float> lB1 = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
            REQUIRE( VectorEqual( lB1, expectedValues1 ) );
        }

        {
            constexpr uint32_t lSubdivisions = 64;
            std::vector<float> lValues_A     = lNodeA.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
            std::vector<float> lValues_B     = lNodeB.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
            std::vector<float> expectedValues1( Prod( dim2 ) * lSubdivisions );
            uint32_t           x = 0;
            uint32_t           y = 0;
            for( uint32_t i = 0; i < dim2[0]; i++ )
            {
                for( uint32_t j = 0; j < dim2[1]; j++ )
                {
                    float Delta = ( lValues_B[x] - lValues_A[x] ) / static_cast<float>( lSubdivisions );
                    for( uint32_t k = 0; k < lSubdivisions; k++ )
                    {
                        expectedValues1[y] = lValues_A[x] + static_cast<float>( k ) * Delta;
                        y++;
                    }
                    x++;
                }
            }
            std::vector<float> lB1 = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
            REQUIRE( VectorEqual( lB1, expectedValues1 ) );
        }
    }
}

TEST_CASE( "ARange node", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    SECTION( "ARange allocation (float)" )
    {
        std::vector<float> lAValues{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        std::vector<float> lBValues{ 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
        std::vector<float> lDValues{ 0.01f, .02f, .03f, .04f, .05f };

        auto lNodeA  = ScalarVectorValue( scope, scalar_type_t::FLOAT32, lAValues );
        auto lNodeB  = ScalarVectorValue( scope, scalar_type_t::FLOAT32, lBValues );
        auto lNode_D = ScalarVectorValue( scope, scalar_type_t::FLOAT32, lDValues );

        auto result0 = ARange( scope, lNodeA, lNodeB, lNode_D );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().CountLayers() == 5 );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Rank == 1 );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[0] ==
                 std::vector<uint32_t>{ static_cast<uint32_t>( std::ceil( ( lBValues[0] - lAValues[0] ) / lDValues[0] ) ) } );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[1] ==
                 std::vector<uint32_t>{ static_cast<uint32_t>( std::ceil( ( lBValues[1] - lAValues[1] ) / lDValues[1] ) ) } );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[2] ==
                 std::vector<uint32_t>{ static_cast<uint32_t>( std::ceil( ( lBValues[2] - lAValues[2] ) / lDValues[2] ) ) } );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[3] ==
                 std::vector<uint32_t>{ static_cast<uint32_t>( std::ceil( ( lBValues[3] - lAValues[3] ) / lDValues[3] ) ) } );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[4] ==
                 std::vector<uint32_t>{ static_cast<uint32_t>( std::ceil( ( lBValues[4] - lAValues[4] ) / lDValues[4] ) ) } );
    }

    SECTION( "ARange (float)" )
    {
        std::vector<float> lAValues{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        std::vector<float> lBValues{ 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
        std::vector<float> lDValues{ 0.01f, .02f, .03f, .04f, .05f };

        auto lNodeA  = ScalarVectorValue( scope, scalar_type_t::FLOAT32, lAValues );
        auto lNodeB  = ScalarVectorValue( scope, scalar_type_t::FLOAT32, lBValues );
        auto lNode_D = ScalarVectorValue( scope, scalar_type_t::FLOAT32, lDValues );

        auto result0 = ARange( scope, lNodeA, lNodeB, lNode_D );

        scope.Run( result0 );

        std::vector<float> expectedValues = {};

        for( uint32_t i = 0; i < lAValues.size(); i++ )
        {
            uint32_t           lSubdivisions = static_cast<uint32_t>( std::ceil( ( lBValues[i] - lAValues[i] ) / lDValues[i] ) );
            std::vector<float> lValues( lSubdivisions );
            for( uint32_t j = 0; j < lSubdivisions; j++ )
            {
                lValues[j] = lAValues[i] + j * lDValues[i];
            }
            expectedValues.insert( expectedValues.end(), lValues.begin(), lValues.end() );
        }

        std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "ARange (float)" )
    {
        std::vector<float> lAValues{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        std::vector<float> lBValues{ 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
        std::vector<float> lDValues{ 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };

        auto lNodeA  = ScalarVectorValue( scope, scalar_type_t::FLOAT32, lAValues );
        auto lNodeB  = ScalarVectorValue( scope, scalar_type_t::FLOAT32, lBValues );
        auto lNode_D = ScalarVectorValue( scope, scalar_type_t::FLOAT32, lDValues );

        auto result0 = ARange( scope, lNodeA, lNodeB, lNode_D );

        scope.Run( result0 );

        std::vector<float> expectedValues = {};

        for( uint32_t i = 0; i < lAValues.size(); i++ )
        {
            uint32_t           lSubdivisions = static_cast<uint32_t>( std::ceil( ( lBValues[i] - lAValues[i] ) / lDValues[i] ) );
            std::vector<float> lValues( lSubdivisions );
            for( uint32_t j = 0; j < lSubdivisions; j++ )
            {
                lValues[j] = lAValues[i] + j * lDValues[i];
            }
            expectedValues.insert( expectedValues.end(), lValues.begin(), lValues.end() );
        }

        std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }
}

TEST_CASE( "Repeat node", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    SECTION( "Repeat node allocation (float)" )
    {
        random_normal_initializer_t initializer{};
        initializer.mType = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto lNodeA = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

        std::vector<uint32_t> lSubdivisions{ 3, 5 };
        auto                  lNode_S = VectorValue( scope, lSubdivisions );

        auto result0 = Repeat( scope, lNodeA, lNode_S );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Rank == 3 );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[0] == std::vector<uint32_t>{ 2, 2, 3 } );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[1] == std::vector<uint32_t>{ 3, 4, 5 } );
    }

    SECTION( "Repeat (float)" )
    {
        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        random_normal_initializer_t lInitializer0{};
        lInitializer0.mType = scalar_type_t::FLOAT32;

        auto lNodeA = MultiTensorValue( scope, lInitializer0, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

        std::vector<uint32_t> lRepetitions{ 3, 5 };
        auto                  lNode_R = VectorValue( scope, lRepetitions );

        auto result0 = Repeat( scope, lNodeA, lNode_R );
        scope.Run( result0 );

        {
            constexpr uint32_t lSubdivisions = 3;
            std::vector<float> lValues_A     = lNodeA.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
            std::vector<float> expectedValues1( Prod( dim1 ) * lSubdivisions );
            uint32_t           x = 0;
            uint32_t           y = 0;
            for( uint32_t i = 0; i < dim1[0]; i++ )
            {
                for( uint32_t j = 0; j < dim1[1]; j++ )
                {
                    for( uint32_t k = 0; k < lSubdivisions; k++ )
                    {
                        expectedValues1[y] = lValues_A[x];
                        y++;
                    }
                    x++;
                }
            }
            std::vector<float> lB1 = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
            REQUIRE( VectorEqual( lB1, expectedValues1 ) );
        }

        {
            constexpr uint32_t lSubdivisions = 5;
            std::vector<float> lValues_A     = lNodeA.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
            std::vector<float> expectedValues1( Prod( dim2 ) * lSubdivisions );
            uint32_t           x = 0;
            uint32_t           y = 0;
            for( uint32_t i = 0; i < dim2[0]; i++ )
            {
                for( uint32_t j = 0; j < dim2[1]; j++ )
                {
                    for( uint32_t k = 0; k < lSubdivisions; k++ )
                    {
                        expectedValues1[y] = lValues_A[x];
                        y++;
                    }
                    x++;
                }
            }
            std::vector<float> lB1 = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
            REQUIRE( VectorEqual( lB1, expectedValues1 ) );
        }
    }
}

TEST_CASE( "Tile node", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    SECTION( "Tile node allocation (float)" )
    {
        random_normal_initializer_t initializer{};
        initializer.mType = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto lNodeA = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

        std::vector<uint32_t> lSubdivisions{ 3, 5 };
        auto                  lNode_S = VectorValue( scope, lSubdivisions );

        auto result0 = Tile( scope, lNodeA, lNode_S );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Rank == 3 );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[0] == std::vector<uint32_t>{ 3, 2, 2 } );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[1] == std::vector<uint32_t>{ 5, 3, 4 } );
    }

    SECTION( "Tile (float)" )
    {
        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        random_normal_initializer_t lInitializer0{};
        lInitializer0.mType = scalar_type_t::FLOAT32;

        auto lNodeA = MultiTensorValue( scope, lInitializer0, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

        std::vector<uint32_t> lRepetitions{ 7, 11 };
        auto                  lNode_R = VectorValue( scope, lRepetitions );

        auto result0 = Tile( scope, lNodeA, lNode_R );
        scope.Run( result0 );

        {
            constexpr uint32_t lSubdivisions = 7;
            std::vector<float> lValues_A     = lNodeA.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
            std::vector<float> expectedValues1( Prod( dim1 ) * lSubdivisions );
            uint32_t           x = 0;
            uint32_t           y = 0;
            for( uint32_t k = 0; k < lSubdivisions; k++ )
            {
                for( uint32_t i = 0; i < dim1[0] * dim1[1]; i++ )
                {
                    expectedValues1[y] = lValues_A[i];
                    y++;
                }
            }
            std::vector<float> lB1 = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
            REQUIRE( VectorEqual( lB1, expectedValues1 ) );
        }

        {
            constexpr uint32_t lSubdivisions = 11;
            std::vector<float> lValues_A     = lNodeA.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
            std::vector<float> expectedValues1( Prod( dim2 ) * lSubdivisions );
            uint32_t           x = 0;
            uint32_t           y = 0;
            for( uint32_t k = 0; k < lSubdivisions; k++ )
            {
                for( uint32_t i = 0; i < dim2[0] * dim2[1]; i++ )
                {
                    expectedValues1[y] = lValues_A[i];
                    y++;
                }
            }
            std::vector<float> lB1 = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
            REQUIRE( VectorEqual( lB1, expectedValues1 ) );
        }
    }
}

TEST_CASE( "Expand MultiTensors", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_uniform_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 5, 23, 42 };

    SECTION( "Expanding multi-tensors preserved types" )
    {
        auto lNodeA =
            MultiTensorValue( scope, initializer, tensor_shape_t( std::vector<std::vector<uint32_t>>{ dim1 }, sizeof( float ) ) );
        auto result0 = Expand( scope, lNodeA );

        REQUIRE( result0.Get<type_t>().mValue == lNodeA.Get<type_t>().mValue );
    }

    SECTION( "Expanding multi-tensors gives the correct dimension" )
    {
        auto lNodeA =
            MultiTensorValue( scope, initializer, tensor_shape_t( std::vector<std::vector<uint32_t>>{ dim1 }, sizeof( float ) ) );
        auto result0 = Expand( scope, lNodeA );

        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Rank == ( dim1.size() - 1 ) );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().CountLayers() == dim1[0] );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[0] == std::vector<uint32_t>{ 23, 42 } );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[1] == std::vector<uint32_t>{ 23, 42 } );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[2] == std::vector<uint32_t>{ 23, 42 } );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[3] == std::vector<uint32_t>{ 23, 42 } );
    }

    SECTION( "Expanding multi-tensors does not change values" )
    {
        auto lNodeA =
            MultiTensorValue( scope, initializer, tensor_shape_t( std::vector<std::vector<uint32_t>>{ dim1 }, sizeof( float ) ) );
        auto result0 = Expand( scope, lNodeA );

        scope.Run( result0 );

        std::vector<float> tensorValues0 = lNodeA.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        std::vector<float> tensorValues1 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        REQUIRE( VectorEqual( tensorValues0, tensorValues0 ) );
    }
}

TEST_CASE( "Collapse MultiTensors", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_uniform_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 23, 42 };
    std::vector<uint32_t> dim2{ 23, 42 };
    std::vector<uint32_t> lDim3{ 23, 42 };
    std::vector<uint32_t> lDim4{ 23, 42 };

    SECTION( "Collapsing multi-tensors preserved types" )
    {
        auto lNodeA   = MultiTensorValue( scope, initializer,
                                          tensor_shape_t( std::vector<std::vector<uint32_t>>{ dim1, dim2 }, sizeof( float ) ) );
        auto result0 = Collapse( scope, lNodeA );

        REQUIRE( result0.Get<type_t>().mValue == lNodeA.Get<type_t>().mValue );
    }

    SECTION( "Collapsing multi-tensors gives the correct dimension" )
    {
        auto lNodeA = MultiTensorValue(
            scope, initializer, tensor_shape_t( std::vector<std::vector<uint32_t>>{ dim1, dim2, lDim3, lDim4 }, sizeof( float ) ) );
        auto result0 = Collapse( scope, lNodeA );

        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Rank == 3 );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().CountLayers() == 1 );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[0] == std::vector<uint32_t>{ 4, 23, 42 } );
    }

    SECTION( "Collapsing multi-tensors does not change values" )
    {
        auto lNodeA = MultiTensorValue(
            scope, initializer, tensor_shape_t( std::vector<std::vector<uint32_t>>{ dim1, dim2, lDim3, lDim4 }, sizeof( float ) ) );
        auto result0 = Collapse( scope, lNodeA );

        scope.Run( result0 );

        std::vector<float> tensorValues0 = lNodeA.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        std::vector<float> tensorValues1 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        REQUIRE( VectorEqual( tensorValues0, tensorValues0 ) );
    }
}

TEST_CASE( "Reshape MultiTensors", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_uniform_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 21, 42 };
    std::vector<uint32_t> dim2{ 25, 40 };
    std::vector<uint32_t> lDim3{ 14, 4 };

    SECTION( "Reshaping multi-tensors preserved types" )
    {
        auto lNodeA = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( float ) ) );

        std::vector<uint32_t> lODim1{ 7, 3, 42 };
        std::vector<uint32_t> lODim2{ 25, 8, 5 };
        std::vector<uint32_t> lODim3{ 7, 2, 4 };
        auto                  result0 = Reshape( scope, lNodeA, tensor_shape_t( { lODim1, lODim2, lODim3 }, sizeof( float ) ) );
        scope.Run( result0 );

        REQUIRE( result0.Get<type_t>().mValue == lNodeA.Get<type_t>().mValue );
    }

    SECTION( "Reshaping multi-tensors gives the correct dimension" )
    {
        auto lNodeA = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( float ) ) );

        std::vector<uint32_t> lODim1{ 7, 3, 42 };
        std::vector<uint32_t> lODim2{ 25, 8, 5 };
        std::vector<uint32_t> lODim3{ 7, 2, 4 };

        auto result0 = Reshape( scope, lNodeA, tensor_shape_t( { lODim1, lODim2, lODim3 }, sizeof( float ) ) );
        scope.Run( result0 );

        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Rank == 3 );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().CountLayers() == 3 );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[0] == lODim1 );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[1] == lODim2 );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[2] == lODim3 );
    }

    SECTION( "Reshaping multi-tensors does not change values" )
    {
        auto lNodeA = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( float ) ) );

        std::vector<uint32_t> lODim1{ 7, 3, 42 };
        std::vector<uint32_t> lODim2{ 25, 8, 5 };
        std::vector<uint32_t> lODim3{ 7, 2, 4 };

        auto result0 = Reshape( scope, lNodeA, tensor_shape_t( { lODim1, lODim2, lODim3 }, sizeof( float ) ) );
        scope.Run( result0 );

        std::vector<float> tensorValues0 = lNodeA.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        std::vector<float> tensorValues1 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        REQUIRE( VectorEqual( tensorValues0, tensorValues0 ) );
    }
}

TEST_CASE( "Flatten MultiTensors", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_uniform_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 21, 42 };
    std::vector<uint32_t> dim2{ 25, 40 };
    std::vector<uint32_t> lDim3{ 14, 4 };

    SECTION( "Reshaping multi-tensors preserved types" )
    {
        auto lNodeA   = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto result0 = Flatten( scope, lNodeA );

        REQUIRE( result0.Get<type_t>().mValue == lNodeA.Get<type_t>().mValue );
    }

    SECTION( "Flattening multi-tensors gives the correct dimension" )
    {
        auto lNodeA = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( float ) ) );

        std::vector<uint32_t> lODim1{ 7, 3, 42 };
        std::vector<uint32_t> lODim2{ 25, 8, 5 };
        std::vector<uint32_t> lODim3{ 7, 2, 4 };

        auto result0 = Flatten( scope, lNodeA );
        scope.Run( result0 );

        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Rank == 1 );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().CountLayers() == 3 );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[0][0] == Prod( lODim1 ) );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[1][0] == Prod( lODim2 ) );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[2][0] == Prod( lODim3 ) );
    }

    SECTION( "Flattening multi-tensors does not change values" )
    {
        auto lNodeA = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( float ) ) );

        std::vector<uint32_t> lODim1{ 7, 3, 42 };
        std::vector<uint32_t> lODim2{ 25, 8, 5 };
        std::vector<uint32_t> lODim3{ 7, 2, 4 };

        auto result0 = Flatten( scope, lNodeA );
        scope.Run( result0 );

        std::vector<float> tensorValues0 = lNodeA.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        std::vector<float> tensorValues1 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
        REQUIRE( VectorEqual( tensorValues0, tensorValues0 ) );
    }
}

TEST_CASE( "Scope operation names", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> value = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    auto                  node  = VectorValue<uint32_t>( scope.WithOpName( "Node_1" ), value );
    REQUIRE( node == scope["Node_1"] );
}

TEST_CASE( "InInterval Tensor_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto opNode     = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto lLowerBound = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto lUpperBound = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto result0    = InInterval( scope, opNode, lLowerBound, lUpperBound, false, false );

    scope.Run( result0 );

    std::vector<float> lXValues          = opNode.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    std::vector<float> lLowerBoundValues = lLowerBound.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    std::vector<float> lUpperBoundValues = lUpperBound.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();

    std::vector<uint8_t> lResultValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues( opNode.Get<multi_tensor_value_t>().mValue.SizeAs<float>() );

    for( uint32_t i = 0; i < expectedValues.size(); i++ )
    {
        expectedValues[i] = ( lLowerBoundValues[i] <= lXValues[i] ) && ( lXValues[i] <= lUpperBoundValues[i] );
    }
    REQUIRE( lResultValues == expectedValues );
}

TEST_CASE( "InInterval Tensor_Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto opNode     = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto lLowerBound = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    std::vector<scalar_value_t> constants{ 0.2345f, 0.345f };
    auto                        lUpperBound = VectorValue( scope, constants );
    auto                        result0    = InInterval( scope, opNode, lLowerBound, lUpperBound, false, false );

    scope.Run( result0 );

    std::vector<float>   lXValues0          = opNode.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
    std::vector<float>   lLowerBoundValues0 = lLowerBound.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
    std::vector<uint8_t> lResultValues0     = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<uint8_t>( 0 );
    std::vector<uint8_t> expectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( lLowerBoundValues0[i] <= lXValues0[i] ) && ( lXValues0[i] <= std::get<float>( constants[0] ) );
    REQUIRE( lResultValues0 == expectedValues0 );

    std::vector<float>   lXValues1          = opNode.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
    std::vector<float>   lLowerBoundValues1 = lLowerBound.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
    std::vector<uint8_t> lResultValues1     = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<uint8_t>( 1 );
    std::vector<uint8_t> expectedValues1( lResultValues1.size() );
    for( uint32_t i = 0; i < expectedValues1.size(); i++ )
        expectedValues1[i] = ( lLowerBoundValues1[i] <= lXValues1[i] ) && ( lXValues1[i] <= std::get<float>( constants[1] ) );
    REQUIRE( lResultValues1 == expectedValues1 );
}

TEST_CASE( "InInterval Tensor_Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto opNode     = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto lLowerBound = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto lUpperBound = ConstantScalarValue( scope, 0.245f );
    auto result0    = InInterval( scope, opNode, lLowerBound, lUpperBound, false, false );

    scope.Run( result0 );

    std::vector<float>   lXValues0          = opNode.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    std::vector<float>   lLowerBoundValues0 = lLowerBound.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    std::vector<uint8_t> lResultValues0     = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( lLowerBoundValues0[i] <= lXValues0[i] ) && ( lXValues0[i] <= 0.245f );
    REQUIRE( lResultValues0 == expectedValues0 );
}

TEST_CASE( "InInterval Vector_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto opNode     = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto lUpperBound = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    std::vector<scalar_value_t> constants{ 0.2345f, 0.345f };
    auto                        lLowerBound = VectorValue( scope, constants );
    auto                        result0    = InInterval( scope, opNode, lLowerBound, lUpperBound, false, false );

    scope.Run( result0 );

    std::vector<float>   lXValues0          = opNode.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
    std::vector<float>   lUpperBoundValues0 = lUpperBound.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
    std::vector<uint8_t> lResultValues0     = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<uint8_t>( 0 );
    std::vector<uint8_t> expectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( std::get<float>( constants[0] ) <= lXValues0[i] ) && ( lXValues0[i] <= lUpperBoundValues0[i] );
    REQUIRE( lResultValues0 == expectedValues0 );

    std::vector<float>   lXValues1          = opNode.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
    std::vector<float>   lUpperBoundValues1 = lUpperBound.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
    std::vector<uint8_t> lResultValues1     = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<uint8_t>( 1 );
    std::vector<uint8_t> expectedValues1( lResultValues1.size() );
    for( uint32_t i = 0; i < expectedValues1.size(); i++ )
        expectedValues1[i] = ( std::get<float>( constants[1] ) <= lXValues1[i] ) && ( lXValues1[i] <= lUpperBoundValues1[i] );
    REQUIRE( lResultValues1 == expectedValues1 );
}

TEST_CASE( "InInterval Scalar_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto opNode     = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto lUpperBound = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto lLowerBound = ConstantScalarValue( scope, 0.245f );
    auto result0    = InInterval( scope, opNode, lLowerBound, lUpperBound, false, false );

    scope.Run( result0 );

    std::vector<float>   lXValues0          = opNode.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    std::vector<float>   lUpperBoundValues0 = lUpperBound.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    std::vector<uint8_t> lResultValues0     = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( 0.245f <= lXValues0[i] ) && ( lXValues0[i] <= lUpperBoundValues0[i] );
    REQUIRE( lResultValues0 == expectedValues0 );
}

TEST_CASE( "InInterval Scalar_Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto opNode     = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto lLowerBound = ConstantScalarValue( scope, 0.245f );
    auto lUpperBound = ConstantScalarValue( scope, 0.75f );

    auto result0 = InInterval( scope, opNode, lLowerBound, lUpperBound, false, false );

    scope.Run( result0 );

    std::vector<float>   lXValues0      = opNode.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    std::vector<uint8_t> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( 0.245f <= lXValues0[i] ) && ( lXValues0[i] <= 0.75f );
    REQUIRE( lResultValues0 == expectedValues0 );
}

TEST_CASE( "LessThan Tensor_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto lX = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto y  = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto result0 = LessThan( scope, lX, y );
    scope.Run( result0 );

    std::vector<float> lXValues = lX.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    std::vector<float> lYValues = y.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();

    std::vector<uint8_t> lResultValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues( lResultValues.size() );

    for( uint32_t i = 0; i < expectedValues.size(); i++ )
    {
        expectedValues[i] = ( lXValues[i] < lYValues[i] );
    }
    REQUIRE( lResultValues == expectedValues );
}

TEST_CASE( "LessThan Tensor_Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto lX = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    std::vector<scalar_value_t> constants{ 0.2345f, 0.345f };
    auto                        y        = VectorValue( scope, constants );
    auto                        result0 = LessThan( scope, lX, y );

    scope.Run( result0 );

    std::vector<float>   lXValues0      = lX.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
    std::vector<uint8_t> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<uint8_t>( 0 );
    std::vector<uint8_t> expectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( lXValues0[i] < std::get<float>( constants[0] ) );
    REQUIRE( lResultValues0 == expectedValues0 );

    std::vector<float>   lXValues1      = lX.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
    std::vector<uint8_t> lResultValues1 = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<uint8_t>( 1 );
    std::vector<uint8_t> expectedValues1( lResultValues1.size() );
    for( uint32_t i = 0; i < expectedValues1.size(); i++ )
        expectedValues1[i] = ( lXValues1[i] < std::get<float>( constants[1] ) );
    REQUIRE( lResultValues1 == expectedValues1 );
}

TEST_CASE( "LessThan Tensor_Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto lX = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto y  = ConstantScalarValue( scope, 0.245f );

    auto result0 = LessThan( scope, lX, y );

    scope.Run( result0 );

    std::vector<float>   lXValues0      = lX.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    std::vector<uint8_t> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( lXValues0[i] < 0.245f );
    REQUIRE( lResultValues0 == expectedValues0 );
}

TEST_CASE( "LessThan Vector_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    std::vector<scalar_value_t> constants{ 0.2345f, 0.345f };
    auto                        lX = VectorValue( scope, constants );
    auto                        y  = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto result0 = LessThan( scope, lX, y );

    scope.Run( result0 );

    std::vector<float>   lYValues0      = y.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
    std::vector<uint8_t> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<uint8_t>( 0 );
    std::vector<uint8_t> expectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( std::get<float>( constants[0] ) < lYValues0[i] );
    REQUIRE( lResultValues0 == expectedValues0 );

    std::vector<float>   lYValues1      = y.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
    std::vector<uint8_t> lResultValues1 = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<uint8_t>( 1 );
    std::vector<uint8_t> expectedValues1( lResultValues1.size() );
    for( uint32_t i = 0; i < expectedValues1.size(); i++ )
        expectedValues1[i] = ( std::get<float>( constants[1] ) < lYValues1[i] );
    REQUIRE( lResultValues1 == expectedValues1 );
}

TEST_CASE( "LessThan Scalar_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto lX = ConstantScalarValue( scope, 0.245f );
    auto y  = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto result0 = LessThan( scope, lX, y );

    scope.Run( result0 );

    std::vector<float>   lYValues0      = y.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    std::vector<uint8_t> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( 0.245f < lYValues0[i] );
    REQUIRE( lResultValues0 == expectedValues0 );
}

TEST_CASE( "LessThanOrEqual Tensor_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto lX = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto y  = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto result0 = LessThanOrEqual( scope, lX, y );
    scope.Run( result0 );

    std::vector<float> lXValues = lX.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    std::vector<float> lYValues = y.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();

    std::vector<uint8_t> lResultValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues( lResultValues.size() );

    for( uint32_t i = 0; i < expectedValues.size(); i++ )
    {
        expectedValues[i] = ( lXValues[i] <= lYValues[i] );
    }
    REQUIRE( lResultValues == expectedValues );
}

TEST_CASE( "LessThanOrEqual Tensor_Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto lX = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    std::vector<scalar_value_t> constants{ 0.2345f, 0.345f };
    auto                        y        = VectorValue( scope, constants );
    auto                        result0 = LessThanOrEqual( scope, lX, y );

    scope.Run( result0 );

    std::vector<float>   lXValues0      = lX.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
    std::vector<uint8_t> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<uint8_t>( 0 );
    std::vector<uint8_t> expectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( lXValues0[i] <= std::get<float>( constants[0] ) );
    REQUIRE( lResultValues0 == expectedValues0 );

    std::vector<float>   lXValues1      = lX.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
    std::vector<uint8_t> lResultValues1 = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<uint8_t>( 1 );
    std::vector<uint8_t> expectedValues1( lResultValues1.size() );
    for( uint32_t i = 0; i < expectedValues1.size(); i++ )
        expectedValues1[i] = ( lXValues1[i] <= std::get<float>( constants[1] ) );
    REQUIRE( lResultValues1 == expectedValues1 );
}

TEST_CASE( "LessThanOrEqual Tensor_Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto lX = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto y  = ConstantScalarValue( scope, 0.245f );

    auto result0 = LessThanOrEqual( scope, lX, y );

    scope.Run( result0 );

    std::vector<float>   lXValues0      = lX.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    std::vector<uint8_t> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( lXValues0[i] <= 0.245f );
    REQUIRE( lResultValues0 == expectedValues0 );
}

TEST_CASE( "LessThanOrEqual Vector_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    std::vector<scalar_value_t> constants{ 0.2345f, 0.345f };
    auto                        lX = VectorValue( scope, constants );
    auto                        y  = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto result0 = LessThanOrEqual( scope, lX, y );

    scope.Run( result0 );

    std::vector<float>   lYValues0      = y.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
    std::vector<uint8_t> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<uint8_t>( 0 );
    std::vector<uint8_t> expectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( std::get<float>( constants[0] ) <= lYValues0[i] );
    REQUIRE( lResultValues0 == expectedValues0 );

    std::vector<float>   lYValues1      = y.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
    std::vector<uint8_t> lResultValues1 = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<uint8_t>( 1 );
    std::vector<uint8_t> expectedValues1( lResultValues1.size() );
    for( uint32_t i = 0; i < expectedValues1.size(); i++ )
        expectedValues1[i] = ( std::get<float>( constants[1] ) <= lYValues1[i] );
    REQUIRE( lResultValues1 == expectedValues1 );
}

TEST_CASE( "LessThanOrEqual Scalar_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto lX = ConstantScalarValue( scope, 0.245f );
    auto y  = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto result0 = LessThanOrEqual( scope, lX, y );

    scope.Run( result0 );

    std::vector<float>   lYValues0      = y.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    std::vector<uint8_t> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( 0.245f <= lYValues0[i] );
    REQUIRE( lResultValues0 == expectedValues0 );
}

TEST_CASE( "GreaterThan Tensor_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto lX = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto y  = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto result0 = GreaterThan( scope, lX, y );
    scope.Run( result0 );

    std::vector<float> lXValues = lX.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    std::vector<float> lYValues = y.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();

    std::vector<uint8_t> lResultValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues( lResultValues.size() );

    for( uint32_t i = 0; i < expectedValues.size(); i++ )
    {
        expectedValues[i] = ( lXValues[i] > lYValues[i] );
    }
    REQUIRE( lResultValues == expectedValues );
}

TEST_CASE( "GreaterThan Tensor_Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto lX = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    std::vector<scalar_value_t> constants{ 0.2345f, 0.345f };
    auto                        y        = VectorValue( scope, constants );
    auto                        result0 = GreaterThan( scope, lX, y );

    scope.Run( result0 );

    std::vector<float>   lXValues0      = lX.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
    std::vector<uint8_t> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<uint8_t>( 0 );
    std::vector<uint8_t> expectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( lXValues0[i] > std::get<float>( constants[0] ) );
    REQUIRE( lResultValues0 == expectedValues0 );

    std::vector<float>   lXValues1      = lX.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
    std::vector<uint8_t> lResultValues1 = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<uint8_t>( 1 );
    std::vector<uint8_t> expectedValues1( lResultValues1.size() );
    for( uint32_t i = 0; i < expectedValues1.size(); i++ )
        expectedValues1[i] = ( lXValues1[i] > std::get<float>( constants[1] ) );
    REQUIRE( lResultValues1 == expectedValues1 );
}

TEST_CASE( "GreaterThan Tensor_Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto lX = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto y  = ConstantScalarValue( scope, 0.245f );

    auto result0 = GreaterThan( scope, lX, y );

    scope.Run( result0 );

    std::vector<float>   lXValues0      = lX.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    std::vector<uint8_t> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( lXValues0[i] > 0.245f );
    REQUIRE( lResultValues0 == expectedValues0 );
}

TEST_CASE( "GreaterThan Vector_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    std::vector<scalar_value_t> constants{ 0.2345f, 0.345f };
    auto                        lX = VectorValue( scope, constants );
    auto                        y  = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto result0 = GreaterThan( scope, lX, y );

    scope.Run( result0 );

    std::vector<float>   lYValues0      = y.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
    std::vector<uint8_t> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<uint8_t>( 0 );
    std::vector<uint8_t> expectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( std::get<float>( constants[0] ) > lYValues0[i] );
    REQUIRE( lResultValues0 == expectedValues0 );

    std::vector<float>   lYValues1      = y.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
    std::vector<uint8_t> lResultValues1 = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<uint8_t>( 1 );
    std::vector<uint8_t> expectedValues1( lResultValues1.size() );
    for( uint32_t i = 0; i < expectedValues1.size(); i++ )
        expectedValues1[i] = ( std::get<float>( constants[1] ) > lYValues1[i] );
    REQUIRE( lResultValues1 == expectedValues1 );
}

TEST_CASE( "GreaterThan Scalar_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto lX = ConstantScalarValue( scope, 0.245f );
    auto y  = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto result0 = GreaterThan( scope, lX, y );

    scope.Run( result0 );

    std::vector<float>   lYValues0      = y.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    std::vector<uint8_t> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( 0.245f > lYValues0[i] );
    REQUIRE( lResultValues0 == expectedValues0 );
}

TEST_CASE( "GreaterThanOrEqual Tensor_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto lX = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto y  = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto result0 = GreaterThanOrEqual( scope, lX, y );
    scope.Run( result0 );

    std::vector<float> lXValues = lX.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    std::vector<float> lYValues = y.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();

    std::vector<uint8_t> lResultValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues( lResultValues.size() );

    for( uint32_t i = 0; i < expectedValues.size(); i++ )
    {
        expectedValues[i] = ( lXValues[i] >= lYValues[i] );
    }
    REQUIRE( lResultValues == expectedValues );
}

TEST_CASE( "GreaterThanOrEqual Tensor_Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto lX = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    std::vector<scalar_value_t> constants{ 0.2345f, 0.345f };
    auto                        y        = VectorValue( scope, constants );
    auto                        result0 = GreaterThanOrEqual( scope, lX, y );

    scope.Run( result0 );

    std::vector<float>   lXValues0      = lX.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
    std::vector<uint8_t> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<uint8_t>( 0 );
    std::vector<uint8_t> expectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( lXValues0[i] >= std::get<float>( constants[0] ) );
    REQUIRE( lResultValues0 == expectedValues0 );

    std::vector<float>   lXValues1      = lX.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
    std::vector<uint8_t> lResultValues1 = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<uint8_t>( 1 );
    std::vector<uint8_t> expectedValues1( lResultValues1.size() );
    for( uint32_t i = 0; i < expectedValues1.size(); i++ )
        expectedValues1[i] = ( lXValues1[i] >= std::get<float>( constants[1] ) );
    REQUIRE( lResultValues1 == expectedValues1 );
}

TEST_CASE( "GreaterThanOrEqual Tensor_Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto lX = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto y  = ConstantScalarValue( scope, 0.245f );

    auto result0 = GreaterThanOrEqual( scope, lX, y );

    scope.Run( result0 );

    std::vector<float>   lXValues0      = lX.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    std::vector<uint8_t> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( lXValues0[i] >= 0.245f );
    REQUIRE( lResultValues0 == expectedValues0 );
}

TEST_CASE( "GreaterThanOrEqual Vector_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    std::vector<scalar_value_t> constants{ 0.2345f, 0.345f };
    auto                        lX = VectorValue( scope, constants );
    auto                        y  = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto result0 = GreaterThanOrEqual( scope, lX, y );

    scope.Run( result0 );

    std::vector<float>   lYValues0      = y.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
    std::vector<uint8_t> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<uint8_t>( 0 );
    std::vector<uint8_t> expectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( std::get<float>( constants[0] ) >= lYValues0[i] );
    REQUIRE( lResultValues0 == expectedValues0 );

    std::vector<float>   lYValues1      = y.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
    std::vector<uint8_t> lResultValues1 = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<uint8_t>( 1 );
    std::vector<uint8_t> expectedValues1( lResultValues1.size() );
    for( uint32_t i = 0; i < expectedValues1.size(); i++ )
        expectedValues1[i] = ( std::get<float>( constants[1] ) >= lYValues1[i] );
    REQUIRE( lResultValues1 == expectedValues1 );
}

TEST_CASE( "GreaterThanOrEqual Scalar_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto lX = ConstantScalarValue( scope, 0.245f );
    auto y  = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto result0 = GreaterThanOrEqual( scope, lX, y );

    scope.Run( result0 );

    std::vector<float>   lYValues0      = y.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    std::vector<uint8_t> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( 0.245f >= lYValues0[i] );
    REQUIRE( lResultValues0 == expectedValues0 );
}

TEST_CASE( "Where Tensor_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto               lValues0 = RandomBool( 12 * 23 + 13 * 24 );
    data_initializer_t lInitializer0( lValues0 );
    auto               lCondition = MultiTensorValue( scope, lInitializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    auto lX = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto y  = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto result0 = Where( scope, lCondition, lX, y );
    scope.Run( result0 );

    std::vector<float> lXValues = lX.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    std::vector<float> lYValues = y.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();

    std::vector<float> lResultValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    std::vector<float> expectedValues( lResultValues.size() );

    for( uint32_t i = 0; i < expectedValues.size(); i++ )
    {
        expectedValues[i] = lValues0[i] ? lXValues[i] : lYValues[i];
    }
    REQUIRE( lResultValues == expectedValues );
}

TEST_CASE( "Where Tensor_Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto                 lValues00 = RandomBool( 12 * 23 );
    auto                 lValues01 = RandomBool( 13 * 24 );
    std::vector<uint8_t> lValues0;
    lValues0.insert( lValues0.end(), lValues00.begin(), lValues00.end() );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    data_initializer_t lInitializer0( lValues0 );
    auto               lCondition = MultiTensorValue( scope, lInitializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    auto lX = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    std::vector<scalar_value_t> constants{ 0.27986534f, 0.31490728f };
    auto                        y = VectorValue( scope, constants );

    auto result0 = Where( scope, lCondition, lX, y );

    scope.Run( result0 );

    std::vector<float> lXValues0      = lX.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
    std::vector<float> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
    std::vector<float> expectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = lValues00[i] ? lXValues0[i] : std::get<float>( constants[0] );
    REQUIRE( lResultValues0 == expectedValues0 );

    std::vector<float> lXValues1      = lX.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
    std::vector<float> lResultValues1 = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
    std::vector<float> expectedValues1( lResultValues1.size() );
    for( uint32_t i = 0; i < expectedValues1.size(); i++ )
        expectedValues1[i] = lValues01[i] ? lXValues1[i] : std::get<float>( constants[1] );
    REQUIRE( lResultValues1 == expectedValues1 );
}

TEST_CASE( "Where Tensor_Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto               lValues0 = RandomBool( 12 * 23 + 13 * 24 );
    data_initializer_t lInitializer0( lValues0 );
    auto               lCondition = MultiTensorValue( scope, lInitializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    auto lX = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto y  = ConstantScalarValue( scope, 0.245f );

    auto result0 = Where( scope, lCondition, lX, y );

    scope.Run( result0 );

    std::vector<float> lXValues0      = lX.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    std::vector<float> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    std::vector<float> expectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = lValues0[i] ? lXValues0[i] : 0.245f;
    REQUIRE( lResultValues0 == expectedValues0 );
}

TEST_CASE( "Where Vector_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto lValues00 = RandomBool( 12 * 23 );
    auto lValues01 = RandomBool( 13 * 24 );

    std::vector<uint8_t> lValues0;
    lValues0.insert( lValues0.end(), lValues00.begin(), lValues00.end() );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    data_initializer_t lInitializer0( lValues0 );

    auto lCondition = MultiTensorValue( scope, lInitializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    std::vector<scalar_value_t> constants{ 0.2345f, 0.345f };
    auto                        lX = VectorValue( scope, constants );

    auto y = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto result0 = Where( scope, lCondition, lX, y );

    scope.Run( result0 );

    std::vector<float> lXValues0      = y.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
    std::vector<float> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
    std::vector<float> expectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = lValues00[i] ? std::get<float>( constants[0] ) : lXValues0[i];
    REQUIRE( lResultValues0 == expectedValues0 );

    std::vector<float> lXValues1      = y.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
    std::vector<float> lResultValues1 = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
    std::vector<float> expectedValues1( lResultValues1.size() );
    for( uint32_t i = 0; i < expectedValues1.size(); i++ )
        expectedValues1[i] = lValues01[i] ? std::get<float>( constants[1] ) : lXValues1[i];
    REQUIRE( lResultValues1 == expectedValues1 );
}

TEST_CASE( "Where Vector_Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto                 lValues00 = RandomBool( 12 * 23 );
    auto                 lValues01 = RandomBool( 13 * 24 );
    std::vector<uint8_t> lValues0;
    lValues0.insert( lValues0.end(), lValues00.begin(), lValues00.end() );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    data_initializer_t lInitializer0( lValues0 );

    auto lCondition = MultiTensorValue( scope, lInitializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    std::vector<scalar_value_t> lConstants0{ 0.2345f, 0.345f };
    auto                        lX = VectorValue( scope, lConstants0 );

    std::vector<scalar_value_t> lConstants1{ 0.26534f, 0.19048265f };
    auto                        y = VectorValue( scope, lConstants1 );

    auto result0 = Where( scope, lCondition, lX, y );

    scope.Run( result0 );

    std::vector<float> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
    std::vector<float> expectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = lValues00[i] ? std::get<float>( lConstants0[0] ) : std::get<float>( lConstants1[0] );
    REQUIRE( lResultValues0 == expectedValues0 );

    std::vector<float> lResultValues1 = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
    std::vector<float> expectedValues1( lResultValues1.size() );
    for( uint32_t i = 0; i < expectedValues1.size(); i++ )
        expectedValues1[i] = lValues01[i] ? std::get<float>( lConstants0[1] ) : std::get<float>( lConstants1[1] );
    REQUIRE( lResultValues1 == expectedValues1 );
}

TEST_CASE( "Where Vector_Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto                 lValues00 = RandomBool( 12 * 23 );
    auto                 lValues01 = RandomBool( 13 * 24 );
    std::vector<uint8_t> lValues0;
    lValues0.insert( lValues0.end(), lValues00.begin(), lValues00.end() );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    data_initializer_t lInitializer0( lValues0 );
    auto               lCondition = MultiTensorValue( scope, lInitializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    std::vector<scalar_value_t> constants{ 0.2345f, 0.345f };
    auto                        lX = VectorValue( scope, constants );
    auto                        y  = ConstantScalarValue( scope, 0.245f );

    auto result0 = Where( scope, lCondition, lX, y );

    scope.Run( result0 );

    std::vector<float> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
    std::vector<float> expectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = lValues00[i] ? std::get<float>( constants[0] ) : 0.245f;
    REQUIRE( lResultValues0 == expectedValues0 );

    std::vector<float> lResultValues1 = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
    std::vector<float> expectedValues1( lResultValues1.size() );
    for( uint32_t i = 0; i < expectedValues1.size(); i++ )
        expectedValues1[i] = lValues01[i] ? std::get<float>( constants[1] ) : 0.245f;
    REQUIRE( lResultValues1 == expectedValues1 );
}

TEST_CASE( "Where Scalar_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto               lValues0 = RandomBool( 12 * 23 + 13 * 24 );
    data_initializer_t lInitializer0( lValues0 );
    auto               lCondition = MultiTensorValue( scope, lInitializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    auto lX = ConstantScalarValue( scope, 0.245f );
    auto y  = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto result0 = Where( scope, lCondition, lX, y );

    scope.Run( result0 );

    std::vector<float> lYValues0      = y.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    std::vector<float> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    std::vector<float> expectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = lValues0[i] ? 0.245f : lYValues0[i];
    REQUIRE( lResultValues0 == expectedValues0 );
}

TEST_CASE( "Where Scalar_Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto                 lValues00 = RandomBool( 12 * 23 );
    auto                 lValues01 = RandomBool( 13 * 24 );
    std::vector<uint8_t> lValues0;
    lValues0.insert( lValues0.end(), lValues00.begin(), lValues00.end() );
    lValues0.insert( lValues0.end(), lValues01.begin(), lValues01.end() );
    data_initializer_t lInitializer0( lValues0 );
    auto               lCondition = MultiTensorValue( scope, lInitializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    auto lX = ConstantScalarValue( scope, 0.245f );

    std::vector<scalar_value_t> constants{ 0.2345f, 0.345f };
    auto                        y = VectorValue( scope, constants );

    auto result0 = Where( scope, lCondition, lX, y );

    scope.Run( result0 );

    std::vector<float> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 0 );
    std::vector<float> expectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = lValues00[i] ? 0.245f : std::get<float>( constants[0] );
    REQUIRE( lResultValues0 == expectedValues0 );

    std::vector<float> lResultValues1 = result0.Get<multi_tensor_value_t>().mValue.FetchBufferAt<float>( 1 );
    std::vector<float> expectedValues1( lResultValues1.size() );
    for( uint32_t i = 0; i < expectedValues1.size(); i++ )
        expectedValues1[i] = lValues01[i] ? 0.245f : std::get<float>( constants[1] );
    REQUIRE( lResultValues1 == expectedValues1 );
}

TEST_CASE( "Where Scalar_Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto               lValues0 = RandomBool( 12 * 23 + 13 * 24 );
    data_initializer_t lInitializer0( lValues0 );
    auto               lCondition = MultiTensorValue( scope, lInitializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    auto lX = ConstantScalarValue( scope, 0.9234587f );
    auto y  = ConstantScalarValue( scope, 0.1324978f );

    auto result0 = Where( scope, lCondition, lX, y );

    scope.Run( result0 );

    std::vector<float> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    std::vector<float> expectedValues0( lResultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = lValues0[i] ? 0.9234587f : 0.1324978f;
    REQUIRE( lResultValues0 == expectedValues0 );
}

TEST_CASE( "ArraySlice VECTOR_VECTOR", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    auto lSliceStart = std::vector<uint32_t>{ 15, 27, 400 };
    auto lSliceEnd   = std::vector<uint32_t>{ 81, 59, 510 };

    std::vector<uint32_t> dim1{ 7, 3, 1024 };
    std::vector<uint64_t> lValues1;
    std::vector<uint64_t> expectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<uint64_t>( 1024 );
            lValues1.insert( lValues1.end(), y.begin(), y.end() );
            expectedValues1.insert( expectedValues1.end(), y.begin() + lSliceStart[0], y.begin() + lSliceEnd[0] + 1 );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<uint64_t> lValues2;
    std::vector<uint64_t> lExpectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<uint64_t>( 256 );
            lValues2.insert( lValues2.end(), y.begin(), y.end() );
            lExpectedValues2.insert( lExpectedValues2.end(), y.begin() + lSliceStart[1], y.begin() + lSliceEnd[1] + 1 );
        }
    }

    std::vector<uint32_t> lDim3{ 3, 5, 512 };
    std::vector<uint64_t> lValues3;
    std::vector<uint64_t> lExpectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto y = RandomNumber<uint64_t>( 512 );
            lValues3.insert( lValues3.end(), y.begin(), y.end() );
            lExpectedValues3.insert( lExpectedValues3.end(), y.begin() + lSliceStart[2], y.begin() + lSliceEnd[2] + 1 );
        }
    }

    std::vector<uint64_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    data_initializer_t lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( scope, lInputInitializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( uint64_t ) ) );

    auto lStart = VectorValue( scope, lSliceStart );
    auto lEnd   = VectorValue( scope, lSliceEnd );

    auto result0 = Slice( scope, lInputTensor, lStart, lEnd );
    scope.Run( result0 );

    auto lOutputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.Rank == 3 );
    REQUIRE( lOutputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, lSliceEnd[0] - lSliceStart[0] + 1 } );
    REQUIRE( lOutputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, lSliceEnd[1] - lSliceStart[1] + 1 } );
    REQUIRE( lOutputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, lSliceEnd[2] - lSliceStart[2] + 1 } );

    std::vector<uint64_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<uint64_t>>{ expectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<uint64_t> lResultValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( lResultValues == expectedValues );
}

TEST_CASE( "ArraySlice SCALAR_VECTOR", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    uint32_t lSliceStart = 25;
    auto     lSliceEnd   = std::vector<uint32_t>{ 81, 59, 51 };

    std::vector<uint32_t> dim1{ 7, 3, 1024 };
    std::vector<uint64_t> lValues1;
    std::vector<uint64_t> expectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<uint64_t>( 1024 );
            lValues1.insert( lValues1.end(), y.begin(), y.end() );
            expectedValues1.insert( expectedValues1.end(), y.begin() + lSliceStart, y.begin() + lSliceEnd[0] + 1 );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<uint64_t> lValues2;
    std::vector<uint64_t> lExpectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<uint64_t>( 256 );
            lValues2.insert( lValues2.end(), y.begin(), y.end() );
            lExpectedValues2.insert( lExpectedValues2.end(), y.begin() + lSliceStart, y.begin() + lSliceEnd[1] + 1 );
        }
    }

    std::vector<uint32_t> lDim3{ 3, 5, 512 };
    std::vector<uint64_t> lValues3;
    std::vector<uint64_t> lExpectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto y = RandomNumber<uint64_t>( 512 );
            lValues3.insert( lValues3.end(), y.begin(), y.end() );
            lExpectedValues3.insert( lExpectedValues3.end(), y.begin() + lSliceStart, y.begin() + lSliceEnd[2] + 1 );
        }
    }

    std::vector<uint64_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    data_initializer_t lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( scope, lInputInitializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( uint64_t ) ) );

    auto lStart = ConstantScalarValue( scope, lSliceStart );
    auto lEnd   = VectorValue( scope, lSliceEnd );

    auto result0 = Slice( scope, lInputTensor, lStart, lEnd );
    scope.Run( result0 );

    auto lOutputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.Rank == 3 );
    REQUIRE( lOutputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, lSliceEnd[0] - lSliceStart + 1 } );
    REQUIRE( lOutputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, lSliceEnd[1] - lSliceStart + 1 } );
    REQUIRE( lOutputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, lSliceEnd[2] - lSliceStart + 1 } );

    std::vector<uint64_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<uint64_t>>{ expectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<uint64_t> lResultValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( lResultValues == expectedValues );
}

TEST_CASE( "ArraySlice VECTOR_SCALAR", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    auto     lSliceStart = std::vector<uint32_t>{ 15, 27, 39 };
    uint32_t lSliceEnd   = 65;

    std::vector<uint32_t> dim1{ 7, 3, 1024 };
    std::vector<uint64_t> lValues1;
    std::vector<uint64_t> expectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<uint64_t>( 1024 );
            lValues1.insert( lValues1.end(), y.begin(), y.end() );
            expectedValues1.insert( expectedValues1.end(), y.begin() + lSliceStart[0], y.begin() + lSliceEnd + 1 );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<uint64_t> lValues2;
    std::vector<uint64_t> lExpectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<uint64_t>( 256 );
            lValues2.insert( lValues2.end(), y.begin(), y.end() );
            lExpectedValues2.insert( lExpectedValues2.end(), y.begin() + lSliceStart[1], y.begin() + lSliceEnd + 1 );
        }
    }

    std::vector<uint32_t> lDim3{ 3, 5, 512 };
    std::vector<uint64_t> lValues3;
    std::vector<uint64_t> lExpectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto y = RandomNumber<uint64_t>( 512 );
            lValues3.insert( lValues3.end(), y.begin(), y.end() );
            lExpectedValues3.insert( lExpectedValues3.end(), y.begin() + lSliceStart[2], y.begin() + lSliceEnd + 1 );
        }
    }

    std::vector<uint64_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    data_initializer_t lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( scope, lInputInitializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( uint64_t ) ) );

    auto lStart = VectorValue( scope, lSliceStart );
    auto lEnd   = ConstantScalarValue( scope, lSliceEnd );

    auto result0 = Slice( scope, lInputTensor, lStart, lEnd );
    scope.Run( result0 );

    auto lOutputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.Rank == 3 );
    REQUIRE( lOutputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, lSliceEnd - lSliceStart[0] + 1 } );
    REQUIRE( lOutputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, lSliceEnd - lSliceStart[1] + 1 } );
    REQUIRE( lOutputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, lSliceEnd - lSliceStart[2] + 1 } );

    std::vector<uint64_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<uint64_t>>{ expectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<uint64_t> lResultValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( lResultValues == expectedValues );
}

TEST_CASE( "ArraySlice SCALAR_SCALAR", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    uint32_t lSliceStart = 15;
    uint32_t lSliceEnd   = 65;

    std::vector<uint32_t> dim1{ 7, 3, 1024 };
    std::vector<uint64_t> lValues1;
    std::vector<uint64_t> expectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<uint64_t>( 1024 );
            lValues1.insert( lValues1.end(), y.begin(), y.end() );
            expectedValues1.insert( expectedValues1.end(), y.begin() + lSliceStart, y.begin() + lSliceEnd + 1 );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<uint64_t> lValues2;
    std::vector<uint64_t> lExpectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<uint64_t>( 256 );
            lValues2.insert( lValues2.end(), y.begin(), y.end() );
            lExpectedValues2.insert( lExpectedValues2.end(), y.begin() + lSliceStart, y.begin() + lSliceEnd + 1 );
        }
    }

    std::vector<uint32_t> lDim3{ 3, 5, 512 };
    std::vector<uint64_t> lValues3;
    std::vector<uint64_t> lExpectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto y = RandomNumber<uint64_t>( 512 );
            lValues3.insert( lValues3.end(), y.begin(), y.end() );
            lExpectedValues3.insert( lExpectedValues3.end(), y.begin() + lSliceStart, y.begin() + lSliceEnd + 1 );
        }
    }

    std::vector<uint64_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    data_initializer_t lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( scope, lInputInitializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( uint64_t ) ) );

    auto lStart = ConstantScalarValue( scope, lSliceStart );
    auto lEnd   = ConstantScalarValue( scope, lSliceEnd );

    auto result0 = Slice( scope, lInputTensor, lStart, lEnd );
    scope.Run( result0 );

    auto lOutputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.Rank == 3 );
    REQUIRE( lOutputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, lSliceEnd - lSliceStart + 1 } );
    REQUIRE( lOutputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, lSliceEnd - lSliceStart + 1 } );
    REQUIRE( lOutputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, lSliceEnd - lSliceStart + 1 } );

    std::vector<uint64_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<uint64_t>>{ expectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<uint64_t> lResultValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( lResultValues == expectedValues );
}

TEST_CASE( "ArraySummation VECTOR_VECTOR", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    auto lSliceStart = std::vector<uint32_t>{ 15, 27, 400 };
    auto lSliceEnd   = std::vector<uint32_t>{ 81, 59, 510 };

    std::vector<uint32_t> dim1{ 7, 3, 1024 };
    std::vector<uint64_t> lValues1;
    std::vector<uint64_t> expectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<uint64_t>( 1024 );
            lValues1.insert( lValues1.end(), y.begin(), y.end() );
            expectedValues1.push_back( std::accumulate( y.begin() + lSliceStart[0], y.begin() + lSliceEnd[0] + 1,
                                                        static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<uint64_t> lValues2;
    std::vector<uint64_t> lExpectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<uint64_t>( 256 );
            lValues2.insert( lValues2.end(), y.begin(), y.end() );
            lExpectedValues2.push_back( std::accumulate( y.begin() + lSliceStart[1], y.begin() + lSliceEnd[1] + 1,
                                                         static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint32_t> lDim3{ 3, 5, 512 };
    std::vector<uint64_t> lValues3;
    std::vector<uint64_t> lExpectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto y = RandomNumber<uint64_t>( 512 );
            lValues3.insert( lValues3.end(), y.begin(), y.end() );
            lExpectedValues3.push_back( std::accumulate( y.begin() + lSliceStart[2], y.begin() + lSliceEnd[2] + 1,
                                                         static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint64_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    data_initializer_t lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( scope, lInputInitializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( uint64_t ) ) );

    auto lStart = VectorValue( scope, lSliceStart );
    auto lEnd   = VectorValue( scope, lSliceEnd );

    auto result0 = Summation( scope, lInputTensor, lStart, lEnd );
    scope.Run( result0 );

    auto lOutputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.Rank == 2 );
    REQUIRE( lOutputShape.Shape[0] == std::vector<uint32_t>{ 7, 3 } );
    REQUIRE( lOutputShape.Shape[1] == std::vector<uint32_t>{ 2, 7 } );
    REQUIRE( lOutputShape.Shape[2] == std::vector<uint32_t>{ 3, 5 } );

    std::vector<uint64_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<uint64_t>>{ expectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<uint64_t> lResultValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( lResultValues == expectedValues );
}

TEST_CASE( "ArraySummation SCALAR_VECTOR", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    uint32_t lSliceStart = 25;
    auto     lSliceEnd   = std::vector<uint32_t>{ 81, 59, 51 };

    std::vector<uint32_t> dim1{ 7, 3, 1024 };
    std::vector<uint64_t> lValues1;
    std::vector<uint64_t> expectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<uint64_t>( 1024 );
            lValues1.insert( lValues1.end(), y.begin(), y.end() );
            expectedValues1.push_back( std::accumulate( y.begin() + lSliceStart, y.begin() + lSliceEnd[0] + 1,
                                                        static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<uint64_t> lValues2;
    std::vector<uint64_t> lExpectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<uint64_t>( 256 );
            lValues2.insert( lValues2.end(), y.begin(), y.end() );
            lExpectedValues2.push_back( std::accumulate( y.begin() + lSliceStart, y.begin() + lSliceEnd[1] + 1,
                                                         static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint32_t> lDim3{ 3, 5, 512 };
    std::vector<uint64_t> lValues3;
    std::vector<uint64_t> lExpectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto y = RandomNumber<uint64_t>( 512 );
            lValues3.insert( lValues3.end(), y.begin(), y.end() );
            lExpectedValues3.push_back( std::accumulate( y.begin() + lSliceStart, y.begin() + lSliceEnd[2] + 1,
                                                         static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint64_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    data_initializer_t lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( scope, lInputInitializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( uint64_t ) ) );

    auto lStart = ConstantScalarValue( scope, lSliceStart );
    auto lEnd   = VectorValue( scope, lSliceEnd );

    auto result0 = Summation( scope, lInputTensor, lStart, lEnd );
    scope.Run( result0 );

    auto lOutputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.Rank == 2 );
    REQUIRE( lOutputShape.Shape[0] == std::vector<uint32_t>{ 7, 3 } );
    REQUIRE( lOutputShape.Shape[1] == std::vector<uint32_t>{ 2, 7 } );
    REQUIRE( lOutputShape.Shape[2] == std::vector<uint32_t>{ 3, 5 } );

    std::vector<uint64_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<uint64_t>>{ expectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<uint64_t> lResultValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( lResultValues == expectedValues );
}

TEST_CASE( "ArraySummation VECTOR_SCALAR", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    auto     lSliceStart = std::vector<uint32_t>{ 15, 27, 39 };
    uint32_t lSliceEnd   = 65;

    std::vector<uint32_t> dim1{ 7, 3, 1024 };
    std::vector<uint64_t> lValues1;
    std::vector<uint64_t> expectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<uint64_t>( 1024 );
            lValues1.insert( lValues1.end(), y.begin(), y.end() );
            expectedValues1.push_back( std::accumulate( y.begin() + lSliceStart[0], y.begin() + lSliceEnd + 1,
                                                        static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<uint64_t> lValues2;
    std::vector<uint64_t> lExpectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<uint64_t>( 256 );
            lValues2.insert( lValues2.end(), y.begin(), y.end() );
            lExpectedValues2.push_back( std::accumulate( y.begin() + lSliceStart[1], y.begin() + lSliceEnd + 1,
                                                         static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint32_t> lDim3{ 3, 5, 512 };
    std::vector<uint64_t> lValues3;
    std::vector<uint64_t> lExpectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto y = RandomNumber<uint64_t>( 512 );
            lValues3.insert( lValues3.end(), y.begin(), y.end() );
            lExpectedValues3.push_back( std::accumulate( y.begin() + lSliceStart[2], y.begin() + lSliceEnd + 1,
                                                         static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint64_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    data_initializer_t lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( scope, lInputInitializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( uint64_t ) ) );

    auto lStart = VectorValue( scope, lSliceStart );
    auto lEnd   = ConstantScalarValue( scope, lSliceEnd );

    auto result0 = Summation( scope, lInputTensor, lStart, lEnd );
    scope.Run( result0 );

    auto lOutputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.Rank == 2 );
    REQUIRE( lOutputShape.Shape[0] == std::vector<uint32_t>{ 7, 3 } );
    REQUIRE( lOutputShape.Shape[1] == std::vector<uint32_t>{ 2, 7 } );
    REQUIRE( lOutputShape.Shape[2] == std::vector<uint32_t>{ 3, 5 } );

    std::vector<uint64_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<uint64_t>>{ expectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<uint64_t> lResultValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( lResultValues == expectedValues );
}

TEST_CASE( "ArraySummation SCALAR_SCALAR", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    uint32_t lSliceStart = 15;
    uint32_t lSliceEnd   = 65;

    std::vector<uint32_t> dim1{ 7, 3, 1024 };
    std::vector<uint64_t> lValues1;
    std::vector<uint64_t> expectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<uint64_t>( 1024 );
            lValues1.insert( lValues1.end(), y.begin(), y.end() );
            expectedValues1.push_back( std::accumulate( y.begin() + lSliceStart, y.begin() + lSliceEnd + 1, static_cast<uint64_t>( 0 ),
                                                        std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<uint64_t> lValues2;
    std::vector<uint64_t> lExpectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<uint64_t>( 256 );
            lValues2.insert( lValues2.end(), y.begin(), y.end() );
            lExpectedValues2.push_back( std::accumulate( y.begin() + lSliceStart, y.begin() + lSliceEnd + 1,
                                                         static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint32_t> lDim3{ 3, 5, 512 };
    std::vector<uint64_t> lValues3;
    std::vector<uint64_t> lExpectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto y = RandomNumber<uint64_t>( 512 );
            lValues3.insert( lValues3.end(), y.begin(), y.end() );
            lExpectedValues3.push_back( std::accumulate( y.begin() + lSliceStart, y.begin() + lSliceEnd + 1,
                                                         static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint64_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    data_initializer_t lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( scope, lInputInitializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( uint64_t ) ) );

    auto lStart = ConstantScalarValue( scope, lSliceStart );
    auto lEnd   = ConstantScalarValue( scope, lSliceEnd );

    auto result0 = Summation( scope, lInputTensor, lStart, lEnd );
    scope.Run( result0 );

    auto lOutputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.Rank == 2 );
    REQUIRE( lOutputShape.Shape[0] == std::vector<uint32_t>{ 7, 3 } );
    REQUIRE( lOutputShape.Shape[1] == std::vector<uint32_t>{ 2, 7 } );
    REQUIRE( lOutputShape.Shape[2] == std::vector<uint32_t>{ 3, 5 } );

    std::vector<uint64_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<uint64_t>>{ expectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<uint64_t> lResultValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( lResultValues == expectedValues );
}

TEST_CASE( "ArraySummation full", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    uint32_t lSliceStart = 15;
    uint32_t lSliceEnd   = 65;

    std::vector<uint32_t> dim1{ 7, 3, 1024 };
    std::vector<uint64_t> lValues1;
    std::vector<uint64_t> expectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<uint64_t>( 1024 );
            lValues1.insert( lValues1.end(), y.begin(), y.end() );
            expectedValues1.push_back( std::accumulate( y.begin(), y.end(), static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<uint64_t> lValues2;
    std::vector<uint64_t> lExpectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<uint64_t>( 256 );
            lValues2.insert( lValues2.end(), y.begin(), y.end() );
            lExpectedValues2.push_back( std::accumulate( y.begin(), y.end(), static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint32_t> lDim3{ 3, 5, 512 };
    std::vector<uint64_t> lValues3;
    std::vector<uint64_t> lExpectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto y = RandomNumber<uint64_t>( 512 );
            lValues3.insert( lValues3.end(), y.begin(), y.end() );
            lExpectedValues3.push_back( std::accumulate( y.begin(), y.end(), static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint64_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    data_initializer_t lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( scope, lInputInitializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( uint64_t ) ) );

    auto lStart = ConstantScalarValue( scope, lSliceStart );
    auto lEnd   = ConstantScalarValue( scope, lSliceEnd );

    auto result0 = Summation( scope, lInputTensor );
    scope.Run( result0 );

    auto lOutputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.Rank == 2 );
    REQUIRE( lOutputShape.Shape[0] == std::vector<uint32_t>{ 7, 3 } );
    REQUIRE( lOutputShape.Shape[1] == std::vector<uint32_t>{ 2, 7 } );
    REQUIRE( lOutputShape.Shape[2] == std::vector<uint32_t>{ 3, 5 } );

    std::vector<uint64_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<uint64_t>>{ expectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<uint64_t> lResultValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint64_t>();
    REQUIRE( lResultValues == expectedValues );
}

TEST_CASE( "CountTrue", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 7, 3, 1024 };
    std::vector<uint8_t>  lValues1;
    std::vector<uint32_t> expectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomBool( 1024 );
            lValues1.insert( lValues1.end(), y.begin(), y.end() );

            uint32_t lTrueCount = 0;
            for( auto &x : y )
                lTrueCount += x ? 1 : 0;
            expectedValues1.push_back( lTrueCount );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<uint8_t>  lValues2;
    std::vector<uint32_t> lExpectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomBool( 256 );
            lValues2.insert( lValues2.end(), y.begin(), y.end() );
            uint32_t lTrueCount = 0;
            for( auto &x : y )
                lTrueCount += x ? 1 : 0;
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
            auto y = RandomBool( 512 );
            lValues3.insert( lValues3.end(), y.begin(), y.end() );
            uint32_t lTrueCount = 0;
            for( auto &x : y )
                lTrueCount += x ? 1 : 0;
            lExpectedValues3.push_back( lTrueCount );
        }
    }

    std::vector<uint8_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    data_initializer_t lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( scope, lInputInitializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( uint8_t ) ) );

    auto result0 = CountTrue( scope, lInputTensor );
    scope.Run( result0 );

    auto lOutputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.Rank == 2 );
    REQUIRE( lOutputShape.Shape[0] == std::vector<uint32_t>{ 7, 3 } );
    REQUIRE( lOutputShape.Shape[1] == std::vector<uint32_t>{ 2, 7 } );
    REQUIRE( lOutputShape.Shape[2] == std::vector<uint32_t>{ 3, 5 } );

    std::vector<uint32_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<uint32_t>>{ expectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<uint32_t> lResultValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint32_t>();
    REQUIRE( lResultValues == expectedValues );
}

TEST_CASE( "CountNonZero", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 7, 3, 1024 };
    std::vector<uint64_t> lValues1;
    std::vector<uint32_t> expectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<uint64_t>( 1024 );
            lValues1.insert( lValues1.end(), y.begin(), y.end() );

            uint32_t lTrueCount = 0;
            for( auto &x : y )
                lTrueCount += ( x != 0 ) ? 1 : 0;
            expectedValues1.push_back( lTrueCount );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<uint64_t> lValues2;
    std::vector<uint32_t> lExpectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<uint64_t>( 256 );
            lValues2.insert( lValues2.end(), y.begin(), y.end() );
            uint32_t lTrueCount = 0;
            for( auto &x : y )
                lTrueCount += ( x != 0 ) ? 1 : 0;
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
            auto y = RandomNumber<uint64_t>( 512 );
            lValues3.insert( lValues3.end(), y.begin(), y.end() );
            uint32_t lTrueCount = 0;
            for( auto &x : y )
                lTrueCount += ( x != 0 ) ? 1 : 0;
            lExpectedValues3.push_back( lTrueCount );
        }
    }

    std::vector<uint64_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    data_initializer_t lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( scope, lInputInitializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( uint64_t ) ) );

    auto result0 = CountNonZero( scope, lInputTensor );
    scope.Run( result0 );

    auto lOutputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.Rank == 2 );
    REQUIRE( lOutputShape.Shape[0] == std::vector<uint32_t>{ 7, 3 } );
    REQUIRE( lOutputShape.Shape[1] == std::vector<uint32_t>{ 2, 7 } );
    REQUIRE( lOutputShape.Shape[2] == std::vector<uint32_t>{ 3, 5 } );

    std::vector<uint32_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<uint32_t>>{ expectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<uint32_t> lResultValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint32_t>();
    REQUIRE( lResultValues == expectedValues );
}

TEST_CASE( "CountZero", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 7, 3, 1024 };
    std::vector<uint64_t> lValues1;
    std::vector<uint32_t> expectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<uint64_t>( 1024 );
            lValues1.insert( lValues1.end(), y.begin(), y.end() );

            uint32_t lTrueCount = 0;
            for( auto &x : y )
                lTrueCount += ( x == 0 ) ? 1 : 0;
            expectedValues1.push_back( lTrueCount );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<uint64_t> lValues2;
    std::vector<uint32_t> lExpectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<uint64_t>( 256 );
            lValues2.insert( lValues2.end(), y.begin(), y.end() );
            uint32_t lTrueCount = 0;
            for( auto &x : y )
                lTrueCount += ( x == 0 ) ? 1 : 0;
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
            auto y = RandomNumber<uint64_t>( 512 );
            lValues3.insert( lValues3.end(), y.begin(), y.end() );
            uint32_t lTrueCount = 0;
            for( auto &x : y )
                lTrueCount += ( x == 0 ) ? 1 : 0;
            lExpectedValues3.push_back( lTrueCount );
        }
    }

    std::vector<uint64_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    data_initializer_t lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( scope, lInputInitializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( uint64_t ) ) );

    auto result0 = CountZero( scope, lInputTensor );
    scope.Run( result0 );

    auto lOutputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.Rank == 2 );
    REQUIRE( lOutputShape.Shape[0] == std::vector<uint32_t>{ 7, 3 } );
    REQUIRE( lOutputShape.Shape[1] == std::vector<uint32_t>{ 2, 7 } );
    REQUIRE( lOutputShape.Shape[2] == std::vector<uint32_t>{ 3, 5 } );

    std::vector<uint32_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<uint32_t>>{ expectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<uint32_t> lResultValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint32_t>();
    REQUIRE( lResultValues == expectedValues );
}

TEST_CASE( "Floor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 7, 3, 500 };
    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<uint32_t> lDim3{ 3, 5, 512 };

    auto opNode = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( float ) ) );

    auto result0 = Floor( scope, opNode );
    scope.Run( result0 );

    std::vector<float> lResultValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    std::vector<float> expectedValues( opNode.Get<multi_tensor_value_t>().mValue.SizeAs<float>() );
    std::vector<float> lOpNodeValues = opNode.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    for( uint32_t i = 0; i < lOpNodeValues.size(); i++ )
    {
        expectedValues[i] = std::floor( lOpNodeValues[i] );
    }

    REQUIRE( lResultValues == expectedValues );
}

TEST_CASE( "Ceiling", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 7, 3, 500 };
    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<uint32_t> lDim3{ 3, 5, 512 };

    auto opNode = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( float ) ) );

    auto result0 = Ceil( scope, opNode );
    scope.Run( result0 );

    std::vector<float> lResultValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    std::vector<float> expectedValues( opNode.Get<multi_tensor_value_t>().mValue.SizeAs<float>() );
    std::vector<float> lOpNodeValues = opNode.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    for( uint32_t i = 0; i < lOpNodeValues.size(); i++ )
    {
        expectedValues[i] = std::ceil( lOpNodeValues[i] );
    }

    REQUIRE( lResultValues == expectedValues );
}

TEST_CASE( "Absolute value", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 7, 3, 500 };
    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<uint32_t> lDim3{ 3, 5, 512 };

    auto opNode = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( float ) ) );

    auto result0 = Abs( scope, opNode );
    scope.Run( result0 );

    std::vector<float> lResultValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    std::vector<float> expectedValues( opNode.Get<multi_tensor_value_t>().mValue.SizeAs<float>() );
    std::vector<float> lOpNodeValues = opNode.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    for( uint32_t i = 0; i < lOpNodeValues.size(); i++ )
    {
        expectedValues[i] = std::abs( lOpNodeValues[i] );
    }

    REQUIRE( lResultValues == expectedValues );
}

TEST_CASE( "Square roots", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 17, 12, 51 };
    std::vector<uint32_t> dim2{ 12, 17, 23 };
    std::vector<uint32_t> lDim3{ 13, 15, 52 };

    auto opNode = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( float ) ) );

    auto result0 = Sqrt( scope, opNode );
    scope.Run( result0 );

    std::vector<float> lResultValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    std::vector<float> expectedValues( opNode.Get<multi_tensor_value_t>().mValue.SizeAs<float>() );
    std::vector<float> lOpNodeValues = opNode.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    for( uint32_t i = 0; i < lOpNodeValues.size(); i++ )
    {
        expectedValues[i] = std::sqrt( lOpNodeValues[i] );
    }

    REQUIRE( lResultValues.size() == expectedValues.size() );
    std::vector<bool> lComparison( lResultValues.size() );
    for( uint32_t i = 0; i < lOpNodeValues.size(); i++ )
    {
        lComparison[i] =
            ( std::isnan( lResultValues[i] ) && std::isnan( expectedValues[i] ) ) || ( lResultValues[i] == expectedValues[i] );
    }

    REQUIRE( std::all_of( lComparison.begin(), lComparison.end(), []( auto x ) { return x; } ) );
}

TEST_CASE( "Rounding", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 17, 12, 51 };
    std::vector<uint32_t> dim2{ 12, 17, 23 };
    std::vector<uint32_t> lDim3{ 13, 15, 52 };

    auto opNode = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( float ) ) );

    auto result0 = Round( scope, opNode );
    scope.Run( result0 );

    std::vector<float> lResultValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    std::vector<float> expectedValues( opNode.Get<multi_tensor_value_t>().mValue.SizeAs<float>() );
    std::vector<float> lOpNodeValues = opNode.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    for( uint32_t i = 0; i < lOpNodeValues.size(); i++ )
    {
        expectedValues[i] = std::round( lOpNodeValues[i] );
    }

    REQUIRE( lResultValues.size() == expectedValues.size() );
    std::vector<bool> lComparison( lResultValues.size() );
    for( uint32_t i = 0; i < lOpNodeValues.size(); i++ )
    {
        lComparison[i] = ( lResultValues[i] == expectedValues[i] );
    }

    REQUIRE( std::all_of( lComparison.begin(), lComparison.end(), []( auto x ) { return x; } ) );
}

TEST_CASE( "Finite differences", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 7, 3, 1024 };
    std::vector<int64_t>  lValues1;
    std::vector<int64_t>  expectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<int64_t>( 1024 );
            lValues1.insert( lValues1.end(), y.begin(), y.end() );
            for( uint32_t l = 0; l < y.size() - 1; l++ )
                expectedValues1.push_back( y[l + 1] - y[l] );
            expectedValues1.push_back( static_cast<int64_t>( 0 ) );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<int64_t>  lValues2;
    std::vector<int64_t>  lExpectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<int64_t>( 256 );
            lValues2.insert( lValues2.end(), y.begin(), y.end() );
            for( uint32_t l = 0; l < y.size() - 1; l++ )
                lExpectedValues2.push_back( y[l + 1] - y[l] );
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
            auto y = RandomNumber<int64_t>( 512 );
            lValues3.insert( lValues3.end(), y.begin(), y.end() );
            for( uint32_t l = 0; l < y.size() - 1; l++ )
                lExpectedValues3.push_back( y[l + 1] - y[l] );
            lExpectedValues3.push_back( static_cast<int64_t>( 0 ) );
        }
    }

    std::vector<int64_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    data_initializer_t lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( scope, lInputInitializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( int64_t ) ) );

    auto result0 = Diff( scope, lInputTensor, 1 );
    scope.Run( result0 );

    auto lOutputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.Rank == 3 );
    REQUIRE( lOutputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 1024 } );
    REQUIRE( lOutputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 256 } );
    REQUIRE( lOutputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 512 } );

    std::vector<int64_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<int64_t>>{ expectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<int64_t> lResultValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<int64_t>();
    REQUIRE( lResultValues == expectedValues );
}

TEST_CASE( "Finite shift to the left  by 1", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 7, 3, 1024 };
    std::vector<int64_t>  lValues1;
    std::vector<int64_t>  expectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<int64_t>( 1024 );
            lValues1.insert( lValues1.end(), y.begin(), y.end() );
            for( uint32_t l = 0; l < y.size() - 1; l++ )
                expectedValues1.push_back( y[l + 1] );
            expectedValues1.push_back( static_cast<int64_t>( 121212 ) );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<int64_t>  lValues2;
    std::vector<int64_t>  lExpectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<int64_t>( 256 );
            lValues2.insert( lValues2.end(), y.begin(), y.end() );
            for( uint32_t l = 0; l < y.size() - 1; l++ )
                lExpectedValues2.push_back( y[l + 1] );
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
            auto y = RandomNumber<int64_t>( 512 );
            lValues3.insert( lValues3.end(), y.begin(), y.end() );
            for( uint32_t l = 0; l < y.size() - 1; l++ )
                lExpectedValues3.push_back( y[l + 1] );
            lExpectedValues3.push_back( static_cast<int64_t>( 121212 ) );
        }
    }

    std::vector<int64_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    data_initializer_t lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( scope, lInputInitializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( int64_t ) ) );

    auto lFillValue = ConstantScalarValue( scope, static_cast<int64_t>( 121212 ) );
    auto result0   = Shift( scope, lInputTensor, -1, lFillValue );
    scope.Run( result0 );

    auto lOutputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.Rank == 3 );
    REQUIRE( lOutputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 1024 } );
    REQUIRE( lOutputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 256 } );
    REQUIRE( lOutputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 512 } );

    std::vector<int64_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<int64_t>>{ expectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<int64_t> lResultValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<int64_t>();
    REQUIRE( lResultValues == expectedValues );
}

TEST_CASE( "Finite shift to the left by 3", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 7, 3, 1024 };
    std::vector<int64_t>  lValues1;
    std::vector<int64_t>  expectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<int64_t>( 1024 );
            lValues1.insert( lValues1.end(), y.begin(), y.end() );
            for( uint32_t l = 0; l < y.size() - 3; l++ )
                expectedValues1.push_back( y[l + 3] );
            expectedValues1.push_back( static_cast<int64_t>( 121212 ) );
            expectedValues1.push_back( static_cast<int64_t>( 121212 ) );
            expectedValues1.push_back( static_cast<int64_t>( 121212 ) );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<int64_t>  lValues2;
    std::vector<int64_t>  lExpectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<int64_t>( 256 );
            lValues2.insert( lValues2.end(), y.begin(), y.end() );
            for( uint32_t l = 0; l < y.size() - 3; l++ )
                lExpectedValues2.push_back( y[l + 3] );
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
            auto y = RandomNumber<int64_t>( 512 );
            lValues3.insert( lValues3.end(), y.begin(), y.end() );
            for( uint32_t l = 0; l < y.size() - 3; l++ )
                lExpectedValues3.push_back( y[l + 3] );
            lExpectedValues3.push_back( static_cast<int64_t>( 121212 ) );
            lExpectedValues3.push_back( static_cast<int64_t>( 121212 ) );
            lExpectedValues3.push_back( static_cast<int64_t>( 121212 ) );
        }
    }

    std::vector<int64_t> lInputValues;
    lInputValues.insert( lInputValues.end(), lValues1.begin(), lValues1.end() );
    lInputValues.insert( lInputValues.end(), lValues2.begin(), lValues2.end() );
    lInputValues.insert( lInputValues.end(), lValues3.begin(), lValues3.end() );

    data_initializer_t lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( scope, lInputInitializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( int64_t ) ) );

    auto lFillValue = ConstantScalarValue( scope, static_cast<int64_t>( 121212 ) );
    auto result0   = Shift( scope, lInputTensor, -3, lFillValue );
    scope.Run( result0 );

    auto lOutputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.Rank == 3 );
    REQUIRE( lOutputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 1024 } );
    REQUIRE( lOutputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 256 } );
    REQUIRE( lOutputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 512 } );

    std::vector<int64_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<int64_t>>{ expectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<int64_t> lResultValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<int64_t>();
    REQUIRE( lResultValues == expectedValues );
}

TEST_CASE( "1D convolution", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.mType = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 7, 3, 124 };
    std::vector<int64_t>  lValues1;
    std::vector<int64_t>  expectedValues1;

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
                if( k <= i )
                    lAcc += aX[i - k] * aY[k];
            }
            lOutput[i] = lAcc;
        }
        return lOutput;
    };

    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<int64_t>( 124, -10000, 10000 );
            lValues1.insert( lValues1.end(), y.begin(), y.end() );

            auto lZ = RandomNumber<int64_t>( 34, -10000, 10000 );
            lKernel1.insert( lKernel1.end(), lZ.begin(), lZ.end() );

            auto lC = lConv1D( y, lZ );
            expectedValues1.insert( expectedValues1.end(), lC.begin(), lC.end() );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 75 };
    std::vector<int64_t>  lValues2;
    std::vector<int64_t>  lExpectedValues2;

    std::vector<uint32_t> lKDim2{ 2, 7, 42 };
    std::vector<int64_t>  lKernel2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<int64_t>( 75, -10000, 10000 );
            lValues2.insert( lValues2.end(), y.begin(), y.end() );

            auto lZ = RandomNumber<int64_t>( 42, -10000, 10000 );
            lKernel2.insert( lKernel2.end(), lZ.begin(), lZ.end() );

            auto lC = lConv1D( y, lZ );
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
            auto y = RandomNumber<int64_t>( 23, -10000, 10000 );
            lValues3.insert( lValues3.end(), y.begin(), y.end() );

            auto lZ = RandomNumber<int64_t>( 5, -10000, 10000 );
            lKernel3.insert( lKernel3.end(), lZ.begin(), lZ.end() );

            auto lC = lConv1D( y, lZ );
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

    data_initializer_t lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( scope, lInputInitializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( int64_t ) ) );

    data_initializer_t lKernelInitializer( lKernelValues );
    auto lKernelensor = MultiTensorValue( scope, lKernelInitializer, tensor_shape_t( { lKDim1, lKDim2, lKDim3 }, sizeof( int64_t ) ) );

    auto result0 = Conv1D( scope, lInputTensor, lKernelensor );
    scope.Run( result0 );

    auto lOutputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.Rank == 3 );
    REQUIRE( lOutputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( lOutputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( lOutputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<int64_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<int64_t>>{ expectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<int64_t> lResultValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<int64_t>();
    REQUIRE( lResultValues == expectedValues );
}

TEST_CASE( "1D convolution (uint32_t)", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> dim1{ 7, 3, 124 };
    std::vector<uint32_t> lValues1;
    std::vector<uint32_t> expectedValues1;

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
                if( k <= i )
                    lAcc += aX[i - k] * aY[k];
            }
            lOutput[i] = lAcc;
        }
        return lOutput;
    };

    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<uint32_t>( 124, 0, 10000 );
            lValues1.insert( lValues1.end(), y.begin(), y.end() );

            auto lZ = RandomNumber<uint32_t>( 34, 0, 10000 );
            lKernel1.insert( lKernel1.end(), lZ.begin(), lZ.end() );

            auto lC = lConv1D( y, lZ );
            expectedValues1.insert( expectedValues1.end(), lC.begin(), lC.end() );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 75 };
    std::vector<uint32_t> lValues2;
    std::vector<uint32_t> lExpectedValues2;

    std::vector<uint32_t> lKDim2{ 2, 7, 42 };
    std::vector<uint32_t> lKernel2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<uint32_t>( 75, 0, 10000 );
            lValues2.insert( lValues2.end(), y.begin(), y.end() );

            auto lZ = RandomNumber<uint32_t>( 42, 0, 10000 );
            lKernel2.insert( lKernel2.end(), lZ.begin(), lZ.end() );

            auto lC = lConv1D( y, lZ );
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
            auto y = RandomNumber<uint32_t>( 23, 0, 10000 );
            lValues3.insert( lValues3.end(), y.begin(), y.end() );

            auto lZ = RandomNumber<uint32_t>( 5, 0, 10000 );
            lKernel3.insert( lKernel3.end(), lZ.begin(), lZ.end() );

            auto lC = lConv1D( y, lZ );
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

    data_initializer_t lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( scope, lInputInitializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( uint32_t ) ) );

    data_initializer_t lKernelInitializer( lKernelValues );
    auto               lKernelensor =
        MultiTensorValue( scope, lKernelInitializer, tensor_shape_t( { lKDim1, lKDim2, lKDim3 }, sizeof( uint32_t ) ) );

    auto result0 = Conv1D( scope, lInputTensor, lKernelensor );
    scope.Run( result0 );

    auto lOutputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.Rank == 3 );
    REQUIRE( lOutputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( lOutputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( lOutputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<uint32_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<uint32_t>>{ expectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<uint32_t> lResultValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint32_t>();
    REQUIRE( lResultValues == expectedValues );
}

TEST_CASE( "HCat (uint32_t)", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> dim1{ 7, 3, 124 };
    std::vector<uint32_t> lValues1;
    std::vector<uint32_t> expectedValues1;

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
            auto y = RandomNumber<uint32_t>( 124, 0, 10000 );
            lValues1.insert( lValues1.end(), y.begin(), y.end() );

            auto lZ = RandomNumber<uint32_t>( 34, 0, 10000 );
            lKernel1.insert( lKernel1.end(), lZ.begin(), lZ.end() );

            auto lC = lHCat( y, lZ );
            expectedValues1.insert( expectedValues1.end(), lC.begin(), lC.end() );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 75 };
    std::vector<uint32_t> lValues2;
    std::vector<uint32_t> lExpectedValues2;

    std::vector<uint32_t> lKDim2{ 2, 7, 42 };
    std::vector<uint32_t> lKernel2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<uint32_t>( 75, 0, 10000 );
            lValues2.insert( lValues2.end(), y.begin(), y.end() );

            auto lZ = RandomNumber<uint32_t>( 42, 0, 10000 );
            lKernel2.insert( lKernel2.end(), lZ.begin(), lZ.end() );

            auto lC = lHCat( y, lZ );
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
            auto y = RandomNumber<uint32_t>( 23, 0, 10000 );
            lValues3.insert( lValues3.end(), y.begin(), y.end() );

            auto lZ = RandomNumber<uint32_t>( 5, 0, 10000 );
            lKernel3.insert( lKernel3.end(), lZ.begin(), lZ.end() );

            auto lC = lHCat( y, lZ );
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

    data_initializer_t lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( scope, lInputInitializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( uint32_t ) ) );

    data_initializer_t lKernelInitializer( lKernelValues );
    auto               lKernelensor =
        MultiTensorValue( scope, lKernelInitializer, tensor_shape_t( { lKDim1, lKDim2, lKDim3 }, sizeof( uint32_t ) ) );

    auto result0 = HCat( scope, lInputTensor, lKernelensor );
    scope.Run( result0 );

    auto lOutputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.Rank == 3 );
    REQUIRE( lOutputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 124 + 34 } );
    REQUIRE( lOutputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 75 + 42 } );
    REQUIRE( lOutputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 23 + 5 } );

    std::vector<uint32_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<uint32_t>>{ expectedValues1, lExpectedValues2, lExpectedValues3 } );

    std::vector<uint32_t> lResultValues = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint32_t>();
    REQUIRE( lResultValues == expectedValues );
}

TEMPLATE_TEST_CASE( "Addition broadcast", "[CORE_COMPUTATION_GRAPH]", uint16_t, uint32_t, uint64_t, int16_t, int32_t, int64_t, float )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t>              dim1{ 7, 3, 124 };
    std::vector<std::vector<TestType>> lValues1 = RandomVector<TestType>( dim1, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              dim2{ 2, 7, 75 };
    std::vector<std::vector<TestType>> lValues2 = RandomVector<TestType>( dim2, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              lDim3{ 3, 5, 23 };
    std::vector<std::vector<TestType>> lValues3 = RandomVector<TestType>( lDim3, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim1( dim1.begin(), dim1.end() - 1 );
    std::vector<TestType> lKernel1 = RandomValues<TestType>( lKDim1, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim2( dim2.begin(), dim2.end() - 1 );
    std::vector<TestType> lKernel2 = RandomValues<TestType>( lKDim2, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim3( lDim3.begin(), lDim3.end() - 1 );
    std::vector<TestType> lKernel3 = RandomValues<TestType>( lKDim3, 0, std::numeric_limits<TestType>::max() / 2 );

    auto expectedValues1  = BroadcastMap<TestType>( lValues1, lKernel1, []( TestType x, TestType y ) { return x + y; } );
    auto lExpectedValues2 = BroadcastMap<TestType>( lValues2, lKernel2, []( TestType x, TestType y ) { return x + y; } );
    auto lExpectedValues3 = BroadcastMap<TestType>( lValues3, lKernel3, []( TestType x, TestType y ) { return x + y; } );

    std::vector<TestType> lInputValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lValues1 ), ConcatenateVectors( lValues2 ), ConcatenateVectors( lValues3 ) } );

    std::vector<TestType> lKernelValues = ConcatenateVectors( std::vector<std::vector<TestType>>{ lKernel1, lKernel2, lKernel3 } );

    data_initializer_t lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( scope, lInputInitializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( TestType ) ) );

    data_initializer_t lKernelInitializer( lKernelValues );
    auto               lKernelensor =
        MultiTensorValue( scope, lKernelInitializer, tensor_shape_t( { lKDim1, lKDim2, lKDim3 }, sizeof( TestType ) ) );

    auto result0 = Add( scope, lInputTensor, lKernelensor );
    auto result1 = Add( scope, lKernelensor, lInputTensor );
    scope.Run( { result0, result1 } );

    auto lOutputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.Rank == 3 );
    REQUIRE( lOutputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( lOutputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( lOutputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<TestType> expectedValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( expectedValues1 ), ConcatenateVectors( lExpectedValues2 ), ConcatenateVectors( lExpectedValues3 ) } );

    std::vector<TestType> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<TestType>();
    REQUIRE( lResultValues0 == expectedValues );

    std::vector<TestType> lResultValues1 = result1.Get<multi_tensor_value_t>().mValue.FetchFlattened<TestType>();
    REQUIRE( lResultValues1 == expectedValues );
}

TEMPLATE_TEST_CASE( "Multiplication broadcast", "[CORE_COMPUTATION_GRAPH]", uint16_t, uint32_t, uint64_t, int16_t, int32_t, int64_t,
                    float )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t>              dim1{ 7, 3, 124 };
    std::vector<std::vector<TestType>> lValues1 = RandomVector<TestType>( dim1, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              dim2{ 2, 7, 75 };
    std::vector<std::vector<TestType>> lValues2 = RandomVector<TestType>( dim2, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              lDim3{ 3, 5, 23 };
    std::vector<std::vector<TestType>> lValues3 = RandomVector<TestType>( lDim3, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim1( dim1.begin(), dim1.end() - 1 );
    std::vector<TestType> lKernel1 = RandomValues<TestType>( lKDim1, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim2( dim2.begin(), dim2.end() - 1 );
    std::vector<TestType> lKernel2 = RandomValues<TestType>( lKDim2, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim3( lDim3.begin(), lDim3.end() - 1 );
    std::vector<TestType> lKernel3 = RandomValues<TestType>( lKDim3, 0, std::numeric_limits<TestType>::max() / 2 );

    auto expectedValues1  = BroadcastMap<TestType>( lValues1, lKernel1, []( TestType x, TestType y ) { return x * y; } );
    auto lExpectedValues2 = BroadcastMap<TestType>( lValues2, lKernel2, []( TestType x, TestType y ) { return x * y; } );
    auto lExpectedValues3 = BroadcastMap<TestType>( lValues3, lKernel3, []( TestType x, TestType y ) { return x * y; } );

    std::vector<TestType> lInputValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lValues1 ), ConcatenateVectors( lValues2 ), ConcatenateVectors( lValues3 ) } );

    std::vector<TestType> lKernelValues = ConcatenateVectors( std::vector<std::vector<TestType>>{ lKernel1, lKernel2, lKernel3 } );

    data_initializer_t lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( scope, lInputInitializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( TestType ) ) );

    data_initializer_t lKernelInitializer( lKernelValues );
    auto               lKernelensor =
        MultiTensorValue( scope, lKernelInitializer, tensor_shape_t( { lKDim1, lKDim2, lKDim3 }, sizeof( TestType ) ) );

    auto result0 = Multiply( scope, lInputTensor, lKernelensor );
    auto result1 = Multiply( scope, lKernelensor, lInputTensor );
    scope.Run( { result0, result1 } );

    auto lOutputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.Rank == 3 );
    REQUIRE( lOutputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( lOutputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( lOutputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<TestType> expectedValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( expectedValues1 ), ConcatenateVectors( lExpectedValues2 ), ConcatenateVectors( lExpectedValues3 ) } );

    std::vector<TestType> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<TestType>();
    REQUIRE( lResultValues0 == expectedValues );

    std::vector<TestType> lResultValues1 = result1.Get<multi_tensor_value_t>().mValue.FetchFlattened<TestType>();
    REQUIRE( lResultValues1 == expectedValues );
}

TEST_CASE( "Divison broadcast", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t>           dim1{ 7, 3, 124 };
    std::vector<std::vector<float>> lValues1 = RandomVector<float>( dim1, 0, std::numeric_limits<float>::max() / 2 );

    std::vector<uint32_t>           dim2{ 2, 7, 75 };
    std::vector<std::vector<float>> lValues2 = RandomVector<float>( dim2, 0, std::numeric_limits<float>::max() / 2 );

    std::vector<uint32_t>           lDim3{ 3, 5, 23 };
    std::vector<std::vector<float>> lValues3 = RandomVector<float>( lDim3, 0, std::numeric_limits<float>::max() / 2 );

    std::vector<uint32_t> lKDim1( dim1.begin(), dim1.end() - 1 );
    std::vector<float>    lKernel1 = RandomValues<float>( lKDim1, 0.001, std::numeric_limits<float>::max() / 2 );

    std::vector<uint32_t> lKDim2( dim2.begin(), dim2.end() - 1 );
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

    data_initializer_t lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( scope, lInputInitializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( float ) ) );

    data_initializer_t lKernelInitializer( lKernelValues );
    auto lKernelensor = MultiTensorValue( scope, lKernelInitializer, tensor_shape_t( { lKDim1, lKDim2, lKDim3 }, sizeof( float ) ) );

    auto result0 = Divide( scope, lInputTensor, lKernelensor );
    auto result1 = Divide( scope, lKernelensor, lInputTensor );
    scope.Run( { result0, result1 } );

    auto lOutputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.Rank == 3 );
    REQUIRE( lOutputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( lOutputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( lOutputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<float> expectedValues0 = ConcatenateVectors( std::vector<std::vector<float>>{
        ConcatenateVectors( lExpectedValues01 ), ConcatenateVectors( lExpectedValues02 ), ConcatenateVectors( lExpectedValues03 ) } );
    std::vector<float> expectedValues1 = ConcatenateVectors( std::vector<std::vector<float>>{
        ConcatenateVectors( lExpectedValues11 ), ConcatenateVectors( lExpectedValues12 ), ConcatenateVectors( lExpectedValues13 ) } );

    std::vector<float> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    REQUIRE( lResultValues0 == expectedValues0 );

    std::vector<float> lResultValues1 = result1.Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    REQUIRE( lResultValues1 == expectedValues1 );
}

TEMPLATE_TEST_CASE( "Subtraction broadcast", "[CORE_COMPUTATION_GRAPH]", int16_t, int32_t, int64_t, float, double )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t>              dim1{ 7, 3, 124 };
    std::vector<std::vector<TestType>> lValues1 = RandomVector<TestType>( dim1, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              dim2{ 2, 7, 75 };
    std::vector<std::vector<TestType>> lValues2 = RandomVector<TestType>( dim2, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              lDim3{ 3, 5, 23 };
    std::vector<std::vector<TestType>> lValues3 = RandomVector<TestType>( lDim3, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim1( dim1.begin(), dim1.end() - 1 );
    std::vector<TestType> lKernel1 = RandomValues<TestType>( lKDim1, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim2( dim2.begin(), dim2.end() - 1 );
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

    data_initializer_t lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( scope, lInputInitializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( TestType ) ) );

    data_initializer_t lKernelInitializer( lKernelValues );
    auto               lKernelensor =
        MultiTensorValue( scope, lKernelInitializer, tensor_shape_t( { lKDim1, lKDim2, lKDim3 }, sizeof( TestType ) ) );

    auto result0 = Subtract( scope, lInputTensor, lKernelensor );
    auto result1 = Subtract( scope, lKernelensor, lInputTensor );
    scope.Run( { result0, result1 } );

    auto lOutputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.Rank == 3 );
    REQUIRE( lOutputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( lOutputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( lOutputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<TestType> expectedValues0 = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lExpectedValues01 ), ConcatenateVectors( lExpectedValues02 ), ConcatenateVectors( lExpectedValues03 ) } );
    std::vector<TestType> expectedValues1 = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lExpectedValues11 ), ConcatenateVectors( lExpectedValues12 ), ConcatenateVectors( lExpectedValues13 ) } );

    std::vector<TestType> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<TestType>();
    REQUIRE( lResultValues0 == expectedValues0 );

    std::vector<TestType> lResultValues1 = result1.Get<multi_tensor_value_t>().mValue.FetchFlattened<TestType>();
    REQUIRE( lResultValues1 == expectedValues1 );
}

TEST_CASE( "AND broadcast", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t>             dim1{ 7, 3, 124 };
    std::vector<std::vector<uint8_t>> lValues1 = RandomBooleanVector( dim1 );

    std::vector<uint32_t>             dim2{ 2, 7, 75 };
    std::vector<std::vector<uint8_t>> lValues2 = RandomBooleanVector( dim2 );

    std::vector<uint32_t>             lDim3{ 3, 5, 23 };
    std::vector<std::vector<uint8_t>> lValues3 = RandomBooleanVector( lDim3 );

    std::vector<uint32_t> lKDim1( dim1.begin(), dim1.end() - 1 );
    std::vector<uint8_t>  lKernel1 = RandomBooleanValues( lKDim1 );

    std::vector<uint32_t> lKDim2( dim2.begin(), dim2.end() - 1 );
    std::vector<uint8_t>  lKernel2 = RandomBooleanValues( lKDim2 );

    std::vector<uint32_t> lKDim3( lDim3.begin(), lDim3.end() - 1 );
    std::vector<uint8_t>  lKernel3 = RandomBooleanValues( lKDim3 );

    auto expectedValues1  = BroadcastMap<uint8_t>( lValues1, lKernel1, []( uint8_t x, uint8_t y ) { return x && y; } );
    auto lExpectedValues2 = BroadcastMap<uint8_t>( lValues2, lKernel2, []( uint8_t x, uint8_t y ) { return x && y; } );
    auto lExpectedValues3 = BroadcastMap<uint8_t>( lValues3, lKernel3, []( uint8_t x, uint8_t y ) { return x && y; } );

    std::vector<uint8_t> lInputValues = ConcatenateVectors( std::vector<std::vector<uint8_t>>{
        ConcatenateVectors( lValues1 ), ConcatenateVectors( lValues2 ), ConcatenateVectors( lValues3 ) } );

    std::vector<uint8_t> lKernelValues = ConcatenateVectors( std::vector<std::vector<uint8_t>>{ lKernel1, lKernel2, lKernel3 } );

    data_initializer_t lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( scope, lInputInitializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( uint8_t ) ) );

    data_initializer_t lKernelInitializer( lKernelValues );
    auto lKernelensor = MultiTensorValue( scope, lKernelInitializer, tensor_shape_t( { lKDim1, lKDim2, lKDim3 }, sizeof( uint8_t ) ) );

    auto result0 = And( scope, lInputTensor, lKernelensor );
    auto result1 = And( scope, lKernelensor, lInputTensor );
    scope.Run( { result0, result1 } );

    auto lOutputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.Rank == 3 );
    REQUIRE( lOutputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( lOutputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( lOutputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<uint8_t> expectedValues = ConcatenateVectors( std::vector<std::vector<uint8_t>>{
        ConcatenateVectors( expectedValues1 ), ConcatenateVectors( lExpectedValues2 ), ConcatenateVectors( lExpectedValues3 ) } );

    std::vector<uint8_t> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( lResultValues0 == expectedValues );

    std::vector<uint8_t> lResultValues1 = result1.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( lResultValues1 == expectedValues );
}

TEST_CASE( "OR broadcast", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t>             dim1{ 7, 3, 124 };
    std::vector<std::vector<uint8_t>> lValues1 = RandomBooleanVector( dim1 );

    std::vector<uint32_t>             dim2{ 2, 7, 75 };
    std::vector<std::vector<uint8_t>> lValues2 = RandomBooleanVector( dim2 );

    std::vector<uint32_t>             lDim3{ 3, 5, 23 };
    std::vector<std::vector<uint8_t>> lValues3 = RandomBooleanVector( lDim3 );

    std::vector<uint32_t> lKDim1( dim1.begin(), dim1.end() - 1 );
    std::vector<uint8_t>  lKernel1 = RandomBooleanValues( lKDim1 );

    std::vector<uint32_t> lKDim2( dim2.begin(), dim2.end() - 1 );
    std::vector<uint8_t>  lKernel2 = RandomBooleanValues( lKDim2 );

    std::vector<uint32_t> lKDim3( lDim3.begin(), lDim3.end() - 1 );
    std::vector<uint8_t>  lKernel3 = RandomBooleanValues( lKDim3 );

    auto expectedValues1  = BroadcastMap<uint8_t>( lValues1, lKernel1, []( uint8_t x, uint8_t y ) { return x || y; } );
    auto lExpectedValues2 = BroadcastMap<uint8_t>( lValues2, lKernel2, []( uint8_t x, uint8_t y ) { return x || y; } );
    auto lExpectedValues3 = BroadcastMap<uint8_t>( lValues3, lKernel3, []( uint8_t x, uint8_t y ) { return x || y; } );

    std::vector<uint8_t> lInputValues = ConcatenateVectors( std::vector<std::vector<uint8_t>>{
        ConcatenateVectors( lValues1 ), ConcatenateVectors( lValues2 ), ConcatenateVectors( lValues3 ) } );

    std::vector<uint8_t> lKernelValues = ConcatenateVectors( std::vector<std::vector<uint8_t>>{ lKernel1, lKernel2, lKernel3 } );

    data_initializer_t lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( scope, lInputInitializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( uint8_t ) ) );

    data_initializer_t lKernelInitializer( lKernelValues );
    auto lKernelensor = MultiTensorValue( scope, lKernelInitializer, tensor_shape_t( { lKDim1, lKDim2, lKDim3 }, sizeof( uint8_t ) ) );

    auto result0 = Or( scope, lInputTensor, lKernelensor );
    auto result1 = Or( scope, lKernelensor, lInputTensor );
    scope.Run( { result0, result1 } );

    auto lOutputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.Rank == 3 );
    REQUIRE( lOutputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( lOutputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( lOutputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<uint8_t> expectedValues = ConcatenateVectors( std::vector<std::vector<uint8_t>>{
        ConcatenateVectors( expectedValues1 ), ConcatenateVectors( lExpectedValues2 ), ConcatenateVectors( lExpectedValues3 ) } );

    std::vector<uint8_t> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( lResultValues0 == expectedValues );

    std::vector<uint8_t> lResultValues1 = result1.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( lResultValues1 == expectedValues );
}

TEMPLATE_TEST_CASE( "Bitwise AND broadcast", "[CORE_COMPUTATION_GRAPH]", uint16_t, uint32_t, uint64_t, int16_t, int32_t, int64_t )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t>              dim1{ 7, 3, 124 };
    std::vector<std::vector<TestType>> lValues1 = RandomVector<TestType>( dim1, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t>              dim2{ 2, 7, 75 };
    std::vector<std::vector<TestType>> lValues2 = RandomVector<TestType>( dim2, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t>              lDim3{ 3, 5, 23 };
    std::vector<std::vector<TestType>> lValues3 = RandomVector<TestType>( lDim3, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t> lKDim1( dim1.begin(), dim1.end() - 1 );
    std::vector<TestType> lKernel1 = RandomValues<TestType>( lKDim1, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t> lKDim2( dim2.begin(), dim2.end() - 1 );
    std::vector<TestType> lKernel2 = RandomValues<TestType>( lKDim2, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t> lKDim3( lDim3.begin(), lDim3.end() - 1 );
    std::vector<TestType> lKernel3 = RandomValues<TestType>( lKDim3, 0, std::numeric_limits<TestType>::max() );

    auto expectedValues1  = BroadcastMap<TestType>( lValues1, lKernel1, []( TestType x, TestType y ) { return x & y; } );
    auto lExpectedValues2 = BroadcastMap<TestType>( lValues2, lKernel2, []( TestType x, TestType y ) { return x & y; } );
    auto lExpectedValues3 = BroadcastMap<TestType>( lValues3, lKernel3, []( TestType x, TestType y ) { return x & y; } );

    std::vector<TestType> lInputValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lValues1 ), ConcatenateVectors( lValues2 ), ConcatenateVectors( lValues3 ) } );

    std::vector<TestType> lKernelValues = ConcatenateVectors( std::vector<std::vector<TestType>>{ lKernel1, lKernel2, lKernel3 } );

    data_initializer_t lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( scope, lInputInitializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( TestType ) ) );

    data_initializer_t lKernelInitializer( lKernelValues );
    auto               lKernelensor =
        MultiTensorValue( scope, lKernelInitializer, tensor_shape_t( { lKDim1, lKDim2, lKDim3 }, sizeof( TestType ) ) );

    auto result0 = BitwiseAnd( scope, lInputTensor, lKernelensor );
    auto result1 = BitwiseAnd( scope, lKernelensor, lInputTensor );
    scope.Run( { result0, result1 } );

    auto lOutputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.Rank == 3 );
    REQUIRE( lOutputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( lOutputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( lOutputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<TestType> expectedValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( expectedValues1 ), ConcatenateVectors( lExpectedValues2 ), ConcatenateVectors( lExpectedValues3 ) } );

    std::vector<TestType> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<TestType>();
    REQUIRE( lResultValues0 == expectedValues );

    std::vector<TestType> lResultValues1 = result1.Get<multi_tensor_value_t>().mValue.FetchFlattened<TestType>();
    REQUIRE( lResultValues1 == expectedValues );
}

TEMPLATE_TEST_CASE( "Bitwise OR broadcast", "[CORE_COMPUTATION_GRAPH]", uint16_t, uint32_t, uint64_t, int16_t, int32_t, int64_t )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t>              dim1{ 7, 3, 124 };
    std::vector<std::vector<TestType>> lValues1 = RandomVector<TestType>( dim1, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t>              dim2{ 2, 7, 75 };
    std::vector<std::vector<TestType>> lValues2 = RandomVector<TestType>( dim2, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t>              lDim3{ 3, 5, 23 };
    std::vector<std::vector<TestType>> lValues3 = RandomVector<TestType>( lDim3, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t> lKDim1( dim1.begin(), dim1.end() - 1 );
    std::vector<TestType> lKernel1 = RandomValues<TestType>( lKDim1, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t> lKDim2( dim2.begin(), dim2.end() - 1 );
    std::vector<TestType> lKernel2 = RandomValues<TestType>( lKDim2, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t> lKDim3( lDim3.begin(), lDim3.end() - 1 );
    std::vector<TestType> lKernel3 = RandomValues<TestType>( lKDim3, 0, std::numeric_limits<TestType>::max() );

    auto expectedValues1  = BroadcastMap<TestType>( lValues1, lKernel1, []( TestType x, TestType y ) { return x | y; } );
    auto lExpectedValues2 = BroadcastMap<TestType>( lValues2, lKernel2, []( TestType x, TestType y ) { return x | y; } );
    auto lExpectedValues3 = BroadcastMap<TestType>( lValues3, lKernel3, []( TestType x, TestType y ) { return x | y; } );

    std::vector<TestType> lInputValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lValues1 ), ConcatenateVectors( lValues2 ), ConcatenateVectors( lValues3 ) } );

    std::vector<TestType> lKernelValues = ConcatenateVectors( std::vector<std::vector<TestType>>{ lKernel1, lKernel2, lKernel3 } );

    data_initializer_t lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( scope, lInputInitializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( TestType ) ) );

    data_initializer_t lKernelInitializer( lKernelValues );
    auto               lKernelensor =
        MultiTensorValue( scope, lKernelInitializer, tensor_shape_t( { lKDim1, lKDim2, lKDim3 }, sizeof( TestType ) ) );

    auto result0 = BitwiseOr( scope, lInputTensor, lKernelensor );
    auto result1 = BitwiseOr( scope, lKernelensor, lInputTensor );
    scope.Run( { result0, result1 } );

    auto lOutputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.Rank == 3 );
    REQUIRE( lOutputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( lOutputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( lOutputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<TestType> expectedValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( expectedValues1 ), ConcatenateVectors( lExpectedValues2 ), ConcatenateVectors( lExpectedValues3 ) } );

    std::vector<TestType> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<TestType>();
    REQUIRE( lResultValues0 == expectedValues );

    std::vector<TestType> lResultValues1 = result1.Get<multi_tensor_value_t>().mValue.FetchFlattened<TestType>();
    REQUIRE( lResultValues1 == expectedValues );
}

TEMPLATE_TEST_CASE( "Equal broadcast", "[CORE_COMPUTATION_GRAPH]", uint16_t, uint32_t, uint64_t, int16_t, int32_t, int64_t, float,
                    double )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t>              dim1{ 7, 3, 124 };
    std::vector<std::vector<TestType>> lValues1 = RandomVector<TestType>( dim1, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              dim2{ 2, 7, 75 };
    std::vector<std::vector<TestType>> lValues2 = RandomVector<TestType>( dim2, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              lDim3{ 3, 5, 23 };
    std::vector<std::vector<TestType>> lValues3 = RandomVector<TestType>( lDim3, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim1( dim1.begin(), dim1.end() - 1 );
    std::vector<TestType> lKernel1 = RandomValues<TestType>( lKDim1, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim2( dim2.begin(), dim2.end() - 1 );
    std::vector<TestType> lKernel2 = RandomValues<TestType>( lKDim2, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim3( lDim3.begin(), lDim3.end() - 1 );
    std::vector<TestType> lKernel3 = RandomValues<TestType>( lKDim3, 0, std::numeric_limits<TestType>::max() / 2 );

    auto expectedValues1  = BroadcastMap<TestType>( lValues1, lKernel1, []( TestType x, TestType y ) { return x == y; } );
    auto lExpectedValues2 = BroadcastMap<TestType>( lValues2, lKernel2, []( TestType x, TestType y ) { return x == y; } );
    auto lExpectedValues3 = BroadcastMap<TestType>( lValues3, lKernel3, []( TestType x, TestType y ) { return x == y; } );

    std::vector<TestType> lInputValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lValues1 ), ConcatenateVectors( lValues2 ), ConcatenateVectors( lValues3 ) } );

    std::vector<TestType> lKernelValues = ConcatenateVectors( std::vector<std::vector<TestType>>{ lKernel1, lKernel2, lKernel3 } );

    data_initializer_t lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( scope, lInputInitializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( TestType ) ) );

    data_initializer_t lKernelInitializer( lKernelValues );
    auto               lKernelensor =
        MultiTensorValue( scope, lKernelInitializer, tensor_shape_t( { lKDim1, lKDim2, lKDim3 }, sizeof( TestType ) ) );

    auto result0 = Equal( scope, lInputTensor, lKernelensor );
    auto result1 = Equal( scope, lKernelensor, lInputTensor );
    scope.Run( { result0, result1 } );

    auto lOutputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.Rank == 3 );
    REQUIRE( lOutputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( lOutputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( lOutputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<TestType> expectedValues0 = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( expectedValues1 ), ConcatenateVectors( lExpectedValues2 ), ConcatenateVectors( lExpectedValues3 ) } );

    std::vector<uint8_t> expectedValues{};
    for( auto x : expectedValues0 )
        expectedValues.push_back( static_cast<uint8_t>( x != 0 ) );
    std::vector<uint8_t> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( lResultValues0 == expectedValues );

    std::vector<uint8_t> lResultValues1 = result1.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( lResultValues1 == expectedValues );
}

TEMPLATE_TEST_CASE( "LessThan broadcast", "[CORE_COMPUTATION_GRAPH]", uint16_t, uint32_t, uint64_t, int16_t, int32_t, int64_t, float,
                    double )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t>              dim1{ 7, 3, 124 };
    std::vector<std::vector<TestType>> lValues1 = RandomVector<TestType>( dim1, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              dim2{ 2, 7, 75 };
    std::vector<std::vector<TestType>> lValues2 = RandomVector<TestType>( dim2, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              lDim3{ 3, 5, 23 };
    std::vector<std::vector<TestType>> lValues3 = RandomVector<TestType>( lDim3, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim1( dim1.begin(), dim1.end() - 1 );
    std::vector<TestType> lKernel1 = RandomValues<TestType>( lKDim1, 0.001, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim2( dim2.begin(), dim2.end() - 1 );
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

    data_initializer_t lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( scope, lInputInitializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( TestType ) ) );

    data_initializer_t lKernelInitializer( lKernelValues );
    auto               lKernelensor =
        MultiTensorValue( scope, lKernelInitializer, tensor_shape_t( { lKDim1, lKDim2, lKDim3 }, sizeof( TestType ) ) );

    auto result0 = LessThan( scope, lInputTensor, lKernelensor );
    auto result1 = LessThan( scope, lKernelensor, lInputTensor );
    scope.Run( { result0, result1 } );

    auto lOutputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.Rank == 3 );
    REQUIRE( lOutputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( lOutputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( lOutputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<TestType> lExpectedValues00 = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lExpectedValues01 ), ConcatenateVectors( lExpectedValues02 ), ConcatenateVectors( lExpectedValues03 ) } );
    std::vector<uint8_t>  expectedValues0{};
    for( auto x : lExpectedValues00 )
        expectedValues0.push_back( static_cast<uint8_t>( x != 0 ) );

    std::vector<TestType> lExpectedValues10 = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lExpectedValues11 ), ConcatenateVectors( lExpectedValues12 ), ConcatenateVectors( lExpectedValues13 ) } );
    std::vector<uint8_t>  expectedValues1{};
    for( auto x : lExpectedValues10 )
        expectedValues1.push_back( static_cast<uint8_t>( x != 0 ) );

    std::vector<uint8_t> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( lResultValues0 == expectedValues0 );

    std::vector<uint8_t> lResultValues1 = result1.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( lResultValues1 == expectedValues1 );
}

TEMPLATE_TEST_CASE( "LessThanOrEqual broadcast", "[CORE_COMPUTATION_GRAPH]", uint16_t, uint32_t, uint64_t, int16_t, int32_t, int64_t,
                    float, double )
{

    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t>              dim1{ 7, 3, 124 };
    std::vector<std::vector<TestType>> lValues1 = RandomVector<TestType>( dim1, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              dim2{ 2, 7, 75 };
    std::vector<std::vector<TestType>> lValues2 = RandomVector<TestType>( dim2, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              lDim3{ 3, 5, 23 };
    std::vector<std::vector<TestType>> lValues3 = RandomVector<TestType>( lDim3, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim1( dim1.begin(), dim1.end() - 1 );
    std::vector<TestType> lKernel1 = RandomValues<TestType>( lKDim1, 0.001, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> lKDim2( dim2.begin(), dim2.end() - 1 );
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

    data_initializer_t lInputInitializer( lInputValues );
    auto lInputTensor = MultiTensorValue( scope, lInputInitializer, tensor_shape_t( { dim1, dim2, lDim3 }, sizeof( TestType ) ) );

    data_initializer_t lKernelInitializer( lKernelValues );
    auto               lKernelensor =
        MultiTensorValue( scope, lKernelInitializer, tensor_shape_t( { lKDim1, lKDim2, lKDim3 }, sizeof( TestType ) ) );

    auto result0 = LessThanOrEqual( scope, lInputTensor, lKernelensor );
    auto result1 = LessThanOrEqual( scope, lKernelensor, lInputTensor );
    scope.Run( { result0, result1 } );

    auto lOutputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( lOutputShape.CountLayers() == 3 );
    REQUIRE( lOutputShape.Rank == 3 );
    REQUIRE( lOutputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( lOutputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( lOutputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<TestType> lExpectedValues00 = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lExpectedValues01 ), ConcatenateVectors( lExpectedValues02 ), ConcatenateVectors( lExpectedValues03 ) } );
    std::vector<uint8_t>  expectedValues0{};
    for( auto x : lExpectedValues00 )
        expectedValues0.push_back( static_cast<uint8_t>( x != 0 ) );

    std::vector<TestType> lExpectedValues10 = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( lExpectedValues11 ), ConcatenateVectors( lExpectedValues12 ), ConcatenateVectors( lExpectedValues13 ) } );
    std::vector<uint8_t>  expectedValues1{};
    for( auto x : lExpectedValues10 )
        expectedValues1.push_back( static_cast<uint8_t>( x != 0 ) );

    std::vector<uint8_t> lResultValues0 = result0.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( lResultValues0 == expectedValues0 );

    std::vector<uint8_t> lResultValues1 = result1.Get<multi_tensor_value_t>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( lResultValues1 == expectedValues1 );
}
