#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "TestUtils.h"

#include "Core/Math/Types.h"

#include "Core/CUDA/Array/MemoryPool.h"
#include "Core/CUDA/Array/MultiTensor.h"

#include "TensorOps/NodeComponents.h"
#include "TensorOps/Scope.h"

using namespace numlua::core;
using namespace numlua::mtops;
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
std::vector<_Ty> Randovalues( std::vector<uint32_t> dim, _Ty min, _Ty max )
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

        REQUIRE( node.Get<vector_buffer_t>().value.SizeAs<uint32_t>() == value.size() );
    }

    SECTION( "Node initialization" )
    {
        std::vector<uint32_t> value = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

        auto node = VectorValue<uint32_t>( scope, value );
        scope.Run( node );

        auto buffer2 = node.Get<vector_buffer_t>().value.Fetch<uint32_t>();
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
        initializer.value = (uint8_t)3;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( int8_t ) ) );
        scope.Run( node );

        auto lBuffer1 = node.Get<multi_tensor_value_t>().value.BufferAt( 0 );
        auto buffer2  = node.Get<multi_tensor_value_t>().value.BufferAt( 1 );
        REQUIRE( lBuffer1.Size() == Prod( dim1 ) );
        REQUIRE( buffer2.Size() == Prod( dim2 ) );
    }

    SECTION( "Constant initializer (float)" )
    {
        constant_value_initializer_t initializer{};
        initializer.value = 3.0f;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        scope.Run( node );

        std::vector<float> expectedValues( node.Get<multi_tensor_value_t>().value.SizeAs<float>() );
        std::vector<float> tensorValues = node.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        for( auto &v : expectedValues )
        {
            v = 3.0f;
        }
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Constant initializer (double)" )
    {
        constant_value_initializer_t initializer{};
        initializer.value = (double)3.0f;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( double ) ) );
        scope.Run( node );

        std::vector<double> expectedValues( node.Get<multi_tensor_value_t>().value.SizeAs<double>() );
        std::vector<double> tensorValues = node.Get<multi_tensor_value_t>().value.FetchFlattened<double>();
        for( auto &v : expectedValues )
        {
            v = 3.0f;
        }
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Constant initializer (uint8_t)" )
    {
        constant_value_initializer_t initializer{};
        initializer.value = (uint8_t)3;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );
        scope.Run( node );

        std::vector<uint8_t> expectedValues( node.Get<multi_tensor_value_t>().value.SizeAs<uint8_t>() );
        std::vector<uint8_t> tensorValues = node.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
        for( auto &v : expectedValues )
        {
            v = 3;
        }
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Constant initializer (uint16_t)" )
    {
        constant_value_initializer_t initializer{};
        initializer.value = (uint16_t)256;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( uint16_t ) ) );
        scope.Run( node );

        std::vector<uint16_t> expectedValues( node.Get<multi_tensor_value_t>().value.SizeAs<uint16_t>() );
        std::vector<uint16_t> tensorValues = node.Get<multi_tensor_value_t>().value.FetchFlattened<uint16_t>();
        for( auto &v : expectedValues )
        {
            v = 256;
        }
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Constant initializer (uint32_t)" )
    {
        constant_value_initializer_t initializer{};
        initializer.value = (uint32_t)1000000;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( uint32_t ) ) );
        scope.Run( node );

        std::vector<uint32_t> expectedValues( node.Get<multi_tensor_value_t>().value.SizeAs<uint32_t>() );
        std::vector<uint32_t> tensorValues = node.Get<multi_tensor_value_t>().value.FetchFlattened<uint32_t>();
        for( auto &v : expectedValues )
        {
            v = 1000000;
        }
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Constant initializer (uint64_t)" )
    {
        constant_value_initializer_t initializer{};
        initializer.value = (uint64_t)10000000000;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( uint64_t ) ) );
        scope.Run( node );

        std::vector<uint64_t> expectedValues( node.Get<multi_tensor_value_t>().value.SizeAs<uint64_t>() );
        std::vector<uint64_t> tensorValues = node.Get<multi_tensor_value_t>().value.FetchFlattened<uint64_t>();
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

        std::vector<float> tensorValues = node.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
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
        std::vector<double> tensorValues = node.Get<multi_tensor_value_t>().value.FetchFlattened<double>();

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

        std::vector<uint8_t> tensorValues = node.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
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
        std::vector<uint16_t> tensorValues = node.Get<multi_tensor_value_t>().value.FetchFlattened<uint16_t>();
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
        std::vector<uint32_t> tensorValues = node.Get<multi_tensor_value_t>().value.FetchFlattened<uint32_t>();
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
        std::vector<uint64_t> tensorValues = node.Get<multi_tensor_value_t>().value.FetchFlattened<uint64_t>();
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

        std::vector<float> tensorValues = node.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
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

        std::vector<double> tensorValues = node.Get<multi_tensor_value_t>().value.FetchFlattened<double>();
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

        std::vector<uint8_t> tensorValues = node.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
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

        std::vector<uint16_t> tensorValues = node.Get<multi_tensor_value_t>().value.FetchFlattened<uint16_t>();
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

        std::vector<uint32_t> tensorValues = node.Get<multi_tensor_value_t>().value.FetchFlattened<uint32_t>();
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

        std::vector<uint64_t> tensorValues = node.Get<multi_tensor_value_t>().value.FetchFlattened<uint64_t>();
        REQUIRE( VectorEqual( expectedValues, tensorValues ) );
    }

    SECTION( "Random uniform initializer (float)" )
    {
        random_uniform_initializer_t initializer{};
        initializer.type = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        scope.Run( node );

        std::vector<float> expectedValues( node.Get<multi_tensor_value_t>().value.SizeAs<float>() );
        std::vector<float> tensorValues = node.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        for( auto &v : expectedValues )
        {
            v = 0.0f;
        }
        REQUIRE( tensorValues != expectedValues );
    }

    SECTION( "Random uniform initializer (double)" )
    {
        random_uniform_initializer_t initializer{};
        initializer.type = scalar_type_t::FLOAT64;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( double ) ) );
        scope.Run( node );

        std::vector<double> expectedValues( node.Get<multi_tensor_value_t>().value.SizeAs<double>() );
        std::vector<double> tensorValues = node.Get<multi_tensor_value_t>().value.FetchFlattened<double>();
        for( auto &v : expectedValues )
        {
            v = 0.0;
        }
        REQUIRE( tensorValues != expectedValues );
    }

    SECTION( "Random normal initializer (float)" )
    {
        random_normal_initializer_t initializer{};
        initializer.type = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        scope.Run( node );

        std::vector<float> expectedValues( node.Get<multi_tensor_value_t>().value.SizeAs<float>() );
        std::vector<float> tensorValues = node.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        for( auto &v : expectedValues )
        {
            v = 0.0f;
        }
        REQUIRE( tensorValues != expectedValues );
    }

    SECTION( "Random normal initializer (double)" )
    {
        random_normal_initializer_t initializer{};
        initializer.type  = scalar_type_t::FLOAT64;
        initializer.mu    = (double)0.0;
        initializer.sigma = (double)1.0;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto node = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( double ) ) );
        scope.Run( node );

        std::vector<double> expectedValues( node.Get<multi_tensor_value_t>().value.SizeAs<double>() );
        std::vector<double> tensorValues = node.Get<multi_tensor_value_t>().value.FetchFlattened<double>();
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
        initializer.type = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto opNode  = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto opSNode = ConstantScalarValue( scope, 1.234f );

        auto result0 = Add( scope, opNode, opSNode );
        auto result1 = Add( scope, opSNode, opNode );

        scope.Run( { result0, result1 } );

        std::vector<float> leftTensorValues = opNode.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        std::vector<float> expectedValues( opNode.Get<multi_tensor_value_t>().value.SizeAs<float>() );
        std::vector<float> tensorValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        std::vector<float> tensorValues1 = result1.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
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
        initializer.type = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto opNode  = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto opSNode = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto result0 = Add( scope, opNode, opSNode );

        scope.Run( result0 );

        std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        std::vector<float> rightTensorValues = opSNode.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        std::vector<float> expectedValues( opNode.Get<multi_tensor_value_t>().value.SizeAs<float>() );
        std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
        {
            expectedValues[i] = leftTensorValues[i] + rightTensorValues[i];
        }
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Add array to vector (float)" )
    {
        random_normal_initializer_t initializer{};
        initializer.type = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto                        opNode = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        std::vector<scalar_value_t> constants{ 3.123f, 4.345f };
        auto                        opSNode = VectorValue( scope, constants );
        auto                        result0 = Add( scope, opNode, opSNode );
        auto                        result1 = Add( scope, opSNode, opNode );
        scope.Run( { result0, result1 } );

        {
            std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
            float              rightTensorValues = std::get<float>( constants[0] );
            std::vector<float> expectedValues( leftTensorValues.size() );
            std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
            for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
            {
                expectedValues[i] = leftTensorValues[i] + rightTensorValues;
            }
            REQUIRE( VectorEqual( tensorValues, expectedValues ) );
        }

        {
            std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
            float              rightTensorValues = std::get<float>( constants[1] );
            std::vector<float> expectedValues( leftTensorValues.size() );
            std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
            for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
            {
                expectedValues[i] = leftTensorValues[i] + rightTensorValues;
            }
            REQUIRE( VectorEqual( tensorValues, expectedValues ) );
        }

        {
            std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
            float              rightTensorValues = std::get<float>( constants[0] );
            std::vector<float> expectedValues( leftTensorValues.size() );
            std::vector<float> tensorValues = result1.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
            for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
            {
                expectedValues[i] = leftTensorValues[i] + rightTensorValues;
            }
            REQUIRE( VectorEqual( tensorValues, expectedValues ) );
        }

        {
            std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
            float              rightTensorValues = std::get<float>( constants[1] );
            std::vector<float> expectedValues( leftTensorValues.size() );
            std::vector<float> tensorValues = result1.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
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
        initializer.type = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto opNode  = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto opSNode = ConstantScalarValue( scope, 1.234f );
        auto result0 = Multiply( scope, opNode, opSNode );
        auto result1 = Multiply( scope, opSNode, opNode );
        scope.Run( { result0, result1 } );

        std::vector<float> leftTensorValues = opNode.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        std::vector<float> expectedValues( opNode.Get<multi_tensor_value_t>().value.SizeAs<float>() );
        std::vector<float> tensorValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        std::vector<float> tensorValues1 = result1.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
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
        initializer.type = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto opNode  = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto opSNode = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto result0 = Multiply( scope, opNode, opSNode );
        scope.Run( result0 );

        std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        std::vector<float> rightTensorValues = opSNode.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        std::vector<float> expectedValues( opNode.Get<multi_tensor_value_t>().value.SizeAs<float>() );
        std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
        {
            expectedValues[i] = leftTensorValues[i] * rightTensorValues[i];
        }
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Multiply array by vector (float)" )
    {
        random_normal_initializer_t initializer{};
        initializer.type = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto                        opNode = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        std::vector<scalar_value_t> constants{ 3.123f, 4.345f };
        auto                        opSNode = VectorValue( scope, constants );
        auto                        result0 = Multiply( scope, opNode, opSNode );
        auto                        result1 = Multiply( scope, opSNode, opNode );
        scope.Run( { result0, result1 } );

        {
            std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
            float              rightTensorValues = std::get<float>( constants[0] );
            std::vector<float> expectedValues( leftTensorValues.size() );
            std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
            for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
            {
                expectedValues[i] = leftTensorValues[i] * rightTensorValues;
            }
            REQUIRE( VectorEqual( tensorValues, expectedValues ) );
        }

        {
            std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
            float              rightTensorValues = std::get<float>( constants[1] );
            std::vector<float> expectedValues( leftTensorValues.size() );
            std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
            for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
            {
                expectedValues[i] = leftTensorValues[i] * rightTensorValues;
            }
            REQUIRE( VectorEqual( tensorValues, expectedValues ) );
        }

        {
            std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
            float              rightTensorValues = std::get<float>( constants[0] );
            std::vector<float> expectedValues( leftTensorValues.size() );
            std::vector<float> tensorValues = result1.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
            for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
            {
                expectedValues[i] = leftTensorValues[i] * rightTensorValues;
            }
            REQUIRE( VectorEqual( tensorValues, expectedValues ) );
        }

        {
            std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
            float              rightTensorValues = std::get<float>( constants[1] );
            std::vector<float> expectedValues( leftTensorValues.size() );
            std::vector<float> tensorValues = result1.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
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
        initializer.type = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto opNode  = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto opSNode = ConstantScalarValue( scope, 1.234f );
        auto result0 = Subtract( scope, opNode, opSNode );
        auto result1 = Subtract( scope, opSNode, opNode );
        scope.Run( { result0, result1 } );

        std::vector<float> leftTensorValues = opNode.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        std::vector<float> expectedValues0( opNode.Get<multi_tensor_value_t>().value.SizeAs<float>() );
        std::vector<float> expectedValues1( opNode.Get<multi_tensor_value_t>().value.SizeAs<float>() );
        std::vector<float> tensorValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        std::vector<float> tensorValues1 = result1.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
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
        initializer.type = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto                        opNode = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        std::vector<scalar_value_t> constants{ 3.123f, 4.345f };
        auto                        opSNode = VectorValue( scope, constants );
        auto                        result0 = Subtract( scope, opNode, opSNode );
        auto                        result1 = Subtract( scope, opSNode, opNode );
        scope.Run( { result0, result1 } );

        {
            std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
            float              rightTensorValues = std::get<float>( constants[0] );
            std::vector<float> expectedValues( leftTensorValues.size() );
            std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
            for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
            {
                expectedValues[i] = leftTensorValues[i] - rightTensorValues;
            }
            REQUIRE( VectorEqual( tensorValues, expectedValues ) );
        }

        {
            std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
            float              rightTensorValues = std::get<float>( constants[1] );
            std::vector<float> expectedValues( leftTensorValues.size() );
            std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
            for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
            {
                expectedValues[i] = leftTensorValues[i] - rightTensorValues;
            }
            REQUIRE( VectorEqual( tensorValues, expectedValues ) );
        }

        {
            std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
            float              rightTensorValues = std::get<float>( constants[0] );
            std::vector<float> expectedValues( leftTensorValues.size() );
            std::vector<float> tensorValues = result1.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
            for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
            {
                expectedValues[i] = rightTensorValues - leftTensorValues[i];
            }
            REQUIRE( VectorEqual( tensorValues, expectedValues ) );
        }

        {
            std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
            float              rightTensorValues = std::get<float>( constants[1] );
            std::vector<float> expectedValues( leftTensorValues.size() );
            std::vector<float> tensorValues = result1.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
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
        initializer.type = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto opNode  = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto opSNode = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto result0 = Subtract( scope, opNode, opSNode );
        scope.Run( result0 );

        std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        std::vector<float> rightTensorValues = opSNode.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        std::vector<float> expectedValues( opNode.Get<multi_tensor_value_t>().value.SizeAs<float>() );
        std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
        {
            expectedValues[i] = leftTensorValues[i] - rightTensorValues[i];
        }
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Divide vector by array (float)" )
    {
        random_normal_initializer_t initializer{};
        initializer.type = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto                        opNode = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        std::vector<scalar_value_t> constants{ 3.123f, 4.345f };
        auto                        opSNode = VectorValue( scope, constants );
        auto                        result0 = Divide( scope, opNode, opSNode );
        auto                        result1 = Divide( scope, opSNode, opNode );
        scope.Run( { result0, result1 } );

        {
            std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
            float              rightTensorValues = std::get<float>( constants[0] );
            std::vector<float> expectedValues( leftTensorValues.size() );
            std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
            for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
            {
                expectedValues[i] = leftTensorValues[i] / rightTensorValues;
            }
            REQUIRE( VectorEqual( tensorValues, expectedValues ) );
        }

        {
            std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
            float              rightTensorValues = std::get<float>( constants[1] );
            std::vector<float> expectedValues( leftTensorValues.size() );
            std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
            for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
            {
                expectedValues[i] = leftTensorValues[i] / rightTensorValues;
            }
            REQUIRE( VectorEqual( tensorValues, expectedValues ) );
        }

        {
            std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
            float              rightTensorValues = std::get<float>( constants[0] );
            std::vector<float> expectedValues( leftTensorValues.size() );
            std::vector<float> tensorValues = result1.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
            for( uint32_t i = 0; i < leftTensorValues.size(); i++ )
            {
                expectedValues[i] = rightTensorValues / leftTensorValues[i];
            }
            REQUIRE( VectorEqual( tensorValues, expectedValues ) );
        }

        {
            std::vector<float> leftTensorValues  = opNode.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
            float              rightTensorValues = std::get<float>( constants[1] );
            std::vector<float> expectedValues( leftTensorValues.size() );
            std::vector<float> tensorValues = result1.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
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
    std::vector<std::vector<TestType>> values1 = RandomVector<TestType>( dim1, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t>              dim2{ 2, 7, 700 };
    std::vector<std::vector<TestType>> values2 = RandomVector<TestType>( dim2, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t>              dim3{ 3, 5, 200 };
    std::vector<std::vector<TestType>> values3 = RandomVector<TestType>( dim3, 0, std::numeric_limits<TestType>::max() );

    auto lScalarNode = ConstantScalarValue( scope, static_cast<TestType>( 13 ) );

    std::vector<TestType> expectedValues;
    auto                  expectedValues1 =
        BroadcastMap<TestType>( values1, static_cast<TestType>( 13 ), []( TestType x, TestType y ) { return x / y; } );
    auto expectedValues2 =
        BroadcastMap<TestType>( values2, static_cast<TestType>( 13 ), []( TestType x, TestType y ) { return x / y; } );
    auto expectedValues3 =
        BroadcastMap<TestType>( values3, static_cast<TestType>( 13 ), []( TestType x, TestType y ) { return x / y; } );

    expectedValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( expectedValues1 ), ConcatenateVectors( expectedValues2 ), ConcatenateVectors( expectedValues3 ) } );

    std::vector<TestType> inputValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( values1 ), ConcatenateVectors( values2 ), ConcatenateVectors( values3 ) } );

    data_initializer_t inputInitializer( inputValues );
    auto inputTensor = MultiTensorValue( scope, inputInitializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( TestType ) ) );

    auto result0 = Divide( scope, inputTensor, lScalarNode );
    scope.Run( result0 );

    auto outputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( outputShape.CountLayers() == 3 );
    REQUIRE( outputShape.Rank == 3 );
    REQUIRE( outputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 1400 } );
    REQUIRE( outputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 700 } );
    REQUIRE( outputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 200 } );

    std::vector<TestType> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<TestType>();
    REQUIRE( resultValues0 == expectedValues );
}

TEST_CASE( "Tensor AND Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> dim1{ 29, 12, 23 };
    std::vector<uint32_t> dim2{ 33, 14, 13 };

    std::vector<uint8_t> values0  = RandomBool( 29 * 12 * 23 );
    std::vector<uint8_t> values01 = RandomBool( 33 * 14 * 13 );
    values0.insert( values0.end(), values01.begin(), values01.end() );
    data_initializer_t initializer0( values0 );
    auto               opNodeLeft = MultiTensorValue( scope, initializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    std::vector<uint8_t> values1  = RandomBool( 29 * 12 * 23 );
    std::vector<uint8_t> values11 = RandomBool( 33 * 14 * 13 );
    values1.insert( values1.end(), values11.begin(), values11.end() );
    data_initializer_t initializer1( values1 );
    auto               opNodeRight = MultiTensorValue( scope, initializer1, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    auto result0 = And( scope, opNodeLeft, opNodeRight );
    auto result1 = And( scope, opNodeRight, opNodeLeft );
    scope.Run( { result0, result1 } );

    std::vector<uint8_t> expectedValues( values0.size() );
    for( uint32_t i = 0; i < values0.size(); i++ )
    {
        expectedValues[i] = ( values0[i] && values1[i] );
    }
    std::vector<uint8_t> tensorValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    REQUIRE( tensorValues0 == expectedValues );

    std::vector<uint8_t> tensorValues1 = result1.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    REQUIRE( tensorValues1 == expectedValues );
}

TEST_CASE( "Tensor AND Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> dim1{ 2, 12, 23 };
    std::vector<uint32_t> dim2{ 3, 14, 13 };

    std::vector<uint8_t> values00 = RandomBool( 2 * 12 * 23 );
    std::vector<uint8_t> values01 = RandomBool( 3 * 14 * 13 );
    std::vector<uint8_t> values0;
    values0.insert( values0.end(), values00.begin(), values00.end() );
    values0.insert( values0.end(), values01.begin(), values01.end() );
    data_initializer_t initializer0( values0 );
    auto               opNodeLeft = MultiTensorValue( scope, initializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    std::vector<scalar_value_t> constants{ static_cast<uint8_t>( 0 ), static_cast<uint8_t>( 1 ) };
    auto                        opNodeRight = VectorValue( scope, constants );

    auto result0 = And( scope, opNodeLeft, opNodeRight );
    auto result1 = And( scope, opNodeRight, opNodeLeft );
    scope.Run( { result0, result1 } );

    std::vector<uint8_t> expectedValues0( values00.size() );
    for( uint32_t i = 0; i < values00.size(); i++ )
    {
        expectedValues0[i] = ( values00[i] && std::get<uint8_t>( constants[0] ) );
    }

    std::vector<uint8_t> expectedValues1( values01.size() );
    for( uint32_t i = 0; i < values01.size(); i++ )
    {
        expectedValues1[i] = ( values01[i] && std::get<uint8_t>( constants[1] ) );
    }

    expectedValues0.insert( expectedValues0.end(), expectedValues1.begin(), expectedValues1.end() );

    std::vector<uint8_t> tensorValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    REQUIRE( tensorValues0 == expectedValues0 );

    std::vector<uint8_t> tensorValues1 = result1.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    REQUIRE( tensorValues1 == expectedValues0 );
}

TEST_CASE( "Tensor AND Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> dim1{ 2, 12, 23 };
    std::vector<uint32_t> dim2{ 3, 14, 13 };

    std::vector<uint8_t> values00 = RandomBool( 2 * 12 * 23 );
    std::vector<uint8_t> values01 = RandomBool( 3 * 14 * 13 );
    std::vector<uint8_t> values0;
    values0.insert( values0.end(), values00.begin(), values00.end() );
    values0.insert( values0.end(), values01.begin(), values01.end() );
    data_initializer_t initializer0( values0 );
    auto               opNodeLeft = MultiTensorValue( scope, initializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    {
        auto opNodeRight = ConstantScalarValue( scope, static_cast<uint8_t>( 0 ) );

        auto result0 = And( scope, opNodeLeft, opNodeRight );
        auto result1 = And( scope, opNodeRight, opNodeLeft );
        scope.Run( { result0, result1 } );

        std::vector<uint8_t> expectedValues( values0.size() );
        std::fill( expectedValues.begin(), expectedValues.end(), static_cast<uint8_t>( 0 ) );

        std::vector<uint8_t> tensorValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
        REQUIRE( tensorValues0 == expectedValues );

        std::vector<uint8_t> tensorValues1 = result1.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
        REQUIRE( tensorValues1 == expectedValues );
    }

    {
        auto opNodeRight = ConstantScalarValue( scope, static_cast<uint8_t>( 1 ) );

        auto result0 = And( scope, opNodeLeft, opNodeRight );
        auto result1 = And( scope, opNodeRight, opNodeLeft );
        scope.Run( { result0, result1 } );

        std::vector<uint8_t> expectedValues( values0.size() );

        for( uint32_t i = 0; i < values0.size(); i++ )
        {
            expectedValues[i] = values0[i];
        }

        std::vector<uint8_t> tensorValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
        REQUIRE( tensorValues0 == expectedValues );

        std::vector<uint8_t> tensorValues1 = result1.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
        REQUIRE( tensorValues1 == expectedValues );
    }
}

TEST_CASE( "Tensor OR Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> dim1{ 29, 12, 23 };
    std::vector<uint32_t> dim2{ 33, 14, 13 };

    std::vector<uint8_t> values0  = RandomBool( 29 * 12 * 23 );
    std::vector<uint8_t> values01 = RandomBool( 33 * 14 * 13 );
    values0.insert( values0.end(), values01.begin(), values01.end() );
    data_initializer_t initializer0( values0 );
    auto               opNodeLeft = MultiTensorValue( scope, initializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    std::vector<uint8_t> values1  = RandomBool( 29 * 12 * 23 );
    std::vector<uint8_t> values11 = RandomBool( 33 * 14 * 13 );
    values1.insert( values1.end(), values11.begin(), values11.end() );
    data_initializer_t initializer1( values1 );
    auto               opNodeRight = MultiTensorValue( scope, initializer1, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    auto result0 = Or( scope, opNodeLeft, opNodeRight );
    auto result1 = Or( scope, opNodeRight, opNodeLeft );
    scope.Run( { result0, result1 } );

    std::vector<uint8_t> expectedValues( values0.size() );
    for( uint32_t i = 0; i < values0.size(); i++ )
    {
        expectedValues[i] = ( values0[i] || values1[i] );
    }

    std::vector<uint8_t> tensorValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    REQUIRE( tensorValues0 == expectedValues );

    std::vector<uint8_t> tensorValues1 = result1.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    REQUIRE( tensorValues1 == expectedValues );
}

TEST_CASE( "Tensor OR Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> dim1{ 2, 12, 23 };
    std::vector<uint32_t> dim2{ 3, 14, 13 };

    std::vector<uint8_t> values00 = RandomBool( 2 * 12 * 23 );
    std::vector<uint8_t> values01 = RandomBool( 3 * 14 * 13 );
    std::vector<uint8_t> values0;
    values0.insert( values0.end(), values00.begin(), values00.end() );
    values0.insert( values0.end(), values01.begin(), values01.end() );
    data_initializer_t initializer0( values0 );
    auto               opNodeLeft = MultiTensorValue( scope, initializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    std::vector<scalar_value_t> constants{ static_cast<uint8_t>( 0 ), static_cast<uint8_t>( 1 ) };
    auto                        opNodeRight = VectorValue( scope, constants );

    auto result0 = Or( scope, opNodeLeft, opNodeRight );
    auto result1 = Or( scope, opNodeRight, opNodeLeft );
    scope.Run( { result0, result1 } );

    std::vector<uint8_t> expectedValues0( values00.size() );
    for( uint32_t i = 0; i < values00.size(); i++ )
    {
        expectedValues0[i] = ( values00[i] || std::get<uint8_t>( constants[0] ) );
    }

    std::vector<uint8_t> expectedValues1( values01.size() );
    for( uint32_t i = 0; i < values01.size(); i++ )
    {
        expectedValues1[i] = ( values01[i] || std::get<uint8_t>( constants[1] ) );
    }

    expectedValues0.insert( expectedValues0.end(), expectedValues1.begin(), expectedValues1.end() );

    std::vector<uint8_t> tensorValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    REQUIRE( tensorValues0 == expectedValues0 );

    std::vector<uint8_t> tensorValues1 = result1.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    REQUIRE( tensorValues1 == expectedValues0 );
}

TEST_CASE( "Tensor OR Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> dim1{ 2, 12, 23 };
    std::vector<uint32_t> dim2{ 3, 14, 13 };

    std::vector<uint8_t> values00 = RandomBool( 2 * 12 * 23 );
    std::vector<uint8_t> values01 = RandomBool( 3 * 14 * 13 );
    std::vector<uint8_t> values0;
    values0.insert( values0.end(), values00.begin(), values00.end() );
    values0.insert( values0.end(), values01.begin(), values01.end() );
    data_initializer_t initializer0( values0 );
    auto               opNodeLeft = MultiTensorValue( scope, initializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    {
        auto opNodeRight = ConstantScalarValue( scope, static_cast<uint8_t>( 1 ) );

        auto result0 = Or( scope, opNodeLeft, opNodeRight );
        auto result1 = Or( scope, opNodeRight, opNodeLeft );
        scope.Run( { result0, result1 } );

        std::vector<uint8_t> expectedValues( values0.size() );
        std::fill( expectedValues.begin(), expectedValues.end(), static_cast<uint8_t>( 1 ) );

        std::vector<uint8_t> tensorValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
        REQUIRE( tensorValues0 == expectedValues );

        std::vector<uint8_t> tensorValues1 = result1.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
        REQUIRE( tensorValues1 == expectedValues );
    }

    {
        auto opNodeRight = ConstantScalarValue( scope, static_cast<uint8_t>( 0 ) );

        auto result0 = Or( scope, opNodeLeft, opNodeRight );
        auto result1 = Or( scope, opNodeRight, opNodeLeft );
        scope.Run( { result0, result1 } );

        std::vector<uint8_t> expectedValues( values0.size() );

        for( uint32_t i = 0; i < values0.size(); i++ )
        {
            expectedValues[i] = values0[i];
        }

        std::vector<uint8_t> tensorValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
        REQUIRE( tensorValues0 == expectedValues );

        std::vector<uint8_t> tensorValues1 = result1.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
        REQUIRE( tensorValues1 == expectedValues );
    }
}

TEST_CASE( "NOT Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> dim1{ 29, 12, 23 };
    std::vector<uint32_t> dim2{ 33, 14, 13 };

    std::vector<uint8_t> values0  = RandomBool( 29 * 12 * 23 );
    std::vector<uint8_t> values01 = RandomBool( 33 * 14 * 13 );
    values0.insert( values0.end(), values01.begin(), values01.end() );
    data_initializer_t initializer0( values0 );
    auto               opNodeLeft = MultiTensorValue( scope, initializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    auto result0 = Not( scope, opNodeLeft );
    scope.Run( result0 );

    std::vector<uint8_t> tensorValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues( values0.size() );
    for( uint32_t i = 0; i < values0.size(); i++ )
    {
        expectedValues[i] = !( values0[i] );
    }
    REQUIRE( tensorValues == expectedValues );
}

TEST_CASE( "Tensor BITWISE_AND Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> dim1{ 29, 12, 23 };
    std::vector<uint32_t> dim2{ 33, 14, 13 };

    auto values0  = RandomNumber<uint64_t>( 29 * 12 * 23 );
    auto values01 = RandomNumber<uint64_t>( 33 * 14 * 13 );
    values0.insert( values0.end(), values01.begin(), values01.end() );
    data_initializer_t initializer0( values0 );
    auto               opNodeLeft = MultiTensorValue( scope, initializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint64_t ) ) );

    auto values1  = RandomNumber<uint64_t>( 29 * 12 * 23 );
    auto values11 = RandomNumber<uint64_t>( 33 * 14 * 13 );
    values1.insert( values1.end(), values11.begin(), values11.end() );
    data_initializer_t initializer1( values1 );
    auto               opNodeRight = MultiTensorValue( scope, initializer1, tensor_shape_t( { dim1, dim2 }, sizeof( uint64_t ) ) );

    auto result0 = BitwiseAnd( scope, opNodeLeft, opNodeRight );
    auto result1 = BitwiseAnd( scope, opNodeRight, opNodeLeft );
    scope.Run( { result0, result1 } );

    std::vector<uint64_t> expectedValues( values0.size() );
    for( uint32_t i = 0; i < values0.size(); i++ )
    {
        expectedValues[i] = ( values0[i] & values1[i] );
    }
    auto tensorValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint64_t>();
    REQUIRE( tensorValues0 == expectedValues );

    auto tensorValues1 = result1.Get<multi_tensor_value_t>().value.FetchFlattened<uint64_t>();
    REQUIRE( tensorValues1 == expectedValues );
}

TEST_CASE( "Tensor BITWISE_AND Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> dim1{ 2, 12, 23 };
    std::vector<uint32_t> dim2{ 3, 14, 13 };

    auto                  values00 = RandomNumber<uint64_t>( 2 * 12 * 23 );
    auto                  values01 = RandomNumber<uint64_t>( 3 * 14 * 13 );
    std::vector<uint64_t> values0;
    values0.insert( values0.end(), values00.begin(), values00.end() );
    values0.insert( values0.end(), values01.begin(), values01.end() );
    data_initializer_t initializer0( values0 );
    auto               opNodeLeft = MultiTensorValue( scope, initializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint64_t ) ) );

    std::vector<scalar_value_t> constants{ static_cast<uint64_t>( 0x1b34d765ef12acac ), static_cast<uint64_t>( 0x1b34d065ef120cfc ) };
    auto                        opNodeRight = VectorValue( scope, constants );

    auto result0 = BitwiseAnd( scope, opNodeLeft, opNodeRight );
    auto result1 = BitwiseAnd( scope, opNodeRight, opNodeLeft );
    scope.Run( { result0, result1 } );

    std::vector<uint64_t> expectedValues0( values00.size() );
    for( uint32_t i = 0; i < values00.size(); i++ )
    {
        expectedValues0[i] = ( values00[i] & std::get<uint64_t>( constants[0] ) );
    }

    std::vector<uint64_t> expectedValues1( values01.size() );
    for( uint32_t i = 0; i < values01.size(); i++ )
    {
        expectedValues1[i] = ( values01[i] & std::get<uint64_t>( constants[1] ) );
    }

    expectedValues0.insert( expectedValues0.end(), expectedValues1.begin(), expectedValues1.end() );

    auto tensorValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint64_t>();
    REQUIRE( tensorValues0 == expectedValues0 );

    auto tensorValues1 = result1.Get<multi_tensor_value_t>().value.FetchFlattened<uint64_t>();
    REQUIRE( tensorValues1 == expectedValues0 );
}

TEST_CASE( "Tensor BITWISE_AND Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> dim1{ 2, 12, 23 };
    std::vector<uint32_t> dim2{ 3, 14, 13 };

    auto                  values00 = RandomNumber<uint64_t>( 2 * 12 * 23 );
    auto                  values01 = RandomNumber<uint64_t>( 3 * 14 * 13 );
    std::vector<uint64_t> values0;
    values0.insert( values0.end(), values00.begin(), values00.end() );
    values0.insert( values0.end(), values01.begin(), values01.end() );
    data_initializer_t initializer0( values0 );
    auto               opNodeLeft = MultiTensorValue( scope, initializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint64_t ) ) );

    auto opNodeRight = ConstantScalarValue( scope, static_cast<uint64_t>( 0x1b34d765ef12acac ) );

    auto result0 = BitwiseAnd( scope, opNodeLeft, opNodeRight );
    auto result1 = BitwiseAnd( scope, opNodeRight, opNodeLeft );
    scope.Run( { result0, result1 } );

    std::vector<uint64_t> expectedValues( values0.size() );
    for( uint32_t i = 0; i < values0.size(); i++ )
    {
        expectedValues[i] = ( values0[i] & static_cast<uint64_t>( 0x1b34d765ef12acac ) );
    }

    auto tensorValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint64_t>();
    REQUIRE( tensorValues0 == expectedValues );

    auto tensorValues1 = result1.Get<multi_tensor_value_t>().value.FetchFlattened<uint64_t>();
    REQUIRE( tensorValues1 == expectedValues );
}

TEST_CASE( "Tensor BITWISE_OR Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> dim1{ 29, 12, 23 };
    std::vector<uint32_t> dim2{ 33, 14, 13 };

    auto values0  = RandomNumber<uint64_t>( 29 * 12 * 23 );
    auto values01 = RandomNumber<uint64_t>( 33 * 14 * 13 );
    values0.insert( values0.end(), values01.begin(), values01.end() );
    data_initializer_t initializer0( values0 );
    auto               opNodeLeft = MultiTensorValue( scope, initializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint64_t ) ) );

    auto values1  = RandomNumber<uint64_t>( 29 * 12 * 23 );
    auto values11 = RandomNumber<uint64_t>( 33 * 14 * 13 );
    values1.insert( values1.end(), values11.begin(), values11.end() );
    data_initializer_t initializer1( values1 );
    auto               opNodeRight = MultiTensorValue( scope, initializer1, tensor_shape_t( { dim1, dim2 }, sizeof( uint64_t ) ) );

    auto result0 = BitwiseOr( scope, opNodeLeft, opNodeRight );
    auto result1 = BitwiseOr( scope, opNodeRight, opNodeLeft );
    scope.Run( { result0, result1 } );

    std::vector<uint64_t> expectedValues( values0.size() );
    for( uint32_t i = 0; i < values0.size(); i++ )
    {
        expectedValues[i] = ( values0[i] | values1[i] );
    }

    auto tensorValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint64_t>();
    REQUIRE( tensorValues0 == expectedValues );

    auto tensorValues1 = result1.Get<multi_tensor_value_t>().value.FetchFlattened<uint64_t>();
    REQUIRE( tensorValues1 == expectedValues );
}

TEST_CASE( "Tensor BITWISE_OR Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> dim1{ 2, 12, 23 };
    std::vector<uint32_t> dim2{ 3, 14, 13 };

    auto                  values00 = RandomNumber<uint64_t>( 2 * 12 * 23 );
    auto                  values01 = RandomNumber<uint64_t>( 3 * 14 * 13 );
    std::vector<uint64_t> values0;
    values0.insert( values0.end(), values00.begin(), values00.end() );
    values0.insert( values0.end(), values01.begin(), values01.end() );
    data_initializer_t initializer0( values0 );
    auto               opNodeLeft = MultiTensorValue( scope, initializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint64_t ) ) );

    std::vector<scalar_value_t> constants{ static_cast<uint64_t>( 0x1b34d765ef12acac ), static_cast<uint64_t>( 0x1b34d065ef120cfc ) };
    auto                        opNodeRight = VectorValue( scope, constants );

    auto result0 = BitwiseOr( scope, opNodeLeft, opNodeRight );
    auto result1 = BitwiseOr( scope, opNodeRight, opNodeLeft );
    scope.Run( { result0, result1 } );

    std::vector<uint64_t> expectedValues0( values00.size() );
    for( uint32_t i = 0; i < values00.size(); i++ )
    {
        expectedValues0[i] = ( values00[i] | std::get<uint64_t>( constants[0] ) );
    }

    std::vector<uint64_t> expectedValues1( values01.size() );
    for( uint32_t i = 0; i < values01.size(); i++ )
    {
        expectedValues1[i] = ( values01[i] | std::get<uint64_t>( constants[1] ) );
    }

    expectedValues0.insert( expectedValues0.end(), expectedValues1.begin(), expectedValues1.end() );

    auto tensorValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint64_t>();
    REQUIRE( tensorValues0 == expectedValues0 );

    auto tensorValues1 = result1.Get<multi_tensor_value_t>().value.FetchFlattened<uint64_t>();
    REQUIRE( tensorValues1 == expectedValues0 );
}

TEST_CASE( "Tensor BITWISE_OR Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> dim1{ 2, 12, 23 };
    std::vector<uint32_t> dim2{ 3, 14, 13 };

    auto                  values00 = RandomNumber<uint64_t>( 2 * 12 * 23 );
    auto                  values01 = RandomNumber<uint64_t>( 3 * 14 * 13 );
    std::vector<uint64_t> values0;
    values0.insert( values0.end(), values00.begin(), values00.end() );
    values0.insert( values0.end(), values01.begin(), values01.end() );
    data_initializer_t initializer0( values0 );
    auto               opNodeLeft = MultiTensorValue( scope, initializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint64_t ) ) );

    auto opNodeRight = ConstantScalarValue( scope, static_cast<uint64_t>( 0x1b34d765ef12acac ) );

    auto result0 = BitwiseOr( scope, opNodeLeft, opNodeRight );
    auto result1 = BitwiseOr( scope, opNodeRight, opNodeLeft );
    scope.Run( { result0, result1 } );

    std::vector<uint64_t> expectedValues( values0.size() );
    for( uint32_t i = 0; i < values0.size(); i++ )
    {
        expectedValues[i] = ( values0[i] | static_cast<uint64_t>( 0x1b34d765ef12acac ) );
    }

    auto tensorValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint64_t>();
    REQUIRE( tensorValues0 == expectedValues );

    auto tensorValues1 = result1.Get<multi_tensor_value_t>().value.FetchFlattened<uint64_t>();
    REQUIRE( tensorValues1 == expectedValues );
}

TEST_CASE( "BITWISE_NOT Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> dim1{ 29, 12, 23 };
    std::vector<uint32_t> dim2{ 33, 14, 13 };

    auto values0  = RandomNumber<uint64_t>( 29 * 12 * 23 );
    auto values01 = RandomNumber<uint64_t>( 33 * 14 * 13 );
    values0.insert( values0.end(), values01.begin(), values01.end() );
    data_initializer_t initializer0( values0 );
    auto               opNodeLeft = MultiTensorValue( scope, initializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint64_t ) ) );

    auto result0 = BitwiseNot( scope, opNodeLeft );
    scope.Run( result0 );

    auto                  tensorValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint64_t>();
    std::vector<uint64_t> expectedValues( values0.size() );
    for( uint32_t i = 0; i < values0.size(); i++ )
    {
        expectedValues[i] = ~( values0[i] );
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
        initializer.type = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 8, 12 };
        std::vector<uint32_t> dim2{ 8, 16 };

        auto nodeA = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto nodeX = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto nodeB = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

        auto result0 = AffineTransform( scope, nodeA, nodeX, nodeB );
        scope.Run( result0 );

        std::vector<float> valuesA = nodeA.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        std::vector<float> valuesX = nodeX.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        std::vector<float> valuesB = nodeB.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        std::vector<float> expectedValues( nodeX.Get<multi_tensor_value_t>().value.SizeAs<float>() );
        for( uint32_t i = 0; i < valuesX.size(); i++ )
        {
            expectedValues[i] = valuesX[i] * valuesA[i] + valuesB[i];
        }
        std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<float>();

        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Affine transform tensor/tensor/vector (float)" )
    {
        random_uniform_initializer_t initializer{};
        initializer.type = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 8, 12 };
        std::vector<uint32_t> dim2{ 8, 16 };

        std::vector<float>          BValues{ 2.142983764918237649f, 3.234987659834765f };
        std::vector<scalar_value_t> B( 5 );
        B[0] = BValues[0];
        B[1] = BValues[1];

        auto nodeA = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto nodeX = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto nodeB = VectorValue( scope, B );

        auto result0 = AffineTransform( scope, nodeA, nodeX, nodeB );
        scope.Run( result0 );

        std::vector<float> expectedValues = {};
        for( uint32_t i = 0; i < BValues.size(); i++ )
        {
            uint32_t           size = nodeX.Get<multi_tensor_value_t>().Shape().GetBufferSizeAs<float>( i ).Size;
            std::vector<float> A    = nodeA.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( i );
            std::vector<float> X    = nodeX.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( i );
            std::vector<float> values( size );
            for( uint32_t j = 0; j < size; j++ )
            {
                values[j] = A[j] * X[j] + BValues[i];
            }
            expectedValues.insert( expectedValues.end(), values.begin(), values.end() );
        }

        std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Affine transform tensor/tensor/scalar (float)" )
    {
        random_uniform_initializer_t initializer{};
        initializer.type = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 8, 12 };
        std::vector<uint32_t> dim2{ 8, 16 };

        float scalarB = 2.142983764918237649f;
        auto  nodeA   = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto  nodeX   = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto  nodeB   = ConstantScalarValue( scope, scalarB );

        auto result0 = AffineTransform( scope, nodeA, nodeX, nodeB );
        scope.Run( result0 );

        std::vector<float> expectedValues = {};
        for( uint32_t i = 0; i < nodeX.Get<multi_tensor_value_t>().Shape().CountLayers(); i++ )
        {
            uint32_t           size = nodeX.Get<multi_tensor_value_t>().Shape().GetBufferSizeAs<float>( i ).Size;
            std::vector<float> A    = nodeA.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( i );
            std::vector<float> X    = nodeX.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( i );
            std::vector<float> values( size );
            for( uint32_t j = 0; j < size; j++ )
            {
                values[j] = A[j] * X[j] + scalarB;
            }
            expectedValues.insert( expectedValues.end(), values.begin(), values.end() );
        }

        std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Affine transform vector/tensor/tensor (float)" )
    {
        random_uniform_initializer_t initializer{};
        initializer.type = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 8, 22 };
        std::vector<uint32_t> dim2{ 8, 64 };

        std::vector<float> AValues{ 1.109238740928374f, 2.12398471982364f };
        std::vector<float> BValues{ 2.142983764918237649f, 3.234987659834765f };

        std::vector<scalar_value_t> A( 5 );
        A[0] = AValues[0];
        A[1] = AValues[1];

        std::vector<scalar_value_t> B( 5 );
        B[0] = BValues[0];
        B[1] = BValues[1];

        auto nodeA = VectorValue( scope, A );
        auto nodeB = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto nodeX = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

        auto result0 = AffineTransform( scope, nodeA, nodeX, nodeB );
        scope.Run( result0 );

        std::vector<float> expectedValues = {};
        for( uint32_t i = 0; i < AValues.size(); i++ )
        {
            uint32_t           size = nodeX.Get<multi_tensor_value_t>().Shape().GetBufferSizeAs<float>( i ).Size;
            std::vector<float> X    = nodeX.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( i );
            std::vector<float> B    = nodeB.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( i );
            std::vector<float> values( size );
            for( uint32_t j = 0; j < size; j++ )
            {
                values[j] = AValues[i] * X[j] + B[j];
            }
            expectedValues.insert( expectedValues.end(), values.begin(), values.end() );
        }

        std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Affine transform vector/tensor/vector (float)" )
    {
        random_uniform_initializer_t initializer{};
        initializer.type = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 8, 22 };
        std::vector<uint32_t> dim2{ 8, 64 };

        std::vector<float> AValues{ 1.109238740928374f, 2.12398471982364f };
        std::vector<float> BValues{ 2.142983764918237649f, 3.234987659834765f };

        std::vector<scalar_value_t> A( 5 );
        A[0] = AValues[0];
        A[1] = AValues[1];

        std::vector<scalar_value_t> B( 5 );
        B[0] = BValues[0];
        B[1] = BValues[1];

        auto nodeA = VectorValue( scope, A );
        auto nodeB = VectorValue( scope, B );
        auto nodeX = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

        auto result0 = AffineTransform( scope, nodeA, nodeX, nodeB );
        scope.Run( result0 );

        std::vector<float> expectedValues = {};
        for( uint32_t i = 0; i < AValues.size(); i++ )
        {
            uint32_t           size = nodeX.Get<multi_tensor_value_t>().Shape().GetBufferSizeAs<float>( i ).Size;
            std::vector<float> X    = nodeX.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( i );
            std::vector<float> values( size );
            for( uint32_t j = 0; j < size; j++ )
            {
                values[j] = AValues[i] * X[j] + BValues[i];
            }
            expectedValues.insert( expectedValues.end(), values.begin(), values.end() );
        }

        std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Affine transform vector/tensor/scalar (float)" )
    {
        random_uniform_initializer_t initializer{};
        initializer.type = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 8, 22 };
        std::vector<uint32_t> dim2{ 8, 64 };

        std::vector<float> AValues{ 1.109238740928374f, 2.12398471982364f };
        std::vector<float> BValues{ 2.142983764918237649f, 3.234987659834765f };

        std::vector<scalar_value_t> A( 5 );
        A[0] = AValues[0];
        A[1] = AValues[1];

        std::vector<scalar_value_t> B( 5 );
        B[0] = BValues[0];
        B[1] = BValues[1];

        float scalarB = 2.142983764918237649f;
        auto  nodeA   = VectorValue( scope, A );
        auto  nodeB   = ConstantScalarValue( scope, scalarB );
        auto  nodeX   = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

        auto result0 = AffineTransform( scope, nodeA, nodeX, nodeB );
        scope.Run( result0 );

        std::vector<float> expectedValues = {};
        for( uint32_t i = 0; i < AValues.size(); i++ )
        {
            uint32_t           size = nodeX.Get<multi_tensor_value_t>().Shape().GetBufferSizeAs<float>( i ).Size;
            std::vector<float> X    = nodeX.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( i );
            std::vector<float> values( size );
            for( uint32_t j = 0; j < size; j++ )
            {
                values[j] = AValues[i] * X[j] + scalarB;
            }
            expectedValues.insert( expectedValues.end(), values.begin(), values.end() );
        }

        std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Affine transform scalar/tensor/tensor (float)" )
    {
        random_uniform_initializer_t initializer{};
        initializer.type = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 8, 22 };
        std::vector<uint32_t> dim2{ 8, 64 };

        std::vector<float> AValues{ 1.109238740928374f, 2.12398471982364f };
        std::vector<float> BValues{ 2.142983764918237649f, 3.234987659834765f };

        std::vector<scalar_value_t> A( 5 );
        A[0] = AValues[0];
        A[1] = AValues[1];

        std::vector<scalar_value_t> B( 5 );
        B[0] = BValues[0];
        B[1] = BValues[1];

        float scalarA = 2.142983764918237649f;
        auto  nodeA   = ConstantScalarValue( scope, scalarA );
        auto  nodeB   = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto  nodeX   = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

        auto result0 = AffineTransform( scope, nodeA, nodeX, nodeB );
        scope.Run( result0 );

        std::vector<float> expectedValues = {};
        for( uint32_t i = 0; i < AValues.size(); i++ )
        {
            uint32_t           size = nodeX.Get<multi_tensor_value_t>().Shape().GetBufferSizeAs<float>( i ).Size;
            std::vector<float> X    = nodeX.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( i );
            std::vector<float> B    = nodeB.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( i );
            std::vector<float> values( size );
            for( uint32_t j = 0; j < size; j++ )
            {
                values[j] = scalarA * X[j] + B[j];
            }
            expectedValues.insert( expectedValues.end(), values.begin(), values.end() );
        }

        std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Affine transform scalar/tensor/vector (float)" )
    {
        random_uniform_initializer_t initializer{};
        initializer.type = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 8, 2 };
        std::vector<uint32_t> dim2{ 8, 6 };

        std::vector<float> AValues{ 1.109238740928374f, 2.12398471982364f };
        std::vector<float> BValues{ 2.142983764918237649f, 3.234987659834765f };

        std::vector<scalar_value_t> A( 5 );
        A[0] = AValues[0];
        A[1] = AValues[1];

        std::vector<scalar_value_t> B( 5 );
        B[0] = BValues[0];
        B[1] = BValues[1];

        float scalarA = 2.142983764918237649f;
        auto  nodeA   = ConstantScalarValue( scope, scalarA );
        auto  nodeB   = VectorValue( scope, B );
        auto  nodeX   = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

        auto result0 = AffineTransform( scope, nodeA, nodeX, nodeB );
        scope.Run( result0 );

        std::vector<float> expectedValues = {};
        for( uint32_t i = 0; i < AValues.size(); i++ )
        {
            uint32_t           size = nodeX.Get<multi_tensor_value_t>().Shape().GetBufferSizeAs<float>( i ).Size;
            std::vector<float> X    = nodeX.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( i );
            std::vector<float> values( size );
            for( uint32_t j = 0; j < size; j++ )
            {
                values[j] = scalarA * X[j] + BValues[i];
            }
            expectedValues.insert( expectedValues.end(), values.begin(), values.end() );
        }

        std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "Affine transform scalar/tensor/scalar (float)" )
    {
        random_uniform_initializer_t initializer{};
        initializer.type = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 8, 22 };
        std::vector<uint32_t> dim2{ 8, 64 };

        std::vector<float> AValues{ 1.109238740928374f, 2.12398471982364f };
        std::vector<float> BValues{ 2.142983764918237649f, 3.234987659834765f };

        std::vector<scalar_value_t> A( 5 );
        A[0] = AValues[0];
        A[1] = AValues[1];

        std::vector<scalar_value_t> B( 5 );
        B[0] = BValues[0];
        B[1] = BValues[1];

        float scalarA = 2.142983764918237649f;
        auto  nodeA   = ConstantScalarValue( scope, scalarA );
        float scalarB = 2.142983764918237649f;
        auto  nodeB   = ConstantScalarValue( scope, scalarB );
        auto  nodeX   = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

        auto result0 = AffineTransform( scope, nodeA, nodeX, nodeB );
        scope.Run( result0 );

        std::vector<float> expectedValues = {};
        for( uint32_t i = 0; i < AValues.size(); i++ )
        {
            uint32_t           size = nodeX.Get<multi_tensor_value_t>().Shape().GetBufferSizeAs<float>( i ).Size;
            std::vector<float> X    = nodeX.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( i );
            std::vector<float> values( size );
            for( uint32_t j = 0; j < size; j++ )
            {
                values[j] = scalarA * X[j] + scalarB;
            }
            expectedValues.insert( expectedValues.end(), values.begin(), values.end() );
        }

        std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
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
        initializer.type = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto nodeA = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto nodeB = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto nodeT = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

        auto result0 = Mix( scope, nodeA, nodeB, nodeT );
        scope.Run( result0 );

        std::vector<float> valuesA   = nodeA.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        std::vector<float> valuesB   = nodeB.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        std::vector<float> lValues_T = nodeT.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        std::vector<float> expectedValues( nodeA.Get<multi_tensor_value_t>().value.SizeAs<float>() );
        for( uint32_t i = 0; i < valuesA.size(); i++ )
        {
            expectedValues[i] = ( 1.0f - lValues_T[i] ) * valuesA[i] + lValues_T[i] * valuesB[i];
        }
        std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<float>();

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
        initializer.type = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto nodeA = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto nodeB = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

        std::vector<uint32_t> subdivisions{ 32, 64 };
        auto                  nodeS = VectorValue( scope, subdivisions );

        auto &result0 = LinearSpace( scope, nodeA, nodeB, nodeS );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Rank == 3 );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[0] == std::vector<uint32_t>{ 2, 2, 32 } );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[1] == std::vector<uint32_t>{ 3, 4, 64 } );
    }

    SECTION( "Linear space (float)" )
    {

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        constant_value_initializer_t initializer0{};
        initializer0.value = 0.5f;

        auto nodeA = MultiTensorValue( scope, initializer0, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

        constant_value_initializer_t initializer1{};
        initializer1.value = 1.5f;
        auto nodeB         = MultiTensorValue( scope, initializer1, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

        std::vector<uint32_t> subdivisions{ 32, 64 };
        auto                  nodeS = VectorValue( scope, subdivisions );

        auto &result0 = LinearSpace( scope, nodeA, nodeB, nodeS );
        scope.Run( result0 );

        {
            constexpr uint32_t subdivisions = 32;
            std::vector<float> valuesA      = nodeA.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
            std::vector<float> valuesB      = nodeB.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
            std::vector<float> expectedValues1( Prod( dim1 ) * subdivisions );
            uint32_t           x = 0;
            uint32_t           y = 0;
            for( uint32_t i = 0; i < dim1[0]; i++ )
            {
                for( uint32_t j = 0; j < dim1[1]; j++ )
                {
                    float Delta = ( valuesB[x] - valuesA[x] ) / static_cast<float>( subdivisions );
                    for( uint32_t k = 0; k < subdivisions; k++ )
                    {
                        expectedValues1[y] = valuesA[x] + static_cast<float>( k ) * Delta;
                        y++;
                    }
                    x++;
                }
            }
            std::vector<float> B1 = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
            REQUIRE( VectorEqual( B1, expectedValues1 ) );
        }

        {
            constexpr uint32_t subdivisions = 64;
            std::vector<float> valuesA      = nodeA.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
            std::vector<float> valuesB      = nodeB.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
            std::vector<float> expectedValues1( Prod( dim2 ) * subdivisions );
            uint32_t           x = 0;
            uint32_t           y = 0;
            for( uint32_t i = 0; i < dim2[0]; i++ )
            {
                for( uint32_t j = 0; j < dim2[1]; j++ )
                {
                    float Delta = ( valuesB[x] - valuesA[x] ) / static_cast<float>( subdivisions );
                    for( uint32_t k = 0; k < subdivisions; k++ )
                    {
                        expectedValues1[y] = valuesA[x] + static_cast<float>( k ) * Delta;
                        y++;
                    }
                    x++;
                }
            }
            std::vector<float> B1 = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
            REQUIRE( VectorEqual( B1, expectedValues1 ) );
        }
    }
}

TEST_CASE( "ARange node", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    SECTION( "ARange allocation (float)" )
    {
        std::vector<float> AValues{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        std::vector<float> BValues{ 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
        std::vector<float> DValues{ 0.01f, .02f, .03f, .04f, .05f };

        auto nodeA   = ScalarVectorValue( scope, scalar_type_t::FLOAT32, AValues );
        auto nodeB   = ScalarVectorValue( scope, scalar_type_t::FLOAT32, BValues );
        auto lNode_D = ScalarVectorValue( scope, scalar_type_t::FLOAT32, DValues );

        auto result0 = ARange( scope, nodeA, nodeB, lNode_D );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().CountLayers() == 5 );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Rank == 1 );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[0] ==
                 std::vector<uint32_t>{ static_cast<uint32_t>( std::ceil( ( BValues[0] - AValues[0] ) / DValues[0] ) ) } );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[1] ==
                 std::vector<uint32_t>{ static_cast<uint32_t>( std::ceil( ( BValues[1] - AValues[1] ) / DValues[1] ) ) } );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[2] ==
                 std::vector<uint32_t>{ static_cast<uint32_t>( std::ceil( ( BValues[2] - AValues[2] ) / DValues[2] ) ) } );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[3] ==
                 std::vector<uint32_t>{ static_cast<uint32_t>( std::ceil( ( BValues[3] - AValues[3] ) / DValues[3] ) ) } );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[4] ==
                 std::vector<uint32_t>{ static_cast<uint32_t>( std::ceil( ( BValues[4] - AValues[4] ) / DValues[4] ) ) } );
    }

    SECTION( "ARange (float)" )
    {
        std::vector<float> AValues{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        std::vector<float> BValues{ 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
        std::vector<float> DValues{ 0.01f, .02f, .03f, .04f, .05f };

        auto nodeA   = ScalarVectorValue( scope, scalar_type_t::FLOAT32, AValues );
        auto nodeB   = ScalarVectorValue( scope, scalar_type_t::FLOAT32, BValues );
        auto lNode_D = ScalarVectorValue( scope, scalar_type_t::FLOAT32, DValues );

        auto result0 = ARange( scope, nodeA, nodeB, lNode_D );

        scope.Run( result0 );

        std::vector<float> expectedValues = {};

        for( uint32_t i = 0; i < AValues.size(); i++ )
        {
            uint32_t           subdivisions = static_cast<uint32_t>( std::ceil( ( BValues[i] - AValues[i] ) / DValues[i] ) );
            std::vector<float> values( subdivisions );
            for( uint32_t j = 0; j < subdivisions; j++ )
            {
                values[j] = AValues[i] + j * DValues[i];
            }
            expectedValues.insert( expectedValues.end(), values.begin(), values.end() );
        }

        std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        REQUIRE( VectorEqual( tensorValues, expectedValues ) );
    }

    SECTION( "ARange (float)" )
    {
        std::vector<float> AValues{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        std::vector<float> BValues{ 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
        std::vector<float> DValues{ 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };

        auto nodeA   = ScalarVectorValue( scope, scalar_type_t::FLOAT32, AValues );
        auto nodeB   = ScalarVectorValue( scope, scalar_type_t::FLOAT32, BValues );
        auto lNode_D = ScalarVectorValue( scope, scalar_type_t::FLOAT32, DValues );

        auto result0 = ARange( scope, nodeA, nodeB, lNode_D );

        scope.Run( result0 );

        std::vector<float> expectedValues = {};

        for( uint32_t i = 0; i < AValues.size(); i++ )
        {
            uint32_t           subdivisions = static_cast<uint32_t>( std::ceil( ( BValues[i] - AValues[i] ) / DValues[i] ) );
            std::vector<float> values( subdivisions );
            for( uint32_t j = 0; j < subdivisions; j++ )
            {
                values[j] = AValues[i] + j * DValues[i];
            }
            expectedValues.insert( expectedValues.end(), values.begin(), values.end() );
        }

        std::vector<float> tensorValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
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
        initializer.type = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto nodeA = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

        std::vector<uint32_t> subdivisions{ 3, 5 };
        auto                  nodeS = VectorValue( scope, subdivisions );

        auto result0 = Repeat( scope, nodeA, nodeS );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Rank == 3 );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[0] == std::vector<uint32_t>{ 2, 2, 3 } );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[1] == std::vector<uint32_t>{ 3, 4, 5 } );
    }

    SECTION( "Repeat (float)" )
    {
        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        random_normal_initializer_t initializer0{};
        initializer0.type = scalar_type_t::FLOAT32;

        auto nodeA = MultiTensorValue( scope, initializer0, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

        std::vector<uint32_t> repetitions{ 3, 5 };
        auto                  nodeR = VectorValue( scope, repetitions );

        auto result0 = Repeat( scope, nodeA, nodeR );
        scope.Run( result0 );

        {
            constexpr uint32_t subdivisions = 3;
            std::vector<float> valuesA      = nodeA.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
            std::vector<float> expectedValues1( Prod( dim1 ) * subdivisions );
            uint32_t           x = 0;
            uint32_t           y = 0;
            for( uint32_t i = 0; i < dim1[0]; i++ )
            {
                for( uint32_t j = 0; j < dim1[1]; j++ )
                {
                    for( uint32_t k = 0; k < subdivisions; k++ )
                    {
                        expectedValues1[y] = valuesA[x];
                        y++;
                    }
                    x++;
                }
            }
            std::vector<float> B1 = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
            REQUIRE( VectorEqual( B1, expectedValues1 ) );
        }

        {
            constexpr uint32_t subdivisions = 5;
            std::vector<float> valuesA      = nodeA.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
            std::vector<float> expectedValues1( Prod( dim2 ) * subdivisions );
            uint32_t           x = 0;
            uint32_t           y = 0;
            for( uint32_t i = 0; i < dim2[0]; i++ )
            {
                for( uint32_t j = 0; j < dim2[1]; j++ )
                {
                    for( uint32_t k = 0; k < subdivisions; k++ )
                    {
                        expectedValues1[y] = valuesA[x];
                        y++;
                    }
                    x++;
                }
            }
            std::vector<float> B1 = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
            REQUIRE( VectorEqual( B1, expectedValues1 ) );
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
        initializer.type = scalar_type_t::FLOAT32;

        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        auto nodeA = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

        std::vector<uint32_t> subdivisions{ 3, 5 };
        auto                  nodeS = VectorValue( scope, subdivisions );

        auto result0 = Tile( scope, nodeA, nodeS );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Rank == 3 );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[0] == std::vector<uint32_t>{ 3, 2, 2 } );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[1] == std::vector<uint32_t>{ 5, 3, 4 } );
    }

    SECTION( "Tile (float)" )
    {
        std::vector<uint32_t> dim1{ 2, 2 };
        std::vector<uint32_t> dim2{ 3, 4 };

        random_normal_initializer_t initializer0{};
        initializer0.type = scalar_type_t::FLOAT32;

        auto nodeA = MultiTensorValue( scope, initializer0, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

        std::vector<uint32_t> repetitions{ 7, 11 };
        auto                  nodeR = VectorValue( scope, repetitions );

        auto result0 = Tile( scope, nodeA, nodeR );
        scope.Run( result0 );

        {
            constexpr uint32_t subdivisions = 7;
            std::vector<float> valuesA      = nodeA.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
            std::vector<float> expectedValues1( Prod( dim1 ) * subdivisions );
            uint32_t           x = 0;
            uint32_t           y = 0;
            for( uint32_t k = 0; k < subdivisions; k++ )
            {
                for( uint32_t i = 0; i < dim1[0] * dim1[1]; i++ )
                {
                    expectedValues1[y] = valuesA[i];
                    y++;
                }
            }
            std::vector<float> B1 = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
            REQUIRE( VectorEqual( B1, expectedValues1 ) );
        }

        {
            constexpr uint32_t subdivisions = 11;
            std::vector<float> valuesA      = nodeA.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
            std::vector<float> expectedValues1( Prod( dim2 ) * subdivisions );
            uint32_t           x = 0;
            uint32_t           y = 0;
            for( uint32_t k = 0; k < subdivisions; k++ )
            {
                for( uint32_t i = 0; i < dim2[0] * dim2[1]; i++ )
                {
                    expectedValues1[y] = valuesA[i];
                    y++;
                }
            }
            std::vector<float> B1 = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
            REQUIRE( VectorEqual( B1, expectedValues1 ) );
        }
    }
}

TEST_CASE( "Expand MultiTensors", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_uniform_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 5, 23, 42 };

    SECTION( "Expanding multi-tensors preserved types" )
    {
        auto nodeA =
            MultiTensorValue( scope, initializer, tensor_shape_t( std::vector<std::vector<uint32_t>>{ dim1 }, sizeof( float ) ) );
        auto result0 = Expand( scope, nodeA );

        REQUIRE( result0.Get<type_t>().value == nodeA.Get<type_t>().value );
    }

    SECTION( "Expanding multi-tensors gives the correct dimension" )
    {
        auto nodeA =
            MultiTensorValue( scope, initializer, tensor_shape_t( std::vector<std::vector<uint32_t>>{ dim1 }, sizeof( float ) ) );
        auto result0 = Expand( scope, nodeA );

        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Rank == ( dim1.size() - 1 ) );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().CountLayers() == dim1[0] );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[0] == std::vector<uint32_t>{ 23, 42 } );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[1] == std::vector<uint32_t>{ 23, 42 } );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[2] == std::vector<uint32_t>{ 23, 42 } );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[3] == std::vector<uint32_t>{ 23, 42 } );
    }

    SECTION( "Expanding multi-tensors does not change values" )
    {
        auto nodeA =
            MultiTensorValue( scope, initializer, tensor_shape_t( std::vector<std::vector<uint32_t>>{ dim1 }, sizeof( float ) ) );
        auto result0 = Expand( scope, nodeA );

        scope.Run( result0 );

        std::vector<float> tensorValues0 = nodeA.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        std::vector<float> tensorValues1 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        REQUIRE( VectorEqual( tensorValues0, tensorValues0 ) );
    }
}

TEST_CASE( "Collapse MultiTensors", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_uniform_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 23, 42 };
    std::vector<uint32_t> dim2{ 23, 42 };
    std::vector<uint32_t> dim3{ 23, 42 };
    std::vector<uint32_t> dim4{ 23, 42 };

    SECTION( "Collapsing multi-tensors preserved types" )
    {
        auto nodeA   = MultiTensorValue( scope, initializer,
                                         tensor_shape_t( std::vector<std::vector<uint32_t>>{ dim1, dim2 }, sizeof( float ) ) );
        auto result0 = Collapse( scope, nodeA );

        REQUIRE( result0.Get<type_t>().value == nodeA.Get<type_t>().value );
    }

    SECTION( "Collapsing multi-tensors gives the correct dimension" )
    {
        auto nodeA = MultiTensorValue(
            scope, initializer, tensor_shape_t( std::vector<std::vector<uint32_t>>{ dim1, dim2, dim3, dim4 }, sizeof( float ) ) );
        auto result0 = Collapse( scope, nodeA );

        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Rank == 3 );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().CountLayers() == 1 );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[0] == std::vector<uint32_t>{ 4, 23, 42 } );
    }

    SECTION( "Collapsing multi-tensors does not change values" )
    {
        auto nodeA = MultiTensorValue(
            scope, initializer, tensor_shape_t( std::vector<std::vector<uint32_t>>{ dim1, dim2, dim3, dim4 }, sizeof( float ) ) );
        auto result0 = Collapse( scope, nodeA );

        scope.Run( result0 );

        std::vector<float> tensorValues0 = nodeA.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        std::vector<float> tensorValues1 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        REQUIRE( VectorEqual( tensorValues0, tensorValues0 ) );
    }
}

TEST_CASE( "Reshape MultiTensors", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_uniform_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 21, 42 };
    std::vector<uint32_t> dim2{ 25, 40 };
    std::vector<uint32_t> dim3{ 14, 4 };

    SECTION( "Reshaping multi-tensors preserved types" )
    {
        auto nodeA = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( float ) ) );

        std::vector<uint32_t> oDim1{ 7, 3, 42 };
        std::vector<uint32_t> oDim2{ 25, 8, 5 };
        std::vector<uint32_t> oDim3{ 7, 2, 4 };
        auto                  result0 = Reshape( scope, nodeA, tensor_shape_t( { oDim1, oDim2, oDim3 }, sizeof( float ) ) );
        scope.Run( result0 );

        REQUIRE( result0.Get<type_t>().value == nodeA.Get<type_t>().value );
    }

    SECTION( "Reshaping multi-tensors gives the correct dimension" )
    {
        auto nodeA = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( float ) ) );

        std::vector<uint32_t> oDim1{ 7, 3, 42 };
        std::vector<uint32_t> oDim2{ 25, 8, 5 };
        std::vector<uint32_t> oDim3{ 7, 2, 4 };

        auto result0 = Reshape( scope, nodeA, tensor_shape_t( { oDim1, oDim2, oDim3 }, sizeof( float ) ) );
        scope.Run( result0 );

        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Rank == 3 );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().CountLayers() == 3 );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[0] == oDim1 );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[1] == oDim2 );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[2] == oDim3 );
    }

    SECTION( "Reshaping multi-tensors does not change values" )
    {
        auto nodeA = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( float ) ) );

        std::vector<uint32_t> oDim1{ 7, 3, 42 };
        std::vector<uint32_t> oDim2{ 25, 8, 5 };
        std::vector<uint32_t> oDim3{ 7, 2, 4 };

        auto result0 = Reshape( scope, nodeA, tensor_shape_t( { oDim1, oDim2, oDim3 }, sizeof( float ) ) );
        scope.Run( result0 );

        std::vector<float> tensorValues0 = nodeA.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        std::vector<float> tensorValues1 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        REQUIRE( VectorEqual( tensorValues0, tensorValues0 ) );
    }
}

TEST_CASE( "Flatten MultiTensors", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_uniform_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 21, 42 };
    std::vector<uint32_t> dim2{ 25, 40 };
    std::vector<uint32_t> dim3{ 14, 4 };

    SECTION( "Reshaping multi-tensors preserved types" )
    {
        auto nodeA   = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
        auto result0 = Flatten( scope, nodeA );

        REQUIRE( result0.Get<type_t>().value == nodeA.Get<type_t>().value );
    }

    SECTION( "Flattening multi-tensors gives the correct dimension" )
    {
        auto nodeA = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( float ) ) );

        std::vector<uint32_t> oDim1{ 7, 3, 42 };
        std::vector<uint32_t> oDim2{ 25, 8, 5 };
        std::vector<uint32_t> oDim3{ 7, 2, 4 };

        auto result0 = Flatten( scope, nodeA );
        scope.Run( result0 );

        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Rank == 1 );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().CountLayers() == 3 );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[0][0] == Prod( oDim1 ) );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[1][0] == Prod( oDim2 ) );
        REQUIRE( result0.Get<multi_tensor_value_t>().Shape().Shape[2][0] == Prod( oDim3 ) );
    }

    SECTION( "Flattening multi-tensors does not change values" )
    {
        auto nodeA = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( float ) ) );

        std::vector<uint32_t> oDim1{ 7, 3, 42 };
        std::vector<uint32_t> oDim2{ 25, 8, 5 };
        std::vector<uint32_t> oDim3{ 7, 2, 4 };

        auto result0 = Flatten( scope, nodeA );
        scope.Run( result0 );

        std::vector<float> tensorValues0 = nodeA.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
        std::vector<float> tensorValues1 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
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
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto opNode     = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto lowerBound = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto upperBound = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto result0    = InInterval( scope, opNode, lowerBound, upperBound, false, false );

    scope.Run( result0 );

    std::vector<float> XValues          = opNode.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    std::vector<float> lowerBoundValues = lowerBound.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    std::vector<float> upperBoundValues = upperBound.Get<multi_tensor_value_t>().value.FetchFlattened<float>();

    std::vector<uint8_t> resultValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues( opNode.Get<multi_tensor_value_t>().value.SizeAs<float>() );

    for( uint32_t i = 0; i < expectedValues.size(); i++ )
    {
        expectedValues[i] = ( lowerBoundValues[i] <= XValues[i] ) && ( XValues[i] <= upperBoundValues[i] );
    }
    REQUIRE( resultValues == expectedValues );
}

TEST_CASE( "InInterval Tensor_Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto opNode     = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto lowerBound = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    std::vector<scalar_value_t> constants{ 0.2345f, 0.345f };
    auto                        upperBound = VectorValue( scope, constants );
    auto                        result0    = InInterval( scope, opNode, lowerBound, upperBound, false, false );

    scope.Run( result0 );

    std::vector<float>   XValues0          = opNode.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
    std::vector<float>   lowerBoundValues0 = lowerBound.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
    std::vector<uint8_t> resultValues0     = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<uint8_t>( 0 );
    std::vector<uint8_t> expectedValues0( resultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( lowerBoundValues0[i] <= XValues0[i] ) && ( XValues0[i] <= std::get<float>( constants[0] ) );
    REQUIRE( resultValues0 == expectedValues0 );

    std::vector<float>   XValues1          = opNode.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
    std::vector<float>   lowerBoundValues1 = lowerBound.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
    std::vector<uint8_t> resultValues1     = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<uint8_t>( 1 );
    std::vector<uint8_t> expectedValues1( resultValues1.size() );
    for( uint32_t i = 0; i < expectedValues1.size(); i++ )
        expectedValues1[i] = ( lowerBoundValues1[i] <= XValues1[i] ) && ( XValues1[i] <= std::get<float>( constants[1] ) );
    REQUIRE( resultValues1 == expectedValues1 );
}

TEST_CASE( "InInterval Tensor_Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto opNode     = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto lowerBound = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto upperBound = ConstantScalarValue( scope, 0.245f );
    auto result0    = InInterval( scope, opNode, lowerBound, upperBound, false, false );

    scope.Run( result0 );

    std::vector<float>   XValues0          = opNode.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    std::vector<float>   lowerBoundValues0 = lowerBound.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    std::vector<uint8_t> resultValues0     = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues0( resultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( lowerBoundValues0[i] <= XValues0[i] ) && ( XValues0[i] <= 0.245f );
    REQUIRE( resultValues0 == expectedValues0 );
}

TEST_CASE( "InInterval Vector_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto opNode     = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto upperBound = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    std::vector<scalar_value_t> constants{ 0.2345f, 0.345f };
    auto                        lowerBound = VectorValue( scope, constants );
    auto                        result0    = InInterval( scope, opNode, lowerBound, upperBound, false, false );

    scope.Run( result0 );

    std::vector<float>   XValues0          = opNode.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
    std::vector<float>   upperBoundValues0 = upperBound.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
    std::vector<uint8_t> resultValues0     = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<uint8_t>( 0 );
    std::vector<uint8_t> expectedValues0( resultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( std::get<float>( constants[0] ) <= XValues0[i] ) && ( XValues0[i] <= upperBoundValues0[i] );
    REQUIRE( resultValues0 == expectedValues0 );

    std::vector<float>   XValues1          = opNode.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
    std::vector<float>   upperBoundValues1 = upperBound.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
    std::vector<uint8_t> resultValues1     = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<uint8_t>( 1 );
    std::vector<uint8_t> expectedValues1( resultValues1.size() );
    for( uint32_t i = 0; i < expectedValues1.size(); i++ )
        expectedValues1[i] = ( std::get<float>( constants[1] ) <= XValues1[i] ) && ( XValues1[i] <= upperBoundValues1[i] );
    REQUIRE( resultValues1 == expectedValues1 );
}

TEST_CASE( "InInterval Scalar_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto opNode     = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto upperBound = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto lowerBound = ConstantScalarValue( scope, 0.245f );
    auto result0    = InInterval( scope, opNode, lowerBound, upperBound, false, false );

    scope.Run( result0 );

    std::vector<float>   XValues0          = opNode.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    std::vector<float>   upperBoundValues0 = upperBound.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    std::vector<uint8_t> resultValues0     = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues0( resultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( 0.245f <= XValues0[i] ) && ( XValues0[i] <= upperBoundValues0[i] );
    REQUIRE( resultValues0 == expectedValues0 );
}

TEST_CASE( "InInterval Scalar_Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto opNode     = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto lowerBound = ConstantScalarValue( scope, 0.245f );
    auto upperBound = ConstantScalarValue( scope, 0.75f );

    auto result0 = InInterval( scope, opNode, lowerBound, upperBound, false, false );

    scope.Run( result0 );

    std::vector<float>   XValues0      = opNode.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    std::vector<uint8_t> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues0( resultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( 0.245f <= XValues0[i] ) && ( XValues0[i] <= 0.75f );
    REQUIRE( resultValues0 == expectedValues0 );
}

TEST_CASE( "LessThan Tensor_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto X = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto y = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto result0 = LessThan( scope, X, y );
    scope.Run( result0 );

    std::vector<float> XValues = X.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    std::vector<float> YValues = y.Get<multi_tensor_value_t>().value.FetchFlattened<float>();

    std::vector<uint8_t> resultValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues( resultValues.size() );

    for( uint32_t i = 0; i < expectedValues.size(); i++ )
    {
        expectedValues[i] = ( XValues[i] < YValues[i] );
    }
    REQUIRE( resultValues == expectedValues );
}

TEST_CASE( "LessThan Tensor_Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto X = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    std::vector<scalar_value_t> constants{ 0.2345f, 0.345f };
    auto                        y       = VectorValue( scope, constants );
    auto                        result0 = LessThan( scope, X, y );

    scope.Run( result0 );

    std::vector<float>   XValues0      = X.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
    std::vector<uint8_t> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<uint8_t>( 0 );
    std::vector<uint8_t> expectedValues0( resultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( XValues0[i] < std::get<float>( constants[0] ) );
    REQUIRE( resultValues0 == expectedValues0 );

    std::vector<float>   XValues1      = X.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
    std::vector<uint8_t> resultValues1 = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<uint8_t>( 1 );
    std::vector<uint8_t> expectedValues1( resultValues1.size() );
    for( uint32_t i = 0; i < expectedValues1.size(); i++ )
        expectedValues1[i] = ( XValues1[i] < std::get<float>( constants[1] ) );
    REQUIRE( resultValues1 == expectedValues1 );
}

TEST_CASE( "LessThan Tensor_Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto X = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto y = ConstantScalarValue( scope, 0.245f );

    auto result0 = LessThan( scope, X, y );

    scope.Run( result0 );

    std::vector<float>   XValues0      = X.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    std::vector<uint8_t> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues0( resultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( XValues0[i] < 0.245f );
    REQUIRE( resultValues0 == expectedValues0 );
}

TEST_CASE( "LessThan Vector_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    std::vector<scalar_value_t> constants{ 0.2345f, 0.345f };
    auto                        X = VectorValue( scope, constants );
    auto                        y = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto result0 = LessThan( scope, X, y );

    scope.Run( result0 );

    std::vector<float>   YValues0      = y.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
    std::vector<uint8_t> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<uint8_t>( 0 );
    std::vector<uint8_t> expectedValues0( resultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( std::get<float>( constants[0] ) < YValues0[i] );
    REQUIRE( resultValues0 == expectedValues0 );

    std::vector<float>   YValues1      = y.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
    std::vector<uint8_t> resultValues1 = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<uint8_t>( 1 );
    std::vector<uint8_t> expectedValues1( resultValues1.size() );
    for( uint32_t i = 0; i < expectedValues1.size(); i++ )
        expectedValues1[i] = ( std::get<float>( constants[1] ) < YValues1[i] );
    REQUIRE( resultValues1 == expectedValues1 );
}

TEST_CASE( "LessThan Scalar_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto X = ConstantScalarValue( scope, 0.245f );
    auto y = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto result0 = LessThan( scope, X, y );

    scope.Run( result0 );

    std::vector<float>   YValues0      = y.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    std::vector<uint8_t> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues0( resultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( 0.245f < YValues0[i] );
    REQUIRE( resultValues0 == expectedValues0 );
}

TEST_CASE( "LessThanOrEqual Tensor_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto X = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto y = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto result0 = LessThanOrEqual( scope, X, y );
    scope.Run( result0 );

    std::vector<float> XValues = X.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    std::vector<float> YValues = y.Get<multi_tensor_value_t>().value.FetchFlattened<float>();

    std::vector<uint8_t> resultValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues( resultValues.size() );

    for( uint32_t i = 0; i < expectedValues.size(); i++ )
    {
        expectedValues[i] = ( XValues[i] <= YValues[i] );
    }
    REQUIRE( resultValues == expectedValues );
}

TEST_CASE( "LessThanOrEqual Tensor_Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto X = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    std::vector<scalar_value_t> constants{ 0.2345f, 0.345f };
    auto                        y       = VectorValue( scope, constants );
    auto                        result0 = LessThanOrEqual( scope, X, y );

    scope.Run( result0 );

    std::vector<float>   XValues0      = X.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
    std::vector<uint8_t> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<uint8_t>( 0 );
    std::vector<uint8_t> expectedValues0( resultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( XValues0[i] <= std::get<float>( constants[0] ) );
    REQUIRE( resultValues0 == expectedValues0 );

    std::vector<float>   XValues1      = X.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
    std::vector<uint8_t> resultValues1 = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<uint8_t>( 1 );
    std::vector<uint8_t> expectedValues1( resultValues1.size() );
    for( uint32_t i = 0; i < expectedValues1.size(); i++ )
        expectedValues1[i] = ( XValues1[i] <= std::get<float>( constants[1] ) );
    REQUIRE( resultValues1 == expectedValues1 );
}

TEST_CASE( "LessThanOrEqual Tensor_Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto X = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto y = ConstantScalarValue( scope, 0.245f );

    auto result0 = LessThanOrEqual( scope, X, y );

    scope.Run( result0 );

    std::vector<float>   XValues0      = X.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    std::vector<uint8_t> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues0( resultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( XValues0[i] <= 0.245f );
    REQUIRE( resultValues0 == expectedValues0 );
}

TEST_CASE( "LessThanOrEqual Vector_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    std::vector<scalar_value_t> constants{ 0.2345f, 0.345f };
    auto                        X = VectorValue( scope, constants );
    auto                        y = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto result0 = LessThanOrEqual( scope, X, y );

    scope.Run( result0 );

    std::vector<float>   YValues0      = y.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
    std::vector<uint8_t> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<uint8_t>( 0 );
    std::vector<uint8_t> expectedValues0( resultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( std::get<float>( constants[0] ) <= YValues0[i] );
    REQUIRE( resultValues0 == expectedValues0 );

    std::vector<float>   YValues1      = y.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
    std::vector<uint8_t> resultValues1 = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<uint8_t>( 1 );
    std::vector<uint8_t> expectedValues1( resultValues1.size() );
    for( uint32_t i = 0; i < expectedValues1.size(); i++ )
        expectedValues1[i] = ( std::get<float>( constants[1] ) <= YValues1[i] );
    REQUIRE( resultValues1 == expectedValues1 );
}

TEST_CASE( "LessThanOrEqual Scalar_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto X = ConstantScalarValue( scope, 0.245f );
    auto y = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto result0 = LessThanOrEqual( scope, X, y );

    scope.Run( result0 );

    std::vector<float>   YValues0      = y.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    std::vector<uint8_t> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues0( resultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( 0.245f <= YValues0[i] );
    REQUIRE( resultValues0 == expectedValues0 );
}

TEST_CASE( "GreaterThan Tensor_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto X = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto y = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto result0 = GreaterThan( scope, X, y );
    scope.Run( result0 );

    std::vector<float> XValues = X.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    std::vector<float> YValues = y.Get<multi_tensor_value_t>().value.FetchFlattened<float>();

    std::vector<uint8_t> resultValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues( resultValues.size() );

    for( uint32_t i = 0; i < expectedValues.size(); i++ )
    {
        expectedValues[i] = ( XValues[i] > YValues[i] );
    }
    REQUIRE( resultValues == expectedValues );
}

TEST_CASE( "GreaterThan Tensor_Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto X = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    std::vector<scalar_value_t> constants{ 0.2345f, 0.345f };
    auto                        y       = VectorValue( scope, constants );
    auto                        result0 = GreaterThan( scope, X, y );

    scope.Run( result0 );

    std::vector<float>   XValues0      = X.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
    std::vector<uint8_t> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<uint8_t>( 0 );
    std::vector<uint8_t> expectedValues0( resultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( XValues0[i] > std::get<float>( constants[0] ) );
    REQUIRE( resultValues0 == expectedValues0 );

    std::vector<float>   XValues1      = X.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
    std::vector<uint8_t> resultValues1 = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<uint8_t>( 1 );
    std::vector<uint8_t> expectedValues1( resultValues1.size() );
    for( uint32_t i = 0; i < expectedValues1.size(); i++ )
        expectedValues1[i] = ( XValues1[i] > std::get<float>( constants[1] ) );
    REQUIRE( resultValues1 == expectedValues1 );
}

TEST_CASE( "GreaterThan Tensor_Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto X = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto y = ConstantScalarValue( scope, 0.245f );

    auto result0 = GreaterThan( scope, X, y );

    scope.Run( result0 );

    std::vector<float>   XValues0      = X.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    std::vector<uint8_t> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues0( resultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( XValues0[i] > 0.245f );
    REQUIRE( resultValues0 == expectedValues0 );
}

TEST_CASE( "GreaterThan Vector_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    std::vector<scalar_value_t> constants{ 0.2345f, 0.345f };
    auto                        X = VectorValue( scope, constants );
    auto                        y = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto result0 = GreaterThan( scope, X, y );

    scope.Run( result0 );

    std::vector<float>   YValues0      = y.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
    std::vector<uint8_t> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<uint8_t>( 0 );
    std::vector<uint8_t> expectedValues0( resultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( std::get<float>( constants[0] ) > YValues0[i] );
    REQUIRE( resultValues0 == expectedValues0 );

    std::vector<float>   YValues1      = y.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
    std::vector<uint8_t> resultValues1 = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<uint8_t>( 1 );
    std::vector<uint8_t> expectedValues1( resultValues1.size() );
    for( uint32_t i = 0; i < expectedValues1.size(); i++ )
        expectedValues1[i] = ( std::get<float>( constants[1] ) > YValues1[i] );
    REQUIRE( resultValues1 == expectedValues1 );
}

TEST_CASE( "GreaterThan Scalar_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto X = ConstantScalarValue( scope, 0.245f );
    auto y = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto result0 = GreaterThan( scope, X, y );

    scope.Run( result0 );

    std::vector<float>   YValues0      = y.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    std::vector<uint8_t> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues0( resultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( 0.245f > YValues0[i] );
    REQUIRE( resultValues0 == expectedValues0 );
}

TEST_CASE( "GreaterThanOrEqual Tensor_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto X = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto y = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto result0 = GreaterThanOrEqual( scope, X, y );
    scope.Run( result0 );

    std::vector<float> XValues = X.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    std::vector<float> YValues = y.Get<multi_tensor_value_t>().value.FetchFlattened<float>();

    std::vector<uint8_t> resultValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues( resultValues.size() );

    for( uint32_t i = 0; i < expectedValues.size(); i++ )
    {
        expectedValues[i] = ( XValues[i] >= YValues[i] );
    }
    REQUIRE( resultValues == expectedValues );
}

TEST_CASE( "GreaterThanOrEqual Tensor_Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto X = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    std::vector<scalar_value_t> constants{ 0.2345f, 0.345f };
    auto                        y       = VectorValue( scope, constants );
    auto                        result0 = GreaterThanOrEqual( scope, X, y );

    scope.Run( result0 );

    std::vector<float>   XValues0      = X.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
    std::vector<uint8_t> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<uint8_t>( 0 );
    std::vector<uint8_t> expectedValues0( resultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( XValues0[i] >= std::get<float>( constants[0] ) );
    REQUIRE( resultValues0 == expectedValues0 );

    std::vector<float>   XValues1      = X.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
    std::vector<uint8_t> resultValues1 = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<uint8_t>( 1 );
    std::vector<uint8_t> expectedValues1( resultValues1.size() );
    for( uint32_t i = 0; i < expectedValues1.size(); i++ )
        expectedValues1[i] = ( XValues1[i] >= std::get<float>( constants[1] ) );
    REQUIRE( resultValues1 == expectedValues1 );
}

TEST_CASE( "GreaterThanOrEqual Tensor_Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto X = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto y = ConstantScalarValue( scope, 0.245f );

    auto result0 = GreaterThanOrEqual( scope, X, y );

    scope.Run( result0 );

    std::vector<float>   XValues0      = X.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    std::vector<uint8_t> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues0( resultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( XValues0[i] >= 0.245f );
    REQUIRE( resultValues0 == expectedValues0 );
}

TEST_CASE( "GreaterThanOrEqual Vector_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    std::vector<scalar_value_t> constants{ 0.2345f, 0.345f };
    auto                        X = VectorValue( scope, constants );
    auto                        y = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto result0 = GreaterThanOrEqual( scope, X, y );

    scope.Run( result0 );

    std::vector<float>   YValues0      = y.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
    std::vector<uint8_t> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<uint8_t>( 0 );
    std::vector<uint8_t> expectedValues0( resultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( std::get<float>( constants[0] ) >= YValues0[i] );
    REQUIRE( resultValues0 == expectedValues0 );

    std::vector<float>   YValues1      = y.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
    std::vector<uint8_t> resultValues1 = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<uint8_t>( 1 );
    std::vector<uint8_t> expectedValues1( resultValues1.size() );
    for( uint32_t i = 0; i < expectedValues1.size(); i++ )
        expectedValues1[i] = ( std::get<float>( constants[1] ) >= YValues1[i] );
    REQUIRE( resultValues1 == expectedValues1 );
}

TEST_CASE( "GreaterThanOrEqual Scalar_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto X = ConstantScalarValue( scope, 0.245f );
    auto y = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto result0 = GreaterThanOrEqual( scope, X, y );

    scope.Run( result0 );

    std::vector<float>   YValues0      = y.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    std::vector<uint8_t> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    std::vector<uint8_t> expectedValues0( resultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = ( 0.245f >= YValues0[i] );
    REQUIRE( resultValues0 == expectedValues0 );
}

TEST_CASE( "Where Tensor_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto               values0 = RandomBool( 12 * 23 + 13 * 24 );
    data_initializer_t initializer0( values0 );
    auto               condition = MultiTensorValue( scope, initializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    auto X = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto y = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto result0 = Where( scope, condition, X, y );
    scope.Run( result0 );

    std::vector<float> XValues = X.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    std::vector<float> YValues = y.Get<multi_tensor_value_t>().value.FetchFlattened<float>();

    std::vector<float> resultValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    std::vector<float> expectedValues( resultValues.size() );

    for( uint32_t i = 0; i < expectedValues.size(); i++ )
    {
        expectedValues[i] = values0[i] ? XValues[i] : YValues[i];
    }
    REQUIRE( resultValues == expectedValues );
}

TEST_CASE( "Where Tensor_Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto                 values00 = RandomBool( 12 * 23 );
    auto                 values01 = RandomBool( 13 * 24 );
    std::vector<uint8_t> values0;
    values0.insert( values0.end(), values00.begin(), values00.end() );
    values0.insert( values0.end(), values01.begin(), values01.end() );
    data_initializer_t initializer0( values0 );
    auto               condition = MultiTensorValue( scope, initializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    auto X = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    std::vector<scalar_value_t> constants{ 0.27986534f, 0.31490728f };
    auto                        y = VectorValue( scope, constants );

    auto result0 = Where( scope, condition, X, y );

    scope.Run( result0 );

    std::vector<float> XValues0      = X.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
    std::vector<float> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
    std::vector<float> expectedValues0( resultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = values00[i] ? XValues0[i] : std::get<float>( constants[0] );
    REQUIRE( resultValues0 == expectedValues0 );

    std::vector<float> XValues1      = X.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
    std::vector<float> resultValues1 = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
    std::vector<float> expectedValues1( resultValues1.size() );
    for( uint32_t i = 0; i < expectedValues1.size(); i++ )
        expectedValues1[i] = values01[i] ? XValues1[i] : std::get<float>( constants[1] );
    REQUIRE( resultValues1 == expectedValues1 );
}

TEST_CASE( "Where Tensor_Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto               values0 = RandomBool( 12 * 23 + 13 * 24 );
    data_initializer_t initializer0( values0 );
    auto               condition = MultiTensorValue( scope, initializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    auto X = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );
    auto y = ConstantScalarValue( scope, 0.245f );

    auto result0 = Where( scope, condition, X, y );

    scope.Run( result0 );

    std::vector<float> XValues0      = X.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    std::vector<float> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    std::vector<float> expectedValues0( resultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = values0[i] ? XValues0[i] : 0.245f;
    REQUIRE( resultValues0 == expectedValues0 );
}

TEST_CASE( "Where Vector_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto values00 = RandomBool( 12 * 23 );
    auto values01 = RandomBool( 13 * 24 );

    std::vector<uint8_t> values0;
    values0.insert( values0.end(), values00.begin(), values00.end() );
    values0.insert( values0.end(), values01.begin(), values01.end() );
    data_initializer_t initializer0( values0 );

    auto condition = MultiTensorValue( scope, initializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    std::vector<scalar_value_t> constants{ 0.2345f, 0.345f };
    auto                        X = VectorValue( scope, constants );

    auto y = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto result0 = Where( scope, condition, X, y );

    scope.Run( result0 );

    std::vector<float> XValues0      = y.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
    std::vector<float> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
    std::vector<float> expectedValues0( resultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = values00[i] ? std::get<float>( constants[0] ) : XValues0[i];
    REQUIRE( resultValues0 == expectedValues0 );

    std::vector<float> XValues1      = y.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
    std::vector<float> resultValues1 = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
    std::vector<float> expectedValues1( resultValues1.size() );
    for( uint32_t i = 0; i < expectedValues1.size(); i++ )
        expectedValues1[i] = values01[i] ? std::get<float>( constants[1] ) : XValues1[i];
    REQUIRE( resultValues1 == expectedValues1 );
}

TEST_CASE( "Where Vector_Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto                 values00 = RandomBool( 12 * 23 );
    auto                 values01 = RandomBool( 13 * 24 );
    std::vector<uint8_t> values0;
    values0.insert( values0.end(), values00.begin(), values00.end() );
    values0.insert( values0.end(), values01.begin(), values01.end() );
    data_initializer_t initializer0( values0 );

    auto condition = MultiTensorValue( scope, initializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    std::vector<scalar_value_t> constants0{ 0.2345f, 0.345f };
    auto                        X = VectorValue( scope, constants0 );

    std::vector<scalar_value_t> constants1{ 0.26534f, 0.19048265f };
    auto                        y = VectorValue( scope, constants1 );

    auto result0 = Where( scope, condition, X, y );

    scope.Run( result0 );

    std::vector<float> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
    std::vector<float> expectedValues0( resultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = values00[i] ? std::get<float>( constants0[0] ) : std::get<float>( constants1[0] );
    REQUIRE( resultValues0 == expectedValues0 );

    std::vector<float> resultValues1 = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
    std::vector<float> expectedValues1( resultValues1.size() );
    for( uint32_t i = 0; i < expectedValues1.size(); i++ )
        expectedValues1[i] = values01[i] ? std::get<float>( constants0[1] ) : std::get<float>( constants1[1] );
    REQUIRE( resultValues1 == expectedValues1 );
}

TEST_CASE( "Where Vector_Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto                 values00 = RandomBool( 12 * 23 );
    auto                 values01 = RandomBool( 13 * 24 );
    std::vector<uint8_t> values0;
    values0.insert( values0.end(), values00.begin(), values00.end() );
    values0.insert( values0.end(), values01.begin(), values01.end() );
    data_initializer_t initializer0( values0 );
    auto               condition = MultiTensorValue( scope, initializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    std::vector<scalar_value_t> constants{ 0.2345f, 0.345f };
    auto                        X = VectorValue( scope, constants );
    auto                        y = ConstantScalarValue( scope, 0.245f );

    auto result0 = Where( scope, condition, X, y );

    scope.Run( result0 );

    std::vector<float> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
    std::vector<float> expectedValues0( resultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = values00[i] ? std::get<float>( constants[0] ) : 0.245f;
    REQUIRE( resultValues0 == expectedValues0 );

    std::vector<float> resultValues1 = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
    std::vector<float> expectedValues1( resultValues1.size() );
    for( uint32_t i = 0; i < expectedValues1.size(); i++ )
        expectedValues1[i] = values01[i] ? std::get<float>( constants[1] ) : 0.245f;
    REQUIRE( resultValues1 == expectedValues1 );
}

TEST_CASE( "Where Scalar_Tensor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto               values0 = RandomBool( 12 * 23 + 13 * 24 );
    data_initializer_t initializer0( values0 );
    auto               condition = MultiTensorValue( scope, initializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    auto X = ConstantScalarValue( scope, 0.245f );
    auto y = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2 }, sizeof( float ) ) );

    auto result0 = Where( scope, condition, X, y );

    scope.Run( result0 );

    std::vector<float> YValues0      = y.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    std::vector<float> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    std::vector<float> expectedValues0( resultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = values0[i] ? 0.245f : YValues0[i];
    REQUIRE( resultValues0 == expectedValues0 );
}

TEST_CASE( "Where Scalar_Vector", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto                 values00 = RandomBool( 12 * 23 );
    auto                 values01 = RandomBool( 13 * 24 );
    std::vector<uint8_t> values0;
    values0.insert( values0.end(), values00.begin(), values00.end() );
    values0.insert( values0.end(), values01.begin(), values01.end() );
    data_initializer_t initializer0( values0 );
    auto               condition = MultiTensorValue( scope, initializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    auto X = ConstantScalarValue( scope, 0.245f );

    std::vector<scalar_value_t> constants{ 0.2345f, 0.345f };
    auto                        y = VectorValue( scope, constants );

    auto result0 = Where( scope, condition, X, y );

    scope.Run( result0 );

    std::vector<float> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 0 );
    std::vector<float> expectedValues0( resultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = values00[i] ? 0.245f : std::get<float>( constants[0] );
    REQUIRE( resultValues0 == expectedValues0 );

    std::vector<float> resultValues1 = result0.Get<multi_tensor_value_t>().value.FetchBufferAt<float>( 1 );
    std::vector<float> expectedValues1( resultValues1.size() );
    for( uint32_t i = 0; i < expectedValues1.size(); i++ )
        expectedValues1[i] = values01[i] ? 0.245f : std::get<float>( constants[1] );
    REQUIRE( resultValues1 == expectedValues1 );
}

TEST_CASE( "Where Scalar_Scalar", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 128 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 12, 23 };
    std::vector<uint32_t> dim2{ 13, 24 };

    auto               values0 = RandomBool( 12 * 23 + 13 * 24 );
    data_initializer_t initializer0( values0 );
    auto               condition = MultiTensorValue( scope, initializer0, tensor_shape_t( { dim1, dim2 }, sizeof( uint8_t ) ) );

    auto X = ConstantScalarValue( scope, 0.9234587f );
    auto y = ConstantScalarValue( scope, 0.1324978f );

    auto result0 = Where( scope, condition, X, y );

    scope.Run( result0 );

    std::vector<float> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    std::vector<float> expectedValues0( resultValues0.size() );
    for( uint32_t i = 0; i < expectedValues0.size(); i++ )
        expectedValues0[i] = values0[i] ? 0.9234587f : 0.1324978f;
    REQUIRE( resultValues0 == expectedValues0 );
}

TEST_CASE( "ArraySlice VECTOR_VECTOR", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    auto sliceStart = std::vector<uint32_t>{ 15, 27, 400 };
    auto sliceEnd   = std::vector<uint32_t>{ 81, 59, 510 };

    std::vector<uint32_t> dim1{ 7, 3, 1024 };
    std::vector<uint64_t> values1;
    std::vector<uint64_t> expectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<uint64_t>( 1024 );
            values1.insert( values1.end(), y.begin(), y.end() );
            expectedValues1.insert( expectedValues1.end(), y.begin() + sliceStart[0], y.begin() + sliceEnd[0] + 1 );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<uint64_t> values2;
    std::vector<uint64_t> expectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<uint64_t>( 256 );
            values2.insert( values2.end(), y.begin(), y.end() );
            expectedValues2.insert( expectedValues2.end(), y.begin() + sliceStart[1], y.begin() + sliceEnd[1] + 1 );
        }
    }

    std::vector<uint32_t> dim3{ 3, 5, 512 };
    std::vector<uint64_t> values3;
    std::vector<uint64_t> expectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto y = RandomNumber<uint64_t>( 512 );
            values3.insert( values3.end(), y.begin(), y.end() );
            expectedValues3.insert( expectedValues3.end(), y.begin() + sliceStart[2], y.begin() + sliceEnd[2] + 1 );
        }
    }

    std::vector<uint64_t> inputValues;
    inputValues.insert( inputValues.end(), values1.begin(), values1.end() );
    inputValues.insert( inputValues.end(), values2.begin(), values2.end() );
    inputValues.insert( inputValues.end(), values3.begin(), values3.end() );

    data_initializer_t inputInitializer( inputValues );
    auto inputTensor = MultiTensorValue( scope, inputInitializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( uint64_t ) ) );

    auto start = VectorValue( scope, sliceStart );
    auto end   = VectorValue( scope, sliceEnd );

    auto result0 = Slice( scope, inputTensor, start, end );
    scope.Run( result0 );

    auto outputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( outputShape.CountLayers() == 3 );
    REQUIRE( outputShape.Rank == 3 );
    REQUIRE( outputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, sliceEnd[0] - sliceStart[0] + 1 } );
    REQUIRE( outputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, sliceEnd[1] - sliceStart[1] + 1 } );
    REQUIRE( outputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, sliceEnd[2] - sliceStart[2] + 1 } );

    std::vector<uint64_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<uint64_t>>{ expectedValues1, expectedValues2, expectedValues3 } );

    std::vector<uint64_t> resultValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint64_t>();
    REQUIRE( resultValues == expectedValues );
}

TEST_CASE( "ArraySlice SCALAR_VECTOR", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    uint32_t sliceStart = 25;
    auto     sliceEnd   = std::vector<uint32_t>{ 81, 59, 51 };

    std::vector<uint32_t> dim1{ 7, 3, 1024 };
    std::vector<uint64_t> values1;
    std::vector<uint64_t> expectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<uint64_t>( 1024 );
            values1.insert( values1.end(), y.begin(), y.end() );
            expectedValues1.insert( expectedValues1.end(), y.begin() + sliceStart, y.begin() + sliceEnd[0] + 1 );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<uint64_t> values2;
    std::vector<uint64_t> expectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<uint64_t>( 256 );
            values2.insert( values2.end(), y.begin(), y.end() );
            expectedValues2.insert( expectedValues2.end(), y.begin() + sliceStart, y.begin() + sliceEnd[1] + 1 );
        }
    }

    std::vector<uint32_t> dim3{ 3, 5, 512 };
    std::vector<uint64_t> values3;
    std::vector<uint64_t> expectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto y = RandomNumber<uint64_t>( 512 );
            values3.insert( values3.end(), y.begin(), y.end() );
            expectedValues3.insert( expectedValues3.end(), y.begin() + sliceStart, y.begin() + sliceEnd[2] + 1 );
        }
    }

    std::vector<uint64_t> inputValues;
    inputValues.insert( inputValues.end(), values1.begin(), values1.end() );
    inputValues.insert( inputValues.end(), values2.begin(), values2.end() );
    inputValues.insert( inputValues.end(), values3.begin(), values3.end() );

    data_initializer_t inputInitializer( inputValues );
    auto inputTensor = MultiTensorValue( scope, inputInitializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( uint64_t ) ) );

    auto start = ConstantScalarValue( scope, sliceStart );
    auto end   = VectorValue( scope, sliceEnd );

    auto result0 = Slice( scope, inputTensor, start, end );
    scope.Run( result0 );

    auto outputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( outputShape.CountLayers() == 3 );
    REQUIRE( outputShape.Rank == 3 );
    REQUIRE( outputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, sliceEnd[0] - sliceStart + 1 } );
    REQUIRE( outputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, sliceEnd[1] - sliceStart + 1 } );
    REQUIRE( outputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, sliceEnd[2] - sliceStart + 1 } );

    std::vector<uint64_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<uint64_t>>{ expectedValues1, expectedValues2, expectedValues3 } );

    std::vector<uint64_t> resultValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint64_t>();
    REQUIRE( resultValues == expectedValues );
}

TEST_CASE( "ArraySlice VECTOR_SCALAR", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    auto     sliceStart = std::vector<uint32_t>{ 15, 27, 39 };
    uint32_t sliceEnd   = 65;

    std::vector<uint32_t> dim1{ 7, 3, 1024 };
    std::vector<uint64_t> values1;
    std::vector<uint64_t> expectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<uint64_t>( 1024 );
            values1.insert( values1.end(), y.begin(), y.end() );
            expectedValues1.insert( expectedValues1.end(), y.begin() + sliceStart[0], y.begin() + sliceEnd + 1 );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<uint64_t> values2;
    std::vector<uint64_t> expectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<uint64_t>( 256 );
            values2.insert( values2.end(), y.begin(), y.end() );
            expectedValues2.insert( expectedValues2.end(), y.begin() + sliceStart[1], y.begin() + sliceEnd + 1 );
        }
    }

    std::vector<uint32_t> dim3{ 3, 5, 512 };
    std::vector<uint64_t> values3;
    std::vector<uint64_t> expectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto y = RandomNumber<uint64_t>( 512 );
            values3.insert( values3.end(), y.begin(), y.end() );
            expectedValues3.insert( expectedValues3.end(), y.begin() + sliceStart[2], y.begin() + sliceEnd + 1 );
        }
    }

    std::vector<uint64_t> inputValues;
    inputValues.insert( inputValues.end(), values1.begin(), values1.end() );
    inputValues.insert( inputValues.end(), values2.begin(), values2.end() );
    inputValues.insert( inputValues.end(), values3.begin(), values3.end() );

    data_initializer_t inputInitializer( inputValues );
    auto inputTensor = MultiTensorValue( scope, inputInitializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( uint64_t ) ) );

    auto start = VectorValue( scope, sliceStart );
    auto end   = ConstantScalarValue( scope, sliceEnd );

    auto result0 = Slice( scope, inputTensor, start, end );
    scope.Run( result0 );

    auto outputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( outputShape.CountLayers() == 3 );
    REQUIRE( outputShape.Rank == 3 );
    REQUIRE( outputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, sliceEnd - sliceStart[0] + 1 } );
    REQUIRE( outputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, sliceEnd - sliceStart[1] + 1 } );
    REQUIRE( outputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, sliceEnd - sliceStart[2] + 1 } );

    std::vector<uint64_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<uint64_t>>{ expectedValues1, expectedValues2, expectedValues3 } );

    std::vector<uint64_t> resultValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint64_t>();
    REQUIRE( resultValues == expectedValues );
}

TEST_CASE( "ArraySlice SCALAR_SCALAR", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    uint32_t sliceStart = 15;
    uint32_t sliceEnd   = 65;

    std::vector<uint32_t> dim1{ 7, 3, 1024 };
    std::vector<uint64_t> values1;
    std::vector<uint64_t> expectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<uint64_t>( 1024 );
            values1.insert( values1.end(), y.begin(), y.end() );
            expectedValues1.insert( expectedValues1.end(), y.begin() + sliceStart, y.begin() + sliceEnd + 1 );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<uint64_t> values2;
    std::vector<uint64_t> expectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<uint64_t>( 256 );
            values2.insert( values2.end(), y.begin(), y.end() );
            expectedValues2.insert( expectedValues2.end(), y.begin() + sliceStart, y.begin() + sliceEnd + 1 );
        }
    }

    std::vector<uint32_t> dim3{ 3, 5, 512 };
    std::vector<uint64_t> values3;
    std::vector<uint64_t> expectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto y = RandomNumber<uint64_t>( 512 );
            values3.insert( values3.end(), y.begin(), y.end() );
            expectedValues3.insert( expectedValues3.end(), y.begin() + sliceStart, y.begin() + sliceEnd + 1 );
        }
    }

    std::vector<uint64_t> inputValues;
    inputValues.insert( inputValues.end(), values1.begin(), values1.end() );
    inputValues.insert( inputValues.end(), values2.begin(), values2.end() );
    inputValues.insert( inputValues.end(), values3.begin(), values3.end() );

    data_initializer_t inputInitializer( inputValues );
    auto inputTensor = MultiTensorValue( scope, inputInitializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( uint64_t ) ) );

    auto start = ConstantScalarValue( scope, sliceStart );
    auto end   = ConstantScalarValue( scope, sliceEnd );

    auto result0 = Slice( scope, inputTensor, start, end );
    scope.Run( result0 );

    auto outputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( outputShape.CountLayers() == 3 );
    REQUIRE( outputShape.Rank == 3 );
    REQUIRE( outputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, sliceEnd - sliceStart + 1 } );
    REQUIRE( outputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, sliceEnd - sliceStart + 1 } );
    REQUIRE( outputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, sliceEnd - sliceStart + 1 } );

    std::vector<uint64_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<uint64_t>>{ expectedValues1, expectedValues2, expectedValues3 } );

    std::vector<uint64_t> resultValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint64_t>();
    REQUIRE( resultValues == expectedValues );
}

TEST_CASE( "ArraySummation VECTOR_VECTOR", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    auto sliceStart = std::vector<uint32_t>{ 15, 27, 400 };
    auto sliceEnd   = std::vector<uint32_t>{ 81, 59, 510 };

    std::vector<uint32_t> dim1{ 7, 3, 1024 };
    std::vector<uint64_t> values1;
    std::vector<uint64_t> expectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<uint64_t>( 1024 );
            values1.insert( values1.end(), y.begin(), y.end() );
            expectedValues1.push_back( std::accumulate( y.begin() + sliceStart[0], y.begin() + sliceEnd[0] + 1,
                                                        static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<uint64_t> values2;
    std::vector<uint64_t> expectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<uint64_t>( 256 );
            values2.insert( values2.end(), y.begin(), y.end() );
            expectedValues2.push_back( std::accumulate( y.begin() + sliceStart[1], y.begin() + sliceEnd[1] + 1,
                                                        static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint32_t> dim3{ 3, 5, 512 };
    std::vector<uint64_t> values3;
    std::vector<uint64_t> expectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto y = RandomNumber<uint64_t>( 512 );
            values3.insert( values3.end(), y.begin(), y.end() );
            expectedValues3.push_back( std::accumulate( y.begin() + sliceStart[2], y.begin() + sliceEnd[2] + 1,
                                                        static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint64_t> inputValues;
    inputValues.insert( inputValues.end(), values1.begin(), values1.end() );
    inputValues.insert( inputValues.end(), values2.begin(), values2.end() );
    inputValues.insert( inputValues.end(), values3.begin(), values3.end() );

    data_initializer_t inputInitializer( inputValues );
    auto inputTensor = MultiTensorValue( scope, inputInitializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( uint64_t ) ) );

    auto start = VectorValue( scope, sliceStart );
    auto end   = VectorValue( scope, sliceEnd );

    auto result0 = Summation( scope, inputTensor, start, end );
    scope.Run( result0 );

    auto outputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( outputShape.CountLayers() == 3 );
    REQUIRE( outputShape.Rank == 2 );
    REQUIRE( outputShape.Shape[0] == std::vector<uint32_t>{ 7, 3 } );
    REQUIRE( outputShape.Shape[1] == std::vector<uint32_t>{ 2, 7 } );
    REQUIRE( outputShape.Shape[2] == std::vector<uint32_t>{ 3, 5 } );

    std::vector<uint64_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<uint64_t>>{ expectedValues1, expectedValues2, expectedValues3 } );

    std::vector<uint64_t> resultValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint64_t>();
    REQUIRE( resultValues == expectedValues );
}

TEST_CASE( "ArraySummation SCALAR_VECTOR", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    uint32_t sliceStart = 25;
    auto     sliceEnd   = std::vector<uint32_t>{ 81, 59, 51 };

    std::vector<uint32_t> dim1{ 7, 3, 1024 };
    std::vector<uint64_t> values1;
    std::vector<uint64_t> expectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<uint64_t>( 1024 );
            values1.insert( values1.end(), y.begin(), y.end() );
            expectedValues1.push_back( std::accumulate( y.begin() + sliceStart, y.begin() + sliceEnd[0] + 1,
                                                        static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<uint64_t> values2;
    std::vector<uint64_t> expectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<uint64_t>( 256 );
            values2.insert( values2.end(), y.begin(), y.end() );
            expectedValues2.push_back( std::accumulate( y.begin() + sliceStart, y.begin() + sliceEnd[1] + 1,
                                                        static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint32_t> dim3{ 3, 5, 512 };
    std::vector<uint64_t> values3;
    std::vector<uint64_t> expectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto y = RandomNumber<uint64_t>( 512 );
            values3.insert( values3.end(), y.begin(), y.end() );
            expectedValues3.push_back( std::accumulate( y.begin() + sliceStart, y.begin() + sliceEnd[2] + 1,
                                                        static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint64_t> inputValues;
    inputValues.insert( inputValues.end(), values1.begin(), values1.end() );
    inputValues.insert( inputValues.end(), values2.begin(), values2.end() );
    inputValues.insert( inputValues.end(), values3.begin(), values3.end() );

    data_initializer_t inputInitializer( inputValues );
    auto inputTensor = MultiTensorValue( scope, inputInitializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( uint64_t ) ) );

    auto start = ConstantScalarValue( scope, sliceStart );
    auto end   = VectorValue( scope, sliceEnd );

    auto result0 = Summation( scope, inputTensor, start, end );
    scope.Run( result0 );

    auto outputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( outputShape.CountLayers() == 3 );
    REQUIRE( outputShape.Rank == 2 );
    REQUIRE( outputShape.Shape[0] == std::vector<uint32_t>{ 7, 3 } );
    REQUIRE( outputShape.Shape[1] == std::vector<uint32_t>{ 2, 7 } );
    REQUIRE( outputShape.Shape[2] == std::vector<uint32_t>{ 3, 5 } );

    std::vector<uint64_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<uint64_t>>{ expectedValues1, expectedValues2, expectedValues3 } );

    std::vector<uint64_t> resultValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint64_t>();
    REQUIRE( resultValues == expectedValues );
}

TEST_CASE( "ArraySummation VECTOR_SCALAR", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    auto     sliceStart = std::vector<uint32_t>{ 15, 27, 39 };
    uint32_t sliceEnd   = 65;

    std::vector<uint32_t> dim1{ 7, 3, 1024 };
    std::vector<uint64_t> values1;
    std::vector<uint64_t> expectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<uint64_t>( 1024 );
            values1.insert( values1.end(), y.begin(), y.end() );
            expectedValues1.push_back( std::accumulate( y.begin() + sliceStart[0], y.begin() + sliceEnd + 1,
                                                        static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<uint64_t> values2;
    std::vector<uint64_t> expectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<uint64_t>( 256 );
            values2.insert( values2.end(), y.begin(), y.end() );
            expectedValues2.push_back( std::accumulate( y.begin() + sliceStart[1], y.begin() + sliceEnd + 1,
                                                        static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint32_t> dim3{ 3, 5, 512 };
    std::vector<uint64_t> values3;
    std::vector<uint64_t> expectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto y = RandomNumber<uint64_t>( 512 );
            values3.insert( values3.end(), y.begin(), y.end() );
            expectedValues3.push_back( std::accumulate( y.begin() + sliceStart[2], y.begin() + sliceEnd + 1,
                                                        static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint64_t> inputValues;
    inputValues.insert( inputValues.end(), values1.begin(), values1.end() );
    inputValues.insert( inputValues.end(), values2.begin(), values2.end() );
    inputValues.insert( inputValues.end(), values3.begin(), values3.end() );

    data_initializer_t inputInitializer( inputValues );
    auto inputTensor = MultiTensorValue( scope, inputInitializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( uint64_t ) ) );

    auto start = VectorValue( scope, sliceStart );
    auto end   = ConstantScalarValue( scope, sliceEnd );

    auto result0 = Summation( scope, inputTensor, start, end );
    scope.Run( result0 );

    auto outputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( outputShape.CountLayers() == 3 );
    REQUIRE( outputShape.Rank == 2 );
    REQUIRE( outputShape.Shape[0] == std::vector<uint32_t>{ 7, 3 } );
    REQUIRE( outputShape.Shape[1] == std::vector<uint32_t>{ 2, 7 } );
    REQUIRE( outputShape.Shape[2] == std::vector<uint32_t>{ 3, 5 } );

    std::vector<uint64_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<uint64_t>>{ expectedValues1, expectedValues2, expectedValues3 } );

    std::vector<uint64_t> resultValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint64_t>();
    REQUIRE( resultValues == expectedValues );
}

TEST_CASE( "ArraySummation SCALAR_SCALAR", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    uint32_t sliceStart = 15;
    uint32_t sliceEnd   = 65;

    std::vector<uint32_t> dim1{ 7, 3, 1024 };
    std::vector<uint64_t> values1;
    std::vector<uint64_t> expectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<uint64_t>( 1024 );
            values1.insert( values1.end(), y.begin(), y.end() );
            expectedValues1.push_back( std::accumulate( y.begin() + sliceStart, y.begin() + sliceEnd + 1, static_cast<uint64_t>( 0 ),
                                                        std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<uint64_t> values2;
    std::vector<uint64_t> expectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<uint64_t>( 256 );
            values2.insert( values2.end(), y.begin(), y.end() );
            expectedValues2.push_back( std::accumulate( y.begin() + sliceStart, y.begin() + sliceEnd + 1, static_cast<uint64_t>( 0 ),
                                                        std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint32_t> dim3{ 3, 5, 512 };
    std::vector<uint64_t> values3;
    std::vector<uint64_t> expectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto y = RandomNumber<uint64_t>( 512 );
            values3.insert( values3.end(), y.begin(), y.end() );
            expectedValues3.push_back( std::accumulate( y.begin() + sliceStart, y.begin() + sliceEnd + 1, static_cast<uint64_t>( 0 ),
                                                        std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint64_t> inputValues;
    inputValues.insert( inputValues.end(), values1.begin(), values1.end() );
    inputValues.insert( inputValues.end(), values2.begin(), values2.end() );
    inputValues.insert( inputValues.end(), values3.begin(), values3.end() );

    data_initializer_t inputInitializer( inputValues );
    auto inputTensor = MultiTensorValue( scope, inputInitializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( uint64_t ) ) );

    auto start = ConstantScalarValue( scope, sliceStart );
    auto end   = ConstantScalarValue( scope, sliceEnd );

    auto result0 = Summation( scope, inputTensor, start, end );
    scope.Run( result0 );

    auto outputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( outputShape.CountLayers() == 3 );
    REQUIRE( outputShape.Rank == 2 );
    REQUIRE( outputShape.Shape[0] == std::vector<uint32_t>{ 7, 3 } );
    REQUIRE( outputShape.Shape[1] == std::vector<uint32_t>{ 2, 7 } );
    REQUIRE( outputShape.Shape[2] == std::vector<uint32_t>{ 3, 5 } );

    std::vector<uint64_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<uint64_t>>{ expectedValues1, expectedValues2, expectedValues3 } );

    std::vector<uint64_t> resultValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint64_t>();
    REQUIRE( resultValues == expectedValues );
}

TEST_CASE( "ArraySummation full", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    uint32_t sliceStart = 15;
    uint32_t sliceEnd   = 65;

    std::vector<uint32_t> dim1{ 7, 3, 1024 };
    std::vector<uint64_t> values1;
    std::vector<uint64_t> expectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<uint64_t>( 1024 );
            values1.insert( values1.end(), y.begin(), y.end() );
            expectedValues1.push_back( std::accumulate( y.begin(), y.end(), static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<uint64_t> values2;
    std::vector<uint64_t> expectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<uint64_t>( 256 );
            values2.insert( values2.end(), y.begin(), y.end() );
            expectedValues2.push_back( std::accumulate( y.begin(), y.end(), static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint32_t> dim3{ 3, 5, 512 };
    std::vector<uint64_t> values3;
    std::vector<uint64_t> expectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto y = RandomNumber<uint64_t>( 512 );
            values3.insert( values3.end(), y.begin(), y.end() );
            expectedValues3.push_back( std::accumulate( y.begin(), y.end(), static_cast<uint64_t>( 0 ), std::plus<uint64_t>() ) );
        }
    }

    std::vector<uint64_t> inputValues;
    inputValues.insert( inputValues.end(), values1.begin(), values1.end() );
    inputValues.insert( inputValues.end(), values2.begin(), values2.end() );
    inputValues.insert( inputValues.end(), values3.begin(), values3.end() );

    data_initializer_t inputInitializer( inputValues );
    auto inputTensor = MultiTensorValue( scope, inputInitializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( uint64_t ) ) );

    auto start = ConstantScalarValue( scope, sliceStart );
    auto end   = ConstantScalarValue( scope, sliceEnd );

    auto result0 = Summation( scope, inputTensor );
    scope.Run( result0 );

    auto outputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( outputShape.CountLayers() == 3 );
    REQUIRE( outputShape.Rank == 2 );
    REQUIRE( outputShape.Shape[0] == std::vector<uint32_t>{ 7, 3 } );
    REQUIRE( outputShape.Shape[1] == std::vector<uint32_t>{ 2, 7 } );
    REQUIRE( outputShape.Shape[2] == std::vector<uint32_t>{ 3, 5 } );

    std::vector<uint64_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<uint64_t>>{ expectedValues1, expectedValues2, expectedValues3 } );

    std::vector<uint64_t> resultValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint64_t>();
    REQUIRE( resultValues == expectedValues );
}

TEST_CASE( "CountTrue", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 7, 3, 1024 };
    std::vector<uint8_t>  values1;
    std::vector<uint32_t> expectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomBool( 1024 );
            values1.insert( values1.end(), y.begin(), y.end() );

            uint32_t trueCount = 0;
            for( auto &x : y )
                trueCount += x ? 1 : 0;
            expectedValues1.push_back( trueCount );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<uint8_t>  values2;
    std::vector<uint32_t> expectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomBool( 256 );
            values2.insert( values2.end(), y.begin(), y.end() );
            uint32_t trueCount = 0;
            for( auto &x : y )
                trueCount += x ? 1 : 0;
            expectedValues2.push_back( trueCount );
        }
    }

    std::vector<uint32_t> dim3{ 3, 5, 512 };
    std::vector<uint8_t>  values3;
    std::vector<uint32_t> expectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto y = RandomBool( 512 );
            values3.insert( values3.end(), y.begin(), y.end() );
            uint32_t trueCount = 0;
            for( auto &x : y )
                trueCount += x ? 1 : 0;
            expectedValues3.push_back( trueCount );
        }
    }

    std::vector<uint8_t> inputValues;
    inputValues.insert( inputValues.end(), values1.begin(), values1.end() );
    inputValues.insert( inputValues.end(), values2.begin(), values2.end() );
    inputValues.insert( inputValues.end(), values3.begin(), values3.end() );

    data_initializer_t inputInitializer( inputValues );
    auto inputTensor = MultiTensorValue( scope, inputInitializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( uint8_t ) ) );

    auto result0 = CountTrue( scope, inputTensor );
    scope.Run( result0 );

    auto outputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( outputShape.CountLayers() == 3 );
    REQUIRE( outputShape.Rank == 2 );
    REQUIRE( outputShape.Shape[0] == std::vector<uint32_t>{ 7, 3 } );
    REQUIRE( outputShape.Shape[1] == std::vector<uint32_t>{ 2, 7 } );
    REQUIRE( outputShape.Shape[2] == std::vector<uint32_t>{ 3, 5 } );

    std::vector<uint32_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<uint32_t>>{ expectedValues1, expectedValues2, expectedValues3 } );

    std::vector<uint32_t> resultValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint32_t>();
    REQUIRE( resultValues == expectedValues );
}

TEST_CASE( "CountNonZero", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 7, 3, 1024 };
    std::vector<uint64_t> values1;
    std::vector<uint32_t> expectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<uint64_t>( 1024 );
            values1.insert( values1.end(), y.begin(), y.end() );

            uint32_t trueCount = 0;
            for( auto &x : y )
                trueCount += ( x != 0 ) ? 1 : 0;
            expectedValues1.push_back( trueCount );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<uint64_t> values2;
    std::vector<uint32_t> expectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<uint64_t>( 256 );
            values2.insert( values2.end(), y.begin(), y.end() );
            uint32_t trueCount = 0;
            for( auto &x : y )
                trueCount += ( x != 0 ) ? 1 : 0;
            expectedValues2.push_back( trueCount );
        }
    }

    std::vector<uint32_t> dim3{ 3, 5, 512 };
    std::vector<uint64_t> values3;
    std::vector<uint32_t> expectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto y = RandomNumber<uint64_t>( 512 );
            values3.insert( values3.end(), y.begin(), y.end() );
            uint32_t trueCount = 0;
            for( auto &x : y )
                trueCount += ( x != 0 ) ? 1 : 0;
            expectedValues3.push_back( trueCount );
        }
    }

    std::vector<uint64_t> inputValues;
    inputValues.insert( inputValues.end(), values1.begin(), values1.end() );
    inputValues.insert( inputValues.end(), values2.begin(), values2.end() );
    inputValues.insert( inputValues.end(), values3.begin(), values3.end() );

    data_initializer_t inputInitializer( inputValues );
    auto inputTensor = MultiTensorValue( scope, inputInitializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( uint64_t ) ) );

    auto result0 = CountNonZero( scope, inputTensor );
    scope.Run( result0 );

    auto outputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( outputShape.CountLayers() == 3 );
    REQUIRE( outputShape.Rank == 2 );
    REQUIRE( outputShape.Shape[0] == std::vector<uint32_t>{ 7, 3 } );
    REQUIRE( outputShape.Shape[1] == std::vector<uint32_t>{ 2, 7 } );
    REQUIRE( outputShape.Shape[2] == std::vector<uint32_t>{ 3, 5 } );

    std::vector<uint32_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<uint32_t>>{ expectedValues1, expectedValues2, expectedValues3 } );

    std::vector<uint32_t> resultValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint32_t>();
    REQUIRE( resultValues == expectedValues );
}

TEST_CASE( "CountZero", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 7, 3, 1024 };
    std::vector<uint64_t> values1;
    std::vector<uint32_t> expectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<uint64_t>( 1024 );
            values1.insert( values1.end(), y.begin(), y.end() );

            uint32_t trueCount = 0;
            for( auto &x : y )
                trueCount += ( x == 0 ) ? 1 : 0;
            expectedValues1.push_back( trueCount );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<uint64_t> values2;
    std::vector<uint32_t> expectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<uint64_t>( 256 );
            values2.insert( values2.end(), y.begin(), y.end() );
            uint32_t trueCount = 0;
            for( auto &x : y )
                trueCount += ( x == 0 ) ? 1 : 0;
            expectedValues2.push_back( trueCount );
        }
    }

    std::vector<uint32_t> dim3{ 3, 5, 512 };
    std::vector<uint64_t> values3;
    std::vector<uint32_t> expectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto y = RandomNumber<uint64_t>( 512 );
            values3.insert( values3.end(), y.begin(), y.end() );
            uint32_t trueCount = 0;
            for( auto &x : y )
                trueCount += ( x == 0 ) ? 1 : 0;
            expectedValues3.push_back( trueCount );
        }
    }

    std::vector<uint64_t> inputValues;
    inputValues.insert( inputValues.end(), values1.begin(), values1.end() );
    inputValues.insert( inputValues.end(), values2.begin(), values2.end() );
    inputValues.insert( inputValues.end(), values3.begin(), values3.end() );

    data_initializer_t inputInitializer( inputValues );
    auto inputTensor = MultiTensorValue( scope, inputInitializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( uint64_t ) ) );

    auto result0 = CountZero( scope, inputTensor );
    scope.Run( result0 );

    auto outputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( outputShape.CountLayers() == 3 );
    REQUIRE( outputShape.Rank == 2 );
    REQUIRE( outputShape.Shape[0] == std::vector<uint32_t>{ 7, 3 } );
    REQUIRE( outputShape.Shape[1] == std::vector<uint32_t>{ 2, 7 } );
    REQUIRE( outputShape.Shape[2] == std::vector<uint32_t>{ 3, 5 } );

    std::vector<uint32_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<uint32_t>>{ expectedValues1, expectedValues2, expectedValues3 } );

    std::vector<uint32_t> resultValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint32_t>();
    REQUIRE( resultValues == expectedValues );
}

TEST_CASE( "Floor", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 7, 3, 500 };
    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<uint32_t> dim3{ 3, 5, 512 };

    auto opNode = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( float ) ) );

    auto result0 = Floor( scope, opNode );
    scope.Run( result0 );

    std::vector<float> resultValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    std::vector<float> expectedValues( opNode.Get<multi_tensor_value_t>().value.SizeAs<float>() );
    std::vector<float> opNodeValues = opNode.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    for( uint32_t i = 0; i < opNodeValues.size(); i++ )
    {
        expectedValues[i] = std::floor( opNodeValues[i] );
    }

    REQUIRE( resultValues == expectedValues );
}

TEST_CASE( "Ceiling", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 7, 3, 500 };
    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<uint32_t> dim3{ 3, 5, 512 };

    auto opNode = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( float ) ) );

    auto result0 = Ceil( scope, opNode );
    scope.Run( result0 );

    std::vector<float> resultValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    std::vector<float> expectedValues( opNode.Get<multi_tensor_value_t>().value.SizeAs<float>() );
    std::vector<float> opNodeValues = opNode.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    for( uint32_t i = 0; i < opNodeValues.size(); i++ )
    {
        expectedValues[i] = std::ceil( opNodeValues[i] );
    }

    REQUIRE( resultValues == expectedValues );
}

TEST_CASE( "Absolute value", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 7, 3, 500 };
    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<uint32_t> dim3{ 3, 5, 512 };

    auto opNode = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( float ) ) );

    auto result0 = Abs( scope, opNode );
    scope.Run( result0 );

    std::vector<float> resultValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    std::vector<float> expectedValues( opNode.Get<multi_tensor_value_t>().value.SizeAs<float>() );
    std::vector<float> opNodeValues = opNode.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    for( uint32_t i = 0; i < opNodeValues.size(); i++ )
    {
        expectedValues[i] = std::abs( opNodeValues[i] );
    }

    REQUIRE( resultValues == expectedValues );
}

TEST_CASE( "Square roots", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 17, 12, 51 };
    std::vector<uint32_t> dim2{ 12, 17, 23 };
    std::vector<uint32_t> dim3{ 13, 15, 52 };

    auto opNode = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( float ) ) );

    auto result0 = Sqrt( scope, opNode );
    scope.Run( result0 );

    std::vector<float> resultValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    std::vector<float> expectedValues( opNode.Get<multi_tensor_value_t>().value.SizeAs<float>() );
    std::vector<float> opNodeValues = opNode.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    for( uint32_t i = 0; i < opNodeValues.size(); i++ )
    {
        expectedValues[i] = std::sqrt( opNodeValues[i] );
    }

    REQUIRE( resultValues.size() == expectedValues.size() );
    std::vector<bool> comparison( resultValues.size() );
    for( uint32_t i = 0; i < opNodeValues.size(); i++ )
    {
        comparison[i] =
            ( std::isnan( resultValues[i] ) && std::isnan( expectedValues[i] ) ) || ( resultValues[i] == expectedValues[i] );
    }

    REQUIRE( std::all_of( comparison.begin(), comparison.end(), []( auto x ) { return x; } ) );
}

TEST_CASE( "Rounding", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 17, 12, 51 };
    std::vector<uint32_t> dim2{ 12, 17, 23 };
    std::vector<uint32_t> dim3{ 13, 15, 52 };

    auto opNode = MultiTensorValue( scope, initializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( float ) ) );

    auto result0 = Round( scope, opNode );
    scope.Run( result0 );

    std::vector<float> resultValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    std::vector<float> expectedValues( opNode.Get<multi_tensor_value_t>().value.SizeAs<float>() );
    std::vector<float> opNodeValues = opNode.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    for( uint32_t i = 0; i < opNodeValues.size(); i++ )
    {
        expectedValues[i] = std::round( opNodeValues[i] );
    }

    REQUIRE( resultValues.size() == expectedValues.size() );
    std::vector<bool> comparison( resultValues.size() );
    for( uint32_t i = 0; i < opNodeValues.size(); i++ )
    {
        comparison[i] = ( resultValues[i] == expectedValues[i] );
    }

    REQUIRE( std::all_of( comparison.begin(), comparison.end(), []( auto x ) { return x; } ) );
}

TEST_CASE( "Finite differences", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 7, 3, 1024 };
    std::vector<int64_t>  values1;
    std::vector<int64_t>  expectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<int64_t>( 1024 );
            values1.insert( values1.end(), y.begin(), y.end() );
            for( uint32_t l = 0; l < y.size() - 1; l++ )
                expectedValues1.push_back( y[l + 1] - y[l] );
            expectedValues1.push_back( static_cast<int64_t>( 0 ) );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<int64_t>  values2;
    std::vector<int64_t>  expectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<int64_t>( 256 );
            values2.insert( values2.end(), y.begin(), y.end() );
            for( uint32_t l = 0; l < y.size() - 1; l++ )
                expectedValues2.push_back( y[l + 1] - y[l] );
            expectedValues2.push_back( static_cast<int64_t>( 0 ) );
        }
    }

    std::vector<uint32_t> dim3{ 3, 5, 512 };
    std::vector<int64_t>  values3;
    std::vector<int64_t>  expectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto y = RandomNumber<int64_t>( 512 );
            values3.insert( values3.end(), y.begin(), y.end() );
            for( uint32_t l = 0; l < y.size() - 1; l++ )
                expectedValues3.push_back( y[l + 1] - y[l] );
            expectedValues3.push_back( static_cast<int64_t>( 0 ) );
        }
    }

    std::vector<int64_t> inputValues;
    inputValues.insert( inputValues.end(), values1.begin(), values1.end() );
    inputValues.insert( inputValues.end(), values2.begin(), values2.end() );
    inputValues.insert( inputValues.end(), values3.begin(), values3.end() );

    data_initializer_t inputInitializer( inputValues );
    auto inputTensor = MultiTensorValue( scope, inputInitializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( int64_t ) ) );

    auto result0 = Diff( scope, inputTensor, 1 );
    scope.Run( result0 );

    auto outputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( outputShape.CountLayers() == 3 );
    REQUIRE( outputShape.Rank == 3 );
    REQUIRE( outputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 1024 } );
    REQUIRE( outputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 256 } );
    REQUIRE( outputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 512 } );

    std::vector<int64_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<int64_t>>{ expectedValues1, expectedValues2, expectedValues3 } );

    std::vector<int64_t> resultValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<int64_t>();
    REQUIRE( resultValues == expectedValues );
}

TEST_CASE( "Finite shift to the left  by 1", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 7, 3, 1024 };
    std::vector<int64_t>  values1;
    std::vector<int64_t>  expectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<int64_t>( 1024 );
            values1.insert( values1.end(), y.begin(), y.end() );
            for( uint32_t l = 0; l < y.size() - 1; l++ )
                expectedValues1.push_back( y[l + 1] );
            expectedValues1.push_back( static_cast<int64_t>( 121212 ) );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<int64_t>  values2;
    std::vector<int64_t>  expectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<int64_t>( 256 );
            values2.insert( values2.end(), y.begin(), y.end() );
            for( uint32_t l = 0; l < y.size() - 1; l++ )
                expectedValues2.push_back( y[l + 1] );
            expectedValues2.push_back( static_cast<int64_t>( 121212 ) );
        }
    }

    std::vector<uint32_t> dim3{ 3, 5, 512 };
    std::vector<int64_t>  values3;
    std::vector<int64_t>  expectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto y = RandomNumber<int64_t>( 512 );
            values3.insert( values3.end(), y.begin(), y.end() );
            for( uint32_t l = 0; l < y.size() - 1; l++ )
                expectedValues3.push_back( y[l + 1] );
            expectedValues3.push_back( static_cast<int64_t>( 121212 ) );
        }
    }

    std::vector<int64_t> inputValues;
    inputValues.insert( inputValues.end(), values1.begin(), values1.end() );
    inputValues.insert( inputValues.end(), values2.begin(), values2.end() );
    inputValues.insert( inputValues.end(), values3.begin(), values3.end() );

    data_initializer_t inputInitializer( inputValues );
    auto inputTensor = MultiTensorValue( scope, inputInitializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( int64_t ) ) );

    auto fillValue = ConstantScalarValue( scope, static_cast<int64_t>( 121212 ) );
    auto result0   = Shift( scope, inputTensor, -1, fillValue );
    scope.Run( result0 );

    auto outputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( outputShape.CountLayers() == 3 );
    REQUIRE( outputShape.Rank == 3 );
    REQUIRE( outputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 1024 } );
    REQUIRE( outputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 256 } );
    REQUIRE( outputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 512 } );

    std::vector<int64_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<int64_t>>{ expectedValues1, expectedValues2, expectedValues3 } );

    std::vector<int64_t> resultValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<int64_t>();
    REQUIRE( resultValues == expectedValues );
}

TEST_CASE( "Finite shift to the left by 3", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 7, 3, 1024 };
    std::vector<int64_t>  values1;
    std::vector<int64_t>  expectedValues1;
    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<int64_t>( 1024 );
            values1.insert( values1.end(), y.begin(), y.end() );
            for( uint32_t l = 0; l < y.size() - 3; l++ )
                expectedValues1.push_back( y[l + 3] );
            expectedValues1.push_back( static_cast<int64_t>( 121212 ) );
            expectedValues1.push_back( static_cast<int64_t>( 121212 ) );
            expectedValues1.push_back( static_cast<int64_t>( 121212 ) );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 256 };
    std::vector<int64_t>  values2;
    std::vector<int64_t>  expectedValues2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<int64_t>( 256 );
            values2.insert( values2.end(), y.begin(), y.end() );
            for( uint32_t l = 0; l < y.size() - 3; l++ )
                expectedValues2.push_back( y[l + 3] );
            expectedValues2.push_back( static_cast<int64_t>( 121212 ) );
            expectedValues2.push_back( static_cast<int64_t>( 121212 ) );
            expectedValues2.push_back( static_cast<int64_t>( 121212 ) );
        }
    }

    std::vector<uint32_t> dim3{ 3, 5, 512 };
    std::vector<int64_t>  values3;
    std::vector<int64_t>  expectedValues3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto y = RandomNumber<int64_t>( 512 );
            values3.insert( values3.end(), y.begin(), y.end() );
            for( uint32_t l = 0; l < y.size() - 3; l++ )
                expectedValues3.push_back( y[l + 3] );
            expectedValues3.push_back( static_cast<int64_t>( 121212 ) );
            expectedValues3.push_back( static_cast<int64_t>( 121212 ) );
            expectedValues3.push_back( static_cast<int64_t>( 121212 ) );
        }
    }

    std::vector<int64_t> inputValues;
    inputValues.insert( inputValues.end(), values1.begin(), values1.end() );
    inputValues.insert( inputValues.end(), values2.begin(), values2.end() );
    inputValues.insert( inputValues.end(), values3.begin(), values3.end() );

    data_initializer_t inputInitializer( inputValues );
    auto inputTensor = MultiTensorValue( scope, inputInitializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( int64_t ) ) );

    auto fillValue = ConstantScalarValue( scope, static_cast<int64_t>( 121212 ) );
    auto result0   = Shift( scope, inputTensor, -3, fillValue );
    scope.Run( result0 );

    auto outputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( outputShape.CountLayers() == 3 );
    REQUIRE( outputShape.Rank == 3 );
    REQUIRE( outputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 1024 } );
    REQUIRE( outputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 256 } );
    REQUIRE( outputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 512 } );

    std::vector<int64_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<int64_t>>{ expectedValues1, expectedValues2, expectedValues3 } );

    std::vector<int64_t> resultValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<int64_t>();
    REQUIRE( resultValues == expectedValues );
}

TEST_CASE( "1D convolution", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    random_normal_initializer_t initializer{};
    initializer.type = scalar_type_t::FLOAT32;

    std::vector<uint32_t> dim1{ 7, 3, 124 };
    std::vector<int64_t>  values1;
    std::vector<int64_t>  expectedValues1;

    std::vector<uint32_t> KDim1{ 7, 3, 34 };
    std::vector<int64_t>  Kernel1;

    auto conv1D = []( std::vector<int64_t> aX, std::vector<int64_t> aY ) -> std::vector<int64_t>
    {
        auto output = std::vector<int64_t>( aX.size() );

        for( uint32_t i = 0; i < aX.size(); i++ )
        {
            int32_t acc = 0;
            for( uint32_t k = 0; k < aY.size(); k++ )
            {
                if( k <= i )
                    acc += aX[i - k] * aY[k];
            }
            output[i] = acc;
        }
        return output;
    };

    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<int64_t>( 124, -10000, 10000 );
            values1.insert( values1.end(), y.begin(), y.end() );

            auto z = RandomNumber<int64_t>( 34, -10000, 10000 );
            Kernel1.insert( Kernel1.end(), z.begin(), z.end() );

            auto c = conv1D( y, z );
            expectedValues1.insert( expectedValues1.end(), c.begin(), c.end() );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 75 };
    std::vector<int64_t>  values2;
    std::vector<int64_t>  expectedValues2;

    std::vector<uint32_t> KDim2{ 2, 7, 42 };
    std::vector<int64_t>  kernel2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<int64_t>( 75, -10000, 10000 );
            values2.insert( values2.end(), y.begin(), y.end() );

            auto z = RandomNumber<int64_t>( 42, -10000, 10000 );
            kernel2.insert( kernel2.end(), z.begin(), z.end() );

            auto c = conv1D( y, z );
            expectedValues2.insert( expectedValues2.end(), c.begin(), c.end() );
        }
    }

    std::vector<uint32_t> dim3{ 3, 5, 23 };
    std::vector<int64_t>  values3;
    std::vector<int64_t>  expectedValues3;

    std::vector<uint32_t> KDim3{ 3, 5, 5 };
    std::vector<int64_t>  kernel3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto y = RandomNumber<int64_t>( 23, -10000, 10000 );
            values3.insert( values3.end(), y.begin(), y.end() );

            auto z = RandomNumber<int64_t>( 5, -10000, 10000 );
            kernel3.insert( kernel3.end(), z.begin(), z.end() );

            auto c = conv1D( y, z );
            expectedValues3.insert( expectedValues3.end(), c.begin(), c.end() );
        }
    }

    std::vector<int64_t> inputValues;
    inputValues.insert( inputValues.end(), values1.begin(), values1.end() );
    inputValues.insert( inputValues.end(), values2.begin(), values2.end() );
    inputValues.insert( inputValues.end(), values3.begin(), values3.end() );

    std::vector<int64_t> kernelValues;
    kernelValues.insert( kernelValues.end(), Kernel1.begin(), Kernel1.end() );
    kernelValues.insert( kernelValues.end(), kernel2.begin(), kernel2.end() );
    kernelValues.insert( kernelValues.end(), kernel3.begin(), kernel3.end() );

    data_initializer_t inputInitializer( inputValues );
    auto inputTensor = MultiTensorValue( scope, inputInitializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( int64_t ) ) );

    data_initializer_t kernelInitializer( kernelValues );
    auto kernelensor = MultiTensorValue( scope, kernelInitializer, tensor_shape_t( { KDim1, KDim2, KDim3 }, sizeof( int64_t ) ) );

    auto result0 = Conv1D( scope, inputTensor, kernelensor );
    scope.Run( result0 );

    auto outputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( outputShape.CountLayers() == 3 );
    REQUIRE( outputShape.Rank == 3 );
    REQUIRE( outputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( outputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( outputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<int64_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<int64_t>>{ expectedValues1, expectedValues2, expectedValues3 } );

    std::vector<int64_t> resultValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<int64_t>();
    REQUIRE( resultValues == expectedValues );
}

TEST_CASE( "1D convolution (uint32_t)", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> dim1{ 7, 3, 124 };
    std::vector<uint32_t> values1;
    std::vector<uint32_t> expectedValues1;

    std::vector<uint32_t> KDim1{ 7, 3, 34 };
    std::vector<uint32_t> Kernel1;

    auto conv1D = []( std::vector<uint32_t> aX, std::vector<uint32_t> aY ) -> std::vector<uint32_t>
    {
        auto output = std::vector<uint32_t>( aX.size() );

        for( uint32_t i = 0; i < aX.size(); i++ )
        {
            uint32_t acc = 0;
            for( uint32_t k = 0; k < aY.size(); k++ )
            {
                if( k <= i )
                    acc += aX[i - k] * aY[k];
            }
            output[i] = acc;
        }
        return output;
    };

    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<uint32_t>( 124, 0, 10000 );
            values1.insert( values1.end(), y.begin(), y.end() );

            auto z = RandomNumber<uint32_t>( 34, 0, 10000 );
            Kernel1.insert( Kernel1.end(), z.begin(), z.end() );

            auto c = conv1D( y, z );
            expectedValues1.insert( expectedValues1.end(), c.begin(), c.end() );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 75 };
    std::vector<uint32_t> values2;
    std::vector<uint32_t> expectedValues2;

    std::vector<uint32_t> KDim2{ 2, 7, 42 };
    std::vector<uint32_t> kernel2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<uint32_t>( 75, 0, 10000 );
            values2.insert( values2.end(), y.begin(), y.end() );

            auto z = RandomNumber<uint32_t>( 42, 0, 10000 );
            kernel2.insert( kernel2.end(), z.begin(), z.end() );

            auto c = conv1D( y, z );
            expectedValues2.insert( expectedValues2.end(), c.begin(), c.end() );
        }
    }

    std::vector<uint32_t> dim3{ 3, 5, 23 };
    std::vector<uint32_t> values3;
    std::vector<uint32_t> expectedValues3;

    std::vector<uint32_t> KDim3{ 3, 5, 5 };
    std::vector<uint32_t> kernel3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto y = RandomNumber<uint32_t>( 23, 0, 10000 );
            values3.insert( values3.end(), y.begin(), y.end() );

            auto z = RandomNumber<uint32_t>( 5, 0, 10000 );
            kernel3.insert( kernel3.end(), z.begin(), z.end() );

            auto c = conv1D( y, z );
            expectedValues3.insert( expectedValues3.end(), c.begin(), c.end() );
        }
    }

    std::vector<uint32_t> inputValues;
    inputValues.insert( inputValues.end(), values1.begin(), values1.end() );
    inputValues.insert( inputValues.end(), values2.begin(), values2.end() );
    inputValues.insert( inputValues.end(), values3.begin(), values3.end() );

    std::vector<uint32_t> kernelValues;
    kernelValues.insert( kernelValues.end(), Kernel1.begin(), Kernel1.end() );
    kernelValues.insert( kernelValues.end(), kernel2.begin(), kernel2.end() );
    kernelValues.insert( kernelValues.end(), kernel3.begin(), kernel3.end() );

    data_initializer_t inputInitializer( inputValues );
    auto inputTensor = MultiTensorValue( scope, inputInitializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( uint32_t ) ) );

    data_initializer_t kernelInitializer( kernelValues );
    auto kernelensor = MultiTensorValue( scope, kernelInitializer, tensor_shape_t( { KDim1, KDim2, KDim3 }, sizeof( uint32_t ) ) );

    auto result0 = Conv1D( scope, inputTensor, kernelensor );
    scope.Run( result0 );

    auto outputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( outputShape.CountLayers() == 3 );
    REQUIRE( outputShape.Rank == 3 );
    REQUIRE( outputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( outputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( outputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<uint32_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<uint32_t>>{ expectedValues1, expectedValues2, expectedValues3 } );

    std::vector<uint32_t> resultValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint32_t>();
    REQUIRE( resultValues == expectedValues );
}

TEST_CASE( "HCat (uint32_t)", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t> dim1{ 7, 3, 124 };
    std::vector<uint32_t> values1;
    std::vector<uint32_t> expectedValues1;

    std::vector<uint32_t> KDim1{ 7, 3, 34 };
    std::vector<uint32_t> Kernel1;

    auto HCat = []( std::vector<uint32_t> aX, std::vector<uint32_t> aY ) -> std::vector<uint32_t>
    {
        auto output = std::vector<uint32_t>{};
        output.insert( output.end(), aX.begin(), aX.end() );
        output.insert( output.end(), aY.begin(), aY.end() );

        return output;
    };

    for( uint32_t i = 0; i < 7; i++ )
    {
        for( uint32_t j = 0; j < 3; j++ )
        {
            auto y = RandomNumber<uint32_t>( 124, 0, 10000 );
            values1.insert( values1.end(), y.begin(), y.end() );

            auto z = RandomNumber<uint32_t>( 34, 0, 10000 );
            Kernel1.insert( Kernel1.end(), z.begin(), z.end() );

            auto c = HCat( y, z );
            expectedValues1.insert( expectedValues1.end(), c.begin(), c.end() );
        }
    }

    std::vector<uint32_t> dim2{ 2, 7, 75 };
    std::vector<uint32_t> values2;
    std::vector<uint32_t> expectedValues2;

    std::vector<uint32_t> KDim2{ 2, 7, 42 };
    std::vector<uint32_t> kernel2;
    for( uint32_t i = 0; i < 2; i++ )
    {
        for( uint32_t j = 0; j < 7; j++ )
        {
            auto y = RandomNumber<uint32_t>( 75, 0, 10000 );
            values2.insert( values2.end(), y.begin(), y.end() );

            auto z = RandomNumber<uint32_t>( 42, 0, 10000 );
            kernel2.insert( kernel2.end(), z.begin(), z.end() );

            auto c = HCat( y, z );
            expectedValues2.insert( expectedValues2.end(), c.begin(), c.end() );
        }
    }

    std::vector<uint32_t> dim3{ 3, 5, 23 };
    std::vector<uint32_t> values3;
    std::vector<uint32_t> expectedValues3;

    std::vector<uint32_t> KDim3{ 3, 5, 5 };
    std::vector<uint32_t> kernel3;
    for( uint32_t i = 0; i < 3; i++ )
    {
        for( uint32_t j = 0; j < 5; j++ )
        {
            auto y = RandomNumber<uint32_t>( 23, 0, 10000 );
            values3.insert( values3.end(), y.begin(), y.end() );

            auto z = RandomNumber<uint32_t>( 5, 0, 10000 );
            kernel3.insert( kernel3.end(), z.begin(), z.end() );

            auto c = HCat( y, z );
            expectedValues3.insert( expectedValues3.end(), c.begin(), c.end() );
        }
    }

    std::vector<uint32_t> inputValues;
    inputValues.insert( inputValues.end(), values1.begin(), values1.end() );
    inputValues.insert( inputValues.end(), values2.begin(), values2.end() );
    inputValues.insert( inputValues.end(), values3.begin(), values3.end() );

    std::vector<uint32_t> kernelValues;
    kernelValues.insert( kernelValues.end(), Kernel1.begin(), Kernel1.end() );
    kernelValues.insert( kernelValues.end(), kernel2.begin(), kernel2.end() );
    kernelValues.insert( kernelValues.end(), kernel3.begin(), kernel3.end() );

    data_initializer_t inputInitializer( inputValues );
    auto inputTensor = MultiTensorValue( scope, inputInitializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( uint32_t ) ) );

    data_initializer_t kernelInitializer( kernelValues );
    auto kernelensor = MultiTensorValue( scope, kernelInitializer, tensor_shape_t( { KDim1, KDim2, KDim3 }, sizeof( uint32_t ) ) );

    auto result0 = numlua::mtops::HCat( scope, inputTensor, kernelensor );
    scope.Run( result0 );

    auto outputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( outputShape.CountLayers() == 3 );
    REQUIRE( outputShape.Rank == 3 );
    REQUIRE( outputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 124 + 34 } );
    REQUIRE( outputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 75 + 42 } );
    REQUIRE( outputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 23 + 5 } );

    std::vector<uint32_t> expectedValues =
        ConcatenateVectors( std::vector<std::vector<uint32_t>>{ expectedValues1, expectedValues2, expectedValues3 } );

    std::vector<uint32_t> resultValues = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint32_t>();
    REQUIRE( resultValues == expectedValues );
}

TEMPLATE_TEST_CASE( "Addition broadcast", "[CORE_COMPUTATION_GRAPH]", uint16_t, uint32_t, uint64_t, int16_t, int32_t, int64_t, float )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t>              dim1{ 7, 3, 124 };
    std::vector<std::vector<TestType>> values1 = RandomVector<TestType>( dim1, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              dim2{ 2, 7, 75 };
    std::vector<std::vector<TestType>> values2 = RandomVector<TestType>( dim2, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              dim3{ 3, 5, 23 };
    std::vector<std::vector<TestType>> values3 = RandomVector<TestType>( dim3, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> KDim1( dim1.begin(), dim1.end() - 1 );
    std::vector<TestType> Kernel1 = Randovalues<TestType>( KDim1, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> KDim2( dim2.begin(), dim2.end() - 1 );
    std::vector<TestType> kernel2 = Randovalues<TestType>( KDim2, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> KDim3( dim3.begin(), dim3.end() - 1 );
    std::vector<TestType> kernel3 = Randovalues<TestType>( KDim3, 0, std::numeric_limits<TestType>::max() / 2 );

    auto expectedValues1 = BroadcastMap<TestType>( values1, Kernel1, []( TestType x, TestType y ) { return x + y; } );
    auto expectedValues2 = BroadcastMap<TestType>( values2, kernel2, []( TestType x, TestType y ) { return x + y; } );
    auto expectedValues3 = BroadcastMap<TestType>( values3, kernel3, []( TestType x, TestType y ) { return x + y; } );

    std::vector<TestType> inputValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( values1 ), ConcatenateVectors( values2 ), ConcatenateVectors( values3 ) } );

    std::vector<TestType> kernelValues = ConcatenateVectors( std::vector<std::vector<TestType>>{ Kernel1, kernel2, kernel3 } );

    data_initializer_t inputInitializer( inputValues );
    auto inputTensor = MultiTensorValue( scope, inputInitializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( TestType ) ) );

    data_initializer_t kernelInitializer( kernelValues );
    auto kernelensor = MultiTensorValue( scope, kernelInitializer, tensor_shape_t( { KDim1, KDim2, KDim3 }, sizeof( TestType ) ) );

    auto result0 = Add( scope, inputTensor, kernelensor );
    auto result1 = Add( scope, kernelensor, inputTensor );
    scope.Run( { result0, result1 } );

    auto outputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( outputShape.CountLayers() == 3 );
    REQUIRE( outputShape.Rank == 3 );
    REQUIRE( outputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( outputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( outputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<TestType> expectedValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( expectedValues1 ), ConcatenateVectors( expectedValues2 ), ConcatenateVectors( expectedValues3 ) } );

    std::vector<TestType> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<TestType>();
    REQUIRE( resultValues0 == expectedValues );

    std::vector<TestType> resultValues1 = result1.Get<multi_tensor_value_t>().value.FetchFlattened<TestType>();
    REQUIRE( resultValues1 == expectedValues );
}

TEMPLATE_TEST_CASE( "Multiplication broadcast", "[CORE_COMPUTATION_GRAPH]", uint16_t, uint32_t, uint64_t, int16_t, int32_t, int64_t,
                    float )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t>              dim1{ 7, 3, 124 };
    std::vector<std::vector<TestType>> values1 = RandomVector<TestType>( dim1, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              dim2{ 2, 7, 75 };
    std::vector<std::vector<TestType>> values2 = RandomVector<TestType>( dim2, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              dim3{ 3, 5, 23 };
    std::vector<std::vector<TestType>> values3 = RandomVector<TestType>( dim3, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> KDim1( dim1.begin(), dim1.end() - 1 );
    std::vector<TestType> Kernel1 = Randovalues<TestType>( KDim1, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> KDim2( dim2.begin(), dim2.end() - 1 );
    std::vector<TestType> kernel2 = Randovalues<TestType>( KDim2, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> KDim3( dim3.begin(), dim3.end() - 1 );
    std::vector<TestType> kernel3 = Randovalues<TestType>( KDim3, 0, std::numeric_limits<TestType>::max() / 2 );

    auto expectedValues1 = BroadcastMap<TestType>( values1, Kernel1, []( TestType x, TestType y ) { return x * y; } );
    auto expectedValues2 = BroadcastMap<TestType>( values2, kernel2, []( TestType x, TestType y ) { return x * y; } );
    auto expectedValues3 = BroadcastMap<TestType>( values3, kernel3, []( TestType x, TestType y ) { return x * y; } );

    std::vector<TestType> inputValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( values1 ), ConcatenateVectors( values2 ), ConcatenateVectors( values3 ) } );

    std::vector<TestType> kernelValues = ConcatenateVectors( std::vector<std::vector<TestType>>{ Kernel1, kernel2, kernel3 } );

    data_initializer_t inputInitializer( inputValues );
    auto inputTensor = MultiTensorValue( scope, inputInitializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( TestType ) ) );

    data_initializer_t kernelInitializer( kernelValues );
    auto kernelensor = MultiTensorValue( scope, kernelInitializer, tensor_shape_t( { KDim1, KDim2, KDim3 }, sizeof( TestType ) ) );

    auto result0 = Multiply( scope, inputTensor, kernelensor );
    auto result1 = Multiply( scope, kernelensor, inputTensor );
    scope.Run( { result0, result1 } );

    auto outputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( outputShape.CountLayers() == 3 );
    REQUIRE( outputShape.Rank == 3 );
    REQUIRE( outputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( outputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( outputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<TestType> expectedValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( expectedValues1 ), ConcatenateVectors( expectedValues2 ), ConcatenateVectors( expectedValues3 ) } );

    std::vector<TestType> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<TestType>();
    REQUIRE( resultValues0 == expectedValues );

    std::vector<TestType> resultValues1 = result1.Get<multi_tensor_value_t>().value.FetchFlattened<TestType>();
    REQUIRE( resultValues1 == expectedValues );
}

TEST_CASE( "Divison broadcast", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t>           dim1{ 7, 3, 124 };
    std::vector<std::vector<float>> values1 = RandomVector<float>( dim1, 0, std::numeric_limits<float>::max() / 2 );

    std::vector<uint32_t>           dim2{ 2, 7, 75 };
    std::vector<std::vector<float>> values2 = RandomVector<float>( dim2, 0, std::numeric_limits<float>::max() / 2 );

    std::vector<uint32_t>           dim3{ 3, 5, 23 };
    std::vector<std::vector<float>> values3 = RandomVector<float>( dim3, 0, std::numeric_limits<float>::max() / 2 );

    std::vector<uint32_t> KDim1( dim1.begin(), dim1.end() - 1 );
    std::vector<float>    Kernel1 = Randovalues<float>( KDim1, 0.001, std::numeric_limits<float>::max() / 2 );

    std::vector<uint32_t> KDim2( dim2.begin(), dim2.end() - 1 );
    std::vector<float>    kernel2 = Randovalues<float>( KDim2, 0.001, std::numeric_limits<float>::max() / 2 );

    std::vector<uint32_t> KDim3( dim3.begin(), dim3.end() - 1 );
    std::vector<float>    kernel3 = Randovalues<float>( KDim3, 0.001, std::numeric_limits<float>::max() / 2 );

    auto expectedValues01 = BroadcastMap<float>( values1, Kernel1, []( float x, float y ) { return x / y; } );
    auto expectedValues02 = BroadcastMap<float>( values2, kernel2, []( float x, float y ) { return x / y; } );
    auto expectedValues03 = BroadcastMap<float>( values3, kernel3, []( float x, float y ) { return x / y; } );

    auto expectedValues11 = BroadcastMap<float>( Kernel1, values1, []( float x, float y ) { return x / y; } );
    auto expectedValues12 = BroadcastMap<float>( kernel2, values2, []( float x, float y ) { return x / y; } );
    auto expectedValues13 = BroadcastMap<float>( kernel3, values3, []( float x, float y ) { return x / y; } );

    std::vector<float> inputValues = ConcatenateVectors( std::vector<std::vector<float>>{
        ConcatenateVectors( values1 ), ConcatenateVectors( values2 ), ConcatenateVectors( values3 ) } );

    std::vector<float> kernelValues = ConcatenateVectors( std::vector<std::vector<float>>{ Kernel1, kernel2, kernel3 } );

    data_initializer_t inputInitializer( inputValues );
    auto inputTensor = MultiTensorValue( scope, inputInitializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( float ) ) );

    data_initializer_t kernelInitializer( kernelValues );
    auto kernelensor = MultiTensorValue( scope, kernelInitializer, tensor_shape_t( { KDim1, KDim2, KDim3 }, sizeof( float ) ) );

    auto result0 = Divide( scope, inputTensor, kernelensor );
    auto result1 = Divide( scope, kernelensor, inputTensor );
    scope.Run( { result0, result1 } );

    auto outputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( outputShape.CountLayers() == 3 );
    REQUIRE( outputShape.Rank == 3 );
    REQUIRE( outputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( outputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( outputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<float> expectedValues0 = ConcatenateVectors( std::vector<std::vector<float>>{
        ConcatenateVectors( expectedValues01 ), ConcatenateVectors( expectedValues02 ), ConcatenateVectors( expectedValues03 ) } );
    std::vector<float> expectedValues1 = ConcatenateVectors( std::vector<std::vector<float>>{
        ConcatenateVectors( expectedValues11 ), ConcatenateVectors( expectedValues12 ), ConcatenateVectors( expectedValues13 ) } );

    std::vector<float> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    REQUIRE( resultValues0 == expectedValues0 );

    std::vector<float> resultValues1 = result1.Get<multi_tensor_value_t>().value.FetchFlattened<float>();
    REQUIRE( resultValues1 == expectedValues1 );
}

TEMPLATE_TEST_CASE( "Subtraction broadcast", "[CORE_COMPUTATION_GRAPH]", int16_t, int32_t, int64_t, float, double )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t>              dim1{ 7, 3, 124 };
    std::vector<std::vector<TestType>> values1 = RandomVector<TestType>( dim1, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              dim2{ 2, 7, 75 };
    std::vector<std::vector<TestType>> values2 = RandomVector<TestType>( dim2, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              dim3{ 3, 5, 23 };
    std::vector<std::vector<TestType>> values3 = RandomVector<TestType>( dim3, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> KDim1( dim1.begin(), dim1.end() - 1 );
    std::vector<TestType> Kernel1 = Randovalues<TestType>( KDim1, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> KDim2( dim2.begin(), dim2.end() - 1 );
    std::vector<TestType> kernel2 = Randovalues<TestType>( KDim2, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> KDim3( dim3.begin(), dim3.end() - 1 );
    std::vector<TestType> kernel3 = Randovalues<TestType>( KDim3, 0, std::numeric_limits<TestType>::max() / 2 );

    auto expectedValues01 = BroadcastMap<TestType>( values1, Kernel1, []( TestType x, TestType y ) { return x - y; } );
    auto expectedValues02 = BroadcastMap<TestType>( values2, kernel2, []( TestType x, TestType y ) { return x - y; } );
    auto expectedValues03 = BroadcastMap<TestType>( values3, kernel3, []( TestType x, TestType y ) { return x - y; } );

    auto expectedValues11 = BroadcastMap<TestType>( Kernel1, values1, []( TestType x, TestType y ) { return x - y; } );
    auto expectedValues12 = BroadcastMap<TestType>( kernel2, values2, []( TestType x, TestType y ) { return x - y; } );
    auto expectedValues13 = BroadcastMap<TestType>( kernel3, values3, []( TestType x, TestType y ) { return x - y; } );

    std::vector<TestType> inputValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( values1 ), ConcatenateVectors( values2 ), ConcatenateVectors( values3 ) } );

    std::vector<TestType> kernelValues = ConcatenateVectors( std::vector<std::vector<TestType>>{ Kernel1, kernel2, kernel3 } );

    data_initializer_t inputInitializer( inputValues );
    auto inputTensor = MultiTensorValue( scope, inputInitializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( TestType ) ) );

    data_initializer_t kernelInitializer( kernelValues );
    auto kernelensor = MultiTensorValue( scope, kernelInitializer, tensor_shape_t( { KDim1, KDim2, KDim3 }, sizeof( TestType ) ) );

    auto result0 = Subtract( scope, inputTensor, kernelensor );
    auto result1 = Subtract( scope, kernelensor, inputTensor );
    scope.Run( { result0, result1 } );

    auto outputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( outputShape.CountLayers() == 3 );
    REQUIRE( outputShape.Rank == 3 );
    REQUIRE( outputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( outputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( outputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<TestType> expectedValues0 = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( expectedValues01 ), ConcatenateVectors( expectedValues02 ), ConcatenateVectors( expectedValues03 ) } );
    std::vector<TestType> expectedValues1 = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( expectedValues11 ), ConcatenateVectors( expectedValues12 ), ConcatenateVectors( expectedValues13 ) } );

    std::vector<TestType> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<TestType>();
    REQUIRE( resultValues0 == expectedValues0 );

    std::vector<TestType> resultValues1 = result1.Get<multi_tensor_value_t>().value.FetchFlattened<TestType>();
    REQUIRE( resultValues1 == expectedValues1 );
}

TEST_CASE( "AND broadcast", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t>             dim1{ 7, 3, 124 };
    std::vector<std::vector<uint8_t>> values1 = RandomBooleanVector( dim1 );

    std::vector<uint32_t>             dim2{ 2, 7, 75 };
    std::vector<std::vector<uint8_t>> values2 = RandomBooleanVector( dim2 );

    std::vector<uint32_t>             dim3{ 3, 5, 23 };
    std::vector<std::vector<uint8_t>> values3 = RandomBooleanVector( dim3 );

    std::vector<uint32_t> KDim1( dim1.begin(), dim1.end() - 1 );
    std::vector<uint8_t>  Kernel1 = RandomBooleanValues( KDim1 );

    std::vector<uint32_t> KDim2( dim2.begin(), dim2.end() - 1 );
    std::vector<uint8_t>  kernel2 = RandomBooleanValues( KDim2 );

    std::vector<uint32_t> KDim3( dim3.begin(), dim3.end() - 1 );
    std::vector<uint8_t>  kernel3 = RandomBooleanValues( KDim3 );

    auto expectedValues1 = BroadcastMap<uint8_t>( values1, Kernel1, []( uint8_t x, uint8_t y ) { return x && y; } );
    auto expectedValues2 = BroadcastMap<uint8_t>( values2, kernel2, []( uint8_t x, uint8_t y ) { return x && y; } );
    auto expectedValues3 = BroadcastMap<uint8_t>( values3, kernel3, []( uint8_t x, uint8_t y ) { return x && y; } );

    std::vector<uint8_t> inputValues = ConcatenateVectors( std::vector<std::vector<uint8_t>>{
        ConcatenateVectors( values1 ), ConcatenateVectors( values2 ), ConcatenateVectors( values3 ) } );

    std::vector<uint8_t> kernelValues = ConcatenateVectors( std::vector<std::vector<uint8_t>>{ Kernel1, kernel2, kernel3 } );

    data_initializer_t inputInitializer( inputValues );
    auto inputTensor = MultiTensorValue( scope, inputInitializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( uint8_t ) ) );

    data_initializer_t kernelInitializer( kernelValues );
    auto kernelensor = MultiTensorValue( scope, kernelInitializer, tensor_shape_t( { KDim1, KDim2, KDim3 }, sizeof( uint8_t ) ) );

    auto result0 = And( scope, inputTensor, kernelensor );
    auto result1 = And( scope, kernelensor, inputTensor );
    scope.Run( { result0, result1 } );

    auto outputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( outputShape.CountLayers() == 3 );
    REQUIRE( outputShape.Rank == 3 );
    REQUIRE( outputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( outputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( outputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<uint8_t> expectedValues = ConcatenateVectors( std::vector<std::vector<uint8_t>>{
        ConcatenateVectors( expectedValues1 ), ConcatenateVectors( expectedValues2 ), ConcatenateVectors( expectedValues3 ) } );

    std::vector<uint8_t> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    REQUIRE( resultValues0 == expectedValues );

    std::vector<uint8_t> resultValues1 = result1.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    REQUIRE( resultValues1 == expectedValues );
}

TEST_CASE( "OR broadcast", "[CORE_COMPUTATION_GRAPH]" )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t>             dim1{ 7, 3, 124 };
    std::vector<std::vector<uint8_t>> values1 = RandomBooleanVector( dim1 );

    std::vector<uint32_t>             dim2{ 2, 7, 75 };
    std::vector<std::vector<uint8_t>> values2 = RandomBooleanVector( dim2 );

    std::vector<uint32_t>             dim3{ 3, 5, 23 };
    std::vector<std::vector<uint8_t>> values3 = RandomBooleanVector( dim3 );

    std::vector<uint32_t> KDim1( dim1.begin(), dim1.end() - 1 );
    std::vector<uint8_t>  Kernel1 = RandomBooleanValues( KDim1 );

    std::vector<uint32_t> KDim2( dim2.begin(), dim2.end() - 1 );
    std::vector<uint8_t>  kernel2 = RandomBooleanValues( KDim2 );

    std::vector<uint32_t> KDim3( dim3.begin(), dim3.end() - 1 );
    std::vector<uint8_t>  kernel3 = RandomBooleanValues( KDim3 );

    auto expectedValues1 = BroadcastMap<uint8_t>( values1, Kernel1, []( uint8_t x, uint8_t y ) { return x || y; } );
    auto expectedValues2 = BroadcastMap<uint8_t>( values2, kernel2, []( uint8_t x, uint8_t y ) { return x || y; } );
    auto expectedValues3 = BroadcastMap<uint8_t>( values3, kernel3, []( uint8_t x, uint8_t y ) { return x || y; } );

    std::vector<uint8_t> inputValues = ConcatenateVectors( std::vector<std::vector<uint8_t>>{
        ConcatenateVectors( values1 ), ConcatenateVectors( values2 ), ConcatenateVectors( values3 ) } );

    std::vector<uint8_t> kernelValues = ConcatenateVectors( std::vector<std::vector<uint8_t>>{ Kernel1, kernel2, kernel3 } );

    data_initializer_t inputInitializer( inputValues );
    auto inputTensor = MultiTensorValue( scope, inputInitializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( uint8_t ) ) );

    data_initializer_t kernelInitializer( kernelValues );
    auto kernelensor = MultiTensorValue( scope, kernelInitializer, tensor_shape_t( { KDim1, KDim2, KDim3 }, sizeof( uint8_t ) ) );

    auto result0 = Or( scope, inputTensor, kernelensor );
    auto result1 = Or( scope, kernelensor, inputTensor );
    scope.Run( { result0, result1 } );

    auto outputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( outputShape.CountLayers() == 3 );
    REQUIRE( outputShape.Rank == 3 );
    REQUIRE( outputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( outputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( outputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<uint8_t> expectedValues = ConcatenateVectors( std::vector<std::vector<uint8_t>>{
        ConcatenateVectors( expectedValues1 ), ConcatenateVectors( expectedValues2 ), ConcatenateVectors( expectedValues3 ) } );

    std::vector<uint8_t> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    REQUIRE( resultValues0 == expectedValues );

    std::vector<uint8_t> resultValues1 = result1.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    REQUIRE( resultValues1 == expectedValues );
}

TEMPLATE_TEST_CASE( "Bitwise AND broadcast", "[CORE_COMPUTATION_GRAPH]", uint16_t, uint32_t, uint64_t, int16_t, int32_t, int64_t )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t>              dim1{ 7, 3, 124 };
    std::vector<std::vector<TestType>> values1 = RandomVector<TestType>( dim1, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t>              dim2{ 2, 7, 75 };
    std::vector<std::vector<TestType>> values2 = RandomVector<TestType>( dim2, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t>              dim3{ 3, 5, 23 };
    std::vector<std::vector<TestType>> values3 = RandomVector<TestType>( dim3, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t> KDim1( dim1.begin(), dim1.end() - 1 );
    std::vector<TestType> Kernel1 = Randovalues<TestType>( KDim1, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t> KDim2( dim2.begin(), dim2.end() - 1 );
    std::vector<TestType> kernel2 = Randovalues<TestType>( KDim2, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t> KDim3( dim3.begin(), dim3.end() - 1 );
    std::vector<TestType> kernel3 = Randovalues<TestType>( KDim3, 0, std::numeric_limits<TestType>::max() );

    auto expectedValues1 = BroadcastMap<TestType>( values1, Kernel1, []( TestType x, TestType y ) { return x & y; } );
    auto expectedValues2 = BroadcastMap<TestType>( values2, kernel2, []( TestType x, TestType y ) { return x & y; } );
    auto expectedValues3 = BroadcastMap<TestType>( values3, kernel3, []( TestType x, TestType y ) { return x & y; } );

    std::vector<TestType> inputValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( values1 ), ConcatenateVectors( values2 ), ConcatenateVectors( values3 ) } );

    std::vector<TestType> kernelValues = ConcatenateVectors( std::vector<std::vector<TestType>>{ Kernel1, kernel2, kernel3 } );

    data_initializer_t inputInitializer( inputValues );
    auto inputTensor = MultiTensorValue( scope, inputInitializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( TestType ) ) );

    data_initializer_t kernelInitializer( kernelValues );
    auto kernelensor = MultiTensorValue( scope, kernelInitializer, tensor_shape_t( { KDim1, KDim2, KDim3 }, sizeof( TestType ) ) );

    auto result0 = BitwiseAnd( scope, inputTensor, kernelensor );
    auto result1 = BitwiseAnd( scope, kernelensor, inputTensor );
    scope.Run( { result0, result1 } );

    auto outputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( outputShape.CountLayers() == 3 );
    REQUIRE( outputShape.Rank == 3 );
    REQUIRE( outputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( outputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( outputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<TestType> expectedValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( expectedValues1 ), ConcatenateVectors( expectedValues2 ), ConcatenateVectors( expectedValues3 ) } );

    std::vector<TestType> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<TestType>();
    REQUIRE( resultValues0 == expectedValues );

    std::vector<TestType> resultValues1 = result1.Get<multi_tensor_value_t>().value.FetchFlattened<TestType>();
    REQUIRE( resultValues1 == expectedValues );
}

TEMPLATE_TEST_CASE( "Bitwise OR broadcast", "[CORE_COMPUTATION_GRAPH]", uint16_t, uint32_t, uint64_t, int16_t, int32_t, int64_t )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t>              dim1{ 7, 3, 124 };
    std::vector<std::vector<TestType>> values1 = RandomVector<TestType>( dim1, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t>              dim2{ 2, 7, 75 };
    std::vector<std::vector<TestType>> values2 = RandomVector<TestType>( dim2, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t>              dim3{ 3, 5, 23 };
    std::vector<std::vector<TestType>> values3 = RandomVector<TestType>( dim3, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t> KDim1( dim1.begin(), dim1.end() - 1 );
    std::vector<TestType> Kernel1 = Randovalues<TestType>( KDim1, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t> KDim2( dim2.begin(), dim2.end() - 1 );
    std::vector<TestType> kernel2 = Randovalues<TestType>( KDim2, 0, std::numeric_limits<TestType>::max() );

    std::vector<uint32_t> KDim3( dim3.begin(), dim3.end() - 1 );
    std::vector<TestType> kernel3 = Randovalues<TestType>( KDim3, 0, std::numeric_limits<TestType>::max() );

    auto expectedValues1 = BroadcastMap<TestType>( values1, Kernel1, []( TestType x, TestType y ) { return x | y; } );
    auto expectedValues2 = BroadcastMap<TestType>( values2, kernel2, []( TestType x, TestType y ) { return x | y; } );
    auto expectedValues3 = BroadcastMap<TestType>( values3, kernel3, []( TestType x, TestType y ) { return x | y; } );

    std::vector<TestType> inputValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( values1 ), ConcatenateVectors( values2 ), ConcatenateVectors( values3 ) } );

    std::vector<TestType> kernelValues = ConcatenateVectors( std::vector<std::vector<TestType>>{ Kernel1, kernel2, kernel3 } );

    data_initializer_t inputInitializer( inputValues );
    auto inputTensor = MultiTensorValue( scope, inputInitializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( TestType ) ) );

    data_initializer_t kernelInitializer( kernelValues );
    auto kernelensor = MultiTensorValue( scope, kernelInitializer, tensor_shape_t( { KDim1, KDim2, KDim3 }, sizeof( TestType ) ) );

    auto result0 = BitwiseOr( scope, inputTensor, kernelensor );
    auto result1 = BitwiseOr( scope, kernelensor, inputTensor );
    scope.Run( { result0, result1 } );

    auto outputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( outputShape.CountLayers() == 3 );
    REQUIRE( outputShape.Rank == 3 );
    REQUIRE( outputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( outputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( outputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<TestType> expectedValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( expectedValues1 ), ConcatenateVectors( expectedValues2 ), ConcatenateVectors( expectedValues3 ) } );

    std::vector<TestType> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<TestType>();
    REQUIRE( resultValues0 == expectedValues );

    std::vector<TestType> resultValues1 = result1.Get<multi_tensor_value_t>().value.FetchFlattened<TestType>();
    REQUIRE( resultValues1 == expectedValues );
}

TEMPLATE_TEST_CASE( "Equal broadcast", "[CORE_COMPUTATION_GRAPH]", uint16_t, uint32_t, uint64_t, int16_t, int32_t, int64_t, float,
                    double )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t>              dim1{ 7, 3, 124 };
    std::vector<std::vector<TestType>> values1 = RandomVector<TestType>( dim1, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              dim2{ 2, 7, 75 };
    std::vector<std::vector<TestType>> values2 = RandomVector<TestType>( dim2, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              dim3{ 3, 5, 23 };
    std::vector<std::vector<TestType>> values3 = RandomVector<TestType>( dim3, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> KDim1( dim1.begin(), dim1.end() - 1 );
    std::vector<TestType> Kernel1 = Randovalues<TestType>( KDim1, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> KDim2( dim2.begin(), dim2.end() - 1 );
    std::vector<TestType> kernel2 = Randovalues<TestType>( KDim2, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> KDim3( dim3.begin(), dim3.end() - 1 );
    std::vector<TestType> kernel3 = Randovalues<TestType>( KDim3, 0, std::numeric_limits<TestType>::max() / 2 );

    auto expectedValues1 = BroadcastMap<TestType>( values1, Kernel1, []( TestType x, TestType y ) { return x == y; } );
    auto expectedValues2 = BroadcastMap<TestType>( values2, kernel2, []( TestType x, TestType y ) { return x == y; } );
    auto expectedValues3 = BroadcastMap<TestType>( values3, kernel3, []( TestType x, TestType y ) { return x == y; } );

    std::vector<TestType> inputValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( values1 ), ConcatenateVectors( values2 ), ConcatenateVectors( values3 ) } );

    std::vector<TestType> kernelValues = ConcatenateVectors( std::vector<std::vector<TestType>>{ Kernel1, kernel2, kernel3 } );

    data_initializer_t inputInitializer( inputValues );
    auto inputTensor = MultiTensorValue( scope, inputInitializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( TestType ) ) );

    data_initializer_t kernelInitializer( kernelValues );
    auto kernelensor = MultiTensorValue( scope, kernelInitializer, tensor_shape_t( { KDim1, KDim2, KDim3 }, sizeof( TestType ) ) );

    auto result0 = Equal( scope, inputTensor, kernelensor );
    auto result1 = Equal( scope, kernelensor, inputTensor );
    scope.Run( { result0, result1 } );

    auto outputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( outputShape.CountLayers() == 3 );
    REQUIRE( outputShape.Rank == 3 );
    REQUIRE( outputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( outputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( outputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<TestType> expectedValues0 = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( expectedValues1 ), ConcatenateVectors( expectedValues2 ), ConcatenateVectors( expectedValues3 ) } );

    std::vector<uint8_t> expectedValues{};
    for( auto x : expectedValues0 )
        expectedValues.push_back( static_cast<uint8_t>( x != 0 ) );
    std::vector<uint8_t> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    REQUIRE( resultValues0 == expectedValues );

    std::vector<uint8_t> resultValues1 = result1.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    REQUIRE( resultValues1 == expectedValues );
}

TEMPLATE_TEST_CASE( "LessThan broadcast", "[CORE_COMPUTATION_GRAPH]", uint16_t, uint32_t, uint64_t, int16_t, int32_t, int64_t, float,
                    double )
{
    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t>              dim1{ 7, 3, 124 };
    std::vector<std::vector<TestType>> values1 = RandomVector<TestType>( dim1, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              dim2{ 2, 7, 75 };
    std::vector<std::vector<TestType>> values2 = RandomVector<TestType>( dim2, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              dim3{ 3, 5, 23 };
    std::vector<std::vector<TestType>> values3 = RandomVector<TestType>( dim3, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> KDim1( dim1.begin(), dim1.end() - 1 );
    std::vector<TestType> Kernel1 = Randovalues<TestType>( KDim1, 0.001, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> KDim2( dim2.begin(), dim2.end() - 1 );
    std::vector<TestType> kernel2 = Randovalues<TestType>( KDim2, 0.001, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> KDim3( dim3.begin(), dim3.end() - 1 );
    std::vector<TestType> kernel3 = Randovalues<TestType>( KDim3, 0.001, std::numeric_limits<TestType>::max() / 2 );

    auto expectedValues01 = BroadcastMap<TestType>( values1, Kernel1, []( TestType x, TestType y ) { return x < y; } );
    auto expectedValues02 = BroadcastMap<TestType>( values2, kernel2, []( TestType x, TestType y ) { return x < y; } );
    auto expectedValues03 = BroadcastMap<TestType>( values3, kernel3, []( TestType x, TestType y ) { return x < y; } );

    auto expectedValues11 = BroadcastMap<TestType>( Kernel1, values1, []( TestType x, TestType y ) { return x < y; } );
    auto expectedValues12 = BroadcastMap<TestType>( kernel2, values2, []( TestType x, TestType y ) { return x < y; } );
    auto expectedValues13 = BroadcastMap<TestType>( kernel3, values3, []( TestType x, TestType y ) { return x < y; } );

    std::vector<TestType> inputValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( values1 ), ConcatenateVectors( values2 ), ConcatenateVectors( values3 ) } );

    std::vector<TestType> kernelValues = ConcatenateVectors( std::vector<std::vector<TestType>>{ Kernel1, kernel2, kernel3 } );

    data_initializer_t inputInitializer( inputValues );
    auto inputTensor = MultiTensorValue( scope, inputInitializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( TestType ) ) );

    data_initializer_t kernelInitializer( kernelValues );
    auto kernelensor = MultiTensorValue( scope, kernelInitializer, tensor_shape_t( { KDim1, KDim2, KDim3 }, sizeof( TestType ) ) );

    auto result0 = LessThan( scope, inputTensor, kernelensor );
    auto result1 = LessThan( scope, kernelensor, inputTensor );
    scope.Run( { result0, result1 } );

    auto outputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( outputShape.CountLayers() == 3 );
    REQUIRE( outputShape.Rank == 3 );
    REQUIRE( outputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( outputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( outputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<TestType> expectedValues00 = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( expectedValues01 ), ConcatenateVectors( expectedValues02 ), ConcatenateVectors( expectedValues03 ) } );
    std::vector<uint8_t>  expectedValues0{};
    for( auto x : expectedValues00 )
        expectedValues0.push_back( static_cast<uint8_t>( x != 0 ) );

    std::vector<TestType> expectedValues10 = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( expectedValues11 ), ConcatenateVectors( expectedValues12 ), ConcatenateVectors( expectedValues13 ) } );
    std::vector<uint8_t>  expectedValues1{};
    for( auto x : expectedValues10 )
        expectedValues1.push_back( static_cast<uint8_t>( x != 0 ) );

    std::vector<uint8_t> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    REQUIRE( resultValues0 == expectedValues0 );

    std::vector<uint8_t> resultValues1 = result1.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    REQUIRE( resultValues1 == expectedValues1 );
}

TEMPLATE_TEST_CASE( "LessThanOrEqual broadcast", "[CORE_COMPUTATION_GRAPH]", uint16_t, uint32_t, uint64_t, int16_t, int32_t, int64_t,
                    float, double )
{

    size_t  poolSize = 3 * 1024 * 1024;
    scope_t scope( poolSize );

    std::vector<uint32_t>              dim1{ 7, 3, 124 };
    std::vector<std::vector<TestType>> values1 = RandomVector<TestType>( dim1, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              dim2{ 2, 7, 75 };
    std::vector<std::vector<TestType>> values2 = RandomVector<TestType>( dim2, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t>              dim3{ 3, 5, 23 };
    std::vector<std::vector<TestType>> values3 = RandomVector<TestType>( dim3, 0, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> KDim1( dim1.begin(), dim1.end() - 1 );
    std::vector<TestType> Kernel1 = Randovalues<TestType>( KDim1, 0.001, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> KDim2( dim2.begin(), dim2.end() - 1 );
    std::vector<TestType> kernel2 = Randovalues<TestType>( KDim2, 0.001, std::numeric_limits<TestType>::max() / 2 );

    std::vector<uint32_t> KDim3( dim3.begin(), dim3.end() - 1 );
    std::vector<TestType> kernel3 = Randovalues<TestType>( KDim3, 0.001, std::numeric_limits<TestType>::max() / 2 );

    auto expectedValues01 = BroadcastMap<TestType>( values1, Kernel1, []( TestType x, TestType y ) { return x <= y; } );
    auto expectedValues02 = BroadcastMap<TestType>( values2, kernel2, []( TestType x, TestType y ) { return x <= y; } );
    auto expectedValues03 = BroadcastMap<TestType>( values3, kernel3, []( TestType x, TestType y ) { return x <= y; } );

    auto expectedValues11 = BroadcastMap<TestType>( Kernel1, values1, []( TestType x, TestType y ) { return x <= y; } );
    auto expectedValues12 = BroadcastMap<TestType>( kernel2, values2, []( TestType x, TestType y ) { return x <= y; } );
    auto expectedValues13 = BroadcastMap<TestType>( kernel3, values3, []( TestType x, TestType y ) { return x <= y; } );

    std::vector<TestType> inputValues = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( values1 ), ConcatenateVectors( values2 ), ConcatenateVectors( values3 ) } );

    std::vector<TestType> kernelValues = ConcatenateVectors( std::vector<std::vector<TestType>>{ Kernel1, kernel2, kernel3 } );

    data_initializer_t inputInitializer( inputValues );
    auto inputTensor = MultiTensorValue( scope, inputInitializer, tensor_shape_t( { dim1, dim2, dim3 }, sizeof( TestType ) ) );

    data_initializer_t kernelInitializer( kernelValues );
    auto kernelensor = MultiTensorValue( scope, kernelInitializer, tensor_shape_t( { KDim1, KDim2, KDim3 }, sizeof( TestType ) ) );

    auto result0 = LessThanOrEqual( scope, inputTensor, kernelensor );
    auto result1 = LessThanOrEqual( scope, kernelensor, inputTensor );
    scope.Run( { result0, result1 } );

    auto outputShape = result0.Get<multi_tensor_value_t>().Shape();
    REQUIRE( outputShape.CountLayers() == 3 );
    REQUIRE( outputShape.Rank == 3 );
    REQUIRE( outputShape.Shape[0] == std::vector<uint32_t>{ 7, 3, 124 } );
    REQUIRE( outputShape.Shape[1] == std::vector<uint32_t>{ 2, 7, 75 } );
    REQUIRE( outputShape.Shape[2] == std::vector<uint32_t>{ 3, 5, 23 } );

    std::vector<TestType> expectedValues00 = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( expectedValues01 ), ConcatenateVectors( expectedValues02 ), ConcatenateVectors( expectedValues03 ) } );
    std::vector<uint8_t>  expectedValues0{};
    for( auto x : expectedValues00 )
        expectedValues0.push_back( static_cast<uint8_t>( x != 0 ) );

    std::vector<TestType> expectedValues10 = ConcatenateVectors( std::vector<std::vector<TestType>>{
        ConcatenateVectors( expectedValues11 ), ConcatenateVectors( expectedValues12 ), ConcatenateVectors( expectedValues13 ) } );
    std::vector<uint8_t>  expectedValues1{};
    for( auto x : expectedValues10 )
        expectedValues1.push_back( static_cast<uint8_t>( x != 0 ) );

    std::vector<uint8_t> resultValues0 = result0.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    REQUIRE( resultValues0 == expectedValues0 );

    std::vector<uint8_t> resultValues1 = result1.Get<multi_tensor_value_t>().value.FetchFlattened<uint8_t>();
    REQUIRE( resultValues1 == expectedValues1 );
}
