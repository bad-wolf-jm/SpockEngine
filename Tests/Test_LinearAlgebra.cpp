
#include <catch2/catch_test_macros.hpp>

#include "Core/LinearAlgebra/vector.h"

using namespace numlua;
using namespace numlua::core;
using namespace numlua::linalg;

TEST_CASE( "2D Vectors", "[LINEAR_ALGEBRA]" )
{
    auto const v0 = linalg::float2( 1.0f );
    REQUIRE( ( v0.x == 1.0f && v0.y == 1.0f ) );

    auto const v1 = linalg::float2( 1.0f, 2.0f );
    REQUIRE( ( v1.x == 1.0f && v1.y == 2.0f ) );
    REQUIRE( ( v1[0] == 1.0f && v1[1] == 2.0f ) );

    auto v3 = v1;
    REQUIRE( ( v3.x == 1.0f && v1.y == 2.0f ) );
}

TEST_CASE( "3D Vectors", "[LINEAR_ALGEBRA]" )
{
    auto const vx = linalg::float2( 1.0f, 2.0f );

    auto const v0 = linalg::float3( 1.0f );
    REQUIRE( ( v0.x == 1.0f && v0.y == 1.0f & v0.z == 1.0f ) );

    auto const v1 = linalg::float3( vx, 3.0f );
    REQUIRE( ( v1.x == 1.0f && v1.y == 2.0f & v1.z == 3.0f ) );

    auto const v2 = linalg::float3( 3.0f, vx );
    REQUIRE( ( v2.x == 3.0f && v2.y == 1.0f & v2.z == 2.0f ) );

    auto const v3 = linalg::float3( 1.0f, 2.0f, 3.0f );
    REQUIRE( ( v3.x == 1.0f && v3.y == 2.0f & v3.z == 3.0f ) );
    REQUIRE( ( v3[0] == 1.0f && v3[1] == 2.0f & v3[2] == 3.0f ) );

    auto v4 = v3;
    REQUIRE( ( v4.x == 1.0f && v4.y == 2.0f & v4.z == 3.0f ) );
}

TEST_CASE( "4D Vectors", "[LINEAR_ALGEBRA]" )
{
    auto const v0 = linalg::float4( 1.0f );
    REQUIRE( ( v0.x == 1.0f && v0.y == 1.0f & v0.z == 1.0f & v0.z == 1.0f ) );

    auto const v1 = linalg::float4( 1.0f, 2.0f, 3.0f, 4.0f );
    REQUIRE( ( v1.x == 1.0f && v1.y == 2.0f & v1.z == 3.0f & v1.w == 4.0f ) );
    REQUIRE( ( v1[0] == 1.0f && v1[1] == 2.0f & v1[2] == 3.0f & v1[3] == 4.0f ) );

    auto const vx = linalg::float2( 1.0f, 2.0f );

    auto const v2 = linalg::float4( vx, 3.0f, 4.0f );
    REQUIRE( ( v2.x == 1.0f && v2.y == 2.0f & v2.z == 3.0f & v2.w == 4.0f ) );

    auto const v3 = linalg::float4( 1.0f, vx, 2.0f );
    REQUIRE( ( v3.x == 1.0f && v3.y == 1.0f & v3.z == 2.0f & v3.w == 2.0f ) );

    auto const v4 = linalg::float4( 1.0f, 2.0f, vx );
    REQUIRE( ( v4.x == 1.0f && v4.y == 2.0f & v4.z == 1.0f & v4.w == 2.0f ) );

    auto const vy = linalg::float3( 1.0f, 2.0f, 3.0f );

    auto const v5 = linalg::float4( vy, 4.0f );
    REQUIRE( ( v5.x == 1.0f && v5.y == 2.0f & v5.z == 3.0f & v5.w == 4.0f ) );

    auto const v6 = linalg::float4( 1.0f, vy );
    REQUIRE( ( v6.x == 1.0f && v6.y == 1.0f & v6.z == 2.0f & v6.w == 3.0f ) );

    auto v7 = v6;
    REQUIRE( ( v7.x == 1.0f && v7.y == 1.0f & v7.z == 2.0f & v7.w == 3.0f ) );
}

TEST_CASE( "Vector addition", "[LINEAR_ALGEBRA]" )
{
    constexpr float scalarConstant = 4.5f;

    {
        auto const v1 = linalg::float2( 1.0f, 2.0f );
        auto const v2 = linalg::float2( 2.0f, 3.0f );
        auto       v3 = v1 + v2;
        auto       v4 = v1 + scalarConstant;
        auto       v5 = scalarConstant + v1;
        auto       v6 = +v3;

        // clang-format off
        REQUIRE( ( (v3.x == v1.x + v2.x)           && (v3.y == v1.y + v2.y)           ) );
        REQUIRE( ( (v4.x == v1.x + scalarConstant) && (v4.y == v1.y + scalarConstant) ) );
        REQUIRE( ( (v5.x == v1.x + scalarConstant) && (v5.y == v1.y + scalarConstant) ) );
        REQUIRE( ( v6.x == v3.x                    && v6.y == v3.y ) );
        // clang-format on
    }

    {
        auto const v1 = linalg::float3( 1.0f, 2.0f, 3.0f );
        auto const v2 = linalg::float3( 2.0f, 3.0f, 4.0f );
        auto       v3 = v1 + v2;
        auto       v4 = v1 + scalarConstant;
        auto       v5 = scalarConstant + v1;
        auto       v6 = +v3;

        // clang-format off
        REQUIRE( ( (v3.x == v1.x + v2.x)           && (v3.y == v1.y + v2.y)           && (v3.z == v1.z + v2.z)           ) );
        REQUIRE( ( (v4.x == v1.x + scalarConstant) && (v4.y == v1.y + scalarConstant) && (v4.z == v1.z + scalarConstant) ) );
        REQUIRE( ( (v5.x == v1.x + scalarConstant) && (v5.y == v1.y + scalarConstant) && (v5.z == v1.z + scalarConstant) ) );
        REQUIRE( ( v6.x == v3.x                    && v6.y == v3.y                    && v6.z == v3.z ) );
        // clang-format on
    }

    {
        auto const v1 = linalg::float4( 1.0f, 2.0f, 3.0f, 4.0f );
        auto const v2 = linalg::float4( 2.0f, 3.0f, 4.0f, 5.0f );
        auto       v3 = v1 + v2;
        auto       v4 = v1 + scalarConstant;
        auto       v5 = scalarConstant + v1;
        auto       v6 = +v3;

        // clang-format off
        REQUIRE( ( (v3.x == v1.x + v2.x)           && (v3.y == v1.y + v2.y)           && (v3.z == v1.z + v2.z)           && (v3.w == v1.w + v2.w) ) );
        REQUIRE( ( (v4.x == v1.x + scalarConstant) && (v4.y == v1.y + scalarConstant) && (v4.z == v1.z + scalarConstant) && (v4.w == v1.w + scalarConstant) ) );
        REQUIRE( ( (v5.x == v1.x + scalarConstant) && (v5.y == v1.y + scalarConstant) && (v5.z == v1.z + scalarConstant) && (v5.w == v1.w + scalarConstant) ) );
        REQUIRE( ( v6.x == v3.x                    && v6.y == v3.y                    && v6.z == v3.z                    && v6.w == v3.w ) );
        // clang-format on
    }
}

TEST_CASE( "Vector multiplication", "[LINEAR_ALGEBRA]" )
{
    constexpr float scalarConstant = 4.5f;

    {
        auto const v1 = linalg::float2( 1.0f, 2.0f );
        auto const v2 = linalg::float2( 2.0f, 3.0f );
        auto       v3 = v1 * v2;
        auto       v4 = v1 * scalarConstant;
        auto       v5 = scalarConstant * v1;

        // clang-format off
        REQUIRE( ( (v3.x == v1.x * v2.x)           && (v3.y == v1.y * v2.y)           ) );
        REQUIRE( ( (v4.x == v1.x * scalarConstant) && (v4.y == v1.y * scalarConstant) ) );
        REQUIRE( ( (v5.x == v1.x * scalarConstant) && (v5.y == v1.y * scalarConstant) ) );
        // clang-format on
    }

    {
        auto const v1 = linalg::float3( 1.0f, 2.0f, 3.0f );
        auto const v2 = linalg::float3( 2.0f, 3.0f, 4.0f );
        auto       v3 = v1 * v2;
        auto       v4 = v1 * scalarConstant;
        auto       v5 = scalarConstant * v1;

        // clang-format off
        REQUIRE( ( (v3.x == v1.x * v2.x)           && (v3.y == v1.y * v2.y)           && (v3.z == v1.z * v2.z)           ) );
        REQUIRE( ( (v4.x == v1.x * scalarConstant) && (v4.y == v1.y * scalarConstant) && (v4.z == v1.z * scalarConstant) ) );
        REQUIRE( ( (v5.x == v1.x * scalarConstant) && (v5.y == v1.y * scalarConstant) && (v5.z == v1.z * scalarConstant) ) );
        // clang-format on
    }

    {
        auto const v1 = linalg::float4( 1.0f, 2.0f, 3.0f, 4.0f );
        auto const v2 = linalg::float4( 2.0f, 3.0f, 4.0f, 5.0f );
        auto       v3 = v1 * v2;
        auto       v4 = v1 * scalarConstant;
        auto       v5 = scalarConstant * v1;

        // clang-format off
        REQUIRE( ( (v3.x == v1.x * v2.x)           && (v3.y == v1.y * v2.y)           && (v3.z == v1.z * v2.z)           && (v3.w == v1.w * v2.w) ) );
        REQUIRE( ( (v4.x == v1.x * scalarConstant) && (v4.y == v1.y * scalarConstant) && (v4.z == v1.z * scalarConstant) && (v4.w == v1.w * scalarConstant) ) );
        REQUIRE( ( (v5.x == v1.x * scalarConstant) && (v5.y == v1.y * scalarConstant) && (v5.z == v1.z * scalarConstant) && (v5.w == v1.w * scalarConstant) ) );
        // clang-format on
    }
}
