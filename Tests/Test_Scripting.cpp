#include <catch2/catch_test_macros.hpp>

#include "TestUtils.h"

#include "Core/CUDA/Texture/TextureTypes.h"
#include "Core/Math/Types.h"

#include "Core/CUDA/Array/MultiTensor.h"
#include "Core/CUDA/Texture/Texture2D.h"

#include "TensorOps/Scope.h"

#include "Scripting/ArrayTypes.h"
#include "Scripting/Core/Vector.h"
#include "Scripting/ScriptingEngine.h"

#define ENTT_DISABLE_ASSERT
#include "Core/Entity/Collection.h"

using namespace math;
using namespace SE::Core;
using namespace SE::Cuda;
using namespace SE::TensorOps;
using namespace TestUtils;

TEST_CASE( "LUA Arrays", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    scriptingEngine.Execute( R"(
value = Core.U8Array()
value:append(123)
len0 = value:length()
len1 = #value
)" );
    {
        auto x = scriptingEngine.Get<u8_array_t>( "value" );
        REQUIRE( x.Length() == 1 );
        REQUIRE( x.mArray[0] == 123 );

        auto l0 = scriptingEngine.Get<uint32_t>( "len0" );
        auto l1 = scriptingEngine.Get<uint32_t>( "len1" );
        REQUIRE( ( ( l0 == 1 ) && ( l1 == 1 ) ) );
    }

    scriptingEngine.Execute( R"(
value0 = Core.U8Array()
value1 = Core.U8Array()
value0:append(123)
value1:append(211)
value0:append(value1)
)" );
    {
        auto x = scriptingEngine.Get<u8_array_t>( "value0" );
        REQUIRE( x.Length() == 2 );
        REQUIRE( ( ( x.mArray[0] == 123 ) && ( x.mArray[1] == 211 ) ) );
    }

    scriptingEngine.Execute( R"(
value0 = Core.U8Array{1, 2, 3, 4, 5, 6, 7, 8, 9}
)" );
    {
        auto x = scriptingEngine.Get<u8_array_t>( "value0" );
        REQUIRE( x.mArray == std::vector<uint8_t>{ 1, 2, 3, 4, 5, 6, 7, 8, 9 } );
    }

    scriptingEngine.Execute( R"(
value0 = Core.U8Array{1, 2, 3, 4, 5, 6, 7, 8, 9}
value1 = Core.U8Array{10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
value0:append(value1)
)" );
    {
        auto x = scriptingEngine.Get<u8_array_t>( "value0" );
        REQUIRE( x.mArray == std::vector<uint8_t>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 } );
    }
}

TEST_CASE( "LUA Vectors", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    scriptingEngine.Execute( R"(
value = Core.Array(dtypes.uint32, 6)
)" );
    {
        auto x = scriptingEngine.Get<std::vector<uint32_t>>( "value" );
        REQUIRE( x.size() == 6 );
    }

    scriptingEngine.Execute( R"(
value = Core.Array(dtypes.uint32, 6)
value[1] = 3
)" );
    {
        auto x = scriptingEngine.Get<std::vector<uint32_t>>( "value" );
        REQUIRE( x[0] == 3 );
    }
}

TEST_CASE( "LUA Random", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    scriptingEngine.Execute( R"(
value = Core.Random(dtypes.uint32, 25, 1, 128)
)" );
    {
        auto x = scriptingEngine.Get<std::vector<uint32_t>>( "value" );
        REQUIRE( x.size() == 25 );
    }
}

TEST_CASE( "LUA Vec2 type", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    scriptingEngine.Execute( "value = Math.vec2(1.0, 2.0)" );
    REQUIRE( scriptingEngine.Get<vec2>( "value" ) == vec2{ 1.0f, 2.0f } );

    scriptingEngine.Execute( "value = Math.vec2{1.0, 2.0}" );
    REQUIRE( scriptingEngine.Get<vec2>( "value" ) == vec2{ 1.0f, 2.0f } );

    scriptingEngine.Execute( "value = Math.vec2{ x=1.0, y=2.0 }" );
    REQUIRE( scriptingEngine.Get<vec2>( "value" ) == vec2{ 1.0f, 2.0f } );

    scriptingEngine.Execute( R"(
value = Math.vec2(0.0, 0.0)
value.x = 3.0
value.y = 4.0
)" );
    REQUIRE( scriptingEngine.Get<vec2>( "value" ) == vec2{ 3.0f, 4.0f } );

    scriptingEngine.Execute( R"(
value0 = Math.vec2(1.0, 2.0)
value1 = Math.vec2(0.5, .25)
value = value0 + value1
)" );
    REQUIRE( scriptingEngine.Get<vec2>( "value" ) == vec2{ 1.5f, 2.25f } );

    scriptingEngine.Execute( "value = Math.vec2(1.0, 2.0) * 2.0" );
    REQUIRE( scriptingEngine.Get<vec2>( "value" ) == vec2{ 2.0f, 4.0f } );

    scriptingEngine.Execute( "value = 2.0 * Math.vec2(1.0, 2.0)" );
    REQUIRE( scriptingEngine.Get<vec2>( "value" ) == vec2{ 2.0f, 4.0f } );

    scriptingEngine.Execute( "value = Math.vec2(1.0, 2.0):normalize()" );
    REQUIRE( scriptingEngine.Get<vec2>( "value" ) == normalize( vec2{ 1.0f, 2.0f } ) );

    scriptingEngine.Execute( "value = Math.vec2(1.0, 2.0):length()" );
    REQUIRE( scriptingEngine.Get<float>( "value" ) == length( vec2{ 1.0f, 2.0f } ) );

    scriptingEngine.Execute( "value = Math.vec2(1.0, 2.0):length2()" );
    REQUIRE( scriptingEngine.Get<float>( "value" ) == length2( vec2{ 1.0f, 2.0f } ) );

    scriptingEngine.Execute( R"(
value0 = Math.vec2(1.0, 2.0)
value1 = Math.vec2(3.0, 6.0)
value = value0:dot(value1)
)" );
    REQUIRE( scriptingEngine.Get<float>( "value" ) == dot( vec2{ 1.0f, 2.0f }, vec2{ 3.0f, 6.0f } ) );
}

TEST_CASE( "LUA Vec3 type", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    scriptingEngine.Execute( "value = Math.vec3(1.0, 2.0, 3.0)" );
    REQUIRE( scriptingEngine.Get<vec3>( "value" ) == vec3{ 1.0f, 2.0f, 3.0f } );

    scriptingEngine.Execute( "value = Math.vec3{1.0, 2.0, 3.0}" );
    REQUIRE( scriptingEngine.Get<vec3>( "value" ) == vec3{ 1.0f, 2.0f, 3.0f } );

    scriptingEngine.Execute( "value = Math.vec3{ x=1.0, y=2.0, z=3.0 }" );
    REQUIRE( scriptingEngine.Get<vec3>( "value" ) == vec3{ 1.0f, 2.0f, 3.0f } );

    scriptingEngine.Execute( R"(
value = Math.vec3(0.0, 0.0, 0.0)
value.x = 3.0
value.y = 4.0
value.z = 1.0
)" );
    REQUIRE( scriptingEngine.Get<vec3>( "value" ) == vec3{ 3.0f, 4.0f, 1.0f } );

    scriptingEngine.Execute( R"(
value0 = Math.vec3(1.0, 2.0, 3.0)
value1 = Math.vec3(0.5, .25, .125)
value = value0 + value1
)" );
    REQUIRE( scriptingEngine.Get<vec3>( "value" ) == vec3{ 1.5f, 2.25f, 3.125f } );

    scriptingEngine.Execute( R"(
value = Math.vec3(1.0, 2.0, 3.0) * 2.0
)" );
    REQUIRE( scriptingEngine.Get<vec3>( "value" ) == vec3{ 2.0f, 4.0f, 6.0f } );

    scriptingEngine.Execute( "value = 2.0 * Math.vec3(1.0, 2.0, 3.0)" );
    REQUIRE( scriptingEngine.Get<vec3>( "value" ) == vec3{ 2.0f, 4.0f, 6.0f } );

    scriptingEngine.Execute( "value = Math.vec3(1.0, 2.0, 3.0):normalize()" );
    REQUIRE( scriptingEngine.Get<vec3>( "value" ) == normalize( vec3{ 1.0f, 2.0f, 3.0f } ) );

    scriptingEngine.Execute( "value = Math.vec3(1.0, 2.0, 3.0):length()" );
    REQUIRE( scriptingEngine.Get<float>( "value" ) == length( vec3{ 1.0f, 2.0f, 3.0f } ) );
    scriptingEngine.Execute( "value = Math.vec3(1.0, 2.0, 3.0):length2()" );
    REQUIRE( scriptingEngine.Get<float>( "value" ) == length2( vec3{ 1.0f, 2.0f, 3.0f } ) );

    scriptingEngine.Execute( R"(
value0 = Math.vec3(1.0, 2.0, 3.0)
value1 = Math.vec3(3.0, 6.0, 9.0)
value = value0:dot(value1)
)" );
    REQUIRE( scriptingEngine.Get<float>( "value" ) == dot( vec3{ 1.0f, 2.0f, 3.0f }, vec3{ 3.0f, 6.0f, 9.0f } ) );

    scriptingEngine.Execute( R"(
value0 = Math.vec3(1.0, 2.0, 3.0)
value1 = Math.vec3(3.0, 1.0, 2.0)
value = value0:cross(value1)
)" );
    REQUIRE( scriptingEngine.Get<vec3>( "value" ) == cross( vec3{ 1.0f, 2.0f, 3.0f }, vec3{ 3.0f, 1.0f, 2.0f } ) );
}

TEST_CASE( "LUA Vec4 type", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    scriptingEngine.Execute( "value = Math.vec4(1.0, 2.0, 3.0, 4.0)" );
    REQUIRE( scriptingEngine.Get<vec4>( "value" ) == vec4{ 1.0f, 2.0f, 3.0f, 4.0f } );

    scriptingEngine.Execute( "value = Math.vec4{1.0, 2.0, 3.0, 4.0}" );
    REQUIRE( scriptingEngine.Get<vec4>( "value" ) == vec4{ 1.0f, 2.0f, 3.0f, 4.0f } );

    scriptingEngine.Execute( "value = Math.vec4{ x=1.0, y=2.0, z=3.0, w=4.0 }" );
    REQUIRE( scriptingEngine.Get<vec4>( "value" ) == vec4{ 1.0f, 2.0f, 3.0f, 4.0f } );

    scriptingEngine.Execute( R"(
value = Math.vec4(0.0, 0.0, 0.0, 0.0)
value.x = 3.0
value.y = 4.0
value.z = 1.0
value.w = 2.0
)" );
    REQUIRE( scriptingEngine.Get<vec4>( "value" ) == vec4{ 3.0f, 4.0f, 1.0f, 2.0f } );

    scriptingEngine.Execute( R"(
value0 = Math.vec4(1.0, 2.0, 3.0, 4.0)
value1 = Math.vec4(0.5, .25, .125, .225)
value = value0 + value1
)" );
    REQUIRE( scriptingEngine.Get<vec4>( "value" ) == vec4{ 1.5f, 2.25f, 3.125f, 4.225f } );

    scriptingEngine.Execute( "value = Math.vec4(1.0, 2.0, 3.0, 4.0) * 2.0" );
    REQUIRE( scriptingEngine.Get<vec4>( "value" ) == vec4{ 2.0f, 4.0f, 6.0f, 8.0f } );

    scriptingEngine.Execute( "value = 2.0 * Math.vec4(1.0, 2.0, 3.0, 4.0)" );
    REQUIRE( scriptingEngine.Get<vec4>( "value" ) == vec4{ 2.0f, 4.0f, 6.0f, 8.0f } );

    scriptingEngine.Execute( "value = Math.vec4(1.0, 2.0, 3.0, 4.0):normalize()" );
    REQUIRE( scriptingEngine.Get<vec4>( "value" ) == normalize( vec4{ 1.0f, 2.0f, 3.0f, 4.0f } ) );

    scriptingEngine.Execute( "value = Math.vec4(1.0, 2.0, 3.0, 4.0):length()" );
    REQUIRE( scriptingEngine.Get<float>( "value" ) == length( vec4{ 1.0f, 2.0f, 3.0f, 4.0f } ) );

    scriptingEngine.Execute( "value = Math.vec4(1.0, 2.0, 3.0, 4.0):length2()" );
    REQUIRE( scriptingEngine.Get<float>( "value" ) == length2( vec4{ 1.0f, 2.0f, 3.0f, 4.0f } ) );

    scriptingEngine.Execute( R"(
value0 = Math.vec4(1.0, 2.0, 3.0, 1.2)
value1 = Math.vec4(3.0, 6.0, 9.0, 1.25)
value = value0:dot(value1)
)" );
    REQUIRE( scriptingEngine.Get<float>( "value" ) == dot( vec4{ 1.0f, 2.0f, 3.0f, 1.2f }, vec4{ 3.0f, 6.0f, 9.0f, 1.25f } ) );
}

TEST_CASE( "LUA iVec2 type", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    scriptingEngine.Execute( "value = Math.ivec2(1, 2)" );
    REQUIRE( scriptingEngine.Get<ivec2>( "value" ) == ivec2{ 1, 2 } );

    scriptingEngine.Execute( R"(
value = Math.ivec2(0, 0)
value.x = 3
value.y = 4
)" );
    REQUIRE( scriptingEngine.Get<ivec2>( "value" ) == ivec2{ 3, 4 } );

    scriptingEngine.Execute( R"(
value0 = Math.ivec2(1, 2)
value1 = Math.ivec2(3, 4)
value = value0 + value1
)" );
    REQUIRE( scriptingEngine.Get<ivec2>( "value" ) == ivec2{ 4, 6 } );

    scriptingEngine.Execute( R"(
value = Math.ivec2(1, 2) * 2
)" );
    REQUIRE( scriptingEngine.Get<ivec2>( "value" ) == ivec2{ 2, 4 } );

    scriptingEngine.Execute( "value = 2 * Math.ivec2(1, 2)" );
    REQUIRE( scriptingEngine.Get<ivec2>( "value" ) == ivec2{ 2, 4 } );
}

TEST_CASE( "LUA iVec3 type", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    scriptingEngine.Execute( "value = Math.ivec3(1, 2, 3)" );
    REQUIRE( scriptingEngine.Get<ivec3>( "value" ) == ivec3{ 1, 2, 3 } );

    scriptingEngine.Execute( R"(
value = Math.ivec3(0, 0, 0)
value.x = 3
value.y = 4
value.z = 1
)" );
    REQUIRE( scriptingEngine.Get<ivec3>( "value" ) == ivec3{ 3, 4, 1 } );

    scriptingEngine.Execute( R"(
value0 = Math.ivec3(1, 2, 3)
value1 = Math.ivec3(3, 4, 5)
value = value0 + value1
)" );
    REQUIRE( scriptingEngine.Get<ivec3>( "value" ) == ivec3{ 4, 6, 8 } );

    scriptingEngine.Execute( "value = Math.ivec3(1, 2, 3) * 2" );
    REQUIRE( scriptingEngine.Get<ivec3>( "value" ) == ivec3{ 2, 4, 6 } );

    scriptingEngine.Execute( R"(
value = 2 * Math.ivec3(1, 2, 3)
)" );
    REQUIRE( scriptingEngine.Get<ivec3>( "value" ) == ivec3{ 2, 4, 6 } );
}

TEST_CASE( "LUA iVec4 type", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    scriptingEngine.Execute( "value = Math.ivec4(1, 2, 3, 4)" );
    REQUIRE( scriptingEngine.Get<ivec4>( "value" ) == ivec4{ 1, 2, 3, 4 } );

    scriptingEngine.Execute( R"(
value = Math.ivec4(0, 0, 0, 4)
value.x = 3
value.y = 4
value.z = 1
value.w = 2
)" );
    REQUIRE( scriptingEngine.Get<ivec4>( "value" ) == ivec4{ 3, 4, 1, 2 } );

    scriptingEngine.Execute( R"(
value0 = Math.ivec4(1, 2, 3, 4)
value1 = Math.ivec4(3, 4, 5, 6)
value = value0 + value1
)" );
    REQUIRE( scriptingEngine.Get<ivec4>( "value" ) == ivec4{ 4, 6, 8, 10 } );

    scriptingEngine.Execute( "value = Math.ivec4(1, 2, 3, 4) * 2" );
    REQUIRE( scriptingEngine.Get<ivec4>( "value" ) == ivec4{ 2, 4, 6, 8 } );

    scriptingEngine.Execute( "value = 2 *Math.ivec4(1, 2, 3, 4)" );
    REQUIRE( scriptingEngine.Get<ivec4>( "value" ) == ivec4{ 2, 4, 6, 8 } );
}

TEST_CASE( "LUA uVec2 type", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    scriptingEngine.Execute( "value = Math.uvec2(1, 2)" );
    REQUIRE( scriptingEngine.Get<uvec2>( "value" ) == uvec2{ 1, 2 } );

    scriptingEngine.Execute( R"(
value = Math.uvec2(0, 0)
value.x = 3
value.y = 4
)" );
    REQUIRE( scriptingEngine.Get<uvec2>( "value" ) == uvec2{ 3, 4 } );

    scriptingEngine.Execute( R"(
value0 = Math.uvec2(1, 2)
value1 = Math.uvec2(3, 4)
value = value0 + value1
)" );
    REQUIRE( scriptingEngine.Get<uvec2>( "value" ) == uvec2{ 4, 6 } );

    scriptingEngine.Execute( "value = Math.uvec2(1, 2) * 2" );
    REQUIRE( scriptingEngine.Get<uvec2>( "value" ) == uvec2{ 2, 4 } );

    scriptingEngine.Execute( R"(
value = 2 * Math.uvec2(1, 2)
)" );
    REQUIRE( scriptingEngine.Get<uvec2>( "value" ) == uvec2{ 2, 4 } );
}

TEST_CASE( "LUA uVec3 type", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    scriptingEngine.Execute( "value = Math.uvec3(1, 2, 3)" );
    REQUIRE( scriptingEngine.Get<uvec3>( "value" ) == uvec3{ 1, 2, 3 } );

    scriptingEngine.Execute( R"(
value = Math.uvec3(0, 0, 0)
value.x = 3
value.y = 4
value.z = 1
)" );
    REQUIRE( scriptingEngine.Get<uvec3>( "value" ) == uvec3{ 3, 4, 1 } );

    scriptingEngine.Execute( R"(
value0 = Math.uvec3(1, 2, 3)
value1 = Math.uvec3(3, 4, 5)
value = value0 + value1
)" );
    REQUIRE( scriptingEngine.Get<uvec3>( "value" ) == uvec3{ 4, 6, 8 } );

    scriptingEngine.Execute( R"(
value = Math.uvec3(1, 2, 3) * 2
)" );
    REQUIRE( scriptingEngine.Get<uvec3>( "value" ) == uvec3{ 2, 4, 6 } );

    scriptingEngine.Execute( "value = 2 * Math.uvec3(1, 2, 3)" );
    REQUIRE( scriptingEngine.Get<uvec3>( "value" ) == uvec3{ 2, 4, 6 } );
}

TEST_CASE( "LUA uVec4 type", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    scriptingEngine.Execute( "value = Math.uvec4(1, 2, 3, 4)" );
    REQUIRE( scriptingEngine.Get<uvec4>( "value" ) == uvec4{ 1, 2, 3, 4 } );

    scriptingEngine.Execute( R"(
value = Math.uvec4(0, 0, 0, 4)
value.x = 3
value.y = 4
value.z = 1
value.w = 2
)" );
    REQUIRE( scriptingEngine.Get<uvec4>( "value" ) == uvec4{ 3, 4, 1, 2 } );

    scriptingEngine.Execute( R"(
value0 = Math.uvec4(1, 2, 3, 4)
value1 = Math.uvec4(3, 4, 5, 6)
value = value0 + value1
)" );
    REQUIRE( scriptingEngine.Get<uvec4>( "value" ) == uvec4{ 4, 6, 8, 10 } );

    scriptingEngine.Execute( "value = Math.uvec4(1, 2, 3, 4) * 2" );
    REQUIRE( scriptingEngine.Get<uvec4>( "value" ) == uvec4{ 2, 4, 6, 8 } );

    scriptingEngine.Execute( "value = 2 * Math.uvec4(1, 2, 3, 4)" );
    REQUIRE( scriptingEngine.Get<uvec4>( "value" ) == uvec4{ 2, 4, 6, 8 } );
}

TEST_CASE( "LUA Mat3 type", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    scriptingEngine.Execute( "value = Math.mat3()" );
    REQUIRE( scriptingEngine.Get<mat3>( "value" ) == mat3{ 1.0 } );

    scriptingEngine.Execute( "value = Math.mat3(4.0)" );
    REQUIRE( scriptingEngine.Get<mat3>( "value" ) == mat3{ 4.0 } );

    scriptingEngine.Execute( "value = Math.mat3(Math.vec3(1, 2, 3))" );
    REQUIRE( scriptingEngine.Get<mat3>( "value" ) == FromDiagonal( vec3{ 1, 2, 3 } ) );

    scriptingEngine.Execute( "value = Math.mat3(Math.vec3(1, 2, 3),Math.vec3(4, 5, 6), Math.vec3(7, 8, 9))" );
    REQUIRE( scriptingEngine.Get<mat3>( "value" ) == mat3( vec3{ 1, 2, 3 }, vec3{ 4, 5, 6 }, vec3{ 7, 8, 9 } ) );
}

TEST_CASE( "LUA Mat4 type", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    scriptingEngine.Execute( "value = Math.mat4()" );
    REQUIRE( scriptingEngine.Get<mat4>( "value" ) == mat4{ 1.0 } );

    scriptingEngine.Execute( "value = Math.mat4(4.0)" );
    REQUIRE( scriptingEngine.Get<mat4>( "value" ) == mat4{ 4.0 } );

    scriptingEngine.Execute( "value = Math.mat4(Math.vec4(1, 2, 3, 4))" );
    REQUIRE( scriptingEngine.Get<mat4>( "value" ) == FromDiagonal( vec4{ 1, 2, 3, 4 } ) );

    scriptingEngine.Execute(
        "value = Math.mat4(Math.vec4(1, 2, 3, 1),Math.vec4(4, 5, 6, 2), Math.vec4(7, 8, 9, 3), Math.vec4(10, 11, 12, 4))" );
    REQUIRE( scriptingEngine.Get<mat4>( "value" ) ==
             mat4( vec4{ 1, 2, 3, 1 }, vec4{ 4, 5, 6, 2 }, vec4{ 7, 8, 9, 3 }, vec4{ 10, 11, 12, 4 } ) );

    scriptingEngine.Execute(
        "value = Math.mat3(Math.mat4(Math.vec4(1, 2, 3, 1),Math.vec4(4, 5, 6, 2), Math.vec4(7, 8, 9, 3), Math.vec4(10, 11, 12, 4)))" );
    REQUIRE( scriptingEngine.Get<mat3>( "value" ) ==
             mat3( mat4( vec4{ 1, 2, 3, 1 }, vec4{ 4, 5, 6, 2 }, vec4{ 7, 8, 9, 3 }, vec4{ 10, 11, 12, 4 } ) ) );
}

TEST_CASE( "LUA Create registry", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    scriptingEngine.Execute( "value = EntityCollection.Registry.new()" );
    REQUIRE( true );
}

TEST_CASE( "LUA Create entity", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    scriptingEngine.Execute( R"(
registry = EntityCollection.Registry.new()
entity0 = registry:create_entity()
entity1 = registry:create_entity("NAME")
entity2 = registry:create_entity(entity0, "NAME_0")
)" );
    auto lEntity0 = scriptingEngine.Get<entity_t>( "entity0" );
    REQUIRE( lEntity0.IsValid() );

    auto lEntity1 = scriptingEngine.Get<entity_t>( "entity1" );
    REQUIRE( lEntity1.Has<tag_t>() );
    REQUIRE( lEntity1.Get<tag_t>().mValue == "NAME" );

    auto lEntity2 = scriptingEngine.Get<entity_t>( "entity2" );
    REQUIRE( lEntity0.Has<relationship_compoment_t>() );
    REQUIRE( lEntity2.Has<relationship_compoment_t>() );
    REQUIRE( lEntity2.Has<tag_t>() );
    REQUIRE( lEntity2.Get<tag_t>().mValue == "NAME_0" );

    REQUIRE( lEntity2.Get<relationship_compoment_t>().mParent == lEntity0 );
    REQUIRE( lEntity0.Get<relationship_compoment_t>().mChildren.size() == 1 );
    REQUIRE( lEntity0.Get<relationship_compoment_t>().mChildren[0] == lEntity2 );
}

TEST_CASE( "LUA create entity with relationship", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    scriptingEngine.Execute( R"(
registry = EntityCollection.Registry.new()
entity0 = registry:create_entity_with_relationship()
entity1 = registry:create_entity_with_relationship("NAME_0")
)" );
    auto lEntity0 = scriptingEngine.Get<entity_t>( "entity0" );
    REQUIRE( lEntity0.IsValid() );
    REQUIRE( lEntity0.Has<relationship_compoment_t>() );

    auto lEntity1 = scriptingEngine.Get<entity_t>( "entity1" );
    REQUIRE( lEntity1.IsValid() );
    REQUIRE( lEntity1.Has<relationship_compoment_t>() );
    REQUIRE( lEntity1.Has<tag_t>() );
    REQUIRE( lEntity1.Get<tag_t>().mValue == "NAME_0" );
}

TEST_CASE( "LUA Destroy entity", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    scriptingEngine.Execute( R"(
registry = EntityCollection.Registry.new()
entity0 = registry:create_entity("NAME")
registry:destroy_entity(entity0)
)" );
    auto lEntity0 = scriptingEngine.Get<entity_t>( "entity0" );
    REQUIRE( !lEntity0.IsValid() );
}

TEST_CASE( "LUA set parent entity", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    scriptingEngine.Execute( R"(
registry = EntityCollection.Registry.new()
entity0 = registry:create_entity()
entity1 = registry:create_entity()
registry:set_parent(entity1, entity0)
)" );

    auto lEntity0 = scriptingEngine.Get<entity_t>( "entity0" );
    auto lEntity1 = scriptingEngine.Get<entity_t>( "entity1" );

    REQUIRE( lEntity0.Has<relationship_compoment_t>() );
    REQUIRE( lEntity1.Has<relationship_compoment_t>() );

    REQUIRE( lEntity1.Get<relationship_compoment_t>().mParent == lEntity0 );
    REQUIRE( lEntity0.Get<relationship_compoment_t>().mChildren.size() == 1 );
    REQUIRE( lEntity0.Get<relationship_compoment_t>().mChildren[0] == lEntity1 );
}

TEST_CASE( "LUA Relationships", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};
    scriptingEngine.Execute( R"(
registry = EntityCollection.Registry.new()
entity0 = registry:create_entity()
entity1 = registry:create_entity()
entity2 = registry:create_entity()
entity3 = registry:create_entity()
registry:set_parent(entity1, entity0)
registry:set_parent(entity2, entity0)
registry:set_parent(entity3, entity0)
)" );

    auto lEntity0 = scriptingEngine.Get<entity_t>( "entity0" );
    auto lEntity1 = scriptingEngine.Get<entity_t>( "entity1" );
    auto lEntity2 = scriptingEngine.Get<entity_t>( "entity2" );
    auto lEntity3 = scriptingEngine.Get<entity_t>( "entity3" );

    REQUIRE( lEntity0.Get<relationship_compoment_t>().mChildren.size() == 3 );
    REQUIRE( lEntity1.Get<relationship_compoment_t>().mParent == lEntity0 );
    REQUIRE( lEntity2.Get<relationship_compoment_t>().mParent == lEntity0 );
    REQUIRE( lEntity3.Get<relationship_compoment_t>().mParent == lEntity0 );
}

TEST_CASE( "LUA Removing parent removes from siblings", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};
    scriptingEngine.Execute( R"(
registry = EntityCollection.Registry.new()
entity0 = registry:create_entity()
entity1 = registry:create_entity()
entity2 = registry:create_entity()
entity3 = registry:create_entity()
registry:set_parent(entity1, entity0)
registry:set_parent(entity2, entity0)
registry:set_parent(entity3, entity0)
registry:set_parent(entity3, entity2)
)" );

    auto lEntity0 = scriptingEngine.Get<entity_t>( "entity0" );
    auto lEntity1 = scriptingEngine.Get<entity_t>( "entity1" );
    auto lEntity2 = scriptingEngine.Get<entity_t>( "entity2" );
    auto lEntity3 = scriptingEngine.Get<entity_t>( "entity3" );

    REQUIRE( lEntity0.Get<relationship_compoment_t>().mChildren.size() == 2 );

    REQUIRE( lEntity1.Get<relationship_compoment_t>().mParent == lEntity0 );
    REQUIRE( lEntity2.Get<relationship_compoment_t>().mParent == lEntity0 );
    REQUIRE( lEntity3.Get<relationship_compoment_t>().mParent == lEntity2 );
}

TEST_CASE( "LUA Ability to set parent to NULL", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};
    scriptingEngine.Execute( R"(
registry = EntityCollection.Registry.new()
entity0 = registry:create_entity()
entity1 = registry:create_entity()
entity2 = registry:create_entity()
entity3 = registry:create_entity()
registry:set_parent(entity1, entity0)
registry:set_parent(entity2, entity0)
registry:set_parent(entity3, entity0)
registry:set_parent(entity3, EntityCollection.Entity.new())
)" );

    auto lEntity0 = scriptingEngine.Get<entity_t>( "entity0" );
    auto lEntity1 = scriptingEngine.Get<entity_t>( "entity1" );
    auto lEntity2 = scriptingEngine.Get<entity_t>( "entity2" );
    auto lEntity3 = scriptingEngine.Get<entity_t>( "entity3" );

    REQUIRE( lEntity0.Get<relationship_compoment_t>().mChildren.size() == 2 );
    REQUIRE( lEntity1.Get<relationship_compoment_t>().mParent == lEntity0 );
    REQUIRE( lEntity2.Get<relationship_compoment_t>().mParent == lEntity0 );
    REQUIRE( !( lEntity3.Get<relationship_compoment_t>().mParent ) );
}

struct ComponentA
{
    float a = 0.0f;

    ComponentA()                     = default;
    ComponentA( const ComponentA & ) = default;
};

TEST_CASE( "LUA add component", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    auto x                   = scriptingEngine.RegisterPrimitiveType<ComponentA>( "ComponentA" );
    x["a"]                   = &ComponentA::a;
    x[sol::call_constructor] = sol::factories( []() { return ComponentA{}; }, []( float x ) { return ComponentA{ x }; } );

    scriptingEngine.Execute( R"(
registry = EntityCollection.Registry.new()
entity0 = registry:create_entity()
entity0:add(dtypes.ComponentA())
entity1 = registry:create_entity()
entity1:add(dtypes.ComponentA(4.0))
)" );
    auto lEntity0 = scriptingEngine.Get<entity_t>( "entity0" );
    REQUIRE( lEntity0.Has<ComponentA>() );
    REQUIRE( lEntity0.Get<ComponentA>().a == ComponentA{}.a );

    auto lEntity1 = scriptingEngine.Get<entity_t>( "entity1" );
    REQUIRE( lEntity1.Has<ComponentA>() );
    REQUIRE( lEntity1.Get<ComponentA>().a == 4.0f );
}

TEST_CASE( "LUA remove component", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};
    auto            x        = scriptingEngine.RegisterPrimitiveType<ComponentA>( "ComponentA" );
    x["a"]                   = &ComponentA::a;
    x[sol::call_constructor] = sol::factories( []() { return ComponentA{}; }, []( float x ) { return ComponentA{ x }; } );

    scriptingEngine.Execute( R"(
registry = EntityCollection.Registry.new()
entity0 = registry:create_entity()
)" );

    auto lEntity0 = scriptingEngine.Get<entity_t>( "entity0" );
    lEntity0.Add<ComponentA>();

    scriptingEngine.Execute( "entity0:remove(dtypes.ComponentA)" );

    REQUIRE( !lEntity0.Has<ComponentA>() );
}

TEST_CASE( "LUA replace component", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};
    auto            x        = scriptingEngine.RegisterPrimitiveType<ComponentA>( "ComponentA" );
    x["a"]                   = &ComponentA::a;
    x[sol::call_constructor] = sol::factories( []() { return ComponentA{}; }, []( float x ) { return ComponentA{ x }; } );

    scriptingEngine.Execute( R"(
registry = EntityCollection.Registry.new()
entity0 = registry:create_entity()
)" );

    auto lEntity0 = scriptingEngine.Get<entity_t>( "entity0" );
    lEntity0.Add<ComponentA>();

    scriptingEngine.Execute( "entity0:replace(dtypes.ComponentA(4.0))" );

    REQUIRE( lEntity0.Get<ComponentA>().a == 4.0f );
}

TEST_CASE( "LUA test external registry", "[CORE_SCRIPTING]" )
{
    entity_registry_t lRegistry{};
    ScriptingEngine  scriptingEngine{};

    scriptingEngine.Define( "registry0", &lRegistry );
    scriptingEngine.Execute( "entity0 = registry0:create_entity()" );

    auto lEntity0 = scriptingEngine.Get<entity_t>( "entity0" );
    REQUIRE( lEntity0.IsValid() );
}

TEST_CASE( "LUA test external entity", "[CORE_SCRIPTING]" )
{
    entity_registry_t lRegistry{};
    auto             lEntity0 = lRegistry.CreateEntity();

    ScriptingEngine scriptingEngine{};
    auto            x        = scriptingEngine.RegisterPrimitiveType<ComponentA>( "ComponentA" );
    x["a"]                   = &ComponentA::a;
    x[sol::call_constructor] = sol::factories( []() { return ComponentA{}; }, []( float x ) { return ComponentA{ x }; } );

    scriptingEngine.Define( "registry0", &lRegistry ).Define( "entity0", lEntity0 );
    scriptingEngine.Execute( "entity0:add(dtypes.ComponentA(4.0))" );

    REQUIRE( lEntity0.Has<ComponentA>() );
    REQUIRE( lEntity0.Get<ComponentA>().a == 4.0 );
}

TEST_CASE( "LUA OnComponentAdded event", "[CORE_ENTITIES]" )
{
    entity_registry_t lRegistry{};
    ScriptingEngine  scriptingEngine{};
    auto             x       = scriptingEngine.RegisterPrimitiveType<ComponentA>( "ComponentA" );
    x["a"]                   = &ComponentA::a;
    x[sol::call_constructor] = sol::factories( []() { return ComponentA{}; }, []( float x ) { return ComponentA{ x }; } );
    scriptingEngine.Define( "registry", &lRegistry );

    bool lComponentAddedCalled = false;
    lRegistry.OnComponentAdded<ComponentA>( [&]( auto lEntity, auto &lComponent ) { lComponentAddedCalled = true; } );

    scriptingEngine.Execute( "entity0 = registry:create_entity()" );
    REQUIRE( !lComponentAddedCalled );

    scriptingEngine.Execute( "entity0:add(dtypes.ComponentA(3.0))" );
    REQUIRE( lComponentAddedCalled );
}

TEST_CASE( "LUA OnComponentUpdated event", "[CORE_ENTITIES]" )
{
    entity_registry_t lRegistry{};
    ScriptingEngine  scriptingEngine{};
    auto             x       = scriptingEngine.RegisterPrimitiveType<ComponentA>( "ComponentA" );
    x["a"]                   = &ComponentA::a;
    x[sol::call_constructor] = sol::factories( []() { return ComponentA{}; }, []( float x ) { return ComponentA{ x }; } );
    scriptingEngine.Define( "registry", &lRegistry );

    bool lComponentUpdatedCalled = false;
    lRegistry.OnComponentUpdated<ComponentA>( [&]( auto lEntity, auto &lComponent ) { lComponentUpdatedCalled = true; } );

    scriptingEngine.Execute( R"(
entity0 = registry:create_entity()
entity0:add(dtypes.ComponentA())
)" );
    REQUIRE( !lComponentUpdatedCalled );

    scriptingEngine.Execute( "entity0:replace(dtypes.ComponentA(3.0))" );
    REQUIRE( lComponentUpdatedCalled );
}

TEST_CASE( "LUA OnComponentDestroyed event", "[CORE_ENTITIES]" )
{
    entity_registry_t lRegistry{};
    ScriptingEngine  scriptingEngine{};
    auto             x       = scriptingEngine.RegisterPrimitiveType<ComponentA>( "ComponentA" );
    x["a"]                   = &ComponentA::a;
    x[sol::call_constructor] = sol::factories( []() { return ComponentA{}; }, []( float x ) { return ComponentA{ x }; } );
    scriptingEngine.Define( "registry", &lRegistry );

    bool lComponentDestroyedCalled = false;
    lRegistry.OnComponentDestroyed<ComponentA>( [&]( auto lEntity, auto &lComponent ) { lComponentDestroyedCalled = true; } );

    scriptingEngine.Execute( R"(
entity0 = registry:create_entity()
entity0:add(dtypes.ComponentA())
)" );
    REQUIRE( !lComponentDestroyedCalled );

    scriptingEngine.Execute( "entity0:remove(dtypes.ComponentA)" );
    REQUIRE( lComponentDestroyedCalled );
}

TEST_CASE( "LUA TensorShape", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    scriptingEngine.Execute( "value = Cuda.TensorShape({{1, 2}, {3, 4}, {5, 6}}, 123)" );

    auto lTensorShape = scriptingEngine.Get<tensor_shape_t>( "value" );

    REQUIRE( lTensorShape.CountLayers() == 3 );
    REQUIRE( lTensorShape.mRank == 2 );
    REQUIRE( lTensorShape.mElementSize == 123 );
    REQUIRE( lTensorShape.GetDimension( 0 ) == std::vector<uint32_t>{ 1, 3, 5 } );
    REQUIRE( lTensorShape.GetDimension( 1 ) == std::vector<uint32_t>{ 2, 4, 6 } );
}

TEST_CASE( "LUA TensorShape GetDimension", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    scriptingEngine.Execute( R"(
ts = Cuda.TensorShape( { { 1, 2, 9 }, { 3, 4, 8 }, { 5, 6, 7 } }, 123 )
value = ts:get_dimension( 0 )
)" );

    auto lDimension0 = scriptingEngine.Get<std::vector<uint32_t>>( "value" );
    REQUIRE( lDimension0 == std::vector<uint32_t>{ 1, 3, 5 } );

    scriptingEngine.Execute( R"(
ts = Cuda.TensorShape( { { 1, 2, 9 }, { 3, 4, 8 }, { 5, 6, 7 } }, 123 )
value = ts:get_dimension( 1 )
)" );

    auto lDimension1 = scriptingEngine.Get<std::vector<uint32_t>>( "value" );
    REQUIRE( lDimension1 == std::vector<uint32_t>{ 2, 4, 6 } );

    scriptingEngine.Execute( R"(
ts = Cuda.TensorShape( { { 1, 2, 9 }, { 3, 4, 8 }, { 5, 6, 7 } }, 123 )
value = ts:get_dimension( -1 )
)" );

    auto lDimension2 = scriptingEngine.Get<std::vector<uint32_t>>( "value" );
    REQUIRE( lDimension2 == std::vector<uint32_t>{ 9, 8, 7 } );
}

TEST_CASE( "LUA TensorShape Trim", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    scriptingEngine.Execute( R"(
value = Cuda.TensorShape( { { 1, 2, 9 }, { 3, 4, 8 }, { 5, 6, 7 } }, 123 )
value:trim( 1 )
)" );

    auto lDimension0 = scriptingEngine.Get<tensor_shape_t>( "value" );
    REQUIRE( lDimension0.mShape == std::vector<std::vector<uint32_t>>{ { 1 }, { 3 }, { 5 } } );

    scriptingEngine.Execute( R"(
value = Cuda.TensorShape( { { 1, 2, 9 }, { 3, 4, 8 }, { 5, 6, 7 } }, 123 )
value:trim( -1 )
)" );

    auto lDimension1 = scriptingEngine.Get<tensor_shape_t>( "value" );
    REQUIRE( lDimension1.mShape == std::vector<std::vector<uint32_t>>{ { 1, 2 }, { 3, 4 }, { 5, 6 } } );

    scriptingEngine.Execute( R"(
value = Cuda.TensorShape( { { 1, 2, 9 }, { 3, 4, 8 }, { 5, 6, 7 } }, 123 )
value:trim( -2 )
)" );

    auto lDimension2 = scriptingEngine.Get<tensor_shape_t>( "value" );
    REQUIRE( lDimension2.mShape == std::vector<std::vector<uint32_t>>{ { 1 }, { 3 }, { 5 } } );
}

TEST_CASE( "LUA TensorShape Flatten", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    scriptingEngine.Execute( R"(
value = Cuda.TensorShape( { { 1, 2, 9 }, { 3, 4, 8 }, { 5, 6, 7 } }, 123 )
value:flatten( 3 )
)" );

    auto lDimension0 = scriptingEngine.Get<tensor_shape_t>( "value" );
    REQUIRE( lDimension0.mShape == std::vector<std::vector<uint32_t>>{ { 18 }, { 96 }, { 210 } } );

    scriptingEngine.Execute( R"(
value = Cuda.TensorShape( { { 1, 2, 9 }, { 3, 4, 8 }, { 5, 6, 7 } }, 123 )
value:flatten( -1 )
)" );

    auto lDimension1 = scriptingEngine.Get<tensor_shape_t>( "value" );
    REQUIRE( lDimension1.mShape == std::vector<std::vector<uint32_t>>{ { 2, 9 }, { 12, 8 }, { 30, 7 } } );

    scriptingEngine.Execute( R"(
value = Cuda.TensorShape( { { 1, 2, 9 }, { 3, 4, 8 }, { 5, 6, 7 } }, 123 )
value:flatten( -2 )
)" );

    auto lDimension2 = scriptingEngine.Get<tensor_shape_t>( "value" );
    REQUIRE( lDimension2.mShape == std::vector<std::vector<uint32_t>>{ { 1, 2, 9 }, { 3, 4, 8 }, { 5, 6, 7 } } );
}

TEST_CASE( "LUA MemoryPool", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    scriptingEngine.Execute( "value = Cuda.MemoryPool.new(123)" );

    REQUIRE( true );
}

TEST_CASE( "LUA MultiTensor", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    scriptingEngine.Execute( R"(
pool  = Cuda.MemoryPool.new(64000)
shape = Cuda.TensorShape( { { 1, 2, 9 }, { 3, 4, 8 }, { 5, 6, 7 } }, 4 )
tensor = Cuda.MultiTensor.new(pool, shape)
value = tensor:size()
)" );

    auto lTensorSize = scriptingEngine.Get<uint32_t>( "value" );
    REQUIRE( lTensorSize == ( ( ( 1 * 2 * 9 ) + ( 3 * 4 * 8 ) + ( 5 * 6 * 7 ) ) * 4 ) );
}

TEST_CASE( "LUA MultiTensor Types", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    scriptingEngine.Execute( R"(
value = dtypes.float32.type_id()
)" );

    auto lTensorValues = scriptingEngine.Get<uint32_t>( "value" );
    REQUIRE( lTensorValues );
}

TEST_CASE( "LUA MultiTensor SizeAs", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    scriptingEngine.Execute( R"(
pool  = Cuda.MemoryPool.new(64000)
shape = Cuda.TensorShape( { { 1, 2, 9 }, { 3, 4, 8 }, { 5, 6, 7 } }, 4 )
tensor = Cuda.MultiTensor.new(pool, shape)
value = tensor:size_as(dtypes.float32)
)" );

    auto lTensorValues = scriptingEngine.Get<uint32_t>( "value" );
    REQUIRE( lTensorValues == ( ( ( 1 * 2 * 9 ) + ( 3 * 4 * 8 ) + ( 5 * 6 * 7 ) ) ) );
}

template <typename _Ty>
std::vector<std::vector<_Ty>> RandomVector2( std::vector<uint32_t> aDim, _Ty aMin, _Ty aMax )
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

TEST_CASE( "LUA MultiTensor fetch_at", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};
    size_t          lPoolSize = 128 * 1024;
    memory_pool_t      lPool( lPoolSize );

    auto lShape  = tensor_shape_t( std::vector<std::vector<uint32_t>>{ { 1, 2, 9 }, { 3, 4, 8 }, { 5, 6, 7 } }, sizeof( float ) );
    auto lTensor = multi_tensor_t( lPool, lShape );

    auto lLayer1 = ConcatenateVectors( RandomVector2<float>( lShape.mShape[0], 1.0f, 150.0f ) );
    auto lLayer2 = ConcatenateVectors( RandomVector2<float>( lShape.mShape[1], 1.0f, 150.0f ) );
    auto lLayer3 = ConcatenateVectors( RandomVector2<float>( lShape.mShape[2], 1.0f, 150.0f ) );

    lTensor.Upload( ConcatenateVectors( std::vector<std::vector<float>>{ lLayer1, lLayer2, lLayer3 } ) );
    scriptingEngine.Define( "tensor", lTensor );

    scriptingEngine.Execute( R"(
value0 = tensor:fetch_f32(0)
value1 = tensor:fetch_f32(1)
value2 = tensor:fetch_f32(2)
)" );

    {
        auto lTensorValues0 = scriptingEngine.Get<std::vector<float>>( "value0" );
        auto lTensorValues1 = lTensor.FetchBufferAt<float>( 0 );
        REQUIRE( lTensorValues0 == lTensorValues1 );
    }

    {
        auto lTensorValues0 = scriptingEngine.Get<std::vector<float>>( "value1" );
        auto lTensorValues1 = lTensor.FetchBufferAt<float>( 1 );
        REQUIRE( lTensorValues0 == lTensorValues1 );
    }

    {
        auto lTensorValues0 = scriptingEngine.Get<std::vector<float>>( "value2" );
        auto lTensorValues1 = lTensor.FetchBufferAt<float>( 2 );
        REQUIRE( lTensorValues0 == lTensorValues1 );
    }
}

TEST_CASE( "LUA MultiTensor fetch_flattened", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};
    size_t          lPoolSize = 128 * 1024;
    memory_pool_t      lPool( lPoolSize );

    auto lShape  = tensor_shape_t( std::vector<std::vector<uint32_t>>{ { 1, 2, 9 }, { 3, 4, 8 }, { 5, 6, 7 } }, sizeof( float ) );
    auto lTensor = multi_tensor_t( lPool, lShape );

    auto lLayer1 = ConcatenateVectors( RandomVector2<float>( lShape.mShape[0], 1.0f, 150.0f ) );
    auto lLayer2 = ConcatenateVectors( RandomVector2<float>( lShape.mShape[1], 1.0f, 150.0f ) );
    auto lLayer3 = ConcatenateVectors( RandomVector2<float>( lShape.mShape[2], 1.0f, 150.0f ) );

    lTensor.Upload( ConcatenateVectors( std::vector<std::vector<float>>{ lLayer1, lLayer2, lLayer3 } ) );
    scriptingEngine.Define( "tensor", lTensor );

    scriptingEngine.Execute( R"(
value = tensor:fetch_f32()
)" );

    auto lTensorValues0 = scriptingEngine.Get<std::vector<float>>( "value" );
    auto lTensorValues1 = lTensor.FetchFlattened<float>();
    REQUIRE( lTensorValues0 == lTensorValues1 );
}

TEST_CASE( "LUA MultiTensor upload", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};
    size_t          lPoolSize = 128 * 1024;
    memory_pool_t      lPool( lPoolSize );

    auto lShape = tensor_shape_t( std::vector<std::vector<uint32_t>>{ { 1, 2, 9 }, { 3, 4, 8 }, { 5, 6, 7 } }, sizeof( float ) );

    auto lLayer1 = ConcatenateVectors( RandomVector2<float>( lShape.mShape[0], 1.0f, 150.0f ) );
    auto lLayer2 = ConcatenateVectors( RandomVector2<float>( lShape.mShape[1], 1.0f, 150.0f ) );
    auto lLayer3 = ConcatenateVectors( RandomVector2<float>( lShape.mShape[2], 1.0f, 150.0f ) );

    auto lData = ConcatenateVectors( std::vector<std::vector<float>>{ lLayer1, lLayer2, lLayer3 } );

    scriptingEngine.Define( "tensor_data", lData );

    scriptingEngine.Execute( R"(
pool   = Cuda.MemoryPool.new(64000)
shape  = Cuda.TensorShape( { { 1, 2, 9 }, { 3, 4, 8 }, { 5, 6, 7 } }, 4 )
tensor = Cuda.MultiTensor.new(pool, shape)
tensor:upload_f32(tensor_data)
)" );

    auto lTensor        = scriptingEngine.Get<multi_tensor_t>( "tensor" );
    auto lTensorValues1 = lTensor.FetchFlattened<float>();
    REQUIRE( lData == lTensorValues1 );
}

TEST_CASE( "LUA MultiTensor upload layers", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};
    size_t          lPoolSize = 128 * 1024;
    memory_pool_t      lPool( lPoolSize );

    auto lShape = tensor_shape_t( std::vector<std::vector<uint32_t>>{ { 1, 2, 9 }, { 3, 4, 8 }, { 5, 6, 7 } }, sizeof( float ) );

    auto lLayer1 = ConcatenateVectors( RandomVector2<float>( lShape.mShape[0], 1.0f, 150.0f ) );
    auto lLayer2 = ConcatenateVectors( RandomVector2<float>( lShape.mShape[1], 1.0f, 150.0f ) );
    auto lLayer3 = ConcatenateVectors( RandomVector2<float>( lShape.mShape[2], 1.0f, 150.0f ) );

    auto lData = ConcatenateVectors( std::vector<std::vector<float>>{ lLayer1, lLayer2, lLayer3 } );

    scriptingEngine.Define( "tensor_data_0", lLayer1 );
    scriptingEngine.Define( "tensor_data_1", lLayer2 );
    scriptingEngine.Define( "tensor_data_2", lLayer3 );

    scriptingEngine.Execute( R"(
pool   = Cuda.MemoryPool.new(64000)
shape  = Cuda.TensorShape( { { 1, 2, 9 }, { 3, 4, 8 }, { 5, 6, 7 } }, 4 )
tensor = Cuda.MultiTensor.new(pool, shape)
tensor:upload_f32(tensor_data_0, 0)
tensor:upload_f32(tensor_data_1, 1)
tensor:upload_f32(tensor_data_2, 2)
)" );

    auto lTensor = scriptingEngine.Get<multi_tensor_t>( "tensor" );

    {
        auto lTensorValues1 = lTensor.FetchBufferAt<float>( 0 );
        REQUIRE( lLayer1 == lTensorValues1 );
    }

    {
        auto lTensorValues1 = lTensor.FetchBufferAt<float>( 1 );
        REQUIRE( lLayer2 == lTensorValues1 );
    }

    {
        auto lTensorValues1 = lTensor.FetchBufferAt<float>( 2 );
        REQUIRE( lLayer3 == lTensorValues1 );
    }
}

TEST_CASE( "LUA Scope", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    scriptingEngine.Execute( "value = Cuda.Scope.new(123)" );

    REQUIRE( true );
}

// TEST_CASE( "LUA TextureData", "[CORE_SCRIPTING]" )
// {
//     ScriptingEngine scriptingEngine{};

//     scriptingEngine.Execute( R"(value = Core.TextureData2D({
//         type         = Core.eTextureType.TEXTURE_2D,
//         color_format = Core.eColorFormat.RGBA32_FLOAT,
//         width        = 128,
//         height       = 256,
//         depth        = 1,
//         mip_levels   = 1})
// )" );
//     auto lTexture = scriptingEngine.Get<texture_data2d_t>( "value" );
//     REQUIRE( lTexture.mSpec.mType == texture_type_t::TEXTURE_2D );
//     REQUIRE( lTexture.mSpec.mFormat == color_format_t::RGBA32_FLOAT );
//     REQUIRE( lTexture.mSpec.mWidth == 128 );
// }

// TEST_CASE( "LUA TextureData from image data", "[CORE_SCRIPTING]" )
// {
//     ScriptingEngine scriptingEngine{};

//     scriptingEngine.Execute( R"(
// create_info = {
//     type         = Core.eTextureType.TEXTURE_2D,
//     color_format = Core.eColorFormat.R8_UNORM,
//     width        = 2,
//     height       = 2,
//     depth        = 1,
//     mip_levels = 1 }

// pixel_data = {
//     color_format = Core.eColorFormat.R8_UNORM,
//     width        = 2,
//     height       = 2,
//     pixel_data   = { 1, 2, 3, 4 } }
// value = Core.TextureData2D(create_info, pixel_data)
// )" );
//     auto lTexture = scriptingEngine.Get<texture_data2d_t>( "value" );
//     REQUIRE( lTexture.mSpec.mType == texture_type_t::TEXTURE_2D );
//     REQUIRE( lTexture.mSpec.mFormat == color_format_t::R8_UNORM );
//     REQUIRE( lTexture.mSpec.mWidth == 2 );

//     auto lPixelData = lTexture.GetImageData();
//     REQUIRE( lPixelData.mByteSize == 4 );
//     REQUIRE( lPixelData.mWidth == 2 );
//     REQUIRE( lPixelData.mHeight == 2 );
//     REQUIRE( lPixelData.mPixelData[0] == 1 );
//     REQUIRE( lPixelData.mPixelData[1] == 2 );
// }

// TEST_CASE( "LUA TextureData load from file", "[CORE_SCRIPTING]" )
// {
//     ScriptingEngine scriptingEngine{};

//     scriptingEngine.Execute( R"(value = Core.TextureData2D("C:/GitLab/SpockEngine/Tests/Data/kueken7_rgb8_unorm.ktx"))" );
//     {
//         auto lTexture = scriptingEngine.Get<texture_data2d_t>( "value" );
//         REQUIRE( lTexture.mSpec.mType == texture_type_t::TEXTURE_2D );
//         REQUIRE( lTexture.mSpec.mFormat == color_format_t::RGB8_UNORM );
//         REQUIRE( lTexture.mSpec.mWidth == 256 );
//     }

//     scriptingEngine.Execute( R"(value = Core.TextureData2D("C:/GitLab/SpockEngine/Tests/Data/kueken7_srgb8.png"))" );
//     {
//         auto lTexture = scriptingEngine.Get<texture_data2d_t>( "value" );
//         REQUIRE( lTexture.mSpec.mType == texture_type_t::TEXTURE_2D );
//         REQUIRE( lTexture.mSpec.mFormat == color_format_t::RGBA8_UNORM );
//         REQUIRE( lTexture.mSpec.mWidth == 256 );
//     }
// }

// TEST_CASE( "LUA TextureData get image data from texture", "[CORE_SCRIPTING]" )
// {
//     ScriptingEngine scriptingEngine{};

//     scriptingEngine.Execute( R"(
//     texture = Core.TextureData2D("C:/GitLab/SpockEngine/Tests/Data/kueken7_rgb8_unorm.ktx")
//     value = texture:get_image_data()
// )" );
//     auto lImageData = scriptingEngine.Get<sol::table>( "value" );
//     REQUIRE( lImageData.get<color_format_t>( "color_format" ) == color_format_t::RGB8_UNORM );
//     REQUIRE( lImageData.get<uint32_t>( "width" ) == 256 );
//     REQUIRE( lImageData.get<uint32_t>( "height" ) == 256 );
// }

TEST_CASE( "LUA TextureData load image data from file", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    scriptingEngine.Execute( R"(
    value = Core.load_image("C:/GitLab/SpockEngine/Tests/Data/kueken7_srgb8.png")
)" );
    auto lImageData = scriptingEngine.Get<sol::table>( "value" );
    REQUIRE( lImageData.get<color_format_t>( "color_format" ) == color_format_t::RGBA8_UNORM );
    REQUIRE( lImageData.get<uint32_t>( "width" ) == 256 );
    REQUIRE( lImageData.get<uint32_t>( "height" ) == 256 );
}

// TEST_CASE( "LUA TextureSampler", "[CORE_SCRIPTING]" )
// {
//     ScriptingEngine scriptingEngine{};

//     scriptingEngine.Execute( R"(
//     texture = Core.TextureData2D("C:/GitLab/SpockEngine/Tests/Data/kueken7_rgb8_unorm.ktx")

//     sampler_create_info = {
//         minification = Core.eSamplerFilter.NEAREST,
//         magnification = Core.eSamplerFilter.LINEAR,
//         mip = Core.eSamplerMipmap.NEAREST,
//         wrapping = Core.eSamplerWrapping.MIRROR_CLAMP_TO_BORDER,
//         offset = { x = 3.0, y = 4.0 },
//         scaling = { x = 5.0, y = 6.0 },
//         border_color = { r = 0.1, g = 0.2, b = 0.3, a = 0.4}
//     }

//     value = Core.TextureSampler2D(texture, sampler_create_info)
// )" );

//     auto lTextureSampler = scriptingEngine.Get<SE::Core::texture_data_sampler2d_t>( "value" );
//     REQUIRE( lTextureSampler.mSamplingSpec.mScaling == std::array<float, 2>{ 5.0f, 6.0f } );
//     REQUIRE( lTextureSampler.mSamplingSpec.mOffset == std::array<float, 2>{ 3.0f, 4.0f } );
//     REQUIRE( lTextureSampler.mSamplingSpec.mBorderColor == std::array<float, 4>{ .1f, .2f, .3f, .4f } );
// }

// TEST_CASE( "LUA Cuda Texture2D", "[CORE_SCRIPTING]" )
// {
//     ScriptingEngine scriptingEngine{};

//     scriptingEngine.Execute( R"(
//     texture = Core.TextureData2D("C:/GitLab/SpockEngine/Tests/Data/kueken7_srgb8.png")

//     texture_create_info = {
//         filter_mode = Core.eSamplerFilter.NEAREST,
//         wrapping    = Core.eSamplerWrapping.MIRROR_CLAMP_TO_BORDER
//     }

//     value = Cuda.Texture2D(texture_create_info, texture:get_image_data())
// )" );

//     auto &lCudaTexture = scriptingEngine.GetRef<SE::Cuda::texture2d_t>( "value" );
//     REQUIRE( lCudaTexture.mSpec.mFormat == color_format_t::RGBA8_UNORM );
//     REQUIRE( lCudaTexture.mSpec.mWidth == 256 );
//     REQUIRE( lCudaTexture.mSpec.mHeight == 256 );
// }

// TEST_CASE( "LUA Cuda TextureSampler2D", "[CORE_SCRIPTING]" )
// {
//     ScriptingEngine scriptingEngine{};

//     scriptingEngine.Execute( R"(
//     texture = Core.TextureData2D("C:/GitLab/SpockEngine/Tests/Data/kueken7_srgb8.png")

//     texture_create_info = {
//         filter_mode = Core.eSamplerFilter.NEAREST,
//         wrapping    = Core.eSamplerWrapping.MIRROR_CLAMP_TO_BORDER
//     }

//     tex2d = Cuda.Texture2D(texture_create_info, texture:get_image_data())

//     sampler_create_info = {
//         minification  = Core.eSamplerFilter.NEAREST,
//         magnification = Core.eSamplerFilter.NEAREST,
//         mip           = Core.eSamplerMipmap.NEAREST,
//         wrapping      = Core.eSamplerWrapping.MIRROR_CLAMP_TO_BORDER,
//         offset        = { x = 3.0, y = 4.0 },
//         scaling       = { x = 5.0, y = 6.0 },
//         border_color  = { r = 0.1, g = 0.2, b = 0.3, a = 0.4}
//     }

//     value = Cuda.TextureSampler2D(tex2d, sampler_create_info)
// )" );

//     auto &lCudaTextureSampler = scriptingEngine.GetRef<SE::Cuda::texture_sampler2d_t>( "value" );
//     REQUIRE( lCudaTextureSampler.mSpec.mScaling == std::array<float, 2>{ 5.0f, 6.0f } );
//     REQUIRE( lCudaTextureSampler.mSpec.mOffset == std::array<float, 2>{ 3.0f, 4.0f } );
//     REQUIRE( lCudaTextureSampler.mSpec.mBorderColor == std::array<float, 4>{ .1f, .2f, .3f, .4f } );
// }

TEST_CASE( "LUA sConstantValueInitializerComponent", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    scriptingEngine.Execute( R"(
    value = Cuda.Ops.sConstantValueInitializerComponent(Cuda.Ops.eScalarType.FLOAT32, 1.234)
)" );

    auto &lCudaTextureSampler = scriptingEngine.GetRef<constant_value_initializer_t>( "value" );
    REQUIRE( std::get<float>( lCudaTextureSampler.mValue ) == 1.234f );
}

TEST_CASE( "LUA sVectorInitializerComponent", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    std::vector<float> lData{ 1.234f, 2.345f, 3.456f, 1.234f, 2.345f, 3.456f };
    scriptingEngine.Define( "data", lData );

    scriptingEngine.Execute( R"(
    value = Cuda.Ops.sVectorInitializerComponent(data)
)" );

    auto &lValues = scriptingEngine.GetRef<vector_initializer_t>( "value" );
    REQUIRE( lValues.mValue.size() == 6 );
    REQUIRE( std::get<float>( lValues.mValue[0] ) == 1.234f );
    REQUIRE( std::get<float>( lValues.mValue[1] ) == 2.345f );
    REQUIRE( std::get<float>( lValues.mValue[2] ) == 3.456f );
    REQUIRE( std::get<float>( lValues.mValue[3] ) == 1.234f );
    REQUIRE( std::get<float>( lValues.mValue[4] ) == 2.345f );
    REQUIRE( std::get<float>( lValues.mValue[5] ) == 3.456f );
}

TEST_CASE( "LUA sDataInitializerComponent", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    std::vector<float> lData{ 1.234f, 2.345f, 3.456f, 1.234f, 2.345f, 3.456f };
    scriptingEngine.Define( "data", lData );

    scriptingEngine.Execute( R"(
    value = Cuda.Ops.sDataInitializerComponent(data)
)" );

    auto &lValues = scriptingEngine.GetRef<data_initializer_t>( "value" );
    REQUIRE( lValues.mValue.size() == 6 );
    REQUIRE( std::get<float>( lValues.mValue[0] ) == 1.234f );
    REQUIRE( std::get<float>( lValues.mValue[1] ) == 2.345f );
    REQUIRE( std::get<float>( lValues.mValue[2] ) == 3.456f );
    REQUIRE( std::get<float>( lValues.mValue[3] ) == 1.234f );
    REQUIRE( std::get<float>( lValues.mValue[4] ) == 2.345f );
    REQUIRE( std::get<float>( lValues.mValue[5] ) == 3.456f );
}

TEST_CASE( "LUA MultiTensorValue initialized with constant", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    scriptingEngine.Execute( R"(
scope    = Cuda.Scope.new(128 * 1024)
constant = Cuda.Ops.sConstantValueInitializerComponent(Cuda.Ops.eScalarType.FLOAT32, 1.234)
shape    = Cuda.TensorShape( { { 1, 2, 9 }, { 3, 4, 8 }, { 5, 6, 7 } }, 4 )
mt       = Cuda.Ops.MultiTensorValue(scope, constant, shape)
scope:run(mt)
)" );

    auto lTensorValues0 = scriptingEngine.Get<graph_node_t>( "mt" ).Get<multi_tensor_value_t>().mValue.FetchFlattened<float>();
    auto lExpected      = std::vector<float>( lTensorValues0.size(), 1.234f );

    REQUIRE( lTensorValues0 == lExpected );
}

TEST_CASE( "LUA MultiTensorValue initialized with vector", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    std::vector<float> lData{ 1.234f, 2.345f, 3.456f };
    scriptingEngine.Define( "data", lData );

    scriptingEngine.Execute( R"(
scope    = Cuda.Scope.new(128 * 1024)
constant = Cuda.Ops.sVectorInitializerComponent(data)
shape    = Cuda.TensorShape( { { 1, 2, 9 }, { 3, 4, 8 }, { 5, 6, 7 } }, 4 )
mt       = Cuda.Ops.MultiTensorValue(scope, constant, shape)
scope:run(mt)
)" );

    auto lTensorValues0 = scriptingEngine.Get<graph_node_t>( "mt" ).Get<multi_tensor_value_t>().mValue;

    {
        auto lLayer1   = lTensorValues0.FetchBufferAt<float>( 0 );
        auto lExpected = std::vector<float>( lLayer1.size(), 1.234f );
        REQUIRE( lLayer1 == lExpected );
    }

    {
        auto lLayer1   = lTensorValues0.FetchBufferAt<float>( 1 );
        auto lExpected = std::vector<float>( lLayer1.size(), 2.345f );
        REQUIRE( lLayer1 == lExpected );
    }

    {
        auto lLayer1   = lTensorValues0.FetchBufferAt<float>( 2 );
        auto lExpected = std::vector<float>( lLayer1.size(), 3.456f );
        REQUIRE( lLayer1 == lExpected );
    }
}

TEST_CASE( "LUA MultiTensorValue initialized with data", "[CORE_SCRIPTING]" )
{
    ScriptingEngine scriptingEngine{};

    auto lShape = tensor_shape_t( std::vector<std::vector<uint32_t>>{ { 1, 2, 9 }, { 3, 4, 8 }, { 5, 6, 7 } }, sizeof( float ) );

    auto lLayer1 = ConcatenateVectors( RandomVector2<float>( lShape.mShape[0], 1.0f, 150.0f ) );
    auto lLayer2 = ConcatenateVectors( RandomVector2<float>( lShape.mShape[1], 1.0f, 150.0f ) );
    auto lLayer3 = ConcatenateVectors( RandomVector2<float>( lShape.mShape[2], 1.0f, 150.0f ) );

    auto lData = ConcatenateVectors( std::vector<std::vector<float>>{ lLayer1, lLayer2, lLayer3 } );
    scriptingEngine.Define( "data", lData );
    scriptingEngine.Define( "shape", lShape );

    scriptingEngine.Execute( R"(
scope    = Cuda.Scope.new(128 * 1024)
constant = Cuda.Ops.sDataInitializerComponent(data)
mt       = Cuda.Ops.MultiTensorValue(scope, constant, shape)
scope:run(mt)
)" );

    auto lTensorValues0 = scriptingEngine.Get<graph_node_t>( "mt" ).Get<multi_tensor_value_t>().mValue;

    {
        auto lValues = lTensorValues0.FetchBufferAt<float>( 0 );
        REQUIRE( lLayer1 == lValues );
    }

    {
        auto lValues = lTensorValues0.FetchBufferAt<float>( 1 );
        REQUIRE( lLayer2 == lValues );
    }

    {
        auto lValues = lTensorValues0.FetchBufferAt<float>( 2 );
        REQUIRE( lLayer3 == lValues );
    }
}
