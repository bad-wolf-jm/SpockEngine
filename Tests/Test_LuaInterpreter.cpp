
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
using namespace numlua::core;
using namespace numlua::cuda;
using namespace numlua::mtops;
using namespace TestUtils;

TEST_CASE( "LUA Arrays", "[CORE_SCRIPTING]" )
{
    script_bindings scriptingEngine{};

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

