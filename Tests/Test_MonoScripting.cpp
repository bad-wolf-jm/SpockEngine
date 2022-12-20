#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <type_traits>

#include "TestUtils.h"

#include "Core/Logging.h"
#include "Mono/MonoScriptEngine.h"

#include "mono/jit/jit.h"
#include "mono/metadata/assembly.h"
#include "mono/metadata/object.h"
#include "mono/metadata/tabledefs.h"

using namespace SE::Core;
using namespace TestUtils;
using namespace math;

namespace fs = std::filesystem;

void InitializeMonoscripting()
{
    fs::path lMonoPath          = "C:\\Program Files\\Mono\\lib\\mono\\4.5";
    fs::path lCoreScriptingPath = "c:\\GitLab\\SpockEngine\\Source\\ScriptCore\\Build\\Debug\\SE_Core.dll";

    MonoScriptEngine::Initialize( lMonoPath, lCoreScriptingPath );
}

void InitializeMonoscripting( fs::path aAppAssemblyPath )
{
    InitializeMonoscripting();
    MonoScriptEngine::SetAppAssemblyPath( aAppAssemblyPath );
}

TEST_CASE( "Initialize scripting engine", "[MONO_SCRIPTING]" )
{
    InitializeMonoscripting();

    REQUIRE( true );
}

TEST_CASE( "Set app assembly path", "[MONO_SCRIPTING]" )
{
    InitializeMonoscripting( "C:\\GitLab\\SpockEngine\\Tests\\Mono\\Build\\Debug\\MonoscriptingTest.dll" );

    REQUIRE( true );
}

inline float ToFloat( MonoObject *x ) { return *(float *)mono_object_unbox( x ); }

inline vec3 ToVec3( MonoObject *x )
{
    float *lV = (float *)mono_object_unbox( x );
    return make_vec3( lV );
}

inline vec4 ToVec4( MonoObject *x )
{
    float *lV = (float *)mono_object_unbox( x );
    return make_vec4( lV );
}

template <typename _RetType, typename... _ArgTypes>
inline _RetType CallMethodHelper( MonoScriptClass &aVectorTest, std::string const &aName, _ArgTypes... aArgs )
{
    auto lR = aVectorTest.CallMethod( aName, std::forward<_ArgTypes>( aArgs )... );

    if constexpr( std::is_same_v<_RetType, vec3> )
        return ToVec3( lR );
    else if constexpr( std::is_same_v<_RetType, vec4> )
        return ToVec4( lR );
    else
        return ToFloat( lR );
}

TEST_CASE( "Vector3 operations", "[MONO_SCRIPTING]" )
{
    InitializeMonoscripting( "C:\\GitLab\\SpockEngine\\Tests\\Mono\\Build\\Debug\\MonoscriptingTest.dll" );

    auto lVectorTest = MonoScriptClass( "SEUnitTest", "Vector3Tests", false );

    float lS = 1.234f;
    auto  lX = vec3{ 1.2f, 3.4f, 5.6f };
    auto  lY = vec3{ 3.4f, 5.6f, 7.8f };

    REQUIRE( CallMethodHelper<vec3, float, float, float>( lVectorTest, "Constructor", lX.x, lX.y, lX.z ) == lX );
    REQUIRE( CallMethodHelper<vec3, vec3, vec3>( lVectorTest, "Add", lX, lY ) == ( lX + lY ) );
    REQUIRE( CallMethodHelper<vec3, vec3, vec3>( lVectorTest, "Subtract", lX, lY ) == ( lX - lY ) );
    REQUIRE( CallMethodHelper<vec3, vec3, float>( lVectorTest, "Divide0", lX, lS ) == ( lX / lS ) );
    REQUIRE( CallMethodHelper<vec3, float, vec3>( lVectorTest, "Divide1", lS, lY ) == ( lS / lY ) );
    REQUIRE( CallMethodHelper<vec3, vec3, float>( lVectorTest, "Multiply0", lX, lS ) == ( lX * lS ) );
    REQUIRE( CallMethodHelper<vec3, float, vec3>( lVectorTest, "Multiply1", lS, lY ) == ( lS * lY ) );
    REQUIRE( CallMethodHelper<vec3, vec3, vec3>( lVectorTest, "Cross", lX, lY ) == ( cross( lX, lY ) ) );
    REQUIRE( CallMethodHelper<float, vec3, vec3>( lVectorTest, "Dot", lX, lY ) == ( dot( lX, lY ) ) );
    REQUIRE( CallMethodHelper<float, vec3>( lVectorTest, "Length", lX ) == ( length( lX ) ) );
    REQUIRE( CallMethodHelper<float, vec3>( lVectorTest, "Norm", lX ) == ( length( lX ) ) );
    REQUIRE( CallMethodHelper<float, vec3>( lVectorTest, "Norm1", lX ) ==
             ( math::abs( lX.x ) + math::abs( lX.y ) + math::abs( lX.z ) ) );
    REQUIRE( CallMethodHelper<float, vec3>( lVectorTest, "Norm2", lX ) == ( length( lX ) ) );

    {
        auto lZ = CallMethodHelper<vec3, vec3>( lVectorTest, "Normalized", lX );
        REQUIRE( length( lZ - normalize( lX ) ) < 0.0000001f );
    }
}

TEST_CASE( "Vector4 operations", "[MONO_SCRIPTING]" )
{
    InitializeMonoscripting( "C:\\GitLab\\SpockEngine\\Tests\\Mono\\Build\\Debug\\MonoscriptingTest.dll" );

    auto lVectorTest = MonoScriptClass( "SEUnitTest", "Vector4Tests", false );

    float lS = 1.234f;
    auto  lX = vec4{ 1.2f, 3.4f, 5.6f, 4.2f };
    auto  lY = vec4{ 3.4f, 5.6f, 7.8f, 7.5f };

    REQUIRE( CallMethodHelper<vec4, float, float, float, float>( lVectorTest, "Constructor0", lX.x, lX.y, lX.z, lX.w ) == lX );
    REQUIRE( CallMethodHelper<vec4, vec3, float>( lVectorTest, "Constructor1", vec3( lX ), lX.w ) == lX );
    REQUIRE( CallMethodHelper<vec4, vec3>( lVectorTest, "Constructor2", vec3( lX ) ) == math::vec4( vec3( lX ), 0.0f ) );
    REQUIRE( CallMethodHelper<vec3, vec4>( lVectorTest, "Projection", lX ) == vec3( lX ) );
    REQUIRE( CallMethodHelper<vec4, vec4, vec4>( lVectorTest, "Add", lX, lY ) == ( lX + lY ) );
    REQUIRE( CallMethodHelper<vec4, vec4, vec4>( lVectorTest, "Subtract", lX, lY ) == ( lX - lY ) );
    REQUIRE( CallMethodHelper<vec4, vec4, float>( lVectorTest, "Divide0", lX, lS ) == ( lX / lS ) );
    REQUIRE( CallMethodHelper<vec4, float, vec4>( lVectorTest, "Divide1", lS, lY ) == ( lS / lY ) );
    REQUIRE( CallMethodHelper<vec4, vec4, float>( lVectorTest, "Multiply0", lX, lS ) == ( lX * lS ) );
    REQUIRE( CallMethodHelper<vec4, float, vec4>( lVectorTest, "Multiply1", lS, lY ) == ( lS * lY ) );
    REQUIRE( CallMethodHelper<float, vec4, vec4>( lVectorTest, "Dot", lX, lY ) == ( dot( lX, lY ) ) );
    REQUIRE( CallMethodHelper<float, vec4>( lVectorTest, "Length", lX ) == ( length( lX ) ) );
    REQUIRE( CallMethodHelper<float, vec4>( lVectorTest, "Norm", lX ) == ( length( lX ) ) );
    REQUIRE( CallMethodHelper<float, vec4>( lVectorTest, "Norm1", lX ) ==
             ( math::abs( lX.x ) + math::abs( lX.y ) + math::abs( lX.z ) + math::abs( lX.w ) ) );
    REQUIRE( CallMethodHelper<float, vec4>( lVectorTest, "Norm2", lX ) == ( length( lX ) ) );

    {
        auto lZ = CallMethodHelper<vec4, vec4>( lVectorTest, "Normalized", lX );
        REQUIRE( length( lZ - normalize( lX ) ) < 0.0000001f );
    }
}
