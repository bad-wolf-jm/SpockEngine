#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <filesystem>

#include "TestUtils.h"

#include "Core/Logging.h"
#include "Mono/MonoScriptEngine.h"

#include "mono/jit/jit.h"
#include "mono/metadata/assembly.h"
#include "mono/metadata/object.h"
#include "mono/metadata/tabledefs.h"

using namespace SE::Core;
using namespace TestUtils;

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

inline math::vec3 ToVec3( MonoObject *x )
{
    float *lV = (float *)mono_object_unbox( x );
    return math::make_vec3( lV );
}

inline math::vec4 ToVec4( MonoObject *x )
{
    float *lV = (float *)mono_object_unbox( x );
    return math::make_vec4( lV );
}

TEST_CASE( "Vector3 operations", "[MONO_SCRIPTING]" )
{
    InitializeMonoscripting( "C:\\GitLab\\SpockEngine\\Tests\\Mono\\Build\\Debug\\MonoscriptingTest.dll" );

    auto lVectorTest = MonoScriptClass( "SEUnitTest", "Vector3Tests", false );

    float lS = 1.234f;
    auto  lX = math::vec3{ 1.2f, 3.4f, 5.6f };
    auto  lY = math::vec3{ 3.4f, 5.6f, 7.8f };

    {
        auto lR = lVectorTest.CallMethod( "Constructor", lX.x, lX.y, lX.z );
        auto lZ = ToVec3( lR );
        REQUIRE( lZ == lX );
    }

    {
        auto lR = lVectorTest.CallMethod( "Add", lX, lY );
        auto lZ = ToVec3( lR );
        REQUIRE( lZ == ( lX + lY ) );
    }

    {
        auto lR = lVectorTest.CallMethod( "Subtract", lX, lY );
        auto lZ = ToVec3( lR );
        REQUIRE( lZ == ( lX - lY ) );
    }

    {
        auto lR = lVectorTest.CallMethod( "Divide0", lX, lS );
        auto lZ = ToVec3( lR );
        REQUIRE( lZ == ( lX / lS ) );
    }

    {
        auto lR = lVectorTest.CallMethod( "Divide1", lS, lY );
        auto lZ = ToVec3( lR );
        REQUIRE( lZ == ( lS / lY ) );
    }

    {
        auto lR = lVectorTest.CallMethod( "Multiply0", lX, lS );
        auto lZ = ToVec3( lR );
        REQUIRE( lZ == ( lX * lS ) );
    }

    {
        auto lR = lVectorTest.CallMethod( "Multiply1", lS, lY );
        auto lZ = ToVec3( lR );
        REQUIRE( lZ == ( lS * lY ) );
    }

    {
        auto lR = lVectorTest.CallMethod( "Cross", lX, lY );
        auto lZ = ToVec3( lR );
        REQUIRE( lZ == math::cross( lX, lY ) );
    }

    {
        auto  lR = lVectorTest.CallMethod( "Dot", lX, lY );
        float lV = ToFloat( lR );
        REQUIRE( lV == ( math::dot( lX, lY ) ) );
    }

    {
        auto  lR = lVectorTest.CallMethod( "Length", lX );
        float lV = ToFloat( lR );
        REQUIRE( lV == ( math::length( lX ) ) );
    }

    {
        auto  lR = lVectorTest.CallMethod( "Norm", lX );
        float lV = ToFloat( lR );
        REQUIRE( lV == ( math::length( lX ) ) );
    }

    {
        auto  lR = lVectorTest.CallMethod( "Norm1", lX );
        float lV = ToFloat( lR );
        REQUIRE( lV == ( math::abs( lX.x ) + math::abs( lX.y ) + math::abs( lX.z ) ) );
    }

    {
        auto  lR = lVectorTest.CallMethod( "Norm2", lX );
        float lV = ToFloat( lR );
        REQUIRE( lV == ( math::length( lX ) ) );
    }

    {
        auto lR = lVectorTest.CallMethod( "Normalized", lX );
        auto lZ = ToVec3( lR );
        REQUIRE( math::length( lZ - math::normalize( lX ) ) < 0.0000001f );
    }
}

TEST_CASE( "Vector4 operations", "[MONO_SCRIPTING]" )
{
    InitializeMonoscripting( "C:\\GitLab\\SpockEngine\\Tests\\Mono\\Build\\Debug\\MonoscriptingTest.dll" );

    auto lVectorTest = MonoScriptClass( "SEUnitTest", "Vector4Tests", false );

    float lS = 1.234f;
    auto  lX = math::vec4{ 1.2f, 3.4f, 5.6f, 4.2f };
    auto  lY = math::vec4{ 3.4f, 5.6f, 7.8f, 7.5f };

    {
        auto lR = lVectorTest.CallMethod( "Constructor0", lX.x, lX.y, lX.z, lX.w );
        auto lZ = ToVec4( lR );
        REQUIRE( lZ == lX );
    }

    {
        auto lR = lVectorTest.CallMethod( "Constructor1", math::vec3( lX ), lX.w );
        auto lZ = ToVec4( lR );
        REQUIRE( lZ == lX );
    }
    {
        auto lR = lVectorTest.CallMethod( "Constructor2", math::vec3( lX ) );
        auto lZ = ToVec4( lR );
        REQUIRE( (( math::vec3( lZ ) == math::vec3( lX ) ) && ( lZ.w == 0.0f )) );
    }

    {
        auto lR = lVectorTest.CallMethod( "Projection", lX );
        auto lZ = ToVec3( lR );
        REQUIRE( lZ == math::vec3( lX ) );
    }

    {
        auto lR = lVectorTest.CallMethod( "Add", lX, lY );
        auto lZ = ToVec4( lR );
        REQUIRE( lZ == ( lX + lY ) );
    }

    {
        auto lR = lVectorTest.CallMethod( "Subtract", lX, lY );
        auto lZ = ToVec4( lR );
        REQUIRE( lZ == ( lX - lY ) );
    }

    {
        auto lR = lVectorTest.CallMethod( "Divide0", lX, lS );
        auto lZ = ToVec4( lR );
        REQUIRE( lZ == ( lX / lS ) );
    }

    {
        auto lR = lVectorTest.CallMethod( "Divide1", lS, lY );
        auto lZ = ToVec4( lR );
        REQUIRE( lZ == ( lS / lY ) );
    }

    {
        auto lR = lVectorTest.CallMethod( "Multiply0", lX, lS );
        auto lZ = ToVec4( lR );
        REQUIRE( lZ == ( lX * lS ) );
    }

    {
        auto lR = lVectorTest.CallMethod( "Multiply1", lS, lY );
        auto lZ = ToVec4( lR );
        REQUIRE( lZ == ( lS * lY ) );
    }

    {
        auto  lR = lVectorTest.CallMethod( "Dot", lX, lY );
        float lV = ToFloat( lR );
        REQUIRE( lV == ( math::dot( lX, lY ) ) );
    }

    {
        auto  lR = lVectorTest.CallMethod( "Length", lX );
        float lV = ToFloat( lR );
        REQUIRE( lV == ( math::length( lX ) ) );
    }

    {
        auto  lR = lVectorTest.CallMethod( "Norm", lX );
        float lV = ToFloat( lR );
        REQUIRE( lV == ( math::length( lX ) ) );
    }

    {
        auto  lR = lVectorTest.CallMethod( "Norm1", lX );
        float lV = ToFloat( lR );
        REQUIRE( lV == ( math::abs( lX.x ) + math::abs( lX.y ) + math::abs( lX.z ) + math::abs( lX.w ) ) );
    }

    {
        auto  lR = lVectorTest.CallMethod( "Norm2", lX );
        float lV = ToFloat( lR );
        REQUIRE( lV == ( math::length( lX ) ) );
    }

    {
        auto lR = lVectorTest.CallMethod( "Normalized", lX );
        auto lZ = ToVec4( lR );
        REQUIRE( math::length( lZ - math::normalize( lX ) ) < 0.0000001f );
    }
}
