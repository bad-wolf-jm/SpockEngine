#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating.hpp>
#include <filesystem>
#include <type_traits>

#include "TestUtils.h"

#include "Core/EntityRegistry/Registry.h"
#include "Core/Logging.h"
#include "Mono/MonoScriptEngine.h"

#include "Scene/Components.h"

#include "mono/jit/jit.h"
#include "mono/metadata/assembly.h"
#include "mono/metadata/object.h"
#include "mono/metadata/tabledefs.h"

using namespace Catch::Matchers;
using namespace SE::Core;
using namespace TestUtils;
using namespace math;
using namespace SE::Core::EntityComponentSystem::Components;

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
inline bool  ToBool( MonoObject *x ) { return *(bool *)mono_object_unbox( x ); }

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

inline mat3 ToMat3( MonoObject *x )
{
    float *lV = (float *)mono_object_unbox( x );
    return make_mat3x3( lV );
}

inline mat4 ToMat4( MonoObject *x )
{
    float *lV = (float *)mono_object_unbox( x );
    return make_mat4x4( lV );
}

template <typename _RetType, typename... _ArgTypes>
inline _RetType CallMethodHelper( MonoScriptClass &aVectorTest, std::string const &aName, _ArgTypes... aArgs )
{
    auto lR = aVectorTest.CallMethod( aName, std::forward<_ArgTypes>( aArgs )... );

    if constexpr( std::is_same_v<_RetType, vec3> )
        return ToVec3( lR );
    else if constexpr( std::is_same_v<_RetType, vec4> )
        return ToVec4( lR );
    else if constexpr( std::is_same_v<_RetType, mat3> )
        return ToMat3( lR );
    else if constexpr( std::is_same_v<_RetType, mat4> )
        return ToMat4( lR );
    else if constexpr( std::is_same_v<_RetType, float> )
        return ToFloat( lR );
    else if constexpr( std::is_same_v<_RetType, bool> )
        return ToBool( lR );
}

TEST_CASE( "Vector3 operations", "[MONO_SCRIPTING]" )
{
    InitializeMonoscripting( "C:\\GitLab\\SpockEngine\\Tests\\Mono\\Build\\Debug\\MonoscriptingTest.dll" );

    auto lVectorTest = MonoScriptClass( "SEUnitTest", "Vector3Tests", false );

    float lS = RandomNumber( -10.0f, 10.0f );
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

    float lS = RandomNumber( -10.0f, 10.0f );
    auto  lX = vec4{ 1.2f, 3.4f, 5.6f, 4.2f };
    auto  lY = vec4{ 3.4f, 5.6f, 7.8f, 7.5f };

    REQUIRE( CallMethodHelper<vec4, float, float, float, float>( lVectorTest, "Constructor0", lX.x, lX.y, lX.z, lX.w ) == lX );
    REQUIRE( CallMethodHelper<vec4, vec3, float>( lVectorTest, "Constructor1", vec3( lX ), lX.w ) == lX );
    REQUIRE( CallMethodHelper<vec4, vec3>( lVectorTest, "Constructor2", vec3( lX ) ) == vec4( vec3( lX ), 0.0f ) );
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

vec3 RandomVec3()
{
    auto lM = RandomNumber( 3, -10.0f, 10.0f );

    return make_vec3( lM.data() );
}

vec4 RandomVec4()
{
    auto lM = RandomNumber( 4, -10.0f, 10.0f );

    return make_vec4( lM.data() );
}

mat3 RandomMat3()
{
    auto lM = RandomNumber( 9, -10.0f, 10.0f );

    return make_mat3x3( lM.data() );
}

mat4 RandomMat4()
{
    auto lM = RandomNumber( 16, -10.0f, 10.0f );

    return make_mat4x4( lM.data() );
}

TEST_CASE( "Matrix3 operations", "[MONO_SCRIPTING]" )
{
    InitializeMonoscripting( "C:\\GitLab\\SpockEngine\\Tests\\Mono\\Build\\Debug\\MonoscriptingTest.dll" );

    auto lVectorTest = MonoScriptClass( "SEUnitTest", "Matrix3Tests", false );

    float lS = RandomNumber( -10.0f, 10.0f );
    auto  lA = RandomMat4();
    auto  lX = RandomMat3();
    auto  lY = RandomMat3();
    auto  lV = RandomVec3();

    REQUIRE( CallMethodHelper<mat3, float>( lVectorTest, "Constructor0", lS ) == mat3( lS ) );
    REQUIRE( CallMethodHelper<mat3, float, float, float, float, float, float, float, float, float>(
                 lVectorTest, "Constructor1", lX[0][0], lX[1][0], lX[2][0], lX[0][1], lX[1][1], lX[2][1], lX[0][2], lX[1][2],
                 lX[2][2] ) == lX );
    REQUIRE( CallMethodHelper<mat3, mat4>( lVectorTest, "Constructor2", lA ) == mat3( lA ) );
    REQUIRE( CallMethodHelper<vec3, mat3>( lVectorTest, "Column0", lX ) == ( lX[0] ) );
    REQUIRE( CallMethodHelper<vec3, mat3>( lVectorTest, "Column1", lX ) == ( lX[1] ) );
    REQUIRE( CallMethodHelper<vec3, mat3>( lVectorTest, "Column2", lX ) == ( lX[2] ) );
    REQUIRE( CallMethodHelper<mat3, mat3, mat3>( lVectorTest, "Add", lX, lY ) == ( lX + lY ) );
    REQUIRE( CallMethodHelper<mat3, mat3, mat3>( lVectorTest, "Subtract", lX, lY ) == ( lX - lY ) );
    REQUIRE( CallMethodHelper<mat3, mat3, float>( lVectorTest, "Divide0", lX, lS ) == ( lX / lS ) );
    REQUIRE( CallMethodHelper<mat3, float, mat3>( lVectorTest, "Divide1", lS, lY ) == ( lS / lY ) );
    REQUIRE( CallMethodHelper<mat3, mat3, float>( lVectorTest, "Multiply0", lX, lS ) == ( lX * lS ) );
    REQUIRE( CallMethodHelper<mat3, float, mat3>( lVectorTest, "Multiply1", lS, lY ) == ( lS * lY ) );
    REQUIRE( CallMethodHelper<mat3, mat3, mat3>( lVectorTest, "Multiply2", lX, lY ) == ( lX * lY ) );
    REQUIRE( CallMethodHelper<vec3, mat3, vec3>( lVectorTest, "Multiply3", lX, lV ) == ( lX * lV ) );
    {
        auto lI0 = CallMethodHelper<mat3, mat3>( lVectorTest, "Inverse", lX );
        auto lI1 = Inverse( lX );

        REQUIRE_THAT( length2( lI0[0] - lI1[0] ), WithinAbs( 0.0f, 0.000001f ) );
        REQUIRE_THAT( length2( lI0[1] - lI1[1] ), WithinAbs( 0.0f, 0.000001f ) );
        REQUIRE_THAT( length2( lI0[2] - lI1[2] ), WithinAbs( 0.0f, 0.000001f ) );
    }
    REQUIRE_THAT( ( CallMethodHelper<float, mat3>( lVectorTest, "Determinant", lX ) ), WithinAbs( Determinant( lX ), 0.001f ) );
    REQUIRE( CallMethodHelper<mat3, mat3>( lVectorTest, "Transposed", lX ) == Transpose( lX ) );
}

#define TEST_MAT4_COLUMNS( C1, C2, e )                                  \
    do                                                                  \
    {                                                                   \
        REQUIRE_THAT( length2( C1[0] - C2[0] ), WithinAbs( 0.0f, e ) ); \
        REQUIRE_THAT( length2( C1[1] - C2[1] ), WithinAbs( 0.0f, e ) ); \
        REQUIRE_THAT( length2( C1[2] - C2[2] ), WithinAbs( 0.0f, e ) ); \
        REQUIRE_THAT( length2( C1[3] - C2[3] ), WithinAbs( 0.0f, e ) ); \
    } while( 0 )

TEST_CASE( "Matrix4 operations", "[MONO_SCRIPTING]" )
{
    InitializeMonoscripting( "C:\\GitLab\\SpockEngine\\Tests\\Mono\\Build\\Debug\\MonoscriptingTest.dll" );

    auto lVectorTest = MonoScriptClass( "SEUnitTest", "Matrix4Tests", false );

    float lS = RandomNumber( -10.0f, 10.0f );
    auto  lA = RandomMat3();
    auto  lX = RandomMat4();
    auto  lY = RandomMat4();
    auto  lV = RandomVec4();

    REQUIRE( CallMethodHelper<mat4, float>( lVectorTest, "Constructor0", lS ) == mat4( lS ) );
    REQUIRE( ( CallMethodHelper<mat4, float, float, float, float, float, float, float, float, float, float, float, float, float, float,
                                float, float>( lVectorTest, "Constructor1", lX[0][0], lX[1][0], lX[2][0], lX[3][0], lX[0][1], lX[1][1],
                                               lX[2][1], lX[3][1], lX[0][2], lX[1][2], lX[2][2], lX[3][2], lX[0][3], lX[1][3],
                                               lX[2][3], lX[3][3] ) ) == lX );
    REQUIRE( CallMethodHelper<mat4, mat3>( lVectorTest, "Constructor2", lA ) == mat4( lA ) );
    REQUIRE( CallMethodHelper<vec4, mat4>( lVectorTest, "Column0", lX ) == ( lX[0] ) );
    REQUIRE( CallMethodHelper<vec4, mat4>( lVectorTest, "Column1", lX ) == ( lX[1] ) );
    REQUIRE( CallMethodHelper<vec4, mat4>( lVectorTest, "Column2", lX ) == ( lX[2] ) );
    REQUIRE( CallMethodHelper<vec4, mat4>( lVectorTest, "Column3", lX ) == ( lX[3] ) );

    REQUIRE( CallMethodHelper<mat4, mat4, mat4>( lVectorTest, "Add", lX, lY ) == ( lX + lY ) );
    REQUIRE( CallMethodHelper<mat4, mat4, mat4>( lVectorTest, "Subtract", lX, lY ) == ( lX - lY ) );
    REQUIRE( CallMethodHelper<mat4, mat4, float>( lVectorTest, "Divide0", lX, lS ) == ( lX / lS ) );
    REQUIRE( CallMethodHelper<mat4, float, mat4>( lVectorTest, "Divide1", lS, lY ) == ( lS / lY ) );
    REQUIRE( CallMethodHelper<mat4, mat4, float>( lVectorTest, "Multiply0", lX, lS ) == ( lX * lS ) );
    REQUIRE( CallMethodHelper<mat4, float, mat4>( lVectorTest, "Multiply1", lS, lY ) == ( lS * lY ) );
    REQUIRE( CallMethodHelper<mat4, mat4, mat4>( lVectorTest, "Multiply2", lX, lY ) == ( lX * lY ) );

    {
        auto lI0 = CallMethodHelper<vec4, mat4, vec4>( lVectorTest, "Multiply3", lX, lV );
        auto lI1 = ( lX * lV );
        REQUIRE_THAT( length2( lI0 - lI1 ), WithinAbs( 0.0f, 0.000001f ) );
    }

    {
        auto lI0 = CallMethodHelper<mat4, mat4>( lVectorTest, "Inverse", lX );
        auto lI1 = Inverse( lX );

        TEST_MAT4_COLUMNS( lI0, lI1, 0.001f );
    }

    REQUIRE_THAT( ( CallMethodHelper<float, mat4>( lVectorTest, "Determinant", lX ) ), WithinAbs( Determinant( lX ), 0.001f ) );

    {
        auto lI0 = CallMethodHelper<mat4, mat4>( lVectorTest, "Transposed", lX );
        auto lI1 = Transpose( lX );

        TEST_MAT4_COLUMNS( lI0, lI1, 0.001f );
    }

    {
        float lRho = RandomNumber( -10.0f, 10.0f );
        auto  lI0  = CallMethodHelper<mat4, float, vec3>( lVectorTest, "Rotation", lRho, lV );
        auto  lI1  = Rotation( lRho, lV );

        TEST_MAT4_COLUMNS( lI0, lI1, 0.001f );
    }

    {
        auto lI0 = CallMethodHelper<mat4, vec3>( lVectorTest, "Scaling0", lV );
        auto lI1 = Scaling( lV );

        TEST_MAT4_COLUMNS( lI0, lI1, 0.001f );
    }

    {
        auto lI0 = CallMethodHelper<mat4, float>( lVectorTest, "Scaling1", lS );
        auto lI1 = Scaling( vec3{ lS, lS, lS } );

        TEST_MAT4_COLUMNS( lI0, lI1, 0.001f );
    }

    {
        auto lI0 = CallMethodHelper<mat4, vec3>( lVectorTest, "Translation0", vec3( lV ) );
        auto lI1 = Translation( vec3( lV ) );

        TEST_MAT4_COLUMNS( lI0, lI1, 0.001f );
    }

    {
        auto lV0 = vec3{ lV };
        auto lI0 = CallMethodHelper<mat4, float, float, float>( lVectorTest, "Translation1", lV0.x, lV0.y, lV0.z );
        auto lI1 = Translation( vec3( lV0 ) );

        TEST_MAT4_COLUMNS( lI0, lI1, 0.001f );
    }

    {
        auto lV0 = RandomVec4();
        auto lV1 = RandomVec4();
        auto lV2 = RandomVec4();
        auto lI0 = CallMethodHelper<mat4, vec3, vec3, vec3>( lVectorTest, "LookAt", lV0, lV1, lV2 );
        auto lI1 = LookAt( lV0, lV1, lV2 );

        TEST_MAT4_COLUMNS( lI0, lI1, 0.001f );
    }

    {
        auto lV0 = RandomNumber( -100.0f, 100.0f );
        auto lV1 = RandomNumber( -100.0f, 100.0f );
        auto lV2 = RandomNumber( -100.0f, 100.0f );
        auto lV3 = RandomNumber( -100.0f, 100.0f );
        auto lV4 = RandomNumber( -100.0f, 100.0f );
        auto lV5 = RandomNumber( -100.0f, 100.0f );
        auto lI0 =
            CallMethodHelper<mat4, float, float, float, float, float, float>( lVectorTest, "Ortho0", lV0, lV1, lV2, lV3, lV4, lV5 );
        auto lI1 = glm::ortho( lV0, lV1, lV2, lV3, lV4, lV5 );

        TEST_MAT4_COLUMNS( lI0, lI1, 0.001f );
    }

    {
        auto lV0 = RandomNumber( -100.0f, 100.0f );
        auto lV1 = RandomNumber( -100.0f, 100.0f );
        auto lV2 = RandomNumber( -100.0f, 100.0f );
        auto lV3 = RandomNumber( -100.0f, 100.0f );
        auto lI0 = CallMethodHelper<mat4, float, float, float, float>( lVectorTest, "Ortho1", lV0, lV1, lV2, lV3 );
        auto lI1 = glm::ortho( lV0, lV1, lV2, lV3 );

        TEST_MAT4_COLUMNS( lI0, lI1, 0.001f );
    }
}

TEST_CASE( "Entity is valid when first created", "[MONO_SCRIPTING]" )
{
    SE::Core::EntityRegistry lRegistry;

    auto lEntity         = lRegistry.CreateEntity();
    auto lEntityID       = static_cast<uint32_t>( lEntity );
    auto lRegistryID     = (size_t)lEntity.GetRegistry();
    auto lEntityClass    = MonoScriptClass( "SpockEngine", "Entity", true );
    auto lEntityInstance = lEntityClass.Instantiate( lEntityID, lRegistryID );

    REQUIRE( lEntityInstance.CallMethod( "IsValid" ) );
}

TEST_CASE( "Entity tag is reflected in scripting world", "[MONO_SCRIPTING]" )
{
    InitializeMonoscripting( "C:\\GitLab\\SpockEngine\\Tests\\Mono\\Build\\Debug\\MonoscriptingTest.dll" );
    auto lEntityTest = MonoScriptClass( "SEUnitTest", "EntityTest", false );

    SE::Core::EntityRegistry lRegistry;

    auto lEntity         = lRegistry.CreateEntity( "TAG_0" );
    auto lEntityID       = static_cast<uint32_t>( lEntity );
    auto lRegistryID     = (size_t)lEntity.GetRegistry();
    auto lEntityClass    = MonoScriptClass( "SpockEngine", "Entity", true );
    auto lEntityInstance = lEntityClass.Instantiate( lEntityID, lRegistryID );

    REQUIRE( CallMethodHelper<bool, MonoObject *>( lEntityTest, "TestHasTag", lEntityInstance.GetInstance() ) );

    MonoString *lManagedSTagValue = MonoScriptEngine::NewString( lEntity.Get<sTag>().mValue );
    REQUIRE( CallMethodHelper<bool, MonoObject *, MonoString *>( lEntityTest, "TestTagValue", lEntityInstance.GetInstance(),
                                                                 lManagedSTagValue ) );
}

TEST_CASE( "Entity tag is reflected in C++ world", "[MONO_SCRIPTING]" )
{
    InitializeMonoscripting( "C:\\GitLab\\SpockEngine\\Tests\\Mono\\Build\\Debug\\MonoscriptingTest.dll" );
    auto lEntityTest = MonoScriptClass( "SEUnitTest", "EntityTest", false );

    SE::Core::EntityRegistry lRegistry;

    auto lEntity         = lRegistry.CreateEntity();
    auto lEntityID       = static_cast<uint32_t>( lEntity );
    auto lRegistryID     = (size_t)lEntity.GetRegistry();
    auto lEntityClass    = MonoScriptClass( "SpockEngine", "Entity", true );
    auto lEntityInstance = lEntityClass.Instantiate( lEntityID, lRegistryID );

    MonoString *lManagedSTagValue = MonoScriptEngine::NewString( "TAG_1" );
    CallMethodHelper<bool, MonoObject *, MonoString *>( lEntityTest, "AddTagValue", lEntityInstance.GetInstance(), lManagedSTagValue );

    REQUIRE( ( lEntity.Has<sTag>() ) );
    REQUIRE( ( lEntity.Get<sTag>().mValue == "TAG_1" ) );
}

TEST_CASE( "Entity transform is reflected in scripting world", "[MONO_SCRIPTING]" )
{
    InitializeMonoscripting( "C:\\GitLab\\SpockEngine\\Tests\\Mono\\Build\\Debug\\MonoscriptingTest.dll" );
    auto lEntityTest = MonoScriptClass( "SEUnitTest", "EntityTest", false );

    SE::Core::EntityRegistry lRegistry;

    auto lEntity = lRegistry.CreateEntity( "TAG_0" );
    lEntity.Add<sNodeTransformComponent>( RandomMat4() );

    auto lEntityID    = static_cast<uint32_t>( lEntity );
    auto lRegistryID  = (size_t)lEntity.GetRegistry();
    auto lEntityClass = MonoScriptClass( "SpockEngine", "Entity", true );

    auto lEntityInstance = lEntityClass.Instantiate( lEntityID, lRegistryID );
    REQUIRE( CallMethodHelper<bool, MonoObject *>( lEntityTest, "TestHasNodeTransform", lEntityInstance.GetInstance() ) );
    REQUIRE( CallMethodHelper<bool, MonoObject *, mat4>( lEntityTest, "TestNodeTransformValue", lEntityInstance.GetInstance(),
                                                         lEntity.Get<sNodeTransformComponent>().mMatrix ) );
}

TEST_CASE( "Entity transform is reflected in C++ world", "[MONO_SCRIPTING]" )
{
    InitializeMonoscripting( "C:\\GitLab\\SpockEngine\\Tests\\Mono\\Build\\Debug\\MonoscriptingTest.dll" );
    auto lEntityTest = MonoScriptClass( "SEUnitTest", "EntityTest", false );

    SE::Core::EntityRegistry lRegistry;

    auto lEntity         = lRegistry.CreateEntity();
    auto lEntityID       = static_cast<uint32_t>( lEntity );
    auto lRegistryID     = (size_t)lEntity.GetRegistry();
    auto lEntityClass    = MonoScriptClass( "SpockEngine", "Entity", true );
    auto lEntityInstance = lEntityClass.Instantiate( lEntityID, lRegistryID );

    auto lMat4 = RandomMat4();

    CallMethodHelper<bool, MonoObject *, mat4>( lEntityTest, "AddNodeTransform", lEntityInstance.GetInstance(), lMat4 );

    REQUIRE( ( lEntity.Has<sNodeTransformComponent>() ) );
    REQUIRE( ( lEntity.Get<sNodeTransformComponent>().mMatrix == lMat4 ) );
}

TEST_CASE( "Entity transform matrix is reflected in scripting world", "[MONO_SCRIPTING]" )
{
    InitializeMonoscripting( "C:\\GitLab\\SpockEngine\\Tests\\Mono\\Build\\Debug\\MonoscriptingTest.dll" );
    auto lEntityTest = MonoScriptClass( "SEUnitTest", "EntityTest", false );

    SE::Core::EntityRegistry lRegistry;

    auto lEntity = lRegistry.CreateEntity( "TAG_0" );
    lEntity.Add<sTransformMatrixComponent>( RandomMat4() );

    auto lEntityID    = static_cast<uint32_t>( lEntity );
    auto lRegistryID  = (size_t)lEntity.GetRegistry();
    auto lEntityClass = MonoScriptClass( "SpockEngine", "Entity", true );

    auto lEntityInstance = lEntityClass.Instantiate( lEntityID, lRegistryID );
    REQUIRE( CallMethodHelper<bool, MonoObject *>( lEntityTest, "TestHasTransformMatrix", lEntityInstance.GetInstance() ) );
    REQUIRE( CallMethodHelper<bool, MonoObject *, mat4>( lEntityTest, "TestTransformMatrixValue", lEntityInstance.GetInstance(),
                                                         lEntity.Get<sTransformMatrixComponent>().Matrix ) );
}

TEST_CASE( "Entity transform matrix is reflected in C++ world", "[MONO_SCRIPTING]" )
{
    InitializeMonoscripting( "C:\\GitLab\\SpockEngine\\Tests\\Mono\\Build\\Debug\\MonoscriptingTest.dll" );
    auto lEntityTest = MonoScriptClass( "SEUnitTest", "EntityTest", false );

    SE::Core::EntityRegistry lRegistry;

    auto lEntity         = lRegistry.CreateEntity();
    auto lEntityID       = static_cast<uint32_t>( lEntity );
    auto lRegistryID     = (size_t)lEntity.GetRegistry();
    auto lEntityClass    = MonoScriptClass( "SpockEngine", "Entity", true );
    auto lEntityInstance = lEntityClass.Instantiate( lEntityID, lRegistryID );

    auto lMat4 = RandomMat4();

    CallMethodHelper<bool, MonoObject *, mat4>( lEntityTest, "AddNodeTransformMartix", lEntityInstance.GetInstance(), lMat4 );

    REQUIRE( ( lEntity.Has<sTransformMatrixComponent>() ) );
    REQUIRE( ( lEntity.Get<sTransformMatrixComponent>().Matrix == lMat4 ) );
}
 
TEST_CASE( "Entity light component is reflected in scripting world", "[MONO_SCRIPTING]" )
{
    InitializeMonoscripting( "C:\\GitLab\\SpockEngine\\Tests\\Mono\\Build\\Debug\\MonoscriptingTest.dll" );
    auto lEntityTest = MonoScriptClass( "SEUnitTest", "EntityTest", false );

    SE::Core::EntityRegistry lRegistry;

    auto lEntity = lRegistry.CreateEntity( "TAG_0" );
    lEntity.Add<sLightComponent>();

    auto lEntityID    = static_cast<uint32_t>( lEntity );
    auto lRegistryID  = (size_t)lEntity.GetRegistry();
    auto lEntityClass = MonoScriptClass( "SpockEngine", "Entity", true );

    auto lEntityInstance = lEntityClass.Instantiate( lEntityID, lRegistryID );
    REQUIRE( CallMethodHelper<bool, MonoObject *>( lEntityTest, "TestHasLight", lEntityInstance.GetInstance() ) );
}

TEST_CASE( "Entity light is reflected in C++ world", "[MONO_SCRIPTING]" )
{
    InitializeMonoscripting( "C:\\GitLab\\SpockEngine\\Tests\\Mono\\Build\\Debug\\MonoscriptingTest.dll" );
    auto lEntityTest = MonoScriptClass( "SEUnitTest", "EntityTest", false );

    SE::Core::EntityRegistry lRegistry;

    auto lEntity         = lRegistry.CreateEntity();
    auto lEntityID       = static_cast<uint32_t>( lEntity );
    auto lRegistryID     = (size_t)lEntity.GetRegistry();
    auto lEntityClass    = MonoScriptClass( "SpockEngine", "Entity", true );
    auto lEntityInstance = lEntityClass.Instantiate( lEntityID, lRegistryID );

    CallMethodHelper<bool, MonoObject *>( lEntityTest, "AddLight", lEntityInstance.GetInstance() );

    REQUIRE( ( lEntity.Has<sLightComponent>() ) );
}
