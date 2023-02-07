#include "MonoScriptEngine.h"

#include "Core/Logging.h"
#include "Core/Memory.h"

#include "Engine/Engine.h"

#include "Scene/Scene.h"

#include "mono/jit/jit.h"
#include "mono/metadata/assembly.h"
#include "mono/metadata/object.h"
#include "mono/metadata/tabledefs.h"

#include <unordered_map>

#ifdef WIN32_LEAN_AND_MEAN
#    undef WIN32_LEAN_AND_MEAN
#endif
#include "Core/FileWatch.hpp"

#include "EntityRegistry.h"
#include "InternalCalls.h"
#include "MonoScriptUtils.h"

namespace SE::Core
{
    struct ScriptEngineData
    {
        MonoDomain   *mRootDomain        = nullptr;
        MonoAssembly *mCoreAssembly      = nullptr;
        MonoImage    *mCoreAssemblyImage = nullptr;

        MonoDomain   *mAppDomain        = nullptr;
        MonoAssembly *mAppAssembly      = nullptr;
        MonoImage    *mAppAssemblyImage = nullptr;

        std::filesystem::path mCoreAssemblyFilepath;
        std::filesystem::path mAppAssemblyFilepath;

        MonoScriptClass                                       mBaseApplicationClass;
        std::unordered_map<std::string, Ref<MonoScriptClass>> mApplicationClasses;

        MonoScriptClass                                       mBaseControllerClass;
        std::unordered_map<std::string, Ref<MonoScriptClass>> mControllerClasses;

        MonoScriptClass                                       mBaseHUDClass;
        std::unordered_map<std::string, Ref<MonoScriptClass>> mHUDClasses;

        MonoScriptClass                                       mBaseComponentClass;
        std::unordered_map<std::string, Ref<MonoScriptClass>> mComponentClasses;

        std::unordered_map<std::string, Ref<MonoScriptClass>> mAllClasses;

        std::unique_ptr<filewatch::FileWatch<std::string>> mAppAssemblyFileWatcher;

        void *mSceneContext = nullptr;

        bool mAssemblyReloadPending = false;
    };

    static ScriptEngineData *sData = nullptr;

    MonoObject *MonoScriptEngine::InstantiateClass( MonoClass *aMonoClass, bool aIsCore )
    {
        MonoObject *aInstance = mono_object_new( sData->mAppDomain, aMonoClass );
        
        // mono_runtime_object_init( aInstance );
        return aInstance;
    }

    void MonoScriptEngine::LoadCoreAssembly( const std::filesystem::path &aFilepath )
    {
        sData->mAppDomain = mono_domain_create_appdomain( "SE_Runtime", nullptr );
        mono_domain_set( sData->mAppDomain, true );

        sData->mCoreAssemblyFilepath = aFilepath;
        sData->mCoreAssembly         = Mono::Utils::LoadMonoAssembly( aFilepath );
        sData->mCoreAssemblyImage    = mono_assembly_get_image( sData->mCoreAssembly );

        sData->mBaseApplicationClass = MonoScriptClass( "SpockEngine", "SEApplication", true );
        sData->mBaseControllerClass  = MonoScriptClass( "SpockEngine", "ActorComponent", true );
        sData->mBaseHUDClass         = MonoScriptClass( "SpockEngine", "HUDComponent", true );
        sData->mBaseComponentClass   = MonoScriptClass( "SpockEngine", "Component", true );

        // Mono::Utils::PrintAssemblyTypes( sData->mCoreAssembly );
    }

    static void OnAppAssemblyFileSystemEvent( const std::string &path, const filewatch::Event change_type )
    {
        if( !sData->mAssemblyReloadPending && change_type == filewatch::Event::modified )
        {
            sData->mAssemblyReloadPending = true;

            Engine::GetInstance()->SubmitToMainThread(
                [&]()
                {
                    sData->mAppAssemblyFileWatcher.reset();
                    MonoScriptEngine::ReloadAssembly();
                } );
        }
    }

    void MonoScriptEngine::SetAppAssemblyPath( const std::filesystem::path &aFilepath )
    {
        if( sData->mAppAssemblyFilepath == aFilepath ) return;

        sData->mAppAssemblyFilepath = aFilepath;

        sData->mAssemblyReloadPending = false;

        if( !sData->mAppAssemblyFilepath.empty() )
            sData->mAppAssemblyFileWatcher =
                std::make_unique<filewatch::FileWatch<std::string>>( aFilepath.string(), OnAppAssemblyFileSystemEvent );

        ReloadAssembly();
    }

    void MonoScriptEngine::Initialize( std::filesystem::path &aMonoPath, const std::filesystem::path &aCoreAssemblyPath )
    {
        if( sData != nullptr ) return;

        sData = new ScriptEngineData();

        InitMono( aMonoPath );

        RegisterInternalCppFunctions();

        LoadCoreAssembly( aCoreAssemblyPath );
    }

    void MonoScriptEngine::RegisterComponentTypes()
    {
        RegisterComponentType<sTag>();
        RegisterComponentType<sNodeTransformComponent>();
        RegisterComponentType<sLightComponent>();
    }

    void MonoScriptEngine::RegisterInternalCppFunctions()
    {
        using namespace MonoInternalCalls;

        SE_ADD_INTERNAL_CALL( Entity_Create );
        SE_ADD_INTERNAL_CALL( Entity_IsValid );
        SE_ADD_INTERNAL_CALL( Entity_Has );
        SE_ADD_INTERNAL_CALL( Entity_Get );
        SE_ADD_INTERNAL_CALL( Entity_Add );
        SE_ADD_INTERNAL_CALL( Entity_Replace );

        SE_ADD_INTERNAL_CALL( OpNode_NewTensorShape );
        SE_ADD_INTERNAL_CALL( OpNode_DestroyTensorShape );
        SE_ADD_INTERNAL_CALL( OpNode_CountLayers );
        SE_ADD_INTERNAL_CALL( OpNode_GetDimension );
        SE_ADD_INTERNAL_CALL( OpNode_Trim );
        SE_ADD_INTERNAL_CALL( OpNode_Flatten );
        SE_ADD_INTERNAL_CALL( OpNode_InsertDimension );

        SE_ADD_INTERNAL_CALL( OpNode_NewScope );
        SE_ADD_INTERNAL_CALL( OpNode_DestroyScope );
        SE_ADD_INTERNAL_CALL( OpNode_CreateMultiTensor_Constant_Initializer );
        SE_ADD_INTERNAL_CALL( OpNode_CreateMultiTensor_Vector_Initializer );
        SE_ADD_INTERNAL_CALL( OpNode_CreateMultiTensor_Data_Initializer );
        SE_ADD_INTERNAL_CALL( OpNode_CreateMultiTensor_Random_Uniform_Initializer );
        SE_ADD_INTERNAL_CALL( OpNode_CreateMultiTensor_Random_Normal_Initializer );
        SE_ADD_INTERNAL_CALL( OpNode_CreateVector );
        SE_ADD_INTERNAL_CALL( OpNode_CreateScalarVector );
        SE_ADD_INTERNAL_CALL( OpNode_CreateScalarValue );

        SE_ADD_INTERNAL_CALL( OpNode_Add );
        SE_ADD_INTERNAL_CALL( OpNode_Subtract );
        SE_ADD_INTERNAL_CALL( OpNode_Divide );
        SE_ADD_INTERNAL_CALL( OpNode_Multiply );
        SE_ADD_INTERNAL_CALL( OpNode_And );
        SE_ADD_INTERNAL_CALL( OpNode_Or );
        SE_ADD_INTERNAL_CALL( OpNode_Not );
        SE_ADD_INTERNAL_CALL( OpNode_BitwiseAnd );
        SE_ADD_INTERNAL_CALL( OpNode_BitwiseOr );
        SE_ADD_INTERNAL_CALL( OpNode_BitwiseNot );
        SE_ADD_INTERNAL_CALL( OpNode_InInterval );
        SE_ADD_INTERNAL_CALL( OpNode_Equal );
        SE_ADD_INTERNAL_CALL( OpNode_LessThan );
        SE_ADD_INTERNAL_CALL( OpNode_LessThanOrEqual );
        SE_ADD_INTERNAL_CALL( OpNode_GreaterThan );
        SE_ADD_INTERNAL_CALL( OpNode_GreaterThanOrEqual );
        SE_ADD_INTERNAL_CALL( OpNode_Where );
        SE_ADD_INTERNAL_CALL( OpNode_Mix );
        SE_ADD_INTERNAL_CALL( OpNode_AffineTransform );
        SE_ADD_INTERNAL_CALL( OpNode_ARange );
        SE_ADD_INTERNAL_CALL( OpNode_LinearSpace );
        SE_ADD_INTERNAL_CALL( OpNode_Repeat );
        SE_ADD_INTERNAL_CALL( OpNode_Tile );
        SE_ADD_INTERNAL_CALL( OpNode_Sample2D );
        SE_ADD_INTERNAL_CALL( OpNode_Collapse );
        SE_ADD_INTERNAL_CALL( OpNode_Expand );
        SE_ADD_INTERNAL_CALL( OpNode_Reshape );
        SE_ADD_INTERNAL_CALL( OpNode_Relayout );
        SE_ADD_INTERNAL_CALL( OpNode_FlattenNode );
        SE_ADD_INTERNAL_CALL( OpNode_Slice );
        SE_ADD_INTERNAL_CALL( OpNode_Summation );
        SE_ADD_INTERNAL_CALL( OpNode_CountTrue );
        SE_ADD_INTERNAL_CALL( OpNode_CountNonZero );
        SE_ADD_INTERNAL_CALL( OpNode_CountZero );
        SE_ADD_INTERNAL_CALL( OpNode_Floor );
        SE_ADD_INTERNAL_CALL( OpNode_Ceil );
        SE_ADD_INTERNAL_CALL( OpNode_Abs );
        SE_ADD_INTERNAL_CALL( OpNode_Sqrt );
        SE_ADD_INTERNAL_CALL( OpNode_Round );
        SE_ADD_INTERNAL_CALL( OpNode_Diff );
        SE_ADD_INTERNAL_CALL( OpNode_Shift );
        SE_ADD_INTERNAL_CALL( OpNode_Conv1D );
        SE_ADD_INTERNAL_CALL( OpNode_HCat );

        SE_ADD_INTERNAL_CALL( UI_Text );
        SE_ADD_INTERNAL_CALL( UI_Button );
    }

    void MonoScriptEngine::Shutdown()
    {
        ShutdownMono();

        delete sData;
        sData = nullptr;
    }

    void MonoScriptEngine::InitMono( std::filesystem::path &aMonoPath )
    {
        mono_set_assemblies_path( aMonoPath.string().c_str() );

        MonoDomain *lRootDomain = mono_jit_init( "SpockEngineRuntime" );

        sData->mRootDomain = lRootDomain;
    }

    void MonoScriptEngine::ShutdownMono()
    {
        mono_domain_set( mono_get_root_domain(), false );

        mono_domain_unload( sData->mAppDomain );
        sData->mAppDomain = nullptr;

        mono_jit_cleanup( sData->mRootDomain );
        sData->mRootDomain = nullptr;
    }

    MonoImage *MonoScriptEngine::GetCoreAssemblyImage() { return sData->mCoreAssemblyImage; }
    MonoImage *MonoScriptEngine::GetAppAssemblyImage() { return sData->mAppAssemblyImage; }

    void *MonoScriptEngine::GetSceneContext() { return sData->mSceneContext; }

    MonoString *MonoScriptEngine::NewString( std::string const &aString )
    {
        return mono_string_new( sData->mAppDomain, aString.c_str() );
    }

    std::string MonoScriptEngine::NewString( MonoString *aString ) { return std::string( mono_string_to_utf8( aString ) ); }

    void MonoScriptEngine::LoadAssemblyClasses()
    {
        if( !sData->mAppAssemblyImage ) return;

        const MonoTableInfo *lTypeDefinitionsTable = mono_image_get_table_info( sData->mAppAssemblyImage, MONO_TABLE_TYPEDEF );
        int32_t              lTypesCount           = mono_table_info_get_rows( lTypeDefinitionsTable );

        for( int32_t i = 0; i < lTypesCount; i++ )
        {
            uint32_t lCols[MONO_TYPEDEF_SIZE];
            mono_metadata_decode_row( lTypeDefinitionsTable, i, lCols, MONO_TYPEDEF_SIZE );

            const char *lNameSpace = mono_metadata_string_heap( sData->mAppAssemblyImage, lCols[MONO_TYPEDEF_NAMESPACE] );
            const char *lClassName = mono_metadata_string_heap( sData->mAppAssemblyImage, lCols[MONO_TYPEDEF_NAME] );

            std::string lFullName;
            if( strlen( lNameSpace ) != 0 )
            {
                lFullName = fmt::format( "{}.{}", lNameSpace, lClassName );
            }
            else
            {
                if( std::strncmp( lClassName, "<Module>", 8 ) ) continue;

                lFullName = lClassName;
            }

            MonoClass *lMonoClass = mono_class_from_name( sData->mAppAssemblyImage, lNameSpace, lClassName );
            if( lMonoClass == sData->mBaseApplicationClass.mMonoClass ) continue;
            if( lMonoClass == sData->mBaseControllerClass.mMonoClass ) continue;
            if( lMonoClass == sData->mBaseHUDClass.mMonoClass ) continue;
            if( lMonoClass == sData->mBaseComponentClass.mMonoClass ) continue;

            auto lNewScriptClass = New<MonoScriptClass>( lNameSpace, lClassName );

            bool lIsApplicationClass =
                mono_class_is_subclass_of( lNewScriptClass->mMonoClass, sData->mBaseApplicationClass.mMonoClass, false );
            if( lIsApplicationClass ) sData->mApplicationClasses[lFullName] = lNewScriptClass;

            bool lIsControllerClass =
                mono_class_is_subclass_of( lNewScriptClass->mMonoClass, sData->mBaseControllerClass.mMonoClass, false );
            if( lIsControllerClass ) sData->mControllerClasses[lFullName] = lNewScriptClass;

            bool lIsHUDClass = mono_class_is_subclass_of( lNewScriptClass->mMonoClass, sData->mBaseHUDClass.mMonoClass, false );
            if( lIsHUDClass ) sData->mHUDClasses[lFullName] = lNewScriptClass;

            bool lIsComponentClass =
                mono_class_is_subclass_of( lNewScriptClass->mMonoClass, sData->mBaseComponentClass.mMonoClass, false );
            if( lIsComponentClass ) sData->mComponentClasses[lFullName] = lNewScriptClass;

            // This routine is an iterator routine for retrieving the fields in a class.
            // You must pass a gpointer that points to zero and is treated as an opaque handle
            // to iterate over all of the elements. When no more values are available, the return value is NULL.
            int   lFieldCount = mono_class_num_fields( lMonoClass );
            void *lIterator   = nullptr;
            while( MonoClassField *lField = mono_class_get_fields( lMonoClass, &lIterator ) )
            {
                const char *lFieldName = mono_field_get_name( lField );
                uint32_t    lFlags     = mono_field_get_flags( lField );

                if( lFlags & FIELD_ATTRIBUTE_PUBLIC )
                {
                    MonoType        *lMonoFieldType = mono_field_get_type( lField );
                    eScriptFieldType lFieldType     = Mono::Utils::MonoTypeToScriptFieldType( lMonoFieldType );

                    lNewScriptClass->mFields[lFieldName] = { lFieldType, lFieldName, lField };
                }
            }
        }
    }

    void MonoScriptEngine::ReloadAssembly()
    {
        mono_domain_set( mono_get_root_domain(), false );
        if( sData->mAppDomain != nullptr ) mono_domain_unload( sData->mAppDomain );

        LoadCoreAssembly( sData->mCoreAssemblyFilepath );

        if( !sData->mAppAssemblyFilepath.empty() )
        {
            sData->mAppAssembly      = Mono::Utils::LoadMonoAssembly( sData->mAppAssemblyFilepath );
            sData->mAppAssemblyImage = mono_assembly_get_image( sData->mAppAssembly );
        }

        LoadAssemblyClasses();
        RegisterComponentTypes();
    }
} // namespace SE::Core