#include "Manager.h"

#include "Core/Logging.h"
#include "Core/Memory.h"

#include "Scene/Scene.h"

#include "mono/jit/jit.h"
#include "mono/metadata/assembly.h"
#include "mono/metadata/object.h"
#include "mono/metadata/tabledefs.h"

#include <unordered_map>

// #include <FileWatch.hpp>

#include "EntityRegistry.h"
#include "InternalCalls.h"

namespace LTSE::Core
{

    namespace Utils
    {

        // TODO: move to FileSystem class
        static char *ReadBytes( const std::filesystem::path &aFilepath, uint32_t *aOutSize )
        {
            std::ifstream lStream( aFilepath, std::ios::binary | std::ios::ate );

            if( !lStream ) return nullptr;

            std::streampos end = lStream.tellg();
            lStream.seekg( 0, std::ios::beg );
            uint64_t size = end - lStream.tellg();

            if( size == 0 ) return nullptr;

            char *buffer = new char[size];
            lStream.read( (char *)buffer, size );
            lStream.close();

            *aOutSize = (uint32_t)size;
            return buffer;
        }

        static MonoAssembly *LoadMonoAssembly( const std::filesystem::path &lAssemblyPath )
        {
            uint32_t lFileSize = 0;
            char    *lFileData = ReadBytes( lAssemblyPath, &lFileSize );

            MonoImageOpenStatus lStatus;
            MonoImage          *lImage = mono_image_open_from_data_full( lFileData, lFileSize, 1, &lStatus, 0 );

            if( lStatus != MONO_IMAGE_OK )
            {
                const char *lErrorMessage = mono_image_strerror( lStatus );
                return nullptr;
            }

            std::string   lPathString = lAssemblyPath.string();
            MonoAssembly *lAssembly   = mono_assembly_load_from_full( lImage, lPathString.c_str(), &lStatus, 0 );
            mono_image_close( lImage );

            delete[] lFileData;

            return lAssembly;
        }

        void PrintAssemblyTypes( MonoAssembly *aAssembly )
        {
            MonoImage           *lImage                = mono_assembly_get_image( aAssembly );
            const MonoTableInfo *lTypeDefinitionsTable = mono_image_get_table_info( lImage, MONO_TABLE_TYPEDEF );
            int32_t              lTypesCount           = mono_table_info_get_rows( lTypeDefinitionsTable );

            for( int32_t i = 0; i < lTypesCount; i++ )
            {
                uint32_t lCols[MONO_TYPEDEF_SIZE];
                mono_metadata_decode_row( lTypeDefinitionsTable, i, lCols, MONO_TYPEDEF_SIZE );

                const char *lNameSpace = mono_metadata_string_heap( lImage, lCols[MONO_TYPEDEF_NAMESPACE] );
                const char *lName      = mono_metadata_string_heap( lImage, lCols[MONO_TYPEDEF_NAME] );
                LTSE::Logging::Info( "{}.{}", lNameSpace, lName );
            }
        }

        static std::unordered_map<std::string, eScriptFieldType> sScriptFieldTypeMap = {
            { "System.Single", eScriptFieldType::Float },  { "System.Double", eScriptFieldType::Double },
            { "System.Boolean", eScriptFieldType::Bool },  { "System.Char", eScriptFieldType::Char },
            { "System.Int16", eScriptFieldType::Short },   { "System.Int32", eScriptFieldType::Int },
            { "System.Int64", eScriptFieldType::Long },    { "System.Byte", eScriptFieldType::Byte },
            { "System.UInt16", eScriptFieldType::UShort }, { "System.UInt32", eScriptFieldType::UInt },
            { "System.UInt64", eScriptFieldType::ULong } };

        eScriptFieldType MonoTypeToScriptFieldType( MonoType *aMonoType )
        {
            std::string typeName = mono_type_get_name( aMonoType );

            auto it = sScriptFieldTypeMap.find( typeName );
            if( it == sScriptFieldTypeMap.end() ) return eScriptFieldType::None;

            return it->second;
        }

    } // namespace Utils

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

        ScriptClass mBaseApplicationClass;
        ScriptClass mBaseControllerClass;
        ScriptClass mBaseComponentClass;

        std::unordered_map<std::string, Ref<ScriptClass>> mApplicationClasses;
        std::unordered_map<std::string, Ref<ScriptClass>> mControllerClasses;
        std::unordered_map<std::string, Ref<ScriptClass>> mComponentClasses;
        std::unordered_map<std::string, Ref<ScriptClass>> mAllClasses;

        void *mSceneContext = nullptr;

        bool mAssemblyReloadPending = false;
    };

    static ScriptEngineData *sData = nullptr;

    MonoObject *ScriptManager::InstantiateClass( MonoClass *aMonoClass, bool aIsCore )
    {
        MonoObject *aInstance = mono_object_new( sData->mAppDomain, aMonoClass );
        mono_runtime_object_init( aInstance );
        return aInstance;
    }

    ScriptClass::ScriptClass( const std::string &aClassNamespace, const std::string &aClassName, bool aIsCore )
        : mClassNamespace( aClassNamespace )
        , mClassName( aClassName )
    {
        mMonoClass = mono_class_from_name( aIsCore ? sData->mCoreAssemblyImage : sData->mAppAssemblyImage, aClassNamespace.c_str(),
                                           aClassName.c_str() );

        int   lFieldCount = mono_class_num_fields( mMonoClass );
        void *lIterator   = nullptr;
        while( MonoClassField *lField = mono_class_get_fields( mMonoClass, &lIterator ) )
        {
            const char *lFieldName = mono_field_get_name( lField );
            uint32_t    lFlags     = mono_field_get_flags( lField );

            if( lFlags & FIELD_ATTRIBUTE_PUBLIC )
            {
                MonoType        *lMonoFieldType = mono_field_get_type( lField );
                eScriptFieldType lFieldType     = Utils::MonoTypeToScriptFieldType( lMonoFieldType );

                mFields[lFieldName] = { lFieldType, lFieldName, lField };
            }
        }
    }

    ScriptClass::ScriptClass( MonoType *aMonoClass )
        : mMonoClass{ mono_class_from_mono_type( aMonoClass ) }
    {
        int   lFieldCount = mono_class_num_fields( mMonoClass );
        void *lIterator   = nullptr;
        while( MonoClassField *lField = mono_class_get_fields( mMonoClass, &lIterator ) )
        {
            const char *lFieldName = mono_field_get_name( lField );
            uint32_t    lFlags     = mono_field_get_flags( lField );

            if( lFlags & FIELD_ATTRIBUTE_PUBLIC )
            {
                MonoType        *lMonoFieldType = mono_field_get_type( lField );
                eScriptFieldType lFieldType     = Utils::MonoTypeToScriptFieldType( lMonoFieldType );

                mFields[lFieldName] = { lFieldType, lFieldName, lField };
            }
        }
    }

    ScriptClassMethod::ScriptClassMethod( MonoMethod *aMonoMethod, ScriptClassInstance *aInstance )
        : mMonoMethod{ aMonoMethod }
        , mInstance{ aInstance }
    {
    }

    ScriptClassInstance ScriptClass::Instantiate()
    {
        MonoObject *lInstance = ScriptManager::InstantiateClass( mMonoClass, mIsCore );

        return ScriptClassInstance( mMonoClass, lInstance );
    }

    ScriptClassInstance::ScriptClassInstance( MonoClass *aMonoClass, MonoObject *aInstance )
        : mMonoClass{ aMonoClass }
        , mInstance{ aInstance }
    {
    }

    ScriptClassMethod ScriptClassInstance::GetBoundMethod( const std::string &aName, int aParameterCount )
    {
        MonoClass  *lClass  = mMonoClass;
        MonoMethod *lMethod = NULL;
        while( lClass != NULL && lMethod == NULL )
        {
            lMethod = mono_class_get_method_from_name( lClass, aName.c_str(), aParameterCount );
            if( lMethod == NULL ) lClass = mono_class_get_parent( lClass );
        }
        return ScriptClassMethod( lMethod, this );
    }

    MonoMethod *ScriptClassInstance::GetMethod( const std::string &aName, int aParameterCount )
    {
        MonoClass  *lClass  = mMonoClass;
        MonoMethod *lMethod = NULL;
        while( lClass != NULL && lMethod == NULL )
        {
            lMethod = mono_class_get_method_from_name( lClass, aName.c_str(), aParameterCount );
            if( lMethod == NULL ) lClass = mono_class_get_parent( lClass );
        }

        return lMethod;
    }

    MonoObject *ScriptClassInstance::InvokeMethod( MonoMethod *aMethod, void **aParameters )
    {
        return mono_runtime_invoke( aMethod, mInstance, aParameters, nullptr );
    }

    MonoObject *ScriptClassInstance::InvokeMethod( const std::string &aName, int aParameterCount, void **aParameters )
    {
        auto lMethod = GetMethod( aName, aParameterCount );
        return InvokeMethod( lMethod, aParameters );
    }

    void ScriptManager::LoadCoreAssembly( const std::filesystem::path &aFilepath )
    {
        sData->mAppDomain = mono_domain_create_appdomain( "SE_Runtime", nullptr );
        mono_domain_set( sData->mAppDomain, true );

        sData->mCoreAssemblyFilepath = aFilepath;
        sData->mCoreAssembly         = Utils::LoadMonoAssembly( aFilepath );
        sData->mCoreAssemblyImage    = mono_assembly_get_image( sData->mCoreAssembly );

        sData->mBaseApplicationClass = ScriptClass( "SpockEngine", "SEApplication", true );
        sData->mBaseControllerClass  = ScriptClass( "SpockEngine", "ActorComponent", true );
        sData->mBaseComponentClass   = ScriptClass( "SpockEngine", "Component", true );

        Utils::PrintAssemblyTypes( sData->mCoreAssembly );
    }

    // static void OnAppAssemblyFileSystemEvent( const std::string &path, const filewatch::Event change_type )
    // {
    //     if( !sData->mAssemblyReloadPending && change_type == filewatch::Event::modified )
    //     {
    //         sData->mAssemblyReloadPending = true;

    //         // Application::Get().SubmitToMainThread(
    //         //     []()
    //         //     {
    //         //         sData->mAppAssemblyFileWatcher.reset();
    //         //         ScriptEngine::ReloadAssembly();
    //         //     } );
    //     }
    // }

    void ScriptManager::SetAppAssemblyPath( const std::filesystem::path &aFilepath )
    {
        sData->mAppAssemblyFilepath = aFilepath;

        sData->mAssemblyReloadPending = false;

        // if( !sData->mAppAssemblyFilepath.empty() )
        //     sData->mAppAssemblyFileWatcher =
        //         std::make_unique<filewatch::FileWatch<std::string>>( aFilepath.string(), OnAppAssemblyFileSystemEvent );

        ReloadAssembly();
    }

    void ScriptManager::Initialize( std::filesystem::path &aMonoPath, const std::filesystem::path &aCoreAssemblyPath )
    {
        sData = new ScriptEngineData();

        InitMono( aMonoPath );

        RegisterInternalCppFunctions();

        LoadCoreAssembly( aCoreAssemblyPath );
    }

    void ScriptManager::RegisterComponentTypes()
    {
        //
        RegisterComponentType<sTag>();
        RegisterComponentType<sNodeTransformComponent>();
    }

    void ScriptManager::RegisterInternalCppFunctions()
    {
        using namespace MonoInternalCalls;

        SE_ADD_INTERNAL_CALL( Entity_IsValid );
        SE_ADD_INTERNAL_CALL( Entity_Has );
        SE_ADD_INTERNAL_CALL( Entity_Get );
        SE_ADD_INTERNAL_CALL( Entity_Replace );
    }

    void ScriptManager::Shutdown()
    {
        ShutdownMono();

        delete sData;
    }

    void ScriptManager::InitMono( std::filesystem::path &aMonoPath )
    {
        mono_set_assemblies_path( aMonoPath.string().c_str() );

        MonoDomain *lRootDomain = mono_jit_init( "SpockEngineRuntime" );

        sData->mRootDomain = lRootDomain;
    }

    void ScriptManager::ShutdownMono()
    {
        mono_domain_set( mono_get_root_domain(), false );

        mono_domain_unload( sData->mAppDomain );
        sData->mAppDomain = nullptr;

        mono_jit_cleanup( sData->mRootDomain );
        sData->mRootDomain = nullptr;
    }

    MonoImage *ScriptManager::GetCoreAssemblyImage() { return sData->mCoreAssemblyImage; }

    void *ScriptManager::GetSceneContext() { return sData->mSceneContext; }

    MonoString *ScriptManager::NewString( std::string const &aString )
    {
        return mono_string_new( sData->mAppDomain, aString.c_str() );
    }

    std::string ScriptManager::NewString( MonoString *aString ) { return std::string( mono_string_to_utf8( aString ) ); }

    void ScriptManager::LoadAssemblyClasses()
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
            if( lMonoClass == sData->mBaseComponentClass.mMonoClass ) continue;

            auto lNewScriptClass = New<ScriptClass>( lNameSpace, lClassName );

            bool lIsApplicationClass =
                mono_class_is_subclass_of( lNewScriptClass->mMonoClass, sData->mBaseApplicationClass.mMonoClass, false );
            if( lIsApplicationClass ) sData->mApplicationClasses[lFullName] = lNewScriptClass;

            bool lIsControllerClass =
                mono_class_is_subclass_of( lNewScriptClass->mMonoClass, sData->mBaseControllerClass.mMonoClass, false );
            if( lIsControllerClass ) sData->mControllerClasses[lFullName] = lNewScriptClass;

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
                    eScriptFieldType lFieldType     = Utils::MonoTypeToScriptFieldType( lMonoFieldType );

                    lNewScriptClass->mFields[lFieldName] = { lFieldType, lFieldName, lField };
                }
            }
        }
    }

    void ScriptManager::ReloadAssembly()
    {
        mono_domain_set( mono_get_root_domain(), false );
        if( sData->mAppDomain != nullptr ) mono_domain_unload( sData->mAppDomain );

        LoadCoreAssembly( sData->mCoreAssemblyFilepath );

        if( !sData->mAppAssemblyFilepath.empty() )
        {
            sData->mAppAssembly      = Utils::LoadMonoAssembly( sData->mAppAssemblyFilepath );
            sData->mAppAssemblyImage = mono_assembly_get_image( sData->mAppAssembly );

            // Utils::PrintAssemblyTypes( sData->mAppAssembly );
        }

        LoadAssemblyClasses();
        RegisterComponentTypes();
    }
} // namespace LTSE::Core