#include "Manager.h"
#include "Core/Logging.h"

#include "mono/jit/jit.h"
#include "mono/metadata/assembly.h"
#include "mono/metadata/object.h"
#include "mono/metadata/tabledefs.h"

namespace LTSE
{
    struct ScriptEngineData
    {
        MonoDomain *mRootDomain = nullptr;
        MonoDomain *mAppDomain  = nullptr;

        MonoAssembly *mCoreAssembly      = nullptr;
        MonoImage    *mCoreAssemblyImage = nullptr;

        MonoAssembly *mAppAssembly      = nullptr;
        MonoImage    *mAppAssemblyImage = nullptr;

        std::filesystem::path mCoreAssemblyFilepath;
        std::filesystem::path mAppAssemblyFilepath;

        // ScriptClass EntityClass;

        // std::unordered_map<std::string, Ref<ScriptClass>> EntityClasses;
        // std::unordered_map<UUID, Ref<ScriptInstance>>     EntityInstances;
        // std::unordered_map<UUID, ScriptFieldMap>          EntityScriptFields;

        // Scope<filewatch::FileWatch<std::string>> AppAssemblyFileWatcher;
        // bool                                     AssemblyReloadPending = false;

        // Runtime

        // Scene *SceneContext = nullptr;
    };

    static ScriptEngineData *sData = nullptr;

    namespace Utils
    {

        // TODO: move to FileSystem class
        static char *ReadBytes( const std::filesystem::path &aFilepath, uint32_t *outSize )
        {
            std::ifstream stream( aFilepath, std::ios::binary | std::ios::ate );

            if( !stream ) return nullptr;

            std::streampos end = stream.tellg();
            stream.seekg( 0, std::ios::beg );
            uint64_t size = end - stream.tellg();

            if( size == 0 ) return nullptr;

            char *buffer = new char[size];
            stream.read( (char *)buffer, size );
            stream.close();

            *outSize = (uint32_t)size;
            return buffer;
        }

        static MonoAssembly *LoadMonoAssembly( const std::filesystem::path &assemblyPath )
        {
            uint32_t fileSize = 0;
            char    *fileData = ReadBytes( assemblyPath, &fileSize );

            // NOTE: We can't use this image for anything other than loading the assembly because this image doesn't have a reference
            // to the assembly
            MonoImageOpenStatus status;
            MonoImage          *image = mono_image_open_from_data_full( fileData, fileSize, 1, &status, 0 );

            if( status != MONO_IMAGE_OK )
            {
                const char *errorMessage = mono_image_strerror( status );
                // Log some error message using the errorMessage data
                return nullptr;
            }

            std::string   pathString = assemblyPath.string();
            MonoAssembly *assembly   = mono_assembly_load_from_full( image, pathString.c_str(), &status, 0 );
            mono_image_close( image );

            // Don't forget to free the file data
            delete[] fileData;

            return assembly;
        }

        void PrintAssemblyTypes( MonoAssembly *assembly )
        {
            MonoImage           *image                = mono_assembly_get_image( assembly );
            const MonoTableInfo *typeDefinitionsTable = mono_image_get_table_info( image, MONO_TABLE_TYPEDEF );
            int32_t              numTypes             = mono_table_info_get_rows( typeDefinitionsTable );

            for( int32_t i = 0; i < numTypes; i++ )
            {
                uint32_t cols[MONO_TYPEDEF_SIZE];
                mono_metadata_decode_row( typeDefinitionsTable, i, cols, MONO_TYPEDEF_SIZE );

                const char *nameSpace = mono_metadata_string_heap( image, cols[MONO_TYPEDEF_NAMESPACE] );
                const char *name      = mono_metadata_string_heap( image, cols[MONO_TYPEDEF_NAME] );
                LTSE::Logging::Info( "{}.{}", nameSpace, name );
            }
        }

        // ScriptFieldType MonoTypeToScriptFieldType( MonoType *monoType )
        // {
        //     std::string typeName = mono_type_get_name( monoType );

        //     auto it = s_ScriptFieldTypeMap.find( typeName );
        //     if( it == s_ScriptFieldTypeMap.end() )
        //     {
        //         HZ_CORE_ERROR( "Unknown type: {}", typeName );
        //         return ScriptFieldType::None;
        //     }

        //     return it->second;
        // }

    } // namespace Utils

    void ScriptManager::LoadAssembly( const std::filesystem::path &aFilepath )
    {
        sData->mAppDomain = mono_domain_create_appdomain( "SE_Runtime", nullptr );
        mono_domain_set( sData->mAppDomain, true );

        sData->mCoreAssemblyFilepath = aFilepath;
        sData->mCoreAssembly         = Utils::LoadMonoAssembly( aFilepath );
        sData->mCoreAssemblyImage    = mono_assembly_get_image( sData->mCoreAssembly );

        Utils::PrintAssemblyTypes(sData->mCoreAssembly);
    }

    void ScriptManager::Initialize()
    {
        sData = new ScriptEngineData();

        InitMono();
        // ScriptGlue::RegisterFunctions();

        // LoadAssembly("Resources/Scripts/ScriptCore.dll");
        // LoadAppAssembly("SandboxProject/Assets/Scripts/Binaries/Sandbox.dll");
        // LoadAssemblyClasses();

        // ScriptGlue::RegisterComponents();

        // Retrieve and instantiate class
        // sData->EntityClass = ScriptClass("Hazel", "Entity", true);
        // #if 0

        // MonoObject *instance = sData->EntityClass.Instantiate();

        // // Call method
        // MonoMethod *printMessageFunc = sData->EntityClass.GetMethod( "PrintMessage", 0 );
        // sData->EntityClass.InvokeMethod( instance, printMessageFunc );

        // // Call method with param
        // MonoMethod *printIntFunc = sData->EntityClass.GetMethod( "PrintInt", 1 );

        // int   value = 5;
        // void *param = &value;

        // sData->EntityClass.InvokeMethod( instance, printIntFunc, &param );

        // MonoMethod *printIntsFunc = sData->EntityClass.GetMethod( "PrintInts", 2 );
        // int         value2        = 508;
        // void       *params[2]     = { &value, &value2 };
        // sData->EntityClass.InvokeMethod( instance, printIntsFunc, params );

        // MonoString *monoString             = mono_string_new( sData->mAppDomain, "Hello World from C++!" );
        // MonoMethod *printCustomMessageFunc = sData->EntityClass.GetMethod( "PrintCustomMessage", 1 );
        // void       *stringParam            = monoString;
        // sData->EntityClass.InvokeMethod( instance, printCustomMessageFunc, &stringParam );

        // HZ_CORE_ASSERT(false);
        // #endif
    }

    void ScriptManager::Shutdown()
    {
        ShutdownMono();

        delete sData;
    }

    void ScriptManager::InitMono()
    {
        mono_set_assemblies_path( "C:\\GitLab\\SpockEngine\\mono\\lib" );

        MonoDomain *rootDomain = mono_jit_init( "HazelJITRuntime" );
        // HZ_CORE_ASSERT( rootDomain );

        // Store the root domain pointer
        sData->mRootDomain = rootDomain;
    }

    void ScriptManager::ShutdownMono()
    {
        mono_domain_set( mono_get_root_domain(), false );

        mono_domain_unload( sData->mAppDomain );
        sData->mAppDomain = nullptr;

        mono_jit_cleanup( sData->mRootDomain );
        sData->mRootDomain = nullptr;
    }

} // namespace LTSE