#include "Manager.h"
#include "Core/Logging.h"
#include "Core/Memory.h"

#include "mono/jit/jit.h"
#include "mono/metadata/assembly.h"
#include "mono/metadata/object.h"
#include "mono/metadata/tabledefs.h"

#include <unordered_map>

namespace LTSE::Core
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

        // std::unique_ptr<filewatch::FileWatch<std::string>> mAppAssemblyFileWatcher;

        bool mAssemblyReloadPending = false;

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

            // NOTE: We can't use this lImage for anything other than loading the aAssembly because this lImage doesn't have a
            // reference to the aAssembly
            MonoImageOpenStatus status;
            MonoImage          *lImage = mono_image_open_from_data_full( fileData, fileSize, 1, &status, 0 );

            if( status != MONO_IMAGE_OK )
            {
                const char *errorMessage = mono_image_strerror( status );
                // Log some error message using the errorMessage data
                return nullptr;
            }

            std::string   pathString = assemblyPath.string();
            MonoAssembly *aAssembly  = mono_assembly_load_from_full( lImage, pathString.c_str(), &status, 0 );
            mono_image_close( lImage );

            // Don't forget to free the file data
            delete[] fileData;

            return aAssembly;
        }

        void PrintAssemblyTypes( MonoAssembly *aAssembly )
        {
            MonoImage           *lImage                = mono_assembly_get_image( aAssembly );
            const MonoTableInfo *lTypeDefinitionsTable = mono_image_get_table_info( lImage, MONO_TABLE_TYPEDEF );
            int32_t              lTypesCount           = mono_table_info_get_rows( lTypeDefinitionsTable );

            for( int32_t i = 0; i < lTypesCount; i++ )
            {
                uint32_t cols[MONO_TYPEDEF_SIZE];
                mono_metadata_decode_row( lTypeDefinitionsTable, i, cols, MONO_TYPEDEF_SIZE );

                const char *lNameSpace = mono_metadata_string_heap( lImage, cols[MONO_TYPEDEF_NAMESPACE] );
                const char *lName      = mono_metadata_string_heap( lImage, cols[MONO_TYPEDEF_NAME] );
                LTSE::Logging::Info( "{}.{}", lNameSpace, lName );
            }
        }

        enum class ScriptFieldType
        {
            None = 0,
            Float,
            Double,
            Bool,
            Char,
            Byte,
            Short,
            Int,
            Long,
            UByte,
            UShort,
            UInt,
            ULong
        };

        struct ScriptField
        {
            ScriptFieldType mType;
            std::string     mName;

            MonoClassField *mClassField;
        };

        static std::unordered_map<std::string, ScriptFieldType> s_ScriptFieldTypeMap = { { "System.Single", ScriptFieldType::Float },
            { "System.Double", ScriptFieldType::Double }, { "System.Boolean", ScriptFieldType::Bool },
            { "System.Char", ScriptFieldType::Char }, { "System.Int16", ScriptFieldType::Short },
            { "System.Int32", ScriptFieldType::Int }, { "System.Int64", ScriptFieldType::Long },
            { "System.Byte", ScriptFieldType::Byte }, { "System.UInt16", ScriptFieldType::UShort },
            { "System.UInt32", ScriptFieldType::UInt }, { "System.UInt64", ScriptFieldType::ULong } };

        ScriptFieldType MonoTypeToScriptFieldType( MonoType *monoType )
        {
            std::string typeName = mono_type_get_name( monoType );

            auto it = s_ScriptFieldTypeMap.find( typeName );
            if( it == s_ScriptFieldTypeMap.end() )
            {
                return ScriptFieldType::None;
            }

            return it->second;
        }

    } // namespace Utils

    class ScriptClass
    {
      public:
        ScriptClass() = default;
        ScriptClass( const std::string &classNamespace, const std::string &className, bool isCore = false );

        MonoObject *Instantiate();
        MonoMethod *GetMethod( const std::string &name, int parameterCount );
        MonoObject *InvokeMethod( MonoObject *instance, MonoMethod *method, void **params = nullptr );

        const std::map<std::string, Utils::ScriptField> &GetFields() const { return m_Fields; }

      private:
        std::string m_ClassNamespace;
        std::string m_ClassName;

        std::map<std::string, Utils::ScriptField> m_Fields;

        MonoClass *m_MonoClass = nullptr;

        friend class ScriptManager;
    };

    MonoObject *ScriptManager::InstantiateClass( MonoClass *monoClass )
    {
        MonoObject *instance = mono_object_new( sData->mAppDomain, monoClass );
        mono_runtime_object_init( instance );
        return instance;
    }

    ScriptClass::ScriptClass( const std::string &classNamespace, const std::string &className, bool isCore )
        : m_ClassNamespace( classNamespace )
        , m_ClassName( className )
    {
        m_MonoClass = mono_class_from_name(
            isCore ? sData->mCoreAssemblyImage : sData->mAppAssemblyImage, classNamespace.c_str(), className.c_str() );
    }

    MonoObject *ScriptClass::Instantiate() { return ScriptManager::InstantiateClass( m_MonoClass ); }

    MonoMethod *ScriptClass::GetMethod( const std::string &name, int parameterCount )
    {
        return mono_class_get_method_from_name( m_MonoClass, name.c_str(), parameterCount );
    }

    MonoObject *ScriptClass::InvokeMethod( MonoObject *instance, MonoMethod *method, void **params )
    {
        return mono_runtime_invoke( method, instance, params, nullptr );
    }

    void ScriptManager::LoadAssembly( const std::filesystem::path &aFilepath )
    {
        sData->mAppDomain = mono_domain_create_appdomain( "SE_Runtime", nullptr );
        mono_domain_set( sData->mAppDomain, true );

        sData->mCoreAssemblyFilepath = aFilepath;
        sData->mCoreAssembly         = Utils::LoadMonoAssembly( aFilepath );
        sData->mCoreAssemblyImage    = mono_assembly_get_image( sData->mCoreAssembly );

        Utils::PrintAssemblyTypes( sData->mCoreAssembly );
    }

    void ScriptManager::Initialize()
    {
        sData = new ScriptEngineData();

        InitMono();
        // ScriptGlue::RegisterFunctions();

        LoadAssembly( "Source\\Mono\\ScriptCore\\Build\\Debug\\SE_Core.dll" );
        // LoadAppAssembly("SandboxProject/Assets/Scripts/Binaries/Sandbox.dll");
        LoadAssemblyClasses();

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

        MonoDomain *lRootDomain = mono_jit_init( "HazelJITRuntime" );

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

    void ScriptManager::LoadAssemblyClasses()
    {
        // sData->EntityClasses.clear();
        if( !sData->mAppAssemblyImage ) return;

        const MonoTableInfo *typeDefinitionsTable = mono_image_get_table_info( sData->mAppAssemblyImage, MONO_TABLE_TYPEDEF );
        int32_t              numTypes             = mono_table_info_get_rows( typeDefinitionsTable );
        // MonoClass           *entityClass          = mono_class_from_name( sData->mCoreAssemblyImage, "Hazel", "Entity" );

        for( int32_t i = 0; i < numTypes; i++ )
        {
            uint32_t cols[MONO_TYPEDEF_SIZE];
            mono_metadata_decode_row( typeDefinitionsTable, i, cols, MONO_TYPEDEF_SIZE );

            const char *nameSpace = mono_metadata_string_heap( sData->mAppAssemblyImage, cols[MONO_TYPEDEF_NAMESPACE] );
            const char *className = mono_metadata_string_heap( sData->mAppAssemblyImage, cols[MONO_TYPEDEF_NAME] );
            std::string fullName;
            if( strlen( nameSpace ) != 0 )
                fullName = fmt::format( "{}.{}", nameSpace, className );
            else
                fullName = className;

            MonoClass *monoClass = mono_class_from_name( sData->mAppAssemblyImage, nameSpace, className );

            // if( monoClass == entityClass ) continue;

            // bool isEntity = mono_class_is_subclass_of( monoClass, entityClass, false );
            // if( !isEntity ) continue;

            Ref<ScriptClass> scriptClass = New<ScriptClass>( nameSpace, className );
            // sData->EntityClasses[fullName] = scriptClass;

            // This routine is an iterator routine for retrieving the fields in a class.
            // You must pass a gpointer that points to zero and is treated as an opaque handle
            // to iterate over all of the elements. When no more values are available, the return value is NULL.

            int fieldCount = mono_class_num_fields( monoClass );
            // HZ_CORE_WARN( "{} has {} fields:", className, fieldCount );
            void *iterator = nullptr;
            while( MonoClassField *field = mono_class_get_fields( monoClass, &iterator ) )
            {
                const char *fieldName = mono_field_get_name( field );
                uint32_t    flags     = mono_field_get_flags( field );
                if( flags & FIELD_ATTRIBUTE_PUBLIC )
                {
                    MonoType              *type      = mono_field_get_type( field );
                    Utils::ScriptFieldType fieldType = Utils::MonoTypeToScriptFieldType( type );
                    // HZ_CORE_WARN( "  {} ({})", fieldName, Utils::ScriptFieldTypeToString( fieldType ) );

                    scriptClass->m_Fields[fieldName] = { fieldType, fieldName, field };
                }
            }
        }

        // auto &entityClasses = sData->EntityClasses;

        // mono_field_get_value()
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

    void ScriptManager::ReloadAssembly()
    {
        mono_domain_set( mono_get_root_domain(), false );

        mono_domain_unload( sData->mAppDomain );

        LoadAssembly( sData->mCoreAssemblyFilepath );
        // LoadAppAssembly( sData->mAppAssemblyFilepath );
        LoadAssemblyClasses();

        // ScriptGlue::RegisterComponents();

        // Retrieve and instantiate class
        // sData->EntityClass = ScriptClass( "Hazel", "Entity", true );
    }
} // namespace LTSE::Core