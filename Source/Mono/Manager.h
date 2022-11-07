#pragma once

#include <filesystem>
#include <map>
#include <string>

extern "C"
{
    typedef struct _MonoClass      MonoClass;
    typedef struct _MonoMethod     MonoMethod;
    typedef struct _MonoString     MonoString;
    typedef struct _MonoType       MonoType;
    typedef struct _MonoObject     MonoObject;
    typedef struct _MonoMethod     MonoMethod;
    typedef struct _MonoAssembly   MonoAssembly;
    typedef struct _MonoImage      MonoImage;
    typedef struct _MonoClassField MonoClassField;
}

namespace LTSE::Core
{

    enum class eScriptFieldType
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
    struct sScriptField
    {
        eScriptFieldType mType;
        std::string      mName;
        MonoClassField  *mClassField;
    };

    class ScriptClassInstance;

    class ScriptClassMethod
    {
      public:
        ScriptClassMethod() = default;
        ScriptClassMethod( MonoMethod *aMonoMethod, ScriptClassInstance *aInstance );

        template <typename... _ArgTypes>
        MonoObject *operator()( _ArgTypes... aArgs )
        {
            void *lParameters[] = { (void *)&aArgs... };

            return mInstance->InvokeMethod( mMonoMethod, lParameters );
        }

      private:
        MonoMethod          *mMonoMethod;
        ScriptClassInstance *mInstance;
    };

    class ScriptClassInstance
    {
      public:
        ScriptClassInstance() = default;
        ScriptClassInstance( MonoClass *aMonoClass, MonoObject *aInstance );

        MonoObject *GetInstance() { return mInstance; };

        void GCAcquire();
        void GCRelease();

        MonoMethod       *GetMethod( const std::string &aName, int aParameterCount );
        ScriptClassMethod GetBoundMethod( const std::string &aName, int aParameterCount );
        MonoObject       *InvokeMethod( MonoMethod *aMethod, void **aParameters = nullptr );
        MonoObject       *InvokeMethod( const std::string &aName, int aParameterCount, void **aParameters = nullptr );

        template <typename... _ArgTypes>
        MonoObject *CallMethod( const std::string &aName, _ArgTypes... aArgs )
        {
            void *lParameters[] = { (void *)&aArgs... };

            return InvokeMethod( aName, sizeof...( _ArgTypes ), lParameters );
        }

        template <typename _StructType>
        _StructType GetFieldValue( std::string const &aName )
        {
            MonoClassField *lClassField = mono_class_get_field_from_name( mMonoClass, aName.c_str() );

            _StructType lValue;
            mono_field_get_value( mInstance, lClassField, &lValue );

            return lValue;
        }

      private:
        MonoClass  *mMonoClass = nullptr;
        MonoObject *mInstance  = nullptr;
        uint32_t    mGCHandle  = 0;

        friend class ScriptManager;
    };

    class ScriptClass
    {
      public:
        ScriptClass() = default;
        ScriptClass( MonoType *aMonoClass );
        ScriptClass( const std::string &aClassNamespace, const std::string &aClassName, bool aIsCore = false );

        ScriptClassInstance Instantiate();

        template <typename... _ArgTypes>
        ScriptClassInstance Instantiate( _ArgTypes... aArgs )
        {
            void *lParameters[] = { (void *)&aArgs... };

            auto lNewInstance = Instantiate();
            lNewInstance.InvokeMethod( ".ctor", sizeof...( _ArgTypes ), lParameters );

            return lNewInstance;
        }

        MonoMethod *GetMethod( const std::string &aName, int aParameterCount );
        MonoObject *InvokeMethod( MonoObject *aInstance, MonoMethod *aMethod, void **aParameters = nullptr );
        MonoObject *InvokeMethod( const std::string &aName, int aParameterCount, void **aParameters = nullptr );

        const std::map<std::string, sScriptField> &GetFields() const { return mFields; }

      private:
        std::string mClassNamespace;
        std::string mClassName;

        std::map<std::string, sScriptField> mFields;

        MonoClass *mMonoClass = nullptr;
        bool       mIsCore    = false;

        friend class ScriptManager;
    };

    class ScriptManager
    {
      public:
        ScriptManager()  = default;
        ~ScriptManager() = default;

        static void Initialize( std::filesystem::path &aMonoPath, const std::filesystem::path &aCoreAssemblyPath );
        static void Shutdown();

        static void SetAppAssemblyPath( const std::filesystem::path &aFilepath );

        static void ReloadAssembly();

        static MonoImage *GetCoreAssemblyImage();

        static MonoString *NewString( std::string const &aString );
        static std::string NewString( MonoString *aString );

        static ScriptClass GetClassType( const std::string &aClassName );

        static void *GetSceneContext();

      private:
        static void RegisterComponentTypes();
        static void RegisterInternalCppFunctions();
        static void InitMono( std::filesystem::path &aMonoPath );
        static void ShutdownMono();

        static void LoadCoreAssembly( const std::filesystem::path &aFilepath );

        static MonoObject *InstantiateClass( MonoClass *aMonoClass, bool aIsCore = false );
        static void        LoadAssemblyClasses();

        friend class ScriptClass;
    };
} // namespace LTSE::Core