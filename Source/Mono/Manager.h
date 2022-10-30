#pragma once

#include <filesystem>
#include <map>
#include <string>

extern "C"
{
    typedef struct _MonoClass      MonoClass;
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

    class ScriptClassInstance
    {
      public:
        ScriptClassInstance() = default;
        ScriptClassInstance( MonoClass *aMonoClass, MonoObject *aInstance );

        MonoMethod *GetMethod( const std::string &aName, int aParameterCount );
        MonoObject *InvokeMethod( MonoMethod *aMethod, void **aParameters = nullptr );
        MonoObject *InvokeMethod( const std::string &aName, int aParameterCount, void **aParameters = nullptr );

      private:
        MonoClass  *mMonoClass = nullptr;
        MonoObject *mInstance  = nullptr;

        friend class ScriptManager;
    };

    class ScriptClass
    {
      public:
        ScriptClass() = default;
        ScriptClass( const std::string &aClassNamespace, const std::string &aClassName, bool aIsCore = false );

        ScriptClassInstance Instantiate();

        MonoMethod *GetMethod( const std::string &aName, int aParameterCount );
        MonoObject *InvokeMethod( MonoObject *aInstance, MonoMethod *aMethod, void **aParameters = nullptr );
        MonoObject *InvokeMethod( const std::string &aName, int aParameterCount, void **aParameters = nullptr );

        const std::map<std::string, sScriptField> &GetFields() const { return mFields; }

      private:
        std::string mClassNamespace;
        std::string mClassName;

        std::map<std::string, sScriptField> mFields;

        MonoClass *mMonoClass = nullptr;

        friend class ScriptManager;
    };

    class ScriptManager
    {
      public:
        ScriptManager()  = default;
        ~ScriptManager() = default;

        static void Initialize();
        static void Shutdown();

        // static void LoadAppAssembly( const std::filesystem::path &aFilepath );
        static void SetAppAssemblyPath( const std::filesystem::path &aFilepath );

        static void ReloadAssembly();

        static MonoImage *GetCoreAssemblyImage();

        static ScriptClass GetClassType( const std::string &aClassName );

      private:
        static void RegisterInternalCppFunctions();
        static void InitMono();
        static void ShutdownMono();

        static void LoadCoreAssembly( const std::filesystem::path &aFilepath );

        static MonoObject *InstantiateClass( MonoClass *aMonoClass );
        static void        LoadAssemblyClasses();

        friend class ScriptClass;
    };
} // namespace LTSE::Core