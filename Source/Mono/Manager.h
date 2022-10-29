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
        static MonoImage* GetCoreAssemblyImage();
        
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