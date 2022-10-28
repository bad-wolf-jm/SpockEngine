#pragma once

#include <filesystem>
#include <string>
#include <map>


extern "C" {
	typedef struct _MonoClass MonoClass;
	typedef struct _MonoObject MonoObject;
	typedef struct _MonoMethod MonoMethod;
	typedef struct _MonoAssembly MonoAssembly;
	typedef struct _MonoImage MonoImage;
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

        static void LoadAssembly( const std::filesystem::path &filepath );
        // static void LoadAppAssembly( const std::filesystem::path &filepath );

        static void ReloadAssembly();

      private:
        static void InitMono();
        static void ShutdownMono();

        static MonoObject *InstantiateClass( MonoClass *monoClass );
        static void        LoadAssemblyClasses();

        friend class ScriptClass;
        // friend class ScriptGlue;
    };
} // namespace LTSE