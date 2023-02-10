#pragma once

#include <filesystem>
#include <map>
#include <string>

#include "MonoScriptClass.h"
#include "MonoTypedefs.h"

namespace SE::Core
{
    class MonoRuntime
    {
      public:
        MonoRuntime()  = default;
        ~MonoRuntime() = default;

        static void Initialize( std::filesystem::path &aMonoPath, const std::filesystem::path &aCoreAssemblyPath );
        static void Shutdown();

        static void AddAppAssemblyPath( const std::filesystem::path &aFilepath );
        static void ReloadAssemblies();

        static uint32_t CountAssemblies();
        static void GetAssemblies(std::vector<fs::path> &lOut);
        static bool AssembliesNeedReloading();

        static MonoImage *GetCoreAssemblyImage();
        static MonoImage *GetAppAssemblyImage();

        static MonoString *NewString( std::string const &aString );
        static std::string NewString( MonoString *aString );

        static MonoScriptClass &GetClassType( const std::string &aClassName );

        static MonoType *GetCoreTypeFromName( std::string &aName );

      private:
        static void RegisterComponentTypes();
        static void RegisterInternalCppFunctions();
        static void InitMono( std::filesystem::path &aMonoPath );
        static void ShutdownMono();

        static void LoadCoreAssembly( const std::filesystem::path &aFilepath );

        static MonoObject *InstantiateClass( MonoClass *aMonoClass, bool aIsCore = false );
        static void        LoadAssemblyClasses();
        static void        RecreateClassTree();

        friend class MonoScriptClass;
    };
} // namespace SE::Core