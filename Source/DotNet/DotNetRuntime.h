#pragma once

#include <filesystem>
#include <functional>
#include <map>
#include <string>

#include "DotNetClass.h"
#include "Typedefs.h"

namespace SE::Core
{
    class DotNetRuntime
    {
      public:
        DotNetRuntime()  = default;
        ~DotNetRuntime() = default;

        static void Initialize( std::filesystem::path &aMonoPath, const std::filesystem::path &aCoreAssemblyPath );
        static void Shutdown();

        static void AddAppAssemblyPath( const std::filesystem::path &aFilepath, std::string const &aCategory );
        static void ReloadAssemblies();

        static uint32_t CountAssemblies();
        static void     GetAssemblies( std::vector<fs::path> &lOut );
        static bool     AssembliesNeedReloading();

        static MonoString *NewString( std::string const &aString );
        static std::string NewString( MonoString *aString );

        template <typename _Ty>
        static std::vector<_Ty> AsVector( MonoObject *aObject )
        {
            uint32_t lArrayLength = static_cast<uint32_t>( mono_array_length( (MonoArray *)aObject ) );

            std::vector<_Ty> lVector( lArrayLength );
            for( uint32_t i = 0; i < lArrayLength; i++ )
            {
                auto lElement = *( mono_array_addr( (MonoArray *)aObject, _Ty, i ) );
                lVector[i]    = lElement;
            }

            return lVector;
        }

        static MonoScriptClass &GetClassType( const std::string &aClassName );

        static MonoType *GetCoreTypeFromName( std::string &aName );

        static void DisplayAssemblies();

        static void OnConsoleOut( std::function<void( std::string const & )> aFunction );

        static std::vector<std::string>                GetClassNames();
        static std::map<std::string, MonoScriptClass> &GetClasses();

      private:
        static void RegisterComponentTypes();
        static void RegisterInternalCppFunctions();
        static void InitMono( std::filesystem::path &aMonoPath );
        static void ShutdownMono();
        static void ConsoleWrite( MonoString *aBuffer );

        static void LoadCoreAssembly( const std::filesystem::path &aFilepath );

        static MonoObject *InstantiateClass( MonoClass *aMonoClass, bool aIsCore = false );
        static void        LoadAssemblyClasses();
        static void        RecreateClassTree();

        friend class DotNetClass;
    };
} // namespace SE::Core