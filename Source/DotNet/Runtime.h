#pragma once

#include <filesystem>
#include <functional>
#include <map>
#include <string>

#include "Class.h"
#include "Typedefs.h"
#include "Core/String.h"
namespace SE::Core
{
    class DotNetRuntime
    {
      public:
        DotNetRuntime()  = default;
        ~DotNetRuntime() = default;

        static void Initialize( path_t &aMonoPath, const path_t &aCoreAssemblyPath );
        static void Shutdown();

        static void AddAppAssemblyPath( const path_t &aFilepath, string_t const &aCategory );
        static void ReloadAssemblies();

        // static uint32_t CountAssemblies();
        // static void     GetAssemblies( std::vector<path_t> &lOut );
        // static bool     AssembliesNeedReloading();

        static MonoString *NewString( string_t const &aString );
        static string_t NewString( MonoString *aString );

        template <typename _Ty>
        static std::vector<_Ty> AsVector( MonoObject *aObject )
        {
            if( aObject == nullptr ) return std::vector<_Ty>( 0 );

            uint32_t lArrayLength = static_cast<uint32_t>( mono_array_length( (MonoArray *)aObject ) );

            std::vector<_Ty> lVector( lArrayLength );
            for( uint32_t i = 0; i < lArrayLength; i++ )
            {
                auto lElement = *( mono_array_addr( (MonoArray *)aObject, _Ty, i ) );
                lVector[i]    = lElement;
            }

            return lVector;
        }

        static DotNetClass &GetClassType( const string_t &aClassName );

        static MonoType *GetCoreTypeFromName( string_t &aName );

        // static std::vector<string_t>            GetClassNames();
        // static std::map<string_t, DotNetClass> &GetClasses();

      private:
        static void RegisterComponentTypes();
        static void RegisterInternalCppFunctions();
        static void InitMono( path_t &aMonoPath );
        static void ShutdownMono();
        static void ConsoleWrite( MonoString *aBuffer );

        static void LoadCoreAssembly( const path_t &aFilepath );

        static MonoObject *InstantiateClass( MonoClass *aMonoClass, bool aIsCore = false );
        // static void        LoadAssemblyClasses();
        // static void        RecreateClassTree();

        friend class DotNetClass;
    };
} // namespace SE::Core