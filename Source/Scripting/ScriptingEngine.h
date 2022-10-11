#pragma once

#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

#include <filesystem>
#include <string>

#include "Cuda/Tensor.h"
#include "Entities/EntityRegistry.h"
#include "PrimitiveTypes.h"

namespace fs = std::filesystem;

namespace LTSE::Core
{

    using ScriptEnvironment = sol::environment;

    class ScriptingEngine
    {
      public:
        ScriptingEngine();

        void Initialize();

        ScriptEnvironment NewEnvironment();

        ScriptEnvironment LoadFile( fs::path aPath );

        void Execute( ScriptEnvironment &aEnvironment, std::string aString );
        void Execute( std::string aString );

        template <typename _Ty> ScriptingEngine &Define( std::string aName, _Ty aValue )
        {
            ScriptState[aName] = aValue;
            return *this;
        }

        template <typename _Ty> _Ty Get( std::string aName ) { return ScriptState.get<_Ty>( aName ); }
        template <typename _Ty> _Ty &GetRef( std::string aName ) { return ScriptState.get<_Ty>( aName ); }
        template <typename _Ty> Ref<_Ty> GetSharedPtr( std::string aName ) { return ScriptState.get<Ref<_Ty>>( aName ); }

        template <typename _Ty> sol::usertype<_Ty> RegisterPrimitiveType( std::string const &aName )
        {
            return DeclarePrimitiveType<_Ty>( mTypesModule, aName );
        }

        template <typename _Ty> sol::usertype<_Ty> RegisterType( std::string const &aName )
        {
            return ScriptState.new_usertype<_Ty>( aName );
        }

      private:
        sol::state ScriptState;
        sol::table mTypesModule;
    };
} // namespace LTSE::Core