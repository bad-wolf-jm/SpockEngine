#pragma once

#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

#include <filesystem>
#include <string>

#include "Cuda/Tensor.h"
#include "Entities/EntityRegistry.h"
#include "PrimitiveTypes.h"

namespace fs = std::filesystem;

namespace SE::Core
{

    using environment_t = sol::environment;

    class script_bindings
    {
      public:
        script_bindings();

        void Initialize();

        environment_t NewEnvironment();

        environment_t LoadFile( fs::path aPath );

        void Execute( environment_t &aEnvironment, std::string aString );
        void Execute( std::string aString );

        template <typename _Ty>
        script_bindings &Define( std::string aName, _Ty aValue )
        {
            _scriptState[aName] = aValue;
            return *this;
        }

        template <typename _Ty>
        _Ty Get( std::string aName )
        {
            return _scriptState.get<_Ty>( aName );
        }

        template <typename _Ty>
        _Ty &GetRef( std::string aName )
        {
            return _scriptState.get<_Ty>( aName );
        }

        template <typename _Ty>
        sol::usertype<_Ty> RegisterPrimitiveType( std::string const &aName )
        {
            return declare_primitive_type<_Ty>( _typesModule, aName );
        }

      private:
        sol::state _scriptState;
        sol::table _typesModule;
    };
} // namespace SE::Core