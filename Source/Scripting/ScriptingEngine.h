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

        environment_t LoadFile( fs::path path );

        void Execute( environment_t &environment, std::string string );
        void Execute( std::string string );

        template <typename _Ty>
        script_bindings &Define( std::string name, _Ty value )
        {
            _scriptState[name] = value;
            return *this;
        }

        template <typename _Ty>
        _Ty Get( std::string name )
        {
            return _scriptState.get<_Ty>( name );
        }

        template <typename _Ty>
        _Ty &GetRef( std::string name )
        {
            return _scriptState.get<_Ty>( name );
        }

        template <typename _Ty>
        sol::usertype<_Ty> RegisterPrimitiveType( std::string const &name )
        {
            return declare_primitive_type<_Ty>( _typesModule, name );
        }

      private:
        sol::state _scriptState;
        sol::table _typesModule;
    };
} // namespace SE::Core