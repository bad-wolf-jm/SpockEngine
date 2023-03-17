#pragma once

#include "Core/Memory.h"
#include <filesystem>
#include <map>
#include <string>
#include <type_traits>

#include "MonoTypedefs.h"

#include "mono/jit/jit.h"
#include "mono/metadata/assembly.h"
#include "mono/metadata/mono-config.h"
#include "mono/metadata/object.h"
#include "mono/metadata/tabledefs.h"

namespace SE::Core
{
    class DotNetClass;

    class DotNetInstance
    {
      public:
        DotNetInstance() = default;
        DotNetInstance( DotNetClass *aScriptClass, MonoClass *aMonoClass, MonoObject *aInstance );
        DotNetInstance( DotNetClass *aScriptClass, MonoClass *aMonoClass, void *aInstance )
            : DotNetInstance( aScriptClass, aMonoClass, (MonoObject *)aInstance )
        {
        }
        DotNetInstance( MonoClass *aMonoClass, MonoObject *aInstance )
            : DotNetInstance( nullptr, aMonoClass, aInstance )
        {
        }

        ~DotNetInstance();

        MonoObject *GetInstance() { return mInstance; };
        MonoMethod *GetMethod( const std::string &aName, int aParameterCount );
        MonoObject *InvokeMethod( MonoMethod *aMethod, void **aParameters = nullptr );
        MonoObject *InvokeMethod( const std::string &aName, int aParameterCount, void **aParameters = nullptr );

        template <typename... _ArgTypes>
        MonoObject *CallMethod( const std::string &aName, _ArgTypes... aArgs )
        {
            if constexpr( sizeof...( _ArgTypes ) == 0 )
            {
                return InvokeMethod( aName, 0, nullptr );
            }
            else
            {
                void *lParameters[] = { (void *)aArgs... };

                return InvokeMethod( aName, sizeof...( _ArgTypes ), lParameters );
            }
        }

        template <typename _Ty>
        _Ty GetFieldValue( std::string const &aName )
        {
            MonoClassField *lClassField = mono_class_get_field_from_name( mMonoClass, aName.c_str() );

            _Ty lValue;
            mono_field_get_value( mInstance, lClassField, &lValue );

            return lValue;
        }

        Ref<DotNetInstance> GetPropertyValue( std::string const &aName, std::string const &aClassName );

        template <typename _Ty>
        _Ty GetPropertyValue( std::string const &aName )
        {
            if( mScriptClass == nullptr ) return _Ty{};

            sScriptProperty &lProperty       = GetProperty( aName );
            MonoMethod      *lPropertyGetter = mono_property_get_get_method( lProperty.mProperty );

            MonoObject *lException = nullptr;
            MonoObject *lValue     = mono_runtime_invoke( lPropertyGetter, mInstance, nullptr, &lException );

            if constexpr( std::is_same<_Ty, MonoString *>::value || std::is_same<_Ty, MonoObject *>::value )
                return ( lException == nullptr ) ? (_Ty)lValue : nullptr;
            else
                return ( lException == nullptr ) ? *(_Ty *)mono_object_unbox( lValue ) : _Ty{};
        }

        operator bool() const { return ( mInstance != nullptr ) && ( mMonoClass != nullptr ); }

      private:
        DotNetClass *mScriptClass = nullptr;
        MonoClass       *mMonoClass   = nullptr;
        MonoObject      *mInstance    = nullptr;
        uint32_t         mGCHandle    = 0;

        friend class MonoScriptEngine;

      private:
        sScriptProperty &GetProperty( std::string const &aName );
    };

} // namespace SE::Core