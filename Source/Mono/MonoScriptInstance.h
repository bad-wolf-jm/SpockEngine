#pragma once

#include <filesystem>
#include <map>
#include <string>

#include "MonoTypedefs.h"

namespace SE::Core
{
    class MonoScriptInstance
    {
      public:
        MonoScriptInstance() = default;
        MonoScriptInstance( MonoClass *aMonoClass, MonoObject *aInstance );
        
        ~MonoScriptInstance();

        MonoObject *GetInstance() { return mInstance; };
        MonoMethod *GetMethod( const std::string &aName, int aParameterCount );
        MonoObject *InvokeMethod( MonoMethod *aMethod, void **aParameters = nullptr );
        MonoObject *InvokeMethod( const std::string &aName, int aParameterCount, void **aParameters = nullptr );

        // void Grab();
        // void Release();

        template <typename... _ArgTypes>
        MonoObject *CallMethod( const std::string &aName, _ArgTypes... aArgs )
        {
            if constexpr( sizeof...( _ArgTypes ) == 0 )
            {
                return InvokeMethod( aName, 0, nullptr );
            }
            else
            {
                void *lParameters[] = { (void *)&aArgs... };

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

        operator bool() const { return ( mInstance != nullptr ) && ( mMonoClass != nullptr ); }

      private:
        MonoClass  *mMonoClass = nullptr;
        MonoObject *mInstance  = nullptr;
        uint32_t    mGCHandle  = 0;

        friend class MonoScriptEngine;
    };

} // namespace SE::Core