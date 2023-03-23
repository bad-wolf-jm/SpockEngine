#include "MonoScriptInstance.h"

#include "mono/jit/jit.h"
#include "mono/metadata/assembly.h"
#include "mono/metadata/object.h"
#include "mono/metadata/tabledefs.h"

#include "MonoRuntime.h"
#include "MonoScriptClass.h"

namespace SE::Core
{
    MonoScriptInstance::MonoScriptInstance( MonoScriptClass *aScriptClass, MonoClass *aMonoClass, MonoObject *aInstance )
        : mScriptClass{ aScriptClass }
        , mMonoClass{ aMonoClass }
        , mInstance{ aInstance }
    {
        mGCHandle = mono_gchandle_new( mInstance, true );
    }

    MonoScriptInstance::~MonoScriptInstance() { mono_gchandle_free( mGCHandle ); }

    MonoMethod *MonoScriptInstance::GetMethod( const std::string &aName, int aParameterCount )
    {
        MonoClass  *lClass  = mMonoClass;
        MonoMethod *lMethod = NULL;
        while( lClass != NULL && lMethod == NULL )
        {
            lMethod = mono_class_get_method_from_name( lClass, aName.c_str(), aParameterCount );
            if( lMethod == NULL ) lClass = mono_class_get_parent( lClass );
        }

        return lMethod;
    }

    MonoObject *MonoScriptInstance::InvokeMethod( MonoMethod *aMethod, void **aParameters )
    {
        MonoObject *lException = nullptr;
        MonoObject *lValue     = mono_runtime_invoke( aMethod, mInstance, aParameters, &lException );

        if( lException == nullptr ) return lValue;

        return nullptr;
    }

    MonoObject *MonoScriptInstance::InvokeMethod( const std::string &aName, int aParameterCount, void **aParameters )
    {
        auto lMethod = GetMethod( aName, aParameterCount );
        return InvokeMethod( lMethod, aParameters );
    }

    Ref<MonoScriptInstance> MonoScriptInstance::GetPropertyValue( std::string const &aName, std::string const &aClassName )
    {
        if( mScriptClass == nullptr ) return nullptr;
        if( mInstance == nullptr ) return nullptr;

        sScriptProperty &aProperty       = mScriptClass->GetProperty( aName );
        MonoMethod      *lPropertyGetter = mono_property_get_get_method( aProperty.mProperty );

        MonoScriptClass &lClass = MonoRuntime::GetClassType( aClassName );

        MonoObject *lException = nullptr;
        MonoObject *lValue = mono_runtime_invoke( lPropertyGetter, mInstance, nullptr, &lException );

        if( lException == nullptr ) return New<MonoScriptInstance>( &lClass, lClass.Class(), lValue );

        return nullptr;
    }

    sScriptProperty &MonoScriptInstance::GetProperty( std::string const &aName ) { return mScriptClass->GetProperty( aName ); }

} // namespace SE::Core