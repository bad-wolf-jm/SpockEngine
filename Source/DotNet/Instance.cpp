#include "Instance.h"

#include "mono/jit/jit.h"
#include "mono/metadata/assembly.h"
#include "mono/metadata/object.h"
#include "mono/metadata/tabledefs.h"

#include "Runtime.h"
#include "Class.h"

namespace SE::Core
{
    DotNetInstance::DotNetInstance( DotNetClass *aScriptClass, MonoClass *aMonoClass, MonoObject *aInstance )
        : mScriptClass{ aScriptClass }
        , mMonoClass{ aMonoClass }
        , mInstance{ aInstance }
    {
        mGCHandle = mono_gchandle_new( mInstance, true );
    }

    DotNetInstance::~DotNetInstance() { mono_gchandle_free( mGCHandle ); }

    MonoMethod *DotNetInstance::GetMethod( const std::string &aName, int aParameterCount )
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

    MonoObject *DotNetInstance::InvokeMethod( MonoMethod *aMethod, void **aParameters )
    {
        MonoObject *lException = nullptr;
        MonoObject *lValue     = mono_runtime_invoke( aMethod, mInstance, aParameters, &lException );

        if( lException == nullptr ) return lValue;

        return nullptr;
    }

    MonoObject *DotNetInstance::InvokeMethod( const std::string &aName, int aParameterCount, void **aParameters )
    {
        auto lMethod = GetMethod( aName, aParameterCount );

        return InvokeMethod( lMethod, aParameters );
    }

    Ref<DotNetInstance> DotNetInstance::GetPropertyValue( std::string const &aName, std::string const &aClassName )
    {
        if( mScriptClass == nullptr ) return nullptr;
        if( mInstance == nullptr ) return nullptr;

        sScriptProperty &aProperty       = mScriptClass->GetProperty( aName );
        MonoMethod      *lPropertyGetter = mono_property_get_get_method( aProperty.mProperty );

        DotNetClass &lClass = DotNetRuntime::GetClassType( aClassName );

        MonoObject *lException = nullptr;
        MonoObject *lValue = mono_runtime_invoke( lPropertyGetter, mInstance, nullptr, &lException );

        if( lException == nullptr ) return New<DotNetInstance>( &lClass, lClass.Class(), lValue );

        return nullptr;
    }

    sScriptProperty &DotNetInstance::GetProperty( std::string const &aName ) { return mScriptClass->GetProperty( aName ); }

} // namespace SE::Core