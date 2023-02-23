#include "MonoScriptInstance.h"

#include "mono/jit/jit.h"
#include "mono/metadata/assembly.h"
#include "mono/metadata/object.h"
#include "mono/metadata/tabledefs.h"

namespace SE::Core
{
    MonoScriptInstance::MonoScriptInstance( MonoClass *aMonoClass, MonoObject *aInstance )
        : mMonoClass{ aMonoClass }
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
        return mono_runtime_invoke( aMethod, mInstance, aParameters, nullptr );
    }

    MonoObject *MonoScriptInstance::InvokeMethod( const std::string &aName, int aParameterCount, void **aParameters )
    {
        auto lMethod = GetMethod( aName, aParameterCount );
        return InvokeMethod( lMethod, aParameters );
    }
} // namespace SE::Core