#include "MonoScriptClass.h"

#include "Core/Logging.h"
#include "Core/Memory.h"

#include "Engine/Engine.h"

#include "Scene/Scene.h"

#include "mono/jit/jit.h"
#include "mono/metadata/assembly.h"
#include "mono/metadata/object.h"
#include "mono/metadata/tabledefs.h"

#include <unordered_map>

#ifdef WIN32_LEAN_AND_MEAN
#    undef WIN32_LEAN_AND_MEAN
#endif
#include "Core/FileWatch.hpp"

#include "EntityRegistry.h"
#include "InternalCalls.h"

#include "MonoRuntime.h"
#include "MonoScriptUtils.h"

namespace SE::Core
{

    namespace
    {
        std::map<std::string, sScriptField> GetClassFields( MonoClass *aMonoClass )
        {
            std::map<std::string, sScriptField> lFields{};

            if( !aMonoClass ) return lFields;

            int lFieldCount = mono_class_num_fields( aMonoClass );

            void *lIterator = nullptr;
            while( MonoClassField *lField = mono_class_get_fields( aMonoClass, &lIterator ) )
            {
                const char *lFieldName = mono_field_get_name( lField );
                uint32_t    lFlags     = mono_field_get_flags( lField );

                if( lFlags & FIELD_ATTRIBUTE_PUBLIC )
                {
                    MonoType        *lMonoFieldType = mono_field_get_type( lField );
                    eScriptFieldType lFieldType     = Mono::Utils::MonoTypeToScriptFieldType( lMonoFieldType );

                    lFields[lFieldName] = { lFieldType, lFieldName, lField };
                }
            }

            return lFields;
        }
    } // namespace

    MonoScriptClass::MonoScriptClass( const std::string &aClassNamespace, const std::string &aClassName, MonoImage *aImage,
                                      fs::path const &aDllPPath )
        : mClassNamespace( aClassNamespace )
        , mClassName( aClassName )
        , mDllPath{ aDllPPath }
    {
        mMonoClass     = mono_class_from_name( aImage, aClassNamespace.c_str(), aClassName.c_str() );
        mFields        = GetClassFields( mMonoClass );
        mClassFullName = fmt::format( "{}.{}", mClassNamespace, mClassName );
    }

    MonoScriptClass::MonoScriptClass( MonoType *aMonoClass )
        : mMonoClass{ mono_class_from_mono_type( aMonoClass ) }
    {
        mFields = GetClassFields( mMonoClass );
    }

    Ref<MonoScriptInstance> MonoScriptClass::DoInstantiate()
    {
        MonoObject *lInstance = MonoRuntime::InstantiateClass( mMonoClass, mIsCore );

        return New<MonoScriptInstance>( mMonoClass, lInstance );
    }

    MonoMethod *MonoScriptClass::GetMethod( const std::string &aName, int aParameterCount )
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

    MonoObject *MonoScriptClass::InvokeMethod( MonoMethod *aMethod, void **aParameters )
    {
        return mono_runtime_invoke( aMethod, nullptr, aParameters, nullptr );
    }

    MonoObject *MonoScriptClass::InvokeMethod( const std::string &aName, int aParameterCount, void **aParameters )
    {
        auto lMethod = GetMethod( aName, aParameterCount );

        return InvokeMethod( lMethod, aParameters );
    }
} // namespace SE::Core