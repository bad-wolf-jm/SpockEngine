#include "Manager.h"

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
#include <FileWatch.hpp>

#include "EntityRegistry.h"
#include "InternalCalls.h"

#include "Manager.h"
#include "MonoScriptUtils.h"

namespace SE::Core
{
    MonoScriptClass::MonoScriptClass( const std::string &aClassNamespace, const std::string &aClassName, bool aIsCore )
        : mClassNamespace( aClassNamespace )
        , mClassName( aClassName )
    {
        mMonoClass = mono_class_from_name( aIsCore ? MonoScriptEngine::GetCoreAssemblyImage() : MonoScriptEngine::GetAppAssemblyImage(), aClassNamespace.c_str(),
                                           aClassName.c_str() );

        int   lFieldCount = mono_class_num_fields( mMonoClass );
        void *lIterator   = nullptr;
        while( MonoClassField *lField = mono_class_get_fields( mMonoClass, &lIterator ) )
        {
            const char *lFieldName = mono_field_get_name( lField );
            uint32_t    lFlags     = mono_field_get_flags( lField );

            if( lFlags & FIELD_ATTRIBUTE_PUBLIC )
            {
                MonoType        *lMonoFieldType = mono_field_get_type( lField );
                eScriptFieldType lFieldType     = Mono::Utils::MonoTypeToScriptFieldType( lMonoFieldType );

                mFields[lFieldName] = { lFieldType, lFieldName, lField };
            }
        }
    }

    MonoScriptClass::MonoScriptClass( MonoType *aMonoClass )
        : mMonoClass{ mono_class_from_mono_type( aMonoClass ) }
    {
        int   lFieldCount = mono_class_num_fields( mMonoClass );
        void *lIterator   = nullptr;
        while( MonoClassField *lField = mono_class_get_fields( mMonoClass, &lIterator ) )
        {
            const char *lFieldName = mono_field_get_name( lField );
            uint32_t    lFlags     = mono_field_get_flags( lField );

            if( lFlags & FIELD_ATTRIBUTE_PUBLIC )
            {
                MonoType        *lMonoFieldType = mono_field_get_type( lField );
                eScriptFieldType lFieldType     = Mono::Utils::MonoTypeToScriptFieldType( lMonoFieldType );

                mFields[lFieldName] = { lFieldType, lFieldName, lField };
            }
        }
    }

    MonoScriptInstance MonoScriptClass::Instantiate()
    {
        MonoObject *lInstance = MonoScriptEngine::InstantiateClass( mMonoClass, mIsCore );

        return MonoScriptInstance( mMonoClass, lInstance );
    }
} // namespace SE::Core