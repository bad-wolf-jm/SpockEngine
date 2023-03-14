#include "Class.h"

#include "Core/Logging.h"
#include "Core/Memory.h"

#include "Engine/Engine.h"

// #include "Scene/Scene.h"

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

#include "Runtime.h"
#include "Utils.h"

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

        std::map<std::string, sScriptProperty> GetClassProperties( MonoClass *aMonoClass )
        {
            std::map<std::string, sScriptProperty> lProperties{};

            if( !aMonoClass ) return lProperties;

            MonoClass *lClass = aMonoClass;
            while( lClass != NULL )
            {
                int lFieldCount = mono_class_num_properties( lClass );

                void *lIterator = nullptr;
                while( MonoProperty *lProperty = mono_class_get_properties( lClass, &lIterator ) )
                {
                    const char *lPropertyName = mono_property_get_name( lProperty );
                    uint32_t    lFlags        = mono_property_get_flags( lProperty );

                    if( lProperties.find( lPropertyName ) == lProperties.end() )
                        lProperties[lPropertyName] = { lPropertyName, lProperty };
                }

                lClass = mono_class_get_parent( lClass );
            }

            return lProperties;
        }
    } // namespace

    DotNetClass::DotNetClass( const std::string &aClassNamespace, const std::string &aClassName, MonoImage *aImage,
                              fs::path const &aDllPPath, bool aIsNested )
        : mClassNamespace( aClassNamespace )
        , mClassName( aClassName )
        , mDllPath{ aDllPPath }
        , mIsNested{ aIsNested }
    {
        mMonoClass     = mono_class_from_name( aImage, aClassNamespace.c_str(), aClassName.c_str() );
        mFields        = GetClassFields( mMonoClass );
        mProperties    = GetClassProperties( mMonoClass );
        mClassFullName = fmt::format( "{}.{}", mClassNamespace, mClassName );
    }

    DotNetClass::DotNetClass( MonoClass *aClass, const std::string &aClassNamespace, const std::string &aClassName, MonoImage *aImage,
                              fs::path const &aDllPPath, bool aIsNested )
        : mClassNamespace( aClassNamespace )
        , mClassName( aClassName )
        , mDllPath{ aDllPPath }
        , mIsNested{ aIsNested }
    {
        mMonoClass     = aClass;
        mFields        = GetClassFields( mMonoClass );
        mProperties    = GetClassProperties( mMonoClass );
        mClassFullName = fmt::format( "{}.{}", mClassNamespace, mClassName );
    }

    DotNetClass::DotNetClass( MonoType *aMonoClass )
        : mMonoClass{ mono_class_from_mono_type( aMonoClass ) }
    {
        mFields     = GetClassFields( mMonoClass );
        mProperties = GetClassProperties( mMonoClass );
    }

    Ref<DotNetInstance> DotNetClass::DoInstantiate()
    {
        MonoObject *lInstance = DotNetRuntime::InstantiateClass( mMonoClass, mIsCore );

        return New<DotNetInstance>( this, mMonoClass, lInstance );
    }

    MonoMethod *DotNetClass::GetMethod( const std::string &aName, int aParameterCount )
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

    MonoObject *DotNetClass::InvokeMethod( MonoMethod *aMethod, void **aParameters )
    {
        MonoObject *lException = nullptr;
        MonoObject *lValue     = mono_runtime_invoke( aMethod, nullptr, aParameters, &lException );

        if( lException == nullptr ) return lValue;

        return nullptr;
    }

    MonoObject *DotNetClass::InvokeMethod( const std::string &aName, int aParameterCount, void **aParameters )
    {
        auto lMethod = GetMethod( aName, aParameterCount );

        return InvokeMethod( lMethod, aParameters );
    }
} // namespace SE::Core