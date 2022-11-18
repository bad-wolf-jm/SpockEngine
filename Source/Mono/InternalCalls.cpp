#include "InternalCalls.h"
#include "EntityRegistry.h"
#include "TypeReflection.h"
#include <iostream>
#include <string>

#include "Core/Logging.h"
#include "Manager.h"

namespace SE::MonoInternalCalls
{
    bool Entity_IsValid( uint32_t aEntityID, EntityRegistry *aRegistry )
    {
        return aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ).IsValid();
    }

    bool Entity_Has( uint32_t aEntityID, EntityRegistry *aRegistry, MonoReflectionType *aComponentType )
    {
        MonoType *lMonoType = mono_reflection_type_get_type( aComponentType );

        const entt::meta_type lMetaType = Core::GetMetaType( lMonoType );
        const entt::meta_any  lMaybeAny =
            Core::InvokeMetaFunction( lMetaType, "Has"_hs, aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ) );

        return static_cast<bool>( lMaybeAny );
    }

    MonoObject *Entity_Get( uint32_t aEntityID, EntityRegistry *aRegistry, MonoReflectionType *aComponentType )
    {
        MonoType *lMonoType = mono_reflection_type_get_type( aComponentType );

        const entt::meta_type lMetaType = Core::GetMetaType( lMonoType );
        const entt::meta_any  lMaybeAny = Core::InvokeMetaFunction(
             lMetaType, "Get"_hs, aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ), ScriptClass( lMonoType ) );

        return lMaybeAny.cast<ScriptClassInstance>().GetInstance();
    }

    void Entity_Replace( uint32_t aEntityID, EntityRegistry *aRegistry, MonoReflectionType *aComponentType, MonoObject *aNewComponent )
    {
        MonoType  *lMonoType  = mono_reflection_type_get_type( aComponentType );
        MonoClass *lMonoClass = mono_class_from_mono_type( lMonoType );

        const entt::meta_type lMetaType = Core::GetMetaType( lMonoType );
        Core::InvokeMetaFunction( lMetaType, "Replace"_hs, aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ),
            ScriptClassInstance( lMonoClass, aNewComponent ) );
    }

    void Entity_Add( uint32_t aEntityID, EntityRegistry *aRegistry, MonoReflectionType *aComponentType, MonoObject *aNewComponent )
    {
        MonoType  *lMonoType  = mono_reflection_type_get_type( aComponentType );
        MonoClass *lMonoClass = mono_class_from_mono_type( lMonoType );

        const entt::meta_type lMetaType = Core::GetMetaType( lMonoType );
        Core::InvokeMetaFunction( lMetaType, "Add"_hs, aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ),
            ScriptClassInstance( lMonoClass, aNewComponent ) );
    }

    void Entity_Remove( uint32_t aEntityID, EntityRegistry *aRegistry, MonoReflectionType *aComponentType )
    {
        MonoType *lMonoType = mono_reflection_type_get_type( aComponentType );

        const entt::meta_type lMetaType = Core::GetMetaType( lMonoType );
        const entt::meta_any  lMaybeAny =
            Core::InvokeMetaFunction( lMetaType, "Remove"_hs, aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ) );
    }


} // namespace SE::MonoInternalCalls