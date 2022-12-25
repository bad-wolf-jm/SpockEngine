#include "InternalCalls.h"
#include "EntityRegistry.h"
#include "TypeReflection.h"
#include <iostream>
#include <string>

#include "Core/Logging.h"
#include "MonoScriptEngine.h"

namespace SE::MonoInternalCalls
{
    uint32_t Entity_Create( EntityRegistry *aRegistry, MonoString *aName, uint32_t aEntityID )
    {
        auto lName      = std::string( mono_string_to_utf8( aName ) );
        auto lNewEntity = aRegistry->CreateEntity( aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ), lName );

        return static_cast<uint32_t>( lNewEntity );
    }

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

        return lMaybeAny.cast<bool>();
    }

    MonoObject *Entity_Get( uint32_t aEntityID, EntityRegistry *aRegistry, MonoReflectionType *aComponentType )
    {
        MonoType *lMonoType = mono_reflection_type_get_type( aComponentType );

        const entt::meta_type lMetaType = Core::GetMetaType( lMonoType );
        const entt::meta_any  lMaybeAny = Core::InvokeMetaFunction(
             lMetaType, "Get"_hs, aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ), MonoScriptClass( lMonoType ) );

        return lMaybeAny.cast<MonoScriptInstance>().GetInstance();
    }

    void Entity_Replace( uint32_t aEntityID, EntityRegistry *aRegistry, MonoReflectionType *aComponentType, MonoObject *aNewComponent )
    {
        MonoType  *lMonoType  = mono_reflection_type_get_type( aComponentType );
        MonoClass *lMonoClass = mono_class_from_mono_type( lMonoType );

        const entt::meta_type lMetaType = Core::GetMetaType( lMonoType );
        Core::InvokeMetaFunction( lMetaType, "Replace"_hs, aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ),
                                  MonoScriptInstance( lMonoClass, aNewComponent ) );
    }

    void Entity_Add( uint32_t aEntityID, EntityRegistry *aRegistry, MonoReflectionType *aComponentType, MonoObject *aNewComponent )
    {
        MonoType  *lMonoType  = mono_reflection_type_get_type( aComponentType );
        MonoClass *lMonoClass = mono_class_from_mono_type( lMonoType );

        const entt::meta_type lMetaType = Core::GetMetaType( lMonoType );
        Core::InvokeMetaFunction( lMetaType, "Add"_hs, aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ),
                                  MonoScriptInstance( lMonoClass, aNewComponent ) );
    }

    void Entity_Remove( uint32_t aEntityID, EntityRegistry *aRegistry, MonoReflectionType *aComponentType )
    {
        MonoType *lMonoType = mono_reflection_type_get_type( aComponentType );

        const entt::meta_type lMetaType = Core::GetMetaType( lMonoType );
        const entt::meta_any  lMaybeAny =
            Core::InvokeMetaFunction( lMetaType, "Remove"_hs, aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ) );
    }

} // namespace SE::MonoInternalCalls