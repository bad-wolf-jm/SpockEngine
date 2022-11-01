#include "InternalCalls.h"
#include "EntityRegistry.h"
#include "TypeReflection.h"
#include <iostream>
#include <string>

#include "Core/Logging.h"
#include "Manager.h"

namespace LTSE::MonoInternalCalls
{
    void NativeLog( MonoString *string, int parameter )
    {
        char       *cStr = mono_string_to_utf8( string );
        std::string str( cStr );
        mono_free( cStr );
        std::cout << str << ", " << parameter << std::endl;
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

} // namespace LTSE::MonoInternalCalls