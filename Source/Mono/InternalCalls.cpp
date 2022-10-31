#include "InternalCalls.h"
#include "EntityRegistry.h"
#include "TypeReflection.h"
#include <iostream>
#include <string>

#include "Core/Logging.h"


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

        LTSE::Logging::Info(" TEST_HAS --->> {} (refl: {})", (uint64_t)lMonoType, (uint64_t)aComponentType);

        const auto lMetaType = Core::GetMetaType( lMonoType );
        const auto lMaybeAny =
            Core::InvokeMetaFunction( lMetaType, "Has"_hs, aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ) );

        return static_cast<bool>( lMaybeAny );
    }
    // {klass=0x0000027fd9818a48 {element_class=0x0000027fd9818a48 {element_class=0x0000027fd9818a48 {element_class=...} ...} ...} ...}
} // namespace LTSE::MonoInternalCalls