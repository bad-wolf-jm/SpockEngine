#include "InternalCalls.h"

#include <iostream>
#include <string>


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

} // namespace LTSE::MonoInternalCalls