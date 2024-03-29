#include "TypeReflection.h"

namespace SE::Core
{
    [[nodiscard]] entt::id_type get_type_id( const sol::table &object )
    {
        const auto function = object["type_id"].get<sol::function>();
        assert( function.valid() && "type_id not exposed to lua!" );
        return function.valid() ? function().get<entt::id_type>() : -1;
    }
}