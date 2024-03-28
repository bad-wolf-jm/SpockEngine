#include "TypeReflection.h"

namespace SE::Core
{
    [[nodiscard]] entt::id_type get_type_id( const sol::table &object )
    {
        const auto lFunction = object["type_id"].get<sol::function>();
        assert( lFunction.valid() && "type_id not exposed to lua!" );
        return lFunction.valid() ? lFunction().get<entt::id_type>() : -1;
    }
}