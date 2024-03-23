#include "Vector.h"
#include "Core/Entity/Collection.h"
#include "TensorOps/ScalarTypes.h"
#include "Scripting/PrimitiveTypes.h"

namespace SE::Core
{
    using namespace entt::literals;

    void CreateReflectedTypes() {}

    void open_vector_library( sol::table &aScriptingState )
    {
        aScriptingState["Array"] = []( const sol::object &aTypeOrID, uint32_t aSize, sol::this_state aScriptState )
        {
            const auto lMaybeAny = invoke_meta_function( deduce_type( aTypeOrID ), "CreateVector0"_hs, aSize, aScriptState );
            return lMaybeAny ? lMaybeAny.cast<sol::reference>() : sol::lua_nil_t{};
        };

        aScriptingState["Random"] = []( const sol::object &aTypeOrID, size_t aSize, double aMin, double aMax, sol::this_state aScriptState )
        {
            const auto lMaybeAny = invoke_meta_function( deduce_type( aTypeOrID ), "RandomVector"_hs, aSize, aMin, aMax, aScriptState );
            return lMaybeAny ? lMaybeAny.cast<sol::reference>() : sol::lua_nil_t{};
        };
    }
}; // namespace SE::Core