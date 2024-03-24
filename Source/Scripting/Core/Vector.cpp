#include "Vector.h"
#include "Core/Entity/Collection.h"
#include "TensorOps/ScalarTypes.h"
#include "Scripting/PrimitiveTypes.h"

namespace SE::Core
{
    using namespace entt::literals;

    void CreateReflectedTypes() {}

    void open_vector_library( sol::table &scriptingState )
    {
        scriptingState["Array"] = []( const sol::object &typeOrID, uint32_t size, sol::this_state scriptState )
        {
            const auto maybeAny = invoke_meta_function( deduce_type( typeOrID ), "CreateVector0"_hs, size, scriptState );
            return maybeAny ? maybeAny.cast<sol::reference>() : sol::lua_nil_t{};
        };

        scriptingState["Random"] = []( const sol::object &typeOrID, size_t size, double min, double max, sol::this_state scriptState )
        {
            const auto maybeAny = invoke_meta_function( deduce_type( typeOrID ), "RandomVector"_hs, size, min, max, scriptState );
            return maybeAny ? maybeAny.cast<sol::reference>() : sol::lua_nil_t{};
        };
    }
}; // namespace SE::Core