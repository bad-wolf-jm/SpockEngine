#include "Vector.h"
#include "Core/EntityRegistry/Registry.h"
#include "TensorOps/ScalarTypes.h"
#include "Scripting/PrimitiveTypes.h"

namespace SE::Core
{
    using namespace entt::literals;

    void CreateReflectedTypes() {}

    void OpenVectorLibrary( sol::table &aScriptingState )
    {
        aScriptingState["Array"] = []( const sol::object &aTypeOrID, uint32_t aSize, sol::this_state aScriptState )
        {
            const auto lMaybeAny = InvokeMetaFunction( DeduceType( aTypeOrID ), "CreateVector0"_hs, aSize, aScriptState );
            return lMaybeAny ? lMaybeAny.cast<sol::reference>() : sol::lua_nil_t{};
        };

        aScriptingState["Random"] = []( const sol::object &aTypeOrID, size_t aSize, double aMin, double aMax, sol::this_state aScriptState )
        {
            const auto lMaybeAny = InvokeMetaFunction( DeduceType( aTypeOrID ), "RandomVector"_hs, aSize, aMin, aMax, aScriptState );
            return lMaybeAny ? lMaybeAny.cast<sol::reference>() : sol::lua_nil_t{};
        };
    }
}; // namespace SE::Core