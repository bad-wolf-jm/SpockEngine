#include "EntityRegistry.h"

#include "Scripting/PrimitiveTypes.h"

namespace SE::Core
{
    using namespace sol;
    using namespace entt::literals;

    [[nodiscard]] entt::id_type GetTypeID( const sol::table &aObject )
    {
        const auto lFunction = aObject["type_id"].get<sol::function>();
        assert( lFunction.valid() && "type_id not exposed to lua!" );
        return lFunction.valid() ? lFunction().get<entt::id_type>() : -1;
    }

    void RequireEntityType( sol::table &aScriptingState )
    {
        auto lEntityType = aScriptingState.new_usertype<entity_t>( "Entity", constructors<entity_t()>() );

        lEntityType["tag"] = []( entity_t &aSelf, const sol::object &aTypeOrID ) -> sol::object
        {
            const auto lMaybeAny = InvokeMetaFunction( DeduceType( aTypeOrID ), "Tag"_hs, aSelf );
            return sol::lua_nil_t{};
        };

        lEntityType["untag"] = []( entity_t &aSelf, const sol::object &aTypeOrID )
        {
            const auto lMaybeAny = InvokeMetaFunction( DeduceType( aTypeOrID ), "Untag"_hs, aSelf );
            return sol::lua_nil_t{};
        };

        lEntityType["has"] = []( entity_t &aSelf, const sol::object &aTypeOrID )
        {
            const auto lMaybeAny = InvokeMetaFunction( DeduceType( aTypeOrID ), "Has"_hs, &aSelf );
            return lMaybeAny ? lMaybeAny.cast<bool>() : false;
        };

        lEntityType["get"] = []( entity_t &aSelf, const sol::object &aTypeOrID )
        {
            const auto lMaybeAny = InvokeMetaFunction( DeduceType( aTypeOrID ), "Get"_hs, &aSelf );
            return lMaybeAny ? lMaybeAny.cast<sol::reference>() : sol::lua_nil_t{};
        };

        lEntityType["try_get"] = []( entity_t &aSelf, const sol::object &aTypeOrID )
        {
            const auto lMaybeAny = InvokeMetaFunction( DeduceType( aTypeOrID ), "TryGet"_hs, &aSelf );
            return lMaybeAny ? lMaybeAny.cast<sol::reference>() : sol::lua_nil_t{};
        };

        lEntityType["add"] = []( entity_t &aSelf, const sol::table &aComponent, sol::this_state s ) -> sol::object
        {
            if( !aComponent.valid() )
                return sol::lua_nil_t{};

            const auto lMaybeAny = InvokeMetaFunction( GetTypeID( aComponent ), "Add"_hs, aSelf, aComponent, s );
            return lMaybeAny ? lMaybeAny.cast<sol::reference>() : sol::lua_nil_t{};
        };

        lEntityType["replace"] = []( entity_t &aSelf, const sol::table &aComponent, sol::this_state s ) -> sol::object
        {
            if( !aComponent.valid() )
                return sol::lua_nil_t{};

            const auto lMaybeAny = InvokeMetaFunction( GetTypeID( aComponent ), "Replace"_hs, aSelf, aComponent, s );
            return lMaybeAny ? lMaybeAny.cast<sol::reference>() : sol::lua_nil_t{};
        };

        lEntityType["add_or_replace"] = []( entity_t &aSelf, const sol::table &aComponent, sol::this_state s ) -> sol::object
        {
            if( !aComponent.valid() )
                return sol::lua_nil_t{};

            const auto lMaybeAny = InvokeMetaFunction( GetTypeID( aComponent ), "AddOrReplace"_hs, aSelf, aComponent, s );
            return lMaybeAny ? lMaybeAny.cast<sol::reference>() : sol::lua_nil_t{};
        };

        lEntityType["try_add"] = []( entity_t &aSelf, const sol::table &aComponent, sol::this_state s ) -> sol::object
        {
            if( !aComponent.valid() )
                return sol::lua_nil_t{};

            const auto lMaybeAny = InvokeMetaFunction( GetTypeID( aComponent ), "TryAdd"_hs, aSelf, aComponent, s );
            return lMaybeAny ? lMaybeAny.cast<sol::reference>() : sol::lua_nil_t{};
        };

        lEntityType["remove"] = []( entity_t &aSelf, const sol::object &aTypeOrID )
        {
            const auto lMaybeAny = InvokeMetaFunction( DeduceType( aTypeOrID ), "Remove"_hs, aSelf );
            return sol::lua_nil_t{};
        };

        lEntityType["try_remove"] = []( entity_t &aSelf, const sol::object &aTypeOrID )
        {
            const auto lMaybeAny = InvokeMetaFunction( DeduceType( aTypeOrID ), "TryRemove"_hs, aSelf );
            return sol::lua_nil_t{};
        };
    }

    void RequireEntityRegistry( sol::table &aScriptingState )
    {
        auto lRegistryType = aScriptingState.new_usertype<entity_registry_t>( "Registry", constructors<entity_registry_t()>() );

        lRegistryType["create_entity"] =
            overload( []( entity_registry_t &aSelf ) { return aSelf.CreateEntity(); }, []( entity_registry_t &aSelf, std::string aName ) { return aSelf.CreateEntity( aName ); },
                      []( entity_registry_t &aSelf, entity_t &aParent, std::string aName ) { return aSelf.CreateEntity( aParent, aName ); } );

        lRegistryType["create_entity_with_relationship"] = overload( []( entity_registry_t &aSelf ) { return aSelf.CreateEntityWithRelationship(); },
                                                                     []( entity_registry_t &aSelf, std::string aName ) { return aSelf.CreateEntityWithRelationship( aName ); } );

        lRegistryType["destroy_entity"] = []( entity_registry_t &aSelf, entity_t &aEntity ) { aSelf.DestroyEntity( aEntity ); };

        lRegistryType["set_parent"] = []( entity_registry_t &aSelf, entity_t &aEntity, entity_t &aParentEntity ) { aSelf.SetParent( aEntity, aParentEntity ); };

        lRegistryType["clear"] = []( entity_registry_t &aSelf ) { aSelf.Clear(); };
    }

    void OpenEntityRegistry( sol::state &aScriptingState )
    {
        auto lEntityRegistryModule = aScriptingState["EntityCollection"].get_or_create<sol::table>();

        RequireEntityType( lEntityRegistryModule );
        RequireEntityRegistry( lEntityRegistryModule );
    }

} // namespace SE::Core