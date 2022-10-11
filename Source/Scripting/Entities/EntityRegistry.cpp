#include "EntityRegistry.h"

#include "Scripting/PrimitiveTypes.h"

namespace LTSE::Core
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
        auto lEntityType = aScriptingState.new_usertype<Entity>( "Entity", constructors<Entity()>() );

        lEntityType["tag"] = []( Entity &aSelf, const sol::object &aTypeOrID ) -> sol::object
        {
            const auto lMaybeAny = InvokeMetaFunction( DeduceType( aTypeOrID ), "Tag"_hs, aSelf );
            return sol::lua_nil_t{};
        };

        lEntityType["untag"] = []( Entity &aSelf, const sol::object &aTypeOrID )
        {
            const auto lMaybeAny = InvokeMetaFunction( DeduceType( aTypeOrID ), "Untag"_hs, aSelf );
            return sol::lua_nil_t{};
        };

        lEntityType["has"] = []( Entity &aSelf, const sol::object &aTypeOrID )
        {
            const auto lMaybeAny = InvokeMetaFunction( DeduceType( aTypeOrID ), "Has"_hs, &aSelf );
            return lMaybeAny ? lMaybeAny.cast<bool>() : false;
        };

        lEntityType["get"] = []( Entity &aSelf, const sol::object &aTypeOrID, sol::this_state s )
        {
            const auto lMaybeAny = InvokeMetaFunction( DeduceType( aTypeOrID ), "Get"_hs, aSelf, s );
            return lMaybeAny ? lMaybeAny.cast<sol::reference>() : sol::lua_nil_t{};
        };

        lEntityType["try_get"] = []( Entity &aSelf, const sol::object &aTypeOrID, sol::this_state s  )
        {
            const auto lMaybeAny = InvokeMetaFunction( DeduceType( aTypeOrID ), "TryGet"_hs, aSelf, s );
            return lMaybeAny ? lMaybeAny.cast<sol::reference>() : sol::lua_nil_t{};
        };

        lEntityType["add"] = []( Entity &aSelf, const sol::table &aComponent, sol::this_state s ) -> sol::object
        {
            if( !aComponent.valid() )
                return sol::lua_nil_t{};

            const auto lMaybeAny = InvokeMetaFunction( GetTypeID( aComponent ), "Add"_hs, aSelf, aComponent, s );
            return lMaybeAny ? lMaybeAny.cast<sol::reference>() : sol::lua_nil_t{};
        };

        lEntityType["replace"] = []( Entity &aSelf, const sol::table &aComponent, sol::this_state s ) -> sol::object
        {
            if( !aComponent.valid() )
                return sol::lua_nil_t{};

            const auto lMaybeAny = InvokeMetaFunction( GetTypeID( aComponent ), "Replace"_hs, aSelf, aComponent, s );
            return lMaybeAny ? lMaybeAny.cast<sol::reference>() : sol::lua_nil_t{};
        };

        lEntityType["add_or_replace"] = []( Entity &aSelf, const sol::table &aComponent, sol::this_state s ) -> sol::object
        {
            if( !aComponent.valid() )
                return sol::lua_nil_t{};

            const auto lMaybeAny = InvokeMetaFunction( GetTypeID( aComponent ), "AddOrReplace"_hs, aSelf, aComponent, s );
            return lMaybeAny ? lMaybeAny.cast<sol::reference>() : sol::lua_nil_t{};
        };

        lEntityType["try_add"] = []( Entity &aSelf, const sol::table &aComponent, sol::this_state s ) -> sol::object
        {
            if( !aComponent.valid() )
                return sol::lua_nil_t{};

            const auto lMaybeAny = InvokeMetaFunction( GetTypeID( aComponent ), "TryAdd"_hs, aSelf, aComponent, s );
            return lMaybeAny ? lMaybeAny.cast<sol::reference>() : sol::lua_nil_t{};
        };

        lEntityType["remove"] = []( Entity &aSelf, const sol::object &aTypeOrID )
        {
            const auto lMaybeAny = InvokeMetaFunction( DeduceType( aTypeOrID ), "Remove"_hs, aSelf );
            return sol::lua_nil_t{};
        };

        lEntityType["try_remove"] = []( Entity &aSelf, const sol::object &aTypeOrID )
        {
            const auto lMaybeAny = InvokeMetaFunction( DeduceType( aTypeOrID ), "TryRemove"_hs, aSelf );
            return sol::lua_nil_t{};
        };

        lEntityType["adjoin"] = []( Entity &aSelf, const sol::object &aTypeOrID, Entity &aOther )
        {
            const auto lMaybeAny = InvokeMetaFunction( DeduceType( aTypeOrID ), "Adjoin"_hs, aSelf, aOther );

            return sol::lua_nil_t{};
        };

        lEntityType["get_joined"] = []( Entity &aSelf, const sol::object &aTypeOrID, sol::this_state s  )
        {
            const auto lMaybeAny = InvokeMetaFunction( DeduceType( aTypeOrID ), "GetJoined"_hs, aSelf, s );

            return lMaybeAny ? lMaybeAny.cast<sol::reference>() : sol::lua_nil_t{};
        };
    }

    void RequireEntityRegistry( sol::table &aScriptingState )
    {
        auto lRegistryType = aScriptingState.new_usertype<EntityRegistry>( "Registry", constructors<EntityRegistry()>() );

        lRegistryType["create_entity"] =
            overload( []( EntityRegistry &aSelf ) { return aSelf.CreateEntity(); }, []( EntityRegistry &aSelf, std::string aName ) { return aSelf.CreateEntity( aName ); },
                      []( EntityRegistry &aSelf, Entity &aParent, std::string aName ) { return aSelf.CreateEntity( aParent, aName ); } );

        lRegistryType["create_entity_with_relationship"] = overload( []( EntityRegistry &aSelf ) { return aSelf.CreateEntityWithRelationship(); },
                                                                     []( EntityRegistry &aSelf, std::string aName ) { return aSelf.CreateEntityWithRelationship( aName ); } );

        lRegistryType["destroy_entity"] = []( EntityRegistry &aSelf, Entity &aEntity ) { aSelf.DestroyEntity( aEntity ); };

        lRegistryType["set_parent"] = []( EntityRegistry &aSelf, Entity &aEntity, Entity &aParentEntity ) { aSelf.SetParent( aEntity, aParentEntity ); };

        lRegistryType["clear"] = []( EntityRegistry &aSelf ) { aSelf.Clear(); };
    }

    void OpenEntityRegistry( sol::table &aScriptingState )
    {
        RequireEntityType( aScriptingState );
        RequireEntityRegistry( aScriptingState );

    }

} // namespace LTSE::Core