#include "EntityRegistry.h"

#include "Scripting/PrimitiveTypes.h"

namespace SE::Core
{
    using namespace sol;
    using namespace entt::literals;

    [[nodiscard]] entt::id_type get_type_id( const sol::table &object )
    {
        const auto lFunction = object["type_id"].get<sol::function>();
        assert( lFunction.valid() && "type_id not exposed to lua!" );
        return lFunction.valid() ? lFunction().get<entt::id_type>() : -1;
    }

    void RequireEntityType( sol::table &scriptingState )
    {
        auto entityType = scriptingState.new_usertype<entity_t>( "Entity", constructors<entity_t()>() );

        entityType["tag"] = []( entity_t &self, const sol::object &typeOrID ) -> sol::object
        {
            const auto maybeAny = invoke_meta_function( deduce_type( typeOrID ), "Tag"_hs, self );
            return sol::lua_nil_t{};
        };

        entityType["untag"] = []( entity_t &self, const sol::object &typeOrID )
        {
            const auto maybeAny = invoke_meta_function( deduce_type( typeOrID ), "Untag"_hs, self );
            return sol::lua_nil_t{};
        };

        entityType["has"] = []( entity_t &self, const sol::object &typeOrID )
        {
            const auto maybeAny = invoke_meta_function( deduce_type( typeOrID ), "Has"_hs, &self );
            return maybeAny ? maybeAny.cast<bool>() : false;
        };

        entityType["get"] = []( entity_t &self, const sol::object &typeOrID )
        {
            const auto maybeAny = invoke_meta_function( deduce_type( typeOrID ), "Get"_hs, &self );
            return maybeAny ? maybeAny.cast<sol::reference>() : sol::lua_nil_t{};
        };

        entityType["try_get"] = []( entity_t &self, const sol::object &typeOrID )
        {
            const auto maybeAny = invoke_meta_function( deduce_type( typeOrID ), "TryGet"_hs, &self );
            return maybeAny ? maybeAny.cast<sol::reference>() : sol::lua_nil_t{};
        };

        entityType["add"] = []( entity_t &self, const sol::table &component, sol::this_state s ) -> sol::object
        {
            if( !component.valid() )
                return sol::lua_nil_t{};

            const auto maybeAny = invoke_meta_function( get_type_id( component ), "Add"_hs, self, component, s );
            return maybeAny ? maybeAny.cast<sol::reference>() : sol::lua_nil_t{};
        };

        entityType["replace"] = []( entity_t &self, const sol::table &component, sol::this_state s ) -> sol::object
        {
            if( !component.valid() )
                return sol::lua_nil_t{};

            const auto maybeAny = invoke_meta_function( get_type_id( component ), "Replace"_hs, self, component, s );
            return maybeAny ? maybeAny.cast<sol::reference>() : sol::lua_nil_t{};
        };

        entityType["add_or_replace"] = []( entity_t &self, const sol::table &component, sol::this_state s ) -> sol::object
        {
            if( !component.valid() )
                return sol::lua_nil_t{};

            const auto maybeAny = invoke_meta_function( get_type_id( component ), "AddOrReplace"_hs, self, component, s );
            return maybeAny ? maybeAny.cast<sol::reference>() : sol::lua_nil_t{};
        };

        entityType["try_add"] = []( entity_t &self, const sol::table &component, sol::this_state s ) -> sol::object
        {
            if( !component.valid() )
                return sol::lua_nil_t{};

            const auto maybeAny = invoke_meta_function( get_type_id( component ), "TryAdd"_hs, self, component, s );
            return maybeAny ? maybeAny.cast<sol::reference>() : sol::lua_nil_t{};
        };

        entityType["remove"] = []( entity_t &self, const sol::object &typeOrID )
        {
            const auto maybeAny = invoke_meta_function( deduce_type( typeOrID ), "Remove"_hs, self );
            return sol::lua_nil_t{};
        };

        entityType["try_remove"] = []( entity_t &self, const sol::object &typeOrID )
        {
            const auto maybeAny = invoke_meta_function( deduce_type( typeOrID ), "TryRemove"_hs, self );
            return sol::lua_nil_t{};
        };
    }

    void RequireEntityRegistry( sol::table &scriptingState )
    {
        auto registryType = scriptingState.new_usertype<entity_registry_t>( "Registry", constructors<entity_registry_t()>() );

        registryType["create_entity"] =
            overload( []( entity_registry_t &self ) { return self.CreateEntity(); }, []( entity_registry_t &self, std::string aName ) { return self.CreateEntity( aName ); },
                      []( entity_registry_t &self, entity_t &aParent, std::string aName ) { return self.CreateEntity( aParent, aName ); } );

        registryType["create_entity_with_relationship"] = overload( []( entity_registry_t &self ) { return self.CreateEntityWithRelationship(); },
                                                                     []( entity_registry_t &self, std::string aName ) { return self.CreateEntityWithRelationship( aName ); } );

        registryType["destroy_entity"] = []( entity_registry_t &self, entity_t &aEntity ) { self.DestroyEntity( aEntity ); };

        registryType["set_parent"] = []( entity_registry_t &self, entity_t &aEntity, entity_t &aParentEntity ) { self.SetParent( aEntity, aParentEntity ); };

        registryType["clear"] = []( entity_registry_t &self ) { self.Clear(); };
    }

    void open_entity_registry_library( sol::state &scriptingState )
    {
        auto entityRegistryModule = scriptingState["EntityCollection"].get_or_create<sol::table>();

        RequireEntityType( entityRegistryModule );
        RequireEntityRegistry( entityRegistryModule );
    }

} // namespace SE::Core