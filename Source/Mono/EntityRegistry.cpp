#include "EntityRegistry.h"

// #include "TypeReflection.h"

// #include "mono/jit/jit.h"
// #include "mono/metadata/assembly.h"
// #include "mono/metadata/object.h"
// #include "mono/metadata/tabledefs.h"

#include <functional>
#include <string>

namespace LTSE::Core
{

    entt::meta_type GetMetaType( MonoType *aObject )
    {
        auto lHashValue = std::hash<uint64_t>()( (uint64_t)aObject );

        return entt::resolve( (uint32_t)( lHashValue & 0xFFFFFFFF ) );
    }

    ScriptClassInstance MarshallComponent( ScriptClass &lMonoType, sNodeTransformComponent const &aComponent )
    {
        return lMonoType.Instantiate();
    }

    // using namespace sol;
    // using namespace entt::literals;
    // namespace
    // {
    //     template <typename _Ty>
    //     std::string_view GetTypeName()
    //     {
    //         std::string_view lTypeName          = typeid( _Ty ).name();
    //         size_t           lLastColonPosition = lTypeName.find_last_of( ':' );

    //         return lTypeName.substr( lLastColonPosition + 1 );
    //     }

    //     template <typename _Ty>
    //     MonoType *RetrieveMonoTypeFromNamespace( std::string const &aNamespace )
    //     {
    //         auto        lStructName   = GetTypeName();
    //         std::string lMonoTypeName = fmt::format( "{}.{}", aNamespace, lStructName );

    //         return mono_reflection_type_from_name( lMonoTypeName.data(), ScriptEngine::GetCoreAssemblyImage() );
    //     }

    //     template <typename _Ty>
    //     auto IsValid( Entity &aEntity )
    //     {
    //         return aEntity.IsValid();
    //     }

    //     template <typename _Ty>
    //     auto Add( Entity &aEntity, const sol::table &aInstance, sol::this_state aScriptState )
    //     {
    //         auto &lNewComponent = aEntity.Add<_Ty>( aInstance.valid() ? aInstance.as<_Ty>() : _Ty{} );

    //         return sol::make_reference( aScriptState, std::ref( lNewComponent ) );
    //     }

    //     template <typename _Ty>
    //     auto AddOrReplace( Entity &aEntity, const sol::table &aInstance, sol::this_state aScriptState )
    //     {
    //         auto &lNewComponent = aEntity.AddOrReplace<_Ty>( aInstance.valid() ? aInstance.as<_Ty>() : _Ty{} );

    //         return sol::make_reference( aScriptState, std::ref( lNewComponent ) );
    //     }

    //     template <typename _Ty>
    //     auto Replace( Entity &aEntity, const sol::table &aInstance, sol::this_state aScriptState )
    //     {
    //         auto &lNewComponent = aEntity.Replace<_Ty>( aInstance.valid() ? aInstance.as<_Ty>() : _Ty{} );

    //         return sol::make_reference( aScriptState, std::ref( lNewComponent ) );
    //     }

    //     template <typename _Ty>
    //     auto Get( Entity &aEntity, sol::this_state aScriptState )
    //     {
    //         auto &lNewComponent = aEntity.Get<_Ty>();

    //         return sol::make_reference( aScriptState, std::ref( lNewComponent ) );
    //     }

    //     template <typename _Ty>
    //     auto Has( Entity &aEntity )
    //     {
    //         return aEntity.Has<_Ty>();
    //     }

    //     template <typename _Ty>
    //     auto Remove( Entity &aEntity )
    //     {
    //         aEntity.Remove<_Ty>();
    //     }
    // } // namespace

    // template <typename _Ty>
    // auto DeclareComponentType( sol::table &aScriptingState, std::string const &aLuaName )
    // {
    //     using namespace entt::literals;

    //     auto lMonoType = RetrieveMonoTypeFromNamespace<_Ty>( "SpockEngine" );

    //     auto lHashValue = std::hash<size_t>( (uint64_t)lMonoType );
    //     auto lNewType   = entt::meta<_Ty>().type( lHashValue & 0xFFFFFFFF );

    //     // if constexpr( std::is_class<_Ty>::value && std::is_empty<_Ty>::value )
    //     // {
    //     //     // lNewType.template func<&IsValid<_Ty>>( "IsValid"_hs );
    //     //     // lNewType.template func<&Tag<_Ty>>( "Tag"_hs );
    //     //     // lNewType.template func<&Untag<_Ty>>( "Untag"_hs );
    //     //     lNewType.template func<&Has<_Ty>>( "Has"_hs );
    //     // }
    //     if constexpr( std::is_class<_Ty>::value )
    //     {
    //         // lNewType.template func<&IsValid<_Ty>>( "IsValid"_hs );
    //         // lNewType.template func<&Add<_Ty>>( "Add"_hs );
    //         // lNewType.template func<&AddOrReplace<_Ty>>( "AddOrReplace"_hs );
    //         // lNewType.template func<&Replace<_Ty>>( "Replace"_hs );
    //         // lNewType.template func<&Get<_Ty>>( "Get"_hs );
    //         lNewType.template func<&Has<_Ty>>( "Has"_hs );
    //         // lNewType.template func<&Remove<_Ty>>( "Remove"_hs );
    //     }

    //     return lNewLuaType;
    // }

    // template <typename _ComponentType>
    // void Get( uint32_t aEntityID, EntityRegistry *aRegistry, MonoType *aTagType, _ComponentType *aOutput )
    // {
    //     //
    // }

    // template <typename _ComponentType>
    // void Add( uint32_t aEntityID, EntityRegistry *aRegistry, MonoType *aTagType, _ComponentType *aValue )
    // {
    //     //
    // }

    // template <typename _ComponentType>
    // void Replace( uint32_t aEntityID, EntityRegistry *aRegistry, MonoType *aTagType, _ComponentType *aValue )
    // {
    //     //
    // }

    // template <typename _ComponentType>
    // void AddOrReplace( uint32_t aEntityID, EntityRegistry *aRegistry, MonoType *aTagType, _ComponentType *aValue )
    // {
    //     //
    // }

    // void Remove( uint32_t aEntityID, EntityRegistry *aRegistry, MonoType *aTagType )
    // {
    //     //
    // }

    // // void RequireEntityType( sol::table &aScriptingState )
    // // {
    // //     auto lEntityType = aScriptingState.new_usertype<Entity>( "Entity", constructors<Entity()>() );

    // //     lEntityType["tag"] = []( Entity &aSelf, const sol::object &aTypeOrID ) -> sol::object
    // //     {
    // //         const auto lMaybeAny = InvokeMetaFunction( DeduceType( aTypeOrID ), "Tag"_hs, aSelf );
    // //         return sol::lua_nil_t{};
    // //     };

    // //     lEntityType["untag"] = []( Entity &aSelf, const sol::object &aTypeOrID )
    // //     {
    // //         const auto lMaybeAny = InvokeMetaFunction( DeduceType( aTypeOrID ), "Untag"_hs, aSelf );
    // //         return sol::lua_nil_t{};
    // //     };

    // //     lEntityType["has"] = []( Entity &aSelf, const sol::object &aTypeOrID )
    // //     {
    // //         const auto lMaybeAny = InvokeMetaFunction( DeduceType( aTypeOrID ), "Has"_hs, &aSelf );
    // //         return lMaybeAny ? lMaybeAny.cast<bool>() : false;
    // //     };

    // //     lEntityType["get"] = []( Entity &aSelf, const sol::object &aTypeOrID, sol::this_state s )
    // //     {
    // //         const auto lMaybeAny = InvokeMetaFunction( DeduceType( aTypeOrID ), "Get"_hs, aSelf, s );
    // //         return lMaybeAny ? lMaybeAny.cast<sol::reference>() : sol::lua_nil_t{};
    // //     };

    // //     lEntityType["add"] = []( Entity &aSelf, const sol::table &aComponent, sol::this_state s ) -> sol::object
    // //     {
    // //         if( !aComponent.valid() ) return sol::lua_nil_t{};

    // //         const auto lMaybeAny = InvokeMetaFunction( GetTypeID( aComponent ), "Add"_hs, aSelf, aComponent, s );
    // //         return lMaybeAny ? lMaybeAny.cast<sol::reference>() : sol::lua_nil_t{};
    // //     };

    // //     lEntityType["replace"] = []( Entity &aSelf, const sol::table &aComponent, sol::this_state s ) -> sol::object
    // //     {
    // //         if( !aComponent.valid() ) return sol::lua_nil_t{};

    // //         const auto lMaybeAny = InvokeMetaFunction( GetTypeID( aComponent ), "Replace"_hs, aSelf, aComponent, s );
    // //         return lMaybeAny ? lMaybeAny.cast<sol::reference>() : sol::lua_nil_t{};
    // //     };

    // //     lEntityType["add_or_replace"] = []( Entity &aSelf, const sol::table &aComponent, sol::this_state s ) -> sol::object
    // //     {
    // //         if( !aComponent.valid() ) return sol::lua_nil_t{};

    // //         const auto lMaybeAny = InvokeMetaFunction( GetTypeID( aComponent ), "AddOrReplace"_hs, aSelf, aComponent, s );
    // //         return lMaybeAny ? lMaybeAny.cast<sol::reference>() : sol::lua_nil_t{};
    // //     };

    // //     lEntityType["remove"] = []( Entity &aSelf, const sol::object &aTypeOrID )
    // //     {
    // //         const auto lMaybeAny = InvokeMetaFunction( DeduceType( aTypeOrID ), "Remove"_hs, aSelf );
    // //         return sol::lua_nil_t{};
    // //     };
    // // }

    // void RequireEntityRegistry( sol::table &aScriptingState )
    // {
    //     auto lRegistryType = aScriptingState.new_usertype<EntityRegistry>( "Registry", constructors<EntityRegistry()>() );

    //     lRegistryType["create_entity"] = overload( []( EntityRegistry &aSelf ) { return aSelf.CreateEntity(); },
    //         []( EntityRegistry &aSelf, std::string aName ) { return aSelf.CreateEntity( aName ); },
    //         []( EntityRegistry &aSelf, Entity &aParent, std::string aName ) { return aSelf.CreateEntity( aParent, aName ); } );

    //     lRegistryType["create_entity_with_relationship"] =
    //         overload( []( EntityRegistry &aSelf ) { return aSelf.CreateEntityWithRelationship(); },
    //             []( EntityRegistry &aSelf, std::string aName ) { return aSelf.CreateEntityWithRelationship( aName ); } );

    //     lRegistryType["destroy_entity"] = []( EntityRegistry &aSelf, Entity &aEntity ) { aSelf.DestroyEntity( aEntity ); };

    //     lRegistryType["set_parent"] = []( EntityRegistry &aSelf, Entity &aEntity, Entity &aParentEntity )
    //     { aSelf.SetParent( aEntity, aParentEntity ); };

    //     lRegistryType["clear"] = []( EntityRegistry &aSelf ) { aSelf.Clear(); };
    // }

    // void OpenEntityRegistry( sol::table &aScriptingState )
    // {
    //     RequireEntityType( aScriptingState );
    //     RequireEntityRegistry( aScriptingState );
    // }

} // namespace LTSE::Core