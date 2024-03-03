/// @file   Entity.h
///
/// @brief  Wrapper around EnTT's registry class
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert. All rights reserved.

#pragma once

#include "entt/entt.hpp"
#include <functional>
#include <optional>

#include "Components.h"
#include "Entity.h"
// #include "ScriptableEntity.h"

#include <unordered_map>

namespace SE::Core
{

    /// \class EntityRegistry
    ///
    /// @brief Abstraction for `entt`'s registry type.
    ///
    class entity_registry_t
    {
      private:
        using entity_type_t       = Internal::entity_t<entity_registry_t *>;
        using relationship_type_t = Internal::relationship_t<entity_registry_t *>;

      public:
        friend class Internal::entity_t<entity_registry_t *>;

        template <typename _ComponentType>
        struct signal_handler_t
        {
            // entt::sigh<void( Internal::Entity<EntityRegistry *>, _ComponentType & )> Signal;
            vector_t<std::function<void( entity_type_t, _ComponentType & )>> mHandlers;

            signal_handler_t()                        = default;
            signal_handler_t( const signal_handler_t & ) = default;
        };

        /// @brief Constructs an empty registry
        entity_registry_t()
        {
            mAddSignalHandlers     = CreateRawEntity();
            mUpdateSignalHandlers  = CreateRawEntity();
            mDestroySignalHandlers = CreateRawEntity();
        };

        entity_registry_t( const entity_registry_t & )            = delete;
        entity_registry_t( entity_registry_t &&other )            = delete;
        entity_registry_t &operator=( entity_registry_t &&other ) = delete;
        entity_registry_t &operator=( const entity_registry_t & ) = delete;

        /// @brief Default destructor
        ~entity_registry_t() = default;

        /// @brief Create a new entity in the registry.
        ///
        /// The new entity initially has no components.
        ///
        entity_type_t CreateRawEntity() { return { mRegistry.create(), this }; };

        /// @brief Create a new entity in the registry.
        ///
        /// The new entity initially has no components.
        ///
        entity_type_t CreateEntity()
        {
            entity_type_t lNewEntity = CreateRawEntity();
            lNewEntity.Add<uuid_t>();

            return lNewEntity;
        };

        /// @brief Create a new entity in the registry.
        ///
        /// The new entity initially has no components.
        ///
        entity_type_t CreateEntity( uuid_t const &aUUID )
        {
            entity_type_t lNewEntity = CreateRawEntity();
            lNewEntity.Add<uuid_t>( aUUID );

            return lNewEntity;
        };

        /// @brief Wrap en existing `entt` ID into our registry class.
        entity_type_t WrapEntity( entt::entity const aEntity ) { return { aEntity, this }; };

        /// @brief Create a new entity and add a `Tag` component with the given name.
        ///
        /// @param a_Name The name tag to add to the entity.
        ///
        entity_type_t CreateEntity( string_t const &aName )
        {
            entity_type_t lNewEntity = CreateEntity();
            lNewEntity.Add<tag_t>( aName.empty() ? "Unnamed_Entity" : aName );

            return lNewEntity;
        }

        /// @brief Create a new entity and add a `sRelationship` component.
        entity_type_t CreateEntityWithRelationship()
        {
            entity_type_t lNewEntity = CreateEntity();
            lNewEntity.Add<relationship_type_t>();

            return lNewEntity;
        }

        /// @brief Create a new entity and add a `sRelationship` component and a `Tag` with the given name.
        ///
        /// @param a_Name The name tag to add to the entity.
        ///
        entity_type_t CreateEntityWithRelationship( string_t const &aName )
        {
            entity_type_t lNewEntity = CreateEntity( aName );
            lNewEntity.Add<relationship_type_t>();

            return lNewEntity;
        }

        /// @brief Create a new entity with the given parent entity and name.
        ///
        /// If the parent entity does not have a `sRelationship` component, it will be added.
        ///
        /// @param aParentEntity The parent entity.
        /// @param aName The name tag to add to the entity.
        ///
        entity_type_t CreateEntity( entity_type_t const &aParentEntity, string_t const &aName )
        {
            entity_type_t lNewEntity = CreateEntityWithRelationship( aName );
            SetParent( lNewEntity, aParentEntity );
            
            return lNewEntity;
        }

        /// @brief Remove an entity from the registry.
        ///
        /// @param aEntity Entity to remove.
        ///
        void DestroyEntity( entity_type_t const &aEntity ) { mRegistry.destroy( aEntity ); }

        /// @brief Iterate over all entities containing the listed components.
        ///
        /// @param aApplyFunction Function to apply to each of the listed elements.
        ///
        template <typename... Args>
        void ForEach( std::function<void( entity_type_t, Args &... )> aApplyFunction )
        {
            mRegistry.view<Args...>().each( [this, &aApplyFunction]( const entt::entity entity, Args &...args )
                                            { aApplyFunction( WrapEntity( entity ), std::forward<Args>( args )... ); } );
        }

        /// @brief Sort the entities with the given components according to the provided sort function.
        ///
        /// @param aCompareFunction Comparison function.
        ///
        template <typename Component>
        void Sort( std::function<bool( Component const &c1, Component const &c2 )> aCompareFunction )
        {
            mRegistry.sort<Component>( aCompareFunction );
        }

        /// @brief Sort the entities with the given components according to the provided sort function.
        ///
        /// @param aCompareFunction Comparison function.
        ///
        template <typename Component>
        void Sort( std::function<bool( entity_type_t const &c1, entity_type_t const &c2 )> aCompareFunction )
        {
            mRegistry.sort<Component>( [&]( entt::entity const lhs, entt::entity const rhs )
                                       { return aCompareFunction( WrapEntity( lhs ), WrapEntity( rhs ) ); } );
        }

        /// @brief Set the given entity's parent.
        ///
        /// The `sRelationship` component will be added to either entity that does not already have it.
        ///
        /// @param aEntity The entity.
        /// @param aParentEntity The parent entity.
        ///
        void SetParent( entity_type_t const &aEntity, entity_type_t const &aParentEntity )
        {
            if( !aEntity ) return;

            if( aEntity.Has<relationship_type_t>() )
            {
                auto &lMyRelationship = aEntity.Get<relationship_type_t>();

                if( lMyRelationship.mParent )
                {
                    if( lMyRelationship.mParent == aParentEntity ) return;

                    auto &lSiblings = lMyRelationship.mParent.Get<relationship_type_t>().mChildren;

                    auto &lPositionInSibling = std::find( lSiblings.begin(), lSiblings.end(), aEntity );
                    if( lPositionInSibling != lSiblings.end() ) lSiblings.erase( lPositionInSibling );
                }

                lMyRelationship.mParent = aParentEntity;

                aEntity.Replace<relationship_type_t>( lMyRelationship );
            }
            else
            {
                relationship_type_t lNewRelationship{ aParentEntity, {} };

                aEntity.Add<relationship_type_t>( lNewRelationship );
            }

            if( aParentEntity )
            {
                auto &lParentRelationship = aParentEntity.TryAdd<relationship_type_t>();

                lParentRelationship.mChildren.push_back( aEntity );

                aParentEntity.Replace<relationship_type_t>( lParentRelationship );
            }
        }

        /// @brief Observe when a component of the given type is added to an entity
        ///
        /// The provided function will be called with a handle to the entity to which the component was added,
        /// as well as a reference to the newly added component.
        ///
        /// @tparam Component Component type to observe
        ///
        /// @param aHandler Function to call when a new component is added.
        ///
        template <typename Component>
        void OnComponentAdded( std::function<void( entity_type_t &, Component & )> aHandler )
        {
            if( !mAddSignalHandlers.Has<signal_handler_t<Component>>() )
            {
                mAddSignalHandlers.Add<signal_handler_t<Component>>();
                mRegistry.on_construct<Component>().connect<&entity_registry_t::OnComponentAdded_Implementation<Component>>( *this );
            }

            mAddSignalHandlers.Get<signal_handler_t<Component>>().mHandlers.push_back( aHandler );
        }

        /// @brief Observe when a component of the given type is updated
        ///
        /// The provided function will be called with a handle to the entity to which the component belongs,
        /// as well as a reference to the new value of the component.
        ///
        /// @tparam Component Component type to observe
        ///
        /// @param aHandler Function to call when a new component is updated.
        ///
        template <typename Component>
        void OnComponentUpdated( std::function<void( entity_type_t &, Component & )> aHandler )
        {
            if( !mUpdateSignalHandlers.Has<signal_handler_t<Component>>() )
            {
                mUpdateSignalHandlers.Add<signal_handler_t<Component>>();
                mRegistry.on_update<Component>().connect<&entity_registry_t::OnComponentUpdated_Implementation<Component>>( *this );
            }

            mUpdateSignalHandlers.Get<signal_handler_t<Component>>().mHandlers.push_back( aHandler );
        }

        /// @brief Observe when a component of the given type is removed
        ///
        /// The provided function will be called with a handle to the entity to which the component belongs,
        /// as well as a reference to the value of the component that was just removed.
        ///
        /// @tparam Component Component type to observe
        ///
        /// @param aHandler Function to call when a new component is destroyed.
        ///
        template <typename Component>
        void OnComponentDestroyed( std::function<void( entity_type_t &, Component & )> aHandler )
        {
            if( !mDestroySignalHandlers.Has<signal_handler_t<Component>>() )
            {
                mDestroySignalHandlers.Add<signal_handler_t<Component>>();
                mRegistry.on_destroy<Component>().connect<&entity_registry_t::OnComponentDestroyed_Implementation<Component>>( *this );
            }

            mDestroySignalHandlers.Get<signal_handler_t<Component>>().mHandlers.push_back( aHandler );
        }

        /// @brief Clear the underlying registry
        void Clear()
        {
            mRegistry.clear<>();

            mAddSignalHandlers     = CreateEntity();
            mUpdateSignalHandlers  = CreateEntity();
            mDestroySignalHandlers = CreateEntity();
        }

        std::unordered_map<UUIDv4::UUID, entity_type_t> GetEntityMap()
        {
            std::unordered_map<UUIDv4::UUID, entity_type_t> lResult;
            ForEach<uuid_t>( [&]( auto aEntity, auto &aComponent ) { lResult[aComponent.mValue] = aEntity; } );

            return lResult;
        }

      private:
        entt::registry mRegistry;

        entity_type_t mAddSignalHandlers;
        entity_type_t mUpdateSignalHandlers;
        entity_type_t mDestroySignalHandlers;

      private:
        template <typename Component>
        void OnSignal_Implementation( entt::registry const &aRegistry, entity_type_t const &aHandlers, entt::entity const aEntity )
        {
            if( aHandlers.Has<signal_handler_t<Component>>() )
            {
                auto  l_Entity    = WrapEntity( aEntity );
                auto &l_Component = l_Entity.Get<Component>();

                for( auto &lHandler : aHandlers.Get<signal_handler_t<Component>>().mHandlers )
                {
                    if( lHandler ) lHandler( l_Entity, l_Component );
                }
            }
        }

        template <typename Component>
        void OnComponentAdded_Implementation( entt::registry const &aRegistry, entt::entity const aEntity )
        {
            OnSignal_Implementation<Component>( aRegistry, mAddSignalHandlers, aEntity );
        }

        template <typename Component>
        void OnComponentUpdated_Implementation( entt::registry const &aRegistry, entt::entity const aEntity )
        {
            OnSignal_Implementation<Component>( aRegistry, mUpdateSignalHandlers, aEntity );
        }

        template <typename Component>
        void OnComponentDestroyed_Implementation( entt::registry const &aRegistry, entt::entity const aEntity )
        {
            OnSignal_Implementation<Component>( aRegistry, mDestroySignalHandlers, aEntity );
        }
    };

    /// @brief Instanciated entity type bound to this registry
    using entity_t = Internal::entity_t<entity_registry_t *>;

    /// @brief Instanciated component type bound to this registry
    using behaviour_component_t = Internal::behaviour_component_t<entity_registry_t *>;

    /// @brief Instanciated component type bound to this registry
    using behaviour_controller_t = Internal::behaviour_controller_t<entity_registry_t *>;

    /// @brief Instanciated component type bound to this registry
    using relationship_compoment_t = Internal::relationship_t<entity_registry_t *>;

    // using sActorComponent = Internal::sMonoActor<EntityCollection *>;

    // using sUIComponent = Internal::sMonoUIComponent<EntityCollection *>;

    /// @brief
    template <typename _Ty>
    using join_component_t = Internal::join_t<entity_registry_t *, _Ty>;

} // namespace SE::Core

/// @brief Hash entities so they can be used in unordered maps and sets
template <>
struct std::hash<SE::Core::entity_t>
{
    std::size_t operator()( SE::Core::entity_t const &k ) const { return std::hash<uint32_t>()( static_cast<uint32_t>( k ) ); }
};
