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
    class EntityCollection
    {
      private:
        using EntityType       = Internal::Entity<EntityCollection *>;
        using RelationshipType = Internal::sRelationship<EntityCollection *>;

      public:
        friend class Internal::Entity<EntityCollection *>;

        template <typename _ComponentType>
        struct SignalHandler
        {
            // entt::sigh<void( Internal::Entity<EntityRegistry *>, _ComponentType & )> Signal;
            vec_t<std::function<void( EntityType, _ComponentType & )>> mHandlers;

            SignalHandler()                        = default;
            SignalHandler( const SignalHandler & ) = default;
        };

        /// @brief Constructs an empty registry
        EntityCollection()
        {
            mAddSignalHandlers     = CreateRawEntity();
            mUpdateSignalHandlers  = CreateRawEntity();
            mDestroySignalHandlers = CreateRawEntity();
        };

        EntityCollection( const EntityCollection & )            = delete;
        EntityCollection( EntityCollection &&other )            = delete;
        EntityCollection &operator=( EntityCollection &&other ) = delete;
        EntityCollection &operator=( const EntityCollection & ) = delete;

        /// @brief Default destructor
        ~EntityCollection() = default;

        /// @brief Create a new entity in the registry.
        ///
        /// The new entity initially has no components.
        ///
        EntityType CreateRawEntity() { return { mRegistry.create(), this }; };

        /// @brief Create a new entity in the registry.
        ///
        /// The new entity initially has no components.
        ///
        EntityType CreateEntity()
        {
            EntityType lNewEntity = CreateRawEntity();
            lNewEntity.Add<sUUID>();

            return lNewEntity;
        };

        /// @brief Create a new entity in the registry.
        ///
        /// The new entity initially has no components.
        ///
        EntityType CreateEntity( sUUID const &aUUID )
        {
            EntityType lNewEntity = CreateRawEntity();
            lNewEntity.Add<sUUID>( aUUID );

            return lNewEntity;
        };

        /// @brief Wrap en existing `entt` ID into our registry class.
        EntityType WrapEntity( entt::entity const aEntity ) { return { aEntity, this }; };

        /// @brief Create a new entity and add a `Tag` component with the given name.
        ///
        /// @param a_Name The name tag to add to the entity.
        ///
        EntityType CreateEntity( string_t const &aName )
        {
            EntityType lNewEntity = CreateEntity();
            lNewEntity.Add<sTag>( aName.empty() ? "Unnamed_Entity" : aName );

            return lNewEntity;
        }

        /// @brief Create a new entity and add a `sRelationship` component.
        EntityType CreateEntityWithRelationship()
        {
            EntityType lNewEntity = CreateEntity();
            lNewEntity.Add<RelationshipType>();

            return lNewEntity;
        }

        /// @brief Create a new entity and add a `sRelationship` component and a `Tag` with the given name.
        ///
        /// @param a_Name The name tag to add to the entity.
        ///
        EntityType CreateEntityWithRelationship( string_t const &aName )
        {
            EntityType lNewEntity = CreateEntity( aName );
            lNewEntity.Add<RelationshipType>();

            return lNewEntity;
        }

        /// @brief Create a new entity with the given parent entity and name.
        ///
        /// If the parent entity does not have a `sRelationship` component, it will be added.
        ///
        /// @param aParentEntity The parent entity.
        /// @param aName The name tag to add to the entity.
        ///
        EntityType CreateEntity( EntityType const &aParentEntity, string_t const &aName )
        {
            EntityType lNewEntity = CreateEntityWithRelationship( aName );
            SetParent( lNewEntity, aParentEntity );
            
            return lNewEntity;
        }

        /// @brief Remove an entity from the registry.
        ///
        /// @param aEntity Entity to remove.
        ///
        void DestroyEntity( EntityType const &aEntity ) { mRegistry.destroy( aEntity ); }

        /// @brief Iterate over all entities containing the listed components.
        ///
        /// @param aApplyFunction Function to apply to each of the listed elements.
        ///
        template <typename... Args>
        void ForEach( std::function<void( EntityType, Args &... )> aApplyFunction )
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
        void Sort( std::function<bool( EntityType const &c1, EntityType const &c2 )> aCompareFunction )
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
        void SetParent( EntityType const &aEntity, EntityType const &aParentEntity )
        {
            if( !aEntity ) return;

            if( aEntity.Has<RelationshipType>() )
            {
                auto &lMyRelationship = aEntity.Get<RelationshipType>();

                if( lMyRelationship.mParent )
                {
                    if( lMyRelationship.mParent == aParentEntity ) return;

                    auto &lSiblings = lMyRelationship.mParent.Get<RelationshipType>().mChildren;

                    auto &lPositionInSibling = std::find( lSiblings.begin(), lSiblings.end(), aEntity );
                    if( lPositionInSibling != lSiblings.end() ) lSiblings.erase( lPositionInSibling );
                }

                lMyRelationship.mParent = aParentEntity;

                aEntity.Replace<RelationshipType>( lMyRelationship );
            }
            else
            {
                RelationshipType lNewRelationship{ aParentEntity, {} };

                aEntity.Add<RelationshipType>( lNewRelationship );
            }

            if( aParentEntity )
            {
                auto &lParentRelationship = aParentEntity.TryAdd<RelationshipType>();

                lParentRelationship.mChildren.push_back( aEntity );

                aParentEntity.Replace<RelationshipType>( lParentRelationship );
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
        void OnComponentAdded( std::function<void( EntityType &, Component & )> aHandler )
        {
            if( !mAddSignalHandlers.Has<SignalHandler<Component>>() )
            {
                mAddSignalHandlers.Add<SignalHandler<Component>>();
                mRegistry.on_construct<Component>().connect<&EntityCollection::OnComponentAdded_Implementation<Component>>( *this );
            }

            mAddSignalHandlers.Get<SignalHandler<Component>>().mHandlers.push_back( aHandler );
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
        void OnComponentUpdated( std::function<void( EntityType &, Component & )> aHandler )
        {
            if( !mUpdateSignalHandlers.Has<SignalHandler<Component>>() )
            {
                mUpdateSignalHandlers.Add<SignalHandler<Component>>();
                mRegistry.on_update<Component>().connect<&EntityCollection::OnComponentUpdated_Implementation<Component>>( *this );
            }

            mUpdateSignalHandlers.Get<SignalHandler<Component>>().mHandlers.push_back( aHandler );
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
        void OnComponentDestroyed( std::function<void( EntityType &, Component & )> aHandler )
        {
            if( !mDestroySignalHandlers.Has<SignalHandler<Component>>() )
            {
                mDestroySignalHandlers.Add<SignalHandler<Component>>();
                mRegistry.on_destroy<Component>().connect<&EntityCollection::OnComponentDestroyed_Implementation<Component>>( *this );
            }

            mDestroySignalHandlers.Get<SignalHandler<Component>>().mHandlers.push_back( aHandler );
        }

        /// @brief Clear the underlying registry
        void Clear()
        {
            mRegistry.clear<>();

            mAddSignalHandlers     = CreateEntity();
            mUpdateSignalHandlers  = CreateEntity();
            mDestroySignalHandlers = CreateEntity();
        }

        std::unordered_map<UUIDv4::UUID, EntityType> GetEntityMap()
        {
            std::unordered_map<UUIDv4::UUID, EntityType> lResult;
            ForEach<sUUID>( [&]( auto aEntity, auto &aComponent ) { lResult[aComponent.mValue] = aEntity; } );

            return lResult;
        }

      private:
        entt::registry mRegistry;

        EntityType mAddSignalHandlers;
        EntityType mUpdateSignalHandlers;
        EntityType mDestroySignalHandlers;

      private:
        template <typename Component>
        void OnSignal_Implementation( entt::registry const &aRegistry, EntityType const &aHandlers, entt::entity const aEntity )
        {
            if( aHandlers.Has<SignalHandler<Component>>() )
            {
                auto  l_Entity    = WrapEntity( aEntity );
                auto &l_Component = l_Entity.Get<Component>();

                for( auto &lHandler : aHandlers.Get<SignalHandler<Component>>().mHandlers )
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
    using Entity = Internal::Entity<EntityCollection *>;

    /// @brief Instanciated component type bound to this registry
    using sBehaviourComponent = Internal::sBehaviourComponent<EntityCollection *>;

    /// @brief Instanciated component type bound to this registry
    using sBehaviourController = Internal::sBehaviourController<EntityCollection *>;

    /// @brief Instanciated component type bound to this registry
    using sRelationshipComponent = Internal::sRelationship<EntityCollection *>;

    using sActorComponent = Internal::sMonoActor<EntityCollection *>;

    using sUIComponent = Internal::sMonoUIComponent<EntityCollection *>;

    /// @brief
    template <typename _Ty>
    using sJoinComponent = Internal::sJoin<EntityCollection *, _Ty>;

} // namespace SE::Core

/// @brief Hash entities so they can be used in unordered maps and sets
template <>
struct std::hash<SE::Core::Entity>
{
    std::size_t operator()( SE::Core::Entity const &k ) const { return std::hash<uint32_t>()( static_cast<uint32_t>( k ) ); }
};
