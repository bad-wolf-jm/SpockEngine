/// @file   Entity.h
///
/// @brief  Wrapper around EnTT's registry class
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#pragma once

#include "entt/entt.hpp"
#include <functional>
#include <optional>

#include "Components.h"
#include "Entity.h"
#include "ScriptableEntity.h"

#include <unordered_map>

namespace LTSE::Core
{

    /// \class EntityRegistry
    ///
    /// @brief Abstraction for `entt`'s registry type.
    ///
    class EntityRegistry
    {
      public:
        friend class Internal::Entity<EntityRegistry *>;

        template <typename _ComponentType>
        struct SignalHandler
        {
            // entt::sigh<void( Internal::Entity<EntityRegistry *>, _ComponentType & )> Signal;
            std::vector<std::function<void( Internal::Entity<EntityRegistry *>, _ComponentType & )>> mHandlers;

            SignalHandler()                        = default;
            SignalHandler( const SignalHandler & ) = default;
        };

        /// @brief Constructs an empty registry
        EntityRegistry()
        {
            mAddSignalHandlers     = CreateEntity();
            mUpdateSignalHandlers  = CreateEntity();
            mDestroySignalHandlers = CreateEntity();
        };

        EntityRegistry( const EntityRegistry & ) = delete;
        EntityRegistry( EntityRegistry &&other ) = delete;
        EntityRegistry &operator=( EntityRegistry &&other ) = delete;
        EntityRegistry &operator=( const EntityRegistry & ) = delete;

        /// @brief Default destructor
        ~EntityRegistry() = default;

        /// @brief Create a new entity in the registry.
        ///
        /// The new entity initially has no components.
        ///
        Internal::Entity<EntityRegistry *> CreateRawEntity() { return { mRegistry.create(), this }; };

        /// @brief Create a new entity in the registry.
        ///
        /// The new entity initially has no components.
        ///
        Internal::Entity<EntityRegistry *> CreateEntity()
        {
            Internal::Entity<EntityRegistry *> l_NewEntity = CreateRawEntity();
            l_NewEntity.Add<sUUID>();

            return l_NewEntity;
        };

        /// @brief Wrap en existing `entt` ID into our registry class.
        Internal::Entity<EntityRegistry *> WrapEntity( entt::entity const aEntity ) { return { aEntity, this }; };

        /// @brief Create a new entity and add a `Tag` component with the given name.
        ///
        /// @param a_Name The name tag to add to the entity.
        ///
        Internal::Entity<EntityRegistry *> CreateEntity( std::string const &aName )
        {
            Internal::Entity<EntityRegistry *> l_NewEntity = CreateEntity();
            l_NewEntity.Add<sTag>( aName.empty() ? "Unnamed_Entity" : aName );
            return l_NewEntity;
        }

        /// @brief Create a new entity and add a `sRelationship` component.
        Internal::Entity<EntityRegistry *> CreateEntityWithRelationship()
        {
            Internal::Entity<EntityRegistry *> l_NewEntity = CreateEntity();
            l_NewEntity.Add<sRelationship<EntityRegistry *>>();
            return l_NewEntity;
        }

        /// @brief Create a new entity and add a `sRelationship` component and a `Tag` with the given name.
        ///
        /// @param a_Name The name tag to add to the entity.
        ///
        Internal::Entity<EntityRegistry *> CreateEntityWithRelationship( std::string const &aName )
        {
            Internal::Entity<EntityRegistry *> l_NewEntity = CreateEntity( aName );
            l_NewEntity.Add<sRelationship<EntityRegistry *>>();
            return l_NewEntity;
        }

        /// @brief Create a new entity with the given parent entity and name.
        ///
        /// If the parent entity does not have a `sRelationship` component, it will be added.
        ///
        /// @param aParentEntity The parent entity.
        /// @param aName The name tag to add to the entity.
        ///
        Internal::Entity<EntityRegistry *> CreateEntity(
            Internal::Entity<EntityRegistry *> const &aParentEntity, std::string const &aName )
        {
            Internal::Entity<EntityRegistry *> l_NewEntity = CreateEntityWithRelationship( aName );
            SetParent( l_NewEntity, aParentEntity );
            return l_NewEntity;
        }

        /// @brief Remove an entity from the registry.
        ///
        /// @param aEntity Entity to remove.
        ///
        void DestroyEntity( Internal::Entity<EntityRegistry *> const &aEntity ) { mRegistry.destroy( aEntity ); }

        /// @brief Iterate over all entities containing the listed components.
        ///
        /// @param aApplyFunction Function to apply to each of the listed elements.
        ///
        template <typename... Args>
        void ForEach( std::function<void( Internal::Entity<EntityRegistry *>, Args &... )> aApplyFunction )
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
        void Sort(
            std::function<bool( Internal::Entity<EntityRegistry *> const &c1, Internal::Entity<EntityRegistry *> const &c2 )>
                aCompareFunction )
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
        void SetParent(
            Internal::Entity<EntityRegistry *> const &aEntity, Internal::Entity<EntityRegistry *> const &aParentEntity )
        {
            if( !aEntity ) return;

            if( aEntity.Has<sRelationship<EntityRegistry *>>() )
            {
                auto &lMyRelationship = aEntity.Get<sRelationship<EntityRegistry *>>();

                if( lMyRelationship.mParent )
                {
                    if( lMyRelationship.mParent == aParentEntity ) return;

                    auto &lSiblings = lMyRelationship.mParent.Get<sRelationship<EntityRegistry *>>().mChildren;

                    auto &lPositionInSibling = std::find( lSiblings.begin(), lSiblings.end(), aEntity );
                    if( lPositionInSibling != lSiblings.end() ) lSiblings.erase( lPositionInSibling );
                }

                lMyRelationship.mParent = aParentEntity;
                aEntity.Replace<sRelationship<EntityRegistry *>>( lMyRelationship );
            }
            else
            {
                sRelationship<EntityRegistry *> lNewRelationship{ aParentEntity, {} };

                aEntity.Add<sRelationship<EntityRegistry *>>( lNewRelationship );
            }

            if( aParentEntity )
            {
                auto &lParentRelationship = aParentEntity.TryAdd<sRelationship<EntityRegistry *>>();

                lParentRelationship.mChildren.push_back( aEntity );

                aParentEntity.Replace<sRelationship<EntityRegistry *>>( lParentRelationship );
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
        void OnComponentAdded(
            std::function<void( Internal::Entity<EntityRegistry *> const &, Component const & )> aHandler )
        {
            if( !mAddSignalHandlers.Has<SignalHandler<Component>>() )
            {
                mAddSignalHandlers.Add<SignalHandler<Component>>();
                mRegistry.on_construct<Component>().connect<&EntityRegistry::OnComponentAdded_Implementation<Component>>(
                    *this );
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
        void OnComponentUpdated(
            std::function<void( Internal::Entity<EntityRegistry *> const &, Component const & )> aHandler )
        {
            if( !mUpdateSignalHandlers.Has<SignalHandler<Component>>() )
            {
                mUpdateSignalHandlers.Add<SignalHandler<Component>>();
                mRegistry.on_update<Component>().connect<&EntityRegistry::OnComponentUpdated_Implementation<Component>>(
                    *this );
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
        void OnComponentDestroyed(
            std::function<void( Internal::Entity<EntityRegistry *> const &, Component const & )> aHandler )
        {
            if( !mDestroySignalHandlers.Has<SignalHandler<Component>>() )
            {
                mDestroySignalHandlers.Add<SignalHandler<Component>>();
                mRegistry.on_destroy<Component>().connect<&EntityRegistry::OnComponentDestroyed_Implementation<Component>>(
                    *this );
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

        std::unordered_map<UUIDv4::UUID, Internal::Entity<EntityRegistry *>> GetEntityMap()
        {
            std::unordered_map<UUIDv4::UUID, Internal::Entity<EntityRegistry *>> lResult;
            ForEach<sUUID>( [&]( auto aEntity, auto &aComponent ) { lResult[aComponent.mValue] = aEntity; } );

            return lResult;
        }

      private:
        entt::registry mRegistry;

        Internal::Entity<EntityRegistry *> mAddSignalHandlers;
        Internal::Entity<EntityRegistry *> mUpdateSignalHandlers;
        Internal::Entity<EntityRegistry *> mDestroySignalHandlers;

      private:
        template <typename Component>
        void OnSignal_Implementation( entt::registry const &aRegistry, Internal::Entity<EntityRegistry *> const &aHandlers,
            entt::entity const aEntity )
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
    using Entity = Internal::Entity<EntityRegistry *>;

    /// @brief Instanciated component type bound to this registry
    using sBehaviourComponent = Internal::sBehaviourComponent<EntityRegistry *>;

    /// @brief Instanciated component type bound to this registry
    using sBehaviourController = Internal::BehaviourController<EntityRegistry *>;

    /// @brief Instanciated component type bound to this registry
    using sRelationshipComponent = sRelationship<EntityRegistry *>;

    /// @brief
    template <typename _Ty>
    using sJoinComponent = Internal::Entity<EntityRegistry *>::sJoin<_Ty>;

} // namespace LTSE::Core

/// @brief Hash entities so they can be used in unordered maps and sets
template <>
struct std::hash<LTSE::Core::Entity>
{
    std::size_t operator()( LTSE::Core::Entity const &k ) const
    {
        return std::hash<uint32_t>()( static_cast<uint32_t>( k ) );
    }
};
