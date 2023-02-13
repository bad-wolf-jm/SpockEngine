/// @file   Entity.h
///
/// @brief  Wrapper for the entity class
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#pragma once

#include "entt/entt.hpp"
#include <functional>

namespace SE::Core::Internal
{

    /// \class Entity
    /// @brief Abstraction for `entt`'s entity type.
    ///
    /// This class maintains a pointer to its parent registry so as to be self contained
    /// when trying to access components.
    ///
    /// @tparam ParentType Always set to `SE::Core::EntityCollection`
    ///
    template <typename ParentType>
    class Entity
    {
      public:
        /// @brief Default constructor.
        Entity() = default;

        /// @brief Copy constructor.
        Entity( const Entity &aOther ) = default;

        /// @brief Construct a `NULL` entity.
        Entity( ParentType aParentRegistry )
            : mEntityHandle( entt::null )
            , mParentRegistry( aParentRegistry )
        {
        }

        /// @brief Wrap constructor.
        Entity( entt::entity const aHandle, ParentType aParentRegistry )
            : mEntityHandle( aHandle )
            , mParentRegistry( aParentRegistry )
        {
        }

        // ///  @brief Tags an entity with an empty component.
        // ///
        // ///  The component type `T` should satisfy `std::is_empty_v<T>()`,
        // ///  and a compilation error wil be raised if not.
        // ///
        // ///  @see @ref Untag()
        // ///
        // ///  @tparam T Component type
        // ///
        // template <typename T>
        // void Tag()
        // {
        //     static_assert( std::is_empty<T>::value, "sTag components should be empty." );
        //     if( Has<T>() ) return;
        //     mParentRegistry->mRegistry.emplace<T>( mEntityHandle );
        // }

        // /// @brief Untags an entity.
        // ///
        // /// The component type `T` should be empty. Does nothing if the entity
        // /// does not have the component.
        // ///
        // /// @see @ref sTag()
        // ///
        // /// @tparam T Component type
        // ///
        // template <typename T>
        // void Untag()
        // {
        //     static_assert( std::is_empty<T>::value, "sTag components should be empty." );
        //     if( !Has<T>() ) return;
        //     mParentRegistry->mRegistry.remove<T>( mEntityHandle );
        // }

        /// @brief Add a component of type `T` to the current entity.
        ///
        /// Attempting to add an empty type as a component will result in an error (use @ref sTag() and
        /// @ref Untag() for this purpose). The arguments to this function are passed directly to `T`'s
        /// constructor. It is an error to add the same component twice to an entity. When in doubt as to
        /// whether a component already exists, use @ref TryAddComponent.
        ///
        /// @tparam T Component type
        ///
        /// @returns A reference to the newly added component.
        ///
        /// @since
        ///
        template <typename T, typename... Args>
        T &Add( Args &&...aArgs ) const
        {
            // static_assert( !std::is_empty<T>::value, "Components should not be empty types." );
            if( mParentRegistry->mRegistry.all_of<T>( mEntityHandle ) ) throw std::runtime_error( "Component already exists" );
            if constexpr( std::is_empty<T>::value )
            {
                mParentRegistry->mRegistry.emplace<T>( mEntityHandle );

                return std::move(T{});
            }
            else
            {

                T &lComponent = mParentRegistry->mRegistry.emplace<T>( mEntityHandle, std::forward<Args>( aArgs )... );

                return lComponent;
            }
        }

        /// @brief Add a component of type `T` to the current entity if it does not apready exist.
        ///
        /// Attempting to add an empty type as a component will result in an error (use @ref sTag() and
        /// @ref Untag() for this purpose). The arguments to this function are passed directly to `T`'s
        /// constructor. If the component already exists, the value already attached to the entity is
        /// returned
        ///
        /// @tparam T Component type
        ///
        /// @returns A reference to the newly added component, or the already existing component.
        ///
        /// @since
        ///
        template <typename T, typename... Args>
        T &TryAdd( Args &&...aArgs ) const
        {
            static_assert( !std::is_empty<T>::value, "Components should not be empty types." );
            if( mParentRegistry->mRegistry.all_of<T>( mEntityHandle ) ) return mParentRegistry->mRegistry.get<T>( mEntityHandle );
            T &lComponent = mParentRegistry->mRegistry.emplace<T>( mEntityHandle, std::forward<Args>( aArgs )... );
            return lComponent;
        }

        /// @brief Replace a component of type `T` to the current entity if it does not already exist.
        ///
        /// The arguments to this function are passed directly to `T`'s constructor. The
        /// entity should already have a component of type `T`.
        ///
        /// @tparam T Component type
        ///
        /// @returns A reference to the newly added component, or the already existing component.
        ///
        /// @since
        ///
        template <typename T, typename... Args>
        T &Replace( Args &&...aArgs ) const
        {
            static_assert( !std::is_empty<T>::value, "Components should not be empty types." );
            T &lComponent = mParentRegistry->mRegistry.replace<T>( mEntityHandle, std::forward<Args>( aArgs )... );
            return lComponent;
        }

        /// @brief Add a component of type `T` to the current entity if it does not already exist. Replace the
        /// component if it does exist
        ///
        /// The arguments to this function are passed directly to `T`'s constructor. If the component already
        /// exists, it is replaced with the new value. If not, then it is created.
        ///
        /// @tparam T Component type
        ///
        /// @returns A reference to the newly added component, or the already existing component.
        ///
        /// @since
        ///
        template <typename T, typename... Args>
        T &AddOrReplace( Args &&...aArgs ) const
        {
            static_assert( !std::is_empty<T>::value, "Components should not be empty types." );
            T &lComponent = mParentRegistry->mRegistry.emplace_or_replace<T>( mEntityHandle, std::forward<Args>( aArgs )... );
            return lComponent;
        }

        /// @brief Retrieve the component with type `T`
        ///
        /// Retrieving a component which hasn't been previously added to the entity results in a runtime error.
        /// Then in doubt, use @ref TryGetComponent or @ref IfExists().
        ///
        /// @tparam T Component type
        ///
        /// @returns A reference to the retrieved component.
        ///
        ///
        template <typename T>
        T &Get() const
        {
            static_assert( !std::is_empty<T>::value, "Components should not be empty types." );
            return mParentRegistry->mRegistry.get<T>( mEntityHandle );
        }

        /// @brief Retrieve the component with type `T`, or a default value
        ///
        /// @tparam T Component type
        ///
        /// @returns A reference to the retrieved component, or the passed in default value if the component
        /// does not exist.
        ///
        template <typename T>
        T &TryGet( T &aDefault ) const
        {
            static_assert( !std::is_empty<T>::value, "Components should not be empty types." );
            if( !Has<T>() ) return aDefault;
            return mParentRegistry->mRegistry.get<T>( mEntityHandle );
        }

        /// @brief Applies a function to the component of type `T` if such a component exists.
        ///
        /// @param a_ApplyFunction The function to apply. This can be a function pointer or a lambda expression
        /// with a reference to `T` as its sole parameter.
        ///
        /// @tparam T Component type
        ///
        template <typename T>
        void IfExists( std::function<void( T & )> aApplyFunction )
        {
            if( !Has<T>() ) return;
            aApplyFunction( Get<T>() );
        }

        /// @brief Test whether component `T` exists in this entity.
        ///
        /// @tparam T Component type
        ///
        /// @returns `true` if the entity has a component of type `T`, and `false` otherwise.
        ///
        template <typename T>
        bool Has() const
        {
            if( mParentRegistry == nullptr ) return false;
            if( mEntityHandle == entt::null ) return false;
            return mParentRegistry->mRegistry.all_of<T>( mEntityHandle );
        }

        /// @brief Test whether this entity contains all of the listed components.
        ///
        /// @tparam Component Component type
        ///
        /// @returns `true` if the entity has a component of type `T` for every `T` in `Components`, and `false` otherwise.
        ///
        template <typename... Component>
        bool HasAll() const
        {
            if( mParentRegistry == nullptr ) return false;
            if( mEntityHandle == entt::null ) return false;
            return mParentRegistry->mRegistry.all_of<Component...>( mEntityHandle );
        }

        /// @brief Test whether this entity contains any of the listed components.
        ///
        /// @tparam Component Component type
        ///
        /// @returns `true` if the entity has a component of type `T` for some `T` in `Components`, and `false` otherwise.
        ///
        template <typename... Component>
        bool HasAny() const
        {
            if( mParentRegistry == nullptr ) return false;

            if( mEntityHandle == entt::null ) return false;

            return mParentRegistry->mRegistry.any_of<Component...>( mEntityHandle );
        }

        /// @brief Removes the component of type `T` from the current entity.
        ///
        /// It is an error to remove a component that has not been previously added, or to remove a component
        /// twice. When in doubt as to whether a component already exists, use @ref TryRemove.
        ///
        /// @tparam T Component type
        ///
        /// @since
        ///
        template <typename T>
        void Remove() const
        {
            if( !( mParentRegistry->mRegistry.all_of<T>( mEntityHandle ) ) ) throw std::runtime_error( "Component does not exists" );
            mParentRegistry->mRegistry.remove<T>( mEntityHandle );
        }

        /// @brief Removes the component of type `T` from the current entity only if it exists.
        ///
        /// It is an error to remove a component that has not been previously added, or to remove a component twice.
        /// When in doubt as to whether a component already exists, use @ref TryRemove.
        ///
        /// @tparam T Component type
        ///
        /// @since
        ///
        template <typename T>
        void TryRemove() const
        {
            if( !( mParentRegistry->mRegistry.all_of<T>( mEntityHandle ) ) ) return;
            mParentRegistry->mRegistry.remove<T>( mEntityHandle );
        }

        /// @brief Joins an entity to the current entity, and link them through the given component.
        ///
        /// Adjoining is a way to access the components of an external entity directly from the curent entity. In some
        /// cases this is preferable to copying the component, since updated to the component in the joined entity are
        /// automatically reflected in the current entity through the join.
        ///
        /// @tparam T Component type
        ///
        /// @param aEntityToBind Second entity to join to the current entity
        ///
        /// @since
        ///
        template <typename _ComponentType>
        void Adjoin( Entity aEntityToBind ) const
        {
            if( !aEntityToBind ) return;

            if( !( aEntityToBind.Has<_ComponentType>() ) ) return;

            AddOrReplace<sJoin<ParentType, _ComponentType>>( aEntityToBind );
        }

        /// @brief Checks whether an entity is valid.
        bool IsValid() const { return mParentRegistry->mRegistry.valid( mEntityHandle ); }

        /// @brief Checks whether an entity is `NULL`.
        operator bool() const { return mEntityHandle != entt::null; }

        /// @brief Use the entity class directly in `entt` functions.
        operator entt::entity() const { return mEntityHandle; }

        /// @brief Retrieve the `entt` ID of an `Entity`
        operator uint32_t() const { return (uint32_t)mEntityHandle; }

        /// @brief Returns a handle to the registry abstraction class.
        ParentType GetRegistry() const { return mParentRegistry; };

        /// @brief Test for equality of two `Entity` instance.
        bool operator==( const Entity &aOther ) const
        {
            return ( mEntityHandle == aOther.mEntityHandle ) && ( mParentRegistry == aOther.mParentRegistry );
        }

        /// @brief Test for difference between two `Entity` instance.
        bool operator!=( const Entity &aOther ) const { return !( *this == aOther ); }

        /// @brief Test for difference between two `Entity` instance.
        Entity &operator=( Entity const &aOther )
        {
            mEntityHandle   = aOther.mEntityHandle;
            mParentRegistry = aOther.mParentRegistry;

            return *this;
        }

      private:
        entt::entity mEntityHandle{ entt::null };
        ParentType   mParentRegistry = nullptr;
    };
} // namespace SE::Core::Internal
