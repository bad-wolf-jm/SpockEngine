/// @file   Components.h.h
///
/// @brief  Basic components for entities
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert. All rights reserved.

#pragma once

#include <functional>
#include <optional>
#include <random>
#include <string>

#include "Core/Definitions.h"
#include "Core/Math/Types.h"

// #include "DotNet/Runtime.h"
#include "Entity.h"

#ifdef LITTLEENDIAN
#    undef LITTLEENDIAN
#endif
#include <uuid_v4.h>

using namespace math;
namespace SE::Core
{
    struct tag_t
    {
        string_t mValue;

        tag_t()                = default;
        tag_t( const tag_t & ) = default;
        tag_t( const string_t &aTag )
            : mValue( aTag )
        {
        }
    };

    struct uuid_t
    {
        UUIDv4::UUID mValue;

        uuid_t()
        {
            UUIDv4::UUIDGenerator<std::mt19937_64> lUuidGenerator;
            mValue = lUuidGenerator.getUUID();
        }
        uuid_t( const uuid_t & ) = default;
        uuid_t( const string_t &aStringUUID )
            : mValue{ UUIDv4::UUID::fromStrFactory( aStringUUID ) }
        {
        }
        uuid_t( const UUIDv4::UUID &aValue )
            : mValue{ aValue }
        {
        }
    };

    namespace Internal
    {
        template <typename ParentType>
        struct relationship_t
        {
            Internal::entity_t<ParentType>           mParent{ entt::null, nullptr };
            vector_t<Internal::entity_t<ParentType>> mChildren = {};

            relationship_t()                         = default;
            relationship_t( const relationship_t & ) = default;
        };

        template <typename ParentType, typename _Ty>
        struct join_t
        {
            Internal::entity_t<ParentType> mJoinEntity{}; //!< Handle to the joined entity

            /// @brief Retrieves a reference to the joined component
            _Ty &JoinedComponent()
            {
                return mJoinEntity.Get<_Ty>();
            }

            join_t()                 = default;
            join_t( const join_t & ) = default;
        };

        template <typename ParentType>
        struct behaviour_controller_t
        {
          public:
            virtual ~behaviour_controller_t() = default;

            template <typename T>
            T &Get()
            {
                return mEntity.Get<T>();
            }
            template <typename T>
            bool Has()
            {
                return mEntity.Has<T>();
            }

            virtual void Initialize( Internal::entity_t<ParentType> aEntity )
            {
                mEntity = aEntity;
            }

            virtual void OnBeginScenario()
            {
            }
            virtual void OnEndScenario()
            {
            }
            virtual void OnTick( Timestep ts )
            {
            }

            Internal::entity_t<ParentType> GetControlledEntity() const
            {
                return mEntity;
            };

          private:
            Internal::entity_t<ParentType> mEntity;
        };

        template <typename ParentType>
        struct behaviour_component_t
        {
            behaviour_controller_t<ParentType> *ControllerInstance = nullptr;

            std::function<behaviour_controller_t<ParentType> *()>      InstantiateController;
            std::function<void( behaviour_component_t<ParentType> * )> DestroyController;

            template <typename T, typename... Args>
            void Bind( Args &&...args )
            {
                InstantiateController = [&]()
                { return reinterpret_cast<behaviour_controller_t<ParentType> *>( new T( std::forward<Args>( args )... ) ); };

                DestroyController = [&]( behaviour_component_t<ParentType> *aNsc )
                {
                    delete aNsc->ControllerInstance;
                    aNsc->ControllerInstance = nullptr;
                };
            }
        };
    } // namespace Internal
} // namespace SE::Core
