/// @file   Components.h.h
///
/// @brief  Basic components for entities
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#pragma once

#include <functional>
#include <optional>
#include <random>
#include <string>

#include "Entity.h"

#ifdef LITTLEENDIAN
#    undef LITTLEENDIAN
#endif
#include <uuid_v4.h>

namespace LTSE::Core
{

    struct sTag
    {
        std::string mValue;

        sTag()               = default;
        sTag( const sTag & ) = default;
        sTag( const std::string &aTag )
            : mValue( aTag )
        {
        }
    };

    struct sUUID
    {
        UUIDv4::UUID mValue;

        sUUID()
        {
            UUIDv4::UUIDGenerator<std::mt19937_64> lUuidGenerator;
            mValue = lUuidGenerator.getUUID();
        }
        sUUID( const sUUID & ) = default;
        sUUID( const std::string &aStringUUID )
            : mValue{ UUIDv4::UUID::fromStrFactory( aStringUUID ) }
        {
        }
    };

    template <typename ParentType> struct sRelationship
    {
        Internal::Entity<ParentType> mParent{ entt::null, nullptr };
        std::vector<Internal::Entity<ParentType>> mChildren = {};

        sRelationship()                        = default;
        sRelationship( const sRelationship & ) = default;
    };

} // namespace LTSE::Core
