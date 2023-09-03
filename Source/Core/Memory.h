/// @file   PointerView.h
///
/// @brief  Wrapper functions for smart pointers
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert All rights reserved.

#pragma once

#include <memory>

namespace SE::Core
{

    template <typename _Ty>
    std::shared_ptr<_Ty> New()
    {
        return std::make_shared<_Ty>();
    }

    template <typename _Tx, typename... _TOtherArgs>
    std::shared_ptr<_Tx> New( _TOtherArgs... args )
    {
        return std::make_shared<_Tx>( std::forward<_TOtherArgs>( args )... );
    }

    template <typename _Ty>
    using ref_t = std::shared_ptr<_Ty>;
    template <typename _Ty>
    using WeakRef = std::weak_ptr<_Ty>;

    template <typename _Ty, typename _Tz>
    ref_t<_Ty> Cast( ref_t<_Tz> aElement )
    {
        return std::reinterpret_pointer_cast<_Ty>( aElement );
    }

    template <typename _Ty, typename _Tz>
    _Ty *Cast( _Tz *aElement )
    {
        return reinterpret_cast<_Ty*>( aElement );
    }

} // namespace SE::Core