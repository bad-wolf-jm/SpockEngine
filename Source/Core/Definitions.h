/// @file   Core.h
///
/// @brief  Global definitions
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert All rights reserved.

#pragma once

#ifdef _MSC_VER
#    define LOG2( X ) ( (unsigned)( 8 * sizeof( uint64_t ) - __lzcnt64( ( (uint64_t)X ) ) - 1 ) )
#else
#    define LOG2( X ) ( (unsigned)( 8 * sizeof( uint64_t ) - __builtin_clzll( ( (uint64_t)X ) ) - 1 ) )
#endif

#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <variant>

namespace SE::Core
{
    template <typename _Ty>
    using ref_t = std::shared_ptr<_Ty>;

    template <typename _Ty>
    ref_t<_Ty> New()
    {
        return std::make_shared<_Ty>();
    }

    template <typename _Tx, typename... _TOtherArgs>
    ref_t<_Tx> New( _TOtherArgs... args )
    {
        return std::make_shared<_Tx>( std::forward<_TOtherArgs>( args )... );
    }

    template <typename _Ty>
    using wref_t = std::weak_ptr<_Ty>;

    template <typename _Ty>
    using uref_t = std::unique_ptr<_Ty>;

    template <typename _Ty, typename _Tz>
    ref_t<_Ty> Cast( ref_t<_Tz> aElement )
    {
        return std::reinterpret_pointer_cast<_Ty>( aElement );
    }

    template <typename _Ty, typename _Tz>
    _Ty *Cast( _Tz *aElement )
    {
        return reinterpret_cast<_Ty *>( aElement );
    }

    using char_t   = char;
    using string_t = std::string;
    using path_t   = std::filesystem::path;

    template <typename _Ty>
    using vector_t = std::vector<_Ty>;

    template <typename _Ty, std::size_t _Nm>
    using array_t = std::array<_Ty, _Nm>;

    /// @enum eScalarType
    ///
    /// Enumeration of all primitive numeric types that dan be contrained in a tensor node. The ordering should match
    /// the ordering in the `ScalarValue` variant type.
    ///
    enum class scalar_type_t : uint8_t
    {
        FLOAT32 = 0,
        FLOAT64 = 1,
        UINT8   = 2,
        UINT16  = 3,
        UINT32  = 4,
        UINT64  = 5,
        INT8    = 6,
        INT16   = 7,
        INT32   = 8,
        INT64   = 9,
        UNKNOWN = 10
    };

    using scalar_value_t = std::variant<float, double, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t>;
    
    /// @brief Returns the size in bytes for the passed in typs
    ///
    /// @param aType Type of element
    ///
    /// @return Size in bytes
    ///
    size_t SizeOf( scalar_type_t aType );

    /// @brief Returns the type contained in the passed in @ref ScalarValue
    ///
    /// @param aValue Type of element
    ///
    /// @return Type
    ///
    scalar_type_t TypeOf( scalar_value_t aValue );

} // namespace SE::Core