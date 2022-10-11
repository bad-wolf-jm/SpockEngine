/// @file   ScalarTypes.h
///
/// @brief  Definitions for Abstract scalar types
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#pragma once

#include <variant>

namespace LTSE::TensorOps
{

    enum class eBroadcastHint : uint8_t
    {
        LEFT  = 0,
        RIGHT = 1,
        NONE  = 2
    };

    using ScalarValue = std::variant<float, double, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t>;

    /// @enum eScalarType
    ///
    /// Enumeration of all primitive numeric types that dan be contrained in a tensor node. The ordering should match
    /// the ordering in the `ScalarValue` variant type.
    ///
    enum class eScalarType : uint8_t
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

    /// @brief Returns the size in bytes for the passed in typs
    ///
    /// @param aType Type of element
    ///
    /// @return Size in bytes
    ///
    size_t SizeOf( eScalarType aType );

    /// @brief Returns the type contained in the passed in @ref ScalarValue
    ///
    /// @param aValue Type of element
    ///
    /// @return Type
    ///
    eScalarType TypeOf( ScalarValue aValue );

} // namespace LTSE::TensorOps
