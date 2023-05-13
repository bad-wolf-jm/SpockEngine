/// @file   ScalarTypes.cpp
///
/// @brief  Definitions for typing functions
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#include "ScalarTypes.h"

#include <cstdint>

namespace SE::TensorOps
{

    size_t SizeOf( eScalarType aType )
    {
        switch( aType )
        {
        case eScalarType::FLOAT32: return sizeof( float );
        case eScalarType::FLOAT64: return sizeof( double );
        case eScalarType::UINT8: return sizeof( uint8_t );
        case eScalarType::UINT16: return sizeof( uint16_t );
        case eScalarType::UINT32: return sizeof( uint32_t );
        case eScalarType::UINT64: return sizeof( uint64_t );
        case eScalarType::INT8: return sizeof( int8_t );
        case eScalarType::INT16: return sizeof( int16_t );
        case eScalarType::INT32: return sizeof( int32_t );
        case eScalarType::INT64: return sizeof( int64_t );
        default: return 0;
        }
    }

    eScalarType TypeOf( ScalarValue aType ) { return static_cast<eScalarType>( aType.index() ); }

} // namespace SE::TensorOps