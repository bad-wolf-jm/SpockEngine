#include "Definitions.h"

namespace SE::Core
{

    size_t size_of( scalar_type_t aType )
    {
        switch( aType )
        {
        case scalar_type_t::FLOAT32:
            return sizeof( float );
        case scalar_type_t::FLOAT64:
            return sizeof( double );
        case scalar_type_t::UINT8:
            return sizeof( uint8_t );
        case scalar_type_t::UINT16:
            return sizeof( uint16_t );
        case scalar_type_t::UINT32:
            return sizeof( uint32_t );
        case scalar_type_t::UINT64:
            return sizeof( uint64_t );
        case scalar_type_t::INT8:
            return sizeof( int8_t );
        case scalar_type_t::INT16:
            return sizeof( int16_t );
        case scalar_type_t::INT32:
            return sizeof( int32_t );
        case scalar_type_t::INT64:
            return sizeof( int64_t );
        default:
            return 0;
        }
    }

    scalar_type_t type_of( scalar_value_t aType )
    {
        return static_cast<scalar_type_t>( aType.index() );
    }

} // namespace SE::Core