#include "Definitions.h"

namespace numlua::core
{
    size_t size_of( scalar_type_t type )
    {
        switch( type )
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
        case scalar_type_t::VEC2:
            return sizeof( math::vec2 );
        case scalar_type_t::VEC3:
            return sizeof( math::vec3 );
        case scalar_type_t::VEC4:
            return sizeof( math::vec4 );
        case scalar_type_t::MAT3:
            return sizeof( math::mat3 );
        case scalar_type_t::MAT4:
            return sizeof( math::mat4 );
        default:
            return 0;
        }
    }

    scalar_type_t type_of( scalar_value_t type )
    {
        return static_cast<scalar_type_t>( type.index() );
    }

} // namespace SE::Core
