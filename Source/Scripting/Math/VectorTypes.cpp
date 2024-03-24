#include "GenericVector.h"

#include "Core/Math/Types.h"

namespace SE::Core
{
    using namespace math;

    template <typename _VecType>
    usertype<_VecType> new_vector2_type( sol::table &scriptState, std::string name )
    {
        auto newType = new_vector_type<_VecType>( scriptState, name );
        newType["x"] = &_VecType::x;
        newType["y"] = &_VecType::y;

        // clang-format off
        newType[call_constructor] = factories(
            []() { return _VecType{}; },
            []( sol::table initializer ) {
                if ( initializer["x"].valid() && initializer["y"].valid() )
                    return _VecType{ initializer.get<_VecType::value_type>("x"), initializer.get<_VecType::value_type>("y") };
                return _VecType{ initializer.get<_VecType::value_type>(1), initializer.get<_VecType::value_type>(2) };
            },
            []( _VecType::value_type x, _VecType::value_type y ) { return _VecType{ x, y }; }
        );
        // clang-format on

        return newType;
    }

    template <typename _VecType>
    usertype<_VecType> new_vector3_type( sol::table &scriptState, std::string name )
    {
        auto newType = new_vector_type<_VecType>( scriptState, name );
        newType["x"] = &_VecType::x;
        newType["y"] = &_VecType::y;
        newType["z"] = &_VecType::z;

        // clang-format off
        newType[call_constructor] = factories(
            []() { return _VecType{}; },
            []( sol::table initializer ) {
                if ( initializer["x"].valid() && initializer["y"].valid() && initializer["z"].valid() )
                    return _VecType{ initializer.get<_VecType::value_type>("x"), initializer.get<_VecType::value_type>("y"), initializer.get<_VecType::value_type>("z") };
                return _VecType{ initializer.get<_VecType::value_type>(1), initializer.get<_VecType::value_type>(2), initializer.get<_VecType::value_type>(3) };
            },
            []( _VecType::value_type x, _VecType::value_type y, _VecType::value_type z ) { return _VecType{ x, y, z }; }
        );
        // clang-format on

        return newType;
    }

    template <typename _VecType>
    usertype<_VecType> new_vector4_type( sol::table &scriptState, std::string name )
    {
        auto newType = new_vector_type<_VecType>( scriptState, name );
        newType["x"] = &_VecType::x;
        newType["y"] = &_VecType::y;
        newType["z"] = &_VecType::z;
        newType["w"] = &_VecType::w;

        // clang-format off
        newType[call_constructor] = factories(
            []() { return _VecType{}; },
            []( sol::table initializer ) {
                if ( initializer["x"].valid() && initializer["y"].valid() && initializer["z"].valid() && initializer["w"].valid() )
                    return _VecType{ initializer.get<_VecType::value_type>("x"), initializer.get<_VecType::value_type>("y"), initializer.get<_VecType::value_type>("z"), initializer.get<_VecType::value_type>("w") };
                return _VecType{ initializer.get<_VecType::value_type>(1), initializer.get<_VecType::value_type>(2), initializer.get<_VecType::value_type>(3), initializer.get<_VecType::value_type>(4) };
            },
            []( _VecType::value_type x, _VecType::value_type y, _VecType::value_type z, _VecType::value_type w ) { return _VecType{ x, y, z, w }; }
        );
        // clang-format on

        return newType;
    }

    void define_vector_types( sol::table &module )
    {
        new_vector2_type<vec2>( module, "vec2" );
        declare_vector_operation<vec2>( module );
        module["perpendicular"] = []( vec2 self ) { return perpendicular( self ); };
        module["det"]           = []( vec2 self, vec2 other ) { return det( self, other ); };

        new_vector2_type<ivec2>( module, "ivec2" );
        declare_vector_operation<ivec2>( module );

        new_vector2_type<uvec2>( module, "uvec2" );
        declare_vector_operation<uvec2>( module );

        auto lVec3Type     = new_vector3_type<vec3>( module, "vec3" );
        lVec3Type["cross"] = []( vec3 self, vec3 other ) -> vec3 { return cross( self, other ); };

        declare_vector_operation<vec3>( module );
        module["cross"] = []( vec3 self, vec3 other ) -> vec3 { return cross( self, other ); };

        new_vector3_type<ivec3>( module, "ivec3" );
        declare_vector_operation<ivec3>( module );

        new_vector3_type<uvec3>( module, "uvec3" );
        declare_vector_operation<uvec3>( module );

        new_vector4_type<vec4>( module, "vec4" );
        declare_vector_operation<vec4>( module );

        new_vector4_type<ivec4>( module, "ivec4" );
        declare_vector_operation<ivec4>( module );

        new_vector4_type<uvec4>( module, "uvec4" );
        declare_vector_operation<uvec4>( module );
    }
} // namespace SE::Core