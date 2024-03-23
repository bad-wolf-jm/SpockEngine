#include "GenericVector.h"

#include "Core/Math/Types.h"

namespace SE::Core
{
    using namespace math;

    template <typename _VecType>
    usertype<_VecType> NewVector2Type( sol::table &aScriptState, std::string aName )
    {
        auto lNewType = NewVectorType<_VecType>( aScriptState, aName );
        lNewType["x"] = &_VecType::x;
        lNewType["y"] = &_VecType::y;

        // clang-format off
        lNewType[call_constructor] = factories(
            []() { return _VecType{}; },
            []( sol::table aInitializer ) {
                if ( aInitializer["x"].valid() && aInitializer["y"].valid() )
                    return _VecType{ aInitializer.get<_VecType::value_type>("x"), aInitializer.get<_VecType::value_type>("y") };
                return _VecType{ aInitializer.get<_VecType::value_type>(1), aInitializer.get<_VecType::value_type>(2) };
            },
            []( _VecType::value_type x, _VecType::value_type y ) { return _VecType{ x, y }; }
        );
        // clang-format on

        return lNewType;
    }

    template <typename _VecType>
    usertype<_VecType> NewVector3Type( sol::table &aScriptState, std::string aName )
    {
        auto lNewType = NewVectorType<_VecType>( aScriptState, aName );
        lNewType["x"] = &_VecType::x;
        lNewType["y"] = &_VecType::y;
        lNewType["z"] = &_VecType::z;

        // clang-format off
        lNewType[call_constructor] = factories(
            []() { return _VecType{}; },
            []( sol::table aInitializer ) {
                if ( aInitializer["x"].valid() && aInitializer["y"].valid() && aInitializer["z"].valid() )
                    return _VecType{ aInitializer.get<_VecType::value_type>("x"), aInitializer.get<_VecType::value_type>("y"), aInitializer.get<_VecType::value_type>("z") };
                return _VecType{ aInitializer.get<_VecType::value_type>(1), aInitializer.get<_VecType::value_type>(2), aInitializer.get<_VecType::value_type>(3) };
            },
            []( _VecType::value_type x, _VecType::value_type y, _VecType::value_type z ) { return _VecType{ x, y, z }; }
        );
        // clang-format on

        return lNewType;
    }

    template <typename _VecType>
    usertype<_VecType> NewVector4Type( sol::table &aScriptState, std::string aName )
    {
        auto lNewType = NewVectorType<_VecType>( aScriptState, aName );
        lNewType["x"] = &_VecType::x;
        lNewType["y"] = &_VecType::y;
        lNewType["z"] = &_VecType::z;
        lNewType["w"] = &_VecType::w;

        // clang-format off
        lNewType[call_constructor] = factories(
            []() { return _VecType{}; },
            []( sol::table aInitializer ) {
                if ( aInitializer["x"].valid() && aInitializer["y"].valid() && aInitializer["z"].valid() && aInitializer["w"].valid() )
                    return _VecType{ aInitializer.get<_VecType::value_type>("x"), aInitializer.get<_VecType::value_type>("y"), aInitializer.get<_VecType::value_type>("z"), aInitializer.get<_VecType::value_type>("w") };
                return _VecType{ aInitializer.get<_VecType::value_type>(1), aInitializer.get<_VecType::value_type>(2), aInitializer.get<_VecType::value_type>(3), aInitializer.get<_VecType::value_type>(4) };
            },
            []( _VecType::value_type x, _VecType::value_type y, _VecType::value_type z, _VecType::value_type w ) { return _VecType{ x, y, z, w }; }
        );
        // clang-format on

        return lNewType;
    }

    void define_vector_types( sol::table &aModule )
    {
        NewVector2Type<vec2>( aModule, "vec2" );
        DeclareVectorOperation<vec2>( aModule );
        aModule["perpendicular"] = []( vec2 aSelf ) { return perpendicular( aSelf ); };
        aModule["det"]           = []( vec2 aSelf, vec2 aOther ) { return det( aSelf, aOther ); };

        NewVector2Type<ivec2>( aModule, "ivec2" );
        DeclareVectorOperation<ivec2>( aModule );

        NewVector2Type<uvec2>( aModule, "uvec2" );
        DeclareVectorOperation<uvec2>( aModule );

        auto lVec3Type     = NewVector3Type<vec3>( aModule, "vec3" );
        lVec3Type["cross"] = []( vec3 aSelf, vec3 aOther ) -> vec3 { return cross( aSelf, aOther ); };

        DeclareVectorOperation<vec3>( aModule );
        aModule["cross"] = []( vec3 aSelf, vec3 aOther ) -> vec3 { return cross( aSelf, aOther ); };

        NewVector3Type<ivec3>( aModule, "ivec3" );
        DeclareVectorOperation<ivec3>( aModule );

        NewVector3Type<uvec3>( aModule, "uvec3" );
        DeclareVectorOperation<uvec3>( aModule );

        NewVector4Type<vec4>( aModule, "vec4" );
        DeclareVectorOperation<vec4>( aModule );

        NewVector4Type<ivec4>( aModule, "ivec4" );
        DeclareVectorOperation<ivec4>( aModule );

        NewVector4Type<uvec4>( aModule, "uvec4" );
        DeclareVectorOperation<uvec4>( aModule );
    }
} // namespace SE::Core