#pragma once

#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

#include "Core/Math/Types.h"
#include "Core/Vector.h"

namespace SE::Core
{
    template <typename _Ty>
    struct numeric_array_t
    {
        vector_t<_Ty> mArray = {};

        void Append( _Ty aValue )
        {
            mArray.push_back( aValue );
        }

        size_t Length()
        {
            return mArray.size();
        }
    };

    using u8_array_t  = numeric_array_t<uint8_t>;
    using u16_array_t = numeric_array_t<uint16_t>;
    using u32_array_t = numeric_array_t<uint32_t>;
    using u64_array_t = numeric_array_t<uint64_t>;

    using i8_array_t  = numeric_array_t<int8_t>;
    using i16_array_t = numeric_array_t<int16_t>;
    using i32_array_t = numeric_array_t<int32_t>;
    using i64_array_t = numeric_array_t<int64_t>;

    using f32_array_t = numeric_array_t<float>;
    using f64_array_t = numeric_array_t<double>;

    using uint2_array_t = numeric_array_t<math::uvec2>;
    using uint3_array_t = numeric_array_t<math::uvec3>;
    using uint4_array_t = numeric_array_t<math::uvec4>;

    using int2_array_t = numeric_array_t<math::ivec2>;
    using int3_array_t = numeric_array_t<math::ivec3>;
    using int4_array_t = numeric_array_t<math::ivec4>;

    using float2_array_t = numeric_array_t<math::vec2>;
    using float3_array_t = numeric_array_t<math::vec3>;
    using float4_array_t = numeric_array_t<math::vec4>;

    using float3x3_array_t = numeric_array_t<math::mat3>;
    using float4x4_array_t = numeric_array_t<math::mat4>;

    template <typename _Ty>
    struct structure_array_t
    {
        vector_t<_Ty> mArray = {};
    };

    using TextureArray = structure_array_t<math::mat3>;

    void DefineArrayTypes( sol::table &aModule );

} // namespace SE::Core
