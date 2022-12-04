#pragma once

#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

#include <vector>

#include "Core/Math/Types.h"
#include "Core/CUDA/Texture/Texture2D.h"

namespace SE::Core
{
    template <typename _Ty> struct NumericArray
    {
        std::vector<_Ty> mArray = {};

        void Append( _Ty aValue ) { mArray.push_back( aValue ); }
        size_t Length( ) { return mArray.size( ); }
    };

    using U8Array  = NumericArray<uint8_t>;
    using U16Array = NumericArray<uint16_t>;
    using U32Array = NumericArray<uint32_t>;
    using U64Array = NumericArray<uint64_t>;

    using I8Array  = NumericArray<int8_t>;
    using I16Array = NumericArray<int16_t>;
    using I32Array = NumericArray<int32_t>;
    using I64Array = NumericArray<int64_t>;

    using F32Array = NumericArray<float>;
    using F64Array = NumericArray<double>;

    using UVec2Array = NumericArray<math::uvec2>;
    using UVec3Array = NumericArray<math::uvec3>;
    using UVec4Array = NumericArray<math::uvec4>;

    using IVec2Array = NumericArray<math::ivec2>;
    using IVec3Array = NumericArray<math::ivec3>;
    using IVec4Array = NumericArray<math::ivec4>;

    using Vec2Array = NumericArray<math::vec2>;
    using Vec3Array = NumericArray<math::vec3>;
    using Vec4Array = NumericArray<math::vec4>;

    using Mat3Array = NumericArray<math::mat3>;
    using Mat4Array = NumericArray<math::mat4>;
    using TextureSamplerArray = NumericArray<Cuda::TextureSampler2D>;
    using CudaTextureSamplerArray = NumericArray<Cuda::TextureSampler2D::DeviceData>;

    template <typename _Ty> struct StructureArray
    {
        std::vector<_Ty> mArray = {};
    };

    using TextureArray = NumericArray<math::mat3>;

    void DefineArrayTypes( sol::table &aModule );

} // namespace SE::Core
