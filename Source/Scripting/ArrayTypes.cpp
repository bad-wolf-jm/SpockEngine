#include "ArrayTypes.h"

namespace LTSE::Core
{
    namespace
    {
        template <typename _Ty> void NewArrayType( sol::table &aModule, std::string aName )
        {
            auto lNewType = aModule.new_usertype<NumericArray<_Ty>>( aName );

            // clang-format off
            lNewType[sol::call_constructor] = sol::factories(
                []() {
                    return NumericArray<_Ty>{};
                },
                [](uint32_t aSize) {
                    return NumericArray<_Ty>{std::vector<_Ty>(aSize)};
                },
                [](uint32_t aSize, _Ty aValue) {
                    return NumericArray<_Ty>{std::vector<_Ty>(aSize, aValue)};
                },
                [](sol::table aArrayData) {
                    std::vector<_Ty> lArray{};
                    auto lArrayDataSize = aArrayData.size();
                    for( uint32_t i = 0; i < lArrayDataSize; i++ )
                        lArray.push_back( aArrayData.get<_Ty>( i + 1 ) );

                    return NumericArray<_Ty>{lArray};
                }
            );
            // clang-format on

            // clang-format off
            lNewType["append"] = sol::overload(
                [](NumericArray<_Ty> &aSelf, _Ty aValue) {
                    aSelf.mArray.push_back(aValue);
                },
                [](NumericArray<_Ty> &aSelf, NumericArray<_Ty> aValue) {
                    aSelf.mArray.insert(aSelf.mArray.end(), aValue.mArray.begin(), aValue.mArray.end());
                }
            );
            // clang-format on

            // clang-format off
            lNewType["insert"] = sol::overload(
                [](NumericArray<_Ty> &aSelf, _Ty aValue, uint32_t aPosition) {
                    aSelf.mArray.insert(aSelf.mArray.begin() + aPosition, aValue);
                },
                [](NumericArray<_Ty> &aSelf, NumericArray<_Ty> aValue, uint32_t aPosition) {
                    aSelf.mArray.insert(aSelf.mArray.begin() + aPosition, aValue.mArray.begin(), aValue.mArray.end());
                }
            );
            // clang-format on

            // &NumericArray<_Ty>::Append;
            lNewType["length"] = &NumericArray<_Ty>::Length;
            lNewType["array"] = &NumericArray<_Ty>::mArray;

            lNewType[sol::meta_method::length]    = &NumericArray<_Ty>::Length;
            lNewType[sol::meta_method::index]     = []( NumericArray<_Ty> &aSelf, int i ) { return aSelf.mArray[i - 1]; };
            lNewType[sol::meta_method::new_index] = []( NumericArray<_Ty> &aSelf, int i, _Ty v ) { aSelf.mArray[i - 1] = v; };
        }
    } // namespace

    void DefineArrayTypes( sol::table &aModule )
    {
        NewArrayType<uint8_t>( aModule, "U8Array" );
        NewArrayType<uint16_t>( aModule, "U16Array" );
        NewArrayType<uint32_t>( aModule, "U32Array" );
        NewArrayType<uint64_t>( aModule, "U64Array" );

        NewArrayType<int8_t>( aModule, "I8Array" );
        NewArrayType<int16_t>( aModule, "I16Array" );
        NewArrayType<int32_t>( aModule, "I32Array" );
        NewArrayType<int64_t>( aModule, "I64Array" );

        NewArrayType<float>( aModule, "F32Array" );
        NewArrayType<double>( aModule, "F64Array" );

        NewArrayType<math::uvec2>( aModule, "UVec2Array" );
        NewArrayType<math::uvec3>( aModule, "UVec3Array" );
        NewArrayType<math::uvec4>( aModule, "UVec4Array" );

        NewArrayType<math::ivec2>( aModule, "IVec2Array" );
        NewArrayType<math::ivec3>( aModule, "IVec3Array" );
        NewArrayType<math::ivec4>( aModule, "IVec4Array" );

        NewArrayType<math::vec2>( aModule, "Vec2Array" );
        NewArrayType<math::vec3>( aModule, "Vec3Array" );
        NewArrayType<math::vec4>( aModule, "Vec4Array" );

        NewArrayType<math::mat3>( aModule, "Mat3Array" );
        NewArrayType<math::mat4>( aModule, "Mat4Array" );

        NewArrayType<Cuda::TextureSampler2D>( aModule, "TextureSamplerArray" );
        NewArrayType<Cuda::TextureSampler2D::DeviceData>( aModule, "CudaTextureSamplerArray" );
    }

} // namespace LTSE::Core