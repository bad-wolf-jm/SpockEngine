#include "ArrayTypes.h"

namespace SE::Core
{
    namespace
    {
        template <typename _Ty>
        void new_array_type( sol::table &module, std::string name )
        {
            auto newType = module.new_usertype<numeric_array_t<_Ty>>( name );

            // clang-format off
            newType[sol::call_constructor] = sol::factories(
                []() {
                    return numeric_array_t<_Ty>{};
                },
                [](uint32_t size) {
                    return numeric_array_t<_Ty>{vector_t<_Ty>(size)};
                },
                [](uint32_t size, _Ty value) {
                    return numeric_array_t<_Ty>{vector_t<_Ty>(size, value)};
                },
                [](sol::table arrayData) {
                    vector_t<_Ty> lArray{};
                    auto lArrayDataSize = arrayData.size();
                    for( uint32_t i = 0; i < lArrayDataSize; i++ )
                        lArray.push_back( arrayData.get<_Ty>( i + 1 ) );

                    return numeric_array_t<_Ty>{lArray};
                }
            );
            // clang-format on

            // clang-format off
            newType["append"] = sol::overload(
                [](numeric_array_t<_Ty> &self, _Ty value) {
                    self.mArray.push_back(value);
                },
                [](numeric_array_t<_Ty> &self, numeric_array_t<_Ty> value) {
                    self.mArray.insert(self.mArray.end(), value.mArray.begin(), value.mArray.end());
                }
            );
            // clang-format on

            // clang-format off
            newType["insert"] = sol::overload(
                [](numeric_array_t<_Ty> &self, _Ty value, uint32_t position) {
                    self.mArray.insert(self.mArray.begin() + position, value);
                },
                [](numeric_array_t<_Ty> &self, numeric_array_t<_Ty> value, uint32_t position) {
                    self.mArray.insert(self.mArray.begin() + position, value.mArray.begin(), value.mArray.end());
                }
            );
            // clang-format on

            // &numeric_array_t<_Ty>::Append;
            newType["length"] = &numeric_array_t<_Ty>::Length;

            newType[sol::meta_method::length] = &numeric_array_t<_Ty>::Length;
        }
    } // namespace

    void define_array_types( sol::table &module )
    {
        new_array_type<uint8_t>( module, "U8Array" );
        new_array_type<uint16_t>( module, "U16Array" );
        new_array_type<uint32_t>( module, "U32Array" );
        new_array_type<uint64_t>( module, "U64Array" );

        new_array_type<int8_t>( module, "I8Array" );
        new_array_type<int16_t>( module, "I16Array" );
        new_array_type<int32_t>( module, "I32Array" );
        new_array_type<int64_t>( module, "I64Array" );

        new_array_type<float>( module, "F32Array" );
        new_array_type<double>( module, "F64Array" );

        new_array_type<math::uvec2>( module, "UVec2Array" );
        new_array_type<math::uvec3>( module, "UVec3Array" );
        new_array_type<math::uvec4>( module, "UVec4Array" );

        new_array_type<math::ivec2>( module, "IVec2Array" );
        new_array_type<math::ivec3>( module, "IVec3Array" );
        new_array_type<math::ivec4>( module, "IVec4Array" );

        new_array_type<math::vec2>( module, "Vec2Array" );
        new_array_type<math::vec3>( module, "Vec3Array" );
        new_array_type<math::vec4>( module, "Vec4Array" );

        new_array_type<math::mat3>( module, "Mat3Array" );
        new_array_type<math::mat4>( module, "Mat4Array" );
    }

} // namespace SE::Core