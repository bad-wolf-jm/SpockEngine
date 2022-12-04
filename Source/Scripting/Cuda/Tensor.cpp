#include "Tensor.h"

#include "Core/GPUResource/Array/MultiTensor.h"
#include "Core/Memory.h"


#include "TensorOps/ScalarTypes.h"
#include "TensorOps/Scope.h"

#include "Core/Logging.h"

#include "Scripting/ArrayTypes.h"
#include "Scripting/PrimitiveTypes.h"

namespace SE::Core
{
    using namespace sol;
    using namespace entt::literals;
    using namespace SE::TensorOps;

    namespace
    {
        template <typename _Ty>
        auto MakeUploadFunction()
        {
            return overload(
                []( Cuda::MultiTensor &aSelf, NumericArray<_Ty> &aValues )
                {
                    aSelf.Upload( aValues.mArray );

                    return aSelf;
                },
                []( Cuda::MultiTensor &aSelf, std::vector<_Ty> &aValues )
                {
                    aSelf.Upload( aValues );

                    return aSelf;
                },
                []( Cuda::MultiTensor &aSelf, NumericArray<_Ty> &aValues, uint32_t aLayer )
                {
                    aSelf.Upload( aValues.mArray, aLayer );

                    return aSelf;
                },
                []( Cuda::MultiTensor &aSelf, std::vector<_Ty> &aValues, uint32_t aLayer )
                {
                    aSelf.Upload( aValues, aLayer );

                    return aSelf;
                } );
        }

        template <typename _Ty>
        auto MakeFetchFunction()
        {
            return overload( []( Cuda::MultiTensor &aSelf ) { return aSelf.FetchFlattened<_Ty>(); },
                             []( Cuda::MultiTensor &aSelf, uint32_t aLayer ) { return aSelf.FetchBufferAt<_Ty>( aLayer ); } );
        }
    } // namespace

    void OpenTensorLibrary( sol::table &aScriptingState )
    {
        auto lTensorShapeType = aScriptingState.new_usertype<Cuda::sTensorShape>( "sTensorShape" );

        // clang-format off
        lTensorShapeType[call_constructor] = factories(
            []( U32Array aInitializer, int32_t aElementSize )
            {
                return Cuda::sTensorShape( aInitializer.mArray, aElementSize );
            }
            ,
            []( sol::table aInitializer, int32_t aElementSize )
            {
                std::vector<std::vector<uint32_t>> lShape{};

                for (uint32_t i=0; i < aInitializer.size(); i++)
                {
                    auto lLayer = aInitializer.get<sol::table>( i + 1 );
                    lShape.push_back( std::vector<uint32_t>{} );

                    for (uint32_t j=0; j < lLayer.size(); j++)
                    {
                        auto lDim = lLayer.get<uint32_t>( j + 1 );

                        lShape.back().push_back( lDim );
                    }
                }

                return Cuda::sTensorShape( lShape, aElementSize );
            }
        );
        // clang-format on

        lTensorShapeType["count_layers"]  = []( Cuda::sTensorShape &aSelf ) { return aSelf.CountLayers(); };
        lTensorShapeType["get_dimension"] = []( Cuda::sTensorShape &aSelf, int32_t i ) { return aSelf.GetDimension( i ); };
        lTensorShapeType["trim"]          = []( Cuda::sTensorShape &aSelf, int32_t i ) { aSelf.Trim( i ); };
        lTensorShapeType["flatten"]       = []( Cuda::sTensorShape &aSelf, int32_t i ) { aSelf.Flatten( i ); };

        auto lMemoryPoolType =
            aScriptingState.new_usertype<Cuda::MemoryPool>( "MemoryPool", constructors<Cuda::MemoryPool( uint32_t aMemorySize )>() );
        lMemoryPoolType["reset"]    = []( Cuda::MemoryPool &aSelf ) { aSelf.Reset(); };
        lMemoryPoolType["allocate"] = []( Cuda::MemoryPool &aSelf, int32_t aBytes ) { return aSelf.Allocate( aBytes ); };

        auto lMultiTensorType = aScriptingState.new_usertype<Cuda::MultiTensor>(
            "MultiTensor", constructors<Cuda::MultiTensor( Cuda::MemoryPool & aMemoryPool, const Cuda::sTensorShape &aShape )>() );
        lMultiTensorType["size"]    = []( Cuda::MultiTensor &aSelf ) { return aSelf.Size(); };
        lMultiTensorType["shape"]   = []( Cuda::MultiTensor &aSelf ) { return aSelf.Shape(); };
        lMultiTensorType["size_as"] = []( Cuda::MultiTensor &aSelf, const sol::object &aTypeOrID )
        {
            const auto lMaybeAny = InvokeMetaFunction( DeduceType( aTypeOrID ), "SizeAs"_hs, aSelf );

            return lMaybeAny ? lMaybeAny.cast<size_t>() : 0;
        };

        lMultiTensorType["upload_u8"]  = MakeUploadFunction<uint8_t>();
        lMultiTensorType["upload_u16"] = MakeUploadFunction<uint16_t>();
        lMultiTensorType["upload_u32"] = MakeUploadFunction<uint32_t>();
        lMultiTensorType["upload_u64"] = MakeUploadFunction<uint32_t>();

        lMultiTensorType["upload_i8"]  = MakeUploadFunction<int8_t>();
        lMultiTensorType["upload_i16"] = MakeUploadFunction<int16_t>();
        lMultiTensorType["upload_i32"] = MakeUploadFunction<int32_t>();
        lMultiTensorType["upload_i64"] = MakeUploadFunction<int32_t>();

        lMultiTensorType["upload_f32"] = MakeUploadFunction<float>();
        lMultiTensorType["upload_f64"] = MakeUploadFunction<double>();

        lMultiTensorType["upload_uvec2"] = MakeUploadFunction<math::uvec2>();
        lMultiTensorType["upload_uvec3"] = MakeUploadFunction<math::uvec3>();
        lMultiTensorType["upload_uvec4"] = MakeUploadFunction<math::uvec4>();

        lMultiTensorType["upload_ivec2"] = MakeUploadFunction<math::ivec2>();
        lMultiTensorType["upload_ivec3"] = MakeUploadFunction<math::ivec3>();
        lMultiTensorType["upload_ivec4"] = MakeUploadFunction<math::ivec4>();

        lMultiTensorType["upload_vec2"] = MakeUploadFunction<math::vec2>();
        lMultiTensorType["upload_vec3"] = MakeUploadFunction<math::vec3>();
        lMultiTensorType["upload_vec4"] = MakeUploadFunction<math::vec4>();

        lMultiTensorType["upload_mat3"] = MakeUploadFunction<math::mat3>();
        lMultiTensorType["upload_mat4"] = MakeUploadFunction<math::mat4>();

        lMultiTensorType["fetch_u8"]  = MakeFetchFunction<uint8_t>();
        lMultiTensorType["fetch_u16"] = MakeFetchFunction<uint16_t>();
        lMultiTensorType["fetch_u32"] = MakeFetchFunction<uint32_t>();
        lMultiTensorType["fetch_u64"] = MakeFetchFunction<uint32_t>();

        lMultiTensorType["fetch_i8"]  = MakeFetchFunction<int8_t>();
        lMultiTensorType["fetch_i16"] = MakeFetchFunction<int16_t>();
        lMultiTensorType["fetch_i32"] = MakeFetchFunction<int32_t>();
        lMultiTensorType["fetch_i64"] = MakeFetchFunction<int32_t>();

        lMultiTensorType["fetch_f32"] = MakeFetchFunction<float>();
        lMultiTensorType["fetch_f64"] = MakeFetchFunction<double>();

        lMultiTensorType["fetch_uvec2"] = MakeFetchFunction<math::uvec2>();
        lMultiTensorType["fetch_uvec3"] = MakeFetchFunction<math::uvec3>();
        lMultiTensorType["fetch_uvec4"] = MakeFetchFunction<math::uvec4>();

        lMultiTensorType["fetch_ivec2"] = MakeFetchFunction<math::ivec2>();
        lMultiTensorType["fetch_ivec3"] = MakeFetchFunction<math::ivec3>();
        lMultiTensorType["fetch_ivec4"] = MakeFetchFunction<math::ivec4>();

        lMultiTensorType["fetch_vec2"] = MakeFetchFunction<math::vec2>();
        lMultiTensorType["fetch_vec3"] = MakeFetchFunction<math::vec3>();
        lMultiTensorType["fetch_vec4"] = MakeFetchFunction<math::vec4>();

        lMultiTensorType["fetch_mat3"] = MakeFetchFunction<math::mat3>();
        lMultiTensorType["fetch_mat4"] = MakeFetchFunction<math::mat4>();

        auto lScopeType =
            aScriptingState.new_usertype<TensorOps::Scope>( "Scope", constructors<TensorOps::Scope( uint32_t aMemorySize )>() );
        lScopeType["ref_new"] = []( uint32_t aMemorySize ) { return New<TensorOps::Scope>( aMemorySize ); };
        lScopeType["reset"]   = []( TensorOps::Scope &aSelf ) { aSelf.Reset(); };

        // clang-format off
        lScopeType["run"] = overload(
            []( TensorOps::Scope &aSelf, TensorOps::OpNode &aNode ) { aSelf.Run( aNode ); },
            []( TensorOps::Scope &aSelf, std::vector<TensorOps::OpNode> aNode ) { aSelf.Run( aNode ); },
            []( TensorOps::Scope &aSelf, sol::table aNode )
            {
                std::vector<TensorOps::OpNode> lOpNodes{};

                for (uint32_t i=0; i < aNode.size(); i++)
                {
                    auto lNode = aNode.get<TensorOps::OpNode>( i + 1 );
                    lOpNodes.push_back( lNode );
                }

                aSelf.Run( lOpNodes );
            }
        );
        // clang-format on

        auto lOpsModule = aScriptingState["Ops"].get_or_create<sol::table>();

        // clang-format off
        lOpsModule.new_enum( "eScalarType",
            "FLOAT32", eScalarType::FLOAT32,
            "FLOAT64", eScalarType::FLOAT64,
            "UINT8",   eScalarType::UINT8,
            "UINT16",  eScalarType::UINT16,
            "UINT32",  eScalarType::UINT32,
            "UINT64",  eScalarType::UINT64,
            "INT8",    eScalarType::INT8,
            "INT16",   eScalarType::INT16,
            "INT32",   eScalarType::INT32,
            "INT64",   eScalarType::INT64,
            "UNKNOWN", eScalarType::UNKNOWN  );
        // clang-format on

        lOpsModule["size_of"] = []( eScalarType t ) { return static_cast<int32_t>( SizeOf( t ) ); };

        auto lMultiensorComponentType     = DeclarePrimitiveType<sMultiTensorComponent>( lOpsModule, "sMultiTensorComponent" );
        lMultiensorComponentType["value"] = &sMultiTensorComponent::mValue;

        // clang-format off
        auto lConstantInitializerComponent = lOpsModule.new_usertype<sConstantValueInitializerComponent>("sConstantValueInitializerComponent");
        lConstantInitializerComponent[call_constructor] = [](eScalarType aType, double value)
        {
            switch(aType)
            {
            case eScalarType::FLOAT32:
                return sConstantValueInitializerComponent{ static_cast<float>(value) };
            case eScalarType::FLOAT64:
                return sConstantValueInitializerComponent{ static_cast<double>(value) };
            case eScalarType::UINT8:
                return sConstantValueInitializerComponent{ static_cast<uint8_t>(value) };
            case eScalarType::UINT16:
                return sConstantValueInitializerComponent{ static_cast<uint16_t>(value) };
            case eScalarType::UINT32:
                return sConstantValueInitializerComponent{ static_cast<uint32_t>(value) };
            case eScalarType::UINT64:
                return sConstantValueInitializerComponent{ static_cast<uint64_t>(value) };
            case eScalarType::INT8:
                return sConstantValueInitializerComponent{ static_cast<int8_t>(value) };
            case eScalarType::INT16:
                return sConstantValueInitializerComponent{ static_cast<int16_t>(value) };
            case eScalarType::INT32:
                return sConstantValueInitializerComponent{ static_cast<int32_t>(value) };
            case eScalarType::INT64:
                return sConstantValueInitializerComponent{ static_cast<int64_t>(value) };
            case eScalarType::UNKNOWN:
                break;
            }
        };

        auto lVectorInitializerComponent = lOpsModule.new_usertype<sVectorInitializerComponent>( "sVectorInitializerComponent" );

        // clang-format off
        lVectorInitializerComponent[call_constructor] = factories(
            []( F32Array value) { return sVectorInitializerComponent{ value.mArray }; },
            []( F64Array value) { return sVectorInitializerComponent{ value.mArray }; },
            []( U8Array value)  { return sVectorInitializerComponent{ value.mArray }; },
            []( U16Array value) { return sVectorInitializerComponent{ value.mArray }; },
            []( U32Array value) { return sVectorInitializerComponent{ value.mArray }; },
            []( U64Array value) { return sVectorInitializerComponent{ value.mArray }; },
            []( I8Array value)  { return sVectorInitializerComponent{ value.mArray }; },
            []( I16Array value) { return sVectorInitializerComponent{ value.mArray }; },
            []( I32Array value) { return sVectorInitializerComponent{ value.mArray }; },
            []( I64Array value) { return sVectorInitializerComponent{ value.mArray }; },
            []( std::vector<float> value)    { return sVectorInitializerComponent{ value }; },
            []( std::vector<double> value)   { return sVectorInitializerComponent{ value }; },
            []( std::vector<uint8_t> value)  { return sVectorInitializerComponent{ value }; },
            []( std::vector<uint16_t> value) { return sVectorInitializerComponent{ value }; },
            []( std::vector<uint32_t> value) { return sVectorInitializerComponent{ value }; },
            []( std::vector<uint64_t> value) { return sVectorInitializerComponent{ value }; },
            []( std::vector<int8_t>  value)  { return sVectorInitializerComponent{ value }; },
            []( std::vector<int16_t> value)  { return sVectorInitializerComponent{ value }; },
            []( std::vector<int32_t> value)  { return sVectorInitializerComponent{ value }; },
            []( std::vector<int64_t> value)  { return sVectorInitializerComponent{ value }; }
        );
        // clang-format on

        auto lDataInitializerComponent = lOpsModule.new_usertype<sDataInitializerComponent>( "sDataInitializerComponent" );
        // clang-format off
        lDataInitializerComponent[call_constructor] = factories(
            []( F32Array value) { return sDataInitializerComponent{ value.mArray }; },
            []( F64Array value) { return sDataInitializerComponent{ value.mArray }; },
            []( U8Array value)  { return sDataInitializerComponent{ value.mArray }; },
            []( U16Array value) { return sDataInitializerComponent{ value.mArray }; },
            []( U32Array value) { return sDataInitializerComponent{ value.mArray }; },
            []( U64Array value) { return sDataInitializerComponent{ value.mArray }; },
            []( I8Array value)  { return sDataInitializerComponent{ value.mArray }; },
            []( I16Array value) { return sDataInitializerComponent{ value.mArray }; },
            []( I32Array value) { return sDataInitializerComponent{ value.mArray }; },
            []( I64Array value) { return sDataInitializerComponent{ value.mArray }; },
            []( std::vector<float> value)    { return sDataInitializerComponent{ value }; },
            []( std::vector<double> value)   { return sDataInitializerComponent{ value }; },
            []( std::vector<uint8_t> value)  { return sDataInitializerComponent{ value }; },
            []( std::vector<uint16_t> value) { return sDataInitializerComponent{ value }; },
            []( std::vector<uint32_t> value) { return sDataInitializerComponent{ value }; },
            []( std::vector<uint64_t> value) { return sDataInitializerComponent{ value }; },
            []( std::vector<int8_t>  value)  { return sDataInitializerComponent{ value }; },
            []( std::vector<int16_t> value)  { return sDataInitializerComponent{ value }; },
            []( std::vector<int32_t> value)  { return sDataInitializerComponent{ value }; },
            []( std::vector<int64_t> value)  { return sDataInitializerComponent{ value }; }
        );
        // clang-format on

        auto lRandomUniformInitializerComponent =
            lOpsModule.new_usertype<sRandomUniformInitializerComponent>( "sRandomUniformInitializerComponent" );
        lRandomUniformInitializerComponent[call_constructor] = []( eScalarType value )
        { return sRandomUniformInitializerComponent{ value }; };

        auto lRandomNormalInitializerComponent =
            lOpsModule.new_usertype<sRandomNormalInitializerComponent>( "sRandomNormalInitializerComponent" );
        lRandomNormalInitializerComponent[call_constructor] = []( eScalarType value, double mean, double std )
        {
            switch( value )
            {
            case eScalarType::FLOAT64: return sRandomNormalInitializerComponent{ value, mean, std };
            default: return sRandomNormalInitializerComponent{ value, static_cast<float>( mean ), static_cast<float>( std ) };
            }
        };

        // clang-format off
        lOpsModule["ScalarVectorValue"] = overload(
            []( Scope &aScope, eScalarType aType, F32Array const &aValue ) {
                return ScalarVectorValue( aScope, aType, aValue.mArray );
            },
            []( Scope &aScope, eScalarType aType, F64Array const &aValue ) {
                return ScalarVectorValue( aScope, aType, aValue.mArray );
            },
            []( Scope &aScope, eScalarType aType, U8Array const &aValue ) {
                return ScalarVectorValue( aScope, aType, aValue.mArray );
            },
            []( Scope &aScope, eScalarType aType, U16Array const &aValue ) {
                return ScalarVectorValue( aScope, aType, aValue.mArray );
            },
            []( Scope &aScope, eScalarType aType, U32Array const &aValue ) {
                return ScalarVectorValue( aScope, aType, aValue.mArray );
            },
            []( Scope &aScope, eScalarType aType, U64Array const &aValue ) {
                return ScalarVectorValue( aScope, aType, aValue.mArray );
            },
            []( Scope &aScope, eScalarType aType, I8Array const &aValue ) {
                return ScalarVectorValue( aScope, aType, aValue.mArray );
            },
            []( Scope &aScope, eScalarType aType, I16Array const &aValue ) {
                return ScalarVectorValue( aScope, aType, aValue.mArray );
            },
            []( Scope &aScope, eScalarType aType, I32Array const &aValue ) {
                return ScalarVectorValue( aScope, aType, aValue.mArray );
            },
            []( Scope &aScope, eScalarType aType, I64Array const &aValue ) {
                return ScalarVectorValue( aScope, aType, aValue.mArray );
            },
            []( Scope &aScope, eScalarType aType, std::vector<float> const &aValue ) {
                return ScalarVectorValue( aScope, aType, aValue );
            },
            []( Scope &aScope, eScalarType aType, std::vector<double> const &aValue ) {
                return ScalarVectorValue( aScope, aType, aValue );
            },
            []( Scope &aScope, eScalarType aType, std::vector<uint8_t> const &aValue ) {
                return ScalarVectorValue( aScope, aType, aValue );
            },
            []( Scope &aScope, eScalarType aType, std::vector<uint16_t> const &aValue ) {
                return ScalarVectorValue( aScope, aType, aValue );
            },
            []( Scope &aScope, eScalarType aType, std::vector<uint32_t> const &aValue ) {
                return ScalarVectorValue( aScope, aType, aValue );
            },
            []( Scope &aScope, eScalarType aType, std::vector<uint64_t> const &aValue ) {
                return ScalarVectorValue( aScope, aType, aValue );
            },
            []( Scope &aScope, eScalarType aType, std::vector<int8_t> const &aValue ) {
                return ScalarVectorValue( aScope, aType, aValue );
            },
            []( Scope &aScope, eScalarType aType, std::vector<int16_t> const &aValue ) {
                return ScalarVectorValue( aScope, aType, aValue );
            },
            []( Scope &aScope, eScalarType aType, std::vector<int32_t> const &aValue ) {
                return ScalarVectorValue( aScope, aType, aValue );
            },
            []( Scope &aScope, eScalarType aType, std::vector<int64_t> const &aValue ) {
                return ScalarVectorValue( aScope, aType, aValue );
            }
        );
        // clang-format on

        // clang-format off
        lOpsModule["VectorValue"] = overload(
            []( Scope &aScope, F32Array const &aValue ) {
                return VectorValue( aScope, aValue.mArray );
            },
            []( Scope &aScope, F64Array const &aValue ) {
                return VectorValue( aScope, aValue.mArray );
            },
            []( Scope &aScope, U8Array const &aValue ) {
                return VectorValue( aScope, aValue.mArray );
            },
            []( Scope &aScope, U16Array const &aValue ) {
                return VectorValue( aScope, aValue.mArray );
            },
            []( Scope &aScope, U32Array const &aValue ) {
                return VectorValue( aScope, aValue.mArray );
            },
            []( Scope &aScope, U64Array const &aValue ) {
                return VectorValue( aScope, aValue.mArray );
            },
            []( Scope &aScope, I8Array const &aValue ) {
                return VectorValue( aScope, aValue.mArray );
            },
            []( Scope &aScope, I16Array const &aValue ) {
                return VectorValue( aScope, aValue.mArray );
            },
            []( Scope &aScope, I32Array const &aValue ) {
                return VectorValue( aScope, aValue.mArray );
            },
            []( Scope &aScope, I64Array const &aValue ) {
                return VectorValue( aScope, aValue.mArray );
            },
            []( Scope &aScope, TextureSamplerArray const &aValue ) {
                return VectorValue( aScope, aValue.mArray );
            },
            []( Scope &aScope, CudaTextureSamplerArray const &aValue ) {
                return VectorValue( aScope, aValue.mArray );
            },
            []( Scope &aScope, std::vector<float> const &aValue ) {
                return VectorValue( aScope, aValue );
            },
            []( Scope &aScope, std::vector<double> const &aValue ) {
                return VectorValue( aScope, aValue );
            },
            []( Scope &aScope, std::vector<uint8_t> const &aValue ) {
                return VectorValue( aScope, aValue );
            },
            []( Scope &aScope, std::vector<uint16_t> const &aValue ) {
                return VectorValue( aScope, aValue );
            },
            []( Scope &aScope, std::vector<uint32_t> const &aValue ) {
                return VectorValue( aScope, aValue );
            },
            []( Scope &aScope, std::vector<uint64_t> const &aValue ) {
                return VectorValue( aScope, aValue );
            },
            []( Scope &aScope, std::vector<int8_t> const &aValue ) {
                return VectorValue( aScope, aValue );
            },
            []( Scope &aScope, std::vector<int16_t> const &aValue ) {
                return VectorValue( aScope, aValue );
            },
            []( Scope &aScope, std::vector<int32_t> const &aValue ) {
                return VectorValue( aScope, aValue );
            },
            []( Scope &aScope, std::vector<int64_t> const &aValue ) {
                return VectorValue( aScope, aValue );
            }
        );
        // clang-format on

        // clang-format off
        lOpsModule["ConstantScalarValue"] = []( Scope &aScope, eScalarType aType, double aValue ) {
            switch(aType)
            {
            case eScalarType::FLOAT32:
                return ConstantScalarValue( aScope, static_cast<float>(aValue) );
            case eScalarType::FLOAT64:
                return ConstantScalarValue( aScope, static_cast<double>(aValue) );
            case eScalarType::UINT8:
                return ConstantScalarValue( aScope, static_cast<uint8_t>(aValue) );
            case eScalarType::UINT16:
                return ConstantScalarValue( aScope, static_cast<uint16_t>(aValue) );
            case eScalarType::UINT32:
                return ConstantScalarValue( aScope, static_cast<uint32_t>(aValue) );
            case eScalarType::UINT64:
                return ConstantScalarValue( aScope, static_cast<uint64_t>(aValue) );
            case eScalarType::INT8:
                return ConstantScalarValue( aScope, static_cast<int8_t>(aValue) );
            case eScalarType::INT16:
                return ConstantScalarValue( aScope, static_cast<int16_t>(aValue) );
            case eScalarType::INT32:
                return ConstantScalarValue( aScope, static_cast<int32_t>(aValue) );
            case eScalarType::INT64:
                return ConstantScalarValue( aScope, static_cast<int64_t>(aValue) );
            case eScalarType::UNKNOWN:
                break;
            }
        };
        // clang-format on

        // clang-format off
        lOpsModule["MultiTensorValue"] = overload(
            []( Scope &aScope, sConstantValueInitializerComponent const &aInitializer, Cuda::sTensorShape const &aShape ) {
                return MultiTensorValue( aScope, aInitializer, aShape );
            },
            []( Scope &aScope, sVectorInitializerComponent const &aInitializer, Cuda::sTensorShape const &aShape ) {
                return MultiTensorValue( aScope, aInitializer, aShape );
            },
            []( Scope &aScope, sDataInitializerComponent const &aInitializer, Cuda::sTensorShape const &aShape ) {
                return MultiTensorValue( aScope, aInitializer, aShape );
            },
            []( Scope &aScope, sRandomUniformInitializerComponent const &aInitializer, Cuda::sTensorShape const &aShape ) {
                return MultiTensorValue( aScope, aInitializer, aShape );
            },
            []( Scope &aScope, sRandomNormalInitializerComponent const &aInitializer, Cuda::sTensorShape const &aShape ) {
                return MultiTensorValue( aScope, aInitializer, aShape );
            }
        );
        // clang-format on

        lOpsModule["Add"]      = TensorOps::Add;
        lOpsModule["Subtract"] = TensorOps::Subtract;
        lOpsModule["Divide"]   = TensorOps::Divide;
        lOpsModule["Multiply"] = TensorOps::Multiply;
        lOpsModule["Floor"]    = TensorOps::Floor;
        lOpsModule["Ceil"]     = TensorOps::Ceil;
        lOpsModule["Abs"]      = TensorOps::Abs;
        lOpsModule["Sqrt"]     = TensorOps::Sqrt;
        lOpsModule["Round"]    = TensorOps::Round;
        lOpsModule["Diff"]     = TensorOps::Diff;
        lOpsModule["Shift"]    = TensorOps::Shift;

        lOpsModule["And"] = TensorOps::And;
        lOpsModule["Or"]  = TensorOps::Or;
        lOpsModule["Not"] = TensorOps::Not;

        lOpsModule["BitwiseAnd"] = TensorOps::BitwiseAnd;
        lOpsModule["BitwiseOr"]  = TensorOps::BitwiseOr;
        lOpsModule["BitwiseNot"] = TensorOps::BitwiseNot;

        lOpsModule["InInterval"] = TensorOps::InInterval;

        lOpsModule["Equal"]              = TensorOps::Equal;
        lOpsModule["LessThan"]           = TensorOps::LessThan;
        lOpsModule["LessThanOrEqual"]    = TensorOps::LessThanOrEqual;
        lOpsModule["GreaterThan"]        = TensorOps::GreaterThan;
        lOpsModule["GreaterThanOrEqual"] = TensorOps::GreaterThanOrEqual;

        lOpsModule["Where"] = TensorOps::Where;

        lOpsModule["Mix"]             = TensorOps::Mix;
        lOpsModule["AffineTransform"] = TensorOps::AffineTransform;

        lOpsModule["ARange"]      = TensorOps::ARange;
        lOpsModule["LinearSpace"] = TensorOps::LinearSpace;
        lOpsModule["Repeat"]      = TensorOps::Repeat;
        lOpsModule["Tile"]        = TensorOps::Tile;

        lOpsModule["Sample2D"]     = TensorOps::Sample2D;
        lOpsModule["ToFixedPoint"] = TensorOps::ToFixedPoint;

        lOpsModule["Collapse"] = TensorOps::Collapse;
        lOpsModule["Expand"]   = TensorOps::Expand;
        lOpsModule["Reshape"]  = TensorOps::Reshape;
        lOpsModule["Relayout"] = TensorOps::Relayout;
        lOpsModule["Flatten"]  = TensorOps::Flatten;
        lOpsModule["Slice"]    = TensorOps::Slice;
        lOpsModule["HCat"]     = TensorOps::HCat;

        lOpsModule["Summation"] = overload( []( Scope &aScope, OpNode const &aArray ) { return Summation( aScope, aArray ); },
                                            []( Scope &aScope, OpNode const &aArray, OpNode const &aBegin, OpNode const &aEnd )
                                            { return Summation( aScope, aArray, aBegin, aEnd ); } );

        lOpsModule["CountTrue"]    = TensorOps::CountTrue;
        lOpsModule["CountNonZero"] = TensorOps::CountNonZero;
        lOpsModule["CountZero"]    = TensorOps::CountZero;

        lOpsModule["Conv1D"] = TensorOps::Conv1D;
    }
}; // namespace SE::Core