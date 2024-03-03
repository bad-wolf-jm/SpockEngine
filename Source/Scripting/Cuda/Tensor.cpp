#include "Tensor.h"

#include "Core/Definitions.h"
#include "Core/CUDA/Array/MultiTensor.h"

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
        template <typename _Ty> auto MakeUploadFunction()
        {
            return overload(
                []( Cuda::multi_tensor_t &aSelf, numeric_array_t<_Ty> &aValues )
                {
                    aSelf.Upload( aValues.mArray );

                    return aSelf;
                },
                []( Cuda::multi_tensor_t &aSelf, vector_t<_Ty> &aValues )
                {
                    aSelf.Upload( aValues );

                    return aSelf;
                },
                []( Cuda::multi_tensor_t &aSelf, numeric_array_t<_Ty> &aValues, uint32_t aLayer )
                {
                    aSelf.Upload( aValues.mArray, aLayer );

                    return aSelf;
                },
                []( Cuda::multi_tensor_t &aSelf, vector_t<_Ty> &aValues, uint32_t aLayer )
                {
                    aSelf.Upload( aValues, aLayer );

                    return aSelf;
                } );
        }

        template <typename _Ty> auto MakeFetchFunction()
        {
            return overload( []( Cuda::multi_tensor_t &aSelf ) { return aSelf.FetchFlattened<_Ty>(); },
                             []( Cuda::multi_tensor_t &aSelf, uint32_t aLayer ) { return aSelf.FetchBufferAt<_Ty>( aLayer ); } );
        }
    } // namespace

    void OpenTensorLibrary( sol::table &aScriptingState )
    {
        auto lTensorShapeType = aScriptingState.new_usertype<Cuda::tensor_shape_t>( "TensorShape" );

        // clang-format off
        lTensorShapeType[call_constructor] = factories(
            []( sol::table aInitializer, int32_t aElementSize )
            {
                vector_t<vector_t<uint32_t>> lShape{};

                for (uint32_t i=0; i < aInitializer.size(); i++)
                {
                    auto lLayer = aInitializer.get<sol::table>( i + 1 );
                    lShape.push_back( vector_t<uint32_t>{} );

                    for (uint32_t j=0; j < lLayer.size(); j++)
                    {
                        auto lDim = lLayer.get<uint32_t>( j + 1 );

                        lShape.back().push_back( lDim );
                    }
                }

                return Cuda::tensor_shape_t( lShape, aElementSize );
            } );
        // clang-format on

        lTensorShapeType["get_dimension"] = []( Cuda::tensor_shape_t &aSelf, int32_t i ) { return aSelf.GetDimension( i ); };
        lTensorShapeType["trim"]          = []( Cuda::tensor_shape_t &aSelf, int32_t i ) { aSelf.Trim( i ); };
        lTensorShapeType["flatten"]       = []( Cuda::tensor_shape_t &aSelf, int32_t i ) { aSelf.Flatten( i ); };

        auto lMemoryPoolType        = aScriptingState.new_usertype<Cuda::memory_pool_t>( "MemoryPool", constructors<Cuda::memory_pool_t( uint32_t aMemorySize )>() );
        lMemoryPoolType["reset"]    = []( Cuda::memory_pool_t &aSelf ) { aSelf.Reset(); };
        lMemoryPoolType["allocate"] = []( Cuda::memory_pool_t &aSelf, int32_t aBytes ) { return aSelf.Allocate( aBytes ); };

        auto lMultiTensorType =
            aScriptingState.new_usertype<Cuda::multi_tensor_t>( "MultiTensor", constructors<Cuda::multi_tensor_t( Cuda::memory_pool_t & aMemoryPool, const Cuda::tensor_shape_t &aShape )>() );
        lMultiTensorType["size"]    = []( Cuda::multi_tensor_t &aSelf ) { return aSelf.Size(); };
        lMultiTensorType["size_as"] = []( Cuda::multi_tensor_t &aSelf, const sol::object &aTypeOrID )
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

        auto lScopeType     = aScriptingState.new_usertype<TensorOps::scope_t>( "Scope", constructors<TensorOps::scope_t( uint32_t aMemorySize )>() );
        lScopeType["reset"] = []( TensorOps::scope_t &aSelf ) { aSelf.Reset(); };

        // clang-format off
        lScopeType["run"] = overload(
            []( TensorOps::scope_t &aSelf, TensorOps::graph_node_t &aNode ) { aSelf.Run( aNode ); },
            []( TensorOps::scope_t &aSelf, vector_t<TensorOps::graph_node_t> aNode ) { aSelf.Run( aNode ); },
            []( TensorOps::scope_t &aSelf, sol::table aNode )
            {
                vector_t<TensorOps::graph_node_t> lOpNodes{};

                for (uint32_t i=0; i < aNode.size(); i++)
                {
                    auto lNode = aNode.get<TensorOps::graph_node_t>( i + 1 );
                    lOpNodes.push_back( lNode );
                }

                aSelf.Run( lOpNodes );
            }
        );
        // clang-format on

        auto lOpsModule = aScriptingState["Ops"].get_or_create<sol::table>();

        // clang-format off
        lOpsModule.new_enum( "eScalarType",
            "FLOAT32", scalar_type_t::FLOAT32,
            "FLOAT64", scalar_type_t::FLOAT64,
            "UINT8",   scalar_type_t::UINT8,
            "UINT16",  scalar_type_t::UINT16,
            "UINT32",  scalar_type_t::UINT32,
            "UINT64",  scalar_type_t::UINT64,
            "INT8",    scalar_type_t::INT8,
            "INT16",   scalar_type_t::INT16,
            "INT32",   scalar_type_t::INT32,
            "INT64",   scalar_type_t::INT64,
            "UNKNOWN", scalar_type_t::UNKNOWN  );
        // clang-format on

        DeclarePrimitiveType<multi_tensor_value_t>( lOpsModule, "sMultiTensorComponent" );

        // clang-format off
        auto lConstantInitializerComponent = lOpsModule.new_usertype<constant_value_initializer_t>("sConstantValueInitializerComponent");
        lConstantInitializerComponent[call_constructor] = [](scalar_type_t aType, double value)
        {
            switch(aType)
            {
            case scalar_type_t::FLOAT32:
                return constant_value_initializer_t{ static_cast<float>(value) };
            case scalar_type_t::FLOAT64:
                return constant_value_initializer_t{ static_cast<double>(value) };
            case scalar_type_t::UINT8:
                return constant_value_initializer_t{ static_cast<uint8_t>(value) };
            case scalar_type_t::UINT16:
                return constant_value_initializer_t{ static_cast<uint16_t>(value) };
            case scalar_type_t::UINT32:
                return constant_value_initializer_t{ static_cast<uint32_t>(value) };
            case scalar_type_t::UINT64:
                return constant_value_initializer_t{ static_cast<uint64_t>(value) };
            case scalar_type_t::INT8:
                return constant_value_initializer_t{ static_cast<int8_t>(value) };
            case scalar_type_t::INT16:
                return constant_value_initializer_t{ static_cast<int16_t>(value) };
            case scalar_type_t::INT32:
                return constant_value_initializer_t{ static_cast<int32_t>(value) };
            case scalar_type_t::INT64:
                return constant_value_initializer_t{ static_cast<int64_t>(value) };
            case scalar_type_t::UNKNOWN:
            default:
                break;
            }
        };

        auto lVectorInitializerComponent = lOpsModule.new_usertype<vector_initializer_t>( "sVectorInitializerComponent" );

        // clang-format off
        lVectorInitializerComponent[call_constructor] = factories(
            []( vector_t<float> value)    { return vector_initializer_t{ value }; },
            []( vector_t<double> value)   { return vector_initializer_t{ value }; },
            []( vector_t<uint8_t> value)  { return vector_initializer_t{ value }; },
            []( vector_t<uint16_t> value) { return vector_initializer_t{ value }; },
            []( vector_t<uint32_t> value) { return vector_initializer_t{ value }; },
            []( vector_t<uint64_t> value) { return vector_initializer_t{ value }; },
            []( vector_t<int8_t>  value)  { return vector_initializer_t{ value }; },
            []( vector_t<int16_t> value)  { return vector_initializer_t{ value }; },
            []( vector_t<int32_t> value)  { return vector_initializer_t{ value }; },
            []( vector_t<int64_t> value)  { return vector_initializer_t{ value }; }
        );
        // clang-format on

        auto lDataInitializerComponent = lOpsModule.new_usertype<data_initializer_t>( "sDataInitializerComponent" );
        // clang-format off
        lDataInitializerComponent[call_constructor] = factories(
            []( vector_t<float> value)    { return data_initializer_t{ value }; },
            []( vector_t<double> value)   { return data_initializer_t{ value }; },
            []( vector_t<uint8_t> value)  { return data_initializer_t{ value }; },
            []( vector_t<uint16_t> value) { return data_initializer_t{ value }; },
            []( vector_t<uint32_t> value) { return data_initializer_t{ value }; },
            []( vector_t<uint64_t> value) { return data_initializer_t{ value }; },
            []( vector_t<int8_t>  value)  { return data_initializer_t{ value }; },
            []( vector_t<int16_t> value)  { return data_initializer_t{ value }; },
            []( vector_t<int32_t> value)  { return data_initializer_t{ value }; },
            []( vector_t<int64_t> value)  { return data_initializer_t{ value }; }
        );
        // clang-format on

        auto lRandomUniformInitializerComponent              = lOpsModule.new_usertype<random_uniform_initializer_t>( "sRandomUniformInitializerComponent" );
        lRandomUniformInitializerComponent[call_constructor] = []( scalar_type_t value ) { return random_uniform_initializer_t{ value }; };

        auto lRandomNormalInitializerComponent              = lOpsModule.new_usertype<random_normal_initializer_t>( "sRandomNormalInitializerComponent" );
        lRandomNormalInitializerComponent[call_constructor] = []( scalar_type_t value, double mean, double std )
        {
            switch( value )
            {
            case scalar_type_t::FLOAT64:
                return random_normal_initializer_t{ value, mean, std };
            default:
                return random_normal_initializer_t{ value, static_cast<float>( mean ), static_cast<float>( std ) };
            }
        };

        // clang-format off
        lOpsModule["MultiTensorValue"] = overload(
            []( scope_t &aScope, constant_value_initializer_t const &aInitializer, Cuda::tensor_shape_t const &aShape ) {
                return MultiTensorValue( aScope, aInitializer, aShape );
            },
            []( scope_t &aScope, vector_initializer_t const &aInitializer, Cuda::tensor_shape_t const &aShape ) {
                return MultiTensorValue( aScope, aInitializer, aShape );
            },
            []( scope_t &aScope, data_initializer_t const &aInitializer, Cuda::tensor_shape_t const &aShape ) {
                return MultiTensorValue( aScope, aInitializer, aShape );
            },
            []( scope_t &aScope, random_uniform_initializer_t const &aInitializer, Cuda::tensor_shape_t const &aShape ) {
                return MultiTensorValue( aScope, aInitializer, aShape );
            },
            []( scope_t &aScope, random_normal_initializer_t const &aInitializer, Cuda::tensor_shape_t const &aShape ) {
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

        lOpsModule["Mix"]    = TensorOps::Mix;
        lOpsModule["Affine"] = TensorOps::AffineTransform;

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

        lOpsModule["Summation"] =
            overload( []( scope_t &aScope, graph_node_t const &aArray ) { return Summation( aScope, aArray ); },
                      []( scope_t &aScope, graph_node_t const &aArray, graph_node_t const &aBegin, graph_node_t const &aEnd ) { return Summation( aScope, aArray, aBegin, aEnd ); } );

        lOpsModule["CountTrue"]    = TensorOps::CountTrue;
        lOpsModule["CountNonZero"] = TensorOps::CountNonZero;
        lOpsModule["CountZero"]    = TensorOps::CountZero;

        lOpsModule["Conv1D"] = TensorOps::Conv1D;
    }
}; // namespace SE::Core