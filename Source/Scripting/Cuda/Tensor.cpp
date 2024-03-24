#include "Tensor.h"

#include "Core/CUDA/Array/MultiTensor.h"
#include "Core/Definitions.h"

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
                []( Cuda::multi_tensor_t &self, numeric_array_t<_Ty> &values )
                {
                    self.Upload( values.mArray );

                    return self;
                },
                []( Cuda::multi_tensor_t &self, vector_t<_Ty> &values )
                {
                    self.Upload( values );

                    return self;
                },
                []( Cuda::multi_tensor_t &self, numeric_array_t<_Ty> &values, uint32_t layer )
                {
                    self.Upload( values.mArray, layer );

                    return self;
                },
                []( Cuda::multi_tensor_t &self, vector_t<_Ty> &values, uint32_t layer )
                {
                    self.Upload( values, layer );

                    return self;
                } );
        }

        template <typename _Ty>
        auto MakeFetchFunction()
        {
            return overload( []( Cuda::multi_tensor_t &self ) { return self.FetchFlattened<_Ty>(); },
                             []( Cuda::multi_tensor_t &self, uint32_t layer ) { return self.FetchBufferAt<_Ty>( layer ); } );
        }
    } // namespace

    void open_tensor_library( sol::table &scriptingState )
    {
        auto tensorShapeType = scriptingState.new_usertype<Cuda::tensor_shape_t>( "TensorShape" );

        // clang-format off
        tensorShapeType[call_constructor] = factories(
            []( sol::table initializer, int32_t elementSize )
            {
                vector_t<vector_t<uint32_t>> shape{};

                for (uint32_t i=0; i < initializer.size(); i++)
                {
                    auto layer = initializer.get<sol::table>( i + 1 );
                    shape.push_back( vector_t<uint32_t>{} );

                    for (uint32_t j=0; j < layer.size(); j++)
                    {
                        auto dim = layer.get<uint32_t>( j + 1 );

                        shape.back().push_back( dim );
                    }
                }

                return Cuda::tensor_shape_t( shape, elementSize );
            } );
        // clang-format on

        tensorShapeType["get_dimension"] = []( Cuda::tensor_shape_t &self, int32_t i ) { return self.GetDimension( i ); };
        tensorShapeType["trim"]          = []( Cuda::tensor_shape_t &self, int32_t i ) { self.Trim( i ); };
        tensorShapeType["flatten"]       = []( Cuda::tensor_shape_t &self, int32_t i ) { self.Flatten( i ); };

        auto memoryPoolType = scriptingState.new_usertype<Cuda::memory_pool_t>(
            "MemoryPool", constructors<Cuda::memory_pool_t( uint32_t aMemorySize )>() );
        memoryPoolType["reset"]    = []( Cuda::memory_pool_t &self ) { self.Reset(); };
        memoryPoolType["allocate"] = []( Cuda::memory_pool_t &self, int32_t bytes ) { return self.Allocate( bytes ); };

        auto multiTensorType = scriptingState.new_usertype<Cuda::multi_tensor_t>(
            "MultiTensor",
            constructors<Cuda::multi_tensor_t( Cuda::memory_pool_t & aMemoryPool, const Cuda::tensor_shape_t &shape )>() );
        multiTensorType["size"]    = []( Cuda::multi_tensor_t &self ) { return self.Size(); };
        multiTensorType["size_as"] = []( Cuda::multi_tensor_t &self, const sol::object &aTypeOrID )
        {
            const auto lMaybeAny = invoke_meta_function( deduce_type( aTypeOrID ), "SizeAs"_hs, self );

            return lMaybeAny ? lMaybeAny.cast<size_t>() : 0;
        };

        multiTensorType["upload_u8"]  = MakeUploadFunction<uint8_t>();
        multiTensorType["upload_u16"] = MakeUploadFunction<uint16_t>();
        multiTensorType["upload_u32"] = MakeUploadFunction<uint32_t>();
        multiTensorType["upload_u64"] = MakeUploadFunction<uint32_t>();

        multiTensorType["upload_i8"]  = MakeUploadFunction<int8_t>();
        multiTensorType["upload_i16"] = MakeUploadFunction<int16_t>();
        multiTensorType["upload_i32"] = MakeUploadFunction<int32_t>();
        multiTensorType["upload_i64"] = MakeUploadFunction<int32_t>();

        multiTensorType["upload_f32"] = MakeUploadFunction<float>();
        multiTensorType["upload_f64"] = MakeUploadFunction<double>();

        multiTensorType["upload_uvec2"] = MakeUploadFunction<math::uvec2>();
        multiTensorType["upload_uvec3"] = MakeUploadFunction<math::uvec3>();
        multiTensorType["upload_uvec4"] = MakeUploadFunction<math::uvec4>();

        multiTensorType["upload_ivec2"] = MakeUploadFunction<math::ivec2>();
        multiTensorType["upload_ivec3"] = MakeUploadFunction<math::ivec3>();
        multiTensorType["upload_ivec4"] = MakeUploadFunction<math::ivec4>();

        multiTensorType["upload_vec2"] = MakeUploadFunction<math::vec2>();
        multiTensorType["upload_vec3"] = MakeUploadFunction<math::vec3>();
        multiTensorType["upload_vec4"] = MakeUploadFunction<math::vec4>();

        multiTensorType["upload_mat3"] = MakeUploadFunction<math::mat3>();
        multiTensorType["upload_mat4"] = MakeUploadFunction<math::mat4>();

        multiTensorType["fetch_u8"]  = MakeFetchFunction<uint8_t>();
        multiTensorType["fetch_u16"] = MakeFetchFunction<uint16_t>();
        multiTensorType["fetch_u32"] = MakeFetchFunction<uint32_t>();
        multiTensorType["fetch_u64"] = MakeFetchFunction<uint32_t>();

        multiTensorType["fetch_i8"]  = MakeFetchFunction<int8_t>();
        multiTensorType["fetch_i16"] = MakeFetchFunction<int16_t>();
        multiTensorType["fetch_i32"] = MakeFetchFunction<int32_t>();
        multiTensorType["fetch_i64"] = MakeFetchFunction<int32_t>();

        multiTensorType["fetch_f32"] = MakeFetchFunction<float>();
        multiTensorType["fetch_f64"] = MakeFetchFunction<double>();

        multiTensorType["fetch_uvec2"] = MakeFetchFunction<math::uvec2>();
        multiTensorType["fetch_uvec3"] = MakeFetchFunction<math::uvec3>();
        multiTensorType["fetch_uvec4"] = MakeFetchFunction<math::uvec4>();

        multiTensorType["fetch_ivec2"] = MakeFetchFunction<math::ivec2>();
        multiTensorType["fetch_ivec3"] = MakeFetchFunction<math::ivec3>();
        multiTensorType["fetch_ivec4"] = MakeFetchFunction<math::ivec4>();

        multiTensorType["fetch_vec2"] = MakeFetchFunction<math::vec2>();
        multiTensorType["fetch_vec3"] = MakeFetchFunction<math::vec3>();
        multiTensorType["fetch_vec4"] = MakeFetchFunction<math::vec4>();

        multiTensorType["fetch_mat3"] = MakeFetchFunction<math::mat3>();
        multiTensorType["fetch_mat4"] = MakeFetchFunction<math::mat4>();

        auto scopeType =
            scriptingState.new_usertype<TensorOps::scope_t>( "Scope", constructors<TensorOps::scope_t( uint32_t aMemorySize )>() );
        scopeType["reset"] = []( TensorOps::scope_t &self ) { self.Reset(); };

        // clang-format off
        scopeType["run"] = overload(
            []( TensorOps::scope_t &self, TensorOps::graph_node_t &aNode ) { self.Run( aNode ); },
            []( TensorOps::scope_t &self, vector_t<TensorOps::graph_node_t> aNode ) { self.Run( aNode ); },
            []( TensorOps::scope_t &self, sol::table aNode )
            {
                vector_t<TensorOps::graph_node_t> lOpNodes{};

                for (uint32_t i=0; i < aNode.size(); i++)
                {
                    auto lNode = aNode.get<TensorOps::graph_node_t>( i + 1 );
                    lOpNodes.push_back( lNode );
                }

                self.Run( lOpNodes );
            }
        );
        // clang-format on

        auto opsModule = scriptingState["Ops"].get_or_create<sol::table>();

        // clang-format off
        opsModule.new_enum( "eScalarType",
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

        declare_primitive_type<multi_tensor_value_t>( opsModule, "sMultiTensorComponent" );

        // clang-format off
        auto lConstantInitializerComponent = opsModule.new_usertype<constant_value_initializer_t>("sConstantValueInitializerComponent");
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

        auto lVectorInitializerComponent = opsModule.new_usertype<vector_initializer_t>( "sVectorInitializerComponent" );

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

        auto lDataInitializerComponent = opsModule.new_usertype<data_initializer_t>( "sDataInitializerComponent" );
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

        auto lRandomUniformInitializerComponent =
            opsModule.new_usertype<random_uniform_initializer_t>( "sRandomUniformInitializerComponent" );
        lRandomUniformInitializerComponent[call_constructor] = []( scalar_type_t value )
        { return random_uniform_initializer_t{ value }; };

        auto lRandomNormalInitializerComponent =
            opsModule.new_usertype<random_normal_initializer_t>( "sRandomNormalInitializerComponent" );
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
        opsModule["MultiTensorValue"] = overload(
            []( scope_t &scope, constant_value_initializer_t const &initializer, Cuda::tensor_shape_t const &shape ) {
                return MultiTensorValue( scope, initializer, shape );
            },
            []( scope_t &scope, vector_initializer_t const &initializer, Cuda::tensor_shape_t const &shape ) {
                return MultiTensorValue( scope, initializer, shape );
            },
            []( scope_t &scope, data_initializer_t const &initializer, Cuda::tensor_shape_t const &shape ) {
                return MultiTensorValue( scope, initializer, shape );
            },
            []( scope_t &scope, random_uniform_initializer_t const &initializer, Cuda::tensor_shape_t const &shape ) {
                return MultiTensorValue( scope, initializer, shape );
            },
            []( scope_t &scope, random_normal_initializer_t const &initializer, Cuda::tensor_shape_t const &shape ) {
                return MultiTensorValue( scope, initializer, shape );
            }
        );
        // clang-format on

        opsModule["Add"]      = TensorOps::Add;
        opsModule["Subtract"] = TensorOps::Subtract;
        opsModule["Divide"]   = TensorOps::Divide;
        opsModule["Multiply"] = TensorOps::Multiply;
        opsModule["Floor"]    = TensorOps::Floor;
        opsModule["Ceil"]     = TensorOps::Ceil;
        opsModule["Abs"]      = TensorOps::Abs;
        opsModule["Sqrt"]     = TensorOps::Sqrt;
        opsModule["Round"]    = TensorOps::Round;
        opsModule["Diff"]     = TensorOps::Diff;
        opsModule["Shift"]    = TensorOps::Shift;

        opsModule["And"] = TensorOps::And;
        opsModule["Or"]  = TensorOps::Or;
        opsModule["Not"] = TensorOps::Not;

        opsModule["BitwiseAnd"] = TensorOps::BitwiseAnd;
        opsModule["BitwiseOr"]  = TensorOps::BitwiseOr;
        opsModule["BitwiseNot"] = TensorOps::BitwiseNot;

        opsModule["InInterval"] = TensorOps::InInterval;

        opsModule["Equal"]              = TensorOps::Equal;
        opsModule["LessThan"]           = TensorOps::LessThan;
        opsModule["LessThanOrEqual"]    = TensorOps::LessThanOrEqual;
        opsModule["GreaterThan"]        = TensorOps::GreaterThan;
        opsModule["GreaterThanOrEqual"] = TensorOps::GreaterThanOrEqual;

        opsModule["Where"] = TensorOps::Where;

        opsModule["Mix"]    = TensorOps::Mix;
        opsModule["Affine"] = TensorOps::AffineTransform;

        opsModule["ARange"]      = TensorOps::ARange;
        opsModule["LinearSpace"] = TensorOps::LinearSpace;
        opsModule["Repeat"]      = TensorOps::Repeat;
        opsModule["Tile"]        = TensorOps::Tile;

        opsModule["Sample2D"]     = TensorOps::Sample2D;
        opsModule["ToFixedPoint"] = TensorOps::ToFixedPoint;

        opsModule["Collapse"] = TensorOps::Collapse;
        opsModule["Expand"]   = TensorOps::Expand;
        opsModule["Reshape"]  = TensorOps::Reshape;
        opsModule["Relayout"] = TensorOps::Relayout;
        opsModule["Flatten"]  = TensorOps::Flatten;
        opsModule["Slice"]    = TensorOps::Slice;
        opsModule["HCat"]     = TensorOps::HCat;

        opsModule["Summation"] = overload( []( scope_t &scope, graph_node_t const &array ) { return Summation( scope, array ); },
                                            []( scope_t &scope, graph_node_t const &array, graph_node_t const &aBegin,
                                                graph_node_t const &aEnd ) { return Summation( scope, array, aBegin, aEnd ); } );

        opsModule["CountTrue"]    = TensorOps::CountTrue;
        opsModule["CountNonZero"] = TensorOps::CountNonZero;
        opsModule["CountZero"]    = TensorOps::CountZero;

        opsModule["Conv1D"] = TensorOps::Conv1D;
    }
}; // namespace SE::Core