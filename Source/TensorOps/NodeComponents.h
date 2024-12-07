/// @file   NodeComponents.h
///
/// @brief  Component definitions for computation graph nodes
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert. All rights reserved.

#pragma once
#include <variant>

#include "Core/Definitions.h"
#include "Core/Logging.h"

#include "Core/CUDA/Array/MemoryPool.h"
#include "Core/CUDA/Array/MultiTensor.h"
#include "Core/Entity/Collection.h"

namespace numlua::mtops
{

    using namespace numlua::core;

    using multi_tensor_t  = numlua::cuda::multi_tensor_t;
    using memory_buffer_t = numlua::cuda::memory_buffer_t;
    using memory_pool_t   = numlua::cuda::memory_pool_t;
    using tensor_shape_t  = numlua::cuda::tensor_shape_t;

    using graph_node_t = numlua::core::entity_t;

    struct node_id_t
    {
        bool          do_not_expand = false;
        bool          is_allocated  = false;
        scalar_type_t element_type  = scalar_type_t::FLOAT32;
    };

    inline scalar_type_t &type_of( graph_node_t node )
    {
        return node.Get<node_id_t>().element_type;
    }

    /// @brief sGraphOperationController
    ///
    /// Base class for graph operation controller
    ///
    class graph_operation_controller_t
    {
      public:
        virtual ~graph_operation_controller_t() = default;

        template <typename T>
        T &Get()
        {
            return _node.Get<T>();
        }

        template <typename T>
        bool Has()
        {
            return _node.Has<T>();
        }

        graph_node_t GetControlledEntity()
        {
            return _node;
        };

      public:
        virtual void Initialize( graph_node_t aEntity )
        {
            _node = aEntity;
        }

        virtual void Run()
        {
        }

      protected:
        graph_node_t _node;
    };

    /// @brief sGraphOperationComponent
    ///
    /// This component marks the entity for execution by the tensor graph running system. In order
    /// for it wo do anything, it must be bound to a controller which provides the actual implementation
    /// of the tensor function.
    ///
    struct graph_operation_t
    {
        graph_operation_controller_t *controller_instance = nullptr;

        std::function<graph_operation_controller_t *()> mInstantiateController;
        std::function<void( graph_operation_t * )>      mDestroyController;

        template <typename T, typename... Args>
        void Bind( Args &&...args )
        {
            mInstantiateController = []() { return reinterpret_cast<graph_operation_controller_t *>( new T() ); };
            mDestroyController     = []( graph_operation_t *nsc )
            {
                delete nsc->controller_instance;
                nsc->controller_instance = nullptr;
            };
        }
    };

    /// @brief sVectorComponent
    ///
    /// Upload a vector of values to the GPU upon running
    ///
    template <typename _Ty>
    struct vector_value_t
    {
        vector_t<_Ty> value = {}; //!< Values to upload

        vector_value_t()                         = default;
        vector_value_t( const vector_value_t & ) = default;

        size_t size()
        {
            return value.size();
        }
    };

    struct vector_buffer_t
    {
        size_t          size = 0; //!< Size hint for the underlying memory buffer
        memory_buffer_t value{};  //!< GPU area where the values should be uploaded. This is typically allocated from a pool

        vector_buffer_t()                          = default;
        vector_buffer_t( const vector_buffer_t & ) = default;
    };

    using u32_vector_t          = vector_value_t<uint32_t>;
    using f32_vector_t          = vector_value_t<float>;
    using scalar_value_vector_t = vector_value_t<scalar_value_t>;

    /// @brief sOperandComponent
    ///
    /// When added to an entity, this component holds a list of all the dependencies of said entity.  When running
    /// the graph, the elements listed in this component will be run first, then the current node will be run.
    ///
    struct operand_t
    {
        vector_t<graph_node_t> mOperands = {}; //!< List of entities that have to be run before the current entity.

        operand_t() = default;
        operand_t( vector_t<graph_node_t> const &aOpNodes )
        {
            mOperands = aOpNodes;
        };
        operand_t( const operand_t & ) = default;
    };

    /// @brief sARangeComponent
    ///
    /// When added to an entity, this component indicates that said entity should compute a range of values
    /// beween the values in `aLeft` and the values in `mRight` with a step of size `mDelta`.  All three of
    /// mLeft, mRight and Delta should be vectors of the same length
    ///
    struct arange_operation_t
    {
        graph_node_t left{};  //!< Lower bound. Should be a vector.
        graph_node_t right{}; //!< Upper bound. Should be a vector.
        graph_node_t delta{}; //!< Difference. Should be a vector.

        arange_operation_t()                             = default;
        arange_operation_t( const arange_operation_t & ) = default;
    };

    /// @brief sBinaryOperationComponent
    ///
    /// When added to an entity, this component indicates that the entity represents a generic binary
    /// operation. The actual operation to perform depends on the operation controller.
    ///
    struct binary_operation_t
    {
        graph_node_t left;  //!< Left operand
        graph_node_t right; //!< Right operand

        binary_operation_t()                             = default;
        binary_operation_t( const binary_operation_t & ) = default;
    };

    struct broadcast_info_t
    {
        broadcast_hint_t mBroadcastHint = broadcast_hint_t::NONE; //!< How to broadcast the operation.

        graph_node_t block_sizes;                //!< Block sizes
        uint32_t     max_block_size = 0;         //!< Maximum value of the `block_sizes` parameter
        graph_node_t mBroadcastDimension;        //!< Size of the broadcast dimension
        uint32_t     mMaxBroadcastDimension = 0; //!< Maximum size of the broadcast dimension

        broadcast_info_t()                           = default;
        broadcast_info_t( const broadcast_info_t & ) = default;
    };

    struct not_operation_t
    {
        graph_node_t operand; //!< Left operand

        not_operation_t()                          = default;
        not_operation_t( const not_operation_t & ) = default;
    };

    struct bitwise_not_operation_t
    {
        graph_node_t operand; //!< Left operand

        bitwise_not_operation_t()                                  = default;
        bitwise_not_operation_t( const bitwise_not_operation_t & ) = default;
    };

    struct in_interval_operation_t
    {
        graph_node_t x;            //!< Value to test
        graph_node_t lower;        //!< Interval lower bound
        graph_node_t upper;        //!< Interval upper bound
        bool         strict_lower; //!< Use strict inequality for lower bound
        bool         strict_upper; //!< Use strict inequality for upper bound

        in_interval_operation_t()                                  = default;
        in_interval_operation_t( const in_interval_operation_t & ) = default;
    };

    /// @brief sRepeatOperationComponent
    ///
    /// When added to an entity, this component indicates that the elements @ref MultiTensor contained in `array`
    /// should be repeated a number of times given by `mRepetitions`, which should contain a vector whose length
    /// should match the number of layers in the @ref MultiTensor contained in `array`.
    ///
    struct repeat_operation_t
    {
        graph_node_t operand;     //!< @ref MultiTensor to repeat.
        graph_node_t repetitions; //!< Number of times the elements of the @ref MultiTensor should be repeated

        repeat_operation_t()                             = default;
        repeat_operation_t( const repeat_operation_t & ) = default;
    };

    /// @brief sTileOperationComponent
    ///
    /// When added to an entity, this component indicates that the elements @ref MultiTensor contained in `array`
    /// should be tiled a number of times given by `mRepetitions`, which should contain a vector whose length
    /// should match the number of layers in the @ref MultiTensor contained in `array`.
    ///
    struct tile_operation_t
    {
        graph_node_t operand;     //!< @ref MultiTensor to repeat.
        graph_node_t repetitions; //!< Number of times the elements of the @ref MultiTensor should be tiled

        tile_operation_t()                           = default;
        tile_operation_t( const tile_operation_t & ) = default;
    };

    /// @brief sLinearSpaceComponent
    ///
    /// When added to an entity, this component indicates that the entity should compute a set of evenly spaced
    /// numbers between two bounds.
    ///
    struct linear_space_operation_t
    {
        graph_node_t left{};         //!< Lower bound
        graph_node_t right{};        //!< Upper bound
        graph_node_t subdivisions{}; //!< Number of subdivisions

        linear_space_operation_t()                                   = default;
        linear_space_operation_t( const linear_space_operation_t & ) = default;
    };

    struct where_operation_t
    {
        graph_node_t condition{};      //!< Lower bound
        graph_node_t value_if_true{};  //!< Upper bound
        graph_node_t value_if_false{}; //!< Number of subdivisions

        where_operation_t()                            = default;
        where_operation_t( const where_operation_t & ) = default;
    };

    /// @brief sMixNodeComponent
    ///
    /// When added to an entity, this component indicated that said entity should compute the linear mix of
    /// the @ref MultiTensors represented by `mA` and `mB` with coefficient `t`.
    ///
    struct mix_operation_t
    {
        graph_node_t A{}; //!< Left
        graph_node_t B{}; //!< Right
        graph_node_t t{}; //!< Coefficient

        mix_operation_t()                          = default;
        mix_operation_t( const mix_operation_t & ) = default;
    };

    /// @brief sScalarNodeComponent
    ///
    /// When added to an entity, this component marks said entity as containing a single scalar
    ///
    struct scalar_node_t
    {
        scalar_value_t value = 0.0f;

        scalar_node_t()                        = default;
        scalar_node_t( const scalar_node_t & ) = default;
    };

    /// @brief sConstantValueInitializerComponent
    ///
    /// When added to an entity, this component marks the entity as containing a constant initializer
    /// for a @ref MultiTensor.
    ///
    struct constant_value_initializer_t
    {
        scalar_value_t value = 0.0f; //!< Value to initialize the @ref MultiTensor with.

        constant_value_initializer_t() = default;

        template <typename _Ty>
        constant_value_initializer_t( _Ty const &aValue )
        {
            value = aValue;
        }

        constant_value_initializer_t( const constant_value_initializer_t & ) = default;
    };

    /// @brief sVectorInitializerComponent
    ///
    /// When added to an entity, this component marks the entity as containing a constant initializer
    /// for a @ref MultiTensor with a different value for each layer.
    ///
    struct vector_initializer_t
    {
        vector_t<scalar_value_t> value = {}; //!< Vector of values
        memory_buffer_t          data;       //!< GPU representation of the data of values

        vector_initializer_t() = default;

        template <typename _Ty>
        vector_initializer_t( vector_t<_Ty> const &values )
        {
            uint32_t size = values.size();
            value         = vector_t<scalar_value_t>( size );

            for( uint32_t i = 0; i < size; i++ )
            {
                value[i] = values[i];
            }
        }

        vector_initializer_t( const vector_initializer_t & ) = default;
    };

    /// @brief sDataInitializerComponent
    ///
    /// When added to an entity, this component marks the entity as containing a data initializer
    /// for a @ref MultiTensor containing data that should be copied to the MultiTensor upon
    /// initialization.
    ///

    struct data_initializer_t
    {
        vector_t<scalar_value_t> value = {}; //!< Vector of values

        data_initializer_t() = default;

        template <typename _Ty>
        data_initializer_t( vector_t<_Ty> const &values )
        {
            uint32_t size = values.size();
            value         = vector_t<scalar_value_t>( size );

            for( uint32_t i = 0; i < size; i++ )
            {
                value[i] = values[i];
            }
        }

        data_initializer_t( const data_initializer_t & ) = default;
    };

    /// @brief sRandomUniformInitializerComponent
    ///
    /// When added to an entity, this component indicates that the corresponding @ref MultiTensor
    /// should be filled with uniformly distributed random values
    ///
    struct random_uniform_initializer_t
    {
        scalar_type_t type = scalar_type_t::FLOAT32; //!< Type

        random_uniform_initializer_t()                                       = default;
        random_uniform_initializer_t( const random_uniform_initializer_t & ) = default;
    };

    /// @brief sRandomNormalInitializerComponent
    ///
    /// When added to an entity, this component indicates that the corresponding @ref MultiTensor
    /// should be filled with normally distributed distributed random values
    ///
    struct random_normal_initializer_t
    {
        scalar_type_t  type  = scalar_type_t::FLOAT32; //!< Type
        scalar_value_t mu    = 0.0f;                   //!< Expected value
        scalar_value_t sigma = 1.0f;                   //!< Standard deviation

        random_normal_initializer_t()                                      = default;
        random_normal_initializer_t( const random_normal_initializer_t & ) = default;
    };

    /// @brief sMultiTensorComponent
    ///
    /// When added to an entity, this component attached a @ref MultiTensor, which can be
    /// filled eith with an initializer, or be used as the output of a tensor operation.
    ///
    struct multi_tensor_value_t
    {
        multi_tensor_t value{}; //!< GPU buffer
        tensor_shape_t shape{}; //!< Shape of the multitensor

        multi_tensor_value_t() = default;
        multi_tensor_value_t( memory_pool_t &memory_pool, const tensor_shape_t &shape )
            : shape{ shape }
        {
        }

        multi_tensor_value_t( memory_pool_t &memory_pool, memory_buffer_t &memory_buffer, const tensor_shape_t &shape )
            : shape{ shape }

        {
        }

        multi_tensor_value_t( const multi_tensor_value_t & ) = default;

        tensor_shape_t &Shape()
        {
            return shape;
        }
    };

    /// @brief sSample2DComponent
    ///
    /// When added to an entity, this component samples a list of textures with the provided coordinates.
    ///
    struct sample2D_operation_t
    {
        graph_node_t x{};        //!< X coordinates of the texture samples
        graph_node_t y{};        //!< Y coordinates of the texture samples
        graph_node_t textures{}; //!< Textures to sample from

        sample2D_operation_t()                               = default;
        sample2D_operation_t( const sample2D_operation_t & ) = default;
    };

    /// @brief sToFixedPointNodeComponent
    ///
    /// When added to an entity, this component defines a node that converts floating point numbers to fixed
    /// point decimal mnumbers encoded as integers.
    ///
    struct convert_to_fixed_point_t
    {
        scalar_type_t output_type = scalar_type_t::UINT32; //!< Integer type to use to engode the fixed point decimal numbers
        graph_node_t  array{};                             //!< Input tensor/
        graph_node_t  mScaling{};                          //!< Scaling factor.

        convert_to_fixed_point_t()                                   = default;
        convert_to_fixed_point_t( const convert_to_fixed_point_t & ) = default;
    };

    /// @brief sAffineNodeComponent
    ///
    /// When added to an entity, this component defines a node that computes an affine transformation.
    ///
    struct affine_transform_operation_t
    {
        graph_node_t A{}; //!< Coefficient
        graph_node_t X{}; //!< Variable
        graph_node_t B{}; //!< Translation

        affine_transform_operation_t()                                       = default;
        affine_transform_operation_t( const affine_transform_operation_t & ) = default;
    };

    /// @brief sArraySliceNodeComponent
    struct array_slice_operation_t
    {
        graph_node_t array; //!< Tensor to transform
        graph_node_t begin; //!< Lower index value for each layer
        graph_node_t end;   //!< Upper index value for each layer

        graph_node_t element_count; //!< Length of the last dimension of `array`
        graph_node_t block_sizes;   //!< Product of the lengths of the first rank-1 dimensions of `array`

        uint32_t max_block_size = 0; //!< Maximum value of the `aBlockSizes` parameter

        array_slice_operation_t()                                  = default;
        array_slice_operation_t( const array_slice_operation_t & ) = default;
    };

    /// @brief sArraySummationNodeComponent
    struct array_sum_operation_t
    {
        graph_node_t array; //!< Tensor to transform
        graph_node_t begin; //!< Lower index value for each layer
        graph_node_t end;   //!< Upper index value for each layer

        graph_node_t element_count; // Length of the last dimension of `array`
        graph_node_t block_sizes;   //!< Product of the lengths of the first rank-1 dimensions of `array`

        uint32_t max_block_size = 0; //!< Maximum value of the `aBlockSizes` parameter

        array_sum_operation_t()                                = default;
        array_sum_operation_t( const array_sum_operation_t & ) = default;
    };

    /// @brief sConv1DNodeComponent
    struct conv1d_operation_t
    {
        graph_node_t array0;                 //!< Tensor to transform
        graph_node_t element_count0;         //!< Length of the last dimension of `array0`
        graph_node_t block_sizes0;           //!< Product of the lengths of the first rank-1 dimensions of `array0`
        uint32_t     max_element_count0 = 0; //!< Maximum value of the `element_count0` parameter
        uint32_t     max_block_size0    = 0; //!< Maximum value of the `aBlockSizes0` parameter

        graph_node_t array1;              //!< Convolution kernel
        graph_node_t element_count1;      //!< Length of the last dimension of `array1`
        graph_node_t block_sizes1;        //!< Product of the lengths of the first rank-1 dimensions of `array1`
        uint32_t     max_block_size1 = 0; //!< Maximum value of the `aBlockSizes1` parameter

        conv1d_operation_t()                             = default;
        conv1d_operation_t( const conv1d_operation_t & ) = default;
    };

    /// @brief sCountTrueNodeComponent
    struct count_true_operation_t
    {
        graph_node_t array;         //!< Tensor to transform
        graph_node_t block_sizes;   //!< Product of the lengths of the first rank-1 dimensions of `array`
        graph_node_t element_count; //!< Length of the last dimension of `array`

        uint32_t max_block_size = 0; //!< Maximum value of the `aBlockSizes` parameter

        count_true_operation_t()                                 = default;
        count_true_operation_t( const count_true_operation_t & ) = default;
    };

    /// @brief sCountNonZeroNodeComponent
    struct count_non_zero_operation_t
    {
        graph_node_t array;         //!< Tensor to transform
        graph_node_t block_sizes;   //!< Product of the lengths of the first rank-1 dimensions of `array`
        graph_node_t element_count; //!< Length of the last dimension of `array`

        uint32_t max_block_size = 0; //!< Maximum value of the `aBlockSizes` parameter

        count_non_zero_operation_t()                                     = default;
        count_non_zero_operation_t( const count_non_zero_operation_t & ) = default;
    };

    /// @brief sCountZeroNodeComponent
    struct count_zero_operation_t
    {
        graph_node_t array;         //!< Tensor to transform
        graph_node_t block_sizes;   //!< Product of the lengths of the first rank-1 dimensions of `array`
        graph_node_t element_count; //!< Length of the last dimension of `array`

        uint32_t max_block_size = 0; //!< Maximum value of the `aBlockSizes` parameter

        count_zero_operation_t()                                 = default;
        count_zero_operation_t( const count_zero_operation_t & ) = default;
    };

    /// @brief sFloorNodeComponent
    struct floor_operation_t
    {
        graph_node_t array; //!< Tensor to transform

        floor_operation_t()                            = default;
        floor_operation_t( const floor_operation_t & ) = default;
    };

    /// @brief sCeilNodeComponent
    struct ceiling_operation_t
    {
        graph_node_t array; //!< Tensor to transform

        ceiling_operation_t()                              = default;
        ceiling_operation_t( const ceiling_operation_t & ) = default;
    };

    /// @brief sAbsNodeComponent
    struct abs_operation_t
    {
        graph_node_t array; //!< Tensor to transform

        abs_operation_t()                          = default;
        abs_operation_t( const abs_operation_t & ) = default;
    };

    /// @brief sSqrtNodeComponent
    struct sqrt_operation_t
    {
        graph_node_t array; //!< Tensor to transform

        sqrt_operation_t()                           = default;
        sqrt_operation_t( const sqrt_operation_t & ) = default;
    };

    /// @brief sRoundNodeComponent
    struct round_operation_t
    {
        graph_node_t array; //!< Tensor to transform

        round_operation_t()                            = default;
        round_operation_t( const round_operation_t & ) = default;
    };

    /// @brief sDiffNodeComponent
    struct diff_operation_t
    {
        graph_node_t array; //!< Tensor to transform
        uint32_t     count = 0;

        graph_node_t block_sizes;   //!< Product of the lengths of the first rank-1 dimensions of `array`
        graph_node_t element_count; //!< Length of the last dimension of `array`

        uint32_t max_block_size = 0; //!< Maximum value of the `aBlockSizes` parameter

        diff_operation_t()                           = default;
        diff_operation_t( const diff_operation_t & ) = default;
    };

    /// @brief sShiftNodeComponent
    struct shift_operation_t
    {
        graph_node_t array; //!< Tensor to transform
        int32_t      count = 0;
        graph_node_t fill_value;

        graph_node_t block_sizes;   //!< Product of the lengths of the first rank-1 dimensions of `array`
        graph_node_t element_count; //!< Length of the last dimension of `array`

        uint32_t max_block_size = 0; //!< Maximum value of the `aBlockSizes` parameter

        shift_operation_t()                            = default;
        shift_operation_t( const shift_operation_t & ) = default;
    };

    /// @brief sHCatNodeComponent
    struct hcat_operation_t
    {
        graph_node_t array0; //!< First tensor to concatenate
        graph_node_t array1; //!< Second tensor to concatenate

        graph_node_t block_sizes;        //!< Product of the lengths of the first rank-1 dimensions of `array0`
        uint32_t     max_block_size = 0; //!< Maximum value of the `block_sizes` parameter

        graph_node_t element_count0; //!< Length of the last dimension of `array0`
        graph_node_t element_count1; //!< Length of the last dimension of `array1`

        hcat_operation_t()                           = default;
        hcat_operation_t( const hcat_operation_t & ) = default;
    };
} // namespace numlua::mtops
