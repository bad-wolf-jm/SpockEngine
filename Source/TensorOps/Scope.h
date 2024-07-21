/// @file   Scope.h
///
/// @brief  Computation scope.
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert. All rights reserved.

#pragma once
#include <deque>
#include <optional>
#include <stack>
#include <unordered_map>

#include "Core/Math/Types.h"

#include "Core/Entity/Collection.h"

#include "Core/CUDA/Array/MemoryPool.h"
#include "Core/CUDA/Array/MultiTensor.h"

#include "NodeComponents.h"
#include "NodeControllers.h"

namespace numlua::mtops
{
    using graph_node_t = numlua::core::entity_t;

    struct scope_t
    {
        memory_pool_t mPool{}; //!< Memory pool

        /// @brief Default constructor
        scope_t() = default;

        /// @brief Copy constructor
        scope_t( const scope_t & ) = default;

        /// @brief Create a scope, and reserves `memorySize` butes of GPU memory for it
        ///
        /// @param memorySize Size, in bytes, of the underlying memory pool
        ///
        scope_t( uint32_t memorySize );

        /// @brief Set `name` to be the name of the next node
        ///
        /// @param name REquested name
        ///
        /// @returns The parent scope for method chaining/
        ///
        scope_t &WithOpName( const string_t &name );

        /// @brief Create a node in the database
        ///
        /// If a name has been set using WithOpName, then it can be used to retrieve the node at a later time. The node will also be
        /// tagged with the requested name
        ///
        /// @return The newly cerated node.
        ///
        graph_node_t CreateNode();

        /// @brief Retrieve a node by name
        graph_node_t operator[]( string_t const &nodeName );

        /// @brief Clears the node registry, and resets the memory pool
        void Reset();

        /// @brief Overloaded method provided for convenience.
        void Run( graph_node_t const &node );

        /// @brief Run a given list of nodes
        ///
        /// This implies running all nodes used as inputs for the nodes to run.
        ///
        void Run( vector_t<graph_node_t> const &node );

        /// @brief Access the underlying nodes registry
        numlua::core::entity_registry_t &GetNodesRegistry()
        {
            return _nodes_registry;
        };

      private:
        numlua::core::entity_registry_t _nodes_registry{};    //!< Underlying node database
        std::optional<string_t>         _name = std::nullopt; //!< If this is set, the next node will be stored under the given value
        std::unordered_map<string_t, graph_node_t> _named_nodes = {}; //!< Mapping of node names to OpNodes
    };

    /// @brief Create a constant @ref MultiTensor initialized with the given constant
    ///
    /// @param scope computation scope
    /// @param initializer Initialization method to use
    /// @param shape Shape pf the tensor
    ///
    /// @return The newly created computation node
    ///
    graph_node_t MultiTensorValue( scope_t &scope, constant_value_initializer_t const &initializer,
                                   cuda::tensor_shape_t const &shape );

    /// @brief Create a constant @ref MultiTensor initialized with the given vector of values
    ///
    /// The length of `initializer` should match the number of layers defined in `shape`. Each layer of the tensor
    /// is initialized with the corresponding value in `initializer`.
    ///
    /// @param scope computation scope
    /// @param initializer Initialization method to use
    /// @param shape Shape pf the tensor
    ///
    /// @return The newly created computation node
    ///
    graph_node_t MultiTensorValue( scope_t &scope, vector_initializer_t const &initializer, cuda::tensor_shape_t const &shape );

    /// @brief Create a constant @ref MultiTensor initialized with the given data
    ///
    /// The length of `initializer` should match the number of elements defined in `shape`.
    ///
    /// @param scope computation scope
    /// @param initializer Initialization method to use
    /// @param shape Shape pf the tensor
    ///
    /// @return The newly created computation node
    ///
    graph_node_t MultiTensorValue( scope_t &scope, data_initializer_t const &initializer, cuda::tensor_shape_t const &shape );

    /// @brief Create a constant @ref MultiTensor initialized with uniformly distributed random values
    ///
    /// @param scope computation scope
    /// @param initializer Initialization method to use
    /// @param shape Shape pf the tensor
    ///
    /// @return The newly created computation node
    ///
    graph_node_t MultiTensorValue( scope_t &scope, random_uniform_initializer_t const &initializer,
                                   cuda::tensor_shape_t const &shape );

    /// @brief Create a constant @ref MultiTensor initialized with normally distributed random values
    ///
    /// @param scope computation scope
    /// @param initializer Initialization method to use
    /// @param shape Shape pf the tensor
    ///
    /// @return The newly created computation node
    ///
    graph_node_t MultiTensorValue( scope_t &scope, random_normal_initializer_t const &initializer, cuda::tensor_shape_t const &shape );

    /// @brief Create a constant @ref MemoryBuffer initialized with the given vector
    ///
    /// @tparam _Ty Type of the elements
    ///
    /// @param scope computation scope
    /// @param value Vector of values to upload to the GPU upon running the node
    ///
    /// @return The newly created computation node
    ///
    template <typename _Ty>
    graph_node_t VectorValue( scope_t &scope, vector_t<_Ty> const &value )
    {
        auto new_entity = scope.CreateNode();

        auto &value_component = new_entity.Add<vector_value_t<_Ty>>();
        value_component.value = value;

        auto &buffer = new_entity.Add<vector_buffer_t>();
        buffer.size  = value.size() * sizeof( _Ty );

        if constexpr( std::is_same_v<_Ty, scalar_value_t> )
        {
            new_entity.Get<node_id_t>().element_type = type_of( value[0] );
        }

        new_entity.Add<graph_operation_t>().Bind<VectorRunner<_Ty>>();

        return new_entity;
    }

    /// @brief Create a constant @ref MemoryBuffer of ScalarValues initialized with the given vector
    ///
    /// @tparam _Ty Type of the elements
    ///
    /// @param scope computation scope
    /// @param value Vector of values to upload to the GPU upon running the node
    ///
    /// @return The newly created computation node
    ///
    template <typename _Ty>
    graph_node_t ScalarVectorValue( scope_t &scope, scalar_type_t type, vector_t<_Ty> const &value )
    {
        uint32_t                 size = value.size();
        vector_t<scalar_value_t> values( size );
        for( uint32_t i = 0; i < size; i++ )
        {
            values[i] = value[i];
        }
        return VectorValue( scope, values );
    }

    /// @brief Create a scalar initialized with the given value.
    ///
    /// @tparam _Ty Type of the elements
    ///
    /// @param scope computation scope
    /// @param value Value to upload to the GPU upon running the node
    ///
    /// @return The newly created computation node
    ///
    template <typename _Ty>
    graph_node_t ConstantScalarValue( scope_t &scope, _Ty const &value )
    {
        auto new_entity = scope.CreateNode();

        auto &value_component = new_entity.Add<scalar_node_t>();
        value_component.value = value;

        new_entity.Get<node_id_t>().element_type = type_of( value_component.value );

        return new_entity;
    }

    /// @brief Adds the outputs of two nodes
    ///
    /// At least one of `left` and `right` should be a @ref MultiTensor nodes. If both operands are @ref MultiTensors,
    /// then they should have the same shape. If one of `left` and `right` is a vector, then its length should match
    /// the number of layers of the other operand (which has to be a tensor). The dimension of the output tensor is the
    /// same as that of the input.
    ///
    /// @param scope Computation scope
    /// @param left Left operand
    /// @param right Right operand
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Add( scope_t &scope, graph_node_t const &left, graph_node_t const &right );

    /// @brief Subtracts the outputs of two nodes
    ///
    /// At least one of `left` and `right` should be a @ref MultiTensor nodes. If both operands are @ref MultiTensors,
    /// then they should have the same shape. If one of `left` and `right` is a vector, then its length should match
    /// the number of layers of the other operand (which has to be a tensor). The dimension of the output tensor is the
    /// same as that of the input.
    ///
    /// @param scope Computation scope
    /// @param left Left operand
    /// @param right Right operand
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Subtract( scope_t &scope, graph_node_t const &left, graph_node_t const &right );

    /// @brief Divides the outputs of two nodes
    ///
    /// At least one of `left` and `right` should be a @ref MultiTensor nodes. If both operands are @ref MultiTensors,
    /// then they should have the same shape. If one of `left` and `right` is a vector, then its length should match
    /// the number of layers of the other operand (which has to be a tensor). The dimension of the output tensor is the
    /// same as that of the input.
    ///
    /// @param scope Computation scope
    /// @param left Left operand
    /// @param right Right operand
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Divide( scope_t &scope, graph_node_t const &left, graph_node_t const &right );

    /// @brief Multiplies the outputs of two nodes
    ///
    /// At least one of `left` and `right` should be a @ref MultiTensor nodes. If both operands are @ref MultiTensors,
    /// then they should have the same shape. If one of `left` and `right` is a vector, then its length should match
    /// the number of layers of the other operand (which has to be a tensor). The dimension of the output tensor is the
    /// same as that of the input.
    ///
    /// @param scope Computation scope
    /// @param left Left operand
    /// @param right Right operand
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Multiply( scope_t &scope, graph_node_t const &left, graph_node_t const &right );

    /// @brief Conjunction of two boolean (uint8_t) nodes
    ///
    /// At least one of `left` and `right` should be a @ref MultiTensor nodes. If both operands are @ref MultiTensors,
    /// then they should have the same shape. If one of `left` and `right` is a vector, then its length should match
    /// the number of layers of the other operand (which has to be a tensor). The dimension of the output tensor is the
    /// same as that of the input.
    ///
    /// @param scope Computation scope
    /// @param left Left operand
    /// @param right Right operand
    ///
    /// @return The newly created computation node
    ///
    graph_node_t And( scope_t &scope, graph_node_t const &left, graph_node_t const &right );

    /// @brief Disjunction of two boolean (uint8_t) nodes
    ///
    /// At least one of `left` and `right` should be a @ref MultiTensor nodes. If both operands are @ref MultiTensors,
    /// then they should have the same shape. If one of `left` and `right` is a vector, then its length should match
    /// the number of layers of the other operand (which has to be a tensor). The dimension of the output tensor is the
    /// same as that of the input.
    ///
    /// @param scope Computation scope
    /// @param left Left operand
    /// @param right Right operand
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Or( scope_t &scope, graph_node_t const &left, graph_node_t const &right );

    /// @brief Negation of two boolean (uint8_t) nodes
    ///
    /// The parameter `operand` should be a @ref MultiTensor node. The dimension of the output tensor is the
    /// same as that of the input.
    ///
    /// @param scope Computation scope
    /// @param operand Operand
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Not( scope_t &scope, graph_node_t const &operand );

    /// @brief Bitwise conjunction of two integer nodes
    ///
    /// At least one of `left` and `right` should be a @ref MultiTensor nodes. If both operands are @ref MultiTensors,
    /// then they should have the same shape. If one of `left` and `right` is a vector, then its length should match
    /// the number of layers of the other operand (which has to be a tensor). The dimension of the output tensor is the
    /// same as that of the input.
    ///
    /// @param scope Computation scope
    /// @param left Left operand
    /// @param right Right operand
    ///
    /// @return The newly created computation node
    ///
    graph_node_t BitwiseAnd( scope_t &scope, graph_node_t const &left, graph_node_t const &right );

    /// @brief Bitwise disjunction of two integer nodes
    ///
    /// At least one of `left` and `right` should be a @ref MultiTensor nodes. If both operands are @ref MultiTensors,
    /// then they should have the same shape. If one of `left` and `right` is a vector, then its length should match
    /// the number of layers of the other operand (which has to be a tensor). The dimension of the output tensor is the
    /// same as that of the input.
    ///
    /// @param scope Computation scope
    /// @param left Left operand
    /// @param right Right operand
    ///
    /// @return The newly created computation node
    ///
    graph_node_t BitwiseOr( scope_t &scope, graph_node_t const &left, graph_node_t const &right );

    /// @brief Bitwise negation of two integer nodes
    ///
    /// The parameter `operand` should be a @ref MultiTensor node. The dimension of the output tensor is the
    /// same as that of the input.
    ///
    /// @param scope Computation scope
    /// @param operand Operand
    ///
    /// @return The newly created computation node
    ///
    graph_node_t BitwiseNot( scope_t &scope, graph_node_t const &operand );

    /// @brief Test whether the values contained in a tensor lie within an interval
    ///
    /// The parameter `x` should be a @ref MultiTensor node. The dimension of the output tensor is the
    /// same as that of the input.
    ///
    /// @param scope Computation scope
    /// @param x Operand
    /// @param lower Lower bound for the interval
    /// @param upper Upper bound for the interval
    /// @param strictLower Use strict inequality for the lower bound
    /// @param strictUpper Use strict inequality for the lower bound
    ///
    /// @return The newly created computation node
    ///
    graph_node_t InInterval( scope_t &scope, graph_node_t const &x, graph_node_t const &lower, graph_node_t const &upper,
                             bool strictLower, bool strictUpper );

    /// @brief Equality
    ///
    /// At least one of `x` and `y` should be a @ref MultiTensor nodes. If both operands are @ref MultiTensors,
    /// then they should have the same shape. If one of `left` and `right` is a vector, then its length should match
    /// the number of layers of the other operand (which has to be a tensor). The dimension of the output tensor is the
    /// same as that of the input.
    ///
    /// @param scope Computation scope
    /// @param x Left operand
    /// @param y Right operand
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Equal( scope_t &scope, graph_node_t const &x, graph_node_t const &y );

    /// @brief Less than
    ///
    /// At least one of `x` and `y` should be a @ref MultiTensor nodes. If both operands are @ref MultiTensors,
    /// then they should have the same shape. If one of `left` and `right` is a vector, then its length should match
    /// the number of layers of the other operand (which has to be a tensor). The dimension of the output tensor is the
    /// same as that of the input.
    ///
    /// @param scope Computation scope
    /// @param x Left operand
    /// @param y Right operand
    ///
    /// @return The newly created computation node
    ///
    graph_node_t LessThan( scope_t &scope, graph_node_t const &x, graph_node_t const &y );

    /// @brief Less than or equal to
    ///
    /// At least one of `x` and `y` should be a @ref MultiTensor nodes. If both operands are @ref MultiTensors,
    /// then they should have the same shape. If one of `left` and `right` is a vector, then its length should match
    /// the number of layers of the other operand (which has to be a tensor). The dimension of the output tensor is the
    /// same as that of the input.
    ///
    /// @param scope Computation scope
    /// @param x Left operand
    /// @param y Right operand
    ///
    /// @return The newly created computation node
    ///
    graph_node_t LessThanOrEqual( scope_t &scope, graph_node_t const &x, graph_node_t const &y );

    /// @brief Greater than
    ///
    /// At least one of `x` and `y` should be a @ref MultiTensor nodes. If both operands are @ref MultiTensors,
    /// then they should have the same shape. If one of `left` and `right` is a vector, then its length should match
    /// the number of layers of the other operand (which has to be a tensor). The dimension of the output tensor is the
    /// same as that of the input.
    ///
    /// @param scope Computation scope
    /// @param x Left operand
    /// @param y Right operand
    ///
    /// @return The newly created computation node
    ///
    graph_node_t GreaterThan( scope_t &scope, graph_node_t const &x, graph_node_t const &y );

    /// @brief Greater than or equal to
    ///
    /// At least one of `x` and `y` should be a @ref MultiTensor nodes. If both operands are @ref MultiTensors,
    /// then they should have the same shape. If one of `left` and `right` is a vector, then its length should match
    /// the number of layers of the other operand (which has to be a tensor). The dimension of the output tensor is the
    /// same as that of the input.
    ///
    /// @param scope Computation scope
    /// @param x Left operand
    /// @param y Right operand
    ///
    /// @return The newly created computation node
    ///
    graph_node_t GreaterThanOrEqual( scope_t &scope, graph_node_t const &x, graph_node_t const &y );

    /// @brief Choose values from one tensor or another based on a given condition
    ///
    /// The parameter `condition` should be a @ref MultiTensor nodes. If all operands are @ref MultiTensors,
    /// then they should have the same shape. If one of `valueIfTrue` and `valueIfFalse` is a vector, then its length should match
    /// the number of layers of the other operand (which has to be a tensor). The dimension of the output tensor is the
    /// same as that of `condition`.
    ///
    /// @param scope Computation scope
    /// @param condition Condition to test
    /// @param valueIfTrue Value to use if condition is true
    /// @param valueIfFalse Value to use if condition is false
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Where( scope_t &scope, graph_node_t const &condition, graph_node_t const &valueIfTrue,
                        graph_node_t const &valueIfFalse );

    /// @brief Computes the pointwise mix of two tensors
    ///
    /// All of `A`, `B` and `t` should be @ref MultiTensors of the same shape and type. This function computes the tensor
    /// @f$ (1-t)\cdot A + t\cdot B @f$, the shape of which is the same as the common shape of `A`, `B` and `t`
    ///
    /// @param scope Parent computation scope
    /// @param array Array to repeat
    /// @param A Input tensor
    /// @param B Input tensor
    /// @param t Input tensor
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Mix( scope_t &scope, graph_node_t const &A, graph_node_t const &B, graph_node_t const &t );

    /// @brief Affine transforms
    ///
    /// The parameter `x` has to be a @ref MultiTensor. The other two parameters can be any vectors, or scalars. This node
    /// computes the affine transformation @f$ a\cdot X+b @f$, the output of which has the same shape as `x`. If either of
    /// `A` or `B` is a @ref MultiTensor, then is should have the same shape as `x`, and is either one is a vector, then
    /// its length should match the number of layers of `x`.
    ///
    /// @param scope Parent computation scope
    /// @param A Input tensor
    /// @param B Input tensor
    /// @param x Input tensor
    ///
    /// @return The newly created computation node
    ///
    graph_node_t AffineTransform( scope_t &scope, graph_node_t const &A, graph_node_t const &x, graph_node_t const &B );

    /// @brief Computes a set of ranges of values with a regular step.
    ///
    /// The two vectors contained in `left` and `right` should have the same length and contain floating
    /// point values. This is roughly a generazed version of numpy's `np.arange`. The resulting output tensor
    /// will have rank 1, with one layer for every element of `left`. Each layer will have dimension
    /// @f$ (R - L) / \Delta @f$
    ///
    /// @param scope Parent computation scope
    /// @param left Lower bounds
    /// @param right Upper bounds
    /// @param delta Range step
    ///
    /// @return The newly created computation node
    ///
    graph_node_t ARange( scope_t &scope, graph_node_t const &left, graph_node_t const &right, graph_node_t const &delta );

    /// @brief Computes evenly spaced numbers in the intervals specified by two tensors
    ///
    /// Roughly equivalent to numpy's np.linspace. The two input tensors should have the same shape. The interval between them
    /// is subdivided into `repetitions` many subintervals, where each element in `repetitions` is matched with the corresponding
    /// layer of the input tensors. If the input multi-tensor have rank @f$ N @f$ , then the output multi-tensor will have rank
    /// @f$ N+1 @f$. The last dimension of the output multi-tensor is the number of subdivisions.
    ///
    /// @param scope Parent computation scope
    /// @param array Array to repeat
    /// @param repetitions Nummber of repetitions
    /// @param aOut Output tensor.
    ///
    /// @return The newly created computation node
    ///
    graph_node_t LinearSpace( scope_t &scope, graph_node_t const &left, graph_node_t const &right, graph_node_t const &subdivisions );

    /// @brief Repeat each element of a multi-tensor.
    ///
    /// Roughly equivalent to numpy's npo.repeat. Each element of the innermost dimension of a multitensor is repeated a given
    /// number of times. Node that a different number of repetitions can be specified for each layer of the multi-tensor. As
    /// far as dimension and rank are concerned, if the input multi-tensor has rank @f$ N @f$ , then, the repeated multi-tensor
    /// will have rank @f$ N+1 @f$. The last dimension of the output multi-tensor is the number of repetitions.
    ///
    /// @param scope Parent computation scope
    /// @param array Array to repeat
    /// @param repetitions Nummber of repetitions
    /// @param aOut Output tensor.
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Repeat( scope_t &scope, graph_node_t const &array, graph_node_t const &repetitions );

    /// @brief Repeat each layer of a multi-tensor.
    ///
    /// Roughly equivalent to numpy's npo.tile. Each layer of the a multitensor is repeated a given number of times. Node that
    /// a different number of repetitions can be specified for each layer of the multi-tensor. As far as dimension and rank are
    /// concerned, if the input multi-tensor has rank @f$ N @f$ , then, the repeated multi-tensor will have rank @f$ N+1 @f$.
    /// The first dimension of the output multi-tensor is the number of repetitions.
    ///
    /// @param scope Parent computation scope
    /// @param array Array to repeat
    /// @param repetitions Nummber of repetitions
    /// @param aOut Output tensor.
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Tile( scope_t &scope, graph_node_t const &array, graph_node_t const &repetitions );

    /// @brief Texture sampling
    ///
    /// Samples the textures in `textures` at the coordinates specified by `x` and `y`. The tensors `x` and `y`
    /// should have the same shape, which will be the output shape. `textures` should represent a vector of
    /// @ref sTextureData whose length matches the number of layers in the tensors `x` and `y`. Each layer of the
    /// output will be sampled from the corresponding texture in `aTestures`
    ///
    /// @param scope Parent computation scope
    /// @param array Array to repeat
    /// @param repetitions Nummber of repetitions
    /// @param aOut Output tensor.
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Sample2D( scope_t &scope, graph_node_t const &x, graph_node_t const &y, graph_node_t const &textures );

    /// @brief Fixed point conversion
    ///
    /// Converts a tensor of floating point numbers into a tensor of integers by multiplying each element by a scaling factor.
    ///
    /// @param scope Parent computation scope
    /// @param array Array to repeat
    /// @param repetitions Nummber of repetitions
    /// @param aOut Output tensor.
    ///
    /// @return The newly created computation node
    ///
    graph_node_t ToFixedPoint( scope_t &scope, scalar_type_t outputType, graph_node_t const &array, graph_node_t const &scaling );

    /// @brief Collapse a @ref MultiTensor into a @ref MultiTensor having only one layer
    ///
    /// The dimensions of each layer of the @ref MultiTensor should be equal. Furthermore, the memory area is shared between the
    /// input and the output multitensors, so that there is no actual copying involved.
    ///
    /// @param scope Parent computation scope
    /// @param array MultiTensor to collapse
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Collapse( scope_t &scope, graph_node_t const &array );

    /// @brief Expand the first dimension of a @ref MultiTensor with only one layer into a multi-layered MultiTensor
    ///
    /// The number of layers in the output @ref MultiTensor is equal to the first dimension of the input @ref MultiTensor.
    /// Furthermore, the memory area is shared between the input and the output multitensors, so that there is no actual copying
    /// involved.
    ///
    /// @param scope Parent computation scope
    /// @param array MultiTensor to expand
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Expand( scope_t &scope, graph_node_t const &array );

    /// @brief Reshape the input @ref MultiTensor
    ///
    /// The new shape should bave the same number of layers as the original shape. Furthermore, the dimensions of each layer of
    /// the old and new shapes should be compatible in having equal products, and equal element sizes. The memory is shared between
    /// the input and the output tensors, so that no copying is involved.
    ///
    /// @param scope Parent computation scope
    /// @param array MultiTensor to reshape
    /// @param newShape New shape for the output tensor
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Reshape( scope_t &scope, graph_node_t const &array, cuda::tensor_shape_t &newShape );

    /// @brief Relayout the input @ref MultiTensor
    ///
    /// Applies a new layout to the input MultiTensor. The new layout should have the same size as the old layout.
    ///
    /// @param scope Parent computation scope
    /// @param array MultiTensor to relayout
    /// @param newLayout New layout
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Relayout( scope_t &scope, graph_node_t const &array, tensor_shape_t &newLayout );

    /// @brief Flatten the input @ref MultiTensor
    ///
    /// The output multi-tensor will have the same number of layers as the input, but it will have rank 1, and the dimension of each
    /// layer will be the product of the corresponding layer in the input.
    ///
    /// @param scope Parent computation scope
    /// @param array MultiTensor to flatten
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Flatten( scope_t &scope, graph_node_t const &array );

    /// @brief Slice the input @ref MultiTensor
    ///
    /// The output multi-tensor will have the same number of layers as the input. The nodes `begin` and `end` denote the start and
    /// end indices of the slice respectively, and should correspond to either a scalar value, or a vector whose entries are uint32_t.
    /// Entries at `begin` and `end` are included in the slice. For now, slicing a multi-tensor only acts on the last dimension.
    ///
    /// @param scope Parent computation scope
    /// @param array MultiTensor to slice
    /// @param begin Lower bound
    /// @param end Upper bound
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Slice( scope_t &scope, graph_node_t const &array, graph_node_t const &begin, graph_node_t const &end );

    /// @brief Sum the last dimension of the input @ref MultiTensor
    ///
    /// The output multi-tensor will have the same number of layers as the input. For now, summing a multi-tensor only considers
    /// the last dimension.
    ///
    /// @param scope Parent computation scope
    /// @param array MultiTensor to sum
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Summation( scope_t &scope, graph_node_t const &array );

    /// @brief Sum the last dimension of the input @ref MultiTensor
    ///
    /// The output multi-tensor will have the same number of layers as the input. The nodes `begin` and `end` denote the start and
    /// end indices of the sum respectively, and should correspond to either a scalar value, or a vector whose entries are uint32_t.
    /// Entries at `begin` and `end` are included in the sum. For now, summing a multi-tensor only considers the last dimension.
    ///
    /// @param scope Parent computation scope
    /// @param array MultiTensor to flatten
    /// @param begin Lower bound
    /// @param end Upper bound
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Summation( scope_t &scope, graph_node_t const &array, graph_node_t const &begin, graph_node_t const &end );

    /// @brief Count the number of `true` elements in the last dimension of @ref MultiTensor
    ///
    /// The output multi-tensor will have the same number of layers as the input. For now, counting the true values in a multi-tensor
    /// only considers the last dimension.
    ///
    /// @param scope Parent computation scope
    /// @param array MultiTensor to process
    ///
    /// @return The newly created computation node
    ///
    graph_node_t CountTrue( scope_t &scope, graph_node_t const &array );

    /// @brief Count the number of non-zero elements in the last dimension of @ref MultiTensor
    ///
    /// The output multi-tensor will have the same number of layers as the input. For now, counting the non-zero values of a
    /// multi-tensor only considers the last dimension.
    ///
    /// @param scope Parent computation scope
    /// @param array MultiTensor to process
    ///
    /// @return The newly created computation node
    ///
    graph_node_t CountNonZero( scope_t &scope, graph_node_t const &array );

    /// @brief Count the number of zero elements in the last dimension of @ref MultiTensor
    ///
    /// The output multi-tensor will have the same number of layers as the input. For now, counting the zero values of a multi-tensor
    /// only considers the last dimension.
    ///
    /// @param scope Parent computation scope
    /// @param array MultiTensor to process
    ///
    /// @return The newly created computation node
    ///
    graph_node_t CountZero( scope_t &scope, graph_node_t const &array );

    /// @brief Compute the pointwise floor of @ref MultiTensor
    ///
    /// The output multi-tensor will have the same dimension as the input.
    ///
    /// @param scope Parent computation scope
    /// @param array MultiTensor to process
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Floor( scope_t &scope, graph_node_t const &array );

    /// @brief Compute the pointwise ceiling of @ref MultiTensor
    ///
    /// The output multi-tensor will have the same dimension as the input.
    ///
    /// @param scope Parent computation scope
    /// @param array MultiTensor to process
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Ceil( scope_t &scope, graph_node_t const &array );

    /// @brief Compute the pointwise absolute value of @ref MultiTensor
    ///
    /// The output multi-tensor will have the same dimension as the input.
    ///
    /// @param scope Parent computation scope
    /// @param array MultiTensor to process
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Abs( scope_t &scope, graph_node_t const &array );

    /// @brief Compute the pointwise square root value of @ref MultiTensor
    ///
    /// The output multi-tensor will have the same dimension as the input.
    ///
    /// @param scope Parent computation scope
    /// @param array MultiTensor to process
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Sqrt( scope_t &scope, graph_node_t const &array );

    /// @brief Compute the pointwise rounded value of @ref MultiTensor
    ///
    /// The output multi-tensor will have the same dimension as the input.
    ///
    /// @param scope Parent computation scope
    /// @param array MultiTensor to process
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Round( scope_t &scope, graph_node_t const &array );

    /// @brief Compute the iterated finite difference along the last dimension of @ref MultiTensor
    ///
    /// The output multi-tensor will have the same dimension as the input. The final entries of the output
    /// tensor are set to 0
    ///
    /// @param scope Parent computation scope
    /// @param array MultiTensor to process
    /// @param count Number of iterations
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Diff( scope_t &scope, graph_node_t const &array, uint32_t count );

    /// @brief Compute the finite shift along the last dimension of @ref MultiTensor
    ///
    /// The output multi-tensor will have the same dimension as the input. The final entries of the output
    /// tensor are set to fillValue.
    ///
    /// @param scope Parent computation scope
    /// @param array MultiTensor to process
    /// @param count Number of images to shift
    /// @param fillValue Value used to fill the missing positions
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Shift( scope_t &scope, graph_node_t const &array, int32_t count, graph_node_t const &fillValue );

    /// @brief Compute the 1-dimensional convolution the last dimension of @ref MultiTensor
    ///
    /// The output multi-tensor will have the same dimension as the left.
    ///
    /// @param scope Parent computation scope
    /// @param array0 MultiTensor to process
    /// @param array1 Convolution kernel
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Conv1D( scope_t &scope, graph_node_t const &array0, graph_node_t const &array1 );

    /// @brief Concatenate the given @ref MultiTensors along the last dimension
    ///
    /// @param scope Parent computation scope
    /// @param array0 MultiTensor to process
    /// @param array1 MultiTensor to process
    ///
    /// @return The newly created computation node
    ///
    graph_node_t HCat( scope_t &scope, graph_node_t const &array0, graph_node_t const &array1 );

} // namespace numlua::mtops
