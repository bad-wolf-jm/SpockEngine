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
#include "Core/CUDA/CudaAssert.h"

#include "NodeComponents.h"
#include "NodeControllers.h"
#include "ScalarTypes.h"

namespace SE::TensorOps
{
    using graph_node_t = SE::Core::entity_t;

    struct scope_t
    {
        memory_pool_t mPool{}; //!< Memory pool

        /// @brief Default constructor
        scope_t() = default;

        /// @brief Copy constructor
        scope_t( const scope_t & ) = default;

        /// @brief Create a scope, and reserves `aMemorySize` butes of GPU memory for it
        ///
        /// @param aMemorySize Size, in bytes, of the underlying memory pool
        ///
        scope_t( uint32_t aMemorySize );

        /// @brief Set `aName` to be the name of the next node
        ///
        /// @param aName REquested name
        ///
        /// @returns The parent scope for method chaining/
        ///
        scope_t &WithOpName( const string_t &aName );

        /// @brief Create a node in the database
        ///
        /// If a name has been set using WithOpName, then it can be used to retrieve the node at a later time. The node will also be
        /// tagged with the requested name
        ///
        /// @return The newly cerated node.
        ///
        graph_node_t CreateNode();

        /// @brief Retrieve a node by name
        graph_node_t operator[]( string_t const &aNodeName );

        /// @brief Clears the node registry, and resets the memory pool
        void Reset();

        /// @brief Overloaded method provided for convenience.
        void Run( graph_node_t const &aNode );

        /// @brief Run a given list of nodes
        ///
        /// This implies running all nodes used as inputs for the nodes to run.
        ///
        void Run( vector_t<graph_node_t> const &aNode );

        /// @brief Access the underlying nodes registry
        SE::Core::entity_registry_t &GetNodesRegistry()
        {
            return mNodesRegistry;
        };

      private:
        SE::Core::entity_registry_t mNodesRegistry{};     //!< Underlying node database
        std::optional<string_t> mName = std::nullopt; //!< If this is set, the next node will be stored under the given value
        std::unordered_map<string_t, graph_node_t> mNamedNodes = {}; //!< Mapping of node names to OpNodes
    };

    /// @brief Create a constant @ref MultiTensor initialized with the given constant
    ///
    /// @param aScope computation scope
    /// @param aInitializer Initialization method to use
    /// @param aShape Shape pf the tensor
    ///
    /// @return The newly created computation node
    ///
    graph_node_t MultiTensorValue( scope_t &aScope, constant_value_initializer_t const &aInitializer, Cuda::tensor_shape_t const &aShape );

    /// @brief Create a constant @ref MultiTensor initialized with the given vector of values
    ///
    /// The length of `aInitializer` should match the number of layers defined in `aShape`. Each layer of the tensor
    /// is initialized with the corresponding value in `aInitializer`.
    ///
    /// @param aScope computation scope
    /// @param aInitializer Initialization method to use
    /// @param aShape Shape pf the tensor
    ///
    /// @return The newly created computation node
    ///
    graph_node_t MultiTensorValue( scope_t &aScope, vector_initializer_t const &aInitializer, Cuda::tensor_shape_t const &aShape );

    /// @brief Create a constant @ref MultiTensor initialized with the given data
    ///
    /// The length of `aInitializer` should match the number of elements defined in `aShape`.
    ///
    /// @param aScope computation scope
    /// @param aInitializer Initialization method to use
    /// @param aShape Shape pf the tensor
    ///
    /// @return The newly created computation node
    ///
    graph_node_t MultiTensorValue( scope_t &aScope, data_initializer_t const &aInitializer, Cuda::tensor_shape_t const &aShape );

    /// @brief Create a constant @ref MultiTensor initialized with uniformly distributed random values
    ///
    /// @param aScope computation scope
    /// @param aInitializer Initialization method to use
    /// @param aShape Shape pf the tensor
    ///
    /// @return The newly created computation node
    ///
    graph_node_t MultiTensorValue( scope_t &aScope, random_uniform_initializer_t const &aInitializer, Cuda::tensor_shape_t const &aShape );

    /// @brief Create a constant @ref MultiTensor initialized with normally distributed random values
    ///
    /// @param aScope computation scope
    /// @param aInitializer Initialization method to use
    /// @param aShape Shape pf the tensor
    ///
    /// @return The newly created computation node
    ///
    graph_node_t MultiTensorValue( scope_t &aScope, random_normal_initializer_t const &aInitializer, Cuda::tensor_shape_t const &aShape );

    /// @brief Create a constant @ref MemoryBuffer initialized with the given vector
    ///
    /// @tparam _Ty Type of the elements
    ///
    /// @param aScope computation scope
    /// @param aValue Vector of values to upload to the GPU upon running the node
    ///
    /// @return The newly created computation node
    ///
    template <typename _Ty>
    graph_node_t VectorValue( scope_t &aScope, vector_t<_Ty> const &aValue )
    {
        auto l_NewEntity = aScope.CreateNode();

        auto &l_Value  = l_NewEntity.Add<vector_value_t<_Ty>>();
        l_Value.mValue = aValue;

        auto &l_Buffer = l_NewEntity.Add<vector_buffer_t>();
        l_Buffer.mSize = aValue.size() * sizeof( _Ty );

        if constexpr( std::is_same_v<_Ty, scalar_value_t> )
        {
            l_NewEntity.Add<type_t>( TypeOf( aValue[0] ) );
        }

        l_NewEntity.Add<graph_operation_t>().Bind<VectorRunner<_Ty>>();

        return l_NewEntity;
    }

    /// @brief Create a constant @ref MemoryBuffer of ScalarValues initialized with the given vector
    ///
    /// @tparam _Ty Type of the elements
    ///
    /// @param aScope computation scope
    /// @param aValue Vector of values to upload to the GPU upon running the node
    ///
    /// @return The newly created computation node
    ///
    template <typename _Ty>
    graph_node_t ScalarVectorValue( scope_t &aScope, scalar_type_t aType, vector_t<_Ty> const &aValue )
    {
        uint32_t                 lSize = aValue.size();
        vector_t<scalar_value_t> lValues( lSize );
        for( uint32_t i = 0; i < lSize; i++ )
        {
            lValues[i] = aValue[i];
        }
        return VectorValue( aScope, lValues );
    }

    /// @brief Create a scalar initialized with the given value.
    ///
    /// @tparam _Ty Type of the elements
    ///
    /// @param aScope computation scope
    /// @param aValue Value to upload to the GPU upon running the node
    ///
    /// @return The newly created computation node
    ///
    template <typename _Ty>
    graph_node_t ConstantScalarValue( scope_t &aScope, _Ty const &aValue )
    {
        auto l_NewEntity = aScope.CreateNode();

        auto &l_Value  = l_NewEntity.Add<scalar_node_t>();
        l_Value.mValue = aValue;

        l_NewEntity.Add<type_t>( TypeOf( l_Value.mValue ) );

        return l_NewEntity;
    }

    /// @brief Adds the outputs of two nodes
    ///
    /// At least one of `aLeft` and `aRight` should be a @ref MultiTensor nodes. If both operands are @ref MultiTensors,
    /// then they should have the same shape. If one of `aLeft` and `aRight` is a vector, then its length should match
    /// the number of layers of the other operand (which has to be a tensor). The dimension of the output tensor is the
    /// same as that of the input.
    ///
    /// @param aScope Computation scope
    /// @param aLeft Left operand
    /// @param aRight Right operand
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Add( scope_t &aScope, graph_node_t const &aLeft, graph_node_t const &aRight );

    /// @brief Subtracts the outputs of two nodes
    ///
    /// At least one of `aLeft` and `aRight` should be a @ref MultiTensor nodes. If both operands are @ref MultiTensors,
    /// then they should have the same shape. If one of `aLeft` and `aRight` is a vector, then its length should match
    /// the number of layers of the other operand (which has to be a tensor). The dimension of the output tensor is the
    /// same as that of the input.
    ///
    /// @param aScope Computation scope
    /// @param aLeft Left operand
    /// @param aRight Right operand
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Subtract( scope_t &aScope, graph_node_t const &aLeft, graph_node_t const &aRight );

    /// @brief Divides the outputs of two nodes
    ///
    /// At least one of `aLeft` and `aRight` should be a @ref MultiTensor nodes. If both operands are @ref MultiTensors,
    /// then they should have the same shape. If one of `aLeft` and `aRight` is a vector, then its length should match
    /// the number of layers of the other operand (which has to be a tensor). The dimension of the output tensor is the
    /// same as that of the input.
    ///
    /// @param aScope Computation scope
    /// @param aLeft Left operand
    /// @param aRight Right operand
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Divide( scope_t &aScope, graph_node_t const &aLeft, graph_node_t const &aRight );

    /// @brief Multiplies the outputs of two nodes
    ///
    /// At least one of `aLeft` and `aRight` should be a @ref MultiTensor nodes. If both operands are @ref MultiTensors,
    /// then they should have the same shape. If one of `aLeft` and `aRight` is a vector, then its length should match
    /// the number of layers of the other operand (which has to be a tensor). The dimension of the output tensor is the
    /// same as that of the input.
    ///
    /// @param aScope Computation scope
    /// @param aLeft Left operand
    /// @param aRight Right operand
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Multiply( scope_t &aScope, graph_node_t const &aLeft, graph_node_t const &aRight );

    /// @brief Conjunction of two boolean (uint8_t) nodes
    ///
    /// At least one of `aLeft` and `aRight` should be a @ref MultiTensor nodes. If both operands are @ref MultiTensors,
    /// then they should have the same shape. If one of `aLeft` and `aRight` is a vector, then its length should match
    /// the number of layers of the other operand (which has to be a tensor). The dimension of the output tensor is the
    /// same as that of the input.
    ///
    /// @param aScope Computation scope
    /// @param aLeft Left operand
    /// @param aRight Right operand
    ///
    /// @return The newly created computation node
    ///
    graph_node_t And( scope_t &aScope, graph_node_t const &aLeft, graph_node_t const &aRight );

    /// @brief Disjunction of two boolean (uint8_t) nodes
    ///
    /// At least one of `aLeft` and `aRight` should be a @ref MultiTensor nodes. If both operands are @ref MultiTensors,
    /// then they should have the same shape. If one of `aLeft` and `aRight` is a vector, then its length should match
    /// the number of layers of the other operand (which has to be a tensor). The dimension of the output tensor is the
    /// same as that of the input.
    ///
    /// @param aScope Computation scope
    /// @param aLeft Left operand
    /// @param aRight Right operand
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Or( scope_t &aScope, graph_node_t const &aLeft, graph_node_t const &aRight );

    /// @brief Negation of two boolean (uint8_t) nodes
    ///
    /// The parameter `aOperand` should be a @ref MultiTensor node. The dimension of the output tensor is the
    /// same as that of the input.
    ///
    /// @param aScope Computation scope
    /// @param aOperand Operand
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Not( scope_t &aScope, graph_node_t const &aOperand );

    /// @brief Bitwise conjunction of two integer nodes
    ///
    /// At least one of `aLeft` and `aRight` should be a @ref MultiTensor nodes. If both operands are @ref MultiTensors,
    /// then they should have the same shape. If one of `aLeft` and `aRight` is a vector, then its length should match
    /// the number of layers of the other operand (which has to be a tensor). The dimension of the output tensor is the
    /// same as that of the input.
    ///
    /// @param aScope Computation scope
    /// @param aLeft Left operand
    /// @param aRight Right operand
    ///
    /// @return The newly created computation node
    ///
    graph_node_t BitwiseAnd( scope_t &aScope, graph_node_t const &aLeft, graph_node_t const &aRight );

    /// @brief Bitwise disjunction of two integer nodes
    ///
    /// At least one of `aLeft` and `aRight` should be a @ref MultiTensor nodes. If both operands are @ref MultiTensors,
    /// then they should have the same shape. If one of `aLeft` and `aRight` is a vector, then its length should match
    /// the number of layers of the other operand (which has to be a tensor). The dimension of the output tensor is the
    /// same as that of the input.
    ///
    /// @param aScope Computation scope
    /// @param aLeft Left operand
    /// @param aRight Right operand
    ///
    /// @return The newly created computation node
    ///
    graph_node_t BitwiseOr( scope_t &aScope, graph_node_t const &aLeft, graph_node_t const &aRight );

    /// @brief Bitwise negation of two integer nodes
    ///
    /// The parameter `aOperand` should be a @ref MultiTensor node. The dimension of the output tensor is the
    /// same as that of the input.
    ///
    /// @param aScope Computation scope
    /// @param aOperand Operand
    ///
    /// @return The newly created computation node
    ///
    graph_node_t BitwiseNot( scope_t &aScope, graph_node_t const &aOperand );

    /// @brief Test whether the values contained in a tensor lie within an interval
    ///
    /// The parameter `aX` should be a @ref MultiTensor node. The dimension of the output tensor is the
    /// same as that of the input.
    ///
    /// @param aScope Computation scope
    /// @param aX Operand
    /// @param aLower Lower bound for the interval
    /// @param aUpper Upper bound for the interval
    /// @param aStrictLower Use strict inequality for the lower bound
    /// @param aStrictUpper Use strict inequality for the lower bound
    ///
    /// @return The newly created computation node
    ///
    graph_node_t InInterval( scope_t &aScope, graph_node_t const &aX, graph_node_t const &aLower, graph_node_t const &aUpper, bool aStrictLower,
                       bool aStrictUpper );

    /// @brief Equality
    ///
    /// At least one of `aX` and `aY` should be a @ref MultiTensor nodes. If both operands are @ref MultiTensors,
    /// then they should have the same shape. If one of `aLeft` and `aRight` is a vector, then its length should match
    /// the number of layers of the other operand (which has to be a tensor). The dimension of the output tensor is the
    /// same as that of the input.
    ///
    /// @param aScope Computation scope
    /// @param aX Left operand
    /// @param aY Right operand
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Equal( scope_t &aScope, graph_node_t const &aX, graph_node_t const &aY );

    /// @brief Less than
    ///
    /// At least one of `aX` and `aY` should be a @ref MultiTensor nodes. If both operands are @ref MultiTensors,
    /// then they should have the same shape. If one of `aLeft` and `aRight` is a vector, then its length should match
    /// the number of layers of the other operand (which has to be a tensor). The dimension of the output tensor is the
    /// same as that of the input.
    ///
    /// @param aScope Computation scope
    /// @param aX Left operand
    /// @param aY Right operand
    ///
    /// @return The newly created computation node
    ///
    graph_node_t LessThan( scope_t &aScope, graph_node_t const &aX, graph_node_t const &aY );

    /// @brief Less than or equal to
    ///
    /// At least one of `aX` and `aY` should be a @ref MultiTensor nodes. If both operands are @ref MultiTensors,
    /// then they should have the same shape. If one of `aLeft` and `aRight` is a vector, then its length should match
    /// the number of layers of the other operand (which has to be a tensor). The dimension of the output tensor is the
    /// same as that of the input.
    ///
    /// @param aScope Computation scope
    /// @param aX Left operand
    /// @param aY Right operand
    ///
    /// @return The newly created computation node
    ///
    graph_node_t LessThanOrEqual( scope_t &aScope, graph_node_t const &aX, graph_node_t const &aY );

    /// @brief Greater than
    ///
    /// At least one of `aX` and `aY` should be a @ref MultiTensor nodes. If both operands are @ref MultiTensors,
    /// then they should have the same shape. If one of `aLeft` and `aRight` is a vector, then its length should match
    /// the number of layers of the other operand (which has to be a tensor). The dimension of the output tensor is the
    /// same as that of the input.
    ///
    /// @param aScope Computation scope
    /// @param aX Left operand
    /// @param aY Right operand
    ///
    /// @return The newly created computation node
    ///
    graph_node_t GreaterThan( scope_t &aScope, graph_node_t const &aX, graph_node_t const &aY );

    /// @brief Greater than or equal to
    ///
    /// At least one of `aX` and `aY` should be a @ref MultiTensor nodes. If both operands are @ref MultiTensors,
    /// then they should have the same shape. If one of `aLeft` and `aRight` is a vector, then its length should match
    /// the number of layers of the other operand (which has to be a tensor). The dimension of the output tensor is the
    /// same as that of the input.
    ///
    /// @param aScope Computation scope
    /// @param aX Left operand
    /// @param aY Right operand
    ///
    /// @return The newly created computation node
    ///
    graph_node_t GreaterThanOrEqual( scope_t &aScope, graph_node_t const &aX, graph_node_t const &aY );

    /// @brief Choose values from one tensor or another based on a given condition
    ///
    /// The parameter `aCondition` should be a @ref MultiTensor nodes. If all operands are @ref MultiTensors,
    /// then they should have the same shape. If one of `aValueIfTrue` and `aValueIfFalse` is a vector, then its length should match
    /// the number of layers of the other operand (which has to be a tensor). The dimension of the output tensor is the
    /// same as that of `aCondition`.
    ///
    /// @param aScope Computation scope
    /// @param aCondition Condition to test
    /// @param aValueIfTrue Value to use if condition is true
    /// @param aValueIfFalse Value to use if condition is false
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Where( scope_t &aScope, graph_node_t const &aCondition, graph_node_t const &aValueIfTrue, graph_node_t const &aValueIfFalse );

    /// @brief Computes the pointwise mix of two tensors
    ///
    /// All of `aA`, `aB` and `aT` should be @ref MultiTensors of the same shape and type. This function computes the tensor
    /// @f$ (1-t)\cdot A + t\cdot B @f$, the shape of which is the same as the common shape of `aA`, `aB` and `aT`
    ///
    /// @param aScope Parent computation scope
    /// @param aArray Array to repeat
    /// @param aA Input tensor
    /// @param aB Input tensor
    /// @param aT Input tensor
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Mix( scope_t &aScope, graph_node_t const &aA, graph_node_t const &aB, graph_node_t const &aT );

    /// @brief Affine transforms
    ///
    /// The parameter `aX` has to be a @ref MultiTensor. The other two parameters can be any vectors, or scalars. This node
    /// computes the affine transformation @f$ a\cdot X+b @f$, the output of which has the same shape as `aX`. If either of
    /// `aA` or `aB` is a @ref MultiTensor, then is should have the same shape as `aX`, and is either one is a vector, then
    /// its length should match the number of layers of `aX`.
    ///
    /// @param aScope Parent computation scope
    /// @param aArray Array to repeat
    /// @param aA Input tensor
    /// @param aB Input tensor
    /// @param aT Input tensor
    ///
    /// @return The newly created computation node
    ///
    graph_node_t AffineTransform( scope_t &aScope, graph_node_t const &aA, graph_node_t const &aX, graph_node_t const &aB );

    /// @brief Computes a set of ranges of values with a regular step.
    ///
    /// The two vectors contained in `aLeft` and `aRight` should have the same length and contain floating
    /// point values. This is roughly a generazed version of numpy's `np.arange`. The resulting output tensor
    /// will have rank 1, with one layer for every element of `aLeft`. Each layer will have dimension
    /// @f$ (R - L) / \Delta @f$
    ///
    /// @param aScope Parent computation scope
    /// @param aLeft Lower bounds
    /// @param aRight Upper bounds
    /// @param aDelta Range step
    ///
    /// @return The newly created computation node
    ///
    graph_node_t ARange( scope_t &aScope, graph_node_t const &aLeft, graph_node_t const &aRight, graph_node_t const &aDelta );

    /// @brief Computes evenly spaced numbers in the intervals specified by two tensors
    ///
    /// Roughly equivalent to numpy's np.linspace. The two input tensors should have the same shape. The interval between them
    /// is subdivided into `aRepetitions` many subintervals, where each element in `aRepetitions` is matched with the corresponding
    /// layer of the input tensors. If the input multi-tensor have rank @f$ N @f$ , then the output multi-tensor will have rank
    /// @f$ N+1 @f$. The last dimension of the output multi-tensor is the number of subdivisions.
    ///
    /// @param aScope Parent computation scope
    /// @param aArray Array to repeat
    /// @param aRepetitions Nummber of repetitions
    /// @param aOut Output tensor.
    ///
    /// @return The newly created computation node
    ///
    graph_node_t LinearSpace( scope_t &aScope, graph_node_t const &aLeft, graph_node_t const &aRight, graph_node_t const &aSubdivisions );

    /// @brief Repeat each element of a multi-tensor.
    ///
    /// Roughly equivalent to numpy's npo.repeat. Each element of the innermost dimension of a multitensor is repeated a given
    /// number of times. Node that a different number of repetitions can be specified for each layer of the multi-tensor. As
    /// far as dimension and rank are concerned, if the input multi-tensor has rank @f$ N @f$ , then, the repeated multi-tensor
    /// will have rank @f$ N+1 @f$. The last dimension of the output multi-tensor is the number of repetitions.
    ///
    /// @param aScope Parent computation scope
    /// @param aArray Array to repeat
    /// @param aRepetitions Nummber of repetitions
    /// @param aOut Output tensor.
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Repeat( scope_t &aScope, graph_node_t const &aArray, graph_node_t const &aRepetitions );

    /// @brief Repeat each layer of a multi-tensor.
    ///
    /// Roughly equivalent to numpy's npo.tile. Each layer of the a multitensor is repeated a given number of times. Node that
    /// a different number of repetitions can be specified for each layer of the multi-tensor. As far as dimension and rank are
    /// concerned, if the input multi-tensor has rank @f$ N @f$ , then, the repeated multi-tensor will have rank @f$ N+1 @f$.
    /// The first dimension of the output multi-tensor is the number of repetitions.
    ///
    /// @param aScope Parent computation scope
    /// @param aArray Array to repeat
    /// @param aRepetitions Nummber of repetitions
    /// @param aOut Output tensor.
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Tile( scope_t &aScope, graph_node_t const &aArray, graph_node_t const &aRepetitions );

    /// @brief Texture sampling
    ///
    /// Samples the textures in `aTextures` at the coordinates specified by `aX` and `aY`. The tensors `aX` and `aY`
    /// should have the same shape, which will be the output shape. `aTextures` should represent a vector of
    /// @ref sTextureData whose length matches the number of layers in the tensors `aX` and `aY`. Each layer of the
    /// output will be sampled from the corresponding texture in `aTestures`
    ///
    /// @param aScope Parent computation scope
    /// @param aArray Array to repeat
    /// @param aRepetitions Nummber of repetitions
    /// @param aOut Output tensor.
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Sample2D( scope_t &aScope, graph_node_t const &aX, graph_node_t const &aY, graph_node_t const &aTextures );

    /// @brief Fixed point conversion
    ///
    /// Converts a tensor of floating point numbers into a tensor of integers by multiplying each element by a scaling factor.
    ///
    /// @param aScope Parent computation scope
    /// @param aArray Array to repeat
    /// @param aRepetitions Nummber of repetitions
    /// @param aOut Output tensor.
    ///
    /// @return The newly created computation node
    ///
    graph_node_t ToFixedPoint( scope_t &aScope, scalar_type_t aOutputType, graph_node_t const &aArray, graph_node_t const &aScaling );

    /// @brief Collapse a @ref MultiTensor into a @ref MultiTensor having only one layer
    ///
    /// The dimensions of each layer of the @ref MultiTensor should be equal. Furthermore, the memory area is shared between the
    /// input and the output multitensors, so that there is no actual copying involved.
    ///
    /// @param aScope Parent computation scope
    /// @param aArray MultiTensor to collapse
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Collapse( scope_t &aScope, graph_node_t const &aArray );

    /// @brief Expand the first dimension of a @ref MultiTensor with only one layer into a multi-layered MultiTensor
    ///
    /// The number of layers in the output @ref MultiTensor is equal to the first dimension of the input @ref MultiTensor.
    /// Furthermore, the memory area is shared between the input and the output multitensors, so that there is no actual copying
    /// involved.
    ///
    /// @param aScope Parent computation scope
    /// @param aArray MultiTensor to expand
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Expand( scope_t &aScope, graph_node_t const &aArray );

    /// @brief Reshape the input @ref MultiTensor
    ///
    /// The new shape should bave the same number of layers as the original shape. Furthermore, the dimensions of each layer of
    /// the old and new shapes should be compatible in having equal products, and equal element sizes. The memory is shared between
    /// the input and the output tensors, so that no copying is involved.
    ///
    /// @param aScope Parent computation scope
    /// @param aArray MultiTensor to reshape
    /// @param aNewShape New shape for the output tensor
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Reshape( scope_t &aScope, graph_node_t const &aArray, Cuda::tensor_shape_t &aNewShape );

    /// @brief Relayout the input @ref MultiTensor
    ///
    /// Applies a new layout to the input MultiTensor. The new layout should have the same size as the old layout.
    ///
    /// @param aScope Parent computation scope
    /// @param aArray MultiTensor to relayout
    /// @param aNewLayout New layout
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Relayout( scope_t &aScope, graph_node_t const &aArray, tensor_shape_t &aNewLayout );

    /// @brief Flatten the input @ref MultiTensor
    ///
    /// The output multi-tensor will have the same number of layers as the input, but it will have rank 1, and the dimension of each
    /// layer will be the product of the corresponding layer in the input.
    ///
    /// @param aScope Parent computation scope
    /// @param aArray MultiTensor to flatten
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Flatten( scope_t &aScope, graph_node_t const &aArray );

    /// @brief Slice the input @ref MultiTensor
    ///
    /// The output multi-tensor will have the same number of layers as the input. The nodes `aBegin` and `aEnd` denote the start and
    /// end indices of the slice respectively, and should correspond to either a scalar value, or a vector whose entries are uint32_t.
    /// Entries at `aBegin` and `aEnd` are included in the slice. For now, slicing a multi-tensor only acts on the last dimension.
    ///
    /// @param aScope Parent computation scope
    /// @param aArray MultiTensor to slice
    /// @param aBegin Lower bound
    /// @param aEnd Upper bound
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Slice( scope_t &aScope, graph_node_t const &aArray, graph_node_t const &aBegin, graph_node_t const &aEnd );

    /// @brief Sum the last dimension of the input @ref MultiTensor
    ///
    /// The output multi-tensor will have the same number of layers as the input. For now, summing a multi-tensor only considers
    /// the last dimension.
    ///
    /// @param aScope Parent computation scope
    /// @param aArray MultiTensor to sum
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Summation( scope_t &aScope, graph_node_t const &aArray );

    /// @brief Sum the last dimension of the input @ref MultiTensor
    ///
    /// The output multi-tensor will have the same number of layers as the input. The nodes `aBegin` and `aEnd` denote the start and
    /// end indices of the sum respectively, and should correspond to either a scalar value, or a vector whose entries are uint32_t.
    /// Entries at `aBegin` and `aEnd` are included in the sum. For now, summing a multi-tensor only considers the last dimension.
    ///
    /// @param aScope Parent computation scope
    /// @param aArray MultiTensor to flatten
    /// @param aBegin Lower bound
    /// @param aEnd Upper bound
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Summation( scope_t &aScope, graph_node_t const &aArray, graph_node_t const &aBegin, graph_node_t const &aEnd );

    /// @brief Count the number of `true` elements in the last dimension of @ref MultiTensor
    ///
    /// The output multi-tensor will have the same number of layers as the input. For now, counting the true values in a multi-tensor
    /// only considers the last dimension.
    ///
    /// @param aScope Parent computation scope
    /// @param aArray MultiTensor to process
    ///
    /// @return The newly created computation node
    ///
    graph_node_t CountTrue( scope_t &aScope, graph_node_t const &aArray );

    /// @brief Count the number of non-zero elements in the last dimension of @ref MultiTensor
    ///
    /// The output multi-tensor will have the same number of layers as the input. For now, counting the non-zero values of a
    /// multi-tensor only considers the last dimension.
    ///
    /// @param aScope Parent computation scope
    /// @param aArray MultiTensor to process
    ///
    /// @return The newly created computation node
    ///
    graph_node_t CountNonZero( scope_t &aScope, graph_node_t const &aArray );

    /// @brief Count the number of zero elements in the last dimension of @ref MultiTensor
    ///
    /// The output multi-tensor will have the same number of layers as the input. For now, counting the zero values of a multi-tensor
    /// only considers the last dimension.
    ///
    /// @param aScope Parent computation scope
    /// @param aArray MultiTensor to process
    ///
    /// @return The newly created computation node
    ///
    graph_node_t CountZero( scope_t &aScope, graph_node_t const &aArray );

    /// @brief Compute the pointwise floor of @ref MultiTensor
    ///
    /// The output multi-tensor will have the same dimension as the input.
    ///
    /// @param aScope Parent computation scope
    /// @param aArray MultiTensor to process
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Floor( scope_t &aScope, graph_node_t const &aArray );

    /// @brief Compute the pointwise ceiling of @ref MultiTensor
    ///
    /// The output multi-tensor will have the same dimension as the input.
    ///
    /// @param aScope Parent computation scope
    /// @param aArray MultiTensor to process
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Ceil( scope_t &aScope, graph_node_t const &aArray );

    /// @brief Compute the pointwise absolute value of @ref MultiTensor
    ///
    /// The output multi-tensor will have the same dimension as the input.
    ///
    /// @param aScope Parent computation scope
    /// @param aArray MultiTensor to process
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Abs( scope_t &aScope, graph_node_t const &aArray );

    /// @brief Compute the pointwise square root value of @ref MultiTensor
    ///
    /// The output multi-tensor will have the same dimension as the input.
    ///
    /// @param aScope Parent computation scope
    /// @param aArray MultiTensor to process
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Sqrt( scope_t &aScope, graph_node_t const &aArray );

    /// @brief Compute the pointwise rounded value of @ref MultiTensor
    ///
    /// The output multi-tensor will have the same dimension as the input.
    ///
    /// @param aScope Parent computation scope
    /// @param aArray MultiTensor to process
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Round( scope_t &aScope, graph_node_t const &aArray );

    /// @brief Compute the iterated finite difference along the last dimension of @ref MultiTensor
    ///
    /// The output multi-tensor will have the same dimension as the input. The final entries of the output
    /// tensor are set to 0
    ///
    /// @param aScope Parent computation scope
    /// @param aArray MultiTensor to process
    /// @param aCount Number of iterations
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Diff( scope_t &aScope, graph_node_t const &aArray, uint32_t aCount );

    /// @brief Compute the finite shift along the last dimension of @ref MultiTensor
    ///
    /// The output multi-tensor will have the same dimension as the input. The final entries of the output
    /// tensor are set to aFillValue.
    ///
    /// @param aScope Parent computation scope
    /// @param aArray MultiTensor to process
    /// @param aCount Number of images to shift
    /// @param aFillValue Value used to fill the missing positions
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Shift( scope_t &aScope, graph_node_t const &aArray, int32_t aCount, graph_node_t const &aFillValue );

    /// @brief Compute the 1-dimensional convolution the last dimension of @ref MultiTensor
    ///
    /// The output multi-tensor will have the same dimension as the left.
    ///
    /// @param aScope Parent computation scope
    /// @param aArray0 MultiTensor to process
    /// @param aArray1 Convolution kernel
    ///
    /// @return The newly created computation node
    ///
    graph_node_t Conv1D( scope_t &aScope, graph_node_t const &aArray0, graph_node_t const &aArray1 );

    /// @brief Concatenate the given @ref MultiTensors along the last dimension
    ///
    /// @param aScope Parent computation scope
    /// @param aArray0 MultiTensor to process
    /// @param aArray1 MultiTensor to process
    ///
    /// @return The newly created computation node
    ///
    graph_node_t HCat( scope_t &aScope, graph_node_t const &aArray0, graph_node_t const &aArray1 );

} // namespace SE::TensorOps
