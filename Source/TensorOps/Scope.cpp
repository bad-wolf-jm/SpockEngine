/// @file   Scope.cpp
///
/// @brief  Definitions for computation scope
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert. All rights reserved.

#include "Scope.h"

#include "Core/CUDA/Texture/Texture2D.h"

#include "Implementation/KernelLaunchers.h"

namespace numlua::mtops
{
    using namespace numlua::cuda;

    scope_t::scope_t( uint32_t memorySize )
    {
        mPool = memory_pool_t( memorySize );
    }

    scope_t &scope_t::WithOpName( const string_t &name )
    {
        _name = name;
        return *this;
    }

    graph_node_t scope_t::CreateNode()
    {
        graph_node_t new_entity;
        if( _name.has_value() )
        {
            new_entity = _nodes_registry.CreateEntity( _name.value() );

            _named_nodes[_name.value()] = new_entity;
            _name.reset();
        }
        else
        {
            new_entity = _nodes_registry.CreateEntity();
        }

        new_entity.Add<node_id_t>();

        return new_entity;
    }

    graph_node_t scope_t::operator[]( const string_t &node_name )
    {
        if( _named_nodes.find( node_name ) != _named_nodes.end() )
            return _named_nodes[node_name];

        return graph_node_t{};
    }

    void scope_t::Reset()
    {
        mPool.Reset();

        _nodes_registry.Clear();
        _named_nodes.clear();
        _name.reset();
    }

    void scope_t::Run( graph_node_t const &node )
    {
        Run( vector_t<graph_node_t>{ node } );
    }

    void scope_t::Run( vector_t<graph_node_t> const &node )
    {
        std::deque<graph_node_t>                         execution_queue;
        std::stack<graph_node_t, vector_t<graph_node_t>> stack( node );

        while( !stack.empty() )
        {
            graph_node_t current = stack.top();
            stack.pop();

            if( current.Get<node_id_t>().do_not_expand )
                continue;

            std::deque<graph_node_t>::iterator pos = std::find( execution_queue.begin(), execution_queue.end(), current );
            if( pos != execution_queue.end() )
            {
                execution_queue.erase( pos );
            }
            execution_queue.push_back( current );
            if( current.Has<operand_t>() )
            {
                for( graph_node_t dependent : current.Get<operand_t>().mOperands )
                    stack.push( dependent );
            }
        }

        // Allocate memory for tensors which are on the stack
        for( auto element = execution_queue.rbegin(); element < execution_queue.rend(); element++ )
        {
            if( ( *element ).Get<node_id_t>().is_allocated )
                continue;

            if( ( *element ).Has<multi_tensor_value_t>() )
            {
                ( *element ).Get<multi_tensor_value_t>().value =
                    multi_tensor_t( mPool, ( *element ).Get<multi_tensor_value_t>().shape );
                ( *element ).Get<node_id_t>().is_allocated = true;
            }

            if( ( *element ).Has<vector_buffer_t>() )
            {
                ( *element ).Get<vector_buffer_t>().value  = mPool.Allocate( ( *element ).Get<vector_buffer_t>().size );
                ( *element ).Get<node_id_t>().is_allocated = true;
            }
        }

        for( auto element = execution_queue.rbegin(); element < execution_queue.rend(); element++ )
        {
            if( !( *element ).Has<graph_operation_t>() )
                continue;

            auto &component = ( *element ).Get<graph_operation_t>();
            if( !component.controller_instance )
            {
                component.controller_instance = component.mInstantiateController();
                component.controller_instance->Initialize( *element );
            }

            component.controller_instance->Run();
        }
        SyncDevice();
    }

    graph_node_t CreateMultiTensor( scope_t &scope, tensor_shape_t const &shape )
    {
        auto new_entity = scope.CreateNode();
        new_entity.Add<multi_tensor_value_t>( scope.mPool, shape );
        new_entity.Add<graph_operation_t>().Bind<sMultiTensorRunner>();

        return new_entity;
    }

    graph_node_t MultiTensorValue( scope_t &scope, constant_value_initializer_t const &initializer, tensor_shape_t const &shape )
    {
        auto new_entity = CreateMultiTensor( scope, shape );

        new_entity.Get<node_id_t>().element_type = type_of( initializer.value );
        new_entity.Add<constant_value_initializer_t>( initializer );

        return new_entity;
    }

    graph_node_t MultiTensorValue( scope_t &scope, vector_initializer_t const &initializer, tensor_shape_t const &shape )
    {
        auto new_entity = CreateMultiTensor( scope, shape );

        new_entity.Get<node_id_t>().element_type = type_of( initializer.value[0] );
        auto &initializer_component              = new_entity.Add<vector_initializer_t>( initializer );
        initializer_component.data = scope.mPool.Allocate( initializer.value.size() * size_of( type_of( initializer.value[0] ) ) );

        return new_entity;
    }

    graph_node_t MultiTensorValue( scope_t &scope, data_initializer_t const &initializer, tensor_shape_t const &shape )
    {
        auto new_entity = CreateMultiTensor( scope, shape );

        new_entity.Get<node_id_t>().element_type = type_of( initializer.value[0] );
        auto &initializer_component              = new_entity.Add<data_initializer_t>( initializer );

        return new_entity;
    }

    graph_node_t MultiTensorValue( scope_t &scope, random_uniform_initializer_t const &initializer, tensor_shape_t const &shape )
    {
        auto new_entity = CreateMultiTensor( scope, shape );

        new_entity.Get<node_id_t>().element_type = initializer.type;
        new_entity.Add<random_uniform_initializer_t>( initializer );

        return new_entity;
    }

    graph_node_t MultiTensorValue( scope_t &scope, random_normal_initializer_t const &initializer, tensor_shape_t const &shape )
    {
        auto new_entity = CreateMultiTensor( scope, shape );

        new_entity.Get<node_id_t>().element_type = initializer.type;
        new_entity.Add<random_normal_initializer_t>( initializer );

        return new_entity;
    }

    static inline bool SameType( graph_node_t const &left, graph_node_t const &right )
    {
        return ( left.Get<node_id_t>().element_type == right.Get<node_id_t>().element_type );
    }

    static inline bool SameShape( graph_node_t const &left, graph_node_t const &right )
    {
        return ( left.Get<multi_tensor_value_t>().Shape() == right.Get<multi_tensor_value_t>().Shape() );
    }

    template <typename T>
    static inline bool SameLength( graph_node_t left, graph_node_t const &right )
    {
        return ( left.Get<vector_value_t<T>>().value.size() == right.Get<vector_value_t<T>>().value.size() );
    }

    graph_node_t BinaryOperation( scope_t &scope, scalar_type_t type, graph_node_t const &left, graph_node_t const &right )
    {
        assert( ( left.HasAny<multi_tensor_value_t, scalar_value_vector_t, scalar_node_t>() ) );
        assert( ( right.HasAny<multi_tensor_value_t, scalar_value_vector_t, scalar_node_t>() ) );

        auto  new_entity   = scope.CreateNode();
        auto &operand_data = new_entity.Add<binary_operation_t>( binary_operation_t{ left, right } );

        if( operand_data.left.Has<multi_tensor_value_t>() )
        {
            if( operand_data.right.Has<multi_tensor_value_t>() )
            {
                auto left_shape  = operand_data.left.Get<multi_tensor_value_t>().Shape();
                auto right_shape = operand_data.right.Get<multi_tensor_value_t>().Shape();

                if( left_shape != right_shape )
                {
                    if( left_shape.Rank == right_shape.Rank - 1 )
                    {
                        right_shape.Trim( -1 );
                        if( right_shape != left_shape )
                            throw std::runtime_error( "Can only add tensors of the same shape" );

                        auto &broadcast_info = new_entity.Add<broadcast_info_t>();
                        right_shape.Flatten( 0 );
                        broadcast_info.mBroadcastHint = broadcast_hint_t::LEFT;
                        broadcast_info.max_block_size = right_shape.MaxDimensions[0];
                        broadcast_info.block_sizes    = VectorValue( scope, right_shape.GetDimension( 0 ) );

                        auto broadcast_shape                  = operand_data.right.Get<multi_tensor_value_t>().Shape();
                        broadcast_info.mBroadcastDimension    = VectorValue( scope, broadcast_shape.GetDimension( -1 ) );
                        broadcast_info.mMaxBroadcastDimension = broadcast_shape.MaxDimensions[broadcast_shape.Rank - 1];

                        new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( broadcast_shape.Shape, size_of( type ) ) );
                    }
                    else if( right_shape.Rank == left_shape.Rank - 1 )
                    {
                        left_shape.Trim( -1 );
                        if( right_shape != left_shape )
                            throw std::runtime_error( "Can only add tensors of the same shape" );

                        auto &broadcast_info = new_entity.Add<broadcast_info_t>();
                        right_shape.Flatten( 0 );
                        broadcast_info.mBroadcastHint = broadcast_hint_t::RIGHT;
                        broadcast_info.max_block_size = right_shape.MaxDimensions[0];
                        broadcast_info.block_sizes    = VectorValue( scope, right_shape.GetDimension( 0 ) );

                        auto broadcast_shape                  = operand_data.left.Get<multi_tensor_value_t>().Shape();
                        broadcast_info.mBroadcastDimension    = VectorValue( scope, broadcast_shape.GetDimension( -1 ) );
                        broadcast_info.mMaxBroadcastDimension = broadcast_shape.MaxDimensions[broadcast_shape.Rank - 1];

                        new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( broadcast_shape.Shape, size_of( type ) ) );
                    }
                    else
                    {
                        throw std::runtime_error( "Can only add tensors of the same shape" );
                    }
                }
                else
                {
                    auto shape = operand_data.left.Get<multi_tensor_value_t>().Shape().Shape;
                    new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( shape, size_of( type ) ) );
                }
            }
            else
            {
                auto shape = operand_data.left.Get<multi_tensor_value_t>().Shape().Shape;
                new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( shape, size_of( type ) ) );
            }
        }
        else
        {
            if( !( operand_data.right.Has<multi_tensor_value_t>() ) )
            {
                throw std::runtime_error( "RHS should have a tensor" );
            }

            auto shape = operand_data.right.Get<multi_tensor_value_t>().Shape().Shape;
            new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( shape, size_of( type ) ) );
        }

        if( new_entity.Has<broadcast_info_t>() )
            new_entity.Add<operand_t>( vector_t<graph_node_t>{ left, right, new_entity.Get<broadcast_info_t>().block_sizes,
                                                               new_entity.Get<broadcast_info_t>().mBroadcastDimension } );
        else
            new_entity.Add<operand_t>( vector_t<graph_node_t>{ left, right } );

        new_entity.Get<node_id_t>().element_type = type;

        return new_entity;
    }

    graph_node_t BinaryOperation( scope_t &scope, graph_node_t const &left, graph_node_t const &right )
    {
        return BinaryOperation( scope, left.Get<node_id_t>().element_type, left, right );
    }

    graph_node_t Add( scope_t &scope, graph_node_t const &left, graph_node_t const &right )
    {
        auto new_entity = BinaryOperation( scope, left, right );
        new_entity.Add<graph_operation_t>().Bind<sAddOperationController>();

        return new_entity;
    }

    graph_node_t Subtract( scope_t &scope, graph_node_t const &left, graph_node_t const &right )
    {
        auto new_entity = BinaryOperation( scope, left, right );
        new_entity.Add<graph_operation_t>().Bind<sSubtractOperationController>();

        return new_entity;
    }

    graph_node_t Divide( scope_t &scope, graph_node_t const &left, graph_node_t const &right )
    {
        auto new_entity = BinaryOperation( scope, left, right );
        new_entity.Add<graph_operation_t>().Bind<sDivideOperationController>();

        return new_entity;
    }

    graph_node_t Multiply( scope_t &scope, graph_node_t const &left, graph_node_t const &right )
    {
        auto new_entity = BinaryOperation( scope, left, right );
        new_entity.Add<graph_operation_t>().Bind<sMultiplyOperationController>();

        return new_entity;
    }

    graph_node_t And( scope_t &scope, graph_node_t const &left, graph_node_t const &right )
    {
        assert( SameType( left, right ) );
        assert( left.Get<node_id_t>().element_type == scalar_type_t::UINT8 );

        auto new_entity = BinaryOperation( scope, left, right );
        new_entity.Add<graph_operation_t>().Bind<sAndOperationController>();

        return new_entity;
    }

    graph_node_t Or( scope_t &scope, graph_node_t const &left, graph_node_t const &right )
    {
        assert( SameType( left, right ) );
        assert( left.Get<node_id_t>().element_type == scalar_type_t::UINT8 );

        auto new_entity = BinaryOperation( scope, left, right );
        new_entity.Add<graph_operation_t>().Bind<sOrOperationController>();

        return new_entity;
    }

    graph_node_t Not( scope_t &scope, graph_node_t const &operand )
    {
        assert( operand.Get<node_id_t>().element_type == scalar_type_t::UINT8 );

        auto  new_entity   = scope.CreateNode();
        auto &operand_data = new_entity.Add<not_operation_t>( not_operation_t{ operand } );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ operand } );
        new_entity.Get<node_id_t>().element_type = operand.Get<node_id_t>().element_type;
        new_entity.Add<multi_tensor_value_t>( scope.mPool, operand_data.operand.Get<multi_tensor_value_t>().Shape() );
        new_entity.Add<graph_operation_t>().Bind<sNotOperationController>();

        return new_entity;
    }

    graph_node_t BitwiseAnd( scope_t &scope, graph_node_t const &left, graph_node_t const &right )
    {
        assert( SameType( left, right ) );
        assert( ( left.Get<node_id_t>().element_type >= scalar_type_t::UINT8 ) &&
                ( left.Get<node_id_t>().element_type <= scalar_type_t::INT64 ) );

        auto new_entity = BinaryOperation( scope, left, right );
        new_entity.Add<graph_operation_t>().Bind<sBitwiseAndOperationController>();

        return new_entity;
    }

    graph_node_t BitwiseOr( scope_t &scope, graph_node_t const &left, graph_node_t const &right )
    {
        assert( SameType( left, right ) );
        assert( ( left.Get<node_id_t>().element_type >= scalar_type_t::UINT8 ) &&
                ( left.Get<node_id_t>().element_type <= scalar_type_t::INT64 ) );

        auto new_entity = BinaryOperation( scope, left, right );
        new_entity.Add<graph_operation_t>().Bind<sBitwiseOrOperationController>();

        return new_entity;
    }

    graph_node_t BitwiseNot( scope_t &scope, graph_node_t const &operand )
    {
        assert( ( operand.Get<node_id_t>().element_type >= scalar_type_t::UINT8 ) &&
                ( operand.Get<node_id_t>().element_type <= scalar_type_t::INT64 ) );

        auto  new_entity   = scope.CreateNode();
        auto &operand_data = new_entity.Add<bitwise_not_operation_t>( bitwise_not_operation_t{ operand } );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ operand } );
        new_entity.Get<node_id_t>().element_type = operand.Get<node_id_t>().element_type;
        new_entity.Add<multi_tensor_value_t>( scope.mPool, operand_data.operand.Get<multi_tensor_value_t>().Shape() );
        new_entity.Add<graph_operation_t>().Bind<sBitwiseNotOperationController>();

        return new_entity;
    }

    graph_node_t InInterval( scope_t &scope, graph_node_t const &x, graph_node_t const &lower, graph_node_t const &upper,
                             bool strictLower, bool strictUpper )
    {
        assert( ( x.Has<multi_tensor_value_t>() ) );
        assert( ( lower.HasAny<multi_tensor_value_t, scalar_value_vector_t, scalar_node_t>() ) );
        assert( ( upper.HasAny<multi_tensor_value_t, scalar_value_vector_t, scalar_node_t>() ) );

        assert( SameType( x, lower ) );
        assert( SameType( x, upper ) );

        auto  new_entity = scope.CreateNode();
        auto &operand_data =
            new_entity.Add<in_interval_operation_t>( in_interval_operation_t{ x, lower, upper, strictLower, strictUpper } );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ x, lower, upper } );
        new_entity.Get<node_id_t>().element_type = scalar_type_t::UINT8;

        vector_t<vector_t<uint32_t>> output_shape = x.Get<multi_tensor_value_t>().Shape().Shape;

        new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( output_shape, size_of( scalar_type_t::UINT8 ) ) );
        new_entity.Add<graph_operation_t>().Bind<sInIntervalOperationController>();

        return new_entity;
    }

    graph_node_t Equal( scope_t &scope, graph_node_t const &x, graph_node_t const &y )
    {
        assert( SameType( x, y ) );

        auto new_entity = BinaryOperation( scope, scalar_type_t::UINT8, x, y );
        new_entity.Add<graph_operation_t>().Bind<sEqualOperationController>();

        return new_entity;
    }

    graph_node_t LessThan( scope_t &scope, graph_node_t const &x, graph_node_t const &y )
    {
        assert( SameType( x, y ) );

        auto new_entity = BinaryOperation( scope, scalar_type_t::UINT8, x, y );
        new_entity.Add<graph_operation_t>().Bind<sLessThanOperationController>();

        return new_entity;
    }

    graph_node_t LessThanOrEqual( scope_t &scope, graph_node_t const &x, graph_node_t const &y )
    {
        assert( SameType( x, y ) );

        auto new_entity = BinaryOperation( scope, scalar_type_t::UINT8, x, y );
        new_entity.Add<graph_operation_t>().Bind<sLessThanOrEqualOperationController>();

        return new_entity;
    }

    graph_node_t GreaterThan( scope_t &scope, graph_node_t const &x, graph_node_t const &y )
    {
        return LessThan( scope, y, x );
    }

    graph_node_t GreaterThanOrEqual( scope_t &scope, graph_node_t const &x, graph_node_t const &y )
    {
        return LessThanOrEqual( scope, y, x );
    }

    graph_node_t Where( scope_t &scope, graph_node_t const &condition, graph_node_t const &value_if_true,
                        graph_node_t const &value_if_false )
    {
        assert( ( value_if_true.HasAny<multi_tensor_value_t, scalar_value_vector_t, scalar_node_t>() ) );
        assert( ( value_if_false.HasAny<multi_tensor_value_t, scalar_value_vector_t, scalar_node_t>() ) );
        assert( ( condition.Get<node_id_t>().element_type == scalar_type_t::UINT8 ) );
        assert( SameType( value_if_true, value_if_false ) );

        auto new_entity = scope.CreateNode();
        new_entity.Add<where_operation_t>( where_operation_t{ condition, value_if_true, value_if_false } );
        new_entity.Get<node_id_t>().element_type = value_if_true.Get<node_id_t>().element_type;
        new_entity.Add<operand_t>( vector_t<graph_node_t>{ condition, value_if_true, value_if_false } );

        auto shape = condition.Get<multi_tensor_value_t>().Shape().Shape;
        new_entity.Add<multi_tensor_value_t>( scope.mPool,
                                              tensor_shape_t( shape, size_of( value_if_true.Get<node_id_t>().element_type ) ) );

        new_entity.Add<graph_operation_t>().Bind<sWhereOperationController>();

        return new_entity;
    }

    graph_node_t Mix( scope_t &scope, graph_node_t const &A, graph_node_t const &B, graph_node_t const &T )
    {
        assert( ( A.Has<multi_tensor_value_t>() ) );
        assert( ( B.Has<multi_tensor_value_t>() ) );
        assert( ( T.Has<multi_tensor_value_t>() ) );

        auto  new_entity   = scope.CreateNode();
        auto &operand_data = new_entity.Add<mix_operation_t>( mix_operation_t{ A, B, T } );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ A, B, T } );
        new_entity.Get<node_id_t>().element_type = A.Get<node_id_t>().element_type;
        new_entity.Add<multi_tensor_value_t>( scope.mPool, operand_data.A.Get<multi_tensor_value_t>().Shape() );
        new_entity.Add<graph_operation_t>().Bind<sMixOperationController>();

        return new_entity;
    }

    graph_node_t ToFixedPoint( scope_t &scope, scalar_type_t output_type, graph_node_t const &array, graph_node_t const &scaling )
    {
        assert( ( array.Has<multi_tensor_value_t>() ) );
        assert( ( scaling.Has<scalar_node_t>() ) );

        auto new_entity = scope.CreateNode();

        auto &operand_data = new_entity.Add<convert_to_fixed_point_t>( convert_to_fixed_point_t{ output_type, array, scaling } );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array, scaling } );

        auto &shape = operand_data.array.Get<multi_tensor_value_t>().Shape().Shape;
        new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( shape, size_of( output_type ) ) );

        new_entity.Add<graph_operation_t>().Bind<sToFixedPointOperationController>();

        return new_entity;
    }

    graph_node_t Repeat( scope_t &scope, graph_node_t const &array, graph_node_t const &repetitions )
    {
        assert( ( array.Has<multi_tensor_value_t>() ) );
        assert( repetitions.Has<u32_vector_t>() );

        auto  new_entity   = scope.CreateNode();
        auto &operand_data = new_entity.Add<repeat_operation_t>( repeat_operation_t{ array, repetitions } );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array, repetitions } );
        new_entity.Get<node_id_t>().element_type = array.Get<node_id_t>().element_type;

        auto &input_shape = operand_data.operand.Get<multi_tensor_value_t>().Shape();

        auto                         repetitionsValue = repetitions.Get<u32_vector_t>().value;
        vector_t<vector_t<uint32_t>> output_shape( input_shape.CountLayers() );
        for( uint32_t i = 0; i < input_shape.CountLayers(); i++ )
        {
            output_shape[i] = vector_t<uint32_t>( input_shape.Shape[i].size() + 1 );
            for( uint32_t j = 0; j < input_shape.Shape[i].size(); j++ )
            {
                output_shape[i][j] = input_shape.Shape[i][j];
            }
            output_shape[i][output_shape[i].size() - 1] = repetitionsValue[i];
        }

        new_entity.Add<multi_tensor_value_t>( scope.mPool,
                                              tensor_shape_t( output_shape, size_of( array.Get<node_id_t>().element_type ) ) );
        new_entity.Add<graph_operation_t>().Bind<sArrayOperationController>();

        return new_entity;
    }

    graph_node_t Tile( scope_t &scope, graph_node_t const &array, graph_node_t const &repetitions )
    {
        assert( ( array.Has<multi_tensor_value_t>() ) );
        assert( repetitions.Has<u32_vector_t>() );

        auto  new_entity   = scope.CreateNode();
        auto &operand_data = new_entity.Add<tile_operation_t>( tile_operation_t{ array, repetitions } );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array, repetitions } );
        // new_entity.Add<type_t>( array.Get<type_t>() );
        new_entity.Get<node_id_t>().element_type = array.Get<node_id_t>().element_type;
        auto &input_shape                        = operand_data.operand.Get<multi_tensor_value_t>().Shape();

        auto                         repetitionsValue = repetitions.Get<u32_vector_t>().value;
        vector_t<vector_t<uint32_t>> output_shape( input_shape.CountLayers() );
        for( uint32_t i = 0; i < input_shape.CountLayers(); i++ )
        {
            output_shape[i]                             = vector_t<uint32_t>( input_shape.Shape[i].size() + 1 );
            output_shape[i][output_shape[i].size() - 1] = repetitionsValue[i];

            output_shape[i][0] = repetitionsValue[i];

            for( uint32_t j = 0; j < input_shape.Shape[i].size(); j++ )
            {
                output_shape[i][j + 1] = input_shape.Shape[i][j];
            }
        }

        new_entity.Add<multi_tensor_value_t>( scope.mPool,
                                              tensor_shape_t( output_shape, size_of( array.Get<node_id_t>().element_type ) ) );
        new_entity.Add<graph_operation_t>().Bind<sArrayOperationController>();

        return new_entity;
    }

    graph_node_t ARange( scope_t &scope, graph_node_t const &left, graph_node_t const &right, graph_node_t const &delta )
    {
        assert( left.Has<scalar_value_vector_t>() && right.Has<scalar_value_vector_t>() && delta.Has<scalar_value_vector_t>() );

        assert( SameType( left, right ) );
        assert( SameType( left, delta ) );
        assert( SameLength<scalar_value_t>( left, right ) );
        assert( SameLength<scalar_value_t>( left, delta ) );

        assert( ( left.Get<node_id_t>().element_type == scalar_type_t::FLOAT32 ) ||
                ( left.Get<node_id_t>().element_type == scalar_type_t::FLOAT64 ) );

        auto  new_entity   = scope.CreateNode();
        auto &operand_data = new_entity.Add<arange_operation_t>( arange_operation_t{ left, right, delta } );

        new_entity.Get<node_id_t>().element_type = left.Get<node_id_t>().element_type;
        new_entity.Add<operand_t>( vector_t<graph_node_t>{ left, right, delta } );

        vector_t<vector_t<uint32_t>> output_shape( left.Get<scalar_value_vector_t>().value.size() );
        auto                         left_values  = left.Get<scalar_value_vector_t>().value;
        auto                         right_values = right.Get<scalar_value_vector_t>().value;
        auto                         delta_values = delta.Get<scalar_value_vector_t>().value;

        for( uint32_t i = 0; i < left.Get<scalar_value_vector_t>().value.size(); i++ )
        {
            output_shape[i] = { static_cast<uint32_t>( std::ceil(
                ( std::get<float>( right_values[i] ) - std::get<float>( left_values[i] ) ) / std::get<float>( delta_values[i] ) ) ) };
        }

        new_entity.Add<multi_tensor_value_t>( scope.mPool,
                                              tensor_shape_t( output_shape, size_of( left.Get<node_id_t>().element_type ) ) );
        new_entity.Add<graph_operation_t>().Bind<sARangeOperationController>();

        return new_entity;
    }

    graph_node_t LinearSpace( scope_t &scope, graph_node_t const &left, graph_node_t const &right, graph_node_t const &subdivisions )
    {
        assert( left.Has<multi_tensor_value_t>() && right.Has<multi_tensor_value_t>() && subdivisions.Has<u32_vector_t>() );

        assert( SameShape( left, right ) );

        assert( left.Get<multi_tensor_value_t>().Shape().CountLayers() == subdivisions.Get<u32_vector_t>().value.size() );
        assert( SameType( left, right ) );
        assert( ( left.Get<node_id_t>().element_type == scalar_type_t::FLOAT32 ) ||
                ( left.Get<node_id_t>().element_type == scalar_type_t::FLOAT64 ) );

        auto  new_entity   = scope.CreateNode();
        auto &operand_data = new_entity.Add<linear_space_operation_t>( linear_space_operation_t{ left, right, subdivisions } );

        new_entity.Get<node_id_t>().element_type = left.Get<node_id_t>().element_type;
        new_entity.Add<operand_t>( vector_t<graph_node_t>{ left, right, subdivisions } );

        auto                         subdivisions_value = subdivisions.Get<u32_vector_t>().value;
        vector_t<vector_t<uint32_t>> output_shape( left.Get<multi_tensor_value_t>().Shape().Shape.size() );

        for( uint32_t i = 0; i < left.Get<multi_tensor_value_t>().Shape().Shape.size(); i++ )
        {
            output_shape[i] = left.Get<multi_tensor_value_t>().Shape().Shape[i];
            output_shape[i].push_back( subdivisions_value[i] );
        }

        new_entity.Add<multi_tensor_value_t>( scope.mPool,
                                              tensor_shape_t( output_shape, size_of( left.Get<node_id_t>().element_type ) ) );
        new_entity.Add<graph_operation_t>().Bind<sLinearSpaceOperationController>();

        return new_entity;
    }

    graph_node_t Sample2D( scope_t &scope, graph_node_t const &x, graph_node_t const &y, graph_node_t const &textures )
    {
        assert( ( x.HasAny<multi_tensor_value_t, scalar_node_t, scalar_value_vector_t>() ) );
        assert( ( y.HasAny<multi_tensor_value_t, scalar_node_t, scalar_value_vector_t>() ) );

        assert( x.Has<multi_tensor_value_t>() || y.Has<multi_tensor_value_t>() );
        assert( textures.Has<vector_value_t<cuda::texture_sampler2d_t::DeviceData>>() );

        if( x.Has<multi_tensor_value_t>() && y.Has<multi_tensor_value_t>() )
            assert( SameShape( x, y ) );

        if( x.Has<multi_tensor_value_t>() )
            assert( x.Get<multi_tensor_value_t>().Shape().CountLayers() ==
                    textures.Get<vector_value_t<cuda::texture_sampler2d_t::DeviceData>>().value.size() );

        if( y.Has<multi_tensor_value_t>() )
            assert( y.Get<multi_tensor_value_t>().Shape().CountLayers() ==
                    textures.Get<vector_value_t<cuda::texture_sampler2d_t::DeviceData>>().value.size() );

        assert( SameType( x, y ) );

        assert( x.Get<node_id_t>().element_type == scalar_type_t::FLOAT32 );

        auto  new_entity   = scope.CreateNode();
        auto &operand_data = new_entity.Add<sample2D_operation_t>( sample2D_operation_t{ x, y, textures } );

        new_entity.Get<node_id_t>().element_type = x.Get<node_id_t>().element_type;
        new_entity.Add<operand_t>( vector_t<graph_node_t>{ x, y, textures } );
        new_entity.Add<multi_tensor_value_t>( scope.mPool, operand_data.x.Get<multi_tensor_value_t>().Shape() );
        new_entity.Add<graph_operation_t>().Bind<sSample2DOperationController>();

        return new_entity;
    }

    graph_node_t AffineTransform( scope_t &scope, graph_node_t const &A, graph_node_t const &x, graph_node_t const &B )
    {
        assert( ( x.HasAll<multi_tensor_value_t>() ) );
        assert( ( A.HasAny<multi_tensor_value_t, scalar_node_t, scalar_value_vector_t>() ) );
        assert( ( B.HasAny<multi_tensor_value_t, scalar_node_t, scalar_value_vector_t>() ) );
        assert( SameType( x, A ) );
        assert( SameType( x, B ) );

        auto  new_entity   = scope.CreateNode();
        auto &operand_data = new_entity.Add<affine_transform_operation_t>( affine_transform_operation_t{ A, x, B } );
        new_entity.Get<node_id_t>().element_type = x.Get<node_id_t>().element_type;

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ A, x, B } );
        new_entity.Add<multi_tensor_value_t>( scope.mPool, operand_data.X.Get<multi_tensor_value_t>().Shape() );
        new_entity.Add<graph_operation_t>().Bind<sAffineNodeController>();

        return new_entity;
    }

    graph_node_t Collapse( scope_t &scope, graph_node_t const &array )
    {
        assert( ( array.Has<multi_tensor_value_t>() ) );

        auto new_entity = scope.CreateNode();

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array } );

        auto &input_shape = array.Get<multi_tensor_value_t>().Shape();
        for( uint32_t i = 0; i < input_shape.Shape.size(); i++ )
        {
            if( input_shape.Shape[i] != input_shape.Shape[0] )
                throw std::runtime_error( "All dimensions should be equal" );
        }

        vector_t<uint32_t> output_dimension( input_shape.Rank + 1 );
        output_dimension[0] = input_shape.CountLayers();
        std::copy( input_shape.Shape[0].begin(), input_shape.Shape[0].end(), output_dimension.begin() + 1 );

        new_entity.Get<node_id_t>().element_type = array.Get<node_id_t>().element_type;
        new_entity.Add<multi_tensor_value_t>(
            scope.mPool, array.Get<multi_tensor_value_t>().value.GetMemoryBuffer(),
            tensor_shape_t( vector_t<vector_t<uint32_t>>{ output_dimension }, static_cast<size_t>( input_shape.ElementSize ) ) );

        return new_entity;
    }

    graph_node_t Expand( scope_t &scope, graph_node_t const &array )
    {
        auto new_entity = scope.CreateNode();
        assert( ( array.Has<multi_tensor_value_t>() ) );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array } );

        auto &input_shape = array.Get<multi_tensor_value_t>().Shape();

        assert( input_shape.CountLayers() == 1 );

        vector_t<vector_t<uint32_t>> output_shape( input_shape.Shape[0][0] );

        for( uint32_t i = 0; i < output_shape.size(); i++ )
        {
            output_shape[i] = vector_t<uint32_t>( input_shape.Rank - 1 );
            std::copy( input_shape.Shape[0].begin() + 1, input_shape.Shape[0].end(), output_shape[i].begin() );
        }

        new_entity.Get<node_id_t>().element_type = array.Get<node_id_t>().element_type;
        new_entity.Add<multi_tensor_value_t>( scope.mPool, array.Get<multi_tensor_value_t>().value.GetMemoryBuffer(),
                                              tensor_shape_t( output_shape, static_cast<size_t>( input_shape.ElementSize ) ) );

        return new_entity;
    }

    graph_node_t Reshape( scope_t &scope, graph_node_t const &array, tensor_shape_t &new_shape )
    {
        auto new_entity = scope.CreateNode();
        assert( ( array.Has<multi_tensor_value_t>() ) );

        auto &input_shape = array.Get<multi_tensor_value_t>().Shape();

        assert( input_shape.CountLayers() == new_shape.CountLayers() );
        assert( input_shape.ElementSize == new_shape.ElementSize );

        for( uint32_t i = 0; i < input_shape.CountLayers(); i++ )
        {
            uint32_t size0 =
                std::accumulate( input_shape.Shape[i].begin(), input_shape.Shape[i].end(), 1, std::multiplies<uint32_t>() );
            uint32_t size1 = std::accumulate( new_shape.Shape[i].begin(), new_shape.Shape[i].end(), 1, std::multiplies<uint32_t>() );

            if( size0 != size1 )
                throw std::runtime_error( "Incompatible dimensions" );
        }

        new_entity.Get<node_id_t>().element_type = array.Get<node_id_t>().element_type;
        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array } );
        new_entity.Add<multi_tensor_value_t>( scope.mPool, array.Get<multi_tensor_value_t>().value.GetMemoryBuffer(), new_shape );

        return new_entity;
    }

    graph_node_t Relayout( scope_t &scope, graph_node_t const &array, tensor_shape_t &new_layout )
    {
        auto new_entity = scope.CreateNode();
        assert( ( array.Has<multi_tensor_value_t>() ) );

        auto &input_shape = array.Get<multi_tensor_value_t>().Shape();

        assert( input_shape.ElementSize == new_layout.ElementSize );

        uint32_t inputSize = 0;
        for( uint32_t i = 0; i < input_shape.CountLayers(); i++ )
            inputSize += std::accumulate( input_shape.Shape[i].begin(), input_shape.Shape[i].end(), 1, std::multiplies<uint32_t>() );

        uint32_t outputSize = 0;
        for( uint32_t i = 0; i < new_layout.CountLayers(); i++ )
            outputSize += std::accumulate( new_layout.Shape[i].begin(), new_layout.Shape[i].end(), 1, std::multiplies<uint32_t>() );

        if( inputSize != outputSize )
            throw std::runtime_error( "Incompatible dimensions" );

        new_entity.Get<node_id_t>().element_type = array.Get<node_id_t>().element_type;
        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array } );
        new_entity.Add<multi_tensor_value_t>( scope.mPool, array.Get<multi_tensor_value_t>().value.GetMemoryBuffer(), new_layout );

        return new_entity;
    }

    graph_node_t Flatten( scope_t &scope, graph_node_t const &array )
    {
        auto new_entity = scope.CreateNode();
        assert( ( array.Has<multi_tensor_value_t>() ) );

        auto input_shape = array.Get<multi_tensor_value_t>().Shape();

        input_shape.Flatten( 0 );
        new_entity.Get<node_id_t>().element_type = array.Get<node_id_t>().element_type;
        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array } );
        new_entity.Add<multi_tensor_value_t>( scope.mPool, array.Get<multi_tensor_value_t>().value.GetMemoryBuffer(), input_shape );

        return new_entity;
    }

    graph_node_t Slice( scope_t &scope, graph_node_t const &array, graph_node_t const &begin, graph_node_t const &end )
    {
        assert( ( array.Has<multi_tensor_value_t>() ) );
        assert( ( ( begin.HasAny<vector_value_t<uint32_t>, scalar_node_t>() ) ) &&
                ( end.HasAny<vector_value_t<uint32_t>, scalar_node_t>() ) );

        auto new_entity = scope.CreateNode();

        new_entity.Get<node_id_t>().element_type = array.Get<node_id_t>().element_type;
        auto &operand_data                       = new_entity.Add<array_slice_operation_t>();
        operand_data.array                       = array;

        auto input_shape = array.Get<multi_tensor_value_t>().Shape();

        uint32_t           maxBlockSize = 0;
        vector_t<uint32_t> blockSizes( input_shape.CountLayers() );

        vector_t<uint32_t> beginValue;
        if( begin.Has<vector_value_t<uint32_t>>() )
        {
            beginValue         = begin.Get<vector_value_t<uint32_t>>().value;
            operand_data.begin = begin;
        }
        else
        {
            beginValue = vector_t<uint32_t>( input_shape.CountLayers(), std::get<uint32_t>( begin.Get<scalar_node_t>().value ) );
            operand_data.begin = VectorValue( scope, beginValue );
        }

        vector_t<uint32_t> end_value;
        if( end.Has<vector_value_t<uint32_t>>() )
        {
            end_value        = end.Get<vector_value_t<uint32_t>>().value;
            operand_data.end = end;
        }
        else
        {
            end_value        = vector_t<uint32_t>( input_shape.CountLayers(), std::get<uint32_t>( end.Get<scalar_node_t>().value ) );
            operand_data.end = VectorValue( scope, end_value );
        }

        vector_t<vector_t<uint32_t>> output_shape( input_shape.CountLayers() );

        for( uint32_t i = 0; i < input_shape.CountLayers(); i++ )
        {
            output_shape[i] = vector_t<uint32_t>( input_shape.Shape[i].size() );
            for( uint32_t j = 0; j < input_shape.Shape[i].size() - 1; j++ )
            {
                output_shape[i][j] = input_shape.Shape[i][j];
            }
            output_shape[i][input_shape.Shape[i].size() - 1] =
                std::max( end_value[i] - beginValue[i] + 1, static_cast<uint32_t>( 0 ) );
        }

        input_shape.Flatten( -1 );
        operand_data.max_block_size = input_shape.MaxDimensions[0];
        operand_data.block_sizes    = VectorValue( scope, input_shape.GetDimension( 0 ) );
        operand_data.element_count  = VectorValue( scope, input_shape.GetDimension( -1 ) );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array, operand_data.begin, operand_data.end, operand_data.block_sizes,
                                                           operand_data.element_count } );

        new_entity.Add<multi_tensor_value_t>( scope.mPool,
                                              tensor_shape_t( output_shape, size_of( array.Get<node_id_t>().element_type ) ) );
        new_entity.Add<graph_operation_t>().Bind<sArraySliceOperationController>();

        return new_entity;
    }

    graph_node_t Summation( scope_t &scope, graph_node_t const &array )
    {
        auto               input_shape = array.Get<multi_tensor_value_t>().Shape();
        vector_t<uint32_t> last_dimensions( input_shape.CountLayers() );
        for( uint32_t i = 0; i < input_shape.CountLayers(); i++ )
            last_dimensions[i] = input_shape.Shape[i][input_shape.Shape[i].size() - 1] - 1;

        auto zero      = ConstantScalarValue( scope, static_cast<uint32_t>( 0 ) );
        auto end_value = VectorValue( scope, last_dimensions );

        return Summation( scope, array, zero, end_value );
    }

    graph_node_t Summation( scope_t &scope, graph_node_t const &array, graph_node_t const &begin, graph_node_t const &end )
    {
        assert( ( array.Has<multi_tensor_value_t>() ) );
        assert( ( ( begin.HasAny<vector_value_t<uint32_t>, scalar_node_t>() ) ) &&
                ( end.HasAny<vector_value_t<uint32_t>, scalar_node_t>() ) );

        auto new_entity = scope.CreateNode();

        new_entity.Get<node_id_t>().element_type = array.Get<node_id_t>().element_type;
        auto &operand_data                       = new_entity.Add<array_sum_operation_t>();
        operand_data.array                       = array;

        auto input_shape = array.Get<multi_tensor_value_t>().Shape();

        if( begin.Has<vector_value_t<uint32_t>>() )
        {
            operand_data.begin = begin;
        }
        else
        {
            vector_t<uint32_t> beginValue( input_shape.CountLayers(), std::get<uint32_t>( begin.Get<scalar_node_t>().value ) );
            operand_data.begin = VectorValue( scope, beginValue );
        }

        if( end.Has<vector_value_t<uint32_t>>() )
        {
            operand_data.end = end;
        }
        else
        {
            vector_t<uint32_t> end_value( input_shape.CountLayers(), std::get<uint32_t>( end.Get<scalar_node_t>().value ) );
            operand_data.end = VectorValue( scope, end_value );
        }

        auto output_shape = input_shape;
        output_shape.Trim( -1 );

        input_shape.Flatten( -1 );
        operand_data.max_block_size = input_shape.MaxDimensions[0];
        operand_data.block_sizes    = VectorValue( scope, input_shape.GetDimension( 0 ) );
        operand_data.element_count  = VectorValue( scope, input_shape.GetDimension( -1 ) );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array, operand_data.begin, operand_data.end, operand_data.block_sizes,
                                                           operand_data.element_count } );
        new_entity.Add<multi_tensor_value_t>( scope.mPool,
                                              tensor_shape_t( output_shape.Shape, size_of( array.Get<node_id_t>().element_type ) ) );
        new_entity.Add<graph_operation_t>().Bind<sArraySummationOperationController>();

        return new_entity;
    }

    graph_node_t CountTrue( scope_t &scope, graph_node_t const &array )
    {
        assert( ( array.Has<multi_tensor_value_t>() ) );

        auto new_entity = scope.CreateNode();

        new_entity.Get<node_id_t>().element_type = scalar_type_t::UINT32;
        auto &operand_data = new_entity.Add<count_true_operation_t>();
        operand_data.array = array;

        auto input_shape  = array.Get<multi_tensor_value_t>().Shape();
        auto output_shape = input_shape;
        output_shape.Trim( -1 );

        input_shape.Flatten( -1 );
        operand_data.max_block_size = input_shape.MaxDimensions[0];
        operand_data.block_sizes    = VectorValue( scope, input_shape.GetDimension( 0 ) );
        operand_data.element_count  = VectorValue( scope, input_shape.GetDimension( -1 ) );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array, operand_data.block_sizes, operand_data.element_count } );
        new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( output_shape.Shape, size_of( scalar_type_t::UINT32 ) ) );
        new_entity.Add<graph_operation_t>().Bind<sCountTrueOperationController>();

        return new_entity;
    }

    graph_node_t CountZero( scope_t &scope, graph_node_t const &array )
    {
        assert( ( array.Has<multi_tensor_value_t>() ) );

        auto new_entity = scope.CreateNode();

        new_entity.Get<node_id_t>().element_type = scalar_type_t::UINT32;
        auto &operand_data = new_entity.Add<count_zero_operation_t>();
        operand_data.array = array;

        auto input_shape = array.Get<multi_tensor_value_t>().Shape();

        auto output_shape = input_shape;
        output_shape.Trim( -1 );

        input_shape.Flatten( -1 );
        operand_data.max_block_size = input_shape.MaxDimensions[0];
        operand_data.block_sizes    = VectorValue( scope, input_shape.GetDimension( 0 ) );
        operand_data.element_count  = VectorValue( scope, input_shape.GetDimension( -1 ) );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array, operand_data.block_sizes, operand_data.element_count } );
        new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( output_shape.Shape, size_of( scalar_type_t::UINT32 ) ) );
        new_entity.Add<graph_operation_t>().Bind<sCountZeroOperationController>();

        return new_entity;
    }

    graph_node_t CountNonZero( scope_t &scope, graph_node_t const &array )
    {
        assert( ( array.Has<multi_tensor_value_t>() ) );

        auto new_entity = scope.CreateNode();

        new_entity.Get<node_id_t>().element_type = scalar_type_t::UINT32;
        auto &operand_data = new_entity.Add<count_non_zero_operation_t>();
        operand_data.array = array;

        auto input_shape = array.Get<multi_tensor_value_t>().Shape();

        auto output_shape = input_shape;
        output_shape.Trim( -1 );

        input_shape.Flatten( -1 );
        operand_data.max_block_size = input_shape.MaxDimensions[0];
        operand_data.block_sizes    = VectorValue( scope, input_shape.GetDimension( 0 ) );
        operand_data.element_count  = VectorValue( scope, input_shape.GetDimension( -1 ) );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array, operand_data.block_sizes, operand_data.element_count } );
        new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( output_shape.Shape, size_of( scalar_type_t::UINT32 ) ) );
        new_entity.Add<graph_operation_t>().Bind<sCountNonZeroOperationController>();

        return new_entity;
    }

    graph_node_t Diff( scope_t &scope, graph_node_t const &array, uint32_t count )
    {
        assert( ( array.Has<multi_tensor_value_t>() ) );

        auto new_entity = scope.CreateNode();

        new_entity.Get<node_id_t>().element_type = array.Get<node_id_t>().element_type;
        auto &operand_data = new_entity.Add<diff_operation_t>();
        operand_data.array = array;
        operand_data.count = count;

        auto input_shape  = array.Get<multi_tensor_value_t>().Shape();
        auto output_shape = array.Get<multi_tensor_value_t>().Shape();

        input_shape.Flatten( -1 );
        operand_data.max_block_size = input_shape.MaxDimensions[0];
        operand_data.block_sizes    = VectorValue( scope, input_shape.GetDimension( 0 ) );
        operand_data.element_count  = VectorValue( scope, input_shape.GetDimension( -1 ) );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array, operand_data.block_sizes, operand_data.element_count } );
        new_entity.Add<multi_tensor_value_t>( scope.mPool, output_shape );
        new_entity.Add<graph_operation_t>().Bind<sDiffOperationController>();

        return new_entity;
    }

    graph_node_t Shift( scope_t &scope, graph_node_t const &array, int32_t count, graph_node_t const &fill_value )
    {
        assert( ( array.Has<multi_tensor_value_t>() ) );
        assert( ( fill_value.Has<scalar_node_t>() ) );
        assert( SameType( fill_value, array ) );

        auto new_entity = scope.CreateNode();

        new_entity.Get<node_id_t>().element_type = array.Get<node_id_t>().element_type;
        auto &operand_data      = new_entity.Add<shift_operation_t>();
        operand_data.array      = array;
        operand_data.count      = count;
        operand_data.fill_value = fill_value;

        auto input_shape  = array.Get<multi_tensor_value_t>().Shape();
        auto output_shape = array.Get<multi_tensor_value_t>().Shape();

        input_shape.Flatten( -1 );
        operand_data.max_block_size = input_shape.MaxDimensions[0];
        operand_data.block_sizes    = VectorValue( scope, input_shape.GetDimension( 0 ) );
        operand_data.element_count  = VectorValue( scope, input_shape.GetDimension( -1 ) );

        new_entity.Add<operand_t>(
            vector_t<graph_node_t>{ array, operand_data.fill_value, operand_data.block_sizes, operand_data.element_count } );
        new_entity.Add<multi_tensor_value_t>( scope.mPool, output_shape );
        new_entity.Add<graph_operation_t>().Bind<sShiftOperationController>();

        return new_entity;
    }

    graph_node_t Floor( scope_t &scope, graph_node_t const &array )
    {
        assert( ( array.Has<multi_tensor_value_t>() ) );
        assert( ( array.Get<node_id_t>().element_type == scalar_type_t::FLOAT32 ) );

        auto new_entity = scope.CreateNode();

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array } );
        new_entity.Get<node_id_t>().element_type = array.Get<node_id_t>().element_type;
        new_entity.Add<floor_operation_t>( floor_operation_t{ array } );

        auto shape = array.Get<multi_tensor_value_t>().Shape().Shape;
        new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( shape, size_of( array.Get<node_id_t>().element_type ) ) );

        new_entity.Add<graph_operation_t>().Bind<sFloorOperationController>();

        return new_entity;
    }

    graph_node_t Ceil( scope_t &scope, graph_node_t const &array )
    {
        assert( ( array.Has<multi_tensor_value_t>() ) );
        assert( ( array.Get<node_id_t>().element_type == scalar_type_t::FLOAT32 ) );

        auto new_entity = scope.CreateNode();

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array } );
        new_entity.Get<node_id_t>().element_type = array.Get<node_id_t>().element_type;
        new_entity.Add<ceiling_operation_t>( ceiling_operation_t{ array } );

        auto shape = array.Get<multi_tensor_value_t>().Shape().Shape;
        new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( shape, size_of( array.Get<node_id_t>().element_type ) ) );

        new_entity.Add<graph_operation_t>().Bind<sCeilOperationController>();

        return new_entity;
    }

    graph_node_t Abs( scope_t &scope, graph_node_t const &array )
    {
        assert( ( array.Has<multi_tensor_value_t>() ) );
        assert( ( array.Get<node_id_t>().element_type == scalar_type_t::FLOAT32 ) ||
                ( ( array.Get<node_id_t>().element_type >= scalar_type_t::INT8 ) &&
                  ( array.Get<node_id_t>().element_type <= scalar_type_t::INT64 ) ) );

        auto new_entity = scope.CreateNode();

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array } );
        new_entity.Get<node_id_t>().element_type = array.Get<node_id_t>().element_type;
        new_entity.Add<abs_operation_t>( abs_operation_t{ array } );

        auto shape = array.Get<multi_tensor_value_t>().Shape().Shape;
        new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( shape, size_of( array.Get<node_id_t>().element_type ) ) );

        new_entity.Add<graph_operation_t>().Bind<sAbsOperationController>();

        return new_entity;
    }

    graph_node_t Sqrt( scope_t &scope, graph_node_t const &array )
    {
        assert( ( array.Has<multi_tensor_value_t>() ) );

        auto new_entity = scope.CreateNode();

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array } );
        new_entity.Get<node_id_t>().element_type = array.Get<node_id_t>().element_type;
        new_entity.Add<sqrt_operation_t>( sqrt_operation_t{ array } );

        auto shape = array.Get<multi_tensor_value_t>().Shape().Shape;
        new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( shape, size_of( array.Get<node_id_t>().element_type ) ) );

        new_entity.Add<graph_operation_t>().Bind<sSqrtOperationController>();

        return new_entity;
    }

    graph_node_t Round( scope_t &scope, graph_node_t const &array )
    {
        assert( ( array.Has<multi_tensor_value_t>() ) );

        auto new_entity = scope.CreateNode();

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array } );
        new_entity.Get<node_id_t>().element_type = array.Get<node_id_t>().element_type;
        new_entity.Add<round_operation_t>( round_operation_t{ array } );

        auto shape = array.Get<multi_tensor_value_t>().Shape().Shape;
        new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( shape, size_of( array.Get<node_id_t>().element_type ) ) );

        new_entity.Add<graph_operation_t>().Bind<sRoundOperationController>();

        return new_entity;
    }

    graph_node_t Conv1D( scope_t &scope, graph_node_t const &array0, graph_node_t const &array1 )
    {
        assert( ( array0.Has<multi_tensor_value_t>() ) );
        assert( ( array1.Has<multi_tensor_value_t>() ) );

        auto new_entity = scope.CreateNode();

        new_entity.Get<node_id_t>().element_type = array0.Get<node_id_t>().element_type;
        auto &operand_data  = new_entity.Add<conv1d_operation_t>();
        operand_data.array0 = array0;
        operand_data.array1 = array1;

        auto input_shape  = array0.Get<multi_tensor_value_t>().Shape();
        auto output_shape = array0.Get<multi_tensor_value_t>().Shape();

        input_shape.Flatten( -1 );
        operand_data.max_block_size0    = input_shape.MaxDimensions[0];
        operand_data.max_element_count0 = input_shape.MaxDimensions[input_shape.Rank - 1];
        operand_data.block_sizes0       = VectorValue( scope, input_shape.GetDimension( 0 ) );
        operand_data.element_count0     = VectorValue( scope, input_shape.GetDimension( -1 ) );

        auto kernelShape = array1.Get<multi_tensor_value_t>().Shape();

        kernelShape.Flatten( -1 );
        operand_data.max_block_size1 = kernelShape.MaxDimensions[0];
        operand_data.block_sizes1    = VectorValue( scope, kernelShape.GetDimension( 0 ) );
        operand_data.element_count1  = VectorValue( scope, kernelShape.GetDimension( -1 ) );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ operand_data.array0, operand_data.block_sizes0, operand_data.element_count0,
                                                           operand_data.array1, operand_data.block_sizes1,
                                                           operand_data.element_count1 } );
        new_entity.Add<multi_tensor_value_t>( scope.mPool, output_shape );
        new_entity.Add<graph_operation_t>().Bind<sConv1DOperationController>();

        return new_entity;
    }

    graph_node_t HCat( scope_t &scope, graph_node_t const &array0, graph_node_t const &array1 )
    {
        assert( ( array0.Has<multi_tensor_value_t>() ) );
        assert( ( array1.Has<multi_tensor_value_t>() ) );

        auto new_entity = scope.CreateNode();

        new_entity.Get<node_id_t>().element_type = array0.Get<node_id_t>().element_type;
        auto &operand_data  = new_entity.Add<hcat_operation_t>();
        operand_data.array0 = array0;
        operand_data.array1 = array1;

        auto input_shape_0 = array0.Get<multi_tensor_value_t>().Shape();
        auto input_shape_1 = array1.Get<multi_tensor_value_t>().Shape();

        auto               last_dim_0 = input_shape_0.GetDimension( -1 );
        auto               last_dim_1 = input_shape_1.GetDimension( -1 );
        vector_t<uint32_t> concatenated{};
        for( uint32_t i = 0; i < last_dim_0.size(); i++ )
            concatenated.push_back( last_dim_0[i] + last_dim_1[i] );

        auto output_shape = array0.Get<multi_tensor_value_t>().Shape();
        output_shape.Trim( -1 );
        output_shape.InsertDimension( -1, concatenated );

        auto block_shape = array0.Get<multi_tensor_value_t>().Shape();
        block_shape.Flatten( -1 );
        operand_data.max_block_size = block_shape.MaxDimensions[0];
        operand_data.block_sizes    = VectorValue( scope, block_shape.GetDimension( 0 ) );
        operand_data.element_count0 = VectorValue( scope, input_shape_0.GetDimension( -1 ) );
        operand_data.element_count1 = VectorValue( scope, input_shape_1.GetDimension( -1 ) );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ operand_data.array0, operand_data.array1, operand_data.block_sizes,
                                                           operand_data.element_count0, operand_data.element_count1 } );
        new_entity.Add<multi_tensor_value_t>( scope.mPool, output_shape );
        new_entity.Add<graph_operation_t>().Bind<sHCatOperationController>();

        return new_entity;
    }

} // namespace numlua::mtops
