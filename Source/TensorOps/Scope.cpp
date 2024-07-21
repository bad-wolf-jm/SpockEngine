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
            new_entity                 = _nodesRegistry.CreateEntity( _name.value() );
            _namedNodes[_name.value()] = new_entity;
            _name.reset();
        }
        else
        {
            new_entity = _nodesRegistry.CreateEntity();
        }

        return new_entity;
    }

    graph_node_t scope_t::operator[]( const string_t &nodeName )
    {
        if( _namedNodes.find( nodeName ) != _namedNodes.end() )
            return _namedNodes[nodeName];
        return graph_node_t{};
    }

    void scope_t::Reset()
    {
        mPool.Reset();

        _nodesRegistry.Clear();
        _namedNodes.clear();
        _name.reset();
    }

    void scope_t::Run( graph_node_t const &node )
    {
        Run( vector_t<graph_node_t>{ node } );
    }

    void scope_t::Run( vector_t<graph_node_t> const &node )
    {
        std::deque<graph_node_t>                         executionQueue;
        std::stack<graph_node_t, vector_t<graph_node_t>> stack( node );

        while( !stack.empty() )
        {
            graph_node_t current = stack.top();
            stack.pop();

            if( current.Has<do_not_expand_t>() )
                continue;

            std::deque<graph_node_t>::iterator lPos = std::find( executionQueue.begin(), executionQueue.end(), current );
            if( lPos != executionQueue.end() )
            {
                executionQueue.erase( lPos );
            }
            executionQueue.push_back( current );
            if( current.Has<operand_t>() )
            {
                for( graph_node_t lDependent : current.Get<operand_t>().mOperands )
                {
                    stack.push( lDependent );
                }
            }
        }

        // Allocate memory for tensors which are on the stack
        for( auto element = executionQueue.rbegin(); element < executionQueue.rend(); element++ )
        {
            if( ( *element ).Has<allocated_tag_t>() )
                continue;

            if( ( *element ).Has<multi_tensor_value_t>() )
            {
                ( *element ).Get<multi_tensor_value_t>().mValue =
                    multi_tensor_t( mPool, ( *element ).Get<multi_tensor_value_t>().mShape );
                ( *element ).Add<allocated_tag_t>();
            }

            if( ( *element ).Has<vector_buffer_t>() )
            {
                ( *element ).Get<vector_buffer_t>().mValue = mPool.Allocate( ( *element ).Get<vector_buffer_t>().mSize );
                ( *element ).Add<allocated_tag_t>();
            }
        }

        for( auto element = executionQueue.rbegin(); element < executionQueue.rend(); element++ )
        {
            if( !( *element ).Has<graph_operation_t>() )
                continue;

            auto &component = ( *element ).Get<graph_operation_t>();
            if( !component.mControllerInstance )
            {
                component.mControllerInstance = component.mInstantiateController();
                component.mControllerInstance->Initialize( *element );
            }

            component.mControllerInstance->Run();
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
        new_entity.Add<type_t>( type_of( initializer.mValue ) );
        new_entity.Add<constant_value_initializer_t>( initializer );

        return new_entity;
    }

    graph_node_t MultiTensorValue( scope_t &scope, vector_initializer_t const &initializer, tensor_shape_t const &shape )
    {
        auto new_entity = CreateMultiTensor( scope, shape );
        new_entity.Add<type_t>( type_of( initializer.mValue[0] ) );
        auto &lInitializerComponent = new_entity.Add<vector_initializer_t>( initializer );
        lInitializerComponent.mData = scope.mPool.Allocate( initializer.mValue.size() * size_of( type_of( initializer.mValue[0] ) ) );
        return new_entity;
    }

    graph_node_t MultiTensorValue( scope_t &scope, data_initializer_t const &initializer, tensor_shape_t const &shape )
    {
        auto new_entity = CreateMultiTensor( scope, shape );
        new_entity.Add<type_t>( type_of( initializer.mValue[0] ) );
        auto &lInitializerComponent = new_entity.Add<data_initializer_t>( initializer );
        return new_entity;
    }

    graph_node_t MultiTensorValue( scope_t &scope, random_uniform_initializer_t const &initializer, tensor_shape_t const &shape )
    {
        auto new_entity = CreateMultiTensor( scope, shape );
        new_entity.Add<type_t>( initializer.mType );
        new_entity.Add<random_uniform_initializer_t>( initializer );
        return new_entity;
    }

    graph_node_t MultiTensorValue( scope_t &scope, random_normal_initializer_t const &initializer, tensor_shape_t const &shape )
    {
        auto new_entity = CreateMultiTensor( scope, shape );
        new_entity.Add<type_t>( initializer.mType );
        new_entity.Add<random_normal_initializer_t>( initializer );
        return new_entity;
    }

    static inline bool SameType( graph_node_t const &left, graph_node_t const &right )
    {
        return ( left.Get<type_t>().mValue == right.Get<type_t>().mValue );
    }

    static inline bool SameShape( graph_node_t const &left, graph_node_t const &right )
    {
        return ( left.Get<multi_tensor_value_t>().Shape() == right.Get<multi_tensor_value_t>().Shape() );
    }

    template <typename T>
    static inline bool SameLength( graph_node_t left, graph_node_t const &right )
    {
        return ( left.Get<vector_value_t<T>>().mValue.size() == right.Get<vector_value_t<T>>().mValue.size() );
    }

    graph_node_t BinaryOperation( scope_t &scope, scalar_type_t type, graph_node_t const &left, graph_node_t const &right )
    {
        assert( left.Has<type_t>() );
        assert( right.Has<type_t>() );
        assert( ( left.HasAny<multi_tensor_value_t, scalar_value_vector_t, scalar_node_t>() ) );
        assert( ( right.HasAny<multi_tensor_value_t, scalar_value_vector_t, scalar_node_t>() ) );

        auto  new_entity  = scope.CreateNode();
        auto &operand_data = new_entity.Add<binary_operation_t>( binary_operation_t{ left, right } );

        if( operand_data.mLeftOperand.Has<multi_tensor_value_t>() )
        {
            if( operand_data.mRightOperand.Has<multi_tensor_value_t>() )
            {
                auto leftShape  = operand_data.mLeftOperand.Get<multi_tensor_value_t>().Shape();
                auto rightShape = operand_data.mRightOperand.Get<multi_tensor_value_t>().Shape();

                if( leftShape != rightShape )
                {
                    if( leftShape.Rank == rightShape.Rank - 1 )
                    {
                        rightShape.Trim( -1 );
                        if( rightShape != leftShape )
                            throw std::runtime_error( "Can only add tensors of the same shape" );

                        auto &broadcastInfo = new_entity.Add<broadcast_info_t>();
                        rightShape.Flatten( 0 );
                        broadcastInfo.mBroadcastHint = broadcast_hint_t::LEFT;
                        broadcastInfo.mMaxBlockSize  = rightShape.MaxDimensions[0];
                        broadcastInfo.mBlockSizes    = VectorValue( scope, rightShape.GetDimension( 0 ) );

                        auto broadcastShape                  = operand_data.mRightOperand.Get<multi_tensor_value_t>().Shape();
                        broadcastInfo.mBroadcastDimension    = VectorValue( scope, broadcastShape.GetDimension( -1 ) );
                        broadcastInfo.mMaxBroadcastDimension = broadcastShape.MaxDimensions[broadcastShape.Rank - 1];

                        new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( broadcastShape.Shape, size_of( type ) ) );
                    }
                    else if( rightShape.Rank == leftShape.Rank - 1 )
                    {
                        leftShape.Trim( -1 );
                        if( rightShape != leftShape )
                            throw std::runtime_error( "Can only add tensors of the same shape" );

                        auto &broadcastInfo = new_entity.Add<broadcast_info_t>();
                        rightShape.Flatten( 0 );
                        broadcastInfo.mBroadcastHint = broadcast_hint_t::RIGHT;
                        broadcastInfo.mMaxBlockSize  = rightShape.MaxDimensions[0];
                        broadcastInfo.mBlockSizes    = VectorValue( scope, rightShape.GetDimension( 0 ) );

                        auto broadcastShape                  = operand_data.mLeftOperand.Get<multi_tensor_value_t>().Shape();
                        broadcastInfo.mBroadcastDimension    = VectorValue( scope, broadcastShape.GetDimension( -1 ) );
                        broadcastInfo.mMaxBroadcastDimension = broadcastShape.MaxDimensions[broadcastShape.Rank - 1];

                        new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( broadcastShape.Shape, size_of( type ) ) );
                    }
                    else
                    {
                        throw std::runtime_error( "Can only add tensors of the same shape" );
                    }
                }
                else
                {
                    auto shape = operand_data.mLeftOperand.Get<multi_tensor_value_t>().Shape().Shape;
                    new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( shape, size_of( type ) ) );
                }
            }
            else
            {
                auto shape = operand_data.mLeftOperand.Get<multi_tensor_value_t>().Shape().Shape;
                new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( shape, size_of( type ) ) );
            }
        }
        else
        {
            if( !( operand_data.mRightOperand.Has<multi_tensor_value_t>() ) )
            {
                throw std::runtime_error( "RHS should have a tensor" );
            }

            auto shape = operand_data.mRightOperand.Get<multi_tensor_value_t>().Shape().Shape;
            new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( shape, size_of( type ) ) );
        }

        if( new_entity.Has<broadcast_info_t>() )
            new_entity.Add<operand_t>( vector_t<graph_node_t>{ left, right, new_entity.Get<broadcast_info_t>().mBlockSizes,
                                                               new_entity.Get<broadcast_info_t>().mBroadcastDimension } );
        else
            new_entity.Add<operand_t>( vector_t<graph_node_t>{ left, right } );

        new_entity.Add<type_t>( type );

        return new_entity;
    }

    graph_node_t BinaryOperation( scope_t &scope, graph_node_t const &left, graph_node_t const &right )
    {
        return BinaryOperation( scope, left.Get<type_t>().mValue, left, right );
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
        assert( ( left.Has<type_t>() ) && ( right.Has<type_t>() ) );
        assert( SameType( left, right ) );
        assert( left.Get<type_t>().mValue == scalar_type_t::UINT8 );

        auto new_entity = BinaryOperation( scope, left, right );
        new_entity.Add<graph_operation_t>().Bind<sAndOperationController>();

        return new_entity;
    }

    graph_node_t Or( scope_t &scope, graph_node_t const &left, graph_node_t const &right )
    {
        assert( ( left.Has<type_t>() ) && ( right.Has<type_t>() ) );
        assert( SameType( left, right ) );
        assert( left.Get<type_t>().mValue == scalar_type_t::UINT8 );

        auto new_entity = BinaryOperation( scope, left, right );
        new_entity.Add<graph_operation_t>().Bind<sOrOperationController>();

        return new_entity;
    }

    graph_node_t Not( scope_t &scope, graph_node_t const &operand )
    {
        assert( ( operand.Has<type_t>() ) );
        assert( operand.Get<type_t>().mValue == scalar_type_t::UINT8 );

        auto  new_entity  = scope.CreateNode();
        auto &operand_data = new_entity.Add<not_operation_t>( not_operation_t{ operand } );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ operand } );
        new_entity.Add<type_t>( operand.Get<type_t>() );
        new_entity.Add<multi_tensor_value_t>( scope.mPool, operand_data.mOperand.Get<multi_tensor_value_t>().Shape() );
        new_entity.Add<graph_operation_t>().Bind<sNotOperationController>();

        return new_entity;
    }

    graph_node_t BitwiseAnd( scope_t &scope, graph_node_t const &left, graph_node_t const &right )
    {
        assert( ( left.Has<type_t>() ) && ( right.Has<type_t>() ) );
        assert( SameType( left, right ) );
        assert( ( left.Get<type_t>().mValue >= scalar_type_t::UINT8 ) && ( left.Get<type_t>().mValue <= scalar_type_t::INT64 ) );

        auto new_entity = BinaryOperation( scope, left, right );
        new_entity.Add<graph_operation_t>().Bind<sBitwiseAndOperationController>();

        return new_entity;
    }

    graph_node_t BitwiseOr( scope_t &scope, graph_node_t const &left, graph_node_t const &right )
    {
        assert( ( left.Has<type_t>() ) && ( right.Has<type_t>() ) );
        assert( SameType( left, right ) );
        assert( ( left.Get<type_t>().mValue >= scalar_type_t::UINT8 ) && ( left.Get<type_t>().mValue <= scalar_type_t::INT64 ) );

        auto new_entity = BinaryOperation( scope, left, right );
        new_entity.Add<graph_operation_t>().Bind<sBitwiseOrOperationController>();

        return new_entity;
    }

    graph_node_t BitwiseNot( scope_t &scope, graph_node_t const &operand )
    {
        assert( ( operand.Has<type_t>() ) );
        assert( ( operand.Get<type_t>().mValue >= scalar_type_t::UINT8 ) && ( operand.Get<type_t>().mValue <= scalar_type_t::INT64 ) );

        auto  new_entity  = scope.CreateNode();
        auto &operand_data = new_entity.Add<bitwise_not_operation_t>( bitwise_not_operation_t{ operand } );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ operand } );
        new_entity.Add<type_t>( operand.Get<type_t>() );
        new_entity.Add<multi_tensor_value_t>( scope.mPool, operand_data.mOperand.Get<multi_tensor_value_t>().Shape() );
        new_entity.Add<graph_operation_t>().Bind<sBitwiseNotOperationController>();

        return new_entity;
    }

    graph_node_t InInterval( scope_t &scope, graph_node_t const &x, graph_node_t const &lower, graph_node_t const &upper,
                             bool strictLower, bool strictUpper )
    {
        assert( x.Has<type_t>() && ( lower.Has<type_t>() ) && ( upper.Has<type_t>() ) );
        assert( ( x.Has<multi_tensor_value_t>() ) );
        assert( ( lower.HasAny<multi_tensor_value_t, scalar_value_vector_t, scalar_node_t>() ) );
        assert( ( upper.HasAny<multi_tensor_value_t, scalar_value_vector_t, scalar_node_t>() ) );

        assert( SameType( x, lower ) );
        assert( SameType( x, upper ) );

        auto  new_entity = scope.CreateNode();
        auto &operand_data =
            new_entity.Add<in_interval_operation_t>( in_interval_operation_t{ x, lower, upper, strictLower, strictUpper } );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ x, lower, upper } );
        new_entity.Add<type_t>( scalar_type_t::UINT8 );

        vector_t<vector_t<uint32_t>> output_shape = x.Get<multi_tensor_value_t>().Shape().Shape;

        new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( output_shape, size_of( scalar_type_t::UINT8 ) ) );
        new_entity.Add<graph_operation_t>().Bind<sInIntervalOperationController>();

        return new_entity;
    }

    graph_node_t Equal( scope_t &scope, graph_node_t const &x, graph_node_t const &y )
    {
        assert( ( x.Has<type_t>() ) && ( y.Has<type_t>() ) );
        assert( SameType( x, y ) );

        auto new_entity = BinaryOperation( scope, scalar_type_t::UINT8, x, y );
        new_entity.Add<graph_operation_t>().Bind<sEqualOperationController>();

        return new_entity;
    }

    graph_node_t LessThan( scope_t &scope, graph_node_t const &x, graph_node_t const &y )
    {
        assert( ( x.Has<type_t>() ) && ( y.Has<type_t>() ) );
        assert( SameType( x, y ) );

        auto new_entity = BinaryOperation( scope, scalar_type_t::UINT8, x, y );
        new_entity.Add<graph_operation_t>().Bind<sLessThanOperationController>();

        return new_entity;
    }

    graph_node_t LessThanOrEqual( scope_t &scope, graph_node_t const &x, graph_node_t const &y )
    {
        assert( ( x.Has<type_t>() ) && ( y.Has<type_t>() ) );
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

    graph_node_t Where( scope_t &scope, graph_node_t const &condition, graph_node_t const &valueIfTrue,
                        graph_node_t const &valueIfFalse )
    {
        assert( ( condition.Has<type_t>() ) && ( valueIfTrue.Has<type_t>() ) && ( valueIfFalse.Has<type_t>() ) );
        assert( ( valueIfTrue.HasAny<multi_tensor_value_t, scalar_value_vector_t, scalar_node_t>() ) );
        assert( ( valueIfFalse.HasAny<multi_tensor_value_t, scalar_value_vector_t, scalar_node_t>() ) );
        assert( ( condition.Get<type_t>().mValue == scalar_type_t::UINT8 ) );
        assert( SameType( valueIfTrue, valueIfFalse ) );

        auto new_entity = scope.CreateNode();
        new_entity.Add<where_operation_t>( where_operation_t{ condition, valueIfTrue, valueIfFalse } );
        new_entity.Add<type_t>( valueIfTrue.Get<type_t>() );
        new_entity.Add<operand_t>( vector_t<graph_node_t>{ condition, valueIfTrue, valueIfFalse } );

        auto shape = condition.Get<multi_tensor_value_t>().Shape().Shape;
        new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( shape, size_of( valueIfTrue.Get<type_t>().mValue ) ) );

        new_entity.Add<graph_operation_t>().Bind<sWhereOperationController>();

        return new_entity;
    }

    graph_node_t Mix( scope_t &scope, graph_node_t const &A, graph_node_t const &B, graph_node_t const &T )
    {
        assert( ( A.HasAll<type_t, multi_tensor_value_t>() ) );
        assert( ( B.HasAll<type_t, multi_tensor_value_t>() ) );
        assert( ( T.HasAll<type_t, multi_tensor_value_t>() ) );

        auto  new_entity  = scope.CreateNode();
        auto &operand_data = new_entity.Add<mix_operation_t>( mix_operation_t{ A, B, T } );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ A, B, T } );
        new_entity.Add<type_t>( A.Get<type_t>() );
        new_entity.Add<multi_tensor_value_t>( scope.mPool, operand_data.mA.Get<multi_tensor_value_t>().Shape() );
        new_entity.Add<graph_operation_t>().Bind<sMixOperationController>();

        return new_entity;
    }

    graph_node_t ToFixedPoint( scope_t &scope, scalar_type_t outputType, graph_node_t const &array, graph_node_t const &scaling )
    {
        assert( ( array.HasAll<type_t, multi_tensor_value_t>() ) );
        assert( ( scaling.HasAll<type_t, scalar_node_t>() ) );

        auto new_entity = scope.CreateNode();

        auto &operand_data = new_entity.Add<convert_to_fixed_point_t>( convert_to_fixed_point_t{ outputType, array, scaling } );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array, scaling } );

        auto &shape = operand_data.mArray.Get<multi_tensor_value_t>().Shape().Shape;
        new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( shape, size_of( outputType ) ) );

        new_entity.Add<graph_operation_t>().Bind<sToFixedPointOperationController>();

        return new_entity;
    }

    graph_node_t Repeat( scope_t &scope, graph_node_t const &array, graph_node_t const &repetitions )
    {
        assert( ( array.HasAll<type_t, multi_tensor_value_t>() ) );
        assert( repetitions.Has<u32_vector_t>() );

        auto  new_entity  = scope.CreateNode();
        auto &operand_data = new_entity.Add<repeat_operation_t>( repeat_operation_t{ array, repetitions } );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array, repetitions } );
        new_entity.Add<type_t>( array.Get<type_t>() );

        auto &inputShape = operand_data.mArray.Get<multi_tensor_value_t>().Shape();

        auto                         repetitionsValue = repetitions.Get<u32_vector_t>().mValue;
        vector_t<vector_t<uint32_t>> output_shape( inputShape.CountLayers() );
        for( uint32_t i = 0; i < inputShape.CountLayers(); i++ )
        {
            output_shape[i] = vector_t<uint32_t>( inputShape.Shape[i].size() + 1 );
            for( uint32_t j = 0; j < inputShape.Shape[i].size(); j++ )
            {
                output_shape[i][j] = inputShape.Shape[i][j];
            }
            output_shape[i][output_shape[i].size() - 1] = repetitionsValue[i];
        }

        new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( output_shape, size_of( array.Get<type_t>().mValue ) ) );
        new_entity.Add<graph_operation_t>().Bind<sArrayOperationController>();

        return new_entity;
    }

    graph_node_t Tile( scope_t &scope, graph_node_t const &array, graph_node_t const &repetitions )
    {
        assert( ( array.HasAll<type_t, multi_tensor_value_t>() ) );
        assert( repetitions.Has<u32_vector_t>() );

        auto  new_entity  = scope.CreateNode();
        auto &operand_data = new_entity.Add<tile_operation_t>( tile_operation_t{ array, repetitions } );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array, repetitions } );
        new_entity.Add<type_t>( array.Get<type_t>() );
        auto &inputShape = operand_data.mArray.Get<multi_tensor_value_t>().Shape();

        auto                         repetitionsValue = repetitions.Get<u32_vector_t>().mValue;
        vector_t<vector_t<uint32_t>> output_shape( inputShape.CountLayers() );
        for( uint32_t i = 0; i < inputShape.CountLayers(); i++ )
        {
            output_shape[i]                            = vector_t<uint32_t>( inputShape.Shape[i].size() + 1 );
            output_shape[i][output_shape[i].size() - 1] = repetitionsValue[i];

            output_shape[i][0] = repetitionsValue[i];

            for( uint32_t j = 0; j < inputShape.Shape[i].size(); j++ )
            {
                output_shape[i][j + 1] = inputShape.Shape[i][j];
            }
        }

        new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( output_shape, size_of( array.Get<type_t>().mValue ) ) );
        new_entity.Add<graph_operation_t>().Bind<sArrayOperationController>();

        return new_entity;
    }

    graph_node_t ARange( scope_t &scope, graph_node_t const &left, graph_node_t const &right, graph_node_t const &delta )
    {
        assert( left.Has<scalar_value_vector_t>() && right.Has<scalar_value_vector_t>() && delta.Has<scalar_value_vector_t>() );
        assert( left.Has<type_t>() && right.Has<type_t>() && delta.Has<type_t>() );

        assert( SameType( left, right ) );
        assert( SameType( left, delta ) );
        assert( SameLength<scalar_value_t>( left, right ) );
        assert( SameLength<scalar_value_t>( left, delta ) );

        assert( ( left.Get<type_t>().mValue == scalar_type_t::FLOAT32 ) || ( left.Get<type_t>().mValue == scalar_type_t::FLOAT64 ) );

        auto  new_entity  = scope.CreateNode();
        auto &operand_data = new_entity.Add<arange_operation_t>( arange_operation_t{ left, right, delta } );

        new_entity.Add<type_t>( left.Get<type_t>() );
        new_entity.Add<operand_t>( vector_t<graph_node_t>{ left, right, delta } );

        vector_t<vector_t<uint32_t>> output_shape( left.Get<scalar_value_vector_t>().mValue.size() );
        auto                         leftValues  = left.Get<scalar_value_vector_t>().mValue;
        auto                         rightValues = right.Get<scalar_value_vector_t>().mValue;
        auto                         deltaValues = delta.Get<scalar_value_vector_t>().mValue;

        for( uint32_t i = 0; i < left.Get<scalar_value_vector_t>().mValue.size(); i++ )
        {
            output_shape[i] = { static_cast<uint32_t>( std::ceil(
                ( std::get<float>( rightValues[i] ) - std::get<float>( leftValues[i] ) ) / std::get<float>( deltaValues[i] ) ) ) };
        }

        new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( output_shape, size_of( left.Get<type_t>().mValue ) ) );
        new_entity.Add<graph_operation_t>().Bind<sARangeOperationController>();

        return new_entity;
    }

    graph_node_t LinearSpace( scope_t &scope, graph_node_t const &left, graph_node_t const &right, graph_node_t const &subdivisions )
    {
        assert( left.Has<multi_tensor_value_t>() && right.Has<multi_tensor_value_t>() && subdivisions.Has<u32_vector_t>() );

        assert( SameShape( left, right ) );

        assert( left.Get<multi_tensor_value_t>().Shape().CountLayers() == subdivisions.Get<u32_vector_t>().mValue.size() );
        assert( SameType( left, right ) );
        assert( ( left.Get<type_t>().mValue == scalar_type_t::FLOAT32 ) || ( left.Get<type_t>().mValue == scalar_type_t::FLOAT64 ) );

        auto  new_entity  = scope.CreateNode();
        auto &operand_data = new_entity.Add<linear_space_operation_t>( linear_space_operation_t{ left, right, subdivisions } );

        new_entity.Add<type_t>( left.Get<type_t>() );
        new_entity.Add<operand_t>( vector_t<graph_node_t>{ left, right, subdivisions } );

        auto                         subdivisionsValue = subdivisions.Get<u32_vector_t>().mValue;
        vector_t<vector_t<uint32_t>> output_shape( left.Get<multi_tensor_value_t>().Shape().Shape.size() );

        for( uint32_t i = 0; i < left.Get<multi_tensor_value_t>().Shape().Shape.size(); i++ )
        {
            output_shape[i] = left.Get<multi_tensor_value_t>().Shape().Shape[i];
            output_shape[i].push_back( subdivisionsValue[i] );
        }

        new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( output_shape, size_of( left.Get<type_t>().mValue ) ) );
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
                    textures.Get<vector_value_t<cuda::texture_sampler2d_t::DeviceData>>().mValue.size() );

        if( y.Has<multi_tensor_value_t>() )
            assert( y.Get<multi_tensor_value_t>().Shape().CountLayers() ==
                    textures.Get<vector_value_t<cuda::texture_sampler2d_t::DeviceData>>().mValue.size() );

        assert( SameType( x, y ) );

        assert( x.Get<type_t>().mValue == scalar_type_t::FLOAT32 );

        auto  new_entity  = scope.CreateNode();
        auto &operand_data = new_entity.Add<sample2D_operation_t>( sample2D_operation_t{ x, y, textures } );

        new_entity.Add<type_t>( x.Get<type_t>() );
        new_entity.Add<operand_t>( vector_t<graph_node_t>{ x, y, textures } );
        new_entity.Add<multi_tensor_value_t>( scope.mPool, operand_data.mX.Get<multi_tensor_value_t>().Shape() );
        new_entity.Add<graph_operation_t>().Bind<sSample2DOperationController>();

        return new_entity;
    }

    graph_node_t AffineTransform( scope_t &scope, graph_node_t const &A, graph_node_t const &x, graph_node_t const &B )
    {
        assert( ( x.HasAll<multi_tensor_value_t, type_t>() ) );
        assert( ( A.Has<type_t>() && B.Has<type_t>() ) );
        assert( ( A.HasAny<multi_tensor_value_t, scalar_node_t, scalar_value_vector_t>() ) );
        assert( ( B.HasAny<multi_tensor_value_t, scalar_node_t, scalar_value_vector_t>() ) );
        assert( SameType( x, A ) );
        assert( SameType( x, B ) );

        auto  new_entity  = scope.CreateNode();
        auto &operand_data = new_entity.Add<affine_transform_operation_t>( affine_transform_operation_t{ A, x, B } );
        new_entity.Add<type_t>( x.Get<type_t>() );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ A, x, B } );
        new_entity.Add<multi_tensor_value_t>( scope.mPool, operand_data.mX.Get<multi_tensor_value_t>().Shape() );
        new_entity.Add<graph_operation_t>().Bind<sAffineNodeController>();

        return new_entity;
    }

    graph_node_t Collapse( scope_t &scope, graph_node_t const &array )
    {
        assert( ( array.Has<multi_tensor_value_t>() ) );

        auto new_entity = scope.CreateNode();

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array } );

        auto &inputShape = array.Get<multi_tensor_value_t>().Shape();
        for( uint32_t i = 0; i < inputShape.Shape.size(); i++ )
        {
            if( inputShape.Shape[i] != inputShape.Shape[0] )
                throw std::runtime_error( "All dimensions should be equal" );
        }

        vector_t<uint32_t> lOutputDimension( inputShape.Rank + 1 );
        lOutputDimension[0] = inputShape.CountLayers();
        std::copy( inputShape.Shape[0].begin(), inputShape.Shape[0].end(), lOutputDimension.begin() + 1 );

        new_entity.Add<type_t>( array.Get<type_t>() );
        new_entity.Add<multi_tensor_value_t>(
            scope.mPool, array.Get<multi_tensor_value_t>().mValue.GetMemoryBuffer(),
            tensor_shape_t( vector_t<vector_t<uint32_t>>{ lOutputDimension }, static_cast<size_t>( inputShape.ElementSize ) ) );

        return new_entity;
    }

    graph_node_t Expand( scope_t &scope, graph_node_t const &array )
    {
        auto new_entity = scope.CreateNode();
        assert( ( array.Has<multi_tensor_value_t>() ) );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array } );

        auto &inputShape = array.Get<multi_tensor_value_t>().Shape();

        assert( inputShape.CountLayers() == 1 );

        vector_t<vector_t<uint32_t>> output_shape( inputShape.Shape[0][0] );

        for( uint32_t i = 0; i < output_shape.size(); i++ )
        {
            output_shape[i] = vector_t<uint32_t>( inputShape.Rank - 1 );
            std::copy( inputShape.Shape[0].begin() + 1, inputShape.Shape[0].end(), output_shape[i].begin() );
        }

        new_entity.Add<type_t>( array.Get<type_t>() );
        new_entity.Add<multi_tensor_value_t>( scope.mPool, array.Get<multi_tensor_value_t>().mValue.GetMemoryBuffer(),
                                              tensor_shape_t( output_shape, static_cast<size_t>( inputShape.ElementSize ) ) );

        return new_entity;
    }

    graph_node_t Reshape( scope_t &scope, graph_node_t const &array, tensor_shape_t &newShape )
    {
        auto new_entity = scope.CreateNode();
        assert( ( array.Has<multi_tensor_value_t>() ) );

        auto &inputShape = array.Get<multi_tensor_value_t>().Shape();

        assert( inputShape.CountLayers() == newShape.CountLayers() );
        assert( inputShape.ElementSize == newShape.ElementSize );

        for( uint32_t i = 0; i < inputShape.CountLayers(); i++ )
        {
            uint32_t size0 = std::accumulate( inputShape.Shape[i].begin(), inputShape.Shape[i].end(), 1, std::multiplies<uint32_t>() );
            uint32_t size1 = std::accumulate( newShape.Shape[i].begin(), newShape.Shape[i].end(), 1, std::multiplies<uint32_t>() );

            if( size0 != size1 )
                throw std::runtime_error( "Incompatible dimensions" );
        }

        new_entity.Add<type_t>( array.Get<type_t>() );
        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array } );
        new_entity.Add<multi_tensor_value_t>( scope.mPool, array.Get<multi_tensor_value_t>().mValue.GetMemoryBuffer(), newShape );

        return new_entity;
    }

    graph_node_t Relayout( scope_t &scope, graph_node_t const &array, tensor_shape_t &newLayout )
    {
        auto new_entity = scope.CreateNode();
        assert( ( array.Has<multi_tensor_value_t>() ) );

        auto &inputShape = array.Get<multi_tensor_value_t>().Shape();

        assert( inputShape.ElementSize == newLayout.ElementSize );

        uint32_t inputSize = 0;
        for( uint32_t i = 0; i < inputShape.CountLayers(); i++ )
            inputSize += std::accumulate( inputShape.Shape[i].begin(), inputShape.Shape[i].end(), 1, std::multiplies<uint32_t>() );

        uint32_t outputSize = 0;
        for( uint32_t i = 0; i < newLayout.CountLayers(); i++ )
            outputSize += std::accumulate( newLayout.Shape[i].begin(), newLayout.Shape[i].end(), 1, std::multiplies<uint32_t>() );

        if( inputSize != outputSize )
            throw std::runtime_error( "Incompatible dimensions" );

        if( array.Has<type_t>() )
            new_entity.Add<type_t>( array.Get<type_t>() );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array } );
        new_entity.Add<multi_tensor_value_t>( scope.mPool, array.Get<multi_tensor_value_t>().mValue.GetMemoryBuffer(), newLayout );

        return new_entity;
    }

    graph_node_t Flatten( scope_t &scope, graph_node_t const &array )
    {
        auto new_entity = scope.CreateNode();
        assert( ( array.Has<multi_tensor_value_t>() ) );

        auto inputShape = array.Get<multi_tensor_value_t>().Shape();

        inputShape.Flatten( 0 );
        new_entity.Add<type_t>( array.Get<type_t>() );
        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array } );
        new_entity.Add<multi_tensor_value_t>( scope.mPool, array.Get<multi_tensor_value_t>().mValue.GetMemoryBuffer(), inputShape );

        return new_entity;
    }

    graph_node_t Slice( scope_t &scope, graph_node_t const &array, graph_node_t const &begin, graph_node_t const &end )
    {
        assert( ( array.HasAll<type_t, multi_tensor_value_t>() ) );
        assert( ( ( begin.HasAny<vector_value_t<uint32_t>, scalar_node_t>() ) ) &&
                ( end.HasAny<vector_value_t<uint32_t>, scalar_node_t>() ) );

        auto new_entity = scope.CreateNode();

        new_entity.Add<type_t>( array.Get<type_t>() );
        auto &operand_data  = new_entity.Add<array_slice_operation_t>();
        operand_data.mArray = array;

        auto inputShape = array.Get<multi_tensor_value_t>().Shape();

        uint32_t           maxBlockSize = 0;
        vector_t<uint32_t> blockSizes( inputShape.CountLayers() );

        vector_t<uint32_t> beginValue;
        if( begin.Has<vector_value_t<uint32_t>>() )
        {
            beginValue         = begin.Get<vector_value_t<uint32_t>>().mValue;
            operand_data.mBegin = begin;
        }
        else
        {
            beginValue = vector_t<uint32_t>( inputShape.CountLayers(), std::get<uint32_t>( begin.Get<scalar_node_t>().mValue ) );
            operand_data.mBegin = VectorValue( scope, beginValue );
        }

        vector_t<uint32_t> endValue;
        if( end.Has<vector_value_t<uint32_t>>() )
        {
            endValue         = end.Get<vector_value_t<uint32_t>>().mValue;
            operand_data.mEnd = end;
        }
        else
        {
            endValue         = vector_t<uint32_t>( inputShape.CountLayers(), std::get<uint32_t>( end.Get<scalar_node_t>().mValue ) );
            operand_data.mEnd = VectorValue( scope, endValue );
        }

        vector_t<vector_t<uint32_t>> output_shape( inputShape.CountLayers() );

        for( uint32_t i = 0; i < inputShape.CountLayers(); i++ )
        {
            output_shape[i] = vector_t<uint32_t>( inputShape.Shape[i].size() );
            for( uint32_t j = 0; j < inputShape.Shape[i].size() - 1; j++ )
            {
                output_shape[i][j] = inputShape.Shape[i][j];
            }
            output_shape[i][inputShape.Shape[i].size() - 1] = std::max( endValue[i] - beginValue[i] + 1, static_cast<uint32_t>( 0 ) );
        }

        inputShape.Flatten( -1 );
        operand_data.mMaxBlockSize = inputShape.MaxDimensions[0];
        operand_data.mBlockSizes   = VectorValue( scope, inputShape.GetDimension( 0 ) );
        operand_data.mElementCount = VectorValue( scope, inputShape.GetDimension( -1 ) );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array, operand_data.mBegin, operand_data.mEnd, operand_data.mBlockSizes,
                                                           operand_data.mElementCount } );

        new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( output_shape, size_of( array.Get<type_t>().mValue ) ) );
        new_entity.Add<graph_operation_t>().Bind<sArraySliceOperationController>();

        return new_entity;
    }

    graph_node_t Summation( scope_t &scope, graph_node_t const &array )
    {
        auto               inputShape = array.Get<multi_tensor_value_t>().Shape();
        vector_t<uint32_t> lLastDimensions( inputShape.CountLayers() );
        for( uint32_t i = 0; i < inputShape.CountLayers(); i++ )
            lLastDimensions[i] = inputShape.Shape[i][inputShape.Shape[i].size() - 1] - 1;

        auto lZero    = ConstantScalarValue( scope, static_cast<uint32_t>( 0 ) );
        auto endValue = VectorValue( scope, lLastDimensions );

        return Summation( scope, array, lZero, endValue );
    }

    graph_node_t Summation( scope_t &scope, graph_node_t const &array, graph_node_t const &begin, graph_node_t const &end )
    {
        assert( ( array.HasAll<type_t, multi_tensor_value_t>() ) );
        assert( ( ( begin.HasAny<vector_value_t<uint32_t>, scalar_node_t>() ) ) &&
                ( end.HasAny<vector_value_t<uint32_t>, scalar_node_t>() ) );

        auto new_entity = scope.CreateNode();

        new_entity.Add<type_t>( array.Get<type_t>() );
        auto &operand_data  = new_entity.Add<array_sum_operation_t>();
        operand_data.mArray = array;

        auto inputShape = array.Get<multi_tensor_value_t>().Shape();

        if( begin.Has<vector_value_t<uint32_t>>() )
        {
            operand_data.mBegin = begin;
        }
        else
        {
            vector_t<uint32_t> beginValue( inputShape.CountLayers(), std::get<uint32_t>( begin.Get<scalar_node_t>().mValue ) );
            operand_data.mBegin = VectorValue( scope, beginValue );
        }

        if( end.Has<vector_value_t<uint32_t>>() )
        {
            operand_data.mEnd = end;
        }
        else
        {
            vector_t<uint32_t> endValue( inputShape.CountLayers(), std::get<uint32_t>( end.Get<scalar_node_t>().mValue ) );
            operand_data.mEnd = VectorValue( scope, endValue );
        }

        auto output_shape = inputShape;
        output_shape.Trim( -1 );

        inputShape.Flatten( -1 );
        operand_data.mMaxBlockSize = inputShape.MaxDimensions[0];
        operand_data.mBlockSizes   = VectorValue( scope, inputShape.GetDimension( 0 ) );
        operand_data.mElementCount = VectorValue( scope, inputShape.GetDimension( -1 ) );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array, operand_data.mBegin, operand_data.mEnd, operand_data.mBlockSizes,
                                                           operand_data.mElementCount } );
        new_entity.Add<multi_tensor_value_t>( scope.mPool,
                                              tensor_shape_t( output_shape.Shape, size_of( array.Get<type_t>().mValue ) ) );
        new_entity.Add<graph_operation_t>().Bind<sArraySummationOperationController>();

        return new_entity;
    }

    graph_node_t CountTrue( scope_t &scope, graph_node_t const &array )
    {
        assert( ( array.HasAll<type_t, multi_tensor_value_t>() ) );

        auto new_entity = scope.CreateNode();

        new_entity.Add<type_t>( type_t{ scalar_type_t::UINT32 } );
        auto &operand_data  = new_entity.Add<count_true_operation_t>();
        operand_data.mArray = array;

        auto inputShape  = array.Get<multi_tensor_value_t>().Shape();
        auto output_shape = inputShape;
        output_shape.Trim( -1 );

        inputShape.Flatten( -1 );
        operand_data.mMaxBlockSize = inputShape.MaxDimensions[0];
        operand_data.mBlockSizes   = VectorValue( scope, inputShape.GetDimension( 0 ) );
        operand_data.mElementCount = VectorValue( scope, inputShape.GetDimension( -1 ) );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array, operand_data.mBlockSizes, operand_data.mElementCount } );
        new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( output_shape.Shape, size_of( scalar_type_t::UINT32 ) ) );
        new_entity.Add<graph_operation_t>().Bind<sCountTrueOperationController>();

        return new_entity;
    }

    graph_node_t CountZero( scope_t &scope, graph_node_t const &array )
    {
        assert( ( array.HasAll<type_t, multi_tensor_value_t>() ) );

        auto new_entity = scope.CreateNode();

        new_entity.Add<type_t>( type_t{ scalar_type_t::UINT32 } );
        auto &operand_data  = new_entity.Add<count_zero_operation_t>();
        operand_data.mArray = array;

        auto inputShape = array.Get<multi_tensor_value_t>().Shape();

        auto output_shape = inputShape;
        output_shape.Trim( -1 );

        inputShape.Flatten( -1 );
        operand_data.mMaxBlockSize = inputShape.MaxDimensions[0];
        operand_data.mBlockSizes   = VectorValue( scope, inputShape.GetDimension( 0 ) );
        operand_data.mElementCount = VectorValue( scope, inputShape.GetDimension( -1 ) );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array, operand_data.mBlockSizes, operand_data.mElementCount } );
        new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( output_shape.Shape, size_of( scalar_type_t::UINT32 ) ) );
        new_entity.Add<graph_operation_t>().Bind<sCountZeroOperationController>();

        return new_entity;
    }

    graph_node_t CountNonZero( scope_t &scope, graph_node_t const &array )
    {
        assert( ( array.HasAll<type_t, multi_tensor_value_t>() ) );

        auto new_entity = scope.CreateNode();

        new_entity.Add<type_t>( type_t{ scalar_type_t::UINT32 } );
        auto &operand_data  = new_entity.Add<count_non_zero_operation_t>();
        operand_data.mArray = array;

        auto inputShape = array.Get<multi_tensor_value_t>().Shape();

        auto output_shape = inputShape;
        output_shape.Trim( -1 );

        inputShape.Flatten( -1 );
        operand_data.mMaxBlockSize = inputShape.MaxDimensions[0];
        operand_data.mBlockSizes   = VectorValue( scope, inputShape.GetDimension( 0 ) );
        operand_data.mElementCount = VectorValue( scope, inputShape.GetDimension( -1 ) );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array, operand_data.mBlockSizes, operand_data.mElementCount } );
        new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( output_shape.Shape, size_of( scalar_type_t::UINT32 ) ) );
        new_entity.Add<graph_operation_t>().Bind<sCountNonZeroOperationController>();

        return new_entity;
    }

    graph_node_t Diff( scope_t &scope, graph_node_t const &array, uint32_t count )
    {
        assert( ( array.HasAll<type_t, multi_tensor_value_t>() ) );

        auto new_entity = scope.CreateNode();

        new_entity.Add<type_t>( array.Get<type_t>() );
        auto &operand_data  = new_entity.Add<diff_operation_t>();
        operand_data.mArray = array;
        operand_data.mCount = count;

        auto inputShape  = array.Get<multi_tensor_value_t>().Shape();
        auto output_shape = array.Get<multi_tensor_value_t>().Shape();

        inputShape.Flatten( -1 );
        operand_data.mMaxBlockSize = inputShape.MaxDimensions[0];
        operand_data.mBlockSizes   = VectorValue( scope, inputShape.GetDimension( 0 ) );
        operand_data.mElementCount = VectorValue( scope, inputShape.GetDimension( -1 ) );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array, operand_data.mBlockSizes, operand_data.mElementCount } );
        new_entity.Add<multi_tensor_value_t>( scope.mPool, output_shape );
        new_entity.Add<graph_operation_t>().Bind<sDiffOperationController>();

        return new_entity;
    }

    graph_node_t Shift( scope_t &scope, graph_node_t const &array, int32_t count, graph_node_t const &fillValue )
    {
        assert( ( array.HasAll<type_t, multi_tensor_value_t>() ) );
        assert( ( fillValue.HasAll<type_t, scalar_node_t>() ) );
        assert( SameType( fillValue, array ) );

        auto new_entity = scope.CreateNode();

        new_entity.Add<type_t>( array.Get<type_t>() );
        auto &operand_data      = new_entity.Add<shift_operation_t>();
        operand_data.mArray     = array;
        operand_data.mCount     = count;
        operand_data.mFillValue = fillValue;

        auto inputShape  = array.Get<multi_tensor_value_t>().Shape();
        auto output_shape = array.Get<multi_tensor_value_t>().Shape();

        inputShape.Flatten( -1 );
        operand_data.mMaxBlockSize = inputShape.MaxDimensions[0];
        operand_data.mBlockSizes   = VectorValue( scope, inputShape.GetDimension( 0 ) );
        operand_data.mElementCount = VectorValue( scope, inputShape.GetDimension( -1 ) );

        new_entity.Add<operand_t>(
            vector_t<graph_node_t>{ array, operand_data.mFillValue, operand_data.mBlockSizes, operand_data.mElementCount } );
        new_entity.Add<multi_tensor_value_t>( scope.mPool, output_shape );
        new_entity.Add<graph_operation_t>().Bind<sShiftOperationController>();

        return new_entity;
    }

    graph_node_t Floor( scope_t &scope, graph_node_t const &array )
    {
        assert( ( array.HasAll<type_t, multi_tensor_value_t>() ) );
        assert( ( array.Get<type_t>().mValue == scalar_type_t::FLOAT32 ) );

        auto new_entity = scope.CreateNode();

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array } );
        new_entity.Add<type_t>( array.Get<type_t>() );
        new_entity.Add<floor_operation_t>( floor_operation_t{ array } );

        auto shape = array.Get<multi_tensor_value_t>().Shape().Shape;
        new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( shape, size_of( array.Get<type_t>().mValue ) ) );

        new_entity.Add<graph_operation_t>().Bind<sFloorOperationController>();

        return new_entity;
    }

    graph_node_t Ceil( scope_t &scope, graph_node_t const &array )
    {
        assert( ( array.HasAll<type_t, multi_tensor_value_t>() ) );
        assert( ( array.Get<type_t>().mValue == scalar_type_t::FLOAT32 ) );

        auto new_entity = scope.CreateNode();

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array } );
        new_entity.Add<type_t>( array.Get<type_t>() );
        new_entity.Add<ceiling_operation_t>( ceiling_operation_t{ array } );

        auto shape = array.Get<multi_tensor_value_t>().Shape().Shape;
        new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( shape, size_of( array.Get<type_t>().mValue ) ) );

        new_entity.Add<graph_operation_t>().Bind<sCeilOperationController>();

        return new_entity;
    }

    graph_node_t Abs( scope_t &scope, graph_node_t const &array )
    {
        assert( ( array.HasAll<type_t, multi_tensor_value_t>() ) );
        assert( ( array.Get<type_t>().mValue == scalar_type_t::FLOAT32 ) ||
                ( ( array.Get<type_t>().mValue >= scalar_type_t::INT8 ) && ( array.Get<type_t>().mValue <= scalar_type_t::INT64 ) ) );

        auto new_entity = scope.CreateNode();

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array } );
        new_entity.Add<type_t>( array.Get<type_t>() );
        new_entity.Add<abs_operation_t>( abs_operation_t{ array } );

        auto shape = array.Get<multi_tensor_value_t>().Shape().Shape;
        new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( shape, size_of( array.Get<type_t>().mValue ) ) );

        new_entity.Add<graph_operation_t>().Bind<sAbsOperationController>();

        return new_entity;
    }

    graph_node_t Sqrt( scope_t &scope, graph_node_t const &array )
    {
        assert( ( array.HasAll<type_t, multi_tensor_value_t>() ) );

        auto new_entity = scope.CreateNode();

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array } );
        new_entity.Add<type_t>( array.Get<type_t>() );
        new_entity.Add<sqrt_operation_t>( sqrt_operation_t{ array } );

        auto shape = array.Get<multi_tensor_value_t>().Shape().Shape;
        new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( shape, size_of( array.Get<type_t>().mValue ) ) );

        new_entity.Add<graph_operation_t>().Bind<sSqrtOperationController>();

        return new_entity;
    }

    graph_node_t Round( scope_t &scope, graph_node_t const &array )
    {
        assert( ( array.HasAll<type_t, multi_tensor_value_t>() ) );

        auto new_entity = scope.CreateNode();

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ array } );
        new_entity.Add<type_t>( array.Get<type_t>() );
        new_entity.Add<round_operation_t>( round_operation_t{ array } );

        auto shape = array.Get<multi_tensor_value_t>().Shape().Shape;
        new_entity.Add<multi_tensor_value_t>( scope.mPool, tensor_shape_t( shape, size_of( array.Get<type_t>().mValue ) ) );

        new_entity.Add<graph_operation_t>().Bind<sRoundOperationController>();

        return new_entity;
    }

    graph_node_t Conv1D( scope_t &scope, graph_node_t const &array0, graph_node_t const &array1 )
    {
        assert( ( array0.HasAll<type_t, multi_tensor_value_t>() ) );
        assert( ( array1.HasAll<type_t, multi_tensor_value_t>() ) );

        auto new_entity = scope.CreateNode();

        new_entity.Add<type_t>( array0.Get<type_t>() );
        auto &operand_data   = new_entity.Add<conv1d_operation_t>();
        operand_data.mArray0 = array0;
        operand_data.mArray1 = array1;

        auto inputShape  = array0.Get<multi_tensor_value_t>().Shape();
        auto output_shape = array0.Get<multi_tensor_value_t>().Shape();

        inputShape.Flatten( -1 );
        operand_data.mMaxBlockSize0    = inputShape.MaxDimensions[0];
        operand_data.mMaxElementCount0 = inputShape.MaxDimensions[inputShape.Rank - 1];
        operand_data.mBlockSizes0      = VectorValue( scope, inputShape.GetDimension( 0 ) );
        operand_data.mElementCount0    = VectorValue( scope, inputShape.GetDimension( -1 ) );

        auto kernelShape = array1.Get<multi_tensor_value_t>().Shape();

        kernelShape.Flatten( -1 );
        operand_data.mMaxBlockSize1 = kernelShape.MaxDimensions[0];
        operand_data.mBlockSizes1   = VectorValue( scope, kernelShape.GetDimension( 0 ) );
        operand_data.mElementCount1 = VectorValue( scope, kernelShape.GetDimension( -1 ) );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ operand_data.mArray0, operand_data.mBlockSizes0, operand_data.mElementCount0,
                                                           operand_data.mArray1, operand_data.mBlockSizes1,
                                                           operand_data.mElementCount1 } );
        new_entity.Add<multi_tensor_value_t>( scope.mPool, output_shape );
        new_entity.Add<graph_operation_t>().Bind<sConv1DOperationController>();

        return new_entity;
    }

    graph_node_t HCat( scope_t &scope, graph_node_t const &array0, graph_node_t const &array1 )
    {
        assert( ( array0.HasAll<type_t, multi_tensor_value_t>() ) );
        assert( ( array1.HasAll<type_t, multi_tensor_value_t>() ) );

        auto new_entity = scope.CreateNode();

        new_entity.Add<type_t>( array0.Get<type_t>() );
        auto &operand_data   = new_entity.Add<hcat_operation_t>();
        operand_data.mArray0 = array0;
        operand_data.mArray1 = array1;

        auto inputShape0 = array0.Get<multi_tensor_value_t>().Shape();
        auto inputShape1 = array1.Get<multi_tensor_value_t>().Shape();

        auto               lastDim0 = inputShape0.GetDimension( -1 );
        auto               lastDim1 = inputShape1.GetDimension( -1 );
        vector_t<uint32_t> concatenated{};
        for( uint32_t i = 0; i < lastDim0.size(); i++ )
            concatenated.push_back( lastDim0[i] + lastDim1[i] );

        auto output_shape = array0.Get<multi_tensor_value_t>().Shape();
        output_shape.Trim( -1 );
        output_shape.InsertDimension( -1, concatenated );

        auto blockShape = array0.Get<multi_tensor_value_t>().Shape();
        blockShape.Flatten( -1 );
        operand_data.mMaxBlockSize  = blockShape.MaxDimensions[0];
        operand_data.mBlockSizes    = VectorValue( scope, blockShape.GetDimension( 0 ) );
        operand_data.mElementCount0 = VectorValue( scope, inputShape0.GetDimension( -1 ) );
        operand_data.mElementCount1 = VectorValue( scope, inputShape1.GetDimension( -1 ) );

        new_entity.Add<operand_t>( vector_t<graph_node_t>{ operand_data.mArray0, operand_data.mArray1, operand_data.mBlockSizes,
                                                           operand_data.mElementCount0, operand_data.mElementCount1 } );
        new_entity.Add<multi_tensor_value_t>( scope.mPool, output_shape );
        new_entity.Add<graph_operation_t>().Bind<sHCatOperationController>();

        return new_entity;
    }

} // namespace numlua::mtops
