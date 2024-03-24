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

namespace SE::TensorOps
{
    using namespace SE::Cuda;

    scope_t::scope_t( uint32_t aMemorySize )
    {
        mPool = memory_pool_t( aMemorySize );
    }

    scope_t &scope_t::WithOpName( const string_t &aName )
    {
        mName = aName;
        return *this;
    }

    graph_node_t scope_t::CreateNode()
    {
        graph_node_t lNewEntity;
        if( mName.has_value() )
        {
            lNewEntity                 = mNodesRegistry.CreateEntity( mName.value() );
            mNamedNodes[mName.value()] = lNewEntity;
            mName.reset();
        }
        else
        {
            lNewEntity = mNodesRegistry.CreateEntity();
        }

        return lNewEntity;
    }

    graph_node_t scope_t::operator[]( const string_t &aNodeName )
    {
        if( mNamedNodes.find( aNodeName ) != mNamedNodes.end() )
            return mNamedNodes[aNodeName];
        return graph_node_t{};
    }

    void scope_t::Reset()
    {
        mPool.Reset();
        mNodesRegistry.Clear();
        mNamedNodes.clear();
        mName.reset();
    }

    void scope_t::Run( graph_node_t const &aNode )
    {
        Run( vector_t<graph_node_t>{ aNode } );
    }

    void scope_t::Run( vector_t<graph_node_t> const &aNode )
    {
        std::deque<graph_node_t>                         lExecutionQueue;
        std::stack<graph_node_t, vector_t<graph_node_t>> lStack( aNode );

        while( !lStack.empty() )
        {
            graph_node_t lCurrent = lStack.top();
            lStack.pop();

            if( lCurrent.Has<do_not_expand_t>() )
                continue;

            std::deque<graph_node_t>::iterator lPos = std::find( lExecutionQueue.begin(), lExecutionQueue.end(), lCurrent );
            if( lPos != lExecutionQueue.end() )
            {
                lExecutionQueue.erase( lPos );
            }
            lExecutionQueue.push_back( lCurrent );
            if( lCurrent.Has<operand_t>() )
            {
                for( graph_node_t lDependent : lCurrent.Get<operand_t>().mOperands )
                {
                    lStack.push( lDependent );
                }
            }
        }

        // Allocate memory for tensors which are on the stack
        for( auto lElement = lExecutionQueue.rbegin(); lElement < lExecutionQueue.rend(); lElement++ )
        {
            if( ( *lElement ).Has<allocated_tag_t>() )
                continue;

            if( ( *lElement ).Has<multi_tensor_value_t>() )
            {
                ( *lElement ).Get<multi_tensor_value_t>().mValue =
                    multi_tensor_t( mPool, ( *lElement ).Get<multi_tensor_value_t>().mShape );
                ( *lElement ).Add<allocated_tag_t>();
            }

            if( ( *lElement ).Has<vector_buffer_t>() )
            {
                ( *lElement ).Get<vector_buffer_t>().mValue =
                    mPool.Allocate( ( *lElement ).Get<vector_buffer_t>().mSize );
                ( *lElement ).Add<allocated_tag_t>();
            }
        }

        for( auto lElement = lExecutionQueue.rbegin(); lElement < lExecutionQueue.rend(); lElement++ )
        {
            if( !( *lElement ).Has<graph_operation_t>() )
                continue;

            auto &lComponent = ( *lElement ).Get<graph_operation_t>();
            if( !lComponent.mControllerInstance )
            {
                lComponent.mControllerInstance = lComponent.mInstantiateController();
                lComponent.mControllerInstance->Initialize( *lElement );
            }

            lComponent.mControllerInstance->Run();
        }
        SyncDevice();
    }

    graph_node_t CreateMultiTensor( scope_t &aScope, tensor_shape_t const &aShape )
    {
        auto lNewEntity = aScope.CreateNode();
        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool, aShape );
        lNewEntity.Add<graph_operation_t>().Bind<sMultiTensorRunner>();
        return lNewEntity;
    }

    graph_node_t MultiTensorValue( scope_t &aScope, constant_value_initializer_t const &aInitializer,
                                   tensor_shape_t const &aShape )
    {
        auto lNewEntity = CreateMultiTensor( aScope, aShape );
        lNewEntity.Add<type_t>( type_of( aInitializer.mValue ) );
        lNewEntity.Add<constant_value_initializer_t>( aInitializer );
        return lNewEntity;
    }

    graph_node_t MultiTensorValue( scope_t &aScope, vector_initializer_t const &aInitializer, tensor_shape_t const &aShape )
    {
        auto lNewEntity = CreateMultiTensor( aScope, aShape );
        lNewEntity.Add<type_t>( type_of( aInitializer.mValue[0] ) );
        auto &lInitializerComponent = lNewEntity.Add<vector_initializer_t>( aInitializer );
        lInitializerComponent.mData = aScope.mPool.Allocate( aInitializer.mValue.size() * size_of( type_of( aInitializer.mValue[0] ) ) );
        return lNewEntity;
    }

    graph_node_t MultiTensorValue( scope_t &aScope, data_initializer_t const &aInitializer, tensor_shape_t const &aShape )
    {
        auto lNewEntity = CreateMultiTensor( aScope, aShape );
        lNewEntity.Add<type_t>( type_of( aInitializer.mValue[0] ) );
        auto &lInitializerComponent = lNewEntity.Add<data_initializer_t>( aInitializer );
        return lNewEntity;
    }

    graph_node_t MultiTensorValue( scope_t &aScope, random_uniform_initializer_t const &aInitializer,
                                   tensor_shape_t const &aShape )
    {
        auto lNewEntity = CreateMultiTensor( aScope, aShape );
        lNewEntity.Add<type_t>( aInitializer.mType );
        lNewEntity.Add<random_uniform_initializer_t>( aInitializer );
        return lNewEntity;
    }

    graph_node_t MultiTensorValue( scope_t &aScope, random_normal_initializer_t const &aInitializer, tensor_shape_t const &aShape )
    {
        auto lNewEntity = CreateMultiTensor( aScope, aShape );
        lNewEntity.Add<type_t>( aInitializer.mType );
        lNewEntity.Add<random_normal_initializer_t>( aInitializer );
        return lNewEntity;
    }

    static inline bool SameType( graph_node_t const &aLeft, graph_node_t const &aRight )
    {
        return ( aLeft.Get<type_t>().mValue == aRight.Get<type_t>().mValue );
    }

    static inline bool SameShape( graph_node_t const &aLeft, graph_node_t const &aRight )
    {
        return ( aLeft.Get<multi_tensor_value_t>().Shape() == aRight.Get<multi_tensor_value_t>().Shape() );
    }

    template <typename T>
    static inline bool SameLength( graph_node_t aLeft, graph_node_t const &aRight )
    {
        return ( aLeft.Get<vector_value_t<T>>().mValue.size() == aRight.Get<vector_value_t<T>>().mValue.size() );
    }

    graph_node_t BinaryOperation( scope_t &aScope, scalar_type_t aType, graph_node_t const &aLeft, graph_node_t const &aRight )
    {
        assert( aLeft.Has<type_t>() );
        assert( aRight.Has<type_t>() );
        assert( ( aLeft.HasAny<multi_tensor_value_t, scalar_value_vector_t, scalar_node_t>() ) );
        assert( ( aRight.HasAny<multi_tensor_value_t, scalar_value_vector_t, scalar_node_t>() ) );

        auto  lNewEntity   = aScope.CreateNode();
        auto &lOperandData = lNewEntity.Add<binary_operation_t>( binary_operation_t{ aLeft, aRight } );

        if( lOperandData.mLeftOperand.Has<multi_tensor_value_t>() )
        {
            if( lOperandData.mRightOperand.Has<multi_tensor_value_t>() )
            {
                auto lLeftShape  = lOperandData.mLeftOperand.Get<multi_tensor_value_t>().Shape();
                auto lRightShape = lOperandData.mRightOperand.Get<multi_tensor_value_t>().Shape();

                if( lLeftShape != lRightShape )
                {
                    if( lLeftShape.mRank == lRightShape.mRank - 1 )
                    {
                        lRightShape.Trim( -1 );
                        if( lRightShape != lLeftShape )
                            throw std::runtime_error( "Can only add tensors of the same shape" );

                        auto &lBroadcastInfo = lNewEntity.Add<broadcast_info_t>();
                        lRightShape.Flatten( 0 );
                        lBroadcastInfo.mBroadcastHint = broadcast_hint_t::LEFT;
                        lBroadcastInfo.mMaxBlockSize  = lRightShape.mMaxDimensions[0];
                        lBroadcastInfo.mBlockSizes    = VectorValue( aScope, lRightShape.GetDimension( 0 ) );

                        auto lBroadcastShape                  = lOperandData.mRightOperand.Get<multi_tensor_value_t>().Shape();
                        lBroadcastInfo.mBroadcastDimension    = VectorValue( aScope, lBroadcastShape.GetDimension( -1 ) );
                        lBroadcastInfo.mMaxBroadcastDimension = lBroadcastShape.mMaxDimensions[lBroadcastShape.mRank - 1];

                        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool,
                                                               tensor_shape_t( lBroadcastShape.mShape, size_of( aType ) ) );
                    }
                    else if( lRightShape.mRank == lLeftShape.mRank - 1 )
                    {
                        lLeftShape.Trim( -1 );
                        if( lRightShape != lLeftShape )
                            throw std::runtime_error( "Can only add tensors of the same shape" );

                        auto &lBroadcastInfo = lNewEntity.Add<broadcast_info_t>();
                        lRightShape.Flatten( 0 );
                        lBroadcastInfo.mBroadcastHint = broadcast_hint_t::RIGHT;
                        lBroadcastInfo.mMaxBlockSize  = lRightShape.mMaxDimensions[0];
                        lBroadcastInfo.mBlockSizes    = VectorValue( aScope, lRightShape.GetDimension( 0 ) );

                        auto lBroadcastShape                  = lOperandData.mLeftOperand.Get<multi_tensor_value_t>().Shape();
                        lBroadcastInfo.mBroadcastDimension    = VectorValue( aScope, lBroadcastShape.GetDimension( -1 ) );
                        lBroadcastInfo.mMaxBroadcastDimension = lBroadcastShape.mMaxDimensions[lBroadcastShape.mRank - 1];

                        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool,
                                                               tensor_shape_t( lBroadcastShape.mShape, size_of( aType ) ) );
                    }
                    else
                    {
                        throw std::runtime_error( "Can only add tensors of the same shape" );
                    }
                }
                else
                {
                    auto lShape = lOperandData.mLeftOperand.Get<multi_tensor_value_t>().Shape().mShape;
                    lNewEntity.Add<multi_tensor_value_t>( aScope.mPool, tensor_shape_t( lShape, size_of( aType ) ) );
                }
            }
            else
            {
                auto lShape = lOperandData.mLeftOperand.Get<multi_tensor_value_t>().Shape().mShape;
                lNewEntity.Add<multi_tensor_value_t>( aScope.mPool, tensor_shape_t( lShape, size_of( aType ) ) );
            }
        }
        else
        {
            if( !( lOperandData.mRightOperand.Has<multi_tensor_value_t>() ) )
            {
                throw std::runtime_error( "RHS should have a tensor" );
            }

            auto lShape = lOperandData.mRightOperand.Get<multi_tensor_value_t>().Shape().mShape;
            lNewEntity.Add<multi_tensor_value_t>( aScope.mPool, tensor_shape_t( lShape, size_of( aType ) ) );
        }

        if( lNewEntity.Has<broadcast_info_t>() )
            lNewEntity.Add<operand_t>(
                vector_t<graph_node_t>{ aLeft, aRight, lNewEntity.Get<broadcast_info_t>().mBlockSizes,
                                        lNewEntity.Get<broadcast_info_t>().mBroadcastDimension } );
        else
            lNewEntity.Add<operand_t>( vector_t<graph_node_t>{ aLeft, aRight } );

        lNewEntity.Add<type_t>( aType );

        return lNewEntity;
    }

    graph_node_t BinaryOperation( scope_t &aScope, graph_node_t const &aLeft, graph_node_t const &aRight )
    {
        return BinaryOperation( aScope, aLeft.Get<type_t>().mValue, aLeft, aRight );
    }

    graph_node_t Add( scope_t &aScope, graph_node_t const &aLeft, graph_node_t const &aRight )
    {
        auto lNewEntity = BinaryOperation( aScope, aLeft, aRight );
        lNewEntity.Add<graph_operation_t>().Bind<sAddOperationController>();

        return lNewEntity;
    }

    graph_node_t Subtract( scope_t &aScope, graph_node_t const &aLeft, graph_node_t const &aRight )
    {
        auto lNewEntity = BinaryOperation( aScope, aLeft, aRight );
        lNewEntity.Add<graph_operation_t>().Bind<sSubtractOperationController>();

        return lNewEntity;
    }

    graph_node_t Divide( scope_t &aScope, graph_node_t const &aLeft, graph_node_t const &aRight )
    {
        auto lNewEntity = BinaryOperation( aScope, aLeft, aRight );
        lNewEntity.Add<graph_operation_t>().Bind<sDivideOperationController>();

        return lNewEntity;
    }

    graph_node_t Multiply( scope_t &aScope, graph_node_t const &aLeft, graph_node_t const &aRight )
    {
        auto lNewEntity = BinaryOperation( aScope, aLeft, aRight );
        lNewEntity.Add<graph_operation_t>().Bind<sMultiplyOperationController>();

        return lNewEntity;
    }

    graph_node_t And( scope_t &aScope, graph_node_t const &aLeft, graph_node_t const &aRight )
    {
        assert( ( aLeft.Has<type_t>() ) && ( aRight.Has<type_t>() ) );
        assert( SameType( aLeft, aRight ) );
        assert( aLeft.Get<type_t>().mValue == scalar_type_t::UINT8 );

        auto lNewEntity = BinaryOperation( aScope, aLeft, aRight );
        lNewEntity.Add<graph_operation_t>().Bind<sAndOperationController>();

        return lNewEntity;
    }

    graph_node_t Or( scope_t &aScope, graph_node_t const &aLeft, graph_node_t const &aRight )
    {
        assert( ( aLeft.Has<type_t>() ) && ( aRight.Has<type_t>() ) );
        assert( SameType( aLeft, aRight ) );
        assert( aLeft.Get<type_t>().mValue == scalar_type_t::UINT8 );

        auto lNewEntity = BinaryOperation( aScope, aLeft, aRight );
        lNewEntity.Add<graph_operation_t>().Bind<sOrOperationController>();

        return lNewEntity;
    }

    graph_node_t Not( scope_t &aScope, graph_node_t const &aOperand )
    {
        assert( ( aOperand.Has<type_t>() ) );
        assert( aOperand.Get<type_t>().mValue == scalar_type_t::UINT8 );

        auto  lNewEntity   = aScope.CreateNode();
        auto &lOperandData = lNewEntity.Add<not_operation_t>( not_operation_t{ aOperand } );

        lNewEntity.Add<operand_t>( vector_t<graph_node_t>{ aOperand } );
        lNewEntity.Add<type_t>( aOperand.Get<type_t>() );
        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool, lOperandData.mOperand.Get<multi_tensor_value_t>().Shape() );
        lNewEntity.Add<graph_operation_t>().Bind<sNotOperationController>();

        return lNewEntity;
    }

    graph_node_t BitwiseAnd( scope_t &aScope, graph_node_t const &aLeft, graph_node_t const &aRight )
    {
        assert( ( aLeft.Has<type_t>() ) && ( aRight.Has<type_t>() ) );
        assert( SameType( aLeft, aRight ) );
        assert( ( aLeft.Get<type_t>().mValue >= scalar_type_t::UINT8 ) &&
                ( aLeft.Get<type_t>().mValue <= scalar_type_t::INT64 ) );

        auto lNewEntity = BinaryOperation( aScope, aLeft, aRight );
        lNewEntity.Add<graph_operation_t>().Bind<sBitwiseAndOperationController>();

        return lNewEntity;
    }

    graph_node_t BitwiseOr( scope_t &aScope, graph_node_t const &aLeft, graph_node_t const &aRight )
    {
        assert( ( aLeft.Has<type_t>() ) && ( aRight.Has<type_t>() ) );
        assert( SameType( aLeft, aRight ) );
        assert( ( aLeft.Get<type_t>().mValue >= scalar_type_t::UINT8 ) &&
                ( aLeft.Get<type_t>().mValue <= scalar_type_t::INT64 ) );

        auto lNewEntity = BinaryOperation( aScope, aLeft, aRight );
        lNewEntity.Add<graph_operation_t>().Bind<sBitwiseOrOperationController>();

        return lNewEntity;
    }

    graph_node_t BitwiseNot( scope_t &aScope, graph_node_t const &aOperand )
    {
        assert( ( aOperand.Has<type_t>() ) );
        assert( ( aOperand.Get<type_t>().mValue >= scalar_type_t::UINT8 ) &&
                ( aOperand.Get<type_t>().mValue <= scalar_type_t::INT64 ) );

        auto  lNewEntity   = aScope.CreateNode();
        auto &lOperandData = lNewEntity.Add<bitwise_not_operation_t>( bitwise_not_operation_t{ aOperand } );

        lNewEntity.Add<operand_t>( vector_t<graph_node_t>{ aOperand } );
        lNewEntity.Add<type_t>( aOperand.Get<type_t>() );
        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool, lOperandData.mOperand.Get<multi_tensor_value_t>().Shape() );
        lNewEntity.Add<graph_operation_t>().Bind<sBitwiseNotOperationController>();

        return lNewEntity;
    }

    graph_node_t InInterval( scope_t &aScope, graph_node_t const &aX, graph_node_t const &aLower, graph_node_t const &aUpper,
                             bool aStrictLower, bool aStrictUpper )
    {
        assert( aX.Has<type_t>() && ( aLower.Has<type_t>() ) && ( aUpper.Has<type_t>() ) );
        assert( ( aX.Has<multi_tensor_value_t>() ) );
        assert( ( aLower.HasAny<multi_tensor_value_t, scalar_value_vector_t, scalar_node_t>() ) );
        assert( ( aUpper.HasAny<multi_tensor_value_t, scalar_value_vector_t, scalar_node_t>() ) );

        assert( SameType( aX, aLower ) );
        assert( SameType( aX, aUpper ) );

        auto  lNewEntity   = aScope.CreateNode();
        auto &lOperandData = lNewEntity.Add<in_interval_operation_t>(
            in_interval_operation_t{ aX, aLower, aUpper, aStrictLower, aStrictUpper } );

        lNewEntity.Add<operand_t>( vector_t<graph_node_t>{ aX, aLower, aUpper } );
        lNewEntity.Add<type_t>( scalar_type_t::UINT8 );

        vector_t<vector_t<uint32_t>> lOutputShape = aX.Get<multi_tensor_value_t>().Shape().mShape;

        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool, tensor_shape_t( lOutputShape, size_of( scalar_type_t::UINT8 ) ) );
        lNewEntity.Add<graph_operation_t>().Bind<sInIntervalOperationController>();

        return lNewEntity;
    }

    graph_node_t Equal( scope_t &aScope, graph_node_t const &aX, graph_node_t const &aY )
    {
        assert( ( aX.Has<type_t>() ) && ( aY.Has<type_t>() ) );
        assert( SameType( aX, aY ) );

        auto lNewEntity = BinaryOperation( aScope, scalar_type_t::UINT8, aX, aY );
        lNewEntity.Add<graph_operation_t>().Bind<sEqualOperationController>();

        return lNewEntity;
    }

    graph_node_t LessThan( scope_t &aScope, graph_node_t const &aX, graph_node_t const &aY )
    {
        assert( ( aX.Has<type_t>() ) && ( aY.Has<type_t>() ) );
        assert( SameType( aX, aY ) );

        auto lNewEntity = BinaryOperation( aScope, scalar_type_t::UINT8, aX, aY );
        lNewEntity.Add<graph_operation_t>().Bind<sLessThanOperationController>();

        return lNewEntity;
    }

    graph_node_t LessThanOrEqual( scope_t &aScope, graph_node_t const &aX, graph_node_t const &aY )
    {
        assert( ( aX.Has<type_t>() ) && ( aY.Has<type_t>() ) );
        assert( SameType( aX, aY ) );

        auto lNewEntity = BinaryOperation( aScope, scalar_type_t::UINT8, aX, aY );
        lNewEntity.Add<graph_operation_t>().Bind<sLessThanOrEqualOperationController>();

        return lNewEntity;
    }

    graph_node_t GreaterThan( scope_t &aScope, graph_node_t const &aX, graph_node_t const &aY )
    {
        return LessThan( aScope, aY, aX );
    }

    graph_node_t GreaterThanOrEqual( scope_t &aScope, graph_node_t const &aX, graph_node_t const &aY )
    {
        return LessThanOrEqual( aScope, aY, aX );
    }

    graph_node_t Where( scope_t &aScope, graph_node_t const &aCondition, graph_node_t const &aValueIfTrue,
                        graph_node_t const &aValueIfFalse )
    {
        assert( ( aCondition.Has<type_t>() ) && ( aValueIfTrue.Has<type_t>() ) &&
                ( aValueIfFalse.Has<type_t>() ) );
        assert( ( aValueIfTrue.HasAny<multi_tensor_value_t, scalar_value_vector_t, scalar_node_t>() ) );
        assert( ( aValueIfFalse.HasAny<multi_tensor_value_t, scalar_value_vector_t, scalar_node_t>() ) );
        assert( ( aCondition.Get<type_t>().mValue == scalar_type_t::UINT8 ) );
        assert( SameType( aValueIfTrue, aValueIfFalse ) );

        auto lNewEntity = aScope.CreateNode();
        lNewEntity.Add<where_operation_t>( where_operation_t{ aCondition, aValueIfTrue, aValueIfFalse } );
        lNewEntity.Add<type_t>( aValueIfTrue.Get<type_t>() );
        lNewEntity.Add<operand_t>( vector_t<graph_node_t>{ aCondition, aValueIfTrue, aValueIfFalse } );

        auto lShape = aCondition.Get<multi_tensor_value_t>().Shape().mShape;
        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool,
                                               tensor_shape_t( lShape, size_of( aValueIfTrue.Get<type_t>().mValue ) ) );

        lNewEntity.Add<graph_operation_t>().Bind<sWhereOperationController>();

        return lNewEntity;
    }

    graph_node_t Mix( scope_t &aScope, graph_node_t const &aA, graph_node_t const &aB, graph_node_t const &aT )
    {
        assert( ( aA.HasAll<type_t, multi_tensor_value_t>() ) );
        assert( ( aB.HasAll<type_t, multi_tensor_value_t>() ) );
        assert( ( aT.HasAll<type_t, multi_tensor_value_t>() ) );

        auto  lNewEntity   = aScope.CreateNode();
        auto &lOperandData = lNewEntity.Add<mix_operation_t>( mix_operation_t{ aA, aB, aT } );

        lNewEntity.Add<operand_t>( vector_t<graph_node_t>{ aA, aB, aT } );
        lNewEntity.Add<type_t>( aA.Get<type_t>() );
        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool, lOperandData.mA.Get<multi_tensor_value_t>().Shape() );
        lNewEntity.Add<graph_operation_t>().Bind<sMixOperationController>();

        return lNewEntity;
    }

    graph_node_t ToFixedPoint( scope_t &aScope, scalar_type_t aOutputType, graph_node_t const &aArray, graph_node_t const &aScaling )
    {
        assert( ( aArray.HasAll<type_t, multi_tensor_value_t>() ) );
        assert( ( aScaling.HasAll<type_t, scalar_node_t>() ) );

        auto lNewEntity = aScope.CreateNode();

        auto &lOperandData = lNewEntity.Add<convert_to_fixed_point_t>( convert_to_fixed_point_t{ aOutputType, aArray, aScaling } );

        lNewEntity.Add<operand_t>( vector_t<graph_node_t>{ aArray, aScaling } );

        auto &lShape = lOperandData.mArray.Get<multi_tensor_value_t>().Shape().mShape;
        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool, tensor_shape_t( lShape, size_of( aOutputType ) ) );

        lNewEntity.Add<graph_operation_t>().Bind<sToFixedPointOperationController>();

        return lNewEntity;
    }

    graph_node_t Repeat( scope_t &aScope, graph_node_t const &aArray, graph_node_t const &aRepetitions )
    {
        assert( ( aArray.HasAll<type_t, multi_tensor_value_t>() ) );
        assert( aRepetitions.Has<u32_vector_t>() );

        auto  lNewEntity   = aScope.CreateNode();
        auto &lOperandData = lNewEntity.Add<repeat_operation_t>( repeat_operation_t{ aArray, aRepetitions } );

        lNewEntity.Add<operand_t>( vector_t<graph_node_t>{ aArray, aRepetitions } );
        lNewEntity.Add<type_t>( aArray.Get<type_t>() );

        auto &lInputShape = lOperandData.mArray.Get<multi_tensor_value_t>().Shape();

        auto                         lRepetitions = aRepetitions.Get<u32_vector_t>().mValue;
        vector_t<vector_t<uint32_t>> lOutputShape( lInputShape.CountLayers() );
        for( uint32_t i = 0; i < lInputShape.CountLayers(); i++ )
        {
            lOutputShape[i] = vector_t<uint32_t>( lInputShape.mShape[i].size() + 1 );
            for( uint32_t j = 0; j < lInputShape.mShape[i].size(); j++ )
            {
                lOutputShape[i][j] = lInputShape.mShape[i][j];
            }
            lOutputShape[i][lOutputShape[i].size() - 1] = lRepetitions[i];
        }

        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool,
                                               tensor_shape_t( lOutputShape, size_of( aArray.Get<type_t>().mValue ) ) );
        lNewEntity.Add<graph_operation_t>().Bind<sArrayOperationController>();

        return lNewEntity;
    }

    graph_node_t Tile( scope_t &aScope, graph_node_t const &aArray, graph_node_t const &aRepetitions )
    {
        assert( ( aArray.HasAll<type_t, multi_tensor_value_t>() ) );
        assert( aRepetitions.Has<u32_vector_t>() );

        auto  lNewEntity   = aScope.CreateNode();
        auto &lOperandData = lNewEntity.Add<tile_operation_t>( tile_operation_t{ aArray, aRepetitions } );

        lNewEntity.Add<operand_t>( vector_t<graph_node_t>{ aArray, aRepetitions } );
        lNewEntity.Add<type_t>( aArray.Get<type_t>() );
        auto &lInputShape = lOperandData.mArray.Get<multi_tensor_value_t>().Shape();

        auto                         lRepetitions = aRepetitions.Get<u32_vector_t>().mValue;
        vector_t<vector_t<uint32_t>> lOutputShape( lInputShape.CountLayers() );
        for( uint32_t i = 0; i < lInputShape.CountLayers(); i++ )
        {
            lOutputShape[i]                             = vector_t<uint32_t>( lInputShape.mShape[i].size() + 1 );
            lOutputShape[i][lOutputShape[i].size() - 1] = lRepetitions[i];

            lOutputShape[i][0] = lRepetitions[i];

            for( uint32_t j = 0; j < lInputShape.mShape[i].size(); j++ )
            {
                lOutputShape[i][j + 1] = lInputShape.mShape[i][j];
            }
        }

        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool,
                                               tensor_shape_t( lOutputShape, size_of( aArray.Get<type_t>().mValue ) ) );
        lNewEntity.Add<graph_operation_t>().Bind<sArrayOperationController>();

        return lNewEntity;
    }

    graph_node_t ARange( scope_t &aScope, graph_node_t const &aLeft, graph_node_t const &aRight, graph_node_t const &aDelta )
    {
        assert( aLeft.Has<scalar_value_vector_t>() && aRight.Has<scalar_value_vector_t>() &&
                aDelta.Has<scalar_value_vector_t>() );
        assert( aLeft.Has<type_t>() && aRight.Has<type_t>() && aDelta.Has<type_t>() );

        assert( SameType( aLeft, aRight ) );
        assert( SameType( aLeft, aDelta ) );
        assert( SameLength<scalar_value_t>( aLeft, aRight ) );
        assert( SameLength<scalar_value_t>( aLeft, aDelta ) );

        assert( ( aLeft.Get<type_t>().mValue == scalar_type_t::FLOAT32 ) ||
                ( aLeft.Get<type_t>().mValue == scalar_type_t::FLOAT64 ) );

        auto  lNewEntity   = aScope.CreateNode();
        auto &lOperandData = lNewEntity.Add<arange_operation_t>( arange_operation_t{ aLeft, aRight, aDelta } );

        lNewEntity.Add<type_t>( aLeft.Get<type_t>() );
        lNewEntity.Add<operand_t>( vector_t<graph_node_t>{ aLeft, aRight, aDelta } );

        vector_t<vector_t<uint32_t>> lOutputShape( aLeft.Get<scalar_value_vector_t>().mValue.size() );
        auto                         lLeftValues  = aLeft.Get<scalar_value_vector_t>().mValue;
        auto                         lRightValues = aRight.Get<scalar_value_vector_t>().mValue;
        auto                         lDeltaValues = aDelta.Get<scalar_value_vector_t>().mValue;

        for( uint32_t i = 0; i < aLeft.Get<scalar_value_vector_t>().mValue.size(); i++ )
        {
            lOutputShape[i] = { static_cast<uint32_t>( std::ceil(
                ( std::get<float>( lRightValues[i] ) - std::get<float>( lLeftValues[i] ) ) / std::get<float>( lDeltaValues[i] ) ) ) };
        }

        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool,
                                               tensor_shape_t( lOutputShape, size_of( aLeft.Get<type_t>().mValue ) ) );
        lNewEntity.Add<graph_operation_t>().Bind<sARangeOperationController>();

        return lNewEntity;
    }

    graph_node_t LinearSpace( scope_t &aScope, graph_node_t const &aLeft, graph_node_t const &aRight,
                              graph_node_t const &aSubdivisions )
    {
        assert( aLeft.Has<multi_tensor_value_t>() && aRight.Has<multi_tensor_value_t>() &&
                aSubdivisions.Has<u32_vector_t>() );

        assert( SameShape( aLeft, aRight ) );

        assert( aLeft.Get<multi_tensor_value_t>().Shape().CountLayers() == aSubdivisions.Get<u32_vector_t>().mValue.size() );
        assert( SameType( aLeft, aRight ) );
        assert( ( aLeft.Get<type_t>().mValue == scalar_type_t::FLOAT32 ) ||
                ( aLeft.Get<type_t>().mValue == scalar_type_t::FLOAT64 ) );

        auto  lNewEntity   = aScope.CreateNode();
        auto &lOperandData = lNewEntity.Add<linear_space_operation_t>( linear_space_operation_t{ aLeft, aRight, aSubdivisions } );

        lNewEntity.Add<type_t>( aLeft.Get<type_t>() );
        lNewEntity.Add<operand_t>( vector_t<graph_node_t>{ aLeft, aRight, aSubdivisions } );

        auto                         lSubdivisions = aSubdivisions.Get<u32_vector_t>().mValue;
        vector_t<vector_t<uint32_t>> lOutputShape( aLeft.Get<multi_tensor_value_t>().Shape().mShape.size() );

        for( uint32_t i = 0; i < aLeft.Get<multi_tensor_value_t>().Shape().mShape.size(); i++ )
        {
            lOutputShape[i] = aLeft.Get<multi_tensor_value_t>().Shape().mShape[i];
            lOutputShape[i].push_back( lSubdivisions[i] );
        }

        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool,
                                               tensor_shape_t( lOutputShape, size_of( aLeft.Get<type_t>().mValue ) ) );
        lNewEntity.Add<graph_operation_t>().Bind<sLinearSpaceOperationController>();

        return lNewEntity;
    }

    graph_node_t Sample2D( scope_t &aScope, graph_node_t const &aX, graph_node_t const &aY, graph_node_t const &aTextures )
    {
        assert( ( aX.HasAny<multi_tensor_value_t, scalar_node_t, scalar_value_vector_t>() ) );
        assert( ( aY.HasAny<multi_tensor_value_t, scalar_node_t, scalar_value_vector_t>() ) );

        assert( aX.Has<multi_tensor_value_t>() || aY.Has<multi_tensor_value_t>() );
        assert( aTextures.Has<vector_value_t<Cuda::texture_sampler2d_t::DeviceData>>() );

        if( aX.Has<multi_tensor_value_t>() && aY.Has<multi_tensor_value_t>() )
            assert( SameShape( aX, aY ) );

        if( aX.Has<multi_tensor_value_t>() )
            assert( aX.Get<multi_tensor_value_t>().Shape().CountLayers() ==
                    aTextures.Get<vector_value_t<Cuda::texture_sampler2d_t::DeviceData>>().mValue.size() );

        if( aY.Has<multi_tensor_value_t>() )
            assert( aY.Get<multi_tensor_value_t>().Shape().CountLayers() ==
                    aTextures.Get<vector_value_t<Cuda::texture_sampler2d_t::DeviceData>>().mValue.size() );

        assert( SameType( aX, aY ) );

        assert( aX.Get<type_t>().mValue == scalar_type_t::FLOAT32 );

        auto  lNewEntity   = aScope.CreateNode();
        auto &lOperandData = lNewEntity.Add<sample2D_operation_t>( sample2D_operation_t{ aX, aY, aTextures } );

        lNewEntity.Add<type_t>( aX.Get<type_t>() );
        lNewEntity.Add<operand_t>( vector_t<graph_node_t>{ aX, aY, aTextures } );
        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool, lOperandData.mX.Get<multi_tensor_value_t>().Shape() );
        lNewEntity.Add<graph_operation_t>().Bind<sSample2DOperationController>();

        return lNewEntity;
    }

    graph_node_t AffineTransform( scope_t &aScope, graph_node_t const &aA, graph_node_t const &aX, graph_node_t const &aB )
    {
        assert( ( aX.HasAll<multi_tensor_value_t, type_t>() ) );
        assert( ( aA.Has<type_t>() && aB.Has<type_t>() ) );
        assert( ( aA.HasAny<multi_tensor_value_t, scalar_node_t, scalar_value_vector_t>() ) );
        assert( ( aB.HasAny<multi_tensor_value_t, scalar_node_t, scalar_value_vector_t>() ) );
        assert( SameType( aX, aA ) );
        assert( SameType( aX, aB ) );

        auto  lNewEntity   = aScope.CreateNode();
        auto &lOperandData = lNewEntity.Add<affine_transform_operation_t>( affine_transform_operation_t{ aA, aX, aB } );
        lNewEntity.Add<type_t>( aX.Get<type_t>() );

        lNewEntity.Add<operand_t>( vector_t<graph_node_t>{ aA, aX, aB } );
        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool, lOperandData.mX.Get<multi_tensor_value_t>().Shape() );
        lNewEntity.Add<graph_operation_t>().Bind<sAffineNodeController>();

        return lNewEntity;
    }

    graph_node_t Collapse( scope_t &aScope, graph_node_t const &aArray )
    {
        assert( ( aArray.Has<multi_tensor_value_t>() ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<operand_t>( vector_t<graph_node_t>{ aArray } );

        auto &lInputShape = aArray.Get<multi_tensor_value_t>().Shape();
        for( uint32_t i = 0; i < lInputShape.mShape.size(); i++ )
        {
            if( lInputShape.mShape[i] != lInputShape.mShape[0] )
                throw std::runtime_error( "All dimensions should be equal" );
        }

        vector_t<uint32_t> lOutputDimension( lInputShape.mRank + 1 );
        lOutputDimension[0] = lInputShape.CountLayers();
        std::copy( lInputShape.mShape[0].begin(), lInputShape.mShape[0].end(), lOutputDimension.begin() + 1 );

        lNewEntity.Add<type_t>( aArray.Get<type_t>() );
        lNewEntity.Add<multi_tensor_value_t>(
            aScope.mPool, aArray.Get<multi_tensor_value_t>().mValue.GetMemoryBuffer(),
            tensor_shape_t( vector_t<vector_t<uint32_t>>{ lOutputDimension }, static_cast<size_t>( lInputShape.mElementSize ) ) );

        return lNewEntity;
    }

    graph_node_t Expand( scope_t &aScope, graph_node_t const &aArray )
    {
        auto lNewEntity = aScope.CreateNode();
        assert( ( aArray.Has<multi_tensor_value_t>() ) );

        lNewEntity.Add<operand_t>( vector_t<graph_node_t>{ aArray } );

        auto &lInputShape = aArray.Get<multi_tensor_value_t>().Shape();

        assert( lInputShape.CountLayers() == 1 );

        vector_t<vector_t<uint32_t>> lOutputShape( lInputShape.mShape[0][0] );

        for( uint32_t i = 0; i < lOutputShape.size(); i++ )
        {
            lOutputShape[i] = vector_t<uint32_t>( lInputShape.mRank - 1 );
            std::copy( lInputShape.mShape[0].begin() + 1, lInputShape.mShape[0].end(), lOutputShape[i].begin() );
        }

        lNewEntity.Add<type_t>( aArray.Get<type_t>() );
        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool, aArray.Get<multi_tensor_value_t>().mValue.GetMemoryBuffer(),
                                               tensor_shape_t( lOutputShape, static_cast<size_t>( lInputShape.mElementSize ) ) );

        return lNewEntity;
    }

    graph_node_t Reshape( scope_t &aScope, graph_node_t const &aArray, tensor_shape_t &aNewShape )
    {
        auto lNewEntity = aScope.CreateNode();
        assert( ( aArray.Has<multi_tensor_value_t>() ) );

        auto &lInputShape = aArray.Get<multi_tensor_value_t>().Shape();

        assert( lInputShape.CountLayers() == aNewShape.CountLayers() );
        assert( lInputShape.mElementSize == aNewShape.mElementSize );

        for( uint32_t i = 0; i < lInputShape.CountLayers(); i++ )
        {
            uint32_t lSize0 =
                std::accumulate( lInputShape.mShape[i].begin(), lInputShape.mShape[i].end(), 1, std::multiplies<uint32_t>() );
            uint32_t lSize1 =
                std::accumulate( aNewShape.mShape[i].begin(), aNewShape.mShape[i].end(), 1, std::multiplies<uint32_t>() );

            if( lSize0 != lSize1 )
                throw std::runtime_error( "Incompatible dimensions" );
        }

        lNewEntity.Add<type_t>( aArray.Get<type_t>() );
        lNewEntity.Add<operand_t>( vector_t<graph_node_t>{ aArray } );
        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool, aArray.Get<multi_tensor_value_t>().mValue.GetMemoryBuffer(), aNewShape );

        return lNewEntity;
    }

    graph_node_t Relayout( scope_t &aScope, graph_node_t const &aArray, tensor_shape_t &aNewLayout )
    {
        auto lNewEntity = aScope.CreateNode();
        assert( ( aArray.Has<multi_tensor_value_t>() ) );

        auto &lInputShape = aArray.Get<multi_tensor_value_t>().Shape();

        assert( lInputShape.mElementSize == aNewLayout.mElementSize );

        uint32_t lInputSize = 0;
        for( uint32_t i = 0; i < lInputShape.CountLayers(); i++ )
            lInputSize +=
                std::accumulate( lInputShape.mShape[i].begin(), lInputShape.mShape[i].end(), 1, std::multiplies<uint32_t>() );

        uint32_t lOutputSize = 0;
        for( uint32_t i = 0; i < aNewLayout.CountLayers(); i++ )
            lOutputSize += std::accumulate( aNewLayout.mShape[i].begin(), aNewLayout.mShape[i].end(), 1, std::multiplies<uint32_t>() );

        if( lInputSize != lOutputSize )
            throw std::runtime_error( "Incompatible dimensions" );

        if( aArray.Has<type_t>() )
            lNewEntity.Add<type_t>( aArray.Get<type_t>() );

        lNewEntity.Add<operand_t>( vector_t<graph_node_t>{ aArray } );
        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool, aArray.Get<multi_tensor_value_t>().mValue.GetMemoryBuffer(),
                                               aNewLayout );

        return lNewEntity;
    }

    graph_node_t Flatten( scope_t &aScope, graph_node_t const &aArray )
    {
        auto lNewEntity = aScope.CreateNode();
        assert( ( aArray.Has<multi_tensor_value_t>() ) );

        auto lInputShape = aArray.Get<multi_tensor_value_t>().Shape();

        lInputShape.Flatten( 0 );
        lNewEntity.Add<type_t>( aArray.Get<type_t>() );
        lNewEntity.Add<operand_t>( vector_t<graph_node_t>{ aArray } );
        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool, aArray.Get<multi_tensor_value_t>().mValue.GetMemoryBuffer(),
                                               lInputShape );

        return lNewEntity;
    }

    graph_node_t Slice( scope_t &aScope, graph_node_t const &aArray, graph_node_t const &aBegin, graph_node_t const &aEnd )
    {
        assert( ( aArray.HasAll<type_t, multi_tensor_value_t>() ) );
        assert( ( ( aBegin.HasAny<vector_value_t<uint32_t>, scalar_node_t>() ) ) &&
                ( aEnd.HasAny<vector_value_t<uint32_t>, scalar_node_t>() ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<type_t>( aArray.Get<type_t>() );
        auto &lOperandData  = lNewEntity.Add<array_slice_operation_t>();
        lOperandData.mArray = aArray;

        auto lInputShape = aArray.Get<multi_tensor_value_t>().Shape();

        uint32_t           lMaxBlockSize = 0;
        vector_t<uint32_t> lBlockSizes( lInputShape.CountLayers() );

        vector_t<uint32_t> lBegin;
        if( aBegin.Has<vector_value_t<uint32_t>>() )
        {
            lBegin              = aBegin.Get<vector_value_t<uint32_t>>().mValue;
            lOperandData.mBegin = aBegin;
        }
        else
        {
            lBegin = vector_t<uint32_t>( lInputShape.CountLayers(), std::get<uint32_t>( aBegin.Get<scalar_node_t>().mValue ) );
            lOperandData.mBegin = VectorValue( aScope, lBegin );
        }

        vector_t<uint32_t> lEnd;
        if( aEnd.Has<vector_value_t<uint32_t>>() )
        {
            lEnd              = aEnd.Get<vector_value_t<uint32_t>>().mValue;
            lOperandData.mEnd = aEnd;
        }
        else
        {
            lEnd = vector_t<uint32_t>( lInputShape.CountLayers(), std::get<uint32_t>( aEnd.Get<scalar_node_t>().mValue ) );
            lOperandData.mEnd = VectorValue( aScope, lEnd );
        }

        vector_t<vector_t<uint32_t>> lOutputShape( lInputShape.CountLayers() );

        for( uint32_t i = 0; i < lInputShape.CountLayers(); i++ )
        {
            lOutputShape[i] = vector_t<uint32_t>( lInputShape.mShape[i].size() );
            for( uint32_t j = 0; j < lInputShape.mShape[i].size() - 1; j++ )
            {
                lOutputShape[i][j] = lInputShape.mShape[i][j];
            }
            lOutputShape[i][lInputShape.mShape[i].size() - 1] = std::max( lEnd[i] - lBegin[i] + 1, static_cast<uint32_t>( 0 ) );
        }

        lInputShape.Flatten( -1 );
        lOperandData.mMaxBlockSize = lInputShape.mMaxDimensions[0];
        lOperandData.mBlockSizes   = VectorValue( aScope, lInputShape.GetDimension( 0 ) );
        lOperandData.mElementCount = VectorValue( aScope, lInputShape.GetDimension( -1 ) );

        lNewEntity.Add<operand_t>( vector_t<graph_node_t>{ aArray, lOperandData.mBegin, lOperandData.mEnd,
                                                                   lOperandData.mBlockSizes, lOperandData.mElementCount } );

        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool,
                                               tensor_shape_t( lOutputShape, size_of( aArray.Get<type_t>().mValue ) ) );
        lNewEntity.Add<graph_operation_t>().Bind<sArraySliceOperationController>();

        return lNewEntity;
    }

    graph_node_t Summation( scope_t &aScope, graph_node_t const &aArray )
    {
        auto               lInputShape = aArray.Get<multi_tensor_value_t>().Shape();
        vector_t<uint32_t> lLastDimensions( lInputShape.CountLayers() );
        for( uint32_t i = 0; i < lInputShape.CountLayers(); i++ )
            lLastDimensions[i] = lInputShape.mShape[i][lInputShape.mShape[i].size() - 1] - 1;

        auto lZero = ConstantScalarValue( aScope, static_cast<uint32_t>( 0 ) );
        auto lEnd  = VectorValue( aScope, lLastDimensions );

        return Summation( aScope, aArray, lZero, lEnd );
    }

    graph_node_t Summation( scope_t &aScope, graph_node_t const &aArray, graph_node_t const &aBegin, graph_node_t const &aEnd )
    {
        assert( ( aArray.HasAll<type_t, multi_tensor_value_t>() ) );
        assert( ( ( aBegin.HasAny<vector_value_t<uint32_t>, scalar_node_t>() ) ) &&
                ( aEnd.HasAny<vector_value_t<uint32_t>, scalar_node_t>() ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<type_t>( aArray.Get<type_t>() );
        auto &lOperandData  = lNewEntity.Add<array_sum_operation_t>();
        lOperandData.mArray = aArray;

        auto lInputShape = aArray.Get<multi_tensor_value_t>().Shape();

        if( aBegin.Has<vector_value_t<uint32_t>>() )
        {
            lOperandData.mBegin = aBegin;
        }
        else
        {
            vector_t<uint32_t> lBegin( lInputShape.CountLayers(), std::get<uint32_t>( aBegin.Get<scalar_node_t>().mValue ) );
            lOperandData.mBegin = VectorValue( aScope, lBegin );
        }

        if( aEnd.Has<vector_value_t<uint32_t>>() )
        {
            lOperandData.mEnd = aEnd;
        }
        else
        {
            vector_t<uint32_t> lEnd( lInputShape.CountLayers(), std::get<uint32_t>( aEnd.Get<scalar_node_t>().mValue ) );
            lOperandData.mEnd = VectorValue( aScope, lEnd );
        }

        auto lOutputShape = lInputShape;
        lOutputShape.Trim( -1 );

        lInputShape.Flatten( -1 );
        lOperandData.mMaxBlockSize = lInputShape.mMaxDimensions[0];
        lOperandData.mBlockSizes   = VectorValue( aScope, lInputShape.GetDimension( 0 ) );
        lOperandData.mElementCount = VectorValue( aScope, lInputShape.GetDimension( -1 ) );

        lNewEntity.Add<operand_t>( vector_t<graph_node_t>{ aArray, lOperandData.mBegin, lOperandData.mEnd,
                                                                   lOperandData.mBlockSizes, lOperandData.mElementCount } );
        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool,
                                               tensor_shape_t( lOutputShape.mShape, size_of( aArray.Get<type_t>().mValue ) ) );
        lNewEntity.Add<graph_operation_t>().Bind<sArraySummationOperationController>();

        return lNewEntity;
    }

    graph_node_t CountTrue( scope_t &aScope, graph_node_t const &aArray )
    {
        assert( ( aArray.HasAll<type_t, multi_tensor_value_t>() ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<type_t>( type_t{ scalar_type_t::UINT32 } );
        auto &lOperandData  = lNewEntity.Add<count_true_operation_t>();
        lOperandData.mArray = aArray;

        auto lInputShape  = aArray.Get<multi_tensor_value_t>().Shape();
        auto lOutputShape = lInputShape;
        lOutputShape.Trim( -1 );

        lInputShape.Flatten( -1 );
        lOperandData.mMaxBlockSize = lInputShape.mMaxDimensions[0];
        lOperandData.mBlockSizes   = VectorValue( aScope, lInputShape.GetDimension( 0 ) );
        lOperandData.mElementCount = VectorValue( aScope, lInputShape.GetDimension( -1 ) );

        lNewEntity.Add<operand_t>( vector_t<graph_node_t>{ aArray, lOperandData.mBlockSizes, lOperandData.mElementCount } );
        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool, tensor_shape_t( lOutputShape.mShape, size_of( scalar_type_t::UINT32 ) ) );
        lNewEntity.Add<graph_operation_t>().Bind<sCountTrueOperationController>();

        return lNewEntity;
    }

    graph_node_t CountZero( scope_t &aScope, graph_node_t const &aArray )
    {
        assert( ( aArray.HasAll<type_t, multi_tensor_value_t>() ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<type_t>( type_t{ scalar_type_t::UINT32 } );
        auto &lOperandData  = lNewEntity.Add<count_zero_operation_t>();
        lOperandData.mArray = aArray;

        auto lInputShape = aArray.Get<multi_tensor_value_t>().Shape();

        auto lOutputShape = lInputShape;
        lOutputShape.Trim( -1 );

        lInputShape.Flatten( -1 );
        lOperandData.mMaxBlockSize = lInputShape.mMaxDimensions[0];
        lOperandData.mBlockSizes   = VectorValue( aScope, lInputShape.GetDimension( 0 ) );
        lOperandData.mElementCount = VectorValue( aScope, lInputShape.GetDimension( -1 ) );

        lNewEntity.Add<operand_t>( vector_t<graph_node_t>{ aArray, lOperandData.mBlockSizes, lOperandData.mElementCount } );
        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool, tensor_shape_t( lOutputShape.mShape, size_of( scalar_type_t::UINT32 ) ) );
        lNewEntity.Add<graph_operation_t>().Bind<sCountZeroOperationController>();

        return lNewEntity;
    }

    graph_node_t CountNonZero( scope_t &aScope, graph_node_t const &aArray )
    {
        assert( ( aArray.HasAll<type_t, multi_tensor_value_t>() ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<type_t>( type_t{ scalar_type_t::UINT32 } );
        auto &lOperandData  = lNewEntity.Add<count_non_zero_operation_t>();
        lOperandData.mArray = aArray;

        auto lInputShape = aArray.Get<multi_tensor_value_t>().Shape();

        auto lOutputShape = lInputShape;
        lOutputShape.Trim( -1 );

        lInputShape.Flatten( -1 );
        lOperandData.mMaxBlockSize = lInputShape.mMaxDimensions[0];
        lOperandData.mBlockSizes   = VectorValue( aScope, lInputShape.GetDimension( 0 ) );
        lOperandData.mElementCount = VectorValue( aScope, lInputShape.GetDimension( -1 ) );

        lNewEntity.Add<operand_t>( vector_t<graph_node_t>{ aArray, lOperandData.mBlockSizes, lOperandData.mElementCount } );
        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool, tensor_shape_t( lOutputShape.mShape, size_of( scalar_type_t::UINT32 ) ) );
        lNewEntity.Add<graph_operation_t>().Bind<sCountNonZeroOperationController>();

        return lNewEntity;
    }

    graph_node_t Diff( scope_t &aScope, graph_node_t const &aArray, uint32_t aCount )
    {
        assert( ( aArray.HasAll<type_t, multi_tensor_value_t>() ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<type_t>( aArray.Get<type_t>() );
        auto &lOperandData  = lNewEntity.Add<diff_operation_t>();
        lOperandData.mArray = aArray;
        lOperandData.mCount = aCount;

        auto lInputShape  = aArray.Get<multi_tensor_value_t>().Shape();
        auto lOutputShape = aArray.Get<multi_tensor_value_t>().Shape();

        lInputShape.Flatten( -1 );
        lOperandData.mMaxBlockSize = lInputShape.mMaxDimensions[0];
        lOperandData.mBlockSizes   = VectorValue( aScope, lInputShape.GetDimension( 0 ) );
        lOperandData.mElementCount = VectorValue( aScope, lInputShape.GetDimension( -1 ) );

        lNewEntity.Add<operand_t>( vector_t<graph_node_t>{ aArray, lOperandData.mBlockSizes, lOperandData.mElementCount } );
        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool, lOutputShape );
        lNewEntity.Add<graph_operation_t>().Bind<sDiffOperationController>();

        return lNewEntity;
    }

    graph_node_t Shift( scope_t &aScope, graph_node_t const &aArray, int32_t aCount, graph_node_t const &aFillValue )
    {
        assert( ( aArray.HasAll<type_t, multi_tensor_value_t>() ) );
        assert( ( aFillValue.HasAll<type_t, scalar_node_t>() ) );
        assert( SameType( aFillValue, aArray ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<type_t>( aArray.Get<type_t>() );
        auto &lOperandData      = lNewEntity.Add<shift_operation_t>();
        lOperandData.mArray     = aArray;
        lOperandData.mCount     = aCount;
        lOperandData.mFillValue = aFillValue;

        auto lInputShape  = aArray.Get<multi_tensor_value_t>().Shape();
        auto lOutputShape = aArray.Get<multi_tensor_value_t>().Shape();

        lInputShape.Flatten( -1 );
        lOperandData.mMaxBlockSize = lInputShape.mMaxDimensions[0];
        lOperandData.mBlockSizes   = VectorValue( aScope, lInputShape.GetDimension( 0 ) );
        lOperandData.mElementCount = VectorValue( aScope, lInputShape.GetDimension( -1 ) );

        lNewEntity.Add<operand_t>(
            vector_t<graph_node_t>{ aArray, lOperandData.mFillValue, lOperandData.mBlockSizes, lOperandData.mElementCount } );
        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool, lOutputShape );
        lNewEntity.Add<graph_operation_t>().Bind<sShiftOperationController>();

        return lNewEntity;
    }

    graph_node_t Floor( scope_t &aScope, graph_node_t const &aArray )
    {
        assert( ( aArray.HasAll<type_t, multi_tensor_value_t>() ) );
        assert( ( aArray.Get<type_t>().mValue == scalar_type_t::FLOAT32 ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<operand_t>( vector_t<graph_node_t>{ aArray } );
        lNewEntity.Add<type_t>( aArray.Get<type_t>() );
        lNewEntity.Add<floor_operation_t>( floor_operation_t{ aArray } );

        auto lShape = aArray.Get<multi_tensor_value_t>().Shape().mShape;
        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool, tensor_shape_t( lShape, size_of( aArray.Get<type_t>().mValue ) ) );

        lNewEntity.Add<graph_operation_t>().Bind<sFloorOperationController>();

        return lNewEntity;
    }

    graph_node_t Ceil( scope_t &aScope, graph_node_t const &aArray )
    {
        assert( ( aArray.HasAll<type_t, multi_tensor_value_t>() ) );
        assert( ( aArray.Get<type_t>().mValue == scalar_type_t::FLOAT32 ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<operand_t>( vector_t<graph_node_t>{ aArray } );
        lNewEntity.Add<type_t>( aArray.Get<type_t>() );
        lNewEntity.Add<ceiling_operation_t>( ceiling_operation_t{ aArray } );

        auto lShape = aArray.Get<multi_tensor_value_t>().Shape().mShape;
        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool, tensor_shape_t( lShape, size_of( aArray.Get<type_t>().mValue ) ) );

        lNewEntity.Add<graph_operation_t>().Bind<sCeilOperationController>();

        return lNewEntity;
    }

    graph_node_t Abs( scope_t &aScope, graph_node_t const &aArray )
    {
        assert( ( aArray.HasAll<type_t, multi_tensor_value_t>() ) );
        assert( ( aArray.Get<type_t>().mValue == scalar_type_t::FLOAT32 ) ||
                ( ( aArray.Get<type_t>().mValue >= scalar_type_t::INT8 ) &&
                  ( aArray.Get<type_t>().mValue <= scalar_type_t::INT64 ) ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<operand_t>( vector_t<graph_node_t>{ aArray } );
        lNewEntity.Add<type_t>( aArray.Get<type_t>() );
        lNewEntity.Add<abs_operation_t>( abs_operation_t{ aArray } );

        auto lShape = aArray.Get<multi_tensor_value_t>().Shape().mShape;
        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool, tensor_shape_t( lShape, size_of( aArray.Get<type_t>().mValue ) ) );

        lNewEntity.Add<graph_operation_t>().Bind<sAbsOperationController>();

        return lNewEntity;
    }

    graph_node_t Sqrt( scope_t &aScope, graph_node_t const &aArray )
    {
        assert( ( aArray.HasAll<type_t, multi_tensor_value_t>() ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<operand_t>( vector_t<graph_node_t>{ aArray } );
        lNewEntity.Add<type_t>( aArray.Get<type_t>() );
        lNewEntity.Add<sqrt_operation_t>( sqrt_operation_t{ aArray } );

        auto lShape = aArray.Get<multi_tensor_value_t>().Shape().mShape;
        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool, tensor_shape_t( lShape, size_of( aArray.Get<type_t>().mValue ) ) );

        lNewEntity.Add<graph_operation_t>().Bind<sSqrtOperationController>();

        return lNewEntity;
    }

    graph_node_t Round( scope_t &aScope, graph_node_t const &aArray )
    {
        assert( ( aArray.HasAll<type_t, multi_tensor_value_t>() ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<operand_t>( vector_t<graph_node_t>{ aArray } );
        lNewEntity.Add<type_t>( aArray.Get<type_t>() );
        lNewEntity.Add<round_operation_t>( round_operation_t{ aArray } );

        auto lShape = aArray.Get<multi_tensor_value_t>().Shape().mShape;
        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool, tensor_shape_t( lShape, size_of( aArray.Get<type_t>().mValue ) ) );

        lNewEntity.Add<graph_operation_t>().Bind<sRoundOperationController>();

        return lNewEntity;
    }

    graph_node_t Conv1D( scope_t &aScope, graph_node_t const &aArray0, graph_node_t const &aArray1 )
    {
        assert( ( aArray0.HasAll<type_t, multi_tensor_value_t>() ) );
        assert( ( aArray1.HasAll<type_t, multi_tensor_value_t>() ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<type_t>( aArray0.Get<type_t>() );
        auto &lOperandData   = lNewEntity.Add<conv1d_operation_t>();
        lOperandData.mArray0 = aArray0;
        lOperandData.mArray1 = aArray1;

        auto lInputShape  = aArray0.Get<multi_tensor_value_t>().Shape();
        auto lOutputShape = aArray0.Get<multi_tensor_value_t>().Shape();

        lInputShape.Flatten( -1 );
        lOperandData.mMaxBlockSize0    = lInputShape.mMaxDimensions[0];
        lOperandData.mMaxElementCount0 = lInputShape.mMaxDimensions[lInputShape.mRank - 1];
        lOperandData.mBlockSizes0      = VectorValue( aScope, lInputShape.GetDimension( 0 ) );
        lOperandData.mElementCount0    = VectorValue( aScope, lInputShape.GetDimension( -1 ) );

        auto lKernelShape = aArray1.Get<multi_tensor_value_t>().Shape();

        lKernelShape.Flatten( -1 );
        lOperandData.mMaxBlockSize1 = lKernelShape.mMaxDimensions[0];
        lOperandData.mBlockSizes1   = VectorValue( aScope, lKernelShape.GetDimension( 0 ) );
        lOperandData.mElementCount1 = VectorValue( aScope, lKernelShape.GetDimension( -1 ) );

        lNewEntity.Add<operand_t>( vector_t<graph_node_t>{ lOperandData.mArray0, lOperandData.mBlockSizes0,
                                                                   lOperandData.mElementCount0, lOperandData.mArray1,
                                                                   lOperandData.mBlockSizes1, lOperandData.mElementCount1 } );
        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool, lOutputShape );
        lNewEntity.Add<graph_operation_t>().Bind<sConv1DOperationController>();

        return lNewEntity;
    }

    graph_node_t HCat( scope_t &aScope, graph_node_t const &aArray0, graph_node_t const &aArray1 )
    {
        assert( ( aArray0.HasAll<type_t, multi_tensor_value_t>() ) );
        assert( ( aArray1.HasAll<type_t, multi_tensor_value_t>() ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<type_t>( aArray0.Get<type_t>() );
        auto &lOperandData   = lNewEntity.Add<hcat_operation_t>();
        lOperandData.mArray0 = aArray0;
        lOperandData.mArray1 = aArray1;

        auto lInputShape0 = aArray0.Get<multi_tensor_value_t>().Shape();
        auto lInputShape1 = aArray1.Get<multi_tensor_value_t>().Shape();

        auto               lLastDim0 = lInputShape0.GetDimension( -1 );
        auto               lLastDim1 = lInputShape1.GetDimension( -1 );
        vector_t<uint32_t> lConcatenated{};
        for( uint32_t i = 0; i < lLastDim0.size(); i++ )
            lConcatenated.push_back( lLastDim0[i] + lLastDim1[i] );

        auto lOutputShape = aArray0.Get<multi_tensor_value_t>().Shape();
        lOutputShape.Trim( -1 );
        lOutputShape.InsertDimension( -1, lConcatenated );

        auto lBlockShape = aArray0.Get<multi_tensor_value_t>().Shape();
        lBlockShape.Flatten( -1 );
        lOperandData.mMaxBlockSize  = lBlockShape.mMaxDimensions[0];
        lOperandData.mBlockSizes    = VectorValue( aScope, lBlockShape.GetDimension( 0 ) );
        lOperandData.mElementCount0 = VectorValue( aScope, lInputShape0.GetDimension( -1 ) );
        lOperandData.mElementCount1 = VectorValue( aScope, lInputShape1.GetDimension( -1 ) );

        lNewEntity.Add<operand_t>( vector_t<graph_node_t>{ lOperandData.mArray0, lOperandData.mArray1,
                                                                   lOperandData.mBlockSizes, lOperandData.mElementCount0,
                                                                   lOperandData.mElementCount1 } );
        lNewEntity.Add<multi_tensor_value_t>( aScope.mPool, lOutputShape );
        lNewEntity.Add<graph_operation_t>().Bind<sHCatOperationController>();

        return lNewEntity;
    }

} // namespace SE::TensorOps
