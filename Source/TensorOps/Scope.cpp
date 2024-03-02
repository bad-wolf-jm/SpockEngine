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
        std::deque<graph_node_t>                   lExecutionQueue;
        std::stack<graph_node_t, vector_t<graph_node_t>> lStack( aNode );

        while( !lStack.empty() )
        {
            graph_node_t lCurrent = lStack.top();
            lStack.pop();

            if( lCurrent.Has<sDoNotExpand>() )
                continue;

            std::deque<graph_node_t>::iterator lPos = std::find( lExecutionQueue.begin(), lExecutionQueue.end(), lCurrent );
            if( lPos != lExecutionQueue.end() )
            {
                lExecutionQueue.erase( lPos );
            }
            lExecutionQueue.push_back( lCurrent );
            if( lCurrent.Has<sOperandComponent>() )
            {
                for( graph_node_t lDependent : lCurrent.Get<sOperandComponent>().mOperands )
                {
                    lStack.push( lDependent );
                }
            }
        }

        // Allocate memory for tensors which are on the stack
        for( auto lElement = lExecutionQueue.rbegin(); lElement < lExecutionQueue.rend(); lElement++ )
        {
            if( ( *lElement ).Has<sAllocatedTag>() )
                continue;

            if( ( *lElement ).Has<sMultiTensorComponent>() )
            {
                ( *lElement ).Get<sMultiTensorComponent>().mValue =
                    multi_tensor_t( mPool, ( *lElement ).Get<sMultiTensorComponent>().mShape );
                ( *lElement ).Add<sAllocatedTag>();
            }

            if( ( *lElement ).Has<sVectorBufferComponent>() )
            {
                ( *lElement ).Get<sVectorBufferComponent>().mValue =
                    mPool.Allocate( ( *lElement ).Get<sVectorBufferComponent>().mSize );
                ( *lElement ).Add<sAllocatedTag>();
            }
        }

        for( auto lElement = lExecutionQueue.rbegin(); lElement < lExecutionQueue.rend(); lElement++ )
        {
            if( !( *lElement ).Has<sGraphOperationComponent>() )
                continue;

            auto &lComponent = ( *lElement ).Get<sGraphOperationComponent>();
            if( !lComponent.mControllerInstance )
            {
                lComponent.mControllerInstance = lComponent.mInstantiateController();
                lComponent.mControllerInstance->Initialize( *lElement );
            }

            lComponent.mControllerInstance->Run();
        }
        SyncDevice();
    }

    graph_node_t CreateMultiTensor( scope_t &aScope, sTensorShape const &aShape )
    {
        auto lNewEntity = aScope.CreateNode();
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, aShape );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sMultiTensorRunner>();
        return lNewEntity;
    }

    graph_node_t MultiTensorValue( scope_t &aScope, sConstantValueInitializerComponent const &aInitializer, sTensorShape const &aShape )
    {
        auto lNewEntity = CreateMultiTensor( aScope, aShape );
        lNewEntity.Add<sTypeComponent>( TypeOf( aInitializer.mValue ) );
        lNewEntity.Add<sConstantValueInitializerComponent>( aInitializer );
        return lNewEntity;
    }

    graph_node_t MultiTensorValue( scope_t &aScope, sVectorInitializerComponent const &aInitializer, sTensorShape const &aShape )
    {
        auto lNewEntity = CreateMultiTensor( aScope, aShape );
        lNewEntity.Add<sTypeComponent>( TypeOf( aInitializer.mValue[0] ) );
        auto &lInitializerComponent = lNewEntity.Add<sVectorInitializerComponent>( aInitializer );
        lInitializerComponent.mData = aScope.mPool.Allocate( aInitializer.mValue.size() * SizeOf( TypeOf( aInitializer.mValue[0] ) ) );
        return lNewEntity;
    }

    graph_node_t MultiTensorValue( scope_t &aScope, sDataInitializerComponent const &aInitializer, sTensorShape const &aShape )
    {
        auto lNewEntity = CreateMultiTensor( aScope, aShape );
        lNewEntity.Add<sTypeComponent>( TypeOf( aInitializer.mValue[0] ) );
        auto &lInitializerComponent = lNewEntity.Add<sDataInitializerComponent>( aInitializer );
        return lNewEntity;
    }

    graph_node_t MultiTensorValue( scope_t &aScope, sRandomUniformInitializerComponent const &aInitializer, sTensorShape const &aShape )
    {
        auto lNewEntity = CreateMultiTensor( aScope, aShape );
        lNewEntity.Add<sTypeComponent>( aInitializer.mType );
        lNewEntity.Add<sRandomUniformInitializerComponent>( aInitializer );
        return lNewEntity;
    }

    graph_node_t MultiTensorValue( scope_t &aScope, sRandomNormalInitializerComponent const &aInitializer, sTensorShape const &aShape )
    {
        auto lNewEntity = CreateMultiTensor( aScope, aShape );
        lNewEntity.Add<sTypeComponent>( aInitializer.mType );
        lNewEntity.Add<sRandomNormalInitializerComponent>( aInitializer );
        return lNewEntity;
    }

    static inline bool SameType( graph_node_t const &aLeft, graph_node_t const &aRight )
    {
        return ( aLeft.Get<sTypeComponent>().mValue == aRight.Get<sTypeComponent>().mValue );
    }

    static inline bool SameShape( graph_node_t const &aLeft, graph_node_t const &aRight )
    {
        return ( aLeft.Get<sMultiTensorComponent>().Shape() == aRight.Get<sMultiTensorComponent>().Shape() );
    }

    template <typename T>
    static inline bool SameLength( graph_node_t aLeft, graph_node_t const &aRight )
    {
        return ( aLeft.Get<sVectorValueComponent<T>>().mValue.size() == aRight.Get<sVectorValueComponent<T>>().mValue.size() );
    }

    graph_node_t BinaryOperation( scope_t &aScope, scalar_type_t aType, graph_node_t const &aLeft, graph_node_t const &aRight )
    {
        assert( aLeft.Has<sTypeComponent>() );
        assert( aRight.Has<sTypeComponent>() );
        assert( ( aLeft.HasAny<sMultiTensorComponent, sScalarValueVectorComponent, sScalarNodeComponent>() ) );
        assert( ( aRight.HasAny<sMultiTensorComponent, sScalarValueVectorComponent, sScalarNodeComponent>() ) );

        auto  lNewEntity   = aScope.CreateNode();
        auto &lOperandData = lNewEntity.Add<sBinaryOperationComponent>( sBinaryOperationComponent{ aLeft, aRight } );

        if( lOperandData.mLeftOperand.Has<sMultiTensorComponent>() )
        {
            if( lOperandData.mRightOperand.Has<sMultiTensorComponent>() )
            {
                auto lLeftShape  = lOperandData.mLeftOperand.Get<sMultiTensorComponent>().Shape();
                auto lRightShape = lOperandData.mRightOperand.Get<sMultiTensorComponent>().Shape();

                if( lLeftShape != lRightShape )
                {
                    if( lLeftShape.mRank == lRightShape.mRank - 1 )
                    {
                        lRightShape.Trim( -1 );
                        if( lRightShape != lLeftShape )
                            throw std::runtime_error( "Can only add tensors of the same shape" );

                        auto &lBroadcastInfo = lNewEntity.Add<sBroadcastInfoComponent>();
                        lRightShape.Flatten( 0 );
                        lBroadcastInfo.mBroadcastHint = eBroadcastHint::LEFT;
                        lBroadcastInfo.mMaxBlockSize  = lRightShape.mMaxDimensions[0];
                        lBroadcastInfo.mBlockSizes    = VectorValue( aScope, lRightShape.GetDimension( 0 ) );

                        auto lBroadcastShape                  = lOperandData.mRightOperand.Get<sMultiTensorComponent>().Shape();
                        lBroadcastInfo.mBroadcastDimension    = VectorValue( aScope, lBroadcastShape.GetDimension( -1 ) );
                        lBroadcastInfo.mMaxBroadcastDimension = lBroadcastShape.mMaxDimensions[lBroadcastShape.mRank - 1];

                        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, tensor_shape_t( lBroadcastShape.mShape, SizeOf( aType ) ) );
                    }
                    else if( lRightShape.mRank == lLeftShape.mRank - 1 )
                    {
                        lLeftShape.Trim( -1 );
                        if( lRightShape != lLeftShape )
                            throw std::runtime_error( "Can only add tensors of the same shape" );

                        auto &lBroadcastInfo = lNewEntity.Add<sBroadcastInfoComponent>();
                        lRightShape.Flatten( 0 );
                        lBroadcastInfo.mBroadcastHint = eBroadcastHint::RIGHT;
                        lBroadcastInfo.mMaxBlockSize  = lRightShape.mMaxDimensions[0];
                        lBroadcastInfo.mBlockSizes    = VectorValue( aScope, lRightShape.GetDimension( 0 ) );

                        auto lBroadcastShape                  = lOperandData.mLeftOperand.Get<sMultiTensorComponent>().Shape();
                        lBroadcastInfo.mBroadcastDimension    = VectorValue( aScope, lBroadcastShape.GetDimension( -1 ) );
                        lBroadcastInfo.mMaxBroadcastDimension = lBroadcastShape.mMaxDimensions[lBroadcastShape.mRank - 1];

                        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, tensor_shape_t( lBroadcastShape.mShape, SizeOf( aType ) ) );
                    }
                    else
                    {
                        throw std::runtime_error( "Can only add tensors of the same shape" );
                    }
                }
                else
                {
                    auto lShape = lOperandData.mLeftOperand.Get<sMultiTensorComponent>().Shape().mShape;
                    lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, tensor_shape_t( lShape, SizeOf( aType ) ) );
                }
            }
            else
            {
                auto lShape = lOperandData.mLeftOperand.Get<sMultiTensorComponent>().Shape().mShape;
                lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, tensor_shape_t( lShape, SizeOf( aType ) ) );
            }
        }
        else
        {
            if( !( lOperandData.mRightOperand.Has<sMultiTensorComponent>() ) )
            {
                throw std::runtime_error( "RHS should have a tensor" );
            }

            auto lShape = lOperandData.mRightOperand.Get<sMultiTensorComponent>().Shape().mShape;
            lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, tensor_shape_t( lShape, SizeOf( aType ) ) );
        }

        if( lNewEntity.Has<sBroadcastInfoComponent>() )
            lNewEntity.Add<sOperandComponent>( vector_t<graph_node_t>{ aLeft, aRight, lNewEntity.Get<sBroadcastInfoComponent>().mBlockSizes,
                                                                 lNewEntity.Get<sBroadcastInfoComponent>().mBroadcastDimension } );
        else
            lNewEntity.Add<sOperandComponent>( vector_t<graph_node_t>{ aLeft, aRight } );

        lNewEntity.Add<sTypeComponent>( aType );

        return lNewEntity;
    }

    graph_node_t BinaryOperation( scope_t &aScope, graph_node_t const &aLeft, graph_node_t const &aRight )
    {
        return BinaryOperation( aScope, aLeft.Get<sTypeComponent>().mValue, aLeft, aRight );
    }

    graph_node_t Add( scope_t &aScope, graph_node_t const &aLeft, graph_node_t const &aRight )
    {
        auto lNewEntity = BinaryOperation( aScope, aLeft, aRight );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sAddOperationController>();

        return lNewEntity;
    }

    graph_node_t Subtract( scope_t &aScope, graph_node_t const &aLeft, graph_node_t const &aRight )
    {
        auto lNewEntity = BinaryOperation( aScope, aLeft, aRight );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sSubtractOperationController>();

        return lNewEntity;
    }

    graph_node_t Divide( scope_t &aScope, graph_node_t const &aLeft, graph_node_t const &aRight )
    {
        auto lNewEntity = BinaryOperation( aScope, aLeft, aRight );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sDivideOperationController>();

        return lNewEntity;
    }

    graph_node_t Multiply( scope_t &aScope, graph_node_t const &aLeft, graph_node_t const &aRight )
    {
        auto lNewEntity = BinaryOperation( aScope, aLeft, aRight );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sMultiplyOperationController>();

        return lNewEntity;
    }

    graph_node_t And( scope_t &aScope, graph_node_t const &aLeft, graph_node_t const &aRight )
    {
        assert( ( aLeft.Has<sTypeComponent>() ) && ( aRight.Has<sTypeComponent>() ) );
        assert( SameType( aLeft, aRight ) );
        assert( aLeft.Get<sTypeComponent>().mValue == scalar_type_t::UINT8 );

        auto lNewEntity = BinaryOperation( aScope, aLeft, aRight );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sAndOperationController>();

        return lNewEntity;
    }

    graph_node_t Or( scope_t &aScope, graph_node_t const &aLeft, graph_node_t const &aRight )
    {
        assert( ( aLeft.Has<sTypeComponent>() ) && ( aRight.Has<sTypeComponent>() ) );
        assert( SameType( aLeft, aRight ) );
        assert( aLeft.Get<sTypeComponent>().mValue == scalar_type_t::UINT8 );

        auto lNewEntity = BinaryOperation( aScope, aLeft, aRight );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sOrOperationController>();

        return lNewEntity;
    }

    graph_node_t Not( scope_t &aScope, graph_node_t const &aOperand )
    {
        assert( ( aOperand.Has<sTypeComponent>() ) );
        assert( aOperand.Get<sTypeComponent>().mValue == scalar_type_t::UINT8 );

        auto  lNewEntity   = aScope.CreateNode();
        auto &lOperandData = lNewEntity.Add<sNotOperationComponent>( sNotOperationComponent{ aOperand } );

        lNewEntity.Add<sOperandComponent>( vector_t<graph_node_t>{ aOperand } );
        lNewEntity.Add<sTypeComponent>( aOperand.Get<sTypeComponent>() );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, lOperandData.mOperand.Get<sMultiTensorComponent>().Shape() );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sNotOperationController>();

        return lNewEntity;
    }

    graph_node_t BitwiseAnd( scope_t &aScope, graph_node_t const &aLeft, graph_node_t const &aRight )
    {
        assert( ( aLeft.Has<sTypeComponent>() ) && ( aRight.Has<sTypeComponent>() ) );
        assert( SameType( aLeft, aRight ) );
        assert( ( aLeft.Get<sTypeComponent>().mValue >= scalar_type_t::UINT8 ) &&
                ( aLeft.Get<sTypeComponent>().mValue <= scalar_type_t::INT64 ) );

        auto lNewEntity = BinaryOperation( aScope, aLeft, aRight );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sBitwiseAndOperationController>();

        return lNewEntity;
    }

    graph_node_t BitwiseOr( scope_t &aScope, graph_node_t const &aLeft, graph_node_t const &aRight )
    {
        assert( ( aLeft.Has<sTypeComponent>() ) && ( aRight.Has<sTypeComponent>() ) );
        assert( SameType( aLeft, aRight ) );
        assert( ( aLeft.Get<sTypeComponent>().mValue >= scalar_type_t::UINT8 ) &&
                ( aLeft.Get<sTypeComponent>().mValue <= scalar_type_t::INT64 ) );

        auto lNewEntity = BinaryOperation( aScope, aLeft, aRight );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sBitwiseOrOperationController>();

        return lNewEntity;
    }

    graph_node_t BitwiseNot( scope_t &aScope, graph_node_t const &aOperand )
    {
        assert( ( aOperand.Has<sTypeComponent>() ) );
        assert( ( aOperand.Get<sTypeComponent>().mValue >= scalar_type_t::UINT8 ) &&
                ( aOperand.Get<sTypeComponent>().mValue <= scalar_type_t::INT64 ) );

        auto  lNewEntity   = aScope.CreateNode();
        auto &lOperandData = lNewEntity.Add<sBitwiseNotOperationComponent>( sBitwiseNotOperationComponent{ aOperand } );

        lNewEntity.Add<sOperandComponent>( vector_t<graph_node_t>{ aOperand } );
        lNewEntity.Add<sTypeComponent>( aOperand.Get<sTypeComponent>() );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, lOperandData.mOperand.Get<sMultiTensorComponent>().Shape() );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sBitwiseNotOperationController>();

        return lNewEntity;
    }

    graph_node_t InInterval( scope_t &aScope, graph_node_t const &aX, graph_node_t const &aLower, graph_node_t const &aUpper, bool aStrictLower,
                       bool aStrictUpper )
    {
        assert( aX.Has<sTypeComponent>() && ( aLower.Has<sTypeComponent>() ) && ( aUpper.Has<sTypeComponent>() ) );
        assert( ( aX.Has<sMultiTensorComponent>() ) );
        assert( ( aLower.HasAny<sMultiTensorComponent, sScalarValueVectorComponent, sScalarNodeComponent>() ) );
        assert( ( aUpper.HasAny<sMultiTensorComponent, sScalarValueVectorComponent, sScalarNodeComponent>() ) );

        assert( SameType( aX, aLower ) );
        assert( SameType( aX, aUpper ) );

        auto  lNewEntity   = aScope.CreateNode();
        auto &lOperandData = lNewEntity.Add<sInIntervalOperationComponent>(
            sInIntervalOperationComponent{ aX, aLower, aUpper, aStrictLower, aStrictUpper } );

        lNewEntity.Add<sOperandComponent>( vector_t<graph_node_t>{ aX, aLower, aUpper } );
        lNewEntity.Add<sTypeComponent>( scalar_type_t::UINT8 );

        vector_t<vector_t<uint32_t>> lOutputShape = aX.Get<sMultiTensorComponent>().Shape().mShape;

        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, tensor_shape_t( lOutputShape, SizeOf( scalar_type_t::UINT8 ) ) );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sInIntervalOperationController>();

        return lNewEntity;
    }

    graph_node_t Equal( scope_t &aScope, graph_node_t const &aX, graph_node_t const &aY )
    {
        assert( ( aX.Has<sTypeComponent>() ) && ( aY.Has<sTypeComponent>() ) );
        assert( SameType( aX, aY ) );

        auto lNewEntity = BinaryOperation( aScope, scalar_type_t::UINT8, aX, aY );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sEqualOperationController>();

        return lNewEntity;
    }

    graph_node_t LessThan( scope_t &aScope, graph_node_t const &aX, graph_node_t const &aY )
    {
        assert( ( aX.Has<sTypeComponent>() ) && ( aY.Has<sTypeComponent>() ) );
        assert( SameType( aX, aY ) );

        auto lNewEntity = BinaryOperation( aScope, scalar_type_t::UINT8, aX, aY );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sLessThanOperationController>();

        return lNewEntity;
    }

    graph_node_t LessThanOrEqual( scope_t &aScope, graph_node_t const &aX, graph_node_t const &aY )
    {
        assert( ( aX.Has<sTypeComponent>() ) && ( aY.Has<sTypeComponent>() ) );
        assert( SameType( aX, aY ) );

        auto lNewEntity = BinaryOperation( aScope, scalar_type_t::UINT8, aX, aY );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sLessThanOrEqualOperationController>();

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

    graph_node_t Where( scope_t &aScope, graph_node_t const &aCondition, graph_node_t const &aValueIfTrue, graph_node_t const &aValueIfFalse )
    {
        assert( ( aCondition.Has<sTypeComponent>() ) && ( aValueIfTrue.Has<sTypeComponent>() ) &&
                ( aValueIfFalse.Has<sTypeComponent>() ) );
        assert( ( aValueIfTrue.HasAny<sMultiTensorComponent, sScalarValueVectorComponent, sScalarNodeComponent>() ) );
        assert( ( aValueIfFalse.HasAny<sMultiTensorComponent, sScalarValueVectorComponent, sScalarNodeComponent>() ) );
        assert( ( aCondition.Get<sTypeComponent>().mValue == scalar_type_t::UINT8 ) );
        assert( SameType( aValueIfTrue, aValueIfFalse ) );

        auto lNewEntity = aScope.CreateNode();
        lNewEntity.Add<sWhereOperationComponent>( sWhereOperationComponent{ aCondition, aValueIfTrue, aValueIfFalse } );
        lNewEntity.Add<sTypeComponent>( aValueIfTrue.Get<sTypeComponent>() );
        lNewEntity.Add<sOperandComponent>( vector_t<graph_node_t>{ aCondition, aValueIfTrue, aValueIfFalse } );

        auto lShape = aCondition.Get<sMultiTensorComponent>().Shape().mShape;
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool,
                                               tensor_shape_t( lShape, SizeOf( aValueIfTrue.Get<sTypeComponent>().mValue ) ) );

        lNewEntity.Add<sGraphOperationComponent>().Bind<sWhereOperationController>();

        return lNewEntity;
    }

    graph_node_t Mix( scope_t &aScope, graph_node_t const &aA, graph_node_t const &aB, graph_node_t const &aT )
    {
        assert( ( aA.HasAll<sTypeComponent, sMultiTensorComponent>() ) );
        assert( ( aB.HasAll<sTypeComponent, sMultiTensorComponent>() ) );
        assert( ( aT.HasAll<sTypeComponent, sMultiTensorComponent>() ) );

        auto  lNewEntity   = aScope.CreateNode();
        auto &lOperandData = lNewEntity.Add<sMixNodeComponent>( sMixNodeComponent{ aA, aB, aT } );

        lNewEntity.Add<sOperandComponent>( vector_t<graph_node_t>{ aA, aB, aT } );
        lNewEntity.Add<sTypeComponent>( aA.Get<sTypeComponent>() );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, lOperandData.mA.Get<sMultiTensorComponent>().Shape() );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sMixOperationController>();

        return lNewEntity;
    }

    graph_node_t ToFixedPoint( scope_t &aScope, scalar_type_t aOutputType, graph_node_t const &aArray, graph_node_t const &aScaling )
    {
        assert( ( aArray.HasAll<sTypeComponent, sMultiTensorComponent>() ) );
        assert( ( aScaling.HasAll<sTypeComponent, sScalarNodeComponent>() ) );

        auto lNewEntity = aScope.CreateNode();

        auto &lOperandData = lNewEntity.Add<sToFixedPointNodeComponent>( sToFixedPointNodeComponent{ aOutputType, aArray, aScaling } );

        lNewEntity.Add<sOperandComponent>( vector_t<graph_node_t>{ aArray, aScaling } );

        auto &lShape = lOperandData.mArray.Get<sMultiTensorComponent>().Shape().mShape;
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, tensor_shape_t( lShape, SizeOf( aOutputType ) ) );

        lNewEntity.Add<sGraphOperationComponent>().Bind<sToFixedPointOperationController>();

        return lNewEntity;
    }

    graph_node_t Repeat( scope_t &aScope, graph_node_t const &aArray, graph_node_t const &aRepetitions )
    {
        assert( ( aArray.HasAll<sTypeComponent, sMultiTensorComponent>() ) );
        assert( aRepetitions.Has<sU32VectorComponent>() );

        auto  lNewEntity   = aScope.CreateNode();
        auto &lOperandData = lNewEntity.Add<sRepeatOperationComponent>( sRepeatOperationComponent{ aArray, aRepetitions } );

        lNewEntity.Add<sOperandComponent>( vector_t<graph_node_t>{ aArray, aRepetitions } );
        lNewEntity.Add<sTypeComponent>( aArray.Get<sTypeComponent>() );

        auto &lInputShape = lOperandData.mArray.Get<sMultiTensorComponent>().Shape();

        auto                         lRepetitions = aRepetitions.Get<sU32VectorComponent>().mValue;
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

        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool,
                                               tensor_shape_t( lOutputShape, SizeOf( aArray.Get<sTypeComponent>().mValue ) ) );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sArrayOperationController>();

        return lNewEntity;
    }

    graph_node_t Tile( scope_t &aScope, graph_node_t const &aArray, graph_node_t const &aRepetitions )
    {
        assert( ( aArray.HasAll<sTypeComponent, sMultiTensorComponent>() ) );
        assert( aRepetitions.Has<sU32VectorComponent>() );

        auto  lNewEntity   = aScope.CreateNode();
        auto &lOperandData = lNewEntity.Add<sTileOperationComponent>( sTileOperationComponent{ aArray, aRepetitions } );

        lNewEntity.Add<sOperandComponent>( vector_t<graph_node_t>{ aArray, aRepetitions } );
        lNewEntity.Add<sTypeComponent>( aArray.Get<sTypeComponent>() );
        auto &lInputShape = lOperandData.mArray.Get<sMultiTensorComponent>().Shape();

        auto                         lRepetitions = aRepetitions.Get<sU32VectorComponent>().mValue;
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

        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool,
                                               tensor_shape_t( lOutputShape, SizeOf( aArray.Get<sTypeComponent>().mValue ) ) );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sArrayOperationController>();

        return lNewEntity;
    }

    graph_node_t ARange( scope_t &aScope, graph_node_t const &aLeft, graph_node_t const &aRight, graph_node_t const &aDelta )
    {
        assert( aLeft.Has<sScalarValueVectorComponent>() && aRight.Has<sScalarValueVectorComponent>() &&
                aDelta.Has<sScalarValueVectorComponent>() );
        assert( aLeft.Has<sTypeComponent>() && aRight.Has<sTypeComponent>() && aDelta.Has<sTypeComponent>() );

        assert( SameType( aLeft, aRight ) );
        assert( SameType( aLeft, aDelta ) );
        assert( SameLength<scalar_value_t>( aLeft, aRight ) );
        assert( SameLength<scalar_value_t>( aLeft, aDelta ) );

        assert( ( aLeft.Get<sTypeComponent>().mValue == scalar_type_t::FLOAT32 ) ||
                ( aLeft.Get<sTypeComponent>().mValue == scalar_type_t::FLOAT64 ) );

        auto  lNewEntity   = aScope.CreateNode();
        auto &lOperandData = lNewEntity.Add<sARangeComponent>( sARangeComponent{ aLeft, aRight, aDelta } );

        lNewEntity.Add<sTypeComponent>( aLeft.Get<sTypeComponent>() );
        lNewEntity.Add<sOperandComponent>( vector_t<graph_node_t>{ aLeft, aRight, aDelta } );

        vector_t<vector_t<uint32_t>> lOutputShape( aLeft.Get<sScalarValueVectorComponent>().mValue.size() );
        auto                         lLeftValues  = aLeft.Get<sScalarValueVectorComponent>().mValue;
        auto                         lRightValues = aRight.Get<sScalarValueVectorComponent>().mValue;
        auto                         lDeltaValues = aDelta.Get<sScalarValueVectorComponent>().mValue;

        for( uint32_t i = 0; i < aLeft.Get<sScalarValueVectorComponent>().mValue.size(); i++ )
        {
            lOutputShape[i] = { static_cast<uint32_t>( std::ceil(
                ( std::get<float>( lRightValues[i] ) - std::get<float>( lLeftValues[i] ) ) / std::get<float>( lDeltaValues[i] ) ) ) };
        }

        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool,
                                               tensor_shape_t( lOutputShape, SizeOf( aLeft.Get<sTypeComponent>().mValue ) ) );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sARangeOperationController>();

        return lNewEntity;
    }

    graph_node_t LinearSpace( scope_t &aScope, graph_node_t const &aLeft, graph_node_t const &aRight, graph_node_t const &aSubdivisions )
    {
        assert( aLeft.Has<sMultiTensorComponent>() && aRight.Has<sMultiTensorComponent>() &&
                aSubdivisions.Has<sU32VectorComponent>() );

        assert( SameShape( aLeft, aRight ) );

        assert( aLeft.Get<sMultiTensorComponent>().Shape().CountLayers() == aSubdivisions.Get<sU32VectorComponent>().mValue.size() );
        assert( SameType( aLeft, aRight ) );
        assert( ( aLeft.Get<sTypeComponent>().mValue == scalar_type_t::FLOAT32 ) ||
                ( aLeft.Get<sTypeComponent>().mValue == scalar_type_t::FLOAT64 ) );

        auto  lNewEntity   = aScope.CreateNode();
        auto &lOperandData = lNewEntity.Add<sLinearSpaceComponent>( sLinearSpaceComponent{ aLeft, aRight, aSubdivisions } );

        lNewEntity.Add<sTypeComponent>( aLeft.Get<sTypeComponent>() );
        lNewEntity.Add<sOperandComponent>( vector_t<graph_node_t>{ aLeft, aRight, aSubdivisions } );

        auto                         lSubdivisions = aSubdivisions.Get<sU32VectorComponent>().mValue;
        vector_t<vector_t<uint32_t>> lOutputShape( aLeft.Get<sMultiTensorComponent>().Shape().mShape.size() );

        for( uint32_t i = 0; i < aLeft.Get<sMultiTensorComponent>().Shape().mShape.size(); i++ )
        {
            lOutputShape[i] = aLeft.Get<sMultiTensorComponent>().Shape().mShape[i];
            lOutputShape[i].push_back( lSubdivisions[i] );
        }

        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool,
                                               tensor_shape_t( lOutputShape, SizeOf( aLeft.Get<sTypeComponent>().mValue ) ) );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sLinearSpaceOperationController>();

        return lNewEntity;
    }

    graph_node_t Sample2D( scope_t &aScope, graph_node_t const &aX, graph_node_t const &aY, graph_node_t const &aTextures )
    {
        assert( ( aX.HasAny<sMultiTensorComponent, sScalarNodeComponent, sScalarValueVectorComponent>() ) );
        assert( ( aY.HasAny<sMultiTensorComponent, sScalarNodeComponent, sScalarValueVectorComponent>() ) );

        assert( aX.Has<sMultiTensorComponent>() || aY.Has<sMultiTensorComponent>() );
        assert( aTextures.Has<sVectorValueComponent<Cuda::texture_sampler2d_t::DeviceData>>() );

        if( aX.Has<sMultiTensorComponent>() && aY.Has<sMultiTensorComponent>() )
            assert( SameShape( aX, aY ) );

        if( aX.Has<sMultiTensorComponent>() )
            assert( aX.Get<sMultiTensorComponent>().Shape().CountLayers() ==
                    aTextures.Get<sVectorValueComponent<Cuda::texture_sampler2d_t::DeviceData>>().mValue.size() );

        if( aY.Has<sMultiTensorComponent>() )
            assert( aY.Get<sMultiTensorComponent>().Shape().CountLayers() ==
                    aTextures.Get<sVectorValueComponent<Cuda::texture_sampler2d_t::DeviceData>>().mValue.size() );

        assert( SameType( aX, aY ) );

        assert( aX.Get<sTypeComponent>().mValue == scalar_type_t::FLOAT32 );

        auto  lNewEntity   = aScope.CreateNode();
        auto &lOperandData = lNewEntity.Add<sSample2DComponent>( sSample2DComponent{ aX, aY, aTextures } );

        lNewEntity.Add<sTypeComponent>( aX.Get<sTypeComponent>() );
        lNewEntity.Add<sOperandComponent>( vector_t<graph_node_t>{ aX, aY, aTextures } );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, lOperandData.mX.Get<sMultiTensorComponent>().Shape() );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sSample2DOperationController>();

        return lNewEntity;
    }

    graph_node_t AffineTransform( scope_t &aScope, graph_node_t const &aA, graph_node_t const &aX, graph_node_t const &aB )
    {
        assert( ( aX.HasAll<sMultiTensorComponent, sTypeComponent>() ) );
        assert( ( aA.Has<sTypeComponent>() && aB.Has<sTypeComponent>() ) );
        assert( ( aA.HasAny<sMultiTensorComponent, sScalarNodeComponent, sScalarValueVectorComponent>() ) );
        assert( ( aB.HasAny<sMultiTensorComponent, sScalarNodeComponent, sScalarValueVectorComponent>() ) );
        assert( SameType( aX, aA ) );
        assert( SameType( aX, aB ) );

        auto  lNewEntity   = aScope.CreateNode();
        auto &lOperandData = lNewEntity.Add<sAffineNodeComponent>( sAffineNodeComponent{ aA, aX, aB } );
        lNewEntity.Add<sTypeComponent>( aX.Get<sTypeComponent>() );

        lNewEntity.Add<sOperandComponent>( vector_t<graph_node_t>{ aA, aX, aB } );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, lOperandData.mX.Get<sMultiTensorComponent>().Shape() );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sAffineNodeController>();

        return lNewEntity;
    }

    graph_node_t Collapse( scope_t &aScope, graph_node_t const &aArray )
    {
        assert( ( aArray.Has<sMultiTensorComponent>() ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<sOperandComponent>( vector_t<graph_node_t>{ aArray } );

        auto &lInputShape = aArray.Get<sMultiTensorComponent>().Shape();
        for( uint32_t i = 0; i < lInputShape.mShape.size(); i++ )
        {
            if( lInputShape.mShape[i] != lInputShape.mShape[0] )
                throw std::runtime_error( "All dimensions should be equal" );
        }

        vector_t<uint32_t> lOutputDimension( lInputShape.mRank + 1 );
        lOutputDimension[0] = lInputShape.CountLayers();
        std::copy( lInputShape.mShape[0].begin(), lInputShape.mShape[0].end(), lOutputDimension.begin() + 1 );

        lNewEntity.Add<sTypeComponent>( aArray.Get<sTypeComponent>() );
        lNewEntity.Add<sMultiTensorComponent>(
            aScope.mPool, aArray.Get<sMultiTensorComponent>().mValue.GetMemoryBuffer(),
            tensor_shape_t( vector_t<vector_t<uint32_t>>{ lOutputDimension }, static_cast<size_t>( lInputShape.mElementSize ) ) );

        return lNewEntity;
    }

    graph_node_t Expand( scope_t &aScope, graph_node_t const &aArray )
    {
        auto lNewEntity = aScope.CreateNode();
        assert( ( aArray.Has<sMultiTensorComponent>() ) );

        lNewEntity.Add<sOperandComponent>( vector_t<graph_node_t>{ aArray } );

        auto &lInputShape = aArray.Get<sMultiTensorComponent>().Shape();

        assert( lInputShape.CountLayers() == 1 );

        vector_t<vector_t<uint32_t>> lOutputShape( lInputShape.mShape[0][0] );

        for( uint32_t i = 0; i < lOutputShape.size(); i++ )
        {
            lOutputShape[i] = vector_t<uint32_t>( lInputShape.mRank - 1 );
            std::copy( lInputShape.mShape[0].begin() + 1, lInputShape.mShape[0].end(), lOutputShape[i].begin() );
        }

        lNewEntity.Add<sTypeComponent>( aArray.Get<sTypeComponent>() );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, aArray.Get<sMultiTensorComponent>().mValue.GetMemoryBuffer(),
                                               tensor_shape_t( lOutputShape, static_cast<size_t>( lInputShape.mElementSize ) ) );

        return lNewEntity;
    }

    graph_node_t Reshape( scope_t &aScope, graph_node_t const &aArray, sTensorShape &aNewShape )
    {
        auto lNewEntity = aScope.CreateNode();
        assert( ( aArray.Has<sMultiTensorComponent>() ) );

        auto &lInputShape = aArray.Get<sMultiTensorComponent>().Shape();

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

        lNewEntity.Add<sTypeComponent>( aArray.Get<sTypeComponent>() );
        lNewEntity.Add<sOperandComponent>( vector_t<graph_node_t>{ aArray } );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, aArray.Get<sMultiTensorComponent>().mValue.GetMemoryBuffer(), aNewShape );

        return lNewEntity;
    }

    graph_node_t Relayout( scope_t &aScope, graph_node_t const &aArray, sTensorShape &aNewLayout )
    {
        auto lNewEntity = aScope.CreateNode();
        assert( ( aArray.Has<sMultiTensorComponent>() ) );

        auto &lInputShape = aArray.Get<sMultiTensorComponent>().Shape();

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

        if( aArray.Has<sTypeComponent>() )
            lNewEntity.Add<sTypeComponent>( aArray.Get<sTypeComponent>() );

        lNewEntity.Add<sOperandComponent>( vector_t<graph_node_t>{ aArray } );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, aArray.Get<sMultiTensorComponent>().mValue.GetMemoryBuffer(),
                                               aNewLayout );

        return lNewEntity;
    }

    graph_node_t Flatten( scope_t &aScope, graph_node_t const &aArray )
    {
        auto lNewEntity = aScope.CreateNode();
        assert( ( aArray.Has<sMultiTensorComponent>() ) );

        auto lInputShape = aArray.Get<sMultiTensorComponent>().Shape();

        lInputShape.Flatten( 0 );
        lNewEntity.Add<sTypeComponent>( aArray.Get<sTypeComponent>() );
        lNewEntity.Add<sOperandComponent>( vector_t<graph_node_t>{ aArray } );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, aArray.Get<sMultiTensorComponent>().mValue.GetMemoryBuffer(),
                                               lInputShape );

        return lNewEntity;
    }

    graph_node_t Slice( scope_t &aScope, graph_node_t const &aArray, graph_node_t const &aBegin, graph_node_t const &aEnd )
    {
        assert( ( aArray.HasAll<sTypeComponent, sMultiTensorComponent>() ) );
        assert( ( ( aBegin.HasAny<sVectorValueComponent<uint32_t>, sScalarNodeComponent>() ) ) &&
                ( aEnd.HasAny<sVectorValueComponent<uint32_t>, sScalarNodeComponent>() ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<sTypeComponent>( aArray.Get<sTypeComponent>() );
        auto &lOperandData  = lNewEntity.Add<sArraySliceNodeComponent>();
        lOperandData.mArray = aArray;

        auto lInputShape = aArray.Get<sMultiTensorComponent>().Shape();

        uint32_t           lMaxBlockSize = 0;
        vector_t<uint32_t> lBlockSizes( lInputShape.CountLayers() );

        vector_t<uint32_t> lBegin;
        if( aBegin.Has<sVectorValueComponent<uint32_t>>() )
        {
            lBegin              = aBegin.Get<sVectorValueComponent<uint32_t>>().mValue;
            lOperandData.mBegin = aBegin;
        }
        else
        {
            lBegin = vector_t<uint32_t>( lInputShape.CountLayers(), std::get<uint32_t>( aBegin.Get<sScalarNodeComponent>().mValue ) );
            lOperandData.mBegin = VectorValue( aScope, lBegin );
        }

        vector_t<uint32_t> lEnd;
        if( aEnd.Has<sVectorValueComponent<uint32_t>>() )
        {
            lEnd              = aEnd.Get<sVectorValueComponent<uint32_t>>().mValue;
            lOperandData.mEnd = aEnd;
        }
        else
        {
            lEnd = vector_t<uint32_t>( lInputShape.CountLayers(), std::get<uint32_t>( aEnd.Get<sScalarNodeComponent>().mValue ) );
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

        lNewEntity.Add<sOperandComponent>(
            vector_t<graph_node_t>{ aArray, lOperandData.mBegin, lOperandData.mEnd, lOperandData.mBlockSizes, lOperandData.mElementCount } );

        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool,
                                               tensor_shape_t( lOutputShape, SizeOf( aArray.Get<sTypeComponent>().mValue ) ) );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sArraySliceOperationController>();

        return lNewEntity;
    }

    graph_node_t Summation( scope_t &aScope, graph_node_t const &aArray )
    {
        auto               lInputShape = aArray.Get<sMultiTensorComponent>().Shape();
        vector_t<uint32_t> lLastDimensions( lInputShape.CountLayers() );
        for( uint32_t i = 0; i < lInputShape.CountLayers(); i++ )
            lLastDimensions[i] = lInputShape.mShape[i][lInputShape.mShape[i].size() - 1] - 1;

        auto lZero = ConstantScalarValue( aScope, static_cast<uint32_t>( 0 ) );
        auto lEnd  = VectorValue( aScope, lLastDimensions );

        return Summation( aScope, aArray, lZero, lEnd );
    }

    graph_node_t Summation( scope_t &aScope, graph_node_t const &aArray, graph_node_t const &aBegin, graph_node_t const &aEnd )
    {
        assert( ( aArray.HasAll<sTypeComponent, sMultiTensorComponent>() ) );
        assert( ( ( aBegin.HasAny<sVectorValueComponent<uint32_t>, sScalarNodeComponent>() ) ) &&
                ( aEnd.HasAny<sVectorValueComponent<uint32_t>, sScalarNodeComponent>() ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<sTypeComponent>( aArray.Get<sTypeComponent>() );
        auto &lOperandData  = lNewEntity.Add<sArraySummationNodeComponent>();
        lOperandData.mArray = aArray;

        auto lInputShape = aArray.Get<sMultiTensorComponent>().Shape();

        if( aBegin.Has<sVectorValueComponent<uint32_t>>() )
        {
            lOperandData.mBegin = aBegin;
        }
        else
        {
            vector_t<uint32_t> lBegin( lInputShape.CountLayers(), std::get<uint32_t>( aBegin.Get<sScalarNodeComponent>().mValue ) );
            lOperandData.mBegin = VectorValue( aScope, lBegin );
        }

        if( aEnd.Has<sVectorValueComponent<uint32_t>>() )
        {
            lOperandData.mEnd = aEnd;
        }
        else
        {
            vector_t<uint32_t> lEnd( lInputShape.CountLayers(), std::get<uint32_t>( aEnd.Get<sScalarNodeComponent>().mValue ) );
            lOperandData.mEnd = VectorValue( aScope, lEnd );
        }

        auto lOutputShape = lInputShape;
        lOutputShape.Trim( -1 );

        lInputShape.Flatten( -1 );
        lOperandData.mMaxBlockSize = lInputShape.mMaxDimensions[0];
        lOperandData.mBlockSizes   = VectorValue( aScope, lInputShape.GetDimension( 0 ) );
        lOperandData.mElementCount = VectorValue( aScope, lInputShape.GetDimension( -1 ) );

        lNewEntity.Add<sOperandComponent>(
            vector_t<graph_node_t>{ aArray, lOperandData.mBegin, lOperandData.mEnd, lOperandData.mBlockSizes, lOperandData.mElementCount } );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool,
                                               tensor_shape_t( lOutputShape.mShape, SizeOf( aArray.Get<sTypeComponent>().mValue ) ) );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sArraySummationOperationController>();

        return lNewEntity;
    }

    graph_node_t CountTrue( scope_t &aScope, graph_node_t const &aArray )
    {
        assert( ( aArray.HasAll<sTypeComponent, sMultiTensorComponent>() ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<sTypeComponent>( sTypeComponent{ scalar_type_t::UINT32 } );
        auto &lOperandData  = lNewEntity.Add<sCountTrueNodeComponent>();
        lOperandData.mArray = aArray;

        auto lInputShape  = aArray.Get<sMultiTensorComponent>().Shape();
        auto lOutputShape = lInputShape;
        lOutputShape.Trim( -1 );

        lInputShape.Flatten( -1 );
        lOperandData.mMaxBlockSize = lInputShape.mMaxDimensions[0];
        lOperandData.mBlockSizes   = VectorValue( aScope, lInputShape.GetDimension( 0 ) );
        lOperandData.mElementCount = VectorValue( aScope, lInputShape.GetDimension( -1 ) );

        lNewEntity.Add<sOperandComponent>( vector_t<graph_node_t>{ aArray, lOperandData.mBlockSizes, lOperandData.mElementCount } );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, tensor_shape_t( lOutputShape.mShape, SizeOf( scalar_type_t::UINT32 ) ) );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sCountTrueOperationController>();

        return lNewEntity;
    }

    graph_node_t CountZero( scope_t &aScope, graph_node_t const &aArray )
    {
        assert( ( aArray.HasAll<sTypeComponent, sMultiTensorComponent>() ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<sTypeComponent>( sTypeComponent{ scalar_type_t::UINT32 } );
        auto &lOperandData  = lNewEntity.Add<sCountZeroNodeComponent>();
        lOperandData.mArray = aArray;

        auto lInputShape = aArray.Get<sMultiTensorComponent>().Shape();

        auto lOutputShape = lInputShape;
        lOutputShape.Trim( -1 );

        lInputShape.Flatten( -1 );
        lOperandData.mMaxBlockSize = lInputShape.mMaxDimensions[0];
        lOperandData.mBlockSizes   = VectorValue( aScope, lInputShape.GetDimension( 0 ) );
        lOperandData.mElementCount = VectorValue( aScope, lInputShape.GetDimension( -1 ) );

        lNewEntity.Add<sOperandComponent>( vector_t<graph_node_t>{ aArray, lOperandData.mBlockSizes, lOperandData.mElementCount } );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, tensor_shape_t( lOutputShape.mShape, SizeOf( scalar_type_t::UINT32 ) ) );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sCountZeroOperationController>();

        return lNewEntity;
    }

    graph_node_t CountNonZero( scope_t &aScope, graph_node_t const &aArray )
    {
        assert( ( aArray.HasAll<sTypeComponent, sMultiTensorComponent>() ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<sTypeComponent>( sTypeComponent{ scalar_type_t::UINT32 } );
        auto &lOperandData  = lNewEntity.Add<sCountNonZeroNodeComponent>();
        lOperandData.mArray = aArray;

        auto lInputShape = aArray.Get<sMultiTensorComponent>().Shape();

        auto lOutputShape = lInputShape;
        lOutputShape.Trim( -1 );

        lInputShape.Flatten( -1 );
        lOperandData.mMaxBlockSize = lInputShape.mMaxDimensions[0];
        lOperandData.mBlockSizes   = VectorValue( aScope, lInputShape.GetDimension( 0 ) );
        lOperandData.mElementCount = VectorValue( aScope, lInputShape.GetDimension( -1 ) );

        lNewEntity.Add<sOperandComponent>( vector_t<graph_node_t>{ aArray, lOperandData.mBlockSizes, lOperandData.mElementCount } );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, tensor_shape_t( lOutputShape.mShape, SizeOf( scalar_type_t::UINT32 ) ) );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sCountNonZeroOperationController>();

        return lNewEntity;
    }

    graph_node_t Diff( scope_t &aScope, graph_node_t const &aArray, uint32_t aCount )
    {
        assert( ( aArray.HasAll<sTypeComponent, sMultiTensorComponent>() ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<sTypeComponent>( aArray.Get<sTypeComponent>() );
        auto &lOperandData  = lNewEntity.Add<sDiffNodeComponent>();
        lOperandData.mArray = aArray;
        lOperandData.mCount = aCount;

        auto lInputShape  = aArray.Get<sMultiTensorComponent>().Shape();
        auto lOutputShape = aArray.Get<sMultiTensorComponent>().Shape();

        lInputShape.Flatten( -1 );
        lOperandData.mMaxBlockSize = lInputShape.mMaxDimensions[0];
        lOperandData.mBlockSizes   = VectorValue( aScope, lInputShape.GetDimension( 0 ) );
        lOperandData.mElementCount = VectorValue( aScope, lInputShape.GetDimension( -1 ) );

        lNewEntity.Add<sOperandComponent>( vector_t<graph_node_t>{ aArray, lOperandData.mBlockSizes, lOperandData.mElementCount } );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, lOutputShape );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sDiffOperationController>();

        return lNewEntity;
    }

    graph_node_t Shift( scope_t &aScope, graph_node_t const &aArray, int32_t aCount, graph_node_t const &aFillValue )
    {
        assert( ( aArray.HasAll<sTypeComponent, sMultiTensorComponent>() ) );
        assert( ( aFillValue.HasAll<sTypeComponent, sScalarNodeComponent>() ) );
        assert( SameType( aFillValue, aArray ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<sTypeComponent>( aArray.Get<sTypeComponent>() );
        auto &lOperandData      = lNewEntity.Add<sShiftNodeComponent>();
        lOperandData.mArray     = aArray;
        lOperandData.mCount     = aCount;
        lOperandData.mFillValue = aFillValue;

        auto lInputShape  = aArray.Get<sMultiTensorComponent>().Shape();
        auto lOutputShape = aArray.Get<sMultiTensorComponent>().Shape();

        lInputShape.Flatten( -1 );
        lOperandData.mMaxBlockSize = lInputShape.mMaxDimensions[0];
        lOperandData.mBlockSizes   = VectorValue( aScope, lInputShape.GetDimension( 0 ) );
        lOperandData.mElementCount = VectorValue( aScope, lInputShape.GetDimension( -1 ) );

        lNewEntity.Add<sOperandComponent>(
            vector_t<graph_node_t>{ aArray, lOperandData.mFillValue, lOperandData.mBlockSizes, lOperandData.mElementCount } );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, lOutputShape );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sShiftOperationController>();

        return lNewEntity;
    }

    graph_node_t Floor( scope_t &aScope, graph_node_t const &aArray )
    {
        assert( ( aArray.HasAll<sTypeComponent, sMultiTensorComponent>() ) );
        assert( ( aArray.Get<sTypeComponent>().mValue == scalar_type_t::FLOAT32 ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<sOperandComponent>( vector_t<graph_node_t>{ aArray } );
        lNewEntity.Add<sTypeComponent>( aArray.Get<sTypeComponent>() );
        lNewEntity.Add<sFloorNodeComponent>( sFloorNodeComponent{ aArray } );

        auto lShape = aArray.Get<sMultiTensorComponent>().Shape().mShape;
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, tensor_shape_t( lShape, SizeOf( aArray.Get<sTypeComponent>().mValue ) ) );

        lNewEntity.Add<sGraphOperationComponent>().Bind<sFloorOperationController>();

        return lNewEntity;
    }

    graph_node_t Ceil( scope_t &aScope, graph_node_t const &aArray )
    {
        assert( ( aArray.HasAll<sTypeComponent, sMultiTensorComponent>() ) );
        assert( ( aArray.Get<sTypeComponent>().mValue == scalar_type_t::FLOAT32 ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<sOperandComponent>( vector_t<graph_node_t>{ aArray } );
        lNewEntity.Add<sTypeComponent>( aArray.Get<sTypeComponent>() );
        lNewEntity.Add<sCeilNodeComponent>( sCeilNodeComponent{ aArray } );

        auto lShape = aArray.Get<sMultiTensorComponent>().Shape().mShape;
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, tensor_shape_t( lShape, SizeOf( aArray.Get<sTypeComponent>().mValue ) ) );

        lNewEntity.Add<sGraphOperationComponent>().Bind<sCeilOperationController>();

        return lNewEntity;
    }

    graph_node_t Abs( scope_t &aScope, graph_node_t const &aArray )
    {
        assert( ( aArray.HasAll<sTypeComponent, sMultiTensorComponent>() ) );
        assert( ( aArray.Get<sTypeComponent>().mValue == scalar_type_t::FLOAT32 ) ||
                ( ( aArray.Get<sTypeComponent>().mValue >= scalar_type_t::INT8 ) &&
                  ( aArray.Get<sTypeComponent>().mValue <= scalar_type_t::INT64 ) ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<sOperandComponent>( vector_t<graph_node_t>{ aArray } );
        lNewEntity.Add<sTypeComponent>( aArray.Get<sTypeComponent>() );
        lNewEntity.Add<sAbsNodeComponent>( sAbsNodeComponent{ aArray } );

        auto lShape = aArray.Get<sMultiTensorComponent>().Shape().mShape;
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, tensor_shape_t( lShape, SizeOf( aArray.Get<sTypeComponent>().mValue ) ) );

        lNewEntity.Add<sGraphOperationComponent>().Bind<sAbsOperationController>();

        return lNewEntity;
    }

    graph_node_t Sqrt( scope_t &aScope, graph_node_t const &aArray )
    {
        assert( ( aArray.HasAll<sTypeComponent, sMultiTensorComponent>() ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<sOperandComponent>( vector_t<graph_node_t>{ aArray } );
        lNewEntity.Add<sTypeComponent>( aArray.Get<sTypeComponent>() );
        lNewEntity.Add<sSqrtNodeComponent>( sSqrtNodeComponent{ aArray } );

        auto lShape = aArray.Get<sMultiTensorComponent>().Shape().mShape;
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, tensor_shape_t( lShape, SizeOf( aArray.Get<sTypeComponent>().mValue ) ) );

        lNewEntity.Add<sGraphOperationComponent>().Bind<sSqrtOperationController>();

        return lNewEntity;
    }

    graph_node_t Round( scope_t &aScope, graph_node_t const &aArray )
    {
        assert( ( aArray.HasAll<sTypeComponent, sMultiTensorComponent>() ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<sOperandComponent>( vector_t<graph_node_t>{ aArray } );
        lNewEntity.Add<sTypeComponent>( aArray.Get<sTypeComponent>() );
        lNewEntity.Add<sRoundNodeComponent>( sRoundNodeComponent{ aArray } );

        auto lShape = aArray.Get<sMultiTensorComponent>().Shape().mShape;
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, tensor_shape_t( lShape, SizeOf( aArray.Get<sTypeComponent>().mValue ) ) );

        lNewEntity.Add<sGraphOperationComponent>().Bind<sRoundOperationController>();

        return lNewEntity;
    }

    graph_node_t Conv1D( scope_t &aScope, graph_node_t const &aArray0, graph_node_t const &aArray1 )
    {
        assert( ( aArray0.HasAll<sTypeComponent, sMultiTensorComponent>() ) );
        assert( ( aArray1.HasAll<sTypeComponent, sMultiTensorComponent>() ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<sTypeComponent>( aArray0.Get<sTypeComponent>() );
        auto &lOperandData   = lNewEntity.Add<sConv1DNodeComponent>();
        lOperandData.mArray0 = aArray0;
        lOperandData.mArray1 = aArray1;

        auto lInputShape  = aArray0.Get<sMultiTensorComponent>().Shape();
        auto lOutputShape = aArray0.Get<sMultiTensorComponent>().Shape();

        lInputShape.Flatten( -1 );
        lOperandData.mMaxBlockSize0    = lInputShape.mMaxDimensions[0];
        lOperandData.mMaxElementCount0 = lInputShape.mMaxDimensions[lInputShape.mRank - 1];
        lOperandData.mBlockSizes0      = VectorValue( aScope, lInputShape.GetDimension( 0 ) );
        lOperandData.mElementCount0    = VectorValue( aScope, lInputShape.GetDimension( -1 ) );

        auto lKernelShape = aArray1.Get<sMultiTensorComponent>().Shape();

        lKernelShape.Flatten( -1 );
        lOperandData.mMaxBlockSize1 = lKernelShape.mMaxDimensions[0];
        lOperandData.mBlockSizes1   = VectorValue( aScope, lKernelShape.GetDimension( 0 ) );
        lOperandData.mElementCount1 = VectorValue( aScope, lKernelShape.GetDimension( -1 ) );

        lNewEntity.Add<sOperandComponent>( vector_t<graph_node_t>{ lOperandData.mArray0, lOperandData.mBlockSizes0,
                                                             lOperandData.mElementCount0, lOperandData.mArray1,
                                                             lOperandData.mBlockSizes1, lOperandData.mElementCount1 } );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, lOutputShape );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sConv1DOperationController>();

        return lNewEntity;
    }

    graph_node_t HCat( scope_t &aScope, graph_node_t const &aArray0, graph_node_t const &aArray1 )
    {
        assert( ( aArray0.HasAll<sTypeComponent, sMultiTensorComponent>() ) );
        assert( ( aArray1.HasAll<sTypeComponent, sMultiTensorComponent>() ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<sTypeComponent>( aArray0.Get<sTypeComponent>() );
        auto &lOperandData   = lNewEntity.Add<sHCatNodeComponent>();
        lOperandData.mArray0 = aArray0;
        lOperandData.mArray1 = aArray1;

        auto lInputShape0 = aArray0.Get<sMultiTensorComponent>().Shape();
        auto lInputShape1 = aArray1.Get<sMultiTensorComponent>().Shape();

        auto               lLastDim0 = lInputShape0.GetDimension( -1 );
        auto               lLastDim1 = lInputShape1.GetDimension( -1 );
        vector_t<uint32_t> lConcatenated{};
        for( uint32_t i = 0; i < lLastDim0.size(); i++ )
            lConcatenated.push_back( lLastDim0[i] + lLastDim1[i] );

        auto lOutputShape = aArray0.Get<sMultiTensorComponent>().Shape();
        lOutputShape.Trim( -1 );
        lOutputShape.InsertDimension( -1, lConcatenated );

        auto lBlockShape = aArray0.Get<sMultiTensorComponent>().Shape();
        lBlockShape.Flatten( -1 );
        lOperandData.mMaxBlockSize  = lBlockShape.mMaxDimensions[0];
        lOperandData.mBlockSizes    = VectorValue( aScope, lBlockShape.GetDimension( 0 ) );
        lOperandData.mElementCount0 = VectorValue( aScope, lInputShape0.GetDimension( -1 ) );
        lOperandData.mElementCount1 = VectorValue( aScope, lInputShape1.GetDimension( -1 ) );

        lNewEntity.Add<sOperandComponent>( vector_t<graph_node_t>{ lOperandData.mArray0, lOperandData.mArray1, lOperandData.mBlockSizes,
                                                             lOperandData.mElementCount0, lOperandData.mElementCount1 } );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, lOutputShape );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sHCatOperationController>();

        return lNewEntity;
    }

} // namespace SE::TensorOps
