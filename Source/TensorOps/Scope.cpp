/// @file   Scope.cpp
///
/// @brief  Definitions for computation scope
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#include "Scope.h"

#include "Cuda/Texture2D.h"

#include "Implementation/KernelLaunchers.h"

namespace SE::TensorOps
{
    using namespace SE::Cuda;

    Scope::Scope( uint32_t aMemorySize ) { mPool = MemoryPool( aMemorySize ); }

    Scope &Scope::WithOpName( const std::string &aName )
    {
        mName = aName;
        return *this;
    }

    OpNode Scope::CreateNode()
    {
        OpNode lNewEntity;
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

    OpNode Scope::operator[]( const std::string &aNodeName )
    {
        if( mNamedNodes.find( aNodeName ) != mNamedNodes.end() ) return mNamedNodes[aNodeName];
        return OpNode{};
    }

    void Scope::Reset()
    {
        mPool.Reset();
        mNodesRegistry.Clear();
        mNamedNodes.clear();
        mName.reset();
    }

    void Scope::Run( OpNode const &aNode ) { Run( std::vector<OpNode>{ aNode } ); }

    void Scope::Run( std::vector<OpNode> const &aNode )
    {
        std::deque<OpNode>                      lExecutionQueue;
        std::stack<OpNode, std::vector<OpNode>> lStack( aNode );

        while( !lStack.empty() )
        {
            OpNode lCurrent = lStack.top();
            lStack.pop();

            if( lCurrent.Has<sDoNotExpand>() ) continue;

            std::deque<OpNode>::iterator lPos = std::find( lExecutionQueue.begin(), lExecutionQueue.end(), lCurrent );
            if( lPos != lExecutionQueue.end() )
            {
                lExecutionQueue.erase( lPos );
            }
            lExecutionQueue.push_back( lCurrent );
            if( lCurrent.Has<sOperandComponent>() )
            {
                for( OpNode lDependent : lCurrent.Get<sOperandComponent>().mOperands )
                {
                    lStack.push( lDependent );
                }
            }
        }

        // Allocate memory for tensors which are on the stack
        for( auto lElement = lExecutionQueue.rbegin(); lElement < lExecutionQueue.rend(); lElement++ )
        {
            if( ( *lElement ).Has<sAllocatedTag>() ) continue;

            if( ( *lElement ).Has<sMultiTensorComponent>() )
            {
                ( *lElement ).Get<sMultiTensorComponent>().mValue =
                    MultiTensor( mPool, ( *lElement ).Get<sMultiTensorComponent>().mShape );
                ( *lElement ).Tag<sAllocatedTag>();
            }

            if( ( *lElement ).Has<sVectorBufferComponent>() )
            {
                ( *lElement ).Get<sVectorBufferComponent>().mValue =
                    mPool.Allocate( ( *lElement ).Get<sVectorBufferComponent>().mSize );
                ( *lElement ).Tag<sAllocatedTag>();
            }
        }

        for( auto lElement = lExecutionQueue.rbegin(); lElement < lExecutionQueue.rend(); lElement++ )
        {
            if( !( *lElement ).Has<sGraphOperationComponent>() ) continue;

            auto &lComponent = ( *lElement ).Get<sGraphOperationComponent>();
            if( !lComponent.mControllerInstance )
            {
                lComponent.mControllerInstance = lComponent.mInstantiateController();
                lComponent.mControllerInstance->Initialize( *lElement );
            }

            lComponent.mControllerInstance->Run();
        }
        CUDA_SYNC_CHECK();
    }

    OpNode CreateMultiTensor( Scope &aScope, sTensorShape const &aShape )
    {
        auto lNewEntity = aScope.CreateNode();
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, aShape );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sMultiTensorRunner>();
        return lNewEntity;
    }

    OpNode MultiTensorValue( Scope &aScope, sConstantValueInitializerComponent const &aInitializer, sTensorShape const &aShape )
    {
        auto lNewEntity = CreateMultiTensor( aScope, aShape );
        lNewEntity.Add<sTypeComponent>( TypeOf( aInitializer.mValue ) );
        lNewEntity.Add<sConstantValueInitializerComponent>( aInitializer );
        return lNewEntity;
    }

    OpNode MultiTensorValue( Scope &aScope, sVectorInitializerComponent const &aInitializer, sTensorShape const &aShape )
    {
        auto lNewEntity = CreateMultiTensor( aScope, aShape );
        lNewEntity.Add<sTypeComponent>( TypeOf( aInitializer.mValue[0] ) );
        auto &lInitializerComponent = lNewEntity.Add<sVectorInitializerComponent>( aInitializer );
        lInitializerComponent.mData = aScope.mPool.Allocate( aInitializer.mValue.size() * SizeOf( TypeOf( aInitializer.mValue[0] ) ) );
        return lNewEntity;
    }

    OpNode MultiTensorValue( Scope &aScope, sDataInitializerComponent const &aInitializer, sTensorShape const &aShape )
    {
        auto lNewEntity = CreateMultiTensor( aScope, aShape );
        lNewEntity.Add<sTypeComponent>( TypeOf( aInitializer.mValue[0] ) );
        auto &lInitializerComponent = lNewEntity.Add<sDataInitializerComponent>( aInitializer );
        return lNewEntity;
    }

    OpNode MultiTensorValue( Scope &aScope, sRandomUniformInitializerComponent const &aInitializer, sTensorShape const &aShape )
    {
        auto lNewEntity = CreateMultiTensor( aScope, aShape );
        lNewEntity.Add<sTypeComponent>( aInitializer.mType );
        lNewEntity.Add<sRandomUniformInitializerComponent>( aInitializer );
        return lNewEntity;
    }

    OpNode MultiTensorValue( Scope &aScope, sRandomNormalInitializerComponent const &aInitializer, sTensorShape const &aShape )
    {
        auto lNewEntity = CreateMultiTensor( aScope, aShape );
        lNewEntity.Add<sTypeComponent>( aInitializer.mType );
        lNewEntity.Add<sRandomNormalInitializerComponent>( aInitializer );
        return lNewEntity;
    }

    static inline bool SameType( OpNode const &aLeft, OpNode const &aRight )
    {
        return ( aLeft.Get<sTypeComponent>().mValue == aRight.Get<sTypeComponent>().mValue );
    }

    static inline bool SameShape( OpNode const &aLeft, OpNode const &aRight )
    {
        return ( aLeft.Get<sMultiTensorComponent>().Shape() == aRight.Get<sMultiTensorComponent>().Shape() );
    }

    template <typename T>
    static inline bool SameLength( OpNode aLeft, OpNode const &aRight )
    {
        return ( aLeft.Get<sVectorValueComponent<T>>().mValue.size() == aRight.Get<sVectorValueComponent<T>>().mValue.size() );
    }

    OpNode BinaryOperation( Scope &aScope, eScalarType aType, OpNode const &aLeft, OpNode const &aRight )
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
                        if( lRightShape != lLeftShape ) throw std::runtime_error( "Can only add tensors of the same shape" );

                        auto &lBroadcastInfo = lNewEntity.Add<sBroadcastInfoComponent>();
                        lRightShape.Flatten( 0 );
                        lBroadcastInfo.mBroadcastHint = eBroadcastHint::LEFT;
                        lBroadcastInfo.mMaxBlockSize  = lRightShape.mMaxDimensions[0];
                        lBroadcastInfo.mBlockSizes    = VectorValue( aScope, lRightShape.GetDimension( 0 ) );

                        auto lBroadcastShape                  = lOperandData.mRightOperand.Get<sMultiTensorComponent>().Shape();
                        lBroadcastInfo.mBroadcastDimension    = VectorValue( aScope, lBroadcastShape.GetDimension( -1 ) );
                        lBroadcastInfo.mMaxBroadcastDimension = lBroadcastShape.mMaxDimensions[lBroadcastShape.mRank - 1];

                        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, sTensorShape( lBroadcastShape.mShape, SizeOf( aType ) ) );
                    }
                    else if( lRightShape.mRank == lLeftShape.mRank - 1 )
                    {
                        lLeftShape.Trim( -1 );
                        if( lRightShape != lLeftShape ) throw std::runtime_error( "Can only add tensors of the same shape" );

                        auto &lBroadcastInfo = lNewEntity.Add<sBroadcastInfoComponent>();
                        lRightShape.Flatten( 0 );
                        lBroadcastInfo.mBroadcastHint = eBroadcastHint::RIGHT;
                        lBroadcastInfo.mMaxBlockSize  = lRightShape.mMaxDimensions[0];
                        lBroadcastInfo.mBlockSizes    = VectorValue( aScope, lRightShape.GetDimension( 0 ) );

                        auto lBroadcastShape                  = lOperandData.mLeftOperand.Get<sMultiTensorComponent>().Shape();
                        lBroadcastInfo.mBroadcastDimension    = VectorValue( aScope, lBroadcastShape.GetDimension( -1 ) );
                        lBroadcastInfo.mMaxBroadcastDimension = lBroadcastShape.mMaxDimensions[lBroadcastShape.mRank - 1];

                        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, sTensorShape( lBroadcastShape.mShape, SizeOf( aType ) ) );
                    }
                    else
                    {
                        throw std::runtime_error( "Can only add tensors of the same shape" );
                    }
                }
                else
                {
                    auto lShape = lOperandData.mLeftOperand.Get<sMultiTensorComponent>().Shape().mShape;
                    lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, sTensorShape( lShape, SizeOf( aType ) ) );
                }
            }
            else
            {
                auto lShape = lOperandData.mLeftOperand.Get<sMultiTensorComponent>().Shape().mShape;
                lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, sTensorShape( lShape, SizeOf( aType ) ) );
            }
        }
        else
        {
            if( !( lOperandData.mRightOperand.Has<sMultiTensorComponent>() ) )
            {
                throw std::runtime_error( "RHS should have a tensor" );
            }

            auto lShape = lOperandData.mRightOperand.Get<sMultiTensorComponent>().Shape().mShape;
            lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, sTensorShape( lShape, SizeOf( aType ) ) );
        }

        if( lNewEntity.Has<sBroadcastInfoComponent>() )
            lNewEntity.Add<sOperandComponent>( std::vector<OpNode>{ aLeft, aRight,
                                                                    lNewEntity.Get<sBroadcastInfoComponent>().mBlockSizes,
                                                                    lNewEntity.Get<sBroadcastInfoComponent>().mBroadcastDimension } );
        else
            lNewEntity.Add<sOperandComponent>( std::vector<OpNode>{ aLeft, aRight } );

        lNewEntity.Add<sTypeComponent>( aType );

        return lNewEntity;
    }

    OpNode BinaryOperation( Scope &aScope, OpNode const &aLeft, OpNode const &aRight )
    {
        return BinaryOperation( aScope, aLeft.Get<sTypeComponent>().mValue, aLeft, aRight );
    }

    OpNode Add( Scope &aScope, OpNode const &aLeft, OpNode const &aRight )
    {
        auto lNewEntity = BinaryOperation( aScope, aLeft, aRight );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sAddOperationController>();

        return lNewEntity;
    }

    OpNode Subtract( Scope &aScope, OpNode const &aLeft, OpNode const &aRight )
    {
        auto lNewEntity = BinaryOperation( aScope, aLeft, aRight );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sSubtractOperationController>();

        return lNewEntity;
    }

    OpNode Divide( Scope &aScope, OpNode const &aLeft, OpNode const &aRight )
    {
        auto lNewEntity = BinaryOperation( aScope, aLeft, aRight );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sDivideOperationController>();

        return lNewEntity;
    }

    OpNode Multiply( Scope &aScope, OpNode const &aLeft, OpNode const &aRight )
    {
        auto lNewEntity = BinaryOperation( aScope, aLeft, aRight );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sMultiplyOperationController>();

        return lNewEntity;
    }

    OpNode And( Scope &aScope, OpNode const &aLeft, OpNode const &aRight )
    {
        assert( ( aLeft.Has<sTypeComponent>() ) && ( aRight.Has<sTypeComponent>() ) );
        assert( SameType( aLeft, aRight ) );
        assert( aLeft.Get<sTypeComponent>().mValue == eScalarType::UINT8 );

        auto lNewEntity = BinaryOperation( aScope, aLeft, aRight );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sAndOperationController>();

        return lNewEntity;
    }

    OpNode Or( Scope &aScope, OpNode const &aLeft, OpNode const &aRight )
    {
        assert( ( aLeft.Has<sTypeComponent>() ) && ( aRight.Has<sTypeComponent>() ) );
        assert( SameType( aLeft, aRight ) );
        assert( aLeft.Get<sTypeComponent>().mValue == eScalarType::UINT8 );

        auto lNewEntity = BinaryOperation( aScope, aLeft, aRight );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sOrOperationController>();

        return lNewEntity;
    }

    OpNode Not( Scope &aScope, OpNode const &aOperand )
    {
        assert( ( aOperand.Has<sTypeComponent>() ) );
        assert( aOperand.Get<sTypeComponent>().mValue == eScalarType::UINT8 );

        auto  lNewEntity   = aScope.CreateNode();
        auto &lOperandData = lNewEntity.Add<sNotOperationComponent>( sNotOperationComponent{ aOperand } );

        lNewEntity.Add<sOperandComponent>( std::vector{ aOperand } );
        lNewEntity.Add<sTypeComponent>( aOperand.Get<sTypeComponent>() );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, lOperandData.mOperand.Get<sMultiTensorComponent>().Shape() );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sNotOperationController>();

        return lNewEntity;
    }

    OpNode BitwiseAnd( Scope &aScope, OpNode const &aLeft, OpNode const &aRight )
    {
        assert( ( aLeft.Has<sTypeComponent>() ) && ( aRight.Has<sTypeComponent>() ) );
        assert( SameType( aLeft, aRight ) );
        assert( ( aLeft.Get<sTypeComponent>().mValue >= eScalarType::UINT8 ) &&
                ( aLeft.Get<sTypeComponent>().mValue <= eScalarType::INT64 ) );

        auto lNewEntity = BinaryOperation( aScope, aLeft, aRight );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sBitwiseAndOperationController>();

        return lNewEntity;
    }

    OpNode BitwiseOr( Scope &aScope, OpNode const &aLeft, OpNode const &aRight )
    {
        assert( ( aLeft.Has<sTypeComponent>() ) && ( aRight.Has<sTypeComponent>() ) );
        assert( SameType( aLeft, aRight ) );
        assert( ( aLeft.Get<sTypeComponent>().mValue >= eScalarType::UINT8 ) &&
                ( aLeft.Get<sTypeComponent>().mValue <= eScalarType::INT64 ) );

        auto lNewEntity = BinaryOperation( aScope, aLeft, aRight );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sBitwiseOrOperationController>();

        return lNewEntity;
    }

    OpNode BitwiseNot( Scope &aScope, OpNode const &aOperand )
    {
        assert( ( aOperand.Has<sTypeComponent>() ) );
        assert( ( aOperand.Get<sTypeComponent>().mValue >= eScalarType::UINT8 ) &&
                ( aOperand.Get<sTypeComponent>().mValue <= eScalarType::INT64 ) );

        auto  lNewEntity   = aScope.CreateNode();
        auto &lOperandData = lNewEntity.Add<sBitwiseNotOperationComponent>( sBitwiseNotOperationComponent{ aOperand } );

        lNewEntity.Add<sOperandComponent>( std::vector{ aOperand } );
        lNewEntity.Add<sTypeComponent>( aOperand.Get<sTypeComponent>() );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, lOperandData.mOperand.Get<sMultiTensorComponent>().Shape() );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sBitwiseNotOperationController>();

        return lNewEntity;
    }

    OpNode InInterval( Scope &aScope, OpNode const &aX, OpNode const &aLower, OpNode const &aUpper, bool aStrictLower,
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

        lNewEntity.Add<sOperandComponent>( std::vector<OpNode>{ aX, aLower, aUpper } );
        lNewEntity.Add<sTypeComponent>( eScalarType::UINT8 );

        std::vector<std::vector<uint32_t>> lOutputShape = aX.Get<sMultiTensorComponent>().Shape().mShape;

        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, sTensorShape( lOutputShape, SizeOf( eScalarType::UINT8 ) ) );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sInIntervalOperationController>();

        return lNewEntity;
    }

    OpNode Equal( Scope &aScope, OpNode const &aX, OpNode const &aY )
    {
        assert( ( aX.Has<sTypeComponent>() ) && ( aY.Has<sTypeComponent>() ) );
        assert( SameType( aX, aY ) );

        auto lNewEntity = BinaryOperation( aScope, eScalarType::UINT8, aX, aY );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sEqualOperationController>();

        return lNewEntity;
    }

    OpNode LessThan( Scope &aScope, OpNode const &aX, OpNode const &aY )
    {
        assert( ( aX.Has<sTypeComponent>() ) && ( aY.Has<sTypeComponent>() ) );
        assert( SameType( aX, aY ) );

        auto lNewEntity = BinaryOperation( aScope, eScalarType::UINT8, aX, aY );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sLessThanOperationController>();

        return lNewEntity;
    }

    OpNode LessThanOrEqual( Scope &aScope, OpNode const &aX, OpNode const &aY )
    {
        assert( ( aX.Has<sTypeComponent>() ) && ( aY.Has<sTypeComponent>() ) );
        assert( SameType( aX, aY ) );

        auto lNewEntity = BinaryOperation( aScope, eScalarType::UINT8, aX, aY );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sLessThanOrEqualOperationController>();

        return lNewEntity;
    }

    OpNode GreaterThan( Scope &aScope, OpNode const &aX, OpNode const &aY ) { return LessThan( aScope, aY, aX ); }

    OpNode GreaterThanOrEqual( Scope &aScope, OpNode const &aX, OpNode const &aY ) { return LessThanOrEqual( aScope, aY, aX ); }

    OpNode Where( Scope &aScope, OpNode const &aCondition, OpNode const &aValueIfTrue, OpNode const &aValueIfFalse )
    {
        assert( ( aCondition.Has<sTypeComponent>() ) && ( aValueIfTrue.Has<sTypeComponent>() ) &&
                ( aValueIfFalse.Has<sTypeComponent>() ) );
        assert( ( aValueIfTrue.HasAny<sMultiTensorComponent, sScalarValueVectorComponent, sScalarNodeComponent>() ) );
        assert( ( aValueIfFalse.HasAny<sMultiTensorComponent, sScalarValueVectorComponent, sScalarNodeComponent>() ) );
        assert( ( aCondition.Get<sTypeComponent>().mValue == eScalarType::UINT8 ) );
        assert( SameType( aValueIfTrue, aValueIfFalse ) );

        auto lNewEntity = aScope.CreateNode();
        lNewEntity.Add<sWhereOperationComponent>( sWhereOperationComponent{ aCondition, aValueIfTrue, aValueIfFalse } );
        lNewEntity.Add<sTypeComponent>( aValueIfTrue.Get<sTypeComponent>() );
        lNewEntity.Add<sOperandComponent>( std::vector<OpNode>{ aCondition, aValueIfTrue, aValueIfFalse } );

        auto lShape = aCondition.Get<sMultiTensorComponent>().Shape().mShape;
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool,
                                               sTensorShape( lShape, SizeOf( aValueIfTrue.Get<sTypeComponent>().mValue ) ) );

        lNewEntity.Add<sGraphOperationComponent>().Bind<sWhereOperationController>();

        return lNewEntity;
    }

    OpNode Mix( Scope &aScope, OpNode const &aA, OpNode const &aB, OpNode const &aT )
    {
        assert( ( aA.HasAll<sTypeComponent, sMultiTensorComponent>() ) );
        assert( ( aB.HasAll<sTypeComponent, sMultiTensorComponent>() ) );
        assert( ( aT.HasAll<sTypeComponent, sMultiTensorComponent>() ) );

        auto  lNewEntity   = aScope.CreateNode();
        auto &lOperandData = lNewEntity.Add<sMixNodeComponent>( sMixNodeComponent{ aA, aB, aT } );

        lNewEntity.Add<sOperandComponent>( std::vector{ aA, aB, aT } );
        lNewEntity.Add<sTypeComponent>( aA.Get<sTypeComponent>() );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, lOperandData.mA.Get<sMultiTensorComponent>().Shape() );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sMixOperationController>();

        return lNewEntity;
    }

    OpNode ToFixedPoint( Scope &aScope, eScalarType aOutputType, OpNode const &aArray, OpNode const &aScaling )
    {
        assert( ( aArray.HasAll<sTypeComponent, sMultiTensorComponent>() ) );
        assert( ( aScaling.HasAll<sTypeComponent, sScalarNodeComponent>() ) );

        auto lNewEntity = aScope.CreateNode();

        auto &lOperandData = lNewEntity.Add<sToFixedPointNodeComponent>( sToFixedPointNodeComponent{ aOutputType, aArray, aScaling } );

        lNewEntity.Add<sOperandComponent>( std::vector{ aArray, aScaling } );

        auto &lShape = lOperandData.mArray.Get<sMultiTensorComponent>().Shape().mShape;
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, sTensorShape( lShape, SizeOf( aOutputType ) ) );

        lNewEntity.Add<sGraphOperationComponent>().Bind<sToFixedPointOperationController>();

        return lNewEntity;
    }

    OpNode Repeat( Scope &aScope, OpNode const &aArray, OpNode const &aRepetitions )
    {
        assert( ( aArray.HasAll<sTypeComponent, sMultiTensorComponent>() ) );
        assert( aRepetitions.Has<sU32VectorComponent>() );

        auto  lNewEntity   = aScope.CreateNode();
        auto &lOperandData = lNewEntity.Add<sRepeatOperationComponent>( sRepeatOperationComponent{ aArray, aRepetitions } );

        lNewEntity.Add<sOperandComponent>( std::vector{ aArray, aRepetitions } );
        lNewEntity.Add<sTypeComponent>( aArray.Get<sTypeComponent>() );

        auto &lInputShape = lOperandData.mArray.Get<sMultiTensorComponent>().Shape();

        auto                               lRepetitions = aRepetitions.Get<sU32VectorComponent>().mValue;
        std::vector<std::vector<uint32_t>> lOutputShape( lInputShape.CountLayers() );
        for( uint32_t i = 0; i < lInputShape.CountLayers(); i++ )
        {
            lOutputShape[i] = std::vector<uint32_t>( lInputShape.mShape[i].size() + 1 );
            for( uint32_t j = 0; j < lInputShape.mShape[i].size(); j++ )
            {
                lOutputShape[i][j] = lInputShape.mShape[i][j];
            }
            lOutputShape[i][lOutputShape[i].size() - 1] = lRepetitions[i];
        }

        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool,
                                               sTensorShape( lOutputShape, SizeOf( aArray.Get<sTypeComponent>().mValue ) ) );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sArrayOperationController>();

        return lNewEntity;
    }

    OpNode Tile( Scope &aScope, OpNode const &aArray, OpNode const &aRepetitions )
    {
        assert( ( aArray.HasAll<sTypeComponent, sMultiTensorComponent>() ) );
        assert( aRepetitions.Has<sU32VectorComponent>() );

        auto  lNewEntity   = aScope.CreateNode();
        auto &lOperandData = lNewEntity.Add<sTileOperationComponent>( sTileOperationComponent{ aArray, aRepetitions } );

        lNewEntity.Add<sOperandComponent>( std::vector{ aArray, aRepetitions } );
        lNewEntity.Add<sTypeComponent>( aArray.Get<sTypeComponent>() );
        auto &lInputShape = lOperandData.mArray.Get<sMultiTensorComponent>().Shape();

        auto                               lRepetitions = aRepetitions.Get<sU32VectorComponent>().mValue;
        std::vector<std::vector<uint32_t>> lOutputShape( lInputShape.CountLayers() );
        for( uint32_t i = 0; i < lInputShape.CountLayers(); i++ )
        {
            lOutputShape[i]                             = std::vector<uint32_t>( lInputShape.mShape[i].size() + 1 );
            lOutputShape[i][lOutputShape[i].size() - 1] = lRepetitions[i];

            lOutputShape[i][0] = lRepetitions[i];

            for( uint32_t j = 0; j < lInputShape.mShape[i].size(); j++ )
            {
                lOutputShape[i][j + 1] = lInputShape.mShape[i][j];
            }
        }

        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool,
                                               sTensorShape( lOutputShape, SizeOf( aArray.Get<sTypeComponent>().mValue ) ) );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sArrayOperationController>();

        return lNewEntity;
    }

    OpNode ARange( Scope &aScope, OpNode const &aLeft, OpNode const &aRight, OpNode const &aDelta )
    {
        assert( aLeft.Has<sScalarValueVectorComponent>() && aRight.Has<sScalarValueVectorComponent>() &&
                aDelta.Has<sScalarValueVectorComponent>() );
        assert( aLeft.Has<sTypeComponent>() && aRight.Has<sTypeComponent>() && aDelta.Has<sTypeComponent>() );

        assert( SameType( aLeft, aRight ) );
        assert( SameType( aLeft, aDelta ) );
        assert( SameLength<ScalarValue>( aLeft, aRight ) );
        assert( SameLength<ScalarValue>( aLeft, aDelta ) );

        assert( ( aLeft.Get<sTypeComponent>().mValue == eScalarType::FLOAT32 ) ||
                ( aLeft.Get<sTypeComponent>().mValue == eScalarType::FLOAT64 ) );

        auto  lNewEntity   = aScope.CreateNode();
        auto &lOperandData = lNewEntity.Add<sARangeComponent>( sARangeComponent{ aLeft, aRight, aDelta } );

        lNewEntity.Add<sTypeComponent>( aLeft.Get<sTypeComponent>() );
        lNewEntity.Add<sOperandComponent>( std::vector{ aLeft, aRight, aDelta } );

        std::vector<std::vector<uint32_t>> lOutputShape( aLeft.Get<sScalarValueVectorComponent>().mValue.size() );
        auto                               lLeftValues  = aLeft.Get<sScalarValueVectorComponent>().mValue;
        auto                               lRightValues = aRight.Get<sScalarValueVectorComponent>().mValue;
        auto                               lDeltaValues = aDelta.Get<sScalarValueVectorComponent>().mValue;

        for( uint32_t i = 0; i < aLeft.Get<sScalarValueVectorComponent>().mValue.size(); i++ )
        {
            lOutputShape[i] = { static_cast<uint32_t>( std::ceil(
                ( std::get<float>( lRightValues[i] ) - std::get<float>( lLeftValues[i] ) ) / std::get<float>( lDeltaValues[i] ) ) ) };
        }

        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool,
                                               sTensorShape( lOutputShape, SizeOf( aLeft.Get<sTypeComponent>().mValue ) ) );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sARangeOperationController>();

        return lNewEntity;
    }

    OpNode LinearSpace( Scope &aScope, OpNode const &aLeft, OpNode const &aRight, OpNode const &aSubdivisions )
    {
        assert( aLeft.Has<sMultiTensorComponent>() && aRight.Has<sMultiTensorComponent>() &&
                aSubdivisions.Has<sU32VectorComponent>() );

        assert( SameShape( aLeft, aRight ) );

        assert( aLeft.Get<sMultiTensorComponent>().Shape().CountLayers() == aSubdivisions.Get<sU32VectorComponent>().mValue.size() );
        assert( SameType( aLeft, aRight ) );
        assert( ( aLeft.Get<sTypeComponent>().mValue == eScalarType::FLOAT32 ) ||
                ( aLeft.Get<sTypeComponent>().mValue == eScalarType::FLOAT64 ) );

        auto  lNewEntity   = aScope.CreateNode();
        auto &lOperandData = lNewEntity.Add<sLinearSpaceComponent>( sLinearSpaceComponent{ aLeft, aRight, aSubdivisions } );

        lNewEntity.Add<sTypeComponent>( aLeft.Get<sTypeComponent>() );
        lNewEntity.Add<sOperandComponent>( std::vector{ aLeft, aRight, aSubdivisions } );

        auto                               lSubdivisions = aSubdivisions.Get<sU32VectorComponent>().mValue;
        std::vector<std::vector<uint32_t>> lOutputShape( aLeft.Get<sMultiTensorComponent>().Shape().mShape.size() );

        for( uint32_t i = 0; i < aLeft.Get<sMultiTensorComponent>().Shape().mShape.size(); i++ )
        {
            lOutputShape[i] = aLeft.Get<sMultiTensorComponent>().Shape().mShape[i];
            lOutputShape[i].push_back( lSubdivisions[i] );
        }

        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool,
                                               sTensorShape( lOutputShape, SizeOf( aLeft.Get<sTypeComponent>().mValue ) ) );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sLinearSpaceOperationController>();

        return lNewEntity;
    }

    OpNode Sample2D( Scope &aScope, OpNode const &aX, OpNode const &aY, OpNode const &aTextures )
    {
        assert( ( aX.HasAny<sMultiTensorComponent, sScalarNodeComponent, sScalarValueVectorComponent>() ) );
        assert( ( aY.HasAny<sMultiTensorComponent, sScalarNodeComponent, sScalarValueVectorComponent>() ) );

        assert( aX.Has<sMultiTensorComponent>() || aY.Has<sMultiTensorComponent>() );
        assert( aTextures.Has<sVectorValueComponent<Cuda::TextureSampler2D::DeviceData>>() );

        if( aX.Has<sMultiTensorComponent>() && aY.Has<sMultiTensorComponent>() ) assert( SameShape( aX, aY ) );

        if( aX.Has<sMultiTensorComponent>() )
            assert( aX.Get<sMultiTensorComponent>().Shape().CountLayers() ==
                    aTextures.Get<sVectorValueComponent<Cuda::TextureSampler2D::DeviceData>>().mValue.size() );

        if( aY.Has<sMultiTensorComponent>() )
            assert( aY.Get<sMultiTensorComponent>().Shape().CountLayers() ==
                    aTextures.Get<sVectorValueComponent<Cuda::TextureSampler2D::DeviceData>>().mValue.size() );

        assert( SameType( aX, aY ) );

        assert( aX.Get<sTypeComponent>().mValue == eScalarType::FLOAT32 );

        auto  lNewEntity   = aScope.CreateNode();
        auto &lOperandData = lNewEntity.Add<sSample2DComponent>( sSample2DComponent{ aX, aY, aTextures } );

        lNewEntity.Add<sTypeComponent>( aX.Get<sTypeComponent>() );
        lNewEntity.Add<sOperandComponent>( std::vector{ aX, aY, aTextures } );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, lOperandData.mX.Get<sMultiTensorComponent>().Shape() );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sSample2DOperationController>();

        return lNewEntity;
    }

    OpNode AffineTransform( Scope &aScope, OpNode const &aA, OpNode const &aX, OpNode const &aB )
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

        lNewEntity.Add<sOperandComponent>( std::vector{ aA, aX, aB } );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, lOperandData.mX.Get<sMultiTensorComponent>().Shape() );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sAffineNodeController>();

        return lNewEntity;
    }

    OpNode Collapse( Scope &aScope, OpNode const &aArray )
    {
        assert( ( aArray.Has<sMultiTensorComponent>() ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<sOperandComponent>( std::vector{ aArray } );

        auto &lInputShape = aArray.Get<sMultiTensorComponent>().Shape();
        for( uint32_t i = 0; i < lInputShape.mShape.size(); i++ )
        {
            if( lInputShape.mShape[i] != lInputShape.mShape[0] ) throw std::runtime_error( "All dimensions should be equal" );
        }

        std::vector<uint32_t> lOutputDimension( lInputShape.mRank + 1 );
        lOutputDimension[0] = lInputShape.CountLayers();
        std::copy( lInputShape.mShape[0].begin(), lInputShape.mShape[0].end(), lOutputDimension.begin() + 1 );

        lNewEntity.Add<sTypeComponent>( aArray.Get<sTypeComponent>() );
        lNewEntity.Add<sMultiTensorComponent>(
            aScope.mPool, aArray.Get<sMultiTensorComponent>().mValue.GetMemoryBuffer(),
            sTensorShape( std::vector<std::vector<uint32_t>>{ lOutputDimension }, static_cast<size_t>( lInputShape.mElementSize ) ) );

        return lNewEntity;
    }

    OpNode Expand( Scope &aScope, OpNode const &aArray )
    {
        auto lNewEntity = aScope.CreateNode();
        assert( ( aArray.Has<sMultiTensorComponent>() ) );

        lNewEntity.Add<sOperandComponent>( std::vector{ aArray } );

        auto &lInputShape = aArray.Get<sMultiTensorComponent>().Shape();

        assert( lInputShape.CountLayers() == 1 );

        std::vector<std::vector<uint32_t>> lOutputShape( lInputShape.mShape[0][0] );

        for( uint32_t i = 0; i < lOutputShape.size(); i++ )
        {
            lOutputShape[i] = std::vector<uint32_t>( lInputShape.mRank - 1 );
            std::copy( lInputShape.mShape[0].begin() + 1, lInputShape.mShape[0].end(), lOutputShape[i].begin() );
        }

        lNewEntity.Add<sTypeComponent>( aArray.Get<sTypeComponent>() );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, aArray.Get<sMultiTensorComponent>().mValue.GetMemoryBuffer(),
                                               sTensorShape( lOutputShape, static_cast<size_t>( lInputShape.mElementSize ) ) );

        return lNewEntity;
    }

    OpNode Reshape( Scope &aScope, OpNode const &aArray, sTensorShape &aNewShape )
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

            if( lSize0 != lSize1 ) throw std::runtime_error( "Incompatible dimensions" );
        }

        lNewEntity.Add<sTypeComponent>( aArray.Get<sTypeComponent>() );
        lNewEntity.Add<sOperandComponent>( std::vector{ aArray } );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, aArray.Get<sMultiTensorComponent>().mValue.GetMemoryBuffer(), aNewShape );

        return lNewEntity;
    }

    OpNode Relayout( Scope &aScope, OpNode const &aArray, sTensorShape &aNewLayout )
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

        if( lInputSize != lOutputSize ) throw std::runtime_error( "Incompatible dimensions" );

        if( aArray.Has<sTypeComponent>() ) lNewEntity.Add<sTypeComponent>( aArray.Get<sTypeComponent>() );

        lNewEntity.Add<sOperandComponent>( std::vector{ aArray } );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, aArray.Get<sMultiTensorComponent>().mValue.GetMemoryBuffer(),
                                               aNewLayout );

        return lNewEntity;
    }

    OpNode Flatten( Scope &aScope, OpNode const &aArray )
    {
        auto lNewEntity = aScope.CreateNode();
        assert( ( aArray.Has<sMultiTensorComponent>() ) );

        auto lInputShape = aArray.Get<sMultiTensorComponent>().Shape();

        lInputShape.Flatten( 0 );
        lNewEntity.Add<sTypeComponent>( aArray.Get<sTypeComponent>() );
        lNewEntity.Add<sOperandComponent>( std::vector{ aArray } );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, aArray.Get<sMultiTensorComponent>().mValue.GetMemoryBuffer(),
                                               lInputShape );

        return lNewEntity;
    }

    OpNode Slice( Scope &aScope, OpNode const &aArray, OpNode const &aBegin, OpNode const &aEnd )
    {
        assert( ( aArray.HasAll<sTypeComponent, sMultiTensorComponent>() ) );
        assert( ( ( aBegin.HasAny<sVectorValueComponent<uint32_t>, sScalarNodeComponent>() ) ) &&
                ( aEnd.HasAny<sVectorValueComponent<uint32_t>, sScalarNodeComponent>() ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<sTypeComponent>( aArray.Get<sTypeComponent>() );
        auto &lOperandData  = lNewEntity.Add<sArraySliceNodeComponent>();
        lOperandData.mArray = aArray;

        auto lInputShape = aArray.Get<sMultiTensorComponent>().Shape();

        uint32_t              lMaxBlockSize = 0;
        std::vector<uint32_t> lBlockSizes( lInputShape.CountLayers() );

        std::vector<uint32_t> lBegin;
        if( aBegin.Has<sVectorValueComponent<uint32_t>>() )
        {
            lBegin              = aBegin.Get<sVectorValueComponent<uint32_t>>().mValue;
            lOperandData.mBegin = aBegin;
        }
        else
        {
            lBegin =
                std::vector<uint32_t>( lInputShape.CountLayers(), std::get<uint32_t>( aBegin.Get<sScalarNodeComponent>().mValue ) );
            lOperandData.mBegin = VectorValue( aScope, lBegin );
        }

        std::vector<uint32_t> lEnd;
        if( aEnd.Has<sVectorValueComponent<uint32_t>>() )
        {
            lEnd              = aEnd.Get<sVectorValueComponent<uint32_t>>().mValue;
            lOperandData.mEnd = aEnd;
        }
        else
        {
            lEnd = std::vector<uint32_t>( lInputShape.CountLayers(), std::get<uint32_t>( aEnd.Get<sScalarNodeComponent>().mValue ) );
            lOperandData.mEnd = VectorValue( aScope, lEnd );
        }

        std::vector<std::vector<uint32_t>> lOutputShape( lInputShape.CountLayers() );

        for( uint32_t i = 0; i < lInputShape.CountLayers(); i++ )
        {
            lOutputShape[i] = std::vector<uint32_t>( lInputShape.mShape[i].size() );
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
            std::vector{ aArray, lOperandData.mBegin, lOperandData.mEnd, lOperandData.mBlockSizes, lOperandData.mElementCount } );

        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool,
                                               sTensorShape( lOutputShape, SizeOf( aArray.Get<sTypeComponent>().mValue ) ) );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sArraySliceOperationController>();

        return lNewEntity;
    }

    OpNode Summation( Scope &aScope, OpNode const &aArray )
    {
        auto                  lInputShape = aArray.Get<sMultiTensorComponent>().Shape();
        std::vector<uint32_t> lLastDimensions( lInputShape.CountLayers() );
        for( uint32_t i = 0; i < lInputShape.CountLayers(); i++ )
            lLastDimensions[i] = lInputShape.mShape[i][lInputShape.mShape[i].size() - 1] - 1;

        auto lZero = ConstantScalarValue( aScope, static_cast<uint32_t>( 0 ) );
        auto lEnd  = VectorValue( aScope, lLastDimensions );

        return Summation( aScope, aArray, lZero, lEnd );
    }

    OpNode Summation( Scope &aScope, OpNode const &aArray, OpNode const &aBegin, OpNode const &aEnd )
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
            std::vector<uint32_t> lBegin( lInputShape.CountLayers(), std::get<uint32_t>( aBegin.Get<sScalarNodeComponent>().mValue ) );
            lOperandData.mBegin = VectorValue( aScope, lBegin );
        }

        if( aEnd.Has<sVectorValueComponent<uint32_t>>() )
        {
            lOperandData.mEnd = aEnd;
        }
        else
        {
            std::vector<uint32_t> lEnd( lInputShape.CountLayers(), std::get<uint32_t>( aEnd.Get<sScalarNodeComponent>().mValue ) );
            lOperandData.mEnd = VectorValue( aScope, lEnd );
        }

        auto lOutputShape = lInputShape;
        lOutputShape.Trim( -1 );

        lInputShape.Flatten( -1 );
        lOperandData.mMaxBlockSize = lInputShape.mMaxDimensions[0];
        lOperandData.mBlockSizes   = VectorValue( aScope, lInputShape.GetDimension( 0 ) );
        lOperandData.mElementCount = VectorValue( aScope, lInputShape.GetDimension( -1 ) );

        lNewEntity.Add<sOperandComponent>(
            std::vector{ aArray, lOperandData.mBegin, lOperandData.mEnd, lOperandData.mBlockSizes, lOperandData.mElementCount } );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool,
                                               sTensorShape( lOutputShape.mShape, SizeOf( aArray.Get<sTypeComponent>().mValue ) ) );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sArraySummationOperationController>();

        return lNewEntity;
    }

    OpNode CountTrue( Scope &aScope, OpNode const &aArray )
    {
        assert( ( aArray.HasAll<sTypeComponent, sMultiTensorComponent>() ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<sTypeComponent>( sTypeComponent{ eScalarType::UINT32 } );
        auto &lOperandData  = lNewEntity.Add<sCountTrueNodeComponent>();
        lOperandData.mArray = aArray;

        auto lInputShape  = aArray.Get<sMultiTensorComponent>().Shape();
        auto lOutputShape = lInputShape;
        lOutputShape.Trim( -1 );

        lInputShape.Flatten( -1 );
        lOperandData.mMaxBlockSize = lInputShape.mMaxDimensions[0];
        lOperandData.mBlockSizes   = VectorValue( aScope, lInputShape.GetDimension( 0 ) );
        lOperandData.mElementCount = VectorValue( aScope, lInputShape.GetDimension( -1 ) );

        lNewEntity.Add<sOperandComponent>( std::vector{ aArray, lOperandData.mBlockSizes, lOperandData.mElementCount } );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, sTensorShape( lOutputShape.mShape, SizeOf( eScalarType::UINT32 ) ) );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sCountTrueOperationController>();

        return lNewEntity;
    }

    OpNode CountZero( Scope &aScope, OpNode const &aArray )
    {
        assert( ( aArray.HasAll<sTypeComponent, sMultiTensorComponent>() ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<sTypeComponent>( sTypeComponent{ eScalarType::UINT32 } );
        auto &lOperandData  = lNewEntity.Add<sCountZeroNodeComponent>();
        lOperandData.mArray = aArray;

        auto lInputShape = aArray.Get<sMultiTensorComponent>().Shape();

        auto lOutputShape = lInputShape;
        lOutputShape.Trim( -1 );

        lInputShape.Flatten( -1 );
        lOperandData.mMaxBlockSize = lInputShape.mMaxDimensions[0];
        lOperandData.mBlockSizes   = VectorValue( aScope, lInputShape.GetDimension( 0 ) );
        lOperandData.mElementCount = VectorValue( aScope, lInputShape.GetDimension( -1 ) );

        lNewEntity.Add<sOperandComponent>( std::vector{ aArray, lOperandData.mBlockSizes, lOperandData.mElementCount } );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, sTensorShape( lOutputShape.mShape, SizeOf( eScalarType::UINT32 ) ) );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sCountZeroOperationController>();

        return lNewEntity;
    }

    OpNode CountNonZero( Scope &aScope, OpNode const &aArray )
    {
        assert( ( aArray.HasAll<sTypeComponent, sMultiTensorComponent>() ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<sTypeComponent>( sTypeComponent{ eScalarType::UINT32 } );
        auto &lOperandData  = lNewEntity.Add<sCountNonZeroNodeComponent>();
        lOperandData.mArray = aArray;

        auto lInputShape = aArray.Get<sMultiTensorComponent>().Shape();

        auto lOutputShape = lInputShape;
        lOutputShape.Trim( -1 );

        lInputShape.Flatten( -1 );
        lOperandData.mMaxBlockSize = lInputShape.mMaxDimensions[0];
        lOperandData.mBlockSizes   = VectorValue( aScope, lInputShape.GetDimension( 0 ) );
        lOperandData.mElementCount = VectorValue( aScope, lInputShape.GetDimension( -1 ) );

        lNewEntity.Add<sOperandComponent>( std::vector{ aArray, lOperandData.mBlockSizes, lOperandData.mElementCount } );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, sTensorShape( lOutputShape.mShape, SizeOf( eScalarType::UINT32 ) ) );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sCountNonZeroOperationController>();

        return lNewEntity;
    }

    OpNode Diff( Scope &aScope, OpNode const &aArray, uint32_t aCount )
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

        lNewEntity.Add<sOperandComponent>( std::vector{ aArray, lOperandData.mBlockSizes, lOperandData.mElementCount } );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, lOutputShape );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sDiffOperationController>();

        return lNewEntity;
    }

    OpNode Shift( Scope &aScope, OpNode const &aArray, int32_t aCount, OpNode const &aFillValue )
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
            std::vector{ aArray, lOperandData.mFillValue, lOperandData.mBlockSizes, lOperandData.mElementCount } );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, lOutputShape );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sShiftOperationController>();

        return lNewEntity;
    }

    OpNode Floor( Scope &aScope, OpNode const &aArray )
    {
        assert( ( aArray.HasAll<sTypeComponent, sMultiTensorComponent>() ) );
        assert( ( aArray.Get<sTypeComponent>().mValue == eScalarType::FLOAT32 ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<sOperandComponent>( std::vector{ aArray } );
        lNewEntity.Add<sTypeComponent>( aArray.Get<sTypeComponent>() );
        lNewEntity.Add<sFloorNodeComponent>( sFloorNodeComponent{ aArray } );

        auto lShape = aArray.Get<sMultiTensorComponent>().Shape().mShape;
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, sTensorShape( lShape, SizeOf( aArray.Get<sTypeComponent>().mValue ) ) );

        lNewEntity.Add<sGraphOperationComponent>().Bind<sFloorOperationController>();

        return lNewEntity;
    }

    OpNode Ceil( Scope &aScope, OpNode const &aArray )
    {
        assert( ( aArray.HasAll<sTypeComponent, sMultiTensorComponent>() ) );
        assert( ( aArray.Get<sTypeComponent>().mValue == eScalarType::FLOAT32 ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<sOperandComponent>( std::vector{ aArray } );
        lNewEntity.Add<sTypeComponent>( aArray.Get<sTypeComponent>() );
        lNewEntity.Add<sCeilNodeComponent>( sCeilNodeComponent{ aArray } );

        auto lShape = aArray.Get<sMultiTensorComponent>().Shape().mShape;
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, sTensorShape( lShape, SizeOf( aArray.Get<sTypeComponent>().mValue ) ) );

        lNewEntity.Add<sGraphOperationComponent>().Bind<sCeilOperationController>();

        return lNewEntity;
    }

    OpNode Abs( Scope &aScope, OpNode const &aArray )
    {
        assert( ( aArray.HasAll<sTypeComponent, sMultiTensorComponent>() ) );
        assert( ( aArray.Get<sTypeComponent>().mValue == eScalarType::FLOAT32 ) ||
                ( ( aArray.Get<sTypeComponent>().mValue >= eScalarType::INT8 ) &&
                  ( aArray.Get<sTypeComponent>().mValue <= eScalarType::INT64 ) ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<sOperandComponent>( std::vector{ aArray } );
        lNewEntity.Add<sTypeComponent>( aArray.Get<sTypeComponent>() );
        lNewEntity.Add<sAbsNodeComponent>( sAbsNodeComponent{ aArray } );

        auto lShape = aArray.Get<sMultiTensorComponent>().Shape().mShape;
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, sTensorShape( lShape, SizeOf( aArray.Get<sTypeComponent>().mValue ) ) );

        lNewEntity.Add<sGraphOperationComponent>().Bind<sAbsOperationController>();

        return lNewEntity;
    }

    OpNode Sqrt( Scope &aScope, OpNode const &aArray )
    {
        assert( ( aArray.HasAll<sTypeComponent, sMultiTensorComponent>() ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<sOperandComponent>( std::vector{ aArray } );
        lNewEntity.Add<sTypeComponent>( aArray.Get<sTypeComponent>() );
        lNewEntity.Add<sSqrtNodeComponent>( sSqrtNodeComponent{ aArray } );

        auto lShape = aArray.Get<sMultiTensorComponent>().Shape().mShape;
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, sTensorShape( lShape, SizeOf( aArray.Get<sTypeComponent>().mValue ) ) );

        lNewEntity.Add<sGraphOperationComponent>().Bind<sSqrtOperationController>();

        return lNewEntity;
    }

    OpNode Round( Scope &aScope, OpNode const &aArray )
    {
        assert( ( aArray.HasAll<sTypeComponent, sMultiTensorComponent>() ) );

        auto lNewEntity = aScope.CreateNode();

        lNewEntity.Add<sOperandComponent>( std::vector{ aArray } );
        lNewEntity.Add<sTypeComponent>( aArray.Get<sTypeComponent>() );
        lNewEntity.Add<sRoundNodeComponent>( sRoundNodeComponent{ aArray } );

        auto lShape = aArray.Get<sMultiTensorComponent>().Shape().mShape;
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, sTensorShape( lShape, SizeOf( aArray.Get<sTypeComponent>().mValue ) ) );

        lNewEntity.Add<sGraphOperationComponent>().Bind<sRoundOperationController>();

        return lNewEntity;
    }

    OpNode Conv1D( Scope &aScope, OpNode const &aArray0, OpNode const &aArray1 )
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

        lNewEntity.Add<sOperandComponent>( std::vector{ lOperandData.mArray0, lOperandData.mBlockSizes0, lOperandData.mElementCount0,
                                                        lOperandData.mArray1, lOperandData.mBlockSizes1,
                                                        lOperandData.mElementCount1 } );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, lOutputShape );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sConv1DOperationController>();

        return lNewEntity;
    }

    OpNode HCat( Scope &aScope, OpNode const &aArray0, OpNode const &aArray1 )
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

        auto                  lLastDim0 = lInputShape0.GetDimension( -1 );
        auto                  lLastDim1 = lInputShape1.GetDimension( -1 );
        std::vector<uint32_t> lConcatenated{};
        for( uint32_t i = 0; i < lLastDim0.size(); i++ ) lConcatenated.push_back( lLastDim0[i] + lLastDim1[i] );

        auto lOutputShape = aArray0.Get<sMultiTensorComponent>().Shape();
        lOutputShape.Trim( -1 );
        lOutputShape.InsertDimension( -1, lConcatenated );

        auto lBlockShape = aArray0.Get<sMultiTensorComponent>().Shape();
        lBlockShape.Flatten( -1 );
        lOperandData.mMaxBlockSize  = lBlockShape.mMaxDimensions[0];
        lOperandData.mBlockSizes    = VectorValue( aScope, lBlockShape.GetDimension( 0 ) );
        lOperandData.mElementCount0 = VectorValue( aScope, lInputShape0.GetDimension( -1 ) );
        lOperandData.mElementCount1 = VectorValue( aScope, lInputShape1.GetDimension( -1 ) );

        lNewEntity.Add<sOperandComponent>( std::vector{ lOperandData.mArray0, lOperandData.mArray1, lOperandData.mBlockSizes,
                                                        lOperandData.mElementCount0, lOperandData.mElementCount1 } );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, lOutputShape );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sHCatOperationController>();

        return lNewEntity;
    }

} // namespace SE::TensorOps
