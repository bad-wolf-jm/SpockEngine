/// @file   NodeControllers.cpp
///
/// @brief  Controller implementation
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#include <algorithm>
#include <cassert>
#include <cmath>

#include "Implementation/KernelLaunchers.h"
#include "NodeComponents.h"
#include "NodeControllers.h"

#include "Cuda/Texture2D.h"

namespace LTSE::TensorOps
{

    using namespace LTSE::Core;

    void sARangeOperationController::Run()
    {
        auto &lValue       = Get<sMultiTensorComponent>().mValue;
        auto &lOperandData = Get<sARangeComponent>();

        auto &lLeft  = lOperandData.mLeft.Get<sVectorComponent<ScalarValue>>().mData;
        auto &lRight = lOperandData.mRight.Get<sVectorComponent<ScalarValue>>().mData;
        auto &lDelta = lOperandData.mDelta.Get<sVectorComponent<ScalarValue>>().mData;

        auto lElementType = Get<sTypeComponent>().mValue;

        uint32_t lMaxSubdivisions = 0;
        for( const auto &lSub : lValue.Shape().mShape )
            lMaxSubdivisions = std::max( lMaxSubdivisions, lSub[0] );

        ARangeOp( lElementType, lValue, lLeft, lRight, lDelta, lMaxSubdivisions );
    }

    void sArrayOperationController::Run()
    {
        auto &lValue = Get<sMultiTensorComponent>().mValue;

        auto lElementType = Get<sTypeComponent>().mValue;

        if( Has<sRepeatOperationComponent>() )
        {
            auto &lOperandData       = Get<sRepeatOperationComponent>();
            auto &lArray             = lOperandData.mArray.Get<sMultiTensorComponent>().mValue;
            auto &lRepetitions       = lOperandData.mRepetitions.Get<sVectorComponent<uint32_t>>();
            uint32_t lMaxRepetitions = 0;
            for( const auto &lSub : lRepetitions.mValue )
                lMaxRepetitions = std::max( lMaxRepetitions, lSub );
            RepeatOp( lElementType, lValue, lArray, lRepetitions.mData, lMaxRepetitions );
            return;
        }

        if( Has<sTileOperationComponent>() )
        {
            auto &lOperandData       = Get<sTileOperationComponent>();
            auto &lArray             = lOperandData.mArray.Get<sMultiTensorComponent>().mValue;
            auto &lRepetitions       = lOperandData.mRepetitions.Get<sVectorComponent<uint32_t>>();
            uint32_t lMaxRepetitions = 0;
            for( const auto &lSub : lRepetitions.mValue )
                lMaxRepetitions = std::max( lMaxRepetitions, lSub );
            TileOp( lElementType, lValue, lArray, lRepetitions.mData, lMaxRepetitions );
            return;
        }
    }

    void sBinaryOperationController::Run()
    {
        auto &lValue       = Get<sMultiTensorComponent>().mValue;
        auto &lOperandData = Get<sBinaryOperationComponent>();
        auto lElementType  = Get<sTypeComponent>().mValue;

        if( lOperandData.mLeftOperand.Has<sMultiTensorComponent>() && lOperandData.mRightOperand.Has<sMultiTensorComponent>() )
        {
            auto &lLeftOperandData    = lOperandData.mLeftOperand.Get<sMultiTensorComponent>();
            auto &lRightOpOperandData = lOperandData.mRightOperand.Get<sMultiTensorComponent>();

            if( Has<sBroadcastInfoComponent>() )
                Op( lElementType, lValue, lLeftOperandData.mValue, lRightOpOperandData.mValue, Get<sBroadcastInfoComponent>() );
            else
                Op( lElementType, lValue, lLeftOperandData.mValue, lRightOpOperandData.mValue );
        }
        else if( lOperandData.mLeftOperand.Has<sMultiTensorComponent>() && lOperandData.mRightOperand.Has<sVectorComponent<ScalarValue>>() )
        {
            auto &lLeftOperandData    = lOperandData.mLeftOperand.Get<sMultiTensorComponent>();
            auto &lRightOpOperandData = lOperandData.mRightOperand.Get<sVectorComponent<ScalarValue>>();

            Op( lElementType, lValue, lLeftOperandData.mValue, lRightOpOperandData.mData );
        }
        else if( lOperandData.mLeftOperand.Has<sVectorComponent<ScalarValue>>() && lOperandData.mRightOperand.Has<sMultiTensorComponent>() )
        {
            auto &lLeftOperandData    = lOperandData.mLeftOperand.Get<sVectorComponent<ScalarValue>>();
            auto &lRightOpOperandData = lOperandData.mRightOperand.Get<sMultiTensorComponent>();

            Op( lElementType, lValue, lLeftOperandData.mData, lRightOpOperandData.mValue );
        }
        else if( lOperandData.mLeftOperand.Has<sMultiTensorComponent>() && lOperandData.mRightOperand.Has<sScalarNodeComponent>() )
        {
            auto &lLeftOperandData    = lOperandData.mLeftOperand.Get<sMultiTensorComponent>();
            auto &lRightOpOperandData = lOperandData.mRightOperand.Get<sScalarNodeComponent>();

            Op( lElementType, lValue, lLeftOperandData.mValue, lRightOpOperandData.mValue );
        }
        else if( lOperandData.mLeftOperand.Has<sScalarNodeComponent>() && lOperandData.mRightOperand.Has<sMultiTensorComponent>() )
        {
            auto &lLeftOperandData    = lOperandData.mLeftOperand.Get<sScalarNodeComponent>();
            auto &lRightOpOperandData = lOperandData.mRightOperand.Get<sMultiTensorComponent>();

            Op( lElementType, lValue, lLeftOperandData.mValue, lRightOpOperandData.mValue );
        }
        else
        {
            throw std::runtime_error( "something's wrong" );
        }
    }

    void sBinaryBooleanOperationController::Run()
    {

        auto &lValue       = Get<sMultiTensorComponent>().mValue;
        auto &lOperandData = Get<sBinaryOperationComponent>();

        if( lOperandData.mLeftOperand.Has<sMultiTensorComponent>() && lOperandData.mRightOperand.Has<sMultiTensorComponent>() )
        {
            auto &lLeftOperandData    = lOperandData.mLeftOperand.Get<sMultiTensorComponent>();
            auto &lRightOpOperandData = lOperandData.mRightOperand.Get<sMultiTensorComponent>();
            auto lElementType         = lOperandData.mLeftOperand.Get<sTypeComponent>().mValue;

            if( Has<sBroadcastInfoComponent>() )
                Op( lElementType, lValue, lLeftOperandData.mValue, lRightOpOperandData.mValue, Get<sBroadcastInfoComponent>() );
            else
                Op( lElementType, lValue, lLeftOperandData.mValue, lRightOpOperandData.mValue );
        }
        else if( lOperandData.mLeftOperand.Has<sMultiTensorComponent>() && lOperandData.mRightOperand.Has<sVectorComponent<ScalarValue>>() )
        {
            auto &lLeftOperandData    = lOperandData.mLeftOperand.Get<sMultiTensorComponent>();
            auto &lRightOpOperandData = lOperandData.mRightOperand.Get<sVectorComponent<ScalarValue>>();
            auto lElementType         = lOperandData.mLeftOperand.Get<sTypeComponent>().mValue;

            Op( lElementType, lValue, lLeftOperandData.mValue, lRightOpOperandData.mData );
        }
        else if( lOperandData.mLeftOperand.Has<sVectorComponent<ScalarValue>>() && lOperandData.mRightOperand.Has<sMultiTensorComponent>() )
        {
            auto &lLeftOperandData    = lOperandData.mLeftOperand.Get<sVectorComponent<ScalarValue>>();
            auto &lRightOpOperandData = lOperandData.mRightOperand.Get<sMultiTensorComponent>();
            auto lElementType         = lOperandData.mRightOperand.Get<sTypeComponent>().mValue;

            Op( lElementType, lValue, lLeftOperandData.mData, lRightOpOperandData.mValue );
        }
        else if( lOperandData.mLeftOperand.Has<sMultiTensorComponent>() && lOperandData.mRightOperand.Has<sScalarNodeComponent>() )
        {
            auto &lLeftOperandData    = lOperandData.mLeftOperand.Get<sMultiTensorComponent>();
            auto &lRightOpOperandData = lOperandData.mRightOperand.Get<sScalarNodeComponent>();
            auto lElementType         = lOperandData.mLeftOperand.Get<sTypeComponent>().mValue;

            Op( lElementType, lValue, lLeftOperandData.mValue, lRightOpOperandData.mValue );
        }
        else if( lOperandData.mLeftOperand.Has<sScalarNodeComponent>() && lOperandData.mRightOperand.Has<sMultiTensorComponent>() )
        {
            auto &lLeftOperandData    = lOperandData.mLeftOperand.Get<sScalarNodeComponent>();
            auto &lRightOpOperandData = lOperandData.mRightOperand.Get<sMultiTensorComponent>();
            auto lElementType         = lOperandData.mRightOperand.Get<sTypeComponent>().mValue;

            Op( lElementType, lValue, lLeftOperandData.mValue, lRightOpOperandData.mValue );
        }
        else
        {
            throw std::runtime_error( "something's wrong" );
        }
    }

    void sAddOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight )
    {
        AddOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sAddOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, sBroadcastInfoComponent &aBroadcast )
    {
        AddOp( aTensorElementType, aOut, aLeft, aRight, aBroadcast.mBroadcastHint, aBroadcast.mBlockSizes.Get<sVectorComponent<uint32_t>>().mData, aBroadcast.mMaxBlockSize,
               aBroadcast.mBroadcastDimension.Get<sVectorComponent<uint32_t>>().mData, aBroadcast.mMaxBroadcastDimension );
    }

    void sAddOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aIn, ScalarValue &aConstant )
    {
        AddOp( aTensorElementType, aOut, aIn, aConstant );
    }

    void sAddOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aConstant, MultiTensor &aIn )
    {
        AddOp( aTensorElementType, aOut, aIn, aConstant );
    }

    void sAddOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight )
    {
        AddOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sAddOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight )
    {
        AddOp( aTensorElementType, aOut, aRight, aLeft );
    }

    void sMultiplyOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight )
    {
        MultiplyOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sMultiplyOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, sBroadcastInfoComponent &aBroadcast )
    {
        MultiplyOp( aTensorElementType, aOut, aLeft, aRight, aBroadcast.mBroadcastHint, aBroadcast.mBlockSizes.Get<sVectorComponent<uint32_t>>().mData, aBroadcast.mMaxBlockSize,
                    aBroadcast.mBroadcastDimension.Get<sVectorComponent<uint32_t>>().mData, aBroadcast.mMaxBroadcastDimension );
    }

    void sMultiplyOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aIn, ScalarValue &aConstant )
    {
        MultiplyOp( aTensorElementType, aOut, aIn, aConstant );
    }

    void sMultiplyOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aConstant, MultiTensor &aIn )
    {
        MultiplyOp( aTensorElementType, aOut, aIn, aConstant );
    }

    void sMultiplyOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight )
    {
        MultiplyOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sMultiplyOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight )
    {
        MultiplyOp( aTensorElementType, aOut, aRight, aLeft );
    }

    void sSubtractOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight )
    {
        SubtractOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sSubtractOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, sBroadcastInfoComponent &aBroadcast )
    {
        SubtractOp( aTensorElementType, aOut, aLeft, aRight, aBroadcast.mBroadcastHint, aBroadcast.mBlockSizes.Get<sVectorComponent<uint32_t>>().mData, aBroadcast.mMaxBlockSize,
                    aBroadcast.mBroadcastDimension.Get<sVectorComponent<uint32_t>>().mData, aBroadcast.mMaxBroadcastDimension );
    }

    void sSubtractOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aIn, ScalarValue &aConstant )
    {
        SubtractOp( aTensorElementType, aOut, aIn, aConstant );
    }

    void sSubtractOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aConstant, MultiTensor &aIn )
    {
        SubtractOp( aTensorElementType, aOut, aConstant, aIn );
    }

    void sSubtractOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight )
    {
        SubtractOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sSubtractOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight )
    {
        SubtractOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sDivideOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight )
    {
        DivideOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sDivideOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aIn, ScalarValue &aConstant )
    {
        DivideOp( aTensorElementType, aOut, aIn, aConstant );
    }

    void sDivideOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, sBroadcastInfoComponent &aBroadcast )
    {
        DivideOp( aTensorElementType, aOut, aLeft, aRight, aBroadcast.mBroadcastHint, aBroadcast.mBlockSizes.Get<sVectorComponent<uint32_t>>().mData, aBroadcast.mMaxBlockSize,
                  aBroadcast.mBroadcastDimension.Get<sVectorComponent<uint32_t>>().mData, aBroadcast.mMaxBroadcastDimension );
    }

    void sDivideOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aConstant, MultiTensor &aIn )
    {
        DivideOp( aTensorElementType, aOut, aConstant, aIn );
    }

    void sDivideOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight )
    {
        DivideOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sDivideOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight )
    {
        DivideOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sAndOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight )
    {
        AndOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sAndOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, sBroadcastInfoComponent &aBroadcast )
    {
        AndOp( aTensorElementType, aOut, aLeft, aRight, aBroadcast.mBroadcastHint, aBroadcast.mBlockSizes.Get<sVectorComponent<uint32_t>>().mData, aBroadcast.mMaxBlockSize,
               aBroadcast.mBroadcastDimension.Get<sVectorComponent<uint32_t>>().mData, aBroadcast.mMaxBroadcastDimension );
    }

    void sAndOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight )
    {
        AndOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sAndOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aLeft, MultiTensor &aRight )
    {
        AndOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sAndOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight )
    {
        AndOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sAndOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight )
    {
        AndOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sOrOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight )
    {
        OrOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sOrOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, sBroadcastInfoComponent &aBroadcast )
    {
        OrOp( aTensorElementType, aOut, aLeft, aRight, aBroadcast.mBroadcastHint, aBroadcast.mBlockSizes.Get<sVectorComponent<uint32_t>>().mData, aBroadcast.mMaxBlockSize,
              aBroadcast.mBroadcastDimension.Get<sVectorComponent<uint32_t>>().mData, aBroadcast.mMaxBroadcastDimension );
    }

    void sOrOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight )
    {
        OrOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sOrOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aLeft, MultiTensor &aRight )
    {
        OrOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sOrOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight )
    {
        OrOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sOrOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight )
    {
        OrOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sNotOperationController::Run()
    {
        auto lElementType  = Get<sTypeComponent>().mValue;
        auto &lValue       = Get<sMultiTensorComponent>().mValue;
        auto &lOperandData = Get<sNotOperationComponent>();

        NotOp( lElementType, lValue, lOperandData.mOperand.Get<sMultiTensorComponent>().mValue );
    }

    void sBitwiseAndOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight )
    {
        BitwiseAndOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sBitwiseAndOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, sBroadcastInfoComponent &aBroadcast )
    {
        BitwiseAndOp( aTensorElementType, aOut, aLeft, aRight, aBroadcast.mBroadcastHint, aBroadcast.mBlockSizes.Get<sVectorComponent<uint32_t>>().mData, aBroadcast.mMaxBlockSize,
                      aBroadcast.mBroadcastDimension.Get<sVectorComponent<uint32_t>>().mData, aBroadcast.mMaxBroadcastDimension );
    }

    void sBitwiseAndOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight )
    {
        BitwiseAndOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sBitwiseAndOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aLeft, MultiTensor &aRight )
    {
        BitwiseAndOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sBitwiseAndOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight )
    {
        BitwiseAndOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sBitwiseAndOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight )
    {
        BitwiseAndOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sBitwiseOrOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight )
    {
        BitwiseOrOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sBitwiseOrOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, sBroadcastInfoComponent &aBroadcast )
    {
        BitwiseOrOp( aTensorElementType, aOut, aLeft, aRight, aBroadcast.mBroadcastHint, aBroadcast.mBlockSizes.Get<sVectorComponent<uint32_t>>().mData, aBroadcast.mMaxBlockSize,
                     aBroadcast.mBroadcastDimension.Get<sVectorComponent<uint32_t>>().mData, aBroadcast.mMaxBroadcastDimension );
    }

    void sBitwiseOrOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight )
    {
        BitwiseOrOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sBitwiseOrOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aLeft, MultiTensor &aRight )
    {
        BitwiseOrOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sBitwiseOrOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight )
    {
        BitwiseOrOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sBitwiseOrOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight )
    {
        BitwiseOrOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sBitwiseNotOperationController::Run()
    {
        auto lElementType  = Get<sTypeComponent>().mValue;
        auto &lValue       = Get<sMultiTensorComponent>().mValue;
        auto &lOperandData = Get<sBitwiseNotOperationComponent>();

        BitwiseNotOp( lElementType, lValue, lOperandData.mOperand.Get<sMultiTensorComponent>().mValue );
    }

    void sInIntervalOperationController::Run()
    {
        auto &lValue       = Get<sMultiTensorComponent>().mValue;
        auto &lOperandData = Get<sInIntervalOperationComponent>();

        auto &lX          = lOperandData.mX.Get<sMultiTensorComponent>().mValue;
        auto lElementType = lOperandData.mX.Get<sTypeComponent>().mValue;

        if( lOperandData.mLower.Has<sMultiTensorComponent>() && lOperandData.mUpper.Has<sMultiTensorComponent>() )
        {
            auto &lLower = lOperandData.mLower.Get<sMultiTensorComponent>();
            auto &lUpper = lOperandData.mUpper.Get<sMultiTensorComponent>();

            InIntervalOp( lElementType, lValue, lX, lLower.mValue, lUpper.mValue, lOperandData.mStrictLower, lOperandData.mStrictUpper );
        }
        else if( lOperandData.mLower.Has<sMultiTensorComponent>() && lOperandData.mUpper.Has<sVectorComponent<ScalarValue>>() )
        {
            auto &lLower = lOperandData.mLower.Get<sMultiTensorComponent>();
            auto &lUpper = lOperandData.mUpper.Get<sVectorComponent<ScalarValue>>();

            InIntervalOp( lElementType, lValue, lX, lLower.mValue, lUpper.mData, lOperandData.mStrictLower, lOperandData.mStrictUpper );
        }
        else if( lOperandData.mLower.Has<sMultiTensorComponent>() && lOperandData.mUpper.Has<sScalarNodeComponent>() )
        {
            auto &lLower = lOperandData.mLower.Get<sMultiTensorComponent>();
            auto &lUpper = lOperandData.mUpper.Get<sScalarNodeComponent>();

            InIntervalOp( lElementType, lValue, lX, lLower.mValue, lUpper.mValue, lOperandData.mStrictLower, lOperandData.mStrictUpper );
        }
        else if( lOperandData.mLower.Has<sVectorComponent<ScalarValue>>() && lOperandData.mUpper.Has<sMultiTensorComponent>() )
        {
            auto &lLower = lOperandData.mLower.Get<sVectorComponent<ScalarValue>>();
            auto &lUpper = lOperandData.mUpper.Get<sMultiTensorComponent>();

            InIntervalOp( lElementType, lValue, lX, lLower.mData, lUpper.mValue, lOperandData.mStrictLower, lOperandData.mStrictUpper );
        }
        else if( lOperandData.mLower.Has<sVectorComponent<ScalarValue>>() && lOperandData.mUpper.Has<sVectorComponent<ScalarValue>>() )
        {
            auto &lLower = lOperandData.mLower.Get<sVectorComponent<ScalarValue>>();
            auto &lUpper = lOperandData.mUpper.Get<sVectorComponent<ScalarValue>>();

            InIntervalOp( lElementType, lValue, lX, lLower.mData, lUpper.mData, lOperandData.mStrictLower, lOperandData.mStrictUpper );
        }
        else if( lOperandData.mLower.Has<sVectorComponent<ScalarValue>>() && lOperandData.mUpper.Has<sScalarNodeComponent>() )
        {
            auto &lLower = lOperandData.mLower.Get<sVectorComponent<ScalarValue>>();
            auto &lUpper = lOperandData.mUpper.Get<sScalarNodeComponent>();

            InIntervalOp( lElementType, lValue, lX, lLower.mData, lUpper.mValue, lOperandData.mStrictLower, lOperandData.mStrictUpper );
        }
        else if( lOperandData.mLower.Has<sScalarNodeComponent>() && lOperandData.mUpper.Has<sMultiTensorComponent>() )
        {
            auto &lLower = lOperandData.mLower.Get<sScalarNodeComponent>();
            auto &lUpper = lOperandData.mUpper.Get<sMultiTensorComponent>();

            InIntervalOp( lElementType, lValue, lX, lLower.mValue, lUpper.mValue, lOperandData.mStrictLower, lOperandData.mStrictUpper );
        }
        else if( lOperandData.mLower.Has<sScalarNodeComponent>() && lOperandData.mUpper.Has<sVectorComponent<ScalarValue>>() )
        {
            auto &lLower = lOperandData.mLower.Get<sScalarNodeComponent>();
            auto &lUpper = lOperandData.mUpper.Get<sVectorComponent<ScalarValue>>();

            InIntervalOp( lElementType, lValue, lX, lLower.mValue, lUpper.mData, lOperandData.mStrictLower, lOperandData.mStrictUpper );
        }
        else if( lOperandData.mLower.Has<sScalarNodeComponent>() && lOperandData.mUpper.Has<sScalarNodeComponent>() )
        {
            auto &lLower = lOperandData.mLower.Get<sScalarNodeComponent>();
            auto &lUpper = lOperandData.mUpper.Get<sScalarNodeComponent>();

            InIntervalOp( lElementType, lValue, lX, lLower.mValue, lUpper.mValue, lOperandData.mStrictLower, lOperandData.mStrictUpper );
        }
        else
        {
            throw std::runtime_error( "something's wrong" );
        }
    }

    void sEqualOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight )
    {
        EqualOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sEqualOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, sBroadcastInfoComponent &aBroadcast )
    {
        EqualOp( aTensorElementType, aOut, aLeft, aRight, aBroadcast.mBroadcastHint, aBroadcast.mBlockSizes.Get<sVectorComponent<uint32_t>>().mData, aBroadcast.mMaxBlockSize,
                 aBroadcast.mBroadcastDimension.Get<sVectorComponent<uint32_t>>().mData, aBroadcast.mMaxBroadcastDimension );
    }

    void sEqualOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight )
    {
        EqualOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sEqualOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aLeft, MultiTensor &aRight )
    {
        EqualOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sEqualOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight )
    {
        EqualOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sEqualOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight )
    {
        EqualOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sLessThanOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight )
    {
        LessThanOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sLessThanOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, sBroadcastInfoComponent &aBroadcast )
    {
        LessThanOp( aTensorElementType, aOut, aLeft, aRight, aBroadcast.mBroadcastHint, aBroadcast.mBlockSizes.Get<sVectorComponent<uint32_t>>().mData, aBroadcast.mMaxBlockSize,
                    aBroadcast.mBroadcastDimension.Get<sVectorComponent<uint32_t>>().mData, aBroadcast.mMaxBroadcastDimension );
    }

    void sLessThanOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight )
    {
        LessThanOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sLessThanOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aLeft, MultiTensor &aRight )
    {
        LessThanOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sLessThanOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight )
    {
        LessThanOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sLessThanOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight )
    {
        LessThanOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sLessThanOrEqualOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight )
    {
        LessThanOrEqualOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sLessThanOrEqualOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, sBroadcastInfoComponent &aBroadcast )
    {
        LessThanOrEqualOp( aTensorElementType, aOut, aLeft, aRight, aBroadcast.mBroadcastHint, aBroadcast.mBlockSizes.Get<sVectorComponent<uint32_t>>().mData,
                           aBroadcast.mMaxBlockSize, aBroadcast.mBroadcastDimension.Get<sVectorComponent<uint32_t>>().mData, aBroadcast.mMaxBroadcastDimension );
    }

    void sLessThanOrEqualOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight )
    {
        LessThanOrEqualOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sLessThanOrEqualOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aLeft, MultiTensor &aRight )
    {
        LessThanOrEqualOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sLessThanOrEqualOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight )
    {
        LessThanOrEqualOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sLessThanOrEqualOperationController::Op( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight )
    {
        LessThanOrEqualOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sWhereOperationController::Run()
    {
        auto &lValue       = Get<sMultiTensorComponent>().mValue;
        auto &lOperandData = Get<sWhereOperationComponent>();

        auto &lCondition  = lOperandData.mCondition.Get<sMultiTensorComponent>().mValue;
        auto lElementType = lOperandData.mValueIfTrue.Get<sTypeComponent>().mValue;

        if( lOperandData.mValueIfTrue.Has<sMultiTensorComponent>() && lOperandData.mValueIfFalse.Has<sMultiTensorComponent>() )
        {
            auto &lValueIfTrue  = lOperandData.mValueIfTrue.Get<sMultiTensorComponent>();
            auto &lValueIfFalse = lOperandData.mValueIfFalse.Get<sMultiTensorComponent>();

            WhereOp( lElementType, lValue, lCondition, lValueIfTrue.mValue, lValueIfFalse.mValue );
        }
        else if( lOperandData.mValueIfTrue.Has<sMultiTensorComponent>() && lOperandData.mValueIfFalse.Has<sVectorComponent<ScalarValue>>() )
        {
            auto &lValueIfTrue  = lOperandData.mValueIfTrue.Get<sMultiTensorComponent>();
            auto &lValueIfFalse = lOperandData.mValueIfFalse.Get<sVectorComponent<ScalarValue>>();

            WhereOp( lElementType, lValue, lCondition, lValueIfTrue.mValue, lValueIfFalse.mData );
        }
        else if( lOperandData.mValueIfTrue.Has<sMultiTensorComponent>() && lOperandData.mValueIfFalse.Has<sScalarNodeComponent>() )
        {
            auto &lValueIfTrue  = lOperandData.mValueIfTrue.Get<sMultiTensorComponent>();
            auto &lValueIfFalse = lOperandData.mValueIfFalse.Get<sScalarNodeComponent>();

            WhereOp( lElementType, lValue, lCondition, lValueIfTrue.mValue, lValueIfFalse.mValue );
        }
        else if( lOperandData.mValueIfTrue.Has<sVectorComponent<ScalarValue>>() && lOperandData.mValueIfFalse.Has<sMultiTensorComponent>() )
        {
            auto &lValueIfTrue  = lOperandData.mValueIfTrue.Get<sVectorComponent<ScalarValue>>();
            auto &lValueIfFalse = lOperandData.mValueIfFalse.Get<sMultiTensorComponent>();

            WhereOp( lElementType, lValue, lCondition, lValueIfTrue.mData, lValueIfFalse.mValue );
        }
        else if( lOperandData.mValueIfTrue.Has<sVectorComponent<ScalarValue>>() && lOperandData.mValueIfFalse.Has<sVectorComponent<ScalarValue>>() )
        {
            auto &lValueIfTrue  = lOperandData.mValueIfTrue.Get<sVectorComponent<ScalarValue>>();
            auto &lValueIfFalse = lOperandData.mValueIfFalse.Get<sVectorComponent<ScalarValue>>();

            WhereOp( lElementType, lValue, lCondition, lValueIfTrue.mData, lValueIfFalse.mData );
        }
        else if( lOperandData.mValueIfTrue.Has<sVectorComponent<ScalarValue>>() && lOperandData.mValueIfFalse.Has<sScalarNodeComponent>() )
        {
            auto &lValueIfTrue  = lOperandData.mValueIfTrue.Get<sVectorComponent<ScalarValue>>();
            auto &lValueIfFalse = lOperandData.mValueIfFalse.Get<sScalarNodeComponent>();

            WhereOp( lElementType, lValue, lCondition, lValueIfTrue.mData, lValueIfFalse.mValue );
        }
        else if( lOperandData.mValueIfTrue.Has<sScalarNodeComponent>() && lOperandData.mValueIfFalse.Has<sMultiTensorComponent>() )
        {
            auto &lValueIfTrue  = lOperandData.mValueIfTrue.Get<sScalarNodeComponent>();
            auto &lValueIfFalse = lOperandData.mValueIfFalse.Get<sMultiTensorComponent>();

            WhereOp( lElementType, lValue, lCondition, lValueIfTrue.mValue, lValueIfFalse.mValue );
        }
        else if( lOperandData.mValueIfTrue.Has<sScalarNodeComponent>() && lOperandData.mValueIfFalse.Has<sVectorComponent<ScalarValue>>() )
        {
            auto &lValueIfTrue  = lOperandData.mValueIfTrue.Get<sScalarNodeComponent>();
            auto &lValueIfFalse = lOperandData.mValueIfFalse.Get<sVectorComponent<ScalarValue>>();

            WhereOp( lElementType, lValue, lCondition, lValueIfTrue.mValue, lValueIfFalse.mData );
        }
        else if( lOperandData.mValueIfTrue.Has<sScalarNodeComponent>() && lOperandData.mValueIfFalse.Has<sScalarNodeComponent>() )
        {
            auto &lValueIfTrue  = lOperandData.mValueIfTrue.Get<sScalarNodeComponent>();
            auto &lValueIfFalse = lOperandData.mValueIfFalse.Get<sScalarNodeComponent>();

            WhereOp( lElementType, lValue, lCondition, lValueIfTrue.mValue, lValueIfFalse.mValue );
        }
        else
        {
            throw std::runtime_error( "something's wrong" );
        }
    }

    void sLinearSpaceOperationController::Run()
    {
        auto &lValue       = Get<sMultiTensorComponent>().mValue;
        auto &lOperandData = Get<sLinearSpaceComponent>();

        auto &lLeft         = lOperandData.mLeft.Get<sMultiTensorComponent>().mValue;
        auto &lRight        = lOperandData.mRight.Get<sMultiTensorComponent>().mValue;
        auto &lSubdivisions = lOperandData.mSubdivisions.Get<sVectorComponent<uint32_t>>();

        auto lElementType = Get<sTypeComponent>().mValue;

        uint32_t lMaxSubdivisions = 0;
        for( const auto &lSub : lSubdivisions.mValue )
            lMaxSubdivisions = std::max( lMaxSubdivisions, lSub );

        LinearSpaceOp( lElementType, lValue, lLeft, lRight, lSubdivisions.mData, lMaxSubdivisions );
    }

    void sMixOperationController::Run()
    {
        auto &lValue       = Get<sMultiTensorComponent>().mValue;
        auto &lOperandData = Get<sMixNodeComponent>();

        auto &lA = lOperandData.mA.Get<sMultiTensorComponent>().mValue;
        auto &lB = lOperandData.mB.Get<sMultiTensorComponent>().mValue;
        auto &lT = lOperandData.mT.Get<sMultiTensorComponent>().mValue;

        auto lElementType = Get<sTypeComponent>().mValue;

        MixOp( lElementType, lValue, lA, lB, lT );
    }

    void sMultiTensorRunner::Run()
    {
        auto &lValue = Get<sMultiTensorComponent>().mValue;
        if( Has<sConstantValueInitializerComponent>() )
        {
            auto &lInitializer = Get<sConstantValueInitializerComponent>();
            ConstantFill( TypeOf( lInitializer.mValue ), lValue, lInitializer.mValue );
        }
        else if( Has<sVectorInitializerComponent>() )
        {
            auto &lInitializer = Get<sVectorInitializerComponent>();
            DISPATCH_BY_TYPE( TypeOf( lInitializer.mValue[0] ), ResolveAndUpload, ( lInitializer ) );
            ConstantFill( TypeOf( lInitializer.mValue[0] ), lValue, lInitializer.mData );
        }
        else if( Has<sDataInitializerComponent>() )
        {
            auto &lInitializer = Get<sDataInitializerComponent>();
            DISPATCH_BY_TYPE( TypeOf( lInitializer.mValue[0] ), ResolveAndUpload, ( lInitializer, lValue ) );
        }
        else if( Has<sRandomUniformInitializerComponent>() )
        {
            auto &lInitializer = Get<sRandomUniformInitializerComponent>();
            RandomUniformFill( lInitializer.mType, lValue );
        }
        else if( Has<sRandomNormalInitializerComponent>() )
        {
            auto &lInitializer = Get<sRandomNormalInitializerComponent>();
            RandomNormalFill( lInitializer.mType, lValue, lInitializer.mMean, lInitializer.mStd );
        }
        else
        {
            throw std::runtime_error( "Invalid initialization method for multi tensor" );
        }
    }

    void sSample2DOperationController::Run()
    {
        auto &lValue       = Get<sMultiTensorComponent>().mValue;
        auto &lOperandData = Get<sSample2DComponent>();

        auto &lTextures = lOperandData.mTextures.Get<sVectorComponent<Cuda::TextureSampler2D::DeviceData>>().mData;

        if( lOperandData.mX.Has<sMultiTensorComponent>() && lOperandData.mY.Has<sMultiTensorComponent>() )
        {
            auto &lX = lOperandData.mX.Get<sMultiTensorComponent>().mValue;
            auto &lY = lOperandData.mY.Get<sMultiTensorComponent>().mValue;

            Sample2DOp( lValue, lX, lY, lTextures );
        }
        else if( lOperandData.mX.Has<sMultiTensorComponent>() && lOperandData.mY.Has<sVectorComponent<ScalarValue>>() )
        {
            auto &lX = lOperandData.mX.Get<sMultiTensorComponent>().mValue;
            auto &lY = lOperandData.mY.Get<sVectorComponent<ScalarValue>>().mData;

            Sample2DOp( lValue, lX, lY, lTextures );
        }
        else if( lOperandData.mX.Has<sMultiTensorComponent>() && lOperandData.mY.Has<sScalarNodeComponent>() )
        {
            auto &lX = lOperandData.mX.Get<sMultiTensorComponent>().mValue;
            auto &lY = lOperandData.mY.Get<sScalarNodeComponent>().mValue;

            Sample2DOp( lValue, lX, lY, lTextures );
        }
        else if( lOperandData.mX.Has<sVectorComponent<ScalarValue>>() && lOperandData.mY.Has<sMultiTensorComponent>() )
        {
            auto &lX = lOperandData.mX.Get<sVectorComponent<ScalarValue>>().mData;
            auto &lY = lOperandData.mY.Get<sMultiTensorComponent>().mValue;

            Sample2DOp( lValue, lX, lY, lTextures );
        }
        else if( lOperandData.mX.Has<sScalarNodeComponent>() && lOperandData.mY.Has<sMultiTensorComponent>() )
        {
            auto &lX = lOperandData.mX.Get<sScalarNodeComponent>().mValue;
            auto &lY = lOperandData.mY.Get<sMultiTensorComponent>().mValue;

            Sample2DOp( lValue, lX, lY, lTextures );
        }
        else
        {
            throw std::runtime_error( "Invalid arguments" );
        }
    }

    void sToFixedPointOperationController::Run()
    {
        auto &lValue       = Get<sMultiTensorComponent>().mValue;
        auto &lOperandData = Get<sToFixedPointNodeComponent>();
        auto lElementType  = lOperandData.mArray.Get<sTypeComponent>().mValue;
        auto &lArray       = lOperandData.mArray.Get<sMultiTensorComponent>().mValue;
        auto &lScaling     = lOperandData.mScaling.Get<sScalarNodeComponent>().mValue;

        ToFixedPointOp( lElementType, lValue, lOperandData.mOutputType, lArray, lScaling );
    }

    void sAffineNodeController::Run()
    {

        auto &lValue       = Get<sMultiTensorComponent>().mValue;
        auto &lOperandData = Get<sAffineNodeComponent>();
        auto lElementType  = Get<sTypeComponent>().mValue;

        auto &lX = lOperandData.mX.Get<sMultiTensorComponent>();

        if( lOperandData.mA.Has<sMultiTensorComponent>() && lOperandData.mB.Has<sMultiTensorComponent>() )
        {
            auto &lA = lOperandData.mA.Get<sMultiTensorComponent>();
            auto &lB = lOperandData.mB.Get<sMultiTensorComponent>();

            AffineTransformOp( lElementType, lValue, lA.mValue, lX.mValue, lB.mValue );
        }
        else if( lOperandData.mA.Has<sMultiTensorComponent>() && lOperandData.mB.Has<sVectorComponent<ScalarValue>>() )
        {
            auto &lA = lOperandData.mA.Get<sMultiTensorComponent>();
            auto &lB = lOperandData.mB.Get<sVectorComponent<ScalarValue>>();

            AffineTransformOp( lElementType, lValue, lA.mValue, lX.mValue, lB.mData );
        }
        else if( lOperandData.mA.Has<sMultiTensorComponent>() && lOperandData.mB.Has<sScalarNodeComponent>() )
        {
            auto &lA = lOperandData.mA.Get<sMultiTensorComponent>();
            auto &lB = lOperandData.mB.Get<sScalarNodeComponent>();

            AffineTransformOp( lElementType, lValue, lA.mValue, lX.mValue, lB.mValue );
        }
        else if( lOperandData.mA.Has<sVectorComponent<ScalarValue>>() && lOperandData.mB.Has<sMultiTensorComponent>() )
        {
            auto &lA = lOperandData.mA.Get<sVectorComponent<ScalarValue>>();
            auto &lB = lOperandData.mB.Get<sMultiTensorComponent>();

            AffineTransformOp( lElementType, lValue, lA.mData, lX.mValue, lB.mValue );
        }
        else if( lOperandData.mA.Has<sVectorComponent<ScalarValue>>() && lOperandData.mB.Has<sVectorComponent<ScalarValue>>() )
        {
            auto &lA = lOperandData.mA.Get<sVectorComponent<ScalarValue>>();
            auto &lB = lOperandData.mB.Get<sVectorComponent<ScalarValue>>();

            AffineTransformOp( lElementType, lValue, lA.mData, lX.mValue, lB.mData );
        }
        else if( lOperandData.mA.Has<sVectorComponent<ScalarValue>>() && lOperandData.mB.Has<sScalarNodeComponent>() )
        {
            auto &lA = lOperandData.mA.Get<sVectorComponent<ScalarValue>>();
            auto &lB = lOperandData.mB.Get<sScalarNodeComponent>();

            AffineTransformOp( lElementType, lValue, lA.mData, lX.mValue, lB.mValue );
        }
        else if( lOperandData.mA.Has<sScalarNodeComponent>() && lOperandData.mB.Has<sMultiTensorComponent>() )
        {
            auto &lA = lOperandData.mA.Get<sScalarNodeComponent>();
            auto &lB = lOperandData.mB.Get<sMultiTensorComponent>();

            AffineTransformOp( lElementType, lValue, lA.mValue, lX.mValue, lB.mValue );
        }
        else if( lOperandData.mA.Has<sScalarNodeComponent>() && lOperandData.mB.Has<sVectorComponent<ScalarValue>>() )
        {
            auto &lA = lOperandData.mA.Get<sScalarNodeComponent>();
            auto &lB = lOperandData.mB.Get<sVectorComponent<ScalarValue>>();

            AffineTransformOp( lElementType, lValue, lA.mValue, lX.mValue, lB.mData );
        }
        else if( lOperandData.mA.Has<sScalarNodeComponent>() && lOperandData.mB.Has<sScalarNodeComponent>() )
        {
            auto &lA = lOperandData.mA.Get<sScalarNodeComponent>();
            auto &lB = lOperandData.mB.Get<sScalarNodeComponent>();

            AffineTransformOp( lElementType, lValue, lA.mValue, lX.mValue, lB.mValue );
        }
        else
        {
            throw std::runtime_error( "something's wrong" );
        }
    }

    void sFloorOperationController::Run()
    {
        auto &lValue       = Get<sMultiTensorComponent>().mValue;
        auto &lOperandData = Get<sFloorNodeComponent>();

        FloorOp( lValue, lOperandData.mArray.Get<sMultiTensorComponent>().mValue );
    }

    void sCeilOperationController::Run()
    {
        auto &lValue       = Get<sMultiTensorComponent>().mValue;
        auto &lOperandData = Get<sCeilNodeComponent>();

        CeilOp( lValue, lOperandData.mArray.Get<sMultiTensorComponent>().mValue );
    }

    void sAbsOperationController::Run()
    {
        auto &lValue       = Get<sMultiTensorComponent>().mValue;
        auto &lOperandData = Get<sAbsNodeComponent>();
        auto lElementType  = Get<sTypeComponent>().mValue;

        AbsOp( lElementType, lValue, lOperandData.mArray.Get<sMultiTensorComponent>().mValue );
    }

    void sSqrtOperationController::Run()
    {
        auto &lValue       = Get<sMultiTensorComponent>().mValue;
        auto &lOperandData = Get<sSqrtNodeComponent>();
        auto lElementType  = Get<sTypeComponent>().mValue;

        SqrtOp( lElementType, lValue, lOperandData.mArray.Get<sMultiTensorComponent>().mValue );
    }

    void sRoundOperationController::Run()
    {
        auto &lValue       = Get<sMultiTensorComponent>().mValue;
        auto &lOperandData = Get<sRoundNodeComponent>();
        auto lElementType  = Get<sTypeComponent>().mValue;

        RoundOp( lElementType, lValue, lOperandData.mArray.Get<sMultiTensorComponent>().mValue );
    }

    void sCountTrueOperationController::Run()
    {
        auto &lValue       = Get<sMultiTensorComponent>().mValue;
        auto &lOperandData = Get<sCountTrueNodeComponent>();
        auto lElementType  = Get<sTypeComponent>().mValue;

        CountTrueOp( lValue, lOperandData.mArray.Get<sMultiTensorComponent>().mValue, lOperandData.mBlockSizes.Get<sVectorComponent<uint32_t>>().mData,
                     lOperandData.mElementCount.Get<sVectorComponent<uint32_t>>().mData, lOperandData.mMaxBlockSize );
    }

    void sCountNonZeroOperationController::Run()
    {
        auto &lValue       = Get<sMultiTensorComponent>().mValue;
        auto &lOperandData = Get<sCountNonZeroNodeComponent>();
        auto lElementType  = Get<sTypeComponent>().mValue;

        CountNonZeroOp( lElementType, lValue, lOperandData.mArray.Get<sMultiTensorComponent>().mValue, lOperandData.mBlockSizes.Get<sVectorComponent<uint32_t>>().mData,
                        lOperandData.mElementCount.Get<sVectorComponent<uint32_t>>().mData, lOperandData.mMaxBlockSize );
    }

    void sCountZeroOperationController::Run()
    {
        auto &lValue       = Get<sMultiTensorComponent>().mValue;
        auto &lOperandData = Get<sCountZeroNodeComponent>();
        auto lElementType  = Get<sTypeComponent>().mValue;

        CountZeroOp( lElementType, lValue, lOperandData.mArray.Get<sMultiTensorComponent>().mValue, lOperandData.mBlockSizes.Get<sVectorComponent<uint32_t>>().mData,
                     lOperandData.mElementCount.Get<sVectorComponent<uint32_t>>().mData, lOperandData.mMaxBlockSize );
    }

    void sArraySummationOperationController::Run()
    {
        auto &lValue       = Get<sMultiTensorComponent>().mValue;
        auto &lOperandData = Get<sArraySummationNodeComponent>();
        auto lElementType  = Get<sTypeComponent>().mValue;

        ArraySummationOp( lElementType, lValue, lOperandData.mArray.Get<sMultiTensorComponent>().mValue, lOperandData.mBegin.Get<sVectorComponent<uint32_t>>().mData,
                          lOperandData.mEnd.Get<sVectorComponent<uint32_t>>().mData, lOperandData.mElementCount.Get<sVectorComponent<uint32_t>>().mData,
                          lOperandData.mBlockSizes.Get<sVectorComponent<uint32_t>>().mData, lOperandData.mMaxBlockSize );
    }

    void sArraySliceOperationController::Run()
    {
        auto &lValue       = Get<sMultiTensorComponent>().mValue;
        auto &lOperandData = Get<sArraySliceNodeComponent>();
        auto lElementType  = Get<sTypeComponent>().mValue;

        ArraySliceOp( lElementType, lValue, lOperandData.mArray.Get<sMultiTensorComponent>().mValue, lOperandData.mBegin.Get<sVectorComponent<uint32_t>>().mData,
                      lOperandData.mEnd.Get<sVectorComponent<uint32_t>>().mData, lOperandData.mElementCount.Get<sVectorComponent<uint32_t>>().mData,
                      lOperandData.mBlockSizes.Get<sVectorComponent<uint32_t>>().mData, lOperandData.mMaxBlockSize );
    }

    void sDiffOperationController::Run()
    {
        auto &lValue       = Get<sMultiTensorComponent>().mValue;
        auto &lOperandData = Get<sDiffNodeComponent>();
        auto lElementType  = Get<sTypeComponent>().mValue;

        DiffOp( lElementType, lValue, lOperandData.mArray.Get<sMultiTensorComponent>().mValue, lOperandData.mCount,
                lOperandData.mElementCount.Get<sVectorComponent<uint32_t>>().mData, lOperandData.mBlockSizes.Get<sVectorComponent<uint32_t>>().mData, lOperandData.mMaxBlockSize );
    }

    void sShiftOperationController::Run()
    {
        auto &lValue       = Get<sMultiTensorComponent>().mValue;
        auto &lOperandData = Get<sShiftNodeComponent>();
        auto lElementType  = Get<sTypeComponent>().mValue;

        ShiftOp( lElementType, lValue, lOperandData.mArray.Get<sMultiTensorComponent>().mValue, lOperandData.mCount, lOperandData.mFillValue.Get<sScalarNodeComponent>().mValue,
                 lOperandData.mElementCount.Get<sVectorComponent<uint32_t>>().mData, lOperandData.mBlockSizes.Get<sVectorComponent<uint32_t>>().mData, lOperandData.mMaxBlockSize );
    }

    void sConv1DOperationController::Run()
    {
        auto &lValue       = Get<sMultiTensorComponent>().mValue;
        auto &lOperandData = Get<sConv1DNodeComponent>();
        auto lElementType  = Get<sTypeComponent>().mValue;

        Conv1DOp( lElementType, lValue, lOperandData.mArray0.Get<sMultiTensorComponent>().mValue, lOperandData.mElementCount0.Get<sVectorComponent<uint32_t>>().mData,
                  lOperandData.mBlockSizes0.Get<sVectorComponent<uint32_t>>().mData, lOperandData.mMaxElementCount0, lOperandData.mMaxBlockSize0,
                  lOperandData.mArray1.Get<sMultiTensorComponent>().mValue, lOperandData.mElementCount1.Get<sVectorComponent<uint32_t>>().mData,
                  lOperandData.mBlockSizes1.Get<sVectorComponent<uint32_t>>().mData, lOperandData.mMaxBlockSize1 );
    }

    void sHCatOperationController::Run()
    {
        auto &lValue       = Get<sMultiTensorComponent>().mValue;
        auto &lOperandData = Get<sHCatNodeComponent>();
        auto lElementType  = Get<sTypeComponent>().mValue;

        HCatOp( lElementType, lValue, lOperandData.mArray0.Get<sMultiTensorComponent>().mValue, lOperandData.mElementCount0.Get<sVectorComponent<uint32_t>>().mData,
                lOperandData.mArray1.Get<sMultiTensorComponent>().mValue, lOperandData.mElementCount1.Get<sVectorComponent<uint32_t>>().mData,
                lOperandData.mBlockSizes.Get<sVectorComponent<uint32_t>>().mData, lOperandData.mMaxBlockSize );
    }
} // namespace LTSE::TensorOps
