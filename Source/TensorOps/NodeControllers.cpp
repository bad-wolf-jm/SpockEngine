/// @file   NodeControllers.cpp
///
/// @brief  Controller implementation
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert. All rights reserved.

#include <algorithm>
#include <cassert>
#include <cmath>

#include "Implementation/KernelLaunchers.h"
#include "NodeComponents.h"
#include "NodeControllers.h"

#include "Core/CUDA/Texture/Texture2D.h"

namespace SE::TensorOps
{

    using namespace SE::Core;

    void sARangeOperationController::Run()
    {
        auto &lValue       = Get<multi_tensor_value_t>().mValue;
        auto &lOperandData = Get<arange_operation_t>();

        auto &lLeft  = lOperandData.mLeft.Get<vector_buffer_t>().mValue;
        auto &lRight = lOperandData.mRight.Get<vector_buffer_t>().mValue;
        auto &lDelta = lOperandData.mDelta.Get<vector_buffer_t>().mValue;

        auto lElementType = Get<type_t>().mValue;

        uint32_t lMaxSubdivisions = 0;
        for( const auto &lSub : lValue.Shape().mShape )
            lMaxSubdivisions = std::max( lMaxSubdivisions, lSub[0] );

        ARangeOp( lElementType, lValue, lLeft, lRight, lDelta, lMaxSubdivisions );
    }

    void sArrayOperationController::Run()
    {
        auto &lValue = Get<multi_tensor_value_t>().mValue;

        auto lElementType = Get<type_t>().mValue;

        if( Has<repeat_operation_t>() )
        {
            auto    &lOperandData    = Get<repeat_operation_t>();
            auto    &lArray          = lOperandData.mArray.Get<multi_tensor_value_t>().mValue;
            auto    &lRepetitions    = lOperandData.mRepetitions.Get<u32_vector_t>();
            uint32_t lMaxRepetitions = 0;
            for( const auto &lSub : lRepetitions.mValue )
                lMaxRepetitions = std::max( lMaxRepetitions, lSub );
            RepeatOp( lElementType, lValue, lArray, lOperandData.mRepetitions.Get<vector_buffer_t>().mValue, lMaxRepetitions );
            return;
        }

        if( Has<tile_operation_t>() )
        {
            auto    &lOperandData    = Get<tile_operation_t>();
            auto    &lArray          = lOperandData.mArray.Get<multi_tensor_value_t>().mValue;
            auto    &lRepetitions    = lOperandData.mRepetitions.Get<u32_vector_t>();
            uint32_t lMaxRepetitions = 0;
            for( const auto &lSub : lRepetitions.mValue )
                lMaxRepetitions = std::max( lMaxRepetitions, lSub );
            TileOp( lElementType, lValue, lArray, lOperandData.mRepetitions.Get<vector_buffer_t>().mValue, lMaxRepetitions );
            return;
        }
    }

    void sBinaryOperationController::Run()
    {
        auto &lValue       = Get<multi_tensor_value_t>().mValue;
        auto &lOperandData = Get<binary_operation_t>();
        auto  lElementType = Get<type_t>().mValue;

        if( lOperandData.mLeftOperand.Has<multi_tensor_value_t>() && lOperandData.mRightOperand.Has<multi_tensor_value_t>() )
        {
            auto &lLeftOperandData    = lOperandData.mLeftOperand.Get<multi_tensor_value_t>();
            auto &lRightOpOperandData = lOperandData.mRightOperand.Get<multi_tensor_value_t>();

            if( Has<broadcast_info_t>() )
                Op( lElementType, lValue, lLeftOperandData.mValue, lRightOpOperandData.mValue, Get<broadcast_info_t>() );
            else
                Op( lElementType, lValue, lLeftOperandData.mValue, lRightOpOperandData.mValue );
        }
        else if( lOperandData.mLeftOperand.Has<multi_tensor_value_t>() &&
                 lOperandData.mRightOperand.Has<scalar_value_vector_t>() )
        {
            auto &lLeftOperandData    = lOperandData.mLeftOperand.Get<multi_tensor_value_t>();
            auto &lRightOpOperandData = lOperandData.mRightOperand.Get<vector_buffer_t>();

            Op( lElementType, lValue, lLeftOperandData.mValue, lRightOpOperandData.mValue );
        }
        else if( lOperandData.mLeftOperand.Has<scalar_value_vector_t>() &&
                 lOperandData.mRightOperand.Has<multi_tensor_value_t>() )
        {
            auto &lLeftOperandData    = lOperandData.mLeftOperand.Get<vector_buffer_t>();
            auto &lRightOpOperandData = lOperandData.mRightOperand.Get<multi_tensor_value_t>();

            Op( lElementType, lValue, lLeftOperandData.mValue, lRightOpOperandData.mValue );
        }
        else if( lOperandData.mLeftOperand.Has<multi_tensor_value_t>() && lOperandData.mRightOperand.Has<scalar_node_t>() )
        {
            auto &lLeftOperandData    = lOperandData.mLeftOperand.Get<multi_tensor_value_t>();
            auto &lRightOpOperandData = lOperandData.mRightOperand.Get<scalar_node_t>();

            Op( lElementType, lValue, lLeftOperandData.mValue, lRightOpOperandData.mValue );
        }
        else if( lOperandData.mLeftOperand.Has<scalar_node_t>() && lOperandData.mRightOperand.Has<multi_tensor_value_t>() )
        {
            auto &lLeftOperandData    = lOperandData.mLeftOperand.Get<scalar_node_t>();
            auto &lRightOpOperandData = lOperandData.mRightOperand.Get<multi_tensor_value_t>();

            Op( lElementType, lValue, lLeftOperandData.mValue, lRightOpOperandData.mValue );
        }
        else
        {
            throw std::runtime_error( "something's wrong" );
        }
    }

    void sBinaryBooleanOperationController::Run()
    {

        auto &lValue       = Get<multi_tensor_value_t>().mValue;
        auto &lOperandData = Get<binary_operation_t>();

        if( lOperandData.mLeftOperand.Has<multi_tensor_value_t>() && lOperandData.mRightOperand.Has<multi_tensor_value_t>() )
        {
            auto &lLeftOperandData    = lOperandData.mLeftOperand.Get<multi_tensor_value_t>();
            auto &lRightOpOperandData = lOperandData.mRightOperand.Get<multi_tensor_value_t>();
            auto  lElementType        = lOperandData.mLeftOperand.Get<type_t>().mValue;

            if( Has<broadcast_info_t>() )
                Op( lElementType, lValue, lLeftOperandData.mValue, lRightOpOperandData.mValue, Get<broadcast_info_t>() );
            else
                Op( lElementType, lValue, lLeftOperandData.mValue, lRightOpOperandData.mValue );
        }
        else if( lOperandData.mLeftOperand.Has<multi_tensor_value_t>() &&
                 lOperandData.mRightOperand.Has<scalar_value_vector_t>() )
        {
            auto &lLeftOperandData    = lOperandData.mLeftOperand.Get<multi_tensor_value_t>();
            auto &lRightOpOperandData = lOperandData.mRightOperand.Get<vector_buffer_t>();
            auto  lElementType        = lOperandData.mLeftOperand.Get<type_t>().mValue;

            Op( lElementType, lValue, lLeftOperandData.mValue, lRightOpOperandData.mValue );
        }
        else if( lOperandData.mLeftOperand.Has<scalar_value_vector_t>() &&
                 lOperandData.mRightOperand.Has<multi_tensor_value_t>() )
        {
            auto &lLeftOperandData    = lOperandData.mLeftOperand.Get<vector_buffer_t>();
            auto &lRightOpOperandData = lOperandData.mRightOperand.Get<multi_tensor_value_t>();
            auto  lElementType        = lOperandData.mRightOperand.Get<type_t>().mValue;

            Op( lElementType, lValue, lLeftOperandData.mValue, lRightOpOperandData.mValue );
        }
        else if( lOperandData.mLeftOperand.Has<multi_tensor_value_t>() && lOperandData.mRightOperand.Has<scalar_node_t>() )
        {
            auto &lLeftOperandData    = lOperandData.mLeftOperand.Get<multi_tensor_value_t>();
            auto &lRightOpOperandData = lOperandData.mRightOperand.Get<scalar_node_t>();
            auto  lElementType        = lOperandData.mLeftOperand.Get<type_t>().mValue;

            Op( lElementType, lValue, lLeftOperandData.mValue, lRightOpOperandData.mValue );
        }
        else if( lOperandData.mLeftOperand.Has<scalar_node_t>() && lOperandData.mRightOperand.Has<multi_tensor_value_t>() )
        {
            auto &lLeftOperandData    = lOperandData.mLeftOperand.Get<scalar_node_t>();
            auto &lRightOpOperandData = lOperandData.mRightOperand.Get<multi_tensor_value_t>();
            auto  lElementType        = lOperandData.mRightOperand.Get<type_t>().mValue;

            Op( lElementType, lValue, lLeftOperandData.mValue, lRightOpOperandData.mValue );
        }
        else
        {
            throw std::runtime_error( "something's wrong" );
        }
    }

    void sAddOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight )
    {
        AddOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sAddOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                                      broadcast_info_t &aBroadcast )
    {
        AddOp( aTensorElementType, aOut, aLeft, aRight, aBroadcast.mBroadcastHint,
               aBroadcast.mBlockSizes.Get<vector_buffer_t>().mValue, aBroadcast.mMaxBlockSize,
               aBroadcast.mBroadcastDimension.Get<vector_buffer_t>().mValue, aBroadcast.mMaxBroadcastDimension );
    }

    void sAddOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aIn, scalar_value_t &aConstant )
    {
        AddOp( aTensorElementType, aOut, aIn, aConstant );
    }

    void sAddOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aConstant, multi_tensor_t &aIn )
    {
        AddOp( aTensorElementType, aOut, aIn, aConstant );
    }

    void sAddOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight )
    {
        AddOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sAddOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight )
    {
        AddOp( aTensorElementType, aOut, aRight, aLeft );
    }

    void sMultiplyOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight )
    {
        MultiplyOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sMultiplyOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                                           broadcast_info_t &aBroadcast )
    {
        MultiplyOp( aTensorElementType, aOut, aLeft, aRight, aBroadcast.mBroadcastHint,
                    aBroadcast.mBlockSizes.Get<vector_buffer_t>().mValue, aBroadcast.mMaxBlockSize,
                    aBroadcast.mBroadcastDimension.Get<vector_buffer_t>().mValue, aBroadcast.mMaxBroadcastDimension );
    }

    void sMultiplyOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aIn,
                                           scalar_value_t &aConstant )
    {
        MultiplyOp( aTensorElementType, aOut, aIn, aConstant );
    }

    void sMultiplyOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aConstant,
                                           multi_tensor_t &aIn )
    {
        MultiplyOp( aTensorElementType, aOut, aIn, aConstant );
    }

    void sMultiplyOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                           memory_buffer_t &aRight )
    {
        MultiplyOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sMultiplyOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft,
                                           multi_tensor_t &aRight )
    {
        MultiplyOp( aTensorElementType, aOut, aRight, aLeft );
    }

    void sSubtractOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight )
    {
        SubtractOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sSubtractOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                                           broadcast_info_t &aBroadcast )
    {
        SubtractOp( aTensorElementType, aOut, aLeft, aRight, aBroadcast.mBroadcastHint,
                    aBroadcast.mBlockSizes.Get<vector_buffer_t>().mValue, aBroadcast.mMaxBlockSize,
                    aBroadcast.mBroadcastDimension.Get<vector_buffer_t>().mValue, aBroadcast.mMaxBroadcastDimension );
    }

    void sSubtractOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aIn,
                                           scalar_value_t &aConstant )
    {
        SubtractOp( aTensorElementType, aOut, aIn, aConstant );
    }

    void sSubtractOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aConstant,
                                           multi_tensor_t &aIn )
    {
        SubtractOp( aTensorElementType, aOut, aConstant, aIn );
    }

    void sSubtractOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                           memory_buffer_t &aRight )
    {
        SubtractOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sSubtractOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft,
                                           multi_tensor_t &aRight )
    {
        SubtractOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sDivideOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight )
    {
        DivideOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sDivideOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aIn, scalar_value_t &aConstant )
    {
        DivideOp( aTensorElementType, aOut, aIn, aConstant );
    }

    void sDivideOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                                         broadcast_info_t &aBroadcast )
    {
        DivideOp( aTensorElementType, aOut, aLeft, aRight, aBroadcast.mBroadcastHint,
                  aBroadcast.mBlockSizes.Get<vector_buffer_t>().mValue, aBroadcast.mMaxBlockSize,
                  aBroadcast.mBroadcastDimension.Get<vector_buffer_t>().mValue, aBroadcast.mMaxBroadcastDimension );
    }

    void sDivideOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aConstant, multi_tensor_t &aIn )
    {
        DivideOp( aTensorElementType, aOut, aConstant, aIn );
    }

    void sDivideOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight )
    {
        DivideOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sDivideOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight )
    {
        DivideOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sAndOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight )
    {
        AndOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sAndOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                                      broadcast_info_t &aBroadcast )
    {
        AndOp( aTensorElementType, aOut, aLeft, aRight, aBroadcast.mBroadcastHint,
               aBroadcast.mBlockSizes.Get<vector_buffer_t>().mValue, aBroadcast.mMaxBlockSize,
               aBroadcast.mBroadcastDimension.Get<vector_buffer_t>().mValue, aBroadcast.mMaxBroadcastDimension );
    }

    void sAndOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, scalar_value_t &aRight )
    {
        AndOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sAndOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aLeft, multi_tensor_t &aRight )
    {
        AndOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sAndOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight )
    {
        AndOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sAndOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight )
    {
        AndOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sOrOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight )
    {
        OrOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sOrOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                                     broadcast_info_t &aBroadcast )
    {
        OrOp( aTensorElementType, aOut, aLeft, aRight, aBroadcast.mBroadcastHint,
              aBroadcast.mBlockSizes.Get<vector_buffer_t>().mValue, aBroadcast.mMaxBlockSize,
              aBroadcast.mBroadcastDimension.Get<vector_buffer_t>().mValue, aBroadcast.mMaxBroadcastDimension );
    }

    void sOrOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, scalar_value_t &aRight )
    {
        OrOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sOrOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aLeft, multi_tensor_t &aRight )
    {
        OrOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sOrOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight )
    {
        OrOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sOrOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight )
    {
        OrOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sNotOperationController::Run()
    {
        auto  lElementType = Get<type_t>().mValue;
        auto &lValue       = Get<multi_tensor_value_t>().mValue;
        auto &lOperandData = Get<not_operation_t>();

        NotOp( lElementType, lValue, lOperandData.mOperand.Get<multi_tensor_value_t>().mValue );
    }

    void sBitwiseAndOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                             multi_tensor_t &aRight )
    {
        BitwiseAndOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sBitwiseAndOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                             multi_tensor_t &aRight, broadcast_info_t &aBroadcast )
    {
        BitwiseAndOp( aTensorElementType, aOut, aLeft, aRight, aBroadcast.mBroadcastHint,
                      aBroadcast.mBlockSizes.Get<vector_buffer_t>().mValue, aBroadcast.mMaxBlockSize,
                      aBroadcast.mBroadcastDimension.Get<vector_buffer_t>().mValue, aBroadcast.mMaxBroadcastDimension );
    }

    void sBitwiseAndOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                             scalar_value_t &aRight )
    {
        BitwiseAndOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sBitwiseAndOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aLeft,
                                             multi_tensor_t &aRight )
    {
        BitwiseAndOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sBitwiseAndOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                             memory_buffer_t &aRight )
    {
        BitwiseAndOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sBitwiseAndOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft,
                                             multi_tensor_t &aRight )
    {
        BitwiseAndOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sBitwiseOrOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                            multi_tensor_t &aRight )
    {
        BitwiseOrOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sBitwiseOrOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                                            broadcast_info_t &aBroadcast )
    {
        BitwiseOrOp( aTensorElementType, aOut, aLeft, aRight, aBroadcast.mBroadcastHint,
                     aBroadcast.mBlockSizes.Get<vector_buffer_t>().mValue, aBroadcast.mMaxBlockSize,
                     aBroadcast.mBroadcastDimension.Get<vector_buffer_t>().mValue, aBroadcast.mMaxBroadcastDimension );
    }

    void sBitwiseOrOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                            scalar_value_t &aRight )
    {
        BitwiseOrOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sBitwiseOrOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aLeft,
                                            multi_tensor_t &aRight )
    {
        BitwiseOrOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sBitwiseOrOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                            memory_buffer_t &aRight )
    {
        BitwiseOrOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sBitwiseOrOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft,
                                            multi_tensor_t &aRight )
    {
        BitwiseOrOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sBitwiseNotOperationController::Run()
    {
        auto  lElementType = Get<type_t>().mValue;
        auto &lValue       = Get<multi_tensor_value_t>().mValue;
        auto &lOperandData = Get<bitwise_not_operation_t>();

        BitwiseNotOp( lElementType, lValue, lOperandData.mOperand.Get<multi_tensor_value_t>().mValue );
    }

    void sInIntervalOperationController::Run()
    {
        auto &lValue       = Get<multi_tensor_value_t>().mValue;
        auto &lOperandData = Get<in_interval_operation_t>();

        auto &lX           = lOperandData.mX.Get<multi_tensor_value_t>().mValue;
        auto  lElementType = lOperandData.mX.Get<type_t>().mValue;

        if( lOperandData.mLower.Has<multi_tensor_value_t>() && lOperandData.mUpper.Has<multi_tensor_value_t>() )
        {
            auto &lLower = lOperandData.mLower.Get<multi_tensor_value_t>();
            auto &lUpper = lOperandData.mUpper.Get<multi_tensor_value_t>();

            InIntervalOp( lElementType, lValue, lX, lLower.mValue, lUpper.mValue, lOperandData.mStrictLower,
                          lOperandData.mStrictUpper );
        }
        else if( lOperandData.mLower.Has<multi_tensor_value_t>() && lOperandData.mUpper.Has<scalar_value_vector_t>() )
        {
            auto &lLower = lOperandData.mLower.Get<multi_tensor_value_t>();
            auto &lUpper = lOperandData.mUpper.Get<vector_buffer_t>();

            InIntervalOp( lElementType, lValue, lX, lLower.mValue, lUpper.mValue, lOperandData.mStrictLower,
                          lOperandData.mStrictUpper );
        }
        else if( lOperandData.mLower.Has<multi_tensor_value_t>() && lOperandData.mUpper.Has<scalar_node_t>() )
        {
            auto &lLower = lOperandData.mLower.Get<multi_tensor_value_t>();
            auto &lUpper = lOperandData.mUpper.Get<scalar_node_t>();

            InIntervalOp( lElementType, lValue, lX, lLower.mValue, lUpper.mValue, lOperandData.mStrictLower,
                          lOperandData.mStrictUpper );
        }
        else if( lOperandData.mLower.Has<scalar_value_vector_t>() && lOperandData.mUpper.Has<multi_tensor_value_t>() )
        {
            auto &lLower = lOperandData.mLower.Get<vector_buffer_t>();
            auto &lUpper = lOperandData.mUpper.Get<multi_tensor_value_t>();

            InIntervalOp( lElementType, lValue, lX, lLower.mValue, lUpper.mValue, lOperandData.mStrictLower,
                          lOperandData.mStrictUpper );
        }
        else if( lOperandData.mLower.Has<scalar_value_vector_t>() && lOperandData.mUpper.Has<scalar_value_vector_t>() )
        {
            auto &lLower = lOperandData.mLower.Get<vector_buffer_t>();
            auto &lUpper = lOperandData.mUpper.Get<vector_buffer_t>();

            InIntervalOp( lElementType, lValue, lX, lLower.mValue, lUpper.mValue, lOperandData.mStrictLower,
                          lOperandData.mStrictUpper );
        }
        else if( lOperandData.mLower.Has<scalar_value_vector_t>() && lOperandData.mUpper.Has<scalar_node_t>() )
        {
            auto &lLower = lOperandData.mLower.Get<vector_buffer_t>();
            auto &lUpper = lOperandData.mUpper.Get<scalar_node_t>();

            InIntervalOp( lElementType, lValue, lX, lLower.mValue, lUpper.mValue, lOperandData.mStrictLower,
                          lOperandData.mStrictUpper );
        }
        else if( lOperandData.mLower.Has<scalar_node_t>() && lOperandData.mUpper.Has<multi_tensor_value_t>() )
        {
            auto &lLower = lOperandData.mLower.Get<scalar_node_t>();
            auto &lUpper = lOperandData.mUpper.Get<multi_tensor_value_t>();

            InIntervalOp( lElementType, lValue, lX, lLower.mValue, lUpper.mValue, lOperandData.mStrictLower,
                          lOperandData.mStrictUpper );
        }
        else if( lOperandData.mLower.Has<scalar_node_t>() && lOperandData.mUpper.Has<scalar_value_vector_t>() )
        {
            auto &lLower = lOperandData.mLower.Get<scalar_node_t>();
            auto &lUpper = lOperandData.mUpper.Get<vector_buffer_t>();

            InIntervalOp( lElementType, lValue, lX, lLower.mValue, lUpper.mValue, lOperandData.mStrictLower,
                          lOperandData.mStrictUpper );
        }
        else if( lOperandData.mLower.Has<scalar_node_t>() && lOperandData.mUpper.Has<scalar_node_t>() )
        {
            auto &lLower = lOperandData.mLower.Get<scalar_node_t>();
            auto &lUpper = lOperandData.mUpper.Get<scalar_node_t>();

            InIntervalOp( lElementType, lValue, lX, lLower.mValue, lUpper.mValue, lOperandData.mStrictLower,
                          lOperandData.mStrictUpper );
        }
        else
        {
            throw std::runtime_error( "something's wrong" );
        }
    }

    void sEqualOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight )
    {
        EqualOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sEqualOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                                        broadcast_info_t &aBroadcast )
    {
        EqualOp( aTensorElementType, aOut, aLeft, aRight, aBroadcast.mBroadcastHint,
                 aBroadcast.mBlockSizes.Get<vector_buffer_t>().mValue, aBroadcast.mMaxBlockSize,
                 aBroadcast.mBroadcastDimension.Get<vector_buffer_t>().mValue, aBroadcast.mMaxBroadcastDimension );
    }

    void sEqualOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, scalar_value_t &aRight )
    {
        EqualOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sEqualOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aLeft, multi_tensor_t &aRight )
    {
        EqualOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sEqualOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight )
    {
        EqualOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sEqualOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight )
    {
        EqualOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sLessThanOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight )
    {
        LessThanOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sLessThanOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                                           broadcast_info_t &aBroadcast )
    {
        LessThanOp( aTensorElementType, aOut, aLeft, aRight, aBroadcast.mBroadcastHint,
                    aBroadcast.mBlockSizes.Get<vector_buffer_t>().mValue, aBroadcast.mMaxBlockSize,
                    aBroadcast.mBroadcastDimension.Get<vector_buffer_t>().mValue, aBroadcast.mMaxBroadcastDimension );
    }

    void sLessThanOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, scalar_value_t &aRight )
    {
        LessThanOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sLessThanOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aLeft, multi_tensor_t &aRight )
    {
        LessThanOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sLessThanOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                           memory_buffer_t &aRight )
    {
        LessThanOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sLessThanOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft,
                                           multi_tensor_t &aRight )
    {
        LessThanOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sLessThanOrEqualOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                                  multi_tensor_t &aRight )
    {
        LessThanOrEqualOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sLessThanOrEqualOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                                  multi_tensor_t &aRight, broadcast_info_t &aBroadcast )
    {
        LessThanOrEqualOp( aTensorElementType, aOut, aLeft, aRight, aBroadcast.mBroadcastHint,
                           aBroadcast.mBlockSizes.Get<vector_buffer_t>().mValue, aBroadcast.mMaxBlockSize,
                           aBroadcast.mBroadcastDimension.Get<vector_buffer_t>().mValue, aBroadcast.mMaxBroadcastDimension );
    }

    void sLessThanOrEqualOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                                  scalar_value_t &aRight )
    {
        LessThanOrEqualOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sLessThanOrEqualOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aLeft,
                                                  multi_tensor_t &aRight )
    {
        LessThanOrEqualOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sLessThanOrEqualOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                                  memory_buffer_t &aRight )
    {
        LessThanOrEqualOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sLessThanOrEqualOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft,
                                                  multi_tensor_t &aRight )
    {
        LessThanOrEqualOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sWhereOperationController::Run()
    {
        auto &lValue       = Get<multi_tensor_value_t>().mValue;
        auto &lOperandData = Get<where_operation_t>();

        auto &lCondition   = lOperandData.mCondition.Get<multi_tensor_value_t>().mValue;
        auto  lElementType = lOperandData.mValueIfTrue.Get<type_t>().mValue;

        if( lOperandData.mValueIfTrue.Has<multi_tensor_value_t>() && lOperandData.mValueIfFalse.Has<multi_tensor_value_t>() )
        {
            auto &lValueIfTrue  = lOperandData.mValueIfTrue.Get<multi_tensor_value_t>();
            auto &lValueIfFalse = lOperandData.mValueIfFalse.Get<multi_tensor_value_t>();

            WhereOp( lElementType, lValue, lCondition, lValueIfTrue.mValue, lValueIfFalse.mValue );
        }
        else if( lOperandData.mValueIfTrue.Has<multi_tensor_value_t>() &&
                 lOperandData.mValueIfFalse.Has<scalar_value_vector_t>() )
        {
            auto &lValueIfTrue  = lOperandData.mValueIfTrue.Get<multi_tensor_value_t>();
            auto &lValueIfFalse = lOperandData.mValueIfFalse.Get<vector_buffer_t>();

            WhereOp( lElementType, lValue, lCondition, lValueIfTrue.mValue, lValueIfFalse.mValue );
        }
        else if( lOperandData.mValueIfTrue.Has<multi_tensor_value_t>() && lOperandData.mValueIfFalse.Has<scalar_node_t>() )
        {
            auto &lValueIfTrue  = lOperandData.mValueIfTrue.Get<multi_tensor_value_t>();
            auto &lValueIfFalse = lOperandData.mValueIfFalse.Get<scalar_node_t>();

            WhereOp( lElementType, lValue, lCondition, lValueIfTrue.mValue, lValueIfFalse.mValue );
        }
        else if( lOperandData.mValueIfTrue.Has<scalar_value_vector_t>() &&
                 lOperandData.mValueIfFalse.Has<multi_tensor_value_t>() )
        {
            auto &lValueIfTrue  = lOperandData.mValueIfTrue.Get<vector_buffer_t>();
            auto &lValueIfFalse = lOperandData.mValueIfFalse.Get<multi_tensor_value_t>();

            WhereOp( lElementType, lValue, lCondition, lValueIfTrue.mValue, lValueIfFalse.mValue );
        }
        else if( lOperandData.mValueIfTrue.Has<scalar_value_vector_t>() &&
                 lOperandData.mValueIfFalse.Has<scalar_value_vector_t>() )
        {
            auto &lValueIfTrue  = lOperandData.mValueIfTrue.Get<vector_buffer_t>();
            auto &lValueIfFalse = lOperandData.mValueIfFalse.Get<vector_buffer_t>();

            WhereOp( lElementType, lValue, lCondition, lValueIfTrue.mValue, lValueIfFalse.mValue );
        }
        else if( lOperandData.mValueIfTrue.Has<scalar_value_vector_t>() &&
                 lOperandData.mValueIfFalse.Has<scalar_node_t>() )
        {
            auto &lValueIfTrue  = lOperandData.mValueIfTrue.Get<vector_buffer_t>();
            auto &lValueIfFalse = lOperandData.mValueIfFalse.Get<scalar_node_t>();

            WhereOp( lElementType, lValue, lCondition, lValueIfTrue.mValue, lValueIfFalse.mValue );
        }
        else if( lOperandData.mValueIfTrue.Has<scalar_node_t>() && lOperandData.mValueIfFalse.Has<multi_tensor_value_t>() )
        {
            auto &lValueIfTrue  = lOperandData.mValueIfTrue.Get<scalar_node_t>();
            auto &lValueIfFalse = lOperandData.mValueIfFalse.Get<multi_tensor_value_t>();

            WhereOp( lElementType, lValue, lCondition, lValueIfTrue.mValue, lValueIfFalse.mValue );
        }
        else if( lOperandData.mValueIfTrue.Has<scalar_node_t>() &&
                 lOperandData.mValueIfFalse.Has<scalar_value_vector_t>() )
        {
            auto &lValueIfTrue  = lOperandData.mValueIfTrue.Get<scalar_node_t>();
            auto &lValueIfFalse = lOperandData.mValueIfFalse.Get<vector_buffer_t>();

            WhereOp( lElementType, lValue, lCondition, lValueIfTrue.mValue, lValueIfFalse.mValue );
        }
        else if( lOperandData.mValueIfTrue.Has<scalar_node_t>() && lOperandData.mValueIfFalse.Has<scalar_node_t>() )
        {
            auto &lValueIfTrue  = lOperandData.mValueIfTrue.Get<scalar_node_t>();
            auto &lValueIfFalse = lOperandData.mValueIfFalse.Get<scalar_node_t>();

            WhereOp( lElementType, lValue, lCondition, lValueIfTrue.mValue, lValueIfFalse.mValue );
        }
        else
        {
            throw std::runtime_error( "something's wrong" );
        }
    }

    void sLinearSpaceOperationController::Run()
    {
        auto &lValue       = Get<multi_tensor_value_t>().mValue;
        auto &lOperandData = Get<linear_space_operation_t>();

        auto &lLeft         = lOperandData.mLeft.Get<multi_tensor_value_t>().mValue;
        auto &lRight        = lOperandData.mRight.Get<multi_tensor_value_t>().mValue;
        auto &lSubdivisions = lOperandData.mSubdivisions.Get<u32_vector_t>();

        auto lElementType = Get<type_t>().mValue;

        uint32_t lMaxSubdivisions = 0;
        for( const auto &lSub : lSubdivisions.mValue )
            lMaxSubdivisions = std::max( lMaxSubdivisions, lSub );

        LinearSpaceOp( lElementType, lValue, lLeft, lRight, lOperandData.mSubdivisions.Get<vector_buffer_t>().mValue,
                       lMaxSubdivisions );
    }

    void sMixOperationController::Run()
    {
        auto &lValue       = Get<multi_tensor_value_t>().mValue;
        auto &lOperandData = Get<mix_operation_t>();

        auto &lA = lOperandData.mA.Get<multi_tensor_value_t>().mValue;
        auto &lB = lOperandData.mB.Get<multi_tensor_value_t>().mValue;
        auto &lT = lOperandData.mT.Get<multi_tensor_value_t>().mValue;

        auto lElementType = Get<type_t>().mValue;

        MixOp( lElementType, lValue, lA, lB, lT );
    }

    void sMultiTensorRunner::Run()
    {
        auto &lValue = Get<multi_tensor_value_t>().mValue;
        if( Has<constant_value_initializer_t>() )
        {
            auto &lInitializer = Get<constant_value_initializer_t>();
            ConstantFill( TypeOf( lInitializer.mValue ), lValue, lInitializer.mValue );
        }
        else if( Has<vector_initializer_t>() )
        {
            auto &lInitializer = Get<vector_initializer_t>();
            DISPATCH_BY_TYPE( TypeOf( lInitializer.mValue[0] ), ResolveAndUpload, ( lInitializer ) );
            ConstantFill( TypeOf( lInitializer.mValue[0] ), lValue, lInitializer.mData );
        }
        else if( Has<data_initializer_t>() )
        {
            auto &lInitializer = Get<data_initializer_t>();
            DISPATCH_BY_TYPE( TypeOf( lInitializer.mValue[0] ), ResolveAndUpload, ( lInitializer, lValue ) );
        }
        else if( Has<random_uniform_initializer_t>() )
        {
            auto &lInitializer = Get<random_uniform_initializer_t>();
            RandomUniformFill( lInitializer.mType, lValue );
        }
        else if( Has<random_normal_initializer_t>() )
        {
            auto &lInitializer = Get<random_normal_initializer_t>();
            RandomNormalFill( lInitializer.mType, lValue, lInitializer.mMean, lInitializer.mStd );
        }
        else
        {
            throw std::runtime_error( "Invalid initialization method for multi tensor" );
        }
    }

    void sSample2DOperationController::Run()
    {
        auto &lValue       = Get<multi_tensor_value_t>().mValue;
        auto &lOperandData = Get<sample2D_operation_t>();

        auto &lTextures = lOperandData.mTextures.Get<vector_buffer_t>().mValue;

        if( lOperandData.mX.Has<multi_tensor_value_t>() && lOperandData.mY.Has<multi_tensor_value_t>() )
        {
            auto &lX = lOperandData.mX.Get<multi_tensor_value_t>().mValue;
            auto &lY = lOperandData.mY.Get<multi_tensor_value_t>().mValue;

            Sample2DOp( lValue, lX, lY, lTextures );
        }
        else if( lOperandData.mX.Has<multi_tensor_value_t>() && lOperandData.mY.Has<scalar_value_vector_t>() )
        {
            auto &lX = lOperandData.mX.Get<multi_tensor_value_t>().mValue;
            auto &lY = lOperandData.mY.Get<vector_buffer_t>().mValue;

            Sample2DOp( lValue, lX, lY, lTextures );
        }
        else if( lOperandData.mX.Has<multi_tensor_value_t>() && lOperandData.mY.Has<scalar_node_t>() )
        {
            auto &lX = lOperandData.mX.Get<multi_tensor_value_t>().mValue;
            auto &lY = lOperandData.mY.Get<scalar_node_t>().mValue;

            Sample2DOp( lValue, lX, lY, lTextures );
        }
        else if( lOperandData.mX.Has<scalar_value_vector_t>() && lOperandData.mY.Has<multi_tensor_value_t>() )
        {
            auto &lX = lOperandData.mX.Get<vector_buffer_t>().mValue;
            auto &lY = lOperandData.mY.Get<multi_tensor_value_t>().mValue;

            Sample2DOp( lValue, lX, lY, lTextures );
        }
        else if( lOperandData.mX.Has<scalar_node_t>() && lOperandData.mY.Has<multi_tensor_value_t>() )
        {
            auto &lX = lOperandData.mX.Get<scalar_node_t>().mValue;
            auto &lY = lOperandData.mY.Get<multi_tensor_value_t>().mValue;

            Sample2DOp( lValue, lX, lY, lTextures );
        }
        else
        {
            throw std::runtime_error( "Invalid arguments" );
        }
    }

    void sToFixedPointOperationController::Run()
    {
        auto &lValue       = Get<multi_tensor_value_t>().mValue;
        auto &lOperandData = Get<convert_to_fixed_point_t>();
        auto  lElementType = lOperandData.mArray.Get<type_t>().mValue;
        auto &lArray       = lOperandData.mArray.Get<multi_tensor_value_t>().mValue;
        auto &lScaling     = lOperandData.mScaling.Get<scalar_node_t>().mValue;

        ToFixedPointOp( lElementType, lValue, lOperandData.mOutputType, lArray, lScaling );
    }

    void sAffineNodeController::Run()
    {

        auto &lValue       = Get<multi_tensor_value_t>().mValue;
        auto &lOperandData = Get<affine_transform_operation_t>();
        auto  lElementType = Get<type_t>().mValue;

        auto &lX = lOperandData.mX.Get<multi_tensor_value_t>();

        if( lOperandData.mA.Has<multi_tensor_value_t>() && lOperandData.mB.Has<multi_tensor_value_t>() )
        {
            auto &lA = lOperandData.mA.Get<multi_tensor_value_t>();
            auto &lB = lOperandData.mB.Get<multi_tensor_value_t>();

            AffineTransformOp( lElementType, lValue, lA.mValue, lX.mValue, lB.mValue );
        }
        else if( lOperandData.mA.Has<multi_tensor_value_t>() && lOperandData.mB.Has<scalar_value_vector_t>() )
        {
            auto &lA = lOperandData.mA.Get<multi_tensor_value_t>();
            auto &lB = lOperandData.mB.Get<vector_buffer_t>();

            AffineTransformOp( lElementType, lValue, lA.mValue, lX.mValue, lB.mValue );
        }
        else if( lOperandData.mA.Has<multi_tensor_value_t>() && lOperandData.mB.Has<scalar_node_t>() )
        {
            auto &lA = lOperandData.mA.Get<multi_tensor_value_t>();
            auto &lB = lOperandData.mB.Get<scalar_node_t>();

            AffineTransformOp( lElementType, lValue, lA.mValue, lX.mValue, lB.mValue );
        }
        else if( lOperandData.mA.Has<scalar_value_vector_t>() && lOperandData.mB.Has<multi_tensor_value_t>() )
        {
            auto &lA = lOperandData.mA.Get<vector_buffer_t>();
            auto &lB = lOperandData.mB.Get<multi_tensor_value_t>();

            AffineTransformOp( lElementType, lValue, lA.mValue, lX.mValue, lB.mValue );
        }
        else if( lOperandData.mA.Has<scalar_value_vector_t>() && lOperandData.mB.Has<scalar_value_vector_t>() )
        {
            auto &lA = lOperandData.mA.Get<vector_buffer_t>();
            auto &lB = lOperandData.mB.Get<vector_buffer_t>();

            AffineTransformOp( lElementType, lValue, lA.mValue, lX.mValue, lB.mValue );
        }
        else if( lOperandData.mA.Has<scalar_value_vector_t>() && lOperandData.mB.Has<scalar_node_t>() )
        {
            auto &lA = lOperandData.mA.Get<vector_buffer_t>();
            auto &lB = lOperandData.mB.Get<scalar_node_t>();

            AffineTransformOp( lElementType, lValue, lA.mValue, lX.mValue, lB.mValue );
        }
        else if( lOperandData.mA.Has<scalar_node_t>() && lOperandData.mB.Has<multi_tensor_value_t>() )
        {
            auto &lA = lOperandData.mA.Get<scalar_node_t>();
            auto &lB = lOperandData.mB.Get<multi_tensor_value_t>();

            AffineTransformOp( lElementType, lValue, lA.mValue, lX.mValue, lB.mValue );
        }
        else if( lOperandData.mA.Has<scalar_node_t>() && lOperandData.mB.Has<scalar_value_vector_t>() )
        {
            auto &lA = lOperandData.mA.Get<scalar_node_t>();
            auto &lB = lOperandData.mB.Get<vector_buffer_t>();

            AffineTransformOp( lElementType, lValue, lA.mValue, lX.mValue, lB.mValue );
        }
        else if( lOperandData.mA.Has<scalar_node_t>() && lOperandData.mB.Has<scalar_node_t>() )
        {
            auto &lA = lOperandData.mA.Get<scalar_node_t>();
            auto &lB = lOperandData.mB.Get<scalar_node_t>();

            AffineTransformOp( lElementType, lValue, lA.mValue, lX.mValue, lB.mValue );
        }
        else
        {
            throw std::runtime_error( "something's wrong" );
        }
    }

    void sFloorOperationController::Run()
    {
        auto &lValue       = Get<multi_tensor_value_t>().mValue;
        auto &lOperandData = Get<floor_operation_t>();

        FloorOp( lValue, lOperandData.mArray.Get<multi_tensor_value_t>().mValue );
    }

    void sCeilOperationController::Run()
    {
        auto &lValue       = Get<multi_tensor_value_t>().mValue;
        auto &lOperandData = Get<ceiling_operation_t>();

        CeilOp( lValue, lOperandData.mArray.Get<multi_tensor_value_t>().mValue );
    }

    void sAbsOperationController::Run()
    {
        auto &lValue       = Get<multi_tensor_value_t>().mValue;
        auto &lOperandData = Get<abs_operation_t>();
        auto  lElementType = Get<type_t>().mValue;

        AbsOp( lElementType, lValue, lOperandData.mArray.Get<multi_tensor_value_t>().mValue );
    }

    void sSqrtOperationController::Run()
    {
        auto &lValue       = Get<multi_tensor_value_t>().mValue;
        auto &lOperandData = Get<sqrt_operation_t>();
        auto  lElementType = Get<type_t>().mValue;

        SqrtOp( lElementType, lValue, lOperandData.mArray.Get<multi_tensor_value_t>().mValue );
    }

    void sRoundOperationController::Run()
    {
        auto &lValue       = Get<multi_tensor_value_t>().mValue;
        auto &lOperandData = Get<round_operation_t>();
        auto  lElementType = Get<type_t>().mValue;

        RoundOp( lElementType, lValue, lOperandData.mArray.Get<multi_tensor_value_t>().mValue );
    }

    void sCountTrueOperationController::Run()
    {
        auto &lValue       = Get<multi_tensor_value_t>().mValue;
        auto &lOperandData = Get<count_true_operation_t>();
        auto  lElementType = Get<type_t>().mValue;

        CountTrueOp( lValue, lOperandData.mArray.Get<multi_tensor_value_t>().mValue,
                     lOperandData.mBlockSizes.Get<vector_buffer_t>().mValue,
                     lOperandData.mElementCount.Get<vector_buffer_t>().mValue, lOperandData.mMaxBlockSize );
    }

    void sCountNonZeroOperationController::Run()
    {
        auto &lValue       = Get<multi_tensor_value_t>().mValue;
        auto &lOperandData = Get<count_non_zero_operation_t>();
        auto  lElementType = Get<type_t>().mValue;

        CountNonZeroOp( lElementType, lValue, lOperandData.mArray.Get<multi_tensor_value_t>().mValue,
                        lOperandData.mBlockSizes.Get<vector_buffer_t>().mValue,
                        lOperandData.mElementCount.Get<vector_buffer_t>().mValue, lOperandData.mMaxBlockSize );
    }

    void sCountZeroOperationController::Run()
    {
        auto &lValue       = Get<multi_tensor_value_t>().mValue;
        auto &lOperandData = Get<count_zero_operation_t>();
        auto  lElementType = Get<type_t>().mValue;

        CountZeroOp( lElementType, lValue, lOperandData.mArray.Get<multi_tensor_value_t>().mValue,
                     lOperandData.mBlockSizes.Get<vector_buffer_t>().mValue,
                     lOperandData.mElementCount.Get<vector_buffer_t>().mValue, lOperandData.mMaxBlockSize );
    }

    void sArraySummationOperationController::Run()
    {
        auto &lValue       = Get<multi_tensor_value_t>().mValue;
        auto &lOperandData = Get<array_sum_operation_t>();
        auto  lElementType = Get<type_t>().mValue;

        ArraySummationOp( lElementType, lValue, lOperandData.mArray.Get<multi_tensor_value_t>().mValue,
                          lOperandData.mBegin.Get<vector_buffer_t>().mValue,
                          lOperandData.mEnd.Get<vector_buffer_t>().mValue,
                          lOperandData.mElementCount.Get<vector_buffer_t>().mValue,
                          lOperandData.mBlockSizes.Get<vector_buffer_t>().mValue, lOperandData.mMaxBlockSize );
    }

    void sArraySliceOperationController::Run()
    {
        auto &lValue       = Get<multi_tensor_value_t>().mValue;
        auto &lOperandData = Get<array_slice_operation_t>();
        auto  lElementType = Get<type_t>().mValue;

        ArraySliceOp( lElementType, lValue, lOperandData.mArray.Get<multi_tensor_value_t>().mValue,
                      lOperandData.mBegin.Get<vector_buffer_t>().mValue, lOperandData.mEnd.Get<vector_buffer_t>().mValue,
                      lOperandData.mElementCount.Get<vector_buffer_t>().mValue,
                      lOperandData.mBlockSizes.Get<vector_buffer_t>().mValue, lOperandData.mMaxBlockSize );
    }

    void sDiffOperationController::Run()
    {
        auto &lValue       = Get<multi_tensor_value_t>().mValue;
        auto &lOperandData = Get<diff_operation_t>();
        auto  lElementType = Get<type_t>().mValue;

        DiffOp( lElementType, lValue, lOperandData.mArray.Get<multi_tensor_value_t>().mValue, lOperandData.mCount,
                lOperandData.mElementCount.Get<vector_buffer_t>().mValue,
                lOperandData.mBlockSizes.Get<vector_buffer_t>().mValue, lOperandData.mMaxBlockSize );
    }

    void sShiftOperationController::Run()
    {
        auto &lValue       = Get<multi_tensor_value_t>().mValue;
        auto &lOperandData = Get<shift_operation_t>();
        auto  lElementType = Get<type_t>().mValue;

        ShiftOp( lElementType, lValue, lOperandData.mArray.Get<multi_tensor_value_t>().mValue, lOperandData.mCount,
                 lOperandData.mFillValue.Get<scalar_node_t>().mValue,
                 lOperandData.mElementCount.Get<vector_buffer_t>().mValue,
                 lOperandData.mBlockSizes.Get<vector_buffer_t>().mValue, lOperandData.mMaxBlockSize );
    }

    void sConv1DOperationController::Run()
    {
        auto &lValue       = Get<multi_tensor_value_t>().mValue;
        auto &lOperandData = Get<conv1d_operation_t>();
        auto  lElementType = Get<type_t>().mValue;

        Conv1DOp( lElementType, lValue, lOperandData.mArray0.Get<multi_tensor_value_t>().mValue,
                  lOperandData.mElementCount0.Get<vector_buffer_t>().mValue,
                  lOperandData.mBlockSizes0.Get<vector_buffer_t>().mValue, lOperandData.mMaxElementCount0,
                  lOperandData.mMaxBlockSize0, lOperandData.mArray1.Get<multi_tensor_value_t>().mValue,
                  lOperandData.mElementCount1.Get<vector_buffer_t>().mValue,
                  lOperandData.mBlockSizes1.Get<vector_buffer_t>().mValue, lOperandData.mMaxBlockSize1 );
    }

    void sHCatOperationController::Run()
    {
        auto &lValue       = Get<multi_tensor_value_t>().mValue;
        auto &lOperandData = Get<hcat_operation_t>();
        auto  lElementType = Get<type_t>().mValue;

        HCatOp( lElementType, lValue, lOperandData.mArray0.Get<multi_tensor_value_t>().mValue,
                lOperandData.mElementCount0.Get<vector_buffer_t>().mValue,
                lOperandData.mArray1.Get<multi_tensor_value_t>().mValue,
                lOperandData.mElementCount1.Get<vector_buffer_t>().mValue,
                lOperandData.mBlockSizes.Get<vector_buffer_t>().mValue, lOperandData.mMaxBlockSize );
    }
} // namespace SE::TensorOps
