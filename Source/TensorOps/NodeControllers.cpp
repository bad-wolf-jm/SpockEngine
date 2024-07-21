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

namespace numlua::mtops
{

    using namespace numlua::core;

    void sARangeOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<arange_operation_t>();

        auto &lLeft  = operand_data.left.Get<vector_buffer_t>().value;
        auto &lRight = operand_data.right.Get<vector_buffer_t>().value;
        auto &lDelta = operand_data.delta.Get<vector_buffer_t>().value;

        auto element_type = Get<type_t>().value;

        uint32_t lMaxSubdivisions = 0;
        for( const auto &sub : value.Shape().Shape )
            lMaxSubdivisions = std::max( lMaxSubdivisions, sub[0] );

        ARangeOp( element_type, value, lLeft, lRight, lDelta, lMaxSubdivisions );
    }

    void sArrayOperationController::Run()
    {
        auto &value = Get<multi_tensor_value_t>().value;

        auto element_type = Get<type_t>().value;

        if( Has<repeat_operation_t>() )
        {
            auto    &operand_data    = Get<repeat_operation_t>();
            auto    &array           = operand_data.operand.Get<multi_tensor_value_t>().value;
            auto    &repetitions     = operand_data.repetitions.Get<u32_vector_t>();
            uint32_t max_repetitions = 0;
            for( const auto &sub : repetitions.value )
                max_repetitions = std::max( max_repetitions, sub );
            RepeatOp( element_type, value, array, operand_data.repetitions.Get<vector_buffer_t>().value, max_repetitions );
            return;
        }

        if( Has<tile_operation_t>() )
        {
            auto    &operand_data    = Get<tile_operation_t>();
            auto    &array           = operand_data.operand.Get<multi_tensor_value_t>().value;
            auto    &repetitions     = operand_data.repetitions.Get<u32_vector_t>();
            uint32_t max_repetitions = 0;
            for( const auto &sub : repetitions.value )
                max_repetitions = std::max( max_repetitions, sub );
            TileOp( element_type, value, array, operand_data.repetitions.Get<vector_buffer_t>().value, max_repetitions );
            return;
        }
    }

    void sBinaryOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<binary_operation_t>();
        auto  element_type = Get<type_t>().value;

        if( operand_data.left.Has<multi_tensor_value_t>() && operand_data.right.Has<multi_tensor_value_t>() )
        {
            auto &left_operand_data  = operand_data.left.Get<multi_tensor_value_t>();
            auto &right_operant_data = operand_data.right.Get<multi_tensor_value_t>();

            if( Has<broadcast_info_t>() )
                Op( element_type, value, left_operand_data.value, right_operant_data.value, Get<broadcast_info_t>() );
            else
                Op( element_type, value, left_operand_data.value, right_operant_data.value );
        }
        else if( operand_data.left.Has<multi_tensor_value_t>() && operand_data.right.Has<scalar_value_vector_t>() )
        {
            auto &left_operand_data  = operand_data.left.Get<multi_tensor_value_t>();
            auto &right_operant_data = operand_data.right.Get<vector_buffer_t>();

            Op( element_type, value, left_operand_data.value, right_operant_data.value );
        }
        else if( operand_data.left.Has<scalar_value_vector_t>() && operand_data.right.Has<multi_tensor_value_t>() )
        {
            auto &left_operand_data  = operand_data.left.Get<vector_buffer_t>();
            auto &right_operant_data = operand_data.right.Get<multi_tensor_value_t>();

            Op( element_type, value, left_operand_data.value, right_operant_data.value );
        }
        else if( operand_data.left.Has<multi_tensor_value_t>() && operand_data.right.Has<scalar_node_t>() )
        {
            auto &left_operand_data  = operand_data.left.Get<multi_tensor_value_t>();
            auto &right_operant_data = operand_data.right.Get<scalar_node_t>();

            Op( element_type, value, left_operand_data.value, right_operant_data.value );
        }
        else if( operand_data.left.Has<scalar_node_t>() && operand_data.right.Has<multi_tensor_value_t>() )
        {
            auto &left_operand_data  = operand_data.left.Get<scalar_node_t>();
            auto &right_operant_data = operand_data.right.Get<multi_tensor_value_t>();

            Op( element_type, value, left_operand_data.value, right_operant_data.value );
        }
        else
        {
            throw std::runtime_error( "something's wrong" );
        }
    }

    void sBinaryBooleanOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<binary_operation_t>();

        if( operand_data.left.Has<multi_tensor_value_t>() && operand_data.right.Has<multi_tensor_value_t>() )
        {
            auto &left_operand_data  = operand_data.left.Get<multi_tensor_value_t>();
            auto &right_operant_data = operand_data.right.Get<multi_tensor_value_t>();
            auto  element_type       = operand_data.left.Get<type_t>().value;

            if( Has<broadcast_info_t>() )
                Op( element_type, value, left_operand_data.value, right_operant_data.value, Get<broadcast_info_t>() );
            else
                Op( element_type, value, left_operand_data.value, right_operant_data.value );
        }
        else if( operand_data.left.Has<multi_tensor_value_t>() && operand_data.right.Has<scalar_value_vector_t>() )
        {
            auto &left_operand_data  = operand_data.left.Get<multi_tensor_value_t>();
            auto &right_operant_data = operand_data.right.Get<vector_buffer_t>();
            auto  element_type       = operand_data.left.Get<type_t>().value;

            Op( element_type, value, left_operand_data.value, right_operant_data.value );
        }
        else if( operand_data.left.Has<scalar_value_vector_t>() && operand_data.right.Has<multi_tensor_value_t>() )
        {
            auto &left_operand_data  = operand_data.left.Get<vector_buffer_t>();
            auto &right_operant_data = operand_data.right.Get<multi_tensor_value_t>();
            auto  element_type       = operand_data.right.Get<type_t>().value;

            Op( element_type, value, left_operand_data.value, right_operant_data.value );
        }
        else if( operand_data.left.Has<multi_tensor_value_t>() && operand_data.right.Has<scalar_node_t>() )
        {
            auto &left_operand_data  = operand_data.left.Get<multi_tensor_value_t>();
            auto &right_operant_data = operand_data.right.Get<scalar_node_t>();
            auto  element_type       = operand_data.left.Get<type_t>().value;

            Op( element_type, value, left_operand_data.value, right_operant_data.value );
        }
        else if( operand_data.left.Has<scalar_node_t>() && operand_data.right.Has<multi_tensor_value_t>() )
        {
            auto &left_operand_data  = operand_data.left.Get<scalar_node_t>();
            auto &right_operant_data = operand_data.right.Get<multi_tensor_value_t>();
            auto  element_type       = operand_data.right.Get<type_t>().value;

            Op( element_type, value, left_operand_data.value, right_operant_data.value );
        }
        else
        {
            throw std::runtime_error( "something's wrong" );
        }
    }

    void sAddOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                      multi_tensor_t &aRight )
    {
        AddOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sAddOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                      multi_tensor_t &aRight, broadcast_info_t &aBroadcast )
    {
        AddOp( aTensorElementType, aOut, aLeft, aRight, aBroadcast.mBroadcastHint, aBroadcast.block_sizes.Get<vector_buffer_t>().value,
               aBroadcast.max_block_size, aBroadcast.mBroadcastDimension.Get<vector_buffer_t>().value,
               aBroadcast.mMaxBroadcastDimension );
    }

    void sAddOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aIn,
                                      scalar_value_t &aConstant )
    {
        AddOp( aTensorElementType, aOut, aIn, aConstant );
    }

    void sAddOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aConstant,
                                      multi_tensor_t &aIn )
    {
        AddOp( aTensorElementType, aOut, aIn, aConstant );
    }

    void sAddOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                      memory_buffer_t &aRight )
    {
        AddOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sAddOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft,
                                      multi_tensor_t &aRight )
    {
        AddOp( aTensorElementType, aOut, aRight, aLeft );
    }

    void sMultiplyOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                           multi_tensor_t &aRight )
    {
        MultiplyOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sMultiplyOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                           multi_tensor_t &aRight, broadcast_info_t &aBroadcast )
    {
        MultiplyOp( aTensorElementType, aOut, aLeft, aRight, aBroadcast.mBroadcastHint,
                    aBroadcast.block_sizes.Get<vector_buffer_t>().value, aBroadcast.max_block_size,
                    aBroadcast.mBroadcastDimension.Get<vector_buffer_t>().value, aBroadcast.mMaxBroadcastDimension );
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

    void sSubtractOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                           multi_tensor_t &aRight )
    {
        SubtractOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sSubtractOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                           multi_tensor_t &aRight, broadcast_info_t &aBroadcast )
    {
        SubtractOp( aTensorElementType, aOut, aLeft, aRight, aBroadcast.mBroadcastHint,
                    aBroadcast.block_sizes.Get<vector_buffer_t>().value, aBroadcast.max_block_size,
                    aBroadcast.mBroadcastDimension.Get<vector_buffer_t>().value, aBroadcast.mMaxBroadcastDimension );
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

    void sDivideOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                         multi_tensor_t &aRight )
    {
        DivideOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sDivideOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aIn,
                                         scalar_value_t &aConstant )
    {
        DivideOp( aTensorElementType, aOut, aIn, aConstant );
    }

    void sDivideOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                         multi_tensor_t &aRight, broadcast_info_t &aBroadcast )
    {
        DivideOp( aTensorElementType, aOut, aLeft, aRight, aBroadcast.mBroadcastHint,
                  aBroadcast.block_sizes.Get<vector_buffer_t>().value, aBroadcast.max_block_size,
                  aBroadcast.mBroadcastDimension.Get<vector_buffer_t>().value, aBroadcast.mMaxBroadcastDimension );
    }

    void sDivideOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aConstant,
                                         multi_tensor_t &aIn )
    {
        DivideOp( aTensorElementType, aOut, aConstant, aIn );
    }

    void sDivideOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                         memory_buffer_t &aRight )
    {
        DivideOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sDivideOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft,
                                         multi_tensor_t &aRight )
    {
        DivideOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sAndOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                      multi_tensor_t &aRight )
    {
        AndOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sAndOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                      multi_tensor_t &aRight, broadcast_info_t &aBroadcast )
    {
        AndOp( aTensorElementType, aOut, aLeft, aRight, aBroadcast.mBroadcastHint, aBroadcast.block_sizes.Get<vector_buffer_t>().value,
               aBroadcast.max_block_size, aBroadcast.mBroadcastDimension.Get<vector_buffer_t>().value,
               aBroadcast.mMaxBroadcastDimension );
    }

    void sAndOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                      scalar_value_t &aRight )
    {
        AndOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sAndOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aLeft,
                                      multi_tensor_t &aRight )
    {
        AndOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sAndOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                      memory_buffer_t &aRight )
    {
        AndOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sAndOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft,
                                      multi_tensor_t &aRight )
    {
        AndOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sOrOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                     multi_tensor_t &aRight )
    {
        OrOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sOrOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                     multi_tensor_t &aRight, broadcast_info_t &aBroadcast )
    {
        OrOp( aTensorElementType, aOut, aLeft, aRight, aBroadcast.mBroadcastHint, aBroadcast.block_sizes.Get<vector_buffer_t>().value,
              aBroadcast.max_block_size, aBroadcast.mBroadcastDimension.Get<vector_buffer_t>().value,
              aBroadcast.mMaxBroadcastDimension );
    }

    void sOrOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                     scalar_value_t &aRight )
    {
        OrOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sOrOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aLeft,
                                     multi_tensor_t &aRight )
    {
        OrOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sOrOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                     memory_buffer_t &aRight )
    {
        OrOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sOrOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft,
                                     multi_tensor_t &aRight )
    {
        OrOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sNotOperationController::Run()
    {
        auto  element_type = Get<type_t>().value;
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<not_operation_t>();

        NotOp( element_type, value, operand_data.operand.Get<multi_tensor_value_t>().value );
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
                      aBroadcast.block_sizes.Get<vector_buffer_t>().value, aBroadcast.max_block_size,
                      aBroadcast.mBroadcastDimension.Get<vector_buffer_t>().value, aBroadcast.mMaxBroadcastDimension );
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

    void sBitwiseOrOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                            multi_tensor_t &aRight, broadcast_info_t &aBroadcast )
    {
        BitwiseOrOp( aTensorElementType, aOut, aLeft, aRight, aBroadcast.mBroadcastHint,
                     aBroadcast.block_sizes.Get<vector_buffer_t>().value, aBroadcast.max_block_size,
                     aBroadcast.mBroadcastDimension.Get<vector_buffer_t>().value, aBroadcast.mMaxBroadcastDimension );
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
        auto  element_type = Get<type_t>().value;
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<bitwise_not_operation_t>();

        BitwiseNotOp( element_type, value, operand_data.operand.Get<multi_tensor_value_t>().value );
    }

    void sInIntervalOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<in_interval_operation_t>();

        auto &lX           = operand_data.x.Get<multi_tensor_value_t>().value;
        auto  element_type = operand_data.x.Get<type_t>().value;

        if( operand_data.lower.Has<multi_tensor_value_t>() && operand_data.upper.Has<multi_tensor_value_t>() )
        {
            auto &lLower = operand_data.lower.Get<multi_tensor_value_t>();
            auto &lUpper = operand_data.upper.Get<multi_tensor_value_t>();

            InIntervalOp( element_type, value, lX, lLower.value, lUpper.value, operand_data.strict_lower, operand_data.strict_upper );
        }
        else if( operand_data.lower.Has<multi_tensor_value_t>() && operand_data.upper.Has<scalar_value_vector_t>() )
        {
            auto &lLower = operand_data.lower.Get<multi_tensor_value_t>();
            auto &lUpper = operand_data.upper.Get<vector_buffer_t>();

            InIntervalOp( element_type, value, lX, lLower.value, lUpper.value, operand_data.strict_lower, operand_data.strict_upper );
        }
        else if( operand_data.lower.Has<multi_tensor_value_t>() && operand_data.upper.Has<scalar_node_t>() )
        {
            auto &lLower = operand_data.lower.Get<multi_tensor_value_t>();
            auto &lUpper = operand_data.upper.Get<scalar_node_t>();

            InIntervalOp( element_type, value, lX, lLower.value, lUpper.value, operand_data.strict_lower, operand_data.strict_upper );
        }
        else if( operand_data.lower.Has<scalar_value_vector_t>() && operand_data.upper.Has<multi_tensor_value_t>() )
        {
            auto &lLower = operand_data.lower.Get<vector_buffer_t>();
            auto &lUpper = operand_data.upper.Get<multi_tensor_value_t>();

            InIntervalOp( element_type, value, lX, lLower.value, lUpper.value, operand_data.strict_lower, operand_data.strict_upper );
        }
        else if( operand_data.lower.Has<scalar_value_vector_t>() && operand_data.upper.Has<scalar_value_vector_t>() )
        {
            auto &lLower = operand_data.lower.Get<vector_buffer_t>();
            auto &lUpper = operand_data.upper.Get<vector_buffer_t>();

            InIntervalOp( element_type, value, lX, lLower.value, lUpper.value, operand_data.strict_lower, operand_data.strict_upper );
        }
        else if( operand_data.lower.Has<scalar_value_vector_t>() && operand_data.upper.Has<scalar_node_t>() )
        {
            auto &lLower = operand_data.lower.Get<vector_buffer_t>();
            auto &lUpper = operand_data.upper.Get<scalar_node_t>();

            InIntervalOp( element_type, value, lX, lLower.value, lUpper.value, operand_data.strict_lower, operand_data.strict_upper );
        }
        else if( operand_data.lower.Has<scalar_node_t>() && operand_data.upper.Has<multi_tensor_value_t>() )
        {
            auto &lLower = operand_data.lower.Get<scalar_node_t>();
            auto &lUpper = operand_data.upper.Get<multi_tensor_value_t>();

            InIntervalOp( element_type, value, lX, lLower.value, lUpper.value, operand_data.strict_lower, operand_data.strict_upper );
        }
        else if( operand_data.lower.Has<scalar_node_t>() && operand_data.upper.Has<scalar_value_vector_t>() )
        {
            auto &lLower = operand_data.lower.Get<scalar_node_t>();
            auto &lUpper = operand_data.upper.Get<vector_buffer_t>();

            InIntervalOp( element_type, value, lX, lLower.value, lUpper.value, operand_data.strict_lower, operand_data.strict_upper );
        }
        else if( operand_data.lower.Has<scalar_node_t>() && operand_data.upper.Has<scalar_node_t>() )
        {
            auto &lLower = operand_data.lower.Get<scalar_node_t>();
            auto &lUpper = operand_data.upper.Get<scalar_node_t>();

            InIntervalOp( element_type, value, lX, lLower.value, lUpper.value, operand_data.strict_lower, operand_data.strict_upper );
        }
        else
        {
            throw std::runtime_error( "something's wrong" );
        }
    }

    void sEqualOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                        multi_tensor_t &aRight )
    {
        EqualOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sEqualOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                        multi_tensor_t &aRight, broadcast_info_t &aBroadcast )
    {
        EqualOp( aTensorElementType, aOut, aLeft, aRight, aBroadcast.mBroadcastHint,
                 aBroadcast.block_sizes.Get<vector_buffer_t>().value, aBroadcast.max_block_size,
                 aBroadcast.mBroadcastDimension.Get<vector_buffer_t>().value, aBroadcast.mMaxBroadcastDimension );
    }

    void sEqualOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                        scalar_value_t &aRight )
    {
        EqualOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sEqualOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aLeft,
                                        multi_tensor_t &aRight )
    {
        EqualOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sEqualOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                        memory_buffer_t &aRight )
    {
        EqualOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sEqualOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft,
                                        multi_tensor_t &aRight )
    {
        EqualOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sLessThanOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                           multi_tensor_t &aRight )
    {
        LessThanOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sLessThanOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                           multi_tensor_t &aRight, broadcast_info_t &aBroadcast )
    {
        LessThanOp( aTensorElementType, aOut, aLeft, aRight, aBroadcast.mBroadcastHint,
                    aBroadcast.block_sizes.Get<vector_buffer_t>().value, aBroadcast.max_block_size,
                    aBroadcast.mBroadcastDimension.Get<vector_buffer_t>().value, aBroadcast.mMaxBroadcastDimension );
    }

    void sLessThanOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft,
                                           scalar_value_t &aRight )
    {
        LessThanOp( aTensorElementType, aOut, aLeft, aRight );
    }

    void sLessThanOperationController::Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aLeft,
                                           multi_tensor_t &aRight )
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
                           aBroadcast.block_sizes.Get<vector_buffer_t>().value, aBroadcast.max_block_size,
                           aBroadcast.mBroadcastDimension.Get<vector_buffer_t>().value, aBroadcast.mMaxBroadcastDimension );
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
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<where_operation_t>();

        auto &lCondition   = operand_data.condition.Get<multi_tensor_value_t>().value;
        auto  element_type = operand_data.value_if_true.Get<type_t>().value;

        if( operand_data.value_if_true.Has<multi_tensor_value_t>() && operand_data.value_if_false.Has<multi_tensor_value_t>() )
        {
            auto &lValueIfTrue  = operand_data.value_if_true.Get<multi_tensor_value_t>();
            auto &lValueIfFalse = operand_data.value_if_false.Get<multi_tensor_value_t>();

            WhereOp( element_type, value, lCondition, lValueIfTrue.value, lValueIfFalse.value );
        }
        else if( operand_data.value_if_true.Has<multi_tensor_value_t>() && operand_data.value_if_false.Has<scalar_value_vector_t>() )
        {
            auto &lValueIfTrue  = operand_data.value_if_true.Get<multi_tensor_value_t>();
            auto &lValueIfFalse = operand_data.value_if_false.Get<vector_buffer_t>();

            WhereOp( element_type, value, lCondition, lValueIfTrue.value, lValueIfFalse.value );
        }
        else if( operand_data.value_if_true.Has<multi_tensor_value_t>() && operand_data.value_if_false.Has<scalar_node_t>() )
        {
            auto &lValueIfTrue  = operand_data.value_if_true.Get<multi_tensor_value_t>();
            auto &lValueIfFalse = operand_data.value_if_false.Get<scalar_node_t>();

            WhereOp( element_type, value, lCondition, lValueIfTrue.value, lValueIfFalse.value );
        }
        else if( operand_data.value_if_true.Has<scalar_value_vector_t>() && operand_data.value_if_false.Has<multi_tensor_value_t>() )
        {
            auto &lValueIfTrue  = operand_data.value_if_true.Get<vector_buffer_t>();
            auto &lValueIfFalse = operand_data.value_if_false.Get<multi_tensor_value_t>();

            WhereOp( element_type, value, lCondition, lValueIfTrue.value, lValueIfFalse.value );
        }
        else if( operand_data.value_if_true.Has<scalar_value_vector_t>() && operand_data.value_if_false.Has<scalar_value_vector_t>() )
        {
            auto &lValueIfTrue  = operand_data.value_if_true.Get<vector_buffer_t>();
            auto &lValueIfFalse = operand_data.value_if_false.Get<vector_buffer_t>();

            WhereOp( element_type, value, lCondition, lValueIfTrue.value, lValueIfFalse.value );
        }
        else if( operand_data.value_if_true.Has<scalar_value_vector_t>() && operand_data.value_if_false.Has<scalar_node_t>() )
        {
            auto &lValueIfTrue  = operand_data.value_if_true.Get<vector_buffer_t>();
            auto &lValueIfFalse = operand_data.value_if_false.Get<scalar_node_t>();

            WhereOp( element_type, value, lCondition, lValueIfTrue.value, lValueIfFalse.value );
        }
        else if( operand_data.value_if_true.Has<scalar_node_t>() && operand_data.value_if_false.Has<multi_tensor_value_t>() )
        {
            auto &lValueIfTrue  = operand_data.value_if_true.Get<scalar_node_t>();
            auto &lValueIfFalse = operand_data.value_if_false.Get<multi_tensor_value_t>();

            WhereOp( element_type, value, lCondition, lValueIfTrue.value, lValueIfFalse.value );
        }
        else if( operand_data.value_if_true.Has<scalar_node_t>() && operand_data.value_if_false.Has<scalar_value_vector_t>() )
        {
            auto &lValueIfTrue  = operand_data.value_if_true.Get<scalar_node_t>();
            auto &lValueIfFalse = operand_data.value_if_false.Get<vector_buffer_t>();

            WhereOp( element_type, value, lCondition, lValueIfTrue.value, lValueIfFalse.value );
        }
        else if( operand_data.value_if_true.Has<scalar_node_t>() && operand_data.value_if_false.Has<scalar_node_t>() )
        {
            auto &lValueIfTrue  = operand_data.value_if_true.Get<scalar_node_t>();
            auto &lValueIfFalse = operand_data.value_if_false.Get<scalar_node_t>();

            WhereOp( element_type, value, lCondition, lValueIfTrue.value, lValueIfFalse.value );
        }
        else
        {
            throw std::runtime_error( "something's wrong" );
        }
    }

    void sLinearSpaceOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<linear_space_operation_t>();

        auto &lLeft         = operand_data.left.Get<multi_tensor_value_t>().value;
        auto &lRight        = operand_data.right.Get<multi_tensor_value_t>().value;
        auto &lSubdivisions = operand_data.subdivisions.Get<u32_vector_t>();

        auto element_type = Get<type_t>().value;

        uint32_t lMaxSubdivisions = 0;
        for( const auto &sub : lSubdivisions.value )
            lMaxSubdivisions = std::max( lMaxSubdivisions, sub );

        LinearSpaceOp( element_type, value, lLeft, lRight, operand_data.subdivisions.Get<vector_buffer_t>().value, lMaxSubdivisions );
    }

    void sMixOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<mix_operation_t>();

        auto &lA = operand_data.A.Get<multi_tensor_value_t>().value;
        auto &lB = operand_data.B.Get<multi_tensor_value_t>().value;
        auto &lT = operand_data.t.Get<multi_tensor_value_t>().value;

        auto element_type = Get<type_t>().value;

        MixOp( element_type, value, lA, lB, lT );
    }

    void sMultiTensorRunner::Run()
    {
        auto &value = Get<multi_tensor_value_t>().value;
        if( Has<constant_value_initializer_t>() )
        {
            auto &lInitializer = Get<constant_value_initializer_t>();
            ConstantFill( type_of( lInitializer.value ), value, lInitializer.value );
        }
        else if( Has<vector_initializer_t>() )
        {
            auto &lInitializer = Get<vector_initializer_t>();
            DISPATCH_BY_TYPE( type_of( lInitializer.value[0] ), ResolveAndUpload, ( lInitializer ) );
            ConstantFill( type_of( lInitializer.value[0] ), value, lInitializer.data );
        }
        else if( Has<data_initializer_t>() )
        {
            auto &lInitializer = Get<data_initializer_t>();
            DISPATCH_BY_TYPE( type_of( lInitializer.value[0] ), ResolveAndUpload, ( lInitializer, value ) );
        }
        else if( Has<random_uniform_initializer_t>() )
        {
            auto &lInitializer = Get<random_uniform_initializer_t>();
            RandomUniformFill( lInitializer.type, value );
        }
        else if( Has<random_normal_initializer_t>() )
        {
            auto &lInitializer = Get<random_normal_initializer_t>();
            RandomNormalFill( lInitializer.type, value, lInitializer.mu, lInitializer.sigma );
        }
        else
        {
            throw std::runtime_error( "Invalid initialization method for multi tensor" );
        }
    }

    void sSample2DOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<sample2D_operation_t>();

        auto &lTextures = operand_data.textures.Get<vector_buffer_t>().value;

        if( operand_data.x.Has<multi_tensor_value_t>() && operand_data.y.Has<multi_tensor_value_t>() )
        {
            auto &lX = operand_data.x.Get<multi_tensor_value_t>().value;
            auto &lY = operand_data.y.Get<multi_tensor_value_t>().value;

            Sample2DOp( value, lX, lY, lTextures );
        }
        else if( operand_data.x.Has<multi_tensor_value_t>() && operand_data.y.Has<scalar_value_vector_t>() )
        {
            auto &lX = operand_data.x.Get<multi_tensor_value_t>().value;
            auto &lY = operand_data.y.Get<vector_buffer_t>().value;

            Sample2DOp( value, lX, lY, lTextures );
        }
        else if( operand_data.x.Has<multi_tensor_value_t>() && operand_data.y.Has<scalar_node_t>() )
        {
            auto &lX = operand_data.x.Get<multi_tensor_value_t>().value;
            auto &lY = operand_data.y.Get<scalar_node_t>().value;

            Sample2DOp( value, lX, lY, lTextures );
        }
        else if( operand_data.x.Has<scalar_value_vector_t>() && operand_data.y.Has<multi_tensor_value_t>() )
        {
            auto &lX = operand_data.x.Get<vector_buffer_t>().value;
            auto &lY = operand_data.y.Get<multi_tensor_value_t>().value;

            Sample2DOp( value, lX, lY, lTextures );
        }
        else if( operand_data.x.Has<scalar_node_t>() && operand_data.y.Has<multi_tensor_value_t>() )
        {
            auto &lX = operand_data.x.Get<scalar_node_t>().value;
            auto &lY = operand_data.y.Get<multi_tensor_value_t>().value;

            Sample2DOp( value, lX, lY, lTextures );
        }
        else
        {
            throw std::runtime_error( "Invalid arguments" );
        }
    }

    void sToFixedPointOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<convert_to_fixed_point_t>();
        auto  element_type = operand_data.array.Get<type_t>().value;
        auto &array        = operand_data.array.Get<multi_tensor_value_t>().value;
        auto &lScaling     = operand_data.mScaling.Get<scalar_node_t>().value;

        ToFixedPointOp( element_type, value, operand_data.mOutputType, array, lScaling );
    }

    void sAffineNodeController::Run()
    {

        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<affine_transform_operation_t>();
        auto  element_type = Get<type_t>().value;

        auto &lX = operand_data.X.Get<multi_tensor_value_t>();

        if( operand_data.A.Has<multi_tensor_value_t>() && operand_data.B.Has<multi_tensor_value_t>() )
        {
            auto &lA = operand_data.A.Get<multi_tensor_value_t>();
            auto &lB = operand_data.B.Get<multi_tensor_value_t>();

            AffineTransformOp( element_type, value, lA.value, lX.value, lB.value );
        }
        else if( operand_data.A.Has<multi_tensor_value_t>() && operand_data.B.Has<scalar_value_vector_t>() )
        {
            auto &lA = operand_data.A.Get<multi_tensor_value_t>();
            auto &lB = operand_data.B.Get<vector_buffer_t>();

            AffineTransformOp( element_type, value, lA.value, lX.value, lB.value );
        }
        else if( operand_data.A.Has<multi_tensor_value_t>() && operand_data.B.Has<scalar_node_t>() )
        {
            auto &lA = operand_data.A.Get<multi_tensor_value_t>();
            auto &lB = operand_data.B.Get<scalar_node_t>();

            AffineTransformOp( element_type, value, lA.value, lX.value, lB.value );
        }
        else if( operand_data.A.Has<scalar_value_vector_t>() && operand_data.B.Has<multi_tensor_value_t>() )
        {
            auto &lA = operand_data.A.Get<vector_buffer_t>();
            auto &lB = operand_data.B.Get<multi_tensor_value_t>();

            AffineTransformOp( element_type, value, lA.value, lX.value, lB.value );
        }
        else if( operand_data.A.Has<scalar_value_vector_t>() && operand_data.B.Has<scalar_value_vector_t>() )
        {
            auto &lA = operand_data.A.Get<vector_buffer_t>();
            auto &lB = operand_data.B.Get<vector_buffer_t>();

            AffineTransformOp( element_type, value, lA.value, lX.value, lB.value );
        }
        else if( operand_data.A.Has<scalar_value_vector_t>() && operand_data.B.Has<scalar_node_t>() )
        {
            auto &lA = operand_data.A.Get<vector_buffer_t>();
            auto &lB = operand_data.B.Get<scalar_node_t>();

            AffineTransformOp( element_type, value, lA.value, lX.value, lB.value );
        }
        else if( operand_data.A.Has<scalar_node_t>() && operand_data.B.Has<multi_tensor_value_t>() )
        {
            auto &lA = operand_data.A.Get<scalar_node_t>();
            auto &lB = operand_data.B.Get<multi_tensor_value_t>();

            AffineTransformOp( element_type, value, lA.value, lX.value, lB.value );
        }
        else if( operand_data.A.Has<scalar_node_t>() && operand_data.B.Has<scalar_value_vector_t>() )
        {
            auto &lA = operand_data.A.Get<scalar_node_t>();
            auto &lB = operand_data.B.Get<vector_buffer_t>();

            AffineTransformOp( element_type, value, lA.value, lX.value, lB.value );
        }
        else if( operand_data.A.Has<scalar_node_t>() && operand_data.B.Has<scalar_node_t>() )
        {
            auto &lA = operand_data.A.Get<scalar_node_t>();
            auto &lB = operand_data.B.Get<scalar_node_t>();

            AffineTransformOp( element_type, value, lA.value, lX.value, lB.value );
        }
        else
        {
            throw std::runtime_error( "something's wrong" );
        }
    }

    void sFloorOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<floor_operation_t>();

        FloorOp( value, operand_data.array.Get<multi_tensor_value_t>().value );
    }

    void sCeilOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<ceiling_operation_t>();

        CeilOp( value, operand_data.array.Get<multi_tensor_value_t>().value );
    }

    void sAbsOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<abs_operation_t>();
        auto  element_type = Get<type_t>().value;

        AbsOp( element_type, value, operand_data.array.Get<multi_tensor_value_t>().value );
    }

    void sSqrtOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<sqrt_operation_t>();
        auto  element_type = Get<type_t>().value;

        SqrtOp( element_type, value, operand_data.array.Get<multi_tensor_value_t>().value );
    }

    void sRoundOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<round_operation_t>();
        auto  element_type = Get<type_t>().value;

        RoundOp( element_type, value, operand_data.array.Get<multi_tensor_value_t>().value );
    }

    void sCountTrueOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<count_true_operation_t>();
        auto  element_type = Get<type_t>().value;

        CountTrueOp( value, operand_data.array.Get<multi_tensor_value_t>().value,
                     operand_data.block_sizes.Get<vector_buffer_t>().value, operand_data.element_count.Get<vector_buffer_t>().value,
                     operand_data.max_block_size );
    }

    void sCountNonZeroOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<count_non_zero_operation_t>();
        auto  element_type = Get<type_t>().value;

        CountNonZeroOp( element_type, value, operand_data.array.Get<multi_tensor_value_t>().value,
                        operand_data.block_sizes.Get<vector_buffer_t>().value, operand_data.element_count.Get<vector_buffer_t>().value,
                        operand_data.max_block_size );
    }

    void sCountZeroOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<count_zero_operation_t>();
        auto  element_type = Get<type_t>().value;

        CountZeroOp( element_type, value, operand_data.array.Get<multi_tensor_value_t>().value,
                     operand_data.block_sizes.Get<vector_buffer_t>().value, operand_data.element_count.Get<vector_buffer_t>().value,
                     operand_data.max_block_size );
    }

    void sArraySummationOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<array_sum_operation_t>();
        auto  element_type = Get<type_t>().value;

        ArraySummationOp( element_type, value, operand_data.array.Get<multi_tensor_value_t>().value,
                          operand_data.begin.Get<vector_buffer_t>().value, operand_data.end.Get<vector_buffer_t>().value,
                          operand_data.element_count.Get<vector_buffer_t>().value,
                          operand_data.block_sizes.Get<vector_buffer_t>().value, operand_data.max_block_size );
    }

    void sArraySliceOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<array_slice_operation_t>();
        auto  element_type = Get<type_t>().value;

        ArraySliceOp( element_type, value, operand_data.array.Get<multi_tensor_value_t>().value,
                      operand_data.begin.Get<vector_buffer_t>().value, operand_data.end.Get<vector_buffer_t>().value,
                      operand_data.element_count.Get<vector_buffer_t>().value, operand_data.block_sizes.Get<vector_buffer_t>().value,
                      operand_data.max_block_size );
    }

    void sDiffOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<diff_operation_t>();
        auto  element_type = Get<type_t>().value;

        DiffOp( element_type, value, operand_data.array.Get<multi_tensor_value_t>().value, operand_data.count,
                operand_data.element_count.Get<vector_buffer_t>().value, operand_data.block_sizes.Get<vector_buffer_t>().value,
                operand_data.max_block_size );
    }

    void sShiftOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<shift_operation_t>();
        auto  element_type = Get<type_t>().value;

        ShiftOp( element_type, value, operand_data.array.Get<multi_tensor_value_t>().value, operand_data.count,
                 operand_data.fill_value.Get<scalar_node_t>().value, operand_data.element_count.Get<vector_buffer_t>().value,
                 operand_data.block_sizes.Get<vector_buffer_t>().value, operand_data.max_block_size );
    }

    void sConv1DOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<conv1d_operation_t>();
        auto  element_type = Get<type_t>().value;

        Conv1DOp( element_type, value, operand_data.array0.Get<multi_tensor_value_t>().value,
                  operand_data.element_count0.Get<vector_buffer_t>().value, operand_data.block_sizes0.Get<vector_buffer_t>().value,
                  operand_data.max_element_count0, operand_data.max_block_size0, operand_data.array1.Get<multi_tensor_value_t>().value,
                  operand_data.element_count1.Get<vector_buffer_t>().value, operand_data.block_sizes1.Get<vector_buffer_t>().value,
                  operand_data.max_block_size1 );
    }

    void sHCatOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<hcat_operation_t>();
        auto  element_type = Get<type_t>().value;

        HCatOp( element_type, value, operand_data.array0.Get<multi_tensor_value_t>().value,
                operand_data.element_count0.Get<vector_buffer_t>().value, operand_data.array1.Get<multi_tensor_value_t>().value,
                operand_data.element_count1.Get<vector_buffer_t>().value, operand_data.block_sizes.Get<vector_buffer_t>().value,
                operand_data.max_block_size );
    }
} // namespace numlua::mtops
