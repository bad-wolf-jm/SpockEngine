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

        auto &left   = operand_data.left.Get<vector_buffer_t>().value;
        auto &right  = operand_data.right.Get<vector_buffer_t>().value;
        auto &lDelta = operand_data.delta.Get<vector_buffer_t>().value;

        auto element_type = type_of( _node );

        uint32_t max_subdivisions = 0;
        for( const auto &sub : value.Shape().Shape )
            max_subdivisions = std::max( max_subdivisions, sub[0] );

        ARangeOp( element_type, value, left, right, lDelta, max_subdivisions );
    }

    void sArrayOperationController::Run()
    {
        auto &value = Get<multi_tensor_value_t>().value;

        auto element_type = type_of( _node );

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
        auto  element_type = type_of( _node );

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
            auto  element_type       = type_of( operand_data.left ); // operand_data.left.type_of( _node );

            if( Has<broadcast_info_t>() )
                Op( element_type, value, left_operand_data.value, right_operant_data.value, Get<broadcast_info_t>() );
            else
                Op( element_type, value, left_operand_data.value, right_operant_data.value );
        }
        else if( operand_data.left.Has<multi_tensor_value_t>() && operand_data.right.Has<scalar_value_vector_t>() )
        {
            auto &left_operand_data  = operand_data.left.Get<multi_tensor_value_t>();
            auto &right_operant_data = operand_data.right.Get<vector_buffer_t>();
            auto  element_type       = type_of( operand_data.left ); // operand_data.left.type_of( _node );

            Op( element_type, value, left_operand_data.value, right_operant_data.value );
        }
        else if( operand_data.left.Has<scalar_value_vector_t>() && operand_data.right.Has<multi_tensor_value_t>() )
        {
            auto &left_operand_data  = operand_data.left.Get<vector_buffer_t>();
            auto &right_operant_data = operand_data.right.Get<multi_tensor_value_t>();
            auto  element_type       = type_of( operand_data.right ); // operand_data.left.type_of( _node );

            Op( element_type, value, left_operand_data.value, right_operant_data.value );
        }
        else if( operand_data.left.Has<multi_tensor_value_t>() && operand_data.right.Has<scalar_node_t>() )
        {
            auto &left_operand_data  = operand_data.left.Get<multi_tensor_value_t>();
            auto &right_operant_data = operand_data.right.Get<scalar_node_t>();
            auto  element_type       = type_of( operand_data.left ); // operand_data.left.type_of( _node );

            Op( element_type, value, left_operand_data.value, right_operant_data.value );
        }
        else if( operand_data.left.Has<scalar_node_t>() && operand_data.right.Has<multi_tensor_value_t>() )
        {
            auto &left_operand_data  = operand_data.left.Get<scalar_node_t>();
            auto &right_operant_data = operand_data.right.Get<multi_tensor_value_t>();
            auto  element_type       = type_of( operand_data.right ); // operand_data.left.type_of( _node );

            Op( element_type, value, left_operand_data.value, right_operant_data.value );
        }
        else
        {
            throw std::runtime_error( "something's wrong" );
        }
    }

    void sAddOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right )
    {
        AddOp( element_type, out, left, right );
    }

    void sAddOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right,
                                      broadcast_info_t &broadcast )
    {
        AddOp( element_type, out, left, right, broadcast.mBroadcastHint, broadcast.block_sizes.Get<vector_buffer_t>().value,
               broadcast.max_block_size, broadcast.mBroadcastDimension.Get<vector_buffer_t>().value,
               broadcast.mMaxBroadcastDimension );
    }

    void sAddOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &aIn, scalar_value_t &aConstant )
    {
        AddOp( element_type, out, aIn, aConstant );
    }

    void sAddOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, scalar_value_t &aConstant, multi_tensor_t &aIn )
    {
        AddOp( element_type, out, aIn, aConstant );
    }

    void sAddOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left, memory_buffer_t &right )
    {
        AddOp( element_type, out, left, right );
    }

    void sAddOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, memory_buffer_t &left, multi_tensor_t &right )
    {
        AddOp( element_type, out, right, left );
    }

    void sMultiplyOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left,
                                           multi_tensor_t &right )
    {
        MultiplyOp( element_type, out, left, right );
    }

    void sMultiplyOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left,
                                           multi_tensor_t &right, broadcast_info_t &broadcast )
    {
        MultiplyOp( element_type, out, left, right, broadcast.mBroadcastHint, broadcast.block_sizes.Get<vector_buffer_t>().value,
                    broadcast.max_block_size, broadcast.mBroadcastDimension.Get<vector_buffer_t>().value,
                    broadcast.mMaxBroadcastDimension );
    }

    void sMultiplyOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &aIn,
                                           scalar_value_t &aConstant )
    {
        MultiplyOp( element_type, out, aIn, aConstant );
    }

    void sMultiplyOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, scalar_value_t &aConstant,
                                           multi_tensor_t &aIn )
    {
        MultiplyOp( element_type, out, aIn, aConstant );
    }

    void sMultiplyOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left,
                                           memory_buffer_t &right )
    {
        MultiplyOp( element_type, out, left, right );
    }

    void sMultiplyOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, memory_buffer_t &left,
                                           multi_tensor_t &right )
    {
        MultiplyOp( element_type, out, right, left );
    }

    void sSubtractOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left,
                                           multi_tensor_t &right )
    {
        SubtractOp( element_type, out, left, right );
    }

    void sSubtractOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left,
                                           multi_tensor_t &right, broadcast_info_t &broadcast )
    {
        SubtractOp( element_type, out, left, right, broadcast.mBroadcastHint, broadcast.block_sizes.Get<vector_buffer_t>().value,
                    broadcast.max_block_size, broadcast.mBroadcastDimension.Get<vector_buffer_t>().value,
                    broadcast.mMaxBroadcastDimension );
    }

    void sSubtractOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &aIn,
                                           scalar_value_t &aConstant )
    {
        SubtractOp( element_type, out, aIn, aConstant );
    }

    void sSubtractOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, scalar_value_t &aConstant,
                                           multi_tensor_t &aIn )
    {
        SubtractOp( element_type, out, aConstant, aIn );
    }

    void sSubtractOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left,
                                           memory_buffer_t &right )
    {
        SubtractOp( element_type, out, left, right );
    }

    void sSubtractOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, memory_buffer_t &left,
                                           multi_tensor_t &right )
    {
        SubtractOp( element_type, out, left, right );
    }

    void sDivideOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right )
    {
        DivideOp( element_type, out, left, right );
    }

    void sDivideOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &aIn,
                                         scalar_value_t &aConstant )
    {
        DivideOp( element_type, out, aIn, aConstant );
    }

    void sDivideOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right,
                                         broadcast_info_t &broadcast )
    {
        DivideOp( element_type, out, left, right, broadcast.mBroadcastHint, broadcast.block_sizes.Get<vector_buffer_t>().value,
                  broadcast.max_block_size, broadcast.mBroadcastDimension.Get<vector_buffer_t>().value,
                  broadcast.mMaxBroadcastDimension );
    }

    void sDivideOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, scalar_value_t &aConstant,
                                         multi_tensor_t &aIn )
    {
        DivideOp( element_type, out, aConstant, aIn );
    }

    void sDivideOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left,
                                         memory_buffer_t &right )
    {
        DivideOp( element_type, out, left, right );
    }

    void sDivideOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, memory_buffer_t &left,
                                         multi_tensor_t &right )
    {
        DivideOp( element_type, out, left, right );
    }

    void sAndOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right )
    {
        AndOp( element_type, out, left, right );
    }

    void sAndOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right,
                                      broadcast_info_t &broadcast )
    {
        AndOp( element_type, out, left, right, broadcast.mBroadcastHint, broadcast.block_sizes.Get<vector_buffer_t>().value,
               broadcast.max_block_size, broadcast.mBroadcastDimension.Get<vector_buffer_t>().value,
               broadcast.mMaxBroadcastDimension );
    }

    void sAndOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left, scalar_value_t &right )
    {
        AndOp( element_type, out, left, right );
    }

    void sAndOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, scalar_value_t &left, multi_tensor_t &right )
    {
        AndOp( element_type, out, left, right );
    }

    void sAndOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left, memory_buffer_t &right )
    {
        AndOp( element_type, out, left, right );
    }

    void sAndOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, memory_buffer_t &left, multi_tensor_t &right )
    {
        AndOp( element_type, out, left, right );
    }

    void sOrOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right )
    {
        OrOp( element_type, out, left, right );
    }

    void sOrOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right,
                                     broadcast_info_t &broadcast )
    {
        OrOp( element_type, out, left, right, broadcast.mBroadcastHint, broadcast.block_sizes.Get<vector_buffer_t>().value,
              broadcast.max_block_size, broadcast.mBroadcastDimension.Get<vector_buffer_t>().value, broadcast.mMaxBroadcastDimension );
    }

    void sOrOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left, scalar_value_t &right )
    {
        OrOp( element_type, out, left, right );
    }

    void sOrOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, scalar_value_t &left, multi_tensor_t &right )
    {
        OrOp( element_type, out, left, right );
    }

    void sOrOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left, memory_buffer_t &right )
    {
        OrOp( element_type, out, left, right );
    }

    void sOrOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, memory_buffer_t &left, multi_tensor_t &right )
    {
        OrOp( element_type, out, left, right );
    }

    void sNotOperationController::Run()
    {
        auto  element_type = type_of( _node );
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<not_operation_t>();

        NotOp( element_type, value, operand_data.operand.Get<multi_tensor_value_t>().value );
    }

    void sBitwiseAndOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left,
                                             multi_tensor_t &right )
    {
        BitwiseAndOp( element_type, out, left, right );
    }

    void sBitwiseAndOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left,
                                             multi_tensor_t &right, broadcast_info_t &broadcast )
    {
        BitwiseAndOp( element_type, out, left, right, broadcast.mBroadcastHint, broadcast.block_sizes.Get<vector_buffer_t>().value,
                      broadcast.max_block_size, broadcast.mBroadcastDimension.Get<vector_buffer_t>().value,
                      broadcast.mMaxBroadcastDimension );
    }

    void sBitwiseAndOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left,
                                             scalar_value_t &right )
    {
        BitwiseAndOp( element_type, out, left, right );
    }

    void sBitwiseAndOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, scalar_value_t &left,
                                             multi_tensor_t &right )
    {
        BitwiseAndOp( element_type, out, left, right );
    }

    void sBitwiseAndOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left,
                                             memory_buffer_t &right )
    {
        BitwiseAndOp( element_type, out, left, right );
    }

    void sBitwiseAndOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, memory_buffer_t &left,
                                             multi_tensor_t &right )
    {
        BitwiseAndOp( element_type, out, left, right );
    }

    void sBitwiseOrOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left,
                                            multi_tensor_t &right )
    {
        BitwiseOrOp( element_type, out, left, right );
    }

    void sBitwiseOrOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left,
                                            multi_tensor_t &right, broadcast_info_t &broadcast )
    {
        BitwiseOrOp( element_type, out, left, right, broadcast.mBroadcastHint, broadcast.block_sizes.Get<vector_buffer_t>().value,
                     broadcast.max_block_size, broadcast.mBroadcastDimension.Get<vector_buffer_t>().value,
                     broadcast.mMaxBroadcastDimension );
    }

    void sBitwiseOrOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left,
                                            scalar_value_t &right )
    {
        BitwiseOrOp( element_type, out, left, right );
    }

    void sBitwiseOrOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, scalar_value_t &left,
                                            multi_tensor_t &right )
    {
        BitwiseOrOp( element_type, out, left, right );
    }

    void sBitwiseOrOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left,
                                            memory_buffer_t &right )
    {
        BitwiseOrOp( element_type, out, left, right );
    }

    void sBitwiseOrOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, memory_buffer_t &left,
                                            multi_tensor_t &right )
    {
        BitwiseOrOp( element_type, out, left, right );
    }

    void sBitwiseNotOperationController::Run()
    {
        auto  element_type = type_of( _node );
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<bitwise_not_operation_t>();

        BitwiseNotOp( element_type, value, operand_data.operand.Get<multi_tensor_value_t>().value );
    }

    void sInIntervalOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<in_interval_operation_t>();

        auto &X            = operand_data.x.Get<multi_tensor_value_t>().value;
        auto  element_type = type_of( operand_data.x ); //.type_of( _node );

        if( operand_data.lower.Has<multi_tensor_value_t>() && operand_data.upper.Has<multi_tensor_value_t>() )
        {
            auto &lLower = operand_data.lower.Get<multi_tensor_value_t>();
            auto &lUpper = operand_data.upper.Get<multi_tensor_value_t>();

            InIntervalOp( element_type, value, X, lLower.value, lUpper.value, operand_data.strict_lower, operand_data.strict_upper );
        }
        else if( operand_data.lower.Has<multi_tensor_value_t>() && operand_data.upper.Has<scalar_value_vector_t>() )
        {
            auto &lLower = operand_data.lower.Get<multi_tensor_value_t>();
            auto &lUpper = operand_data.upper.Get<vector_buffer_t>();

            InIntervalOp( element_type, value, X, lLower.value, lUpper.value, operand_data.strict_lower, operand_data.strict_upper );
        }
        else if( operand_data.lower.Has<multi_tensor_value_t>() && operand_data.upper.Has<scalar_node_t>() )
        {
            auto &lLower = operand_data.lower.Get<multi_tensor_value_t>();
            auto &lUpper = operand_data.upper.Get<scalar_node_t>();

            InIntervalOp( element_type, value, X, lLower.value, lUpper.value, operand_data.strict_lower, operand_data.strict_upper );
        }
        else if( operand_data.lower.Has<scalar_value_vector_t>() && operand_data.upper.Has<multi_tensor_value_t>() )
        {
            auto &lLower = operand_data.lower.Get<vector_buffer_t>();
            auto &lUpper = operand_data.upper.Get<multi_tensor_value_t>();

            InIntervalOp( element_type, value, X, lLower.value, lUpper.value, operand_data.strict_lower, operand_data.strict_upper );
        }
        else if( operand_data.lower.Has<scalar_value_vector_t>() && operand_data.upper.Has<scalar_value_vector_t>() )
        {
            auto &lLower = operand_data.lower.Get<vector_buffer_t>();
            auto &lUpper = operand_data.upper.Get<vector_buffer_t>();

            InIntervalOp( element_type, value, X, lLower.value, lUpper.value, operand_data.strict_lower, operand_data.strict_upper );
        }
        else if( operand_data.lower.Has<scalar_value_vector_t>() && operand_data.upper.Has<scalar_node_t>() )
        {
            auto &lLower = operand_data.lower.Get<vector_buffer_t>();
            auto &lUpper = operand_data.upper.Get<scalar_node_t>();

            InIntervalOp( element_type, value, X, lLower.value, lUpper.value, operand_data.strict_lower, operand_data.strict_upper );
        }
        else if( operand_data.lower.Has<scalar_node_t>() && operand_data.upper.Has<multi_tensor_value_t>() )
        {
            auto &lLower = operand_data.lower.Get<scalar_node_t>();
            auto &lUpper = operand_data.upper.Get<multi_tensor_value_t>();

            InIntervalOp( element_type, value, X, lLower.value, lUpper.value, operand_data.strict_lower, operand_data.strict_upper );
        }
        else if( operand_data.lower.Has<scalar_node_t>() && operand_data.upper.Has<scalar_value_vector_t>() )
        {
            auto &lLower = operand_data.lower.Get<scalar_node_t>();
            auto &lUpper = operand_data.upper.Get<vector_buffer_t>();

            InIntervalOp( element_type, value, X, lLower.value, lUpper.value, operand_data.strict_lower, operand_data.strict_upper );
        }
        else if( operand_data.lower.Has<scalar_node_t>() && operand_data.upper.Has<scalar_node_t>() )
        {
            auto &lLower = operand_data.lower.Get<scalar_node_t>();
            auto &lUpper = operand_data.upper.Get<scalar_node_t>();

            InIntervalOp( element_type, value, X, lLower.value, lUpper.value, operand_data.strict_lower, operand_data.strict_upper );
        }
        else
        {
            throw std::runtime_error( "something's wrong" );
        }
    }

    void sEqualOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right )
    {
        EqualOp( element_type, out, left, right );
    }

    void sEqualOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left, multi_tensor_t &right,
                                        broadcast_info_t &broadcast )
    {
        EqualOp( element_type, out, left, right, broadcast.mBroadcastHint, broadcast.block_sizes.Get<vector_buffer_t>().value,
                 broadcast.max_block_size, broadcast.mBroadcastDimension.Get<vector_buffer_t>().value,
                 broadcast.mMaxBroadcastDimension );
    }

    void sEqualOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left, scalar_value_t &right )
    {
        EqualOp( element_type, out, left, right );
    }

    void sEqualOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, scalar_value_t &left, multi_tensor_t &right )
    {
        EqualOp( element_type, out, left, right );
    }

    void sEqualOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left, memory_buffer_t &right )
    {
        EqualOp( element_type, out, left, right );
    }

    void sEqualOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, memory_buffer_t &left, multi_tensor_t &right )
    {
        EqualOp( element_type, out, left, right );
    }

    void sLessThanOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left,
                                           multi_tensor_t &right )
    {
        LessThanOp( element_type, out, left, right );
    }

    void sLessThanOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left,
                                           multi_tensor_t &right, broadcast_info_t &broadcast )
    {
        LessThanOp( element_type, out, left, right, broadcast.mBroadcastHint, broadcast.block_sizes.Get<vector_buffer_t>().value,
                    broadcast.max_block_size, broadcast.mBroadcastDimension.Get<vector_buffer_t>().value,
                    broadcast.mMaxBroadcastDimension );
    }

    void sLessThanOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left,
                                           scalar_value_t &right )
    {
        LessThanOp( element_type, out, left, right );
    }

    void sLessThanOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, scalar_value_t &left,
                                           multi_tensor_t &right )
    {
        LessThanOp( element_type, out, left, right );
    }

    void sLessThanOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left,
                                           memory_buffer_t &right )
    {
        LessThanOp( element_type, out, left, right );
    }

    void sLessThanOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, memory_buffer_t &left,
                                           multi_tensor_t &right )
    {
        LessThanOp( element_type, out, left, right );
    }

    void sLessThanOrEqualOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left,
                                                  multi_tensor_t &right )
    {
        LessThanOrEqualOp( element_type, out, left, right );
    }

    void sLessThanOrEqualOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left,
                                                  multi_tensor_t &right, broadcast_info_t &broadcast )
    {
        LessThanOrEqualOp( element_type, out, left, right, broadcast.mBroadcastHint,
                           broadcast.block_sizes.Get<vector_buffer_t>().value, broadcast.max_block_size,
                           broadcast.mBroadcastDimension.Get<vector_buffer_t>().value, broadcast.mMaxBroadcastDimension );
    }

    void sLessThanOrEqualOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left,
                                                  scalar_value_t &right )
    {
        LessThanOrEqualOp( element_type, out, left, right );
    }

    void sLessThanOrEqualOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, scalar_value_t &left,
                                                  multi_tensor_t &right )
    {
        LessThanOrEqualOp( element_type, out, left, right );
    }

    void sLessThanOrEqualOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, multi_tensor_t &left,
                                                  memory_buffer_t &right )
    {
        LessThanOrEqualOp( element_type, out, left, right );
    }

    void sLessThanOrEqualOperationController::Op( scalar_type_t element_type, multi_tensor_t &out, memory_buffer_t &left,
                                                  multi_tensor_t &right )
    {
        LessThanOrEqualOp( element_type, out, left, right );
    }

    void sWhereOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<where_operation_t>();

        auto &lCondition   = operand_data.condition.Get<multi_tensor_value_t>().value;
        auto  element_type = type_of( operand_data.value_if_true ); //.type_of( _node );

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

        auto &left         = operand_data.left.Get<multi_tensor_value_t>().value;
        auto &right        = operand_data.right.Get<multi_tensor_value_t>().value;
        auto &subdivisions = operand_data.subdivisions.Get<u32_vector_t>();

        auto element_type = type_of( _node );

        uint32_t max_subdivisions = 0;
        for( const auto &sub : subdivisions.value )
            max_subdivisions = std::max( max_subdivisions, sub );

        LinearSpaceOp( element_type, value, left, right, operand_data.subdivisions.Get<vector_buffer_t>().value, max_subdivisions );
    }

    void sMixOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<mix_operation_t>();

        auto &A  = operand_data.A.Get<multi_tensor_value_t>().value;
        auto &B  = operand_data.B.Get<multi_tensor_value_t>().value;
        auto &lT = operand_data.t.Get<multi_tensor_value_t>().value;

        auto element_type = type_of( _node );

        MixOp( element_type, value, A, B, lT );
    }

    void sMultiTensorRunner::Run()
    {
        auto &value = Get<multi_tensor_value_t>().value;
        if( Has<constant_value_initializer_t>() )
        {
            auto &initializer = Get<constant_value_initializer_t>();
            ConstantFill( core::type_of( initializer.value ), value, initializer.value );
        }
        else if( Has<vector_initializer_t>() )
        {
            auto &initializer = Get<vector_initializer_t>();
            DISPATCH_BY_TYPE( core::type_of( initializer.value[0] ), ResolveAndUpload, ( initializer ) );
            ConstantFill( core::type_of( initializer.value[0] ), value, initializer.data );
        }
        else if( Has<data_initializer_t>() )
        {
            auto &initializer = Get<data_initializer_t>();
            DISPATCH_BY_TYPE( core::type_of( initializer.value[0] ), ResolveAndUpload, ( initializer, value ) );
        }
        else if( Has<random_uniform_initializer_t>() )
        {
            auto &initializer = Get<random_uniform_initializer_t>();
            RandomUniformFill( initializer.type, value );
        }
        else if( Has<random_normal_initializer_t>() )
        {
            auto &initializer = Get<random_normal_initializer_t>();
            RandomNormalFill( initializer.type, value, initializer.mu, initializer.sigma );
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

        auto &textures = operand_data.textures.Get<vector_buffer_t>().value;

        if( operand_data.x.Has<multi_tensor_value_t>() && operand_data.y.Has<multi_tensor_value_t>() )
        {
            auto &X  = operand_data.x.Get<multi_tensor_value_t>().value;
            auto &lY = operand_data.y.Get<multi_tensor_value_t>().value;

            Sample2DOp( value, X, lY, textures );
        }
        else if( operand_data.x.Has<multi_tensor_value_t>() && operand_data.y.Has<scalar_value_vector_t>() )
        {
            auto &X  = operand_data.x.Get<multi_tensor_value_t>().value;
            auto &lY = operand_data.y.Get<vector_buffer_t>().value;

            Sample2DOp( value, X, lY, textures );
        }
        else if( operand_data.x.Has<multi_tensor_value_t>() && operand_data.y.Has<scalar_node_t>() )
        {
            auto &X  = operand_data.x.Get<multi_tensor_value_t>().value;
            auto &lY = operand_data.y.Get<scalar_node_t>().value;

            Sample2DOp( value, X, lY, textures );
        }
        else if( operand_data.x.Has<scalar_value_vector_t>() && operand_data.y.Has<multi_tensor_value_t>() )
        {
            auto &X  = operand_data.x.Get<vector_buffer_t>().value;
            auto &lY = operand_data.y.Get<multi_tensor_value_t>().value;

            Sample2DOp( value, X, lY, textures );
        }
        else if( operand_data.x.Has<scalar_node_t>() && operand_data.y.Has<multi_tensor_value_t>() )
        {
            auto &X  = operand_data.x.Get<scalar_node_t>().value;
            auto &lY = operand_data.y.Get<multi_tensor_value_t>().value;

            Sample2DOp( value, X, lY, textures );
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
        auto  element_type = type_of( operand_data.array ); //.ctype_of( _node );
        auto &array        = operand_data.array.Get<multi_tensor_value_t>().value;
        auto &lScaling     = operand_data.mScaling.Get<scalar_node_t>().value;

        ToFixedPointOp( element_type, value, operand_data.output_type, array, lScaling );
    }

    void sAffineNodeController::Run()
    {

        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<affine_transform_operation_t>();
        auto  element_type = type_of( _node );

        auto &X = operand_data.X.Get<multi_tensor_value_t>();

        if( operand_data.A.Has<multi_tensor_value_t>() && operand_data.B.Has<multi_tensor_value_t>() )
        {
            auto &A = operand_data.A.Get<multi_tensor_value_t>();
            auto &B = operand_data.B.Get<multi_tensor_value_t>();

            AffineTransformOp( element_type, value, A.value, X.value, B.value );
        }
        else if( operand_data.A.Has<multi_tensor_value_t>() && operand_data.B.Has<scalar_value_vector_t>() )
        {
            auto &A = operand_data.A.Get<multi_tensor_value_t>();
            auto &B = operand_data.B.Get<vector_buffer_t>();

            AffineTransformOp( element_type, value, A.value, X.value, B.value );
        }
        else if( operand_data.A.Has<multi_tensor_value_t>() && operand_data.B.Has<scalar_node_t>() )
        {
            auto &A = operand_data.A.Get<multi_tensor_value_t>();
            auto &B = operand_data.B.Get<scalar_node_t>();

            AffineTransformOp( element_type, value, A.value, X.value, B.value );
        }
        else if( operand_data.A.Has<scalar_value_vector_t>() && operand_data.B.Has<multi_tensor_value_t>() )
        {
            auto &A = operand_data.A.Get<vector_buffer_t>();
            auto &B = operand_data.B.Get<multi_tensor_value_t>();

            AffineTransformOp( element_type, value, A.value, X.value, B.value );
        }
        else if( operand_data.A.Has<scalar_value_vector_t>() && operand_data.B.Has<scalar_value_vector_t>() )
        {
            auto &A = operand_data.A.Get<vector_buffer_t>();
            auto &B = operand_data.B.Get<vector_buffer_t>();

            AffineTransformOp( element_type, value, A.value, X.value, B.value );
        }
        else if( operand_data.A.Has<scalar_value_vector_t>() && operand_data.B.Has<scalar_node_t>() )
        {
            auto &A = operand_data.A.Get<vector_buffer_t>();
            auto &B = operand_data.B.Get<scalar_node_t>();

            AffineTransformOp( element_type, value, A.value, X.value, B.value );
        }
        else if( operand_data.A.Has<scalar_node_t>() && operand_data.B.Has<multi_tensor_value_t>() )
        {
            auto &A = operand_data.A.Get<scalar_node_t>();
            auto &B = operand_data.B.Get<multi_tensor_value_t>();

            AffineTransformOp( element_type, value, A.value, X.value, B.value );
        }
        else if( operand_data.A.Has<scalar_node_t>() && operand_data.B.Has<scalar_value_vector_t>() )
        {
            auto &A = operand_data.A.Get<scalar_node_t>();
            auto &B = operand_data.B.Get<vector_buffer_t>();

            AffineTransformOp( element_type, value, A.value, X.value, B.value );
        }
        else if( operand_data.A.Has<scalar_node_t>() && operand_data.B.Has<scalar_node_t>() )
        {
            auto &A = operand_data.A.Get<scalar_node_t>();
            auto &B = operand_data.B.Get<scalar_node_t>();

            AffineTransformOp( element_type, value, A.value, X.value, B.value );
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
        auto  element_type = type_of( _node );

        AbsOp( element_type, value, operand_data.array.Get<multi_tensor_value_t>().value );
    }

    void sSqrtOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<sqrt_operation_t>();
        auto  element_type = type_of( _node );

        SqrtOp( element_type, value, operand_data.array.Get<multi_tensor_value_t>().value );
    }

    void sRoundOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<round_operation_t>();
        auto  element_type = type_of( _node );

        RoundOp( element_type, value, operand_data.array.Get<multi_tensor_value_t>().value );
    }

    void sCountTrueOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<count_true_operation_t>();
        auto  element_type = type_of( _node );

        CountTrueOp( value, operand_data.array.Get<multi_tensor_value_t>().value,
                     operand_data.block_sizes.Get<vector_buffer_t>().value, operand_data.element_count.Get<vector_buffer_t>().value,
                     operand_data.max_block_size );
    }

    void sCountNonZeroOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<count_non_zero_operation_t>();
        auto  element_type = type_of( _node );

        CountNonZeroOp( element_type, value, operand_data.array.Get<multi_tensor_value_t>().value,
                        operand_data.block_sizes.Get<vector_buffer_t>().value, operand_data.element_count.Get<vector_buffer_t>().value,
                        operand_data.max_block_size );
    }

    void sCountZeroOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<count_zero_operation_t>();
        auto  element_type = type_of( _node );

        CountZeroOp( element_type, value, operand_data.array.Get<multi_tensor_value_t>().value,
                     operand_data.block_sizes.Get<vector_buffer_t>().value, operand_data.element_count.Get<vector_buffer_t>().value,
                     operand_data.max_block_size );
    }

    void sArraySummationOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<array_sum_operation_t>();
        auto  element_type = type_of( _node );

        ArraySummationOp( element_type, value, operand_data.array.Get<multi_tensor_value_t>().value,
                          operand_data.begin.Get<vector_buffer_t>().value, operand_data.end.Get<vector_buffer_t>().value,
                          operand_data.element_count.Get<vector_buffer_t>().value,
                          operand_data.block_sizes.Get<vector_buffer_t>().value, operand_data.max_block_size );
    }

    void sArraySliceOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<array_slice_operation_t>();
        auto  element_type = type_of( _node );

        ArraySliceOp( element_type, value, operand_data.array.Get<multi_tensor_value_t>().value,
                      operand_data.begin.Get<vector_buffer_t>().value, operand_data.end.Get<vector_buffer_t>().value,
                      operand_data.element_count.Get<vector_buffer_t>().value, operand_data.block_sizes.Get<vector_buffer_t>().value,
                      operand_data.max_block_size );
    }

    void sDiffOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<diff_operation_t>();
        auto  element_type = type_of( _node );

        DiffOp( element_type, value, operand_data.array.Get<multi_tensor_value_t>().value, operand_data.count,
                operand_data.element_count.Get<vector_buffer_t>().value, operand_data.block_sizes.Get<vector_buffer_t>().value,
                operand_data.max_block_size );
    }

    void sShiftOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<shift_operation_t>();
        auto  element_type = type_of( _node );

        ShiftOp( element_type, value, operand_data.array.Get<multi_tensor_value_t>().value, operand_data.count,
                 operand_data.fill_value.Get<scalar_node_t>().value, operand_data.element_count.Get<vector_buffer_t>().value,
                 operand_data.block_sizes.Get<vector_buffer_t>().value, operand_data.max_block_size );
    }

    void sConv1DOperationController::Run()
    {
        auto &value        = Get<multi_tensor_value_t>().value;
        auto &operand_data = Get<conv1d_operation_t>();
        auto  element_type = type_of( _node );

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
        auto  element_type = type_of( _node );

        HCatOp( element_type, value, operand_data.array0.Get<multi_tensor_value_t>().value,
                operand_data.element_count0.Get<vector_buffer_t>().value, operand_data.array1.Get<multi_tensor_value_t>().value,
                operand_data.element_count1.Get<vector_buffer_t>().value, operand_data.block_sizes.Get<vector_buffer_t>().value,
                operand_data.max_block_size );
    }
} // namespace numlua::mtops
