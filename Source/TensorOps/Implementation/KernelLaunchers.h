/// @file   KernelLaunchers.h
///
/// @brief  C++ API for Cuda computation launchers
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert. All rights reserved.

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>

#include "Core/CUDA/Array/MemoryPool.h"
#include "Core/CUDA/Array/MultiTensor.h"
#include "Core/Math/Types.h"

#include "../ScalarTypes.h"

namespace SE::TensorOps
{

    using namespace SE::Cuda;

    struct TextureData
    {
        cudaTextureObject_t Texture = 0;
        math::vec4          SubArea = { 0.0f, 0.0f, 1.0, 1.0 };
        math::vec2          Scaling = { 1.0f, 1.0f };

        TextureData()                      = default;
        TextureData( const TextureData & ) = default;
    };

    /// @brief Fill @ref multi_tensor_t with a constant value
    ///
    /// The entire tensor is filled with the value specofoed by `aConstant`
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aArray Generalized tensor to process
    /// @param aConstant Value to fill the buffer with
    ///
    void ConstantFill( scalar_type_t aTensorElementType, multi_tensor_t &aArray, scalar_value_t &aInitialValue );

    /// @brief Fill @ref multi_tensor_t with a constant value
    ///
    /// The entire tensor is filled with the values specofoed in `aConstants`. The vector represented by
    /// `aConstants` should have as many elements as `aArray` has layers.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aArray Generalized tensor to process
    /// @param aConstants Vector of values to fill the buffer with
    ///
    void ConstantFill( scalar_type_t aTensorElementType, multi_tensor_t &aArray, memory_buffer_t &aInitialValues );
    void RandomUniformFill( scalar_type_t aTensorElementType, multi_tensor_t &aArray );
    void RandomNormalFill( scalar_type_t aTensorElementType, multi_tensor_t &aArray, scalar_value_t &a_Mu, scalar_value_t &a_Sigma );

    /// @brief Add a single scalar to all elements of a tensor
    ///
    /// All multi-tensors should have the same shape and the same element types `_Ty`
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aArray Left operand
    /// @param aConstant Element to add to `aArray`
    ///
    void AddOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, scalar_value_t &aRight );

    /// @brief Pointwise addition of two @ref multi_tensor_ts
    ///
    /// All multi-tensors should have the same shape and the same element types
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aLeft Left operand
    /// @param aRight Right operand
    ///
    void AddOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight );
    void AddOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                eBroadcastHint aBroadcastHint, memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize, memory_buffer_t &aBroadcastSizes,
                uint32_t aMaxBroadcastSizes );

    /// @brief Add a list of scalars to all elements of a tensor
    ///
    /// All multi-tensors should have the same shape and the same element types `_Ty`. The length of the constant
    /// vector should match the number of layers of the tensor in `aArray`.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aArray Left operand
    /// @param aConstants Elements to add to `aArray`
    ///
    void AddOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight );

    /// @brief Pointwise multiplication of two @ref multi_tensor_ts
    ///
    /// All multi-tensors should have the same shape and the same element types
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aLeft Left operand
    /// @param aRight Right operand
    ///
    void MultiplyOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight );
    void MultiplyOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                     eBroadcastHint aBroadcastHint, memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize, memory_buffer_t &aBroadcastSizes,
                     uint32_t aMaxBroadcastSizes );

    /// @brief Multiply the elements of a tensor by a vector of values
    ///
    /// All multi-tensors should have the same shape and the same element types `_Ty`. The length of the constant
    /// vector should match the number of layers of the tensor in `aArray`.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aArray Left operand
    /// @param aConstants Elements to add to `aArray`
    /// @param aOut Output tensor.
    ///
    void MultiplyOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight );

    /// @brief Multiply the elements of an array by a single scalar
    ///
    /// All multi-tensors should have the same shape and the same element types `_Ty`
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aArray Left operand
    /// @param aConstant Element to add to `aArray`
    ///
    void MultiplyOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, scalar_value_t &aRight );

    /// @brief Subtract the elements of an array from a scalar
    ///
    /// All multi-tensors should have the same shape and the same element types `_Ty`.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aArray Left operand
    /// @param aConstant Element to add to `aArray`
    ///
    void SubtractOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aLeft, multi_tensor_t &aRight );

    /// @brief Subtract a scalar from every element of an array
    ///
    /// All multi-tensors should have the same shape and the same element types `_Ty`.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aArray Left operand
    /// @param aConstant Element to add to `aArray`
    ///
    void SubtractOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, scalar_value_t &aRight );

    /// @brief Pointwise subtraction of two arrays
    ///
    /// All multi-tensors should have the same shape and the same element types `_Ty`.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aArray Left operand
    /// @param aConstant Element to add to `aArray`
    /// @param aOut Output tensor.
    ///
    void SubtractOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight );
    void SubtractOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                     eBroadcastHint aBroadcastHint, memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize, memory_buffer_t &aBroadcastSizes,
                     uint32_t aMaxBroadcastSizes );

    /// @brief Subtract elements of a multi-tensor from the elements of a vector
    ///
    /// All multi-tensors should have the same shape and the same element types `_Ty`. The length of the constant
    /// vector should match the number of layers of the tensor in `aArray`.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aArray Left operand
    /// @param aConstants Elements to add to `aArray`
    ///
    void SubtractOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight );

    /// @brief Subtract the elements of a vector of values from the elements of a multi-tensor
    ///
    /// All multi-tensors should have the same shape and the same element types `_Ty`. The length of the constant
    /// vector should match the number of layers of the tensor in `aArray`.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aArray Left operand
    /// @param aConstants Elements to add to `aArray`
    ///
    void SubtractOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight );

    /// @brief Divide the elements of an array by a scalar
    ///
    /// All multi-tensors should have the same shape and the same element types `_Ty`.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aArray Left operand
    /// @param aConstant Element to add to `aArray`
    ///
    void DivideOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, scalar_value_t &aRight );

    /// @brief Divide a scalar by every element of an array
    ///
    /// All multi-tensors should have the same shape and the same element types `_Ty`.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aArray Left operand
    /// @param aConstant Element to add to `aArray`
    ///
    void DivideOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aLeft, multi_tensor_t &aRight );

    /// @brief Pointwise division of twu multitensors
    ///
    /// All multi-tensors should have the same shape and the same element types `_Ty`.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aArray Left operand
    /// @param aConstant Element to add to `aArray`
    ///
    void DivideOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight );
    void DivideOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                   eBroadcastHint aBroadcastHint, memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize, memory_buffer_t &aBroadcastSizes,
                   uint32_t aMaxBroadcastSizes );

    /// @brief Divide the elements of a multi-tensor by the elements of a vector
    ///
    /// All multi-tensors should have the same shape and the same element types `_Ty`. The length of the constant
    /// vector should match the number of layers of the tensor in `aArray`.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aArray Left operand
    /// @param aConstants Elements to add to `aArray`
    ///
    void DivideOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight );

    /// @brief Divide the elements of a vector of values from the elements of a multi-tensor
    ///
    /// All multi-tensors should have the same shape and the same element types `_Ty`. The length of the constant
    /// vector should match the number of layers of the tensor in `aArray`.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aArray Left operand
    /// @param aConstants Elements to add to `aArray`
    ///
    void DivideOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight );

    /// @brief Conjunction operators
    ///
    /// All multi-tensors should have the same shape and the same element types`. If one of the arguments is a
    /// MemoryBuffer, then it should hold a vector of values whose length is equal to the number of layers of
    /// the other argument (a multi_tensor_t)
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aLeft Left operand
    /// @param aRight Right operand
    ///
    void AndOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, scalar_value_t &aRight );
    void AndOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aLeft, multi_tensor_t &aRight );
    void AndOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight );
    void AndOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                eBroadcastHint aBroadcastHint, memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize, memory_buffer_t &aBroadcastSizes,
                uint32_t aMaxBroadcastSizes );
    void AndOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight );
    void AndOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight );

    /// @brief Disjunction operators
    ///
    /// All multi-tensors should have the same shape and the same element types`. If one of the arguments is a
    /// MemoryBuffer, then it should hold a vector of values whose length is equal to the number of layers of
    /// the other argument (a multi_tensor_t)
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aLeft Left operand
    /// @param aRight Right operand
    ///
    void OrOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, scalar_value_t &aRight );
    void OrOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aLeft, multi_tensor_t &aRight );
    void OrOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight );
    void OrOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
               eBroadcastHint aBroadcastHint, memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize, memory_buffer_t &aBroadcastSizes,
               uint32_t aMaxBroadcastSizes );
    void OrOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight );
    void OrOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight );

    /// @brief Negation operator
    ///
    /// All multi-tensors should have the same shape and the same element types`.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aOperand Operand
    ///
    void NotOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aOperand );

    /// @brief Bitwise conjunction operators
    ///
    /// All multi-tensors should have the same shape and the same element types`. If one of the arguments is a
    /// MemoryBuffer, then it should hold a vector of values whose length is equal to the number of layers of
    /// the other argument (a multi_tensor_t)
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aLeft Left operand
    /// @param aRight Right operand
    ///
    void BitwiseAndOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, scalar_value_t &aRight );
    void BitwiseAndOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aLeft, multi_tensor_t &aRight );
    void BitwiseAndOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight );
    void BitwiseAndOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                       eBroadcastHint aBroadcastHint, memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize, memory_buffer_t &aBroadcastSizes,
                       uint32_t aMaxBroadcastSizes );
    void BitwiseAndOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight );
    void BitwiseAndOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight );

    /// @brief Bitwise disjunction operators
    ///
    /// All multi-tensors should have the same shape and the same element types`. If one of the arguments is a
    /// MemoryBuffer, then it should hold a vector of values whose length is equal to the number of layers of
    /// the other argument (a multi_tensor_t)
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aLeft Left operand
    /// @param aRight Right operand
    ///
    void BitwiseOrOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, scalar_value_t &aRight );
    void BitwiseOrOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aLeft, multi_tensor_t &aRight );
    void BitwiseOrOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight );
    void BitwiseOrOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                      eBroadcastHint aBroadcastHint, memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize, memory_buffer_t &aBroadcastSizes,
                      uint32_t aMaxBroadcastSizes );
    void BitwiseOrOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight );
    void BitwiseOrOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight );

    /// @brief Bitwise negation operators
    ///
    /// All multi-tensors should have the same shape and the same element types`.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aOperand Operand
    ///
    void BitwiseNotOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aOperand );

    /// @brief Check whether a value lies in an interval
    ///
    /// All multi-tensors should have the same shape and the same element types`. If one of the arguments is a
    /// MemoryBuffer, then it should hold a vector of values whose length is equal to the number of layers of
    /// the other argument (a multi_tensor_t)
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aX multi_tensor_t holding the value to test
    /// @param aLower Lower bound
    /// @param aUpper Upper bound
    /// @param aStrictLower Should the comparison with the lower bound be strict?
    /// @param aStrictUpper Should the comparison with the upper bound be strict?
    ///
    void InIntervalOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aX, multi_tensor_t &aLower, multi_tensor_t &aUpper,
                       bool aStrictLower, bool aStrictUpper );
    void InIntervalOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aX, multi_tensor_t &aLower, memory_buffer_t &aUpper,
                       bool aStrictLower, bool aStrictUpper );
    void InIntervalOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aX, multi_tensor_t &aLower, scalar_value_t &aUpper,
                       bool aStrictLower, bool aStrictUpper );
    void InIntervalOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aX, memory_buffer_t &aLower, multi_tensor_t &aUpper,
                       bool aStrictLower, bool aStrictUpper );
    void InIntervalOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aX, memory_buffer_t &aLower, memory_buffer_t &aUpper,
                       bool aStrictLower, bool aStrictUpper );
    void InIntervalOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aX, memory_buffer_t &aLower, scalar_value_t &aUpper,
                       bool aStrictLower, bool aStrictUpper );
    void InIntervalOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aX, scalar_value_t &aLower, multi_tensor_t &aUpper,
                       bool aStrictLower, bool aStrictUpper );
    void InIntervalOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aX, scalar_value_t &aLower, memory_buffer_t &aUpper,
                       bool aStrictLower, bool aStrictUpper );
    void InIntervalOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aX, scalar_value_t &aLower, scalar_value_t &aUpper,
                       bool aStrictLower, bool aStrictUpper );

    /// @brief Test equaliy between tensors
    ///
    /// All multi-tensors should have the same shape and the same element types`.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aLeft Left operand
    /// @param aRight Right operand
    ///
    void EqualOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, scalar_value_t &aRight );
    void EqualOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aLeft, multi_tensor_t &aRight );
    void EqualOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight );
    void EqualOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                  eBroadcastHint aBroadcastHint, memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize, memory_buffer_t &aBroadcastSizes,
                  uint32_t aMaxBroadcastSizes );
    void EqualOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight );
    void EqualOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight );

    /// @brief Test whether a tensor is pointwise less than another tensor
    ///
    /// All multi-tensors should have the same shape and the same element types`.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aLeft Left operand
    /// @param aRight Right operand
    ///
    void LessThanOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, scalar_value_t &aRight );
    void LessThanOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aLeft, multi_tensor_t &aRight );
    void LessThanOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight );
    void LessThanOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                     eBroadcastHint aBroadcastHint, memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize, memory_buffer_t &aBroadcastSizes,
                     uint32_t aMaxBroadcastSizes );
    void LessThanOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight );
    void LessThanOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight );

    /// @brief Test whether a tensor is pointwise less than or equal to another tensor
    ///
    /// All multi-tensors should have the same shape and the same element types`.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aLeft Left operand
    /// @param aRight Right operand
    ///
    void LessThanOrEqualOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, scalar_value_t &aRight );
    void LessThanOrEqualOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aLeft, multi_tensor_t &aRight );
    void LessThanOrEqualOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight );
    void LessThanOrEqualOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                            eBroadcastHint aBroadcastHint, memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize,
                            memory_buffer_t &aBroadcastSizes, uint32_t aMaxBroadcastSizes );
    void LessThanOrEqualOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight );
    void LessThanOrEqualOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight );

    /// @brief Choose values from one tensor or another based on a given boolean condition
    ///
    /// All multi-tensors should have the same shape and the same element types`.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aCondition Condition to test
    /// @param aValueIfTrue Value to use if condition is true
    /// @param aValueIfFalse Value to use if condition is false
    ///
    void WhereOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aCondition, multi_tensor_t &aValueIfTrue,
                  multi_tensor_t &aValueIfFalse );
    void WhereOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aCondition, multi_tensor_t &aValueIfTrue,
                  memory_buffer_t &aValueIfFalse );
    void WhereOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aCondition, multi_tensor_t &aValueIfTrue,
                  scalar_value_t &aValueIfFalse );
    void WhereOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aCondition, memory_buffer_t &aValueIfTrue,
                  multi_tensor_t &aValueIfFalse );
    void WhereOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aCondition, memory_buffer_t &aValueIfTrue,
                  memory_buffer_t &aValueIfFalse );
    void WhereOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aCondition, memory_buffer_t &aValueIfTrue,
                  scalar_value_t &aValueIfFalse );
    void WhereOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aCondition, scalar_value_t &aValueIfTrue,
                  multi_tensor_t &aValueIfFalse );
    void WhereOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aCondition, scalar_value_t &aValueIfTrue,
                  memory_buffer_t &aValueIfFalse );
    void WhereOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aCondition, scalar_value_t &aValueIfTrue,
                  scalar_value_t &aValueIfFalse );

    /// @brief Computes a set of ranges of values with a regular step.
    ///
    /// The two vectors contained in `aLeft` and `aRight` should have the same length and contain floating
    /// point values. This is roughly a generazed version of numpy's `np.arange`. The resulting output tensor
    /// will have one layer for every element of `aLeft`
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aLeft Lower bounds
    /// @param aRight Upper bounds
    /// @param aDelta Range step
    ///
    void ARangeOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, memory_buffer_t &aRight, memory_buffer_t &aDelta,
                   uint32_t aMaxSubdivisions );

    /// @brief Repeat each element of a multi-tensor.
    ///
    /// Roughly equivalent to numpy's npo.repeat. Each element of the innermost dimension of a multitensor is repeated a given
    /// number of times. Node that a different number of repetitions can be specified for each layer of the multi-tensor. As
    /// far as dimension and rank are concerned, if the input multi-tensor has rank @f$ N @f$ , then, the repeated multi-tensor
    /// will have rank @f$ N+1 @f$. The last dimension of the output multi-tensor is the number of repetitions.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aArray Array to repeat
    /// @param aRepetitions Nummber of repetitions
    ///
    void RepeatOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &Array, memory_buffer_t &aRepetitions,
                   uint32_t aMaxRepetitions );

    /// @brief Repeat each layer of a multi-tensor.
    ///
    /// Roughly equivalent to numpy's npo.tile. Each layer of the a multitensor is repeated a given number of times. Node that
    /// a different number of repetitions can be specified for each layer of the multi-tensor. As far as dimension and rank are
    /// concerned, if the input multi-tensor has rank @f$ N @f$ , then, the repeated multi-tensor will have rank @f$ N+1 @f$.
    /// The first dimension of the output multi-tensor is the number of repetitions.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aArray Array to repeat
    /// @param aRepetitions Nummber of repetitions
    ///
    void TileOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &Array, memory_buffer_t &aRepetitions,
                 uint32_t aMaxRepetitions );

    /// @brief Computes evenly spaced numbers in the intervals specified by two tensors
    ///
    /// Roughly equivalent to numpy's np.linspace. The two input tensors should have the same shape. The interval betrween them
    /// is subdivided into `aRepetitions` many subintervals, where each element in `aRepetitions` is matched with the corresponding
    /// layer of the input tensors.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aArray Array to repeat
    /// @param aRepetitions Nummber of repetitions
    ///
    void LinearSpaceOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                        memory_buffer_t &Subdivisions, uint32_t aMaxSubdivisions );

    /// @brief Computes the pointwise mix of two tensors
    ///
    /// This function computes the tensor @f$ (1-t)\cdot A + t\cdot B @f$
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aArray Array to repeat
    /// @param aRepetitions Nummber of repetitions
    /// @param aOut Output tensor.
    ///
    void MixOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aA, multi_tensor_t &aB, multi_tensor_t &t );

    /// @brief Testure sampling
    ///
    /// Samples the textures in `aTextures` at the coordinates specified by `aX` and `aY`. The tensors `aX`, `aY` and `aOut`
    /// should all have the same shape.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aArray Array to repeat
    /// @param aRepetitions Nummber of repetitions
    ///
    void Sample2DOp( multi_tensor_t &aOut, multi_tensor_t &aX, multi_tensor_t &aY, memory_buffer_t &aTextures );
    void Sample2DOp( multi_tensor_t &aOut, multi_tensor_t &aX, memory_buffer_t &aY, memory_buffer_t &aTextures );
    void Sample2DOp( multi_tensor_t &aOut, multi_tensor_t &aX, scalar_value_t &aY, memory_buffer_t &aTextures );
    void Sample2DOp( multi_tensor_t &aOut, memory_buffer_t &aX, multi_tensor_t &aY, memory_buffer_t &aTextures );
    void Sample2DOp( multi_tensor_t &aOut, scalar_value_t &aX, multi_tensor_t &aY, memory_buffer_t &aTextures );

    /// @brief Fixed point conversion
    ///
    /// Converts a tensor of floating point numbers into a tensor of integers by multiplying each element by a scaling factor.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aArray Array to repeat
    /// @param aRepetitions Nummber of repetitions
    ///
    void ToFixedPointOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_type_t a_OutputElementType, multi_tensor_t &Array,
                         scalar_value_t &Scaling );

    /// @brief Affine transformation
    ///
    /// Given multitensors @f$ A @f$, @f$ X @f$ and @f$ B @f$, computes the multi_tensor_t @f$ A\cdot X + B @f$ using pointwise operations
    ///
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aA Coefficient tensor
    /// @param aX Tensor to transform
    /// @param aB Shift tensor
    ///
    void AffineTransformOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aA, multi_tensor_t &aX, multi_tensor_t &aB );

    /// @brief Affine transformation
    ///
    /// Given multitensors @f$ A @f$, @f$ X @f$ and @f$ B @f$, computes the multi_tensor_t @f$ A\cdot X + B @f$ using pointwise operations
    /// for the product, and considering each element of @f$ B @f$ to be a constant to be added to the individual layers of @f$ A\cdot
    /// X @f$
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aA Coefficient tensor
    /// @param aX Tensor to transform
    /// @param aB Shift vector
    ///
    void AffineTransformOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aA, multi_tensor_t &aX, memory_buffer_t &aB );

    /// @brief Affine transformation
    ///
    /// Given multitensors @f$ A @f$, @f$ X @f$ and @f$ B @f$, computes the multi_tensor_t @f$ A\cdot X + B @f$ using pointwise operations
    /// for the product, and considering @f$ B @f$ to be a constant to be added to the individual elements of @f$ A\cdot X @f$
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aA Coefficient tensor
    /// @param aX Tensor to transform
    /// @param aB Shift constant
    ///
    void AffineTransformOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aA, multi_tensor_t &aX, scalar_value_t &aB );

    /// @brief Affine transformation
    ///
    /// Given multitensors @f$ A @f$, @f$ X @f$ and @f$ B @f$, computes the multi_tensor_t @f$ A\cdot X + B @f$. The values contained in
    /// @f$ A @f$ are constant coefficients to be applied to each layer in @f$ X @f$. The final sum is calculated pointwise.
    ///
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aA Coefficient vector
    /// @param aX Tensor to transform
    /// @param aB Shift tensor
    ///
    void AffineTransformOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aA, multi_tensor_t &aX, multi_tensor_t &aB );

    /// @brief Affine transformation
    ///
    /// Given multitensors @f$ A @f$, @f$ X @f$ and @f$ B @f$, computes the multi_tensor_t @f$ A\cdot X + B @f$. The values contained in
    /// @f$ A @f$ and @f$ B @f$ are constant coefficients and shift values to be applied to each layer in @f$ X @f$.
    ///
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aA Coefficient vector
    /// @param aX Tensor to transform
    /// @param aB Shift vector
    ///
    void AffineTransformOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aA, multi_tensor_t &aX, memory_buffer_t &aB );

    /// @brief Affine transformation
    ///
    /// Given multitensors @f$ A @f$, @f$ X @f$ and @f$ B @f$, computes the multi_tensor_t @f$ A\cdot X + B @f$. The values contained in
    /// @f$ A @f$ are constant coefficients and shift values to be applied to each layer in @f$ X @f$. The constant value contrained
    /// in @f$ B @f$ is then added to every element.
    ///
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aA Coefficient vector
    /// @param aX Tensor to transform
    /// @param aB Shift constant
    ///
    void AffineTransformOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aA, multi_tensor_t &aX, scalar_value_t &aB );

    /// @brief Affine transformation
    ///
    /// Given multitensors @f$ A @f$, @f$ X @f$ and @f$ B @f$, computes the multi_tensor_t @f$ A\cdot X + B @f$. The value contained in
    /// @f$ A @f$ is a constant coefficient to be applied to each layer in @f$ X @f$. The values in @f$ B @f$ is then added pointwise
    /// to every element.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aA Coefficient constant
    /// @param aX Tensor to transform
    /// @param aB Shift constant
    ///
    void AffineTransformOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aA, multi_tensor_t &aX, multi_tensor_t &aB );

    /// @brief Affine transformation
    ///
    /// Given multitensors @f$ A @f$, @f$ X @f$ and @f$ B @f$, computes the multi_tensor_t @f$ A\cdot X + B @f$. The value contained in
    /// @f$ A @f$ is a constant coefficient to be applied to each layer in @f$ X @f$. The values in @f$ B @f$ is then added layerwise
    /// to every element.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aA Coefficient constant
    /// @param aX Tensor to transform
    /// @param aB Shift constant
    ///
    void AffineTransformOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aA, multi_tensor_t &aX, memory_buffer_t &aB );

    /// @brief Affine transformation
    ///
    /// Given multitensors @f$ A @f$, @f$ X @f$ and @f$ B @f$, computes the multi_tensor_t @f$ A\cdot X + B @f$. The value contained in
    /// @f$ A @f$ is a constant coefficient to be applied to each layer in @f$ X @f$. The value in @f$ B @f$ is then added
    /// to every element.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aA Coefficient constant
    /// @param aX Tensor to transform
    /// @param aB Shift constant
    ///
    void AffineTransformOp( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aA, multi_tensor_t &aX, scalar_value_t &aB );

    /// @brief Floor
    ///
    /// Given a multitensor @f$ X @f$, computes the multi_tensor_t @f$ Y @f$ whose values are the floor value of the corresponding
    /// element in @f$ X @f$.
    ///
    /// @param aOut Output tensor.
    /// @param aX Tensor to transform
    ///
    void FloorOp( multi_tensor_t &aOut, multi_tensor_t &aX );

    /// @brief Ceiling
    ///
    /// Given a multitensor @f$ X @f$, computes the multi_tensor_t @f$ Y @f$ whose values are the ceiling value of the corresponding
    /// element in @f$ X @f$.
    ///
    /// @param aOut Output tensor.
    /// @param aX Tensor to transform
    ///
    void CeilOp( multi_tensor_t &aOut, multi_tensor_t &aX );

    /// @brief Absolute value
    ///
    /// Given a multitensor @f$ X @f$, computes the multi_tensor_t @f$ Y @f$ whose values are the absolute values of the corresponding
    /// element in @f$ X @f$.
    ///
    /// @param aOutputElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aX Tensor to transform
    ///
    void AbsOp( scalar_type_t aOutputElementType, multi_tensor_t &aOut, multi_tensor_t &aX );

    /// @brief Square root
    ///
    /// Given a multitensor @f$ X @f$, computes the multi_tensor_t @f$ Y @f$ whose values are the square roots of the corresponding
    /// element in @f$ X @f$. The value if an element is negative is unspecified.
    ///
    /// @param aOutputElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aX Tensor to transform
    ///
    void SqrtOp( scalar_type_t aOutputElementType, multi_tensor_t &aOut, multi_tensor_t &aX );

    /// @brief Round
    ///
    /// Given a multitensor @f$ X @f$, computes the multi_tensor_t @f$ Y @f$ whose values are the rounded values of the corresponding
    /// element in @f$ X @f$. The value if an element is negative is unspecified.
    ///
    /// @param aOutputElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aX Tensor to transform
    ///
    void RoundOp( scalar_type_t aOutputElementType, multi_tensor_t &aOut, multi_tensor_t &aX );

    /// @brief Count the number of true elements
    ///
    /// Given a multitensor @f$ X @f$, computes the multi_tensor_t @f$ Y @f$ whose values are the number of true (non-zero) elements
    /// in @f$ X @f$. This effectively calls CountNonZeroOp.
    ///
    /// @param aOut Output tensor.
    /// @param aX Tensor to transform.
    /// @param aBlockSizes Product of the lengths of the first rank-1 dimensions of the input tensor
    /// @param aElementCount Length of the last dimension of the input tensor
    /// @param aMaxBlockSize Maximum value of the `aBlockSizes` parameter
    ///
    void CountTrueOp( multi_tensor_t &aOut, multi_tensor_t &aX, memory_buffer_t &aBlockSizes, memory_buffer_t &aElementCount,
                      uint32_t aMaxBlockSize );

    /// @brief Count the number of non-zero elements
    ///
    /// Given a multitensor @f$ X @f$, computes the multi_tensor_t @f$ Y @f$ whose values are the number of non-zero elements
    /// in @f$ X @f$.
    ///
    /// @param aOutputElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aX Tensor to transform
    /// @param aBlockSizes Product of the lengths of the first rank-1 dimensions of the input tensor
    /// @param aElementCount Length of the last dimension of the input tensor
    /// @param aMaxBlockSize Maximum value of the `aBlockSizes` parameter
    ///
    void CountNonZeroOp( scalar_type_t aOutputElementType, multi_tensor_t &aOut, multi_tensor_t &aX, memory_buffer_t &aBlockSizes,
                         memory_buffer_t &aElementCount, uint32_t aMaxBlockSize );

    /// @brief Count the number of zero elements
    ///
    /// Given a multitensor @f$ X @f$, computes the multi_tensor_t @f$ Y @f$ whose values are the number of zero elements
    /// in @f$ X @f$.
    ///
    /// @param aOutputElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aX Tensor to transform
    /// @param aBlockSizes Product of the lengths of the first rank-1 dimensions of the input tensor
    /// @param aElementCount Length of the last dimension of the input tensor
    /// @param aMaxBlockSize Maximum value of the `aBlockSizes` parameter
    ///
    void CountZeroOp( scalar_type_t aOutputElementType, multi_tensor_t &aOut, multi_tensor_t &aX, memory_buffer_t &aBlockSizes,
                      memory_buffer_t &aElementCount, uint32_t aMaxBlockSize );

    /// @brief Sum the elements of a given tensor along the last dimension
    ///
    /// Given a multitensor @f$ X @f$, computes the multi_tensor_t @f$ Y @f$ whose values are the sums of elements of
    /// @f$ X @f$ in the specified intervals along the last dimension
    ///
    /// @param aOutputElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aX Tensor to transform
    /// @param aBegin Lower index value for each layer
    /// @param aEnd Upper index value for every layer
    /// @param aElementCount Length of the last dimension of the input tensor
    /// @param aBlockSizes Product of the lengths of the first rank-1 dimensions of the input tensor
    /// @param aMaxBlockSize Maximum value of the `aBlockSizes` parameter
    ///
    void ArraySummationOp( scalar_type_t aOutputElementType, multi_tensor_t &aOut, multi_tensor_t &aX, memory_buffer_t &aBegin,
                           memory_buffer_t &aEnd, memory_buffer_t &aElementCount, memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize );

    /// @brief Slice the last dimension of a given tensor
    ///
    /// Given a multitensor @f$ X @f$, computes the multi_tensor_t @f$ Y @f$ whose values are the elements of
    /// @f$ X @f$ in the specified intervals along the last dimension.
    ///
    /// @param aOutputElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aX Tensor to transform
    /// @param aBegin Lower index value for each layer
    /// @param aEnd Upper index value for every layer
    /// @param aElementCount Length of the last dimension of the input tensor
    /// @param aBlockSizes Product of the lengths of the first rank-1 dimensions of the input tensor
    /// @param aMaxBlockSize Maximum value of the `aBlockSizes` parameter
    ///
    void ArraySliceOp( scalar_type_t aOutputElementType, multi_tensor_t &aOut, multi_tensor_t &aX, memory_buffer_t &aBegin, memory_buffer_t &aEnd,
                       memory_buffer_t &aElementCount, memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize );

    /// @brief Finite differences along the last dimension of a given tensor
    ///
    /// Given a multitensor @f$ X @f$, computes the multi_tensor_t @f$ Y @f$ whose values are the finite differences of
    /// @f$ X @f$ along the last dimension.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aX Tensor to transform
    /// @param aCount Order of the finite difference operator
    /// @param aElementCount Length of the last dimension of the input tensor
    /// @param aBlockSizes Product of the lengths of the first rank-1 dimensions of the input tensor
    /// @param aMaxBlockSize Maximum value of the `aBlockSizes` parameter
    ///
    void DiffOp( scalar_type_t aOutputElementType, multi_tensor_t &aOut, multi_tensor_t &aX, uint32_t aCount, memory_buffer_t &aElementCount,
                 memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize );

    /// @brief Shifts along the last dimension of a given tensor
    ///
    /// Given a multitensor @f$ X @f$, computes the multi_tensor_t @f$ Y @f$ whose values are the finite shifts of
    /// @f$ X @f$ along the last dimension.
    ///
    /// @param aOutputElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aX Tensor to transform
    /// @param aCount Order of the shift difference operator
    /// @param aElementCount Length of the last dimension of the input tensor
    /// @param aBlockSizes Product of the lengths of the first rank-1 dimensions of the input tensor
    /// @param aMaxBlockSize Maximum value of the `aBlockSizes` parameter
    ///
    void ShiftOp( scalar_type_t aOutputElementType, multi_tensor_t &aOut, multi_tensor_t &aX, int32_t aCount, scalar_value_t &aFillValue,
                  memory_buffer_t &aElementCount, memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize );

    /// @brief One-dimensional convolution along the last dimension
    ///
    /// Given multitensors @f$ X @f$ and @f$ K @f$, computes the multi_tensor_t @f$ Y @f$ whose values are the convolutions of
    /// @f$ X @f$ and @f$ Y @f$ along the last dimension.
    ///
    /// @param aOutputElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aArray0 Tensor to transform
    /// @param aElementCount0 Length of the last dimension of the input tensor
    /// @param aBlockSizes0 Product of the lengths of the first rank-1 dimensions of the input tensor `aArray0`
    /// @param aMaxElementCount0 Maximum value of the `aElementCount0` parameter
    /// @param aMaxBlockSize0 Maximum value of the `aBlockSizes0` parameter
    /// @param aArray1 convolution kernel
    /// @param aElementCount1 Length of the last dimension of the convolution kernel
    /// @param aBlockSizes1 Product of the lengths of the first rank-1 dimensions of the convolution kernel
    /// @param aMaxBlockSize1 Maximum value of the `aBlockSizes1` parameter
    ///
    void Conv1DOp( scalar_type_t aOutputElementType, multi_tensor_t &aOut, multi_tensor_t &aArray0, memory_buffer_t &aElementCount0,
                   memory_buffer_t &aBlockSizes0, uint32_t aMaxElementCount0, uint32_t aMaxBlockSize0, multi_tensor_t &aArray1,
                   memory_buffer_t &aElementCount1, memory_buffer_t &aBlockSizes1, uint32_t aMaxBlockSize1 );

    /// @brief Concatenation along the last dimension
    ///
    /// Given multitensors @f$ X @f$ and @f$ K @f$, computes the multi_tensor_t @f$ Y @f$ whose values are the concatenation of
    /// @f$ X @f$ and @f$ Y @f$ along the last dimension.
    ///
    /// @param aOutputElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aArray0 Tensor to concatenate
    /// @param aElementCount0 Length of the last dimension of the first input tensor
    /// @param aArray1 Tensor to concatenate
    /// @param aElementCount1 Length of the last dimension of the second input tensor
    /// @param aBlockSizes Product of the lengths of the first rank-1 dimensions of the input tensor `aArray0`
    /// @param aMaxBlockSize Maximum value of the `aBlockSizes` parameter
    ///
    void HCatOp( scalar_type_t aOutputElementType, multi_tensor_t &aOut, multi_tensor_t &aArray0, memory_buffer_t &aElementCount0,
                 multi_tensor_t &aArray1, memory_buffer_t &aElementCount1, memory_buffer_t &aBlockSizes, uint32_t aMaxBlockSize );

} // namespace SE::TensorOps