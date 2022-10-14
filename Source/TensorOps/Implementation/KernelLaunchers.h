/// @file   KernelLaunchers.h
///
/// @brief  C++ API for Cuda computation launchers
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>

#include "Core/Math/Types.h"
#include "Core/Cuda/MemoryPool.h"
#include "Core/Cuda/MultiTensor.h"

#include "../ScalarTypes.h"

namespace LTSE::TensorOps
{

    using namespace LTSE::Cuda;

    struct TextureData
    {
        cudaTextureObject_t Texture = 0;
        math::vec4 SubArea          = { 0.0f, 0.0f, 1.0, 1.0 };
        math::vec2 Scaling          = { 1.0f, 1.0f };

        TextureData()                      = default;
        TextureData( const TextureData & ) = default;
    };

    /// @brief Fill @ref MultiTensor with a constant value
    ///
    /// The entire tensor is filled with the value specofoed by `aConstant`
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aArray Generalized tensor to process
    /// @param aConstant Value to fill the buffer with
    ///
    void ConstantFill( eScalarType aTensorElementType, MultiTensor &aArray, ScalarValue &aInitialValue );

    /// @brief Fill @ref MultiTensor with a constant value
    ///
    /// The entire tensor is filled with the values specofoed in `aConstants`. The vector represented by
    /// `aConstants` should have as many elements as `aArray` has layers.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aArray Generalized tensor to process
    /// @param aConstants Vector of values to fill the buffer with
    ///
    void ConstantFill( eScalarType aTensorElementType, MultiTensor &aArray, MemoryBuffer &aInitialValues );
    void RandomUniformFill( eScalarType aTensorElementType, MultiTensor &aArray );
    void RandomNormalFill( eScalarType aTensorElementType, MultiTensor &aArray, ScalarValue &a_Mu, ScalarValue &a_Sigma );

    /// @brief Add a single scalar to all elements of a tensor
    ///
    /// All multi-tensors should have the same shape and the same element types `_Ty`
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aArray Left operand
    /// @param aConstant Element to add to `aArray`
    ///
    void AddOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight );

    /// @brief Pointwise addition of two @ref MultiTensors
    ///
    /// All multi-tensors should have the same shape and the same element types
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aLeft Left operand
    /// @param aRight Right operand
    ///
    void AddOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight );
    void AddOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, eBroadcastHint aBroadcastHint, MemoryBuffer &aBlockSizes,
                uint32_t aMaxBlockSize, MemoryBuffer &aBroadcastSizes, uint32_t aMaxBroadcastSizes );

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
    void AddOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight );

    /// @brief Pointwise multiplication of two @ref MultiTensors
    ///
    /// All multi-tensors should have the same shape and the same element types
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aLeft Left operand
    /// @param aRight Right operand
    ///
    void MultiplyOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight );
    void MultiplyOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, eBroadcastHint aBroadcastHint, MemoryBuffer &aBlockSizes,
                     uint32_t aMaxBlockSize, MemoryBuffer &aBroadcastSizes, uint32_t aMaxBroadcastSizes );

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
    void MultiplyOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight );

    /// @brief Multiply the elements of an array by a single scalar
    ///
    /// All multi-tensors should have the same shape and the same element types `_Ty`
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aArray Left operand
    /// @param aConstant Element to add to `aArray`
    ///
    void MultiplyOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight );

    /// @brief Subtract the elements of an array from a scalar
    ///
    /// All multi-tensors should have the same shape and the same element types `_Ty`.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aArray Left operand
    /// @param aConstant Element to add to `aArray`
    ///
    void SubtractOp( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aLeft, MultiTensor &aRight );

    /// @brief Subtract a scalar from every element of an array
    ///
    /// All multi-tensors should have the same shape and the same element types `_Ty`.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aArray Left operand
    /// @param aConstant Element to add to `aArray`
    ///
    void SubtractOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight );

    /// @brief Pointwise subtraction of two arrays
    ///
    /// All multi-tensors should have the same shape and the same element types `_Ty`.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aArray Left operand
    /// @param aConstant Element to add to `aArray`
    /// @param aOut Output tensor.
    ///
    void SubtractOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight );
    void SubtractOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, eBroadcastHint aBroadcastHint, MemoryBuffer &aBlockSizes,
                     uint32_t aMaxBlockSize, MemoryBuffer &aBroadcastSizes, uint32_t aMaxBroadcastSizes );

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
    void SubtractOp( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight );

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
    void SubtractOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight );

    /// @brief Divide the elements of an array by a scalar
    ///
    /// All multi-tensors should have the same shape and the same element types `_Ty`.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aArray Left operand
    /// @param aConstant Element to add to `aArray`
    ///
    void DivideOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight );

    /// @brief Divide a scalar by every element of an array
    ///
    /// All multi-tensors should have the same shape and the same element types `_Ty`.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aArray Left operand
    /// @param aConstant Element to add to `aArray`
    ///
    void DivideOp( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aLeft, MultiTensor &aRight );

    /// @brief Pointwise division of twu multitensors
    ///
    /// All multi-tensors should have the same shape and the same element types `_Ty`.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aArray Left operand
    /// @param aConstant Element to add to `aArray`
    ///
    void DivideOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight );
    void DivideOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, eBroadcastHint aBroadcastHint, MemoryBuffer &aBlockSizes,
                   uint32_t aMaxBlockSize, MemoryBuffer &aBroadcastSizes, uint32_t aMaxBroadcastSizes );

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
    void DivideOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight );

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
    void DivideOp( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight );

    /// @brief Conjunction operators
    ///
    /// All multi-tensors should have the same shape and the same element types`. If one of the arguments is a
    /// MemoryBuffer, then it should hold a vector of values whose length is equal to the number of layers of
    /// the other argument (a MultiTensor)
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aLeft Left operand
    /// @param aRight Right operand
    ///
    void AndOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight );
    void AndOp( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aLeft, MultiTensor &aRight );
    void AndOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight );
    void AndOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, eBroadcastHint aBroadcastHint, MemoryBuffer &aBlockSizes,
                uint32_t aMaxBlockSize, MemoryBuffer &aBroadcastSizes, uint32_t aMaxBroadcastSizes );
    void AndOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight );
    void AndOp( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight );

    /// @brief Disjunction operators
    ///
    /// All multi-tensors should have the same shape and the same element types`. If one of the arguments is a
    /// MemoryBuffer, then it should hold a vector of values whose length is equal to the number of layers of
    /// the other argument (a MultiTensor)
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aLeft Left operand
    /// @param aRight Right operand
    ///
    void OrOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight );
    void OrOp( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aLeft, MultiTensor &aRight );
    void OrOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight );
    void OrOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, eBroadcastHint aBroadcastHint, MemoryBuffer &aBlockSizes,
               uint32_t aMaxBlockSize, MemoryBuffer &aBroadcastSizes, uint32_t aMaxBroadcastSizes );
    void OrOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight );
    void OrOp( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight );

    /// @brief Negation operator
    ///
    /// All multi-tensors should have the same shape and the same element types`.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aOperand Operand
    ///
    void NotOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aOperand );

    /// @brief Bitwise conjunction operators
    ///
    /// All multi-tensors should have the same shape and the same element types`. If one of the arguments is a
    /// MemoryBuffer, then it should hold a vector of values whose length is equal to the number of layers of
    /// the other argument (a MultiTensor)
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aLeft Left operand
    /// @param aRight Right operand
    ///
    void BitwiseAndOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight );
    void BitwiseAndOp( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aLeft, MultiTensor &aRight );
    void BitwiseAndOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight );
    void BitwiseAndOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, eBroadcastHint aBroadcastHint, MemoryBuffer &aBlockSizes,
                       uint32_t aMaxBlockSize, MemoryBuffer &aBroadcastSizes, uint32_t aMaxBroadcastSizes );
    void BitwiseAndOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight );
    void BitwiseAndOp( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight );

    /// @brief Bitwise disjunction operators
    ///
    /// All multi-tensors should have the same shape and the same element types`. If one of the arguments is a
    /// MemoryBuffer, then it should hold a vector of values whose length is equal to the number of layers of
    /// the other argument (a MultiTensor)
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aLeft Left operand
    /// @param aRight Right operand
    ///
    void BitwiseOrOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight );
    void BitwiseOrOp( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aLeft, MultiTensor &aRight );
    void BitwiseOrOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight );
    void BitwiseOrOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, eBroadcastHint aBroadcastHint, MemoryBuffer &aBlockSizes,
                      uint32_t aMaxBlockSize, MemoryBuffer &aBroadcastSizes, uint32_t aMaxBroadcastSizes );
    void BitwiseOrOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight );
    void BitwiseOrOp( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight );

    /// @brief Bitwise negation operators
    ///
    /// All multi-tensors should have the same shape and the same element types`.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aOperand Operand
    ///
    void BitwiseNotOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aOperand );

    /// @brief Check whether a value lies in an interval
    ///
    /// All multi-tensors should have the same shape and the same element types`. If one of the arguments is a
    /// MemoryBuffer, then it should hold a vector of values whose length is equal to the number of layers of
    /// the other argument (a MultiTensor)
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aX MultiTensor holding the value to test
    /// @param aLower Lower bound
    /// @param aUpper Upper bound
    /// @param aStrictLower Should the comparison with the lower bound be strict?
    /// @param aStrictUpper Should the comparison with the upper bound be strict?
    ///
    void InIntervalOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aX, MultiTensor &aLower, MultiTensor &aUpper, bool aStrictLower, bool aStrictUpper );
    void InIntervalOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aX, MultiTensor &aLower, MemoryBuffer &aUpper, bool aStrictLower, bool aStrictUpper );
    void InIntervalOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aX, MultiTensor &aLower, ScalarValue &aUpper, bool aStrictLower, bool aStrictUpper );
    void InIntervalOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aX, MemoryBuffer &aLower, MultiTensor &aUpper, bool aStrictLower, bool aStrictUpper );
    void InIntervalOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aX, MemoryBuffer &aLower, MemoryBuffer &aUpper, bool aStrictLower, bool aStrictUpper );
    void InIntervalOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aX, MemoryBuffer &aLower, ScalarValue &aUpper, bool aStrictLower, bool aStrictUpper );
    void InIntervalOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aX, ScalarValue &aLower, MultiTensor &aUpper, bool aStrictLower, bool aStrictUpper );
    void InIntervalOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aX, ScalarValue &aLower, MemoryBuffer &aUpper, bool aStrictLower, bool aStrictUpper );
    void InIntervalOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aX, ScalarValue &aLower, ScalarValue &aUpper, bool aStrictLower, bool aStrictUpper );

    /// @brief Test equaliy between tensors
    ///
    /// All multi-tensors should have the same shape and the same element types`.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aLeft Left operand
    /// @param aRight Right operand
    ///
    void EqualOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight );
    void EqualOp( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aLeft, MultiTensor &aRight );
    void EqualOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight );
    void EqualOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, eBroadcastHint aBroadcastHint, MemoryBuffer &aBlockSizes,
                  uint32_t aMaxBlockSize, MemoryBuffer &aBroadcastSizes, uint32_t aMaxBroadcastSizes );
    void EqualOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight );
    void EqualOp( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight );

    /// @brief Test whether a tensor is pointwise less than another tensor
    ///
    /// All multi-tensors should have the same shape and the same element types`.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aLeft Left operand
    /// @param aRight Right operand
    ///
    void LessThanOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight );
    void LessThanOp( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aLeft, MultiTensor &aRight );
    void LessThanOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight );
    void LessThanOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, eBroadcastHint aBroadcastHint, MemoryBuffer &aBlockSizes,
                     uint32_t aMaxBlockSize, MemoryBuffer &aBroadcastSizes, uint32_t aMaxBroadcastSizes );
    void LessThanOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight );
    void LessThanOp( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight );

    /// @brief Test whether a tensor is pointwise less than or equal to another tensor
    ///
    /// All multi-tensors should have the same shape and the same element types`.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aLeft Left operand
    /// @param aRight Right operand
    ///
    void LessThanOrEqualOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight );
    void LessThanOrEqualOp( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aLeft, MultiTensor &aRight );
    void LessThanOrEqualOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight );
    void LessThanOrEqualOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, eBroadcastHint aBroadcastHint, MemoryBuffer &aBlockSizes,
                            uint32_t aMaxBlockSize, MemoryBuffer &aBroadcastSizes, uint32_t aMaxBroadcastSizes );
    void LessThanOrEqualOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight );
    void LessThanOrEqualOp( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight );

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
    void WhereOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aCondition, MultiTensor &aValueIfTrue, MultiTensor &aValueIfFalse );
    void WhereOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aCondition, MultiTensor &aValueIfTrue, MemoryBuffer &aValueIfFalse );
    void WhereOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aCondition, MultiTensor &aValueIfTrue, ScalarValue &aValueIfFalse );
    void WhereOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aCondition, MemoryBuffer &aValueIfTrue, MultiTensor &aValueIfFalse );
    void WhereOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aCondition, MemoryBuffer &aValueIfTrue, MemoryBuffer &aValueIfFalse );
    void WhereOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aCondition, MemoryBuffer &aValueIfTrue, ScalarValue &aValueIfFalse );
    void WhereOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aCondition, ScalarValue &aValueIfTrue, MultiTensor &aValueIfFalse );
    void WhereOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aCondition, ScalarValue &aValueIfTrue, MemoryBuffer &aValueIfFalse );
    void WhereOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aCondition, ScalarValue &aValueIfTrue, ScalarValue &aValueIfFalse );

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
    void ARangeOp( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MemoryBuffer &aRight, MemoryBuffer &aDelta, uint32_t aMaxSubdivisions );

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
    void RepeatOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &Array, MemoryBuffer &aRepetitions, uint32_t aMaxRepetitions );

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
    void TileOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &Array, MemoryBuffer &aRepetitions, uint32_t aMaxRepetitions );

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
    void LinearSpaceOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight, MemoryBuffer &Subdivisions, uint32_t aMaxSubdivisions );

    /// @brief Computes the pointwise mix of two tensors
    ///
    /// This function computes the tensor @f$ (1-t)\cdot A + t\cdot B @f$
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aArray Array to repeat
    /// @param aRepetitions Nummber of repetitions
    /// @param aOut Output tensor.
    ///
    void MixOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aA, MultiTensor &aB, MultiTensor &t );

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
    void Sample2DOp( MultiTensor &aOut, MultiTensor &aX, MultiTensor &aY, MemoryBuffer &aTextures );
    void Sample2DOp( MultiTensor &aOut, MultiTensor &aX, MemoryBuffer &aY, MemoryBuffer &aTextures );
    void Sample2DOp( MultiTensor &aOut, MultiTensor &aX, ScalarValue &aY, MemoryBuffer &aTextures );
    void Sample2DOp( MultiTensor &aOut, MemoryBuffer &aX, MultiTensor &aY, MemoryBuffer &aTextures );
    void Sample2DOp( MultiTensor &aOut, ScalarValue &aX, MultiTensor &aY, MemoryBuffer &aTextures );

    /// @brief Fixed point conversion
    ///
    /// Converts a tensor of floating point numbers into a tensor of integers by multiplying each element by a scaling factor.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aArray Array to repeat
    /// @param aRepetitions Nummber of repetitions
    ///
    void ToFixedPointOp( eScalarType aTensorElementType, MultiTensor &aOut, eScalarType a_OutputElementType, MultiTensor &Array, ScalarValue &Scaling );

    /// @brief Affine transformation
    ///
    /// Given multitensors @f$ A @f$, @f$ X @f$ and @f$ B @f$, computes the MultiTensor @f$ A\cdot X + B @f$ using pointwise operations
    ///
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aA Coefficient tensor
    /// @param aX Tensor to transform
    /// @param aB Shift tensor
    ///
    void AffineTransformOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aA, MultiTensor &aX, MultiTensor &aB );

    /// @brief Affine transformation
    ///
    /// Given multitensors @f$ A @f$, @f$ X @f$ and @f$ B @f$, computes the MultiTensor @f$ A\cdot X + B @f$ using pointwise operations
    /// for the product, and considering each element of @f$ B @f$ to be a constant to be added to the individual layers of @f$ A\cdot X @f$
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aA Coefficient tensor
    /// @param aX Tensor to transform
    /// @param aB Shift vector
    ///
    void AffineTransformOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aA, MultiTensor &aX, MemoryBuffer &aB );

    /// @brief Affine transformation
    ///
    /// Given multitensors @f$ A @f$, @f$ X @f$ and @f$ B @f$, computes the MultiTensor @f$ A\cdot X + B @f$ using pointwise operations
    /// for the product, and considering @f$ B @f$ to be a constant to be added to the individual elements of @f$ A\cdot X @f$
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aA Coefficient tensor
    /// @param aX Tensor to transform
    /// @param aB Shift constant
    ///
    void AffineTransformOp( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aA, MultiTensor &aX, ScalarValue &aB );

    /// @brief Affine transformation
    ///
    /// Given multitensors @f$ A @f$, @f$ X @f$ and @f$ B @f$, computes the MultiTensor @f$ A\cdot X + B @f$. The values contained in
    /// @f$ A @f$ are constant coefficients to be applied to each layer in @f$ X @f$. The final sum is calculated pointwise.
    ///
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aA Coefficient vector
    /// @param aX Tensor to transform
    /// @param aB Shift tensor
    ///
    void AffineTransformOp( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aA, MultiTensor &aX, MultiTensor &aB );

    /// @brief Affine transformation
    ///
    /// Given multitensors @f$ A @f$, @f$ X @f$ and @f$ B @f$, computes the MultiTensor @f$ A\cdot X + B @f$. The values contained in
    /// @f$ A @f$ and @f$ B @f$ are constant coefficients and shift values to be applied to each layer in @f$ X @f$.
    ///
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aA Coefficient vector
    /// @param aX Tensor to transform
    /// @param aB Shift vector
    ///
    void AffineTransformOp( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aA, MultiTensor &aX, MemoryBuffer &aB );

    /// @brief Affine transformation
    ///
    /// Given multitensors @f$ A @f$, @f$ X @f$ and @f$ B @f$, computes the MultiTensor @f$ A\cdot X + B @f$. The values contained in
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
    void AffineTransformOp( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aA, MultiTensor &aX, ScalarValue &aB );

    /// @brief Affine transformation
    ///
    /// Given multitensors @f$ A @f$, @f$ X @f$ and @f$ B @f$, computes the MultiTensor @f$ A\cdot X + B @f$. The value contained in
    /// @f$ A @f$ is a constant coefficient to be applied to each layer in @f$ X @f$. The values in @f$ B @f$ is then added pointwise
    /// to every element.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aA Coefficient constant
    /// @param aX Tensor to transform
    /// @param aB Shift constant
    ///
    void AffineTransformOp( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aA, MultiTensor &aX, MultiTensor &aB );

    /// @brief Affine transformation
    ///
    /// Given multitensors @f$ A @f$, @f$ X @f$ and @f$ B @f$, computes the MultiTensor @f$ A\cdot X + B @f$. The value contained in
    /// @f$ A @f$ is a constant coefficient to be applied to each layer in @f$ X @f$. The values in @f$ B @f$ is then added layerwise
    /// to every element.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aA Coefficient constant
    /// @param aX Tensor to transform
    /// @param aB Shift constant
    ///
    void AffineTransformOp( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aA, MultiTensor &aX, MemoryBuffer &aB );

    /// @brief Affine transformation
    ///
    /// Given multitensors @f$ A @f$, @f$ X @f$ and @f$ B @f$, computes the MultiTensor @f$ A\cdot X + B @f$. The value contained in
    /// @f$ A @f$ is a constant coefficient to be applied to each layer in @f$ X @f$. The value in @f$ B @f$ is then added
    /// to every element.
    ///
    /// @param aTensorElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aA Coefficient constant
    /// @param aX Tensor to transform
    /// @param aB Shift constant
    ///
    void AffineTransformOp( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aA, MultiTensor &aX, ScalarValue &aB );

    /// @brief Floor
    ///
    /// Given a multitensor @f$ X @f$, computes the MultiTensor @f$ Y @f$ whose values are the floor value of the corresponding
    /// element in @f$ X @f$.
    ///
    /// @param aOut Output tensor.
    /// @param aX Tensor to transform
    ///
    void FloorOp( MultiTensor &aOut, MultiTensor &aX );

    /// @brief Ceiling
    ///
    /// Given a multitensor @f$ X @f$, computes the MultiTensor @f$ Y @f$ whose values are the ceiling value of the corresponding
    /// element in @f$ X @f$.
    ///
    /// @param aOut Output tensor.
    /// @param aX Tensor to transform
    ///
    void CeilOp( MultiTensor &aOut, MultiTensor &aX );

    /// @brief Absolute value
    ///
    /// Given a multitensor @f$ X @f$, computes the MultiTensor @f$ Y @f$ whose values are the absolute values of the corresponding
    /// element in @f$ X @f$.
    ///
    /// @param aOutputElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aX Tensor to transform
    ///
    void AbsOp( eScalarType aOutputElementType, MultiTensor &aOut, MultiTensor &aX );

    /// @brief Square root
    ///
    /// Given a multitensor @f$ X @f$, computes the MultiTensor @f$ Y @f$ whose values are the square roots of the corresponding
    /// element in @f$ X @f$. The value if an element is negative is unspecified.
    ///
    /// @param aOutputElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aX Tensor to transform
    ///
    void SqrtOp( eScalarType aOutputElementType, MultiTensor &aOut, MultiTensor &aX );

    /// @brief Round
    ///
    /// Given a multitensor @f$ X @f$, computes the MultiTensor @f$ Y @f$ whose values are the rounded values of the corresponding
    /// element in @f$ X @f$. The value if an element is negative is unspecified.
    ///
    /// @param aOutputElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aX Tensor to transform
    ///
    void RoundOp( eScalarType aOutputElementType, MultiTensor &aOut, MultiTensor &aX );

    /// @brief Count the number of true elements
    ///
    /// Given a multitensor @f$ X @f$, computes the MultiTensor @f$ Y @f$ whose values are the number of true (non-zero) elements
    /// in @f$ X @f$. This effectively calls CountNonZeroOp.
    ///
    /// @param aOut Output tensor.
    /// @param aX Tensor to transform.
    /// @param aBlockSizes Product of the lengths of the first rank-1 dimensions of the input tensor
    /// @param aElementCount Length of the last dimension of the input tensor
    /// @param aMaxBlockSize Maximum value of the `aBlockSizes` parameter
    ///
    void CountTrueOp( MultiTensor &aOut, MultiTensor &aX, MemoryBuffer &aBlockSizes, MemoryBuffer &aElementCount, uint32_t aMaxBlockSize );

    /// @brief Count the number of non-zero elements
    ///
    /// Given a multitensor @f$ X @f$, computes the MultiTensor @f$ Y @f$ whose values are the number of non-zero elements
    /// in @f$ X @f$.
    ///
    /// @param aOutputElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aX Tensor to transform
    /// @param aBlockSizes Product of the lengths of the first rank-1 dimensions of the input tensor
    /// @param aElementCount Length of the last dimension of the input tensor
    /// @param aMaxBlockSize Maximum value of the `aBlockSizes` parameter
    ///
    void CountNonZeroOp( eScalarType aOutputElementType, MultiTensor &aOut, MultiTensor &aX, MemoryBuffer &aBlockSizes, MemoryBuffer &aElementCount, uint32_t aMaxBlockSize );

    /// @brief Count the number of zero elements
    ///
    /// Given a multitensor @f$ X @f$, computes the MultiTensor @f$ Y @f$ whose values are the number of zero elements
    /// in @f$ X @f$.
    ///
    /// @param aOutputElementType Type of element in the buffer
    /// @param aOut Output tensor.
    /// @param aX Tensor to transform
    /// @param aBlockSizes Product of the lengths of the first rank-1 dimensions of the input tensor
    /// @param aElementCount Length of the last dimension of the input tensor
    /// @param aMaxBlockSize Maximum value of the `aBlockSizes` parameter
    ///
    void CountZeroOp( eScalarType aOutputElementType, MultiTensor &aOut, MultiTensor &aX, MemoryBuffer &aBlockSizes, MemoryBuffer &aElementCount, uint32_t aMaxBlockSize );

    /// @brief Sum the elements of a given tensor along the last dimension
    ///
    /// Given a multitensor @f$ X @f$, computes the MultiTensor @f$ Y @f$ whose values are the sums of elements of
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
    void ArraySummationOp( eScalarType aOutputElementType, MultiTensor &aOut, MultiTensor &aX, MemoryBuffer &aBegin, MemoryBuffer &aEnd, MemoryBuffer &aElementCount,
                           MemoryBuffer &aBlockSizes, uint32_t aMaxBlockSize );

    /// @brief Slice the last dimension of a given tensor
    ///
    /// Given a multitensor @f$ X @f$, computes the MultiTensor @f$ Y @f$ whose values are the elements of
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
    void ArraySliceOp( eScalarType aOutputElementType, MultiTensor &aOut, MultiTensor &aX, MemoryBuffer &aBegin, MemoryBuffer &aEnd, MemoryBuffer &aElementCount,
                       MemoryBuffer &aBlockSizes, uint32_t aMaxBlockSize );

    /// @brief Finite differences along the last dimension of a given tensor
    ///
    /// Given a multitensor @f$ X @f$, computes the MultiTensor @f$ Y @f$ whose values are the finite differences of
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
    void DiffOp( eScalarType aOutputElementType, MultiTensor &aOut, MultiTensor &aX, uint32_t aCount, MemoryBuffer &aElementCount, MemoryBuffer &aBlockSizes,
                 uint32_t aMaxBlockSize );

    /// @brief Shifts along the last dimension of a given tensor
    ///
    /// Given a multitensor @f$ X @f$, computes the MultiTensor @f$ Y @f$ whose values are the finite shifts of
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
    void ShiftOp( eScalarType aOutputElementType, MultiTensor &aOut, MultiTensor &aX, int32_t aCount, ScalarValue &aFillValue, MemoryBuffer &aElementCount,
                  MemoryBuffer &aBlockSizes, uint32_t aMaxBlockSize );

    /// @brief One-dimensional convolution along the last dimension
    ///
    /// Given multitensors @f$ X @f$ and @f$ K @f$, computes the MultiTensor @f$ Y @f$ whose values are the convolutions of
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
    void Conv1DOp( eScalarType aOutputElementType, MultiTensor &aOut, MultiTensor &aArray0, MemoryBuffer &aElementCount0, MemoryBuffer &aBlockSizes0, uint32_t aMaxElementCount0,
                   uint32_t aMaxBlockSize0, MultiTensor &aArray1, MemoryBuffer &aElementCount1, MemoryBuffer &aBlockSizes1, uint32_t aMaxBlockSize1 );

    /// @brief Concatenation along the last dimension
    ///
    /// Given multitensors @f$ X @f$ and @f$ K @f$, computes the MultiTensor @f$ Y @f$ whose values are the concatenation of
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
    void HCatOp( eScalarType aOutputElementType, MultiTensor &aOut, MultiTensor &aArray0, MemoryBuffer &aElementCount0, MultiTensor &aArray1, MemoryBuffer &aElementCount1,
                 MemoryBuffer &aBlockSizes, uint32_t aMaxBlockSize );

} // namespace LTSE::TensorOps