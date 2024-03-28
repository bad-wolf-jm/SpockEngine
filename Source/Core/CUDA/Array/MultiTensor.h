/// @file   MultiTensor.h
///
/// @brief  MultiTensor class definition
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert. All rights reserved.

#pragma once

#include <algorithm>
#include <fmt/core.h>
#include <numeric>
#include <stdexcept>

#include "MemoryPool.h"

#include "Core/multi_tensor/tensor_shape.h"

namespace SE::Cuda
{
    /// @brief Generalized tensor structures
    ///
    /// A tensor van be viewed as a generalization of the concepts of scalars, vectors and matrices. They provide
    /// a form of closure under algebraic operations which is impossible when one is restricted to vectors and matrices.
    /// For example, the tensor product of two vectors is generally a matrix, but the tensor product of two tensors is
    /// another tensor. One way we may represent them is via multidimensional arrays. For our purposes, if is more convenient
    /// to stack several such tensors into a single structure, provided they all share the same number of dimensions.
    /// The main purpose of this class is to provide such a structure. A *generalized tensor* is characterized by a shape
    /// which holds the dimension data as well as all the information required to access the different parts of
    /// the generalized tensor. For the sake of generality and code simplicity, the generalized tensor itself has
    /// no knowledge of the actual type of the elements it contains.
    ///
    /// Generalized tensor are continuous  segments of GPU memory which is split among several buffers.
    /// All buffers in the stack must have the same rank, but can have different sizes. This class
    /// is especially well suited for passing multiple buffers with similar layout but different sizes
    /// to a Cuda kernel. The `MultiTensor` class can be passed directly to Cuda kernels which can access
    /// the various layers through device methods.
    ///
    /// @section s1 Creating a stack
    ///
    /// @code{.cpp}
    ///   // Create an empty stack of buffer containing elements of type math::vec3
    ///   // Each layer of the stack is a 3-dimensional array.
    ///   size_t lPoolSize = 1024;
    ///   MemoryPool lMemoryPool(lPoolSize);
    ///   MultiTensor lTestTensor(lMemoryPool, sTensorShape({{1, 2, 3}, {4, 5, 6}}, sizeof(math::vec3)));
    /// @endcode
    ///
    class multi_tensor_t
    {
      public:
        multi_tensor_t()  = default;
        ~multi_tensor_t() = default;

        /// @brief Allocates a generalized tensor of the given shape from a memory pool
        ///
        /// @param aMemoryPool The memory pool from which to allocate the tensor
        /// @param aShape      The shape of the tensor to allocate
        ///
        multi_tensor_t( memory_pool_t &aMemoryPool, const tensor_shape_t &aShape );

        /// @brief Create a generalized tensor of the given shape using a preallocated buffer from a memory pool
        ///
        /// @param aMemoryPool The memory pool from which to allocate the tensor
        /// @param aMemoryBuffer Preallocated buffer to hold data
        /// @param aShape      The shape of the tensor to allocate
        ///
        multi_tensor_t( memory_pool_t &aMemoryPool, memory_buffer_t &aMemoryBuffer, const tensor_shape_t &aShape );

        /// @brief Retrieves the shape of the tensor
        SE_CUDA_INLINE SE_CUDA_HOST_DEVICE_FUNCTION_DEF tensor_shape_t &Shape()
        {
            return mShape;
        }

        /// @brief Retrieve a pointer to the i-th layer
        ///
        /// @param i The index of the stack layer to retrieve
        ///
        /// @return  A MemoryBuffer pointing to the layer
        ///
        template <typename _Ty>
        SE_CUDA_INLINE SE_CUDA_DEVICE_FUNCTION_DEF _Ty *DeviceBufferAt( uint32_t i ) const
        {
            buffer_size_info_t lBufferSize = mShape.GetBufferSizeAs<_Ty>( i );
            return DataAs<_Ty>() + lBufferSize.mOffset;
        }

        /// @brief Retrieve a view into the i-th layer
        ///
        /// @param i The index of the stack layer to retrieve
        ///
        /// @return  A MemoryBuffer pointing to the layer
        ///
        memory_buffer_t BufferAt( uint32_t i ) const
        {
            auto &lBufferInfo = mShape.GetBufferSize( i );
            return mMemoryBuffer.View( lBufferInfo.mSize, lBufferInfo.mOffset );
        }

        /// @brief Fetch the contents of the i-th layer
        ///
        /// Retrieve the contents of the i-th layer of the tensor into a newly allocated vector.
        ///
        /// @tparam _Tx Type of the elements to retrieve.
        ///
        /// @param i The index of the stack layer to retrieve
        ///
        /// @return New vector of type _Tx containing the data.
        ///
        template <typename _Tx>
        vector_t<_Tx> FetchBufferAt( uint32_t i ) const
        {
            auto &lBufferInfo = mShape.GetBufferSizeAs<_Tx>( i );
            return mMemoryBuffer.Fetch<_Tx>( lBufferInfo.mOffset, lBufferInfo.mSize );
        }

        /// @brief Fetch the contents of the underlying buffer
        ///
        /// Retrieve the contents of the underlying GPU buffer as a single continus vector, without any notion
        /// of dimensionality.
        ///
        /// @tparam _Tx Type of the elements to retrieve.
        ///
        /// @param i The index of the stack layer to retrieve
        ///
        /// @return New vector of type _Tx containing the data.
        ///
        template <typename _Tx>
        vector_t<_Tx> FetchFlattened() const
        {
            return mMemoryBuffer.Fetch<_Tx>();
        }

        /// @brief Upload the contents of a vector to the tensor
        ///
        /// Data is uploaded as a flat vector, with no notion of dimensionality
        ///
        /// @tparam _Tx Type of the elements to upload.
        ///
        /// @param aArray Data to upload
        ///
        template <typename _Tx>
        void Upload( vector_t<_Tx> const &aArray ) const
        {
            mMemoryBuffer.Upload<_Tx>( aArray );
        }

        /// @brief Upload the contents of a vector to the i-thy layer of a tensor
        ///
        /// Data is uploaded as a flat vector, with no notion of dimensionality
        ///
        /// @tparam _Tx Type of the elements to upload.
        ///
        /// @param aArray  Data to upload
        /// @param aLayer  Layer into which the data should be copied
        /// @param aOffset Offset into the layer, in `_Ty`
        ///
        template <typename _Tx>
        void Upload( vector_t<_Tx> const &aArray, uint32_t aLayer, uint32_t aOffset ) const
        {
            BufferAt( aLayer ).Upload<_Tx>( aArray, aOffset );
        }

        /// @brief Overloaded member provided for convenience.
        ///
        /// Upload the contents of a vector to the i-thy layer of a tensor. Data is uploaded as a flat vector,
        /// with no notion of dimensionality
        ///
        /// @tparam _Tx Type of the elements to upload.
        ///
        /// @param aArray  Data to upload
        /// @param aLayer  Layer into which the data should be copied
        ///
        template <typename _Tx>
        void Upload( vector_t<_Tx> const &aArray, uint32_t aLayer ) const
        {
            Upload( aArray, aLayer, 0 );
        }

        /// @brief Size, in bytes, of the tensor.
        SE_CUDA_INLINE SE_CUDA_HOST_DEVICE_FUNCTION_DEF size_t Size() const
        {
            return mMemoryBuffer.Size();
        }

        /// @brief Size of the tensor as elements of type `_Ty`.
        template <typename _Tx>
        SE_CUDA_INLINE SE_CUDA_HOST_DEVICE_FUNCTION_DEF size_t SizeAs() const
        {
            return mMemoryBuffer.SizeAs<_Tx>();
        }

        /// @brief Pointer to the underlying data as type `_Ty`.
        template <typename _Tx>
        SE_CUDA_INLINE SE_CUDA_HOST_DEVICE_FUNCTION_DEF _Tx *DataAs() const
        {
            return mMemoryBuffer.DataAs<_Tx>();
        }

        memory_buffer_t &GetMemoryBuffer()
        {
            return mMemoryBuffer;
        }

      private:
        tensor_shape_t  mShape{};        //!< Shape of the tensor
        memory_buffer_t mMemoryBuffer{}; //!< Memory buffer assigned to the tensor
    };

} // namespace SE::Cuda
