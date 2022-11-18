/// @file   MultiTensor.h
///
/// @brief  MultiTensor class definition
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#pragma once

#include <algorithm>
#include <fmt/core.h>
#include <numeric>
#include <stdexcept>

#include "MemoryPool.h"

namespace SE::Cuda
{

    /// @brief Buffer offset structure
    struct sBufferSizeInfo
    {
        uint32_t mSize   = 0; //!< Size of current buffer
        uint32_t mOffset = 0; //!< Offset of current buffer

        sBufferSizeInfo()                          = default;
        sBufferSizeInfo( const sBufferSizeInfo & ) = default;
    };

    bool operator==( const sBufferSizeInfo &lLhs, const sBufferSizeInfo &aRhs );

    /// @brief Shape of a gemeralized tensor.
    ///
    /// This is an abstract representation of the shape of a generalized tensor. The main structure consists of a vector
    /// containing the dimensions of each individual *layer* in the generalized tensor. The only constraint is that each
    /// layer have the same rank, i.e. the same number of elements. This allows for a well-defined notion of rank for the
    /// entire generalized tensor, and makes it easy to share data between the CPU and the GPU.
    ///
    /// @code{.cpp}
    ///   // Creates a sTensorShape of rank 3, which will hold elements of type `math::vec3`
    ///   sTensorShape lTestTensor({{1, 2, 3}, {4, 5, 6}}, sizeof(math::vec3));
    /// @endcode
    ///
    struct sTensorShape
    {
        std::vector<std::vector<uint32_t>> mShape         = {}; //!< Shape
        std::vector<std::vector<uint32_t>> mStrides       = {}; //!< Strides
        uint32_t                           mRank          = 0;  //!< Dimension of each element in the shape array
        uint32_t                           mLayerCount    = 0;  //!< Number of layers
        uint32_t                           mElementSize   = 0;  //!< Size, in bytes, of each element in the tensor
        std::vector<uint32_t>              mMaxDimensions = {}; //!< Pointwise maximum of the elements in the shape vector
        uint32_t                           mMaxBufferSize = 0;  //!< Size, in bytes, of the largest tensor.
        size_t                             mByteSize      = 0;  //!< Size, in bytes, of the entire tensor
        std::vector<sBufferSizeInfo>       mBufferSizes = {}; //!< Size and offsets information of each layer in the tensor, in bytes.

        struct
        {
            MemoryBuffer mShape{};
            MemoryBuffer mMaxDimensions{};
            MemoryBuffer mBufferSizes{};
        } mDeviceSideData; //!< Data shared with GPU.

        sTensorShape()                       = default;
        sTensorShape( const sTensorShape & ) = default;

        ~sTensorShape() = default;

        /// @brief Constructs a tensor shape from the data provided.
        ///
        ///
        /// @param aShape       Vector of individual tensor dimensions.  All elements of `aShape` should have the same size.
        /// @param aElementSize Size, in bytes, of individual tensor elements.
        ///
        sTensorShape( std::vector<std::vector<uint32_t>> const &aShape, size_t aElementSize );

        /// @brief Constructs a tensor shape of rank 1 from the data provided.
        ///
        /// This is an overload provided for convenience. The passed-in shape will be converted to a vector of size one vectors
        /// and passed to the real constructor. Use this to build tensor shapes synamically.
        ///
        /// @param aShape       Vector of individual tensor dimensions.  All elements of `aShape` should have the same size.
        /// @param aElementSize Size, in bytes, of individual tensor elements.
        ///
        sTensorShape( std::vector<uint32_t> const &aShape, size_t aElementSize );

        /** @brief Returns the number of layers in the sTensorShape*/
        size_t CountLayers() const { return mLayerCount; }

        /// @brief Retrieves the dimension of the i-th layer of the sTensorShape
        std::vector<uint32_t> const &GetShapeForLayer( uint32_t i ) const
        {
            if( i >= CountLayers() )
                throw std::out_of_range(
                    fmt::format( "Attempted to access layer {}, but the stack only has {} layers", i + 1, CountLayers() ) );

            return mShape[i];
        }

        /// @brief Retrieves the stride of the i-th layer of the sTensorShape
        std::vector<uint32_t> const &GetStridesForLayer( uint32_t i ) const
        {
            if( i >= CountLayers() )
                throw std::out_of_range(
                    fmt::format( "Attempted to access layer {}, but the stack only has {} layers", i + 1, CountLayers() ) );

            return mStrides[i];
        }

        /// @brief Flatten the tensor shape up to a given dimension
        ///
        /// The dimension values up to but not including `aToDimension` are multiplied together and thus collapsed into a
        /// single dimension. The remaining dimensions are left alone. This operation does not change the buffer size. If
        /// `aToDimension` is negative, then it is interpreted as an index from the end of the dimension array. Tn this case,
        /// the final `aToDimension` values are left untouched, and the others are multiplied together and collapsed into a
        /// single dimension.
        ///
        /// @param aToDimension Index of dimensions to collapse
        ///
        void Flatten( int32_t aToDimension );

        /// @brief Trim the tensor shape up to a given dimension
        ///
        /// The dimension values up to but not including `aToDimension` are preservedm and the remaining dimensions are discarded.
        /// This operation does changes the buffer size. If `aToDimension` is negative, then it is interpreted as an index from
        /// the end of the dimension array. Tn this case, the final `aToDimension` values are preserved, and the others are discarded.
        ///
        /// @param aToDimension Index of dimensions to collapse
        ///
        void Trim( int32_t aToDimension );

        /// @brief Retrieves the vector of i-th dimensions of the sTensorShape
        ///
        /// If i >= 0, this is the ordinary i-th dimension. If i < 0, then we return the i-th dimension counted from
        /// the end of the shape vector.  For example, if the shape of a multi-tensor x is given by {{1, 2, 3}, {4, 5, 6}},
        /// then  x.GetDimension(-1) = {3, 6}, whereas x.GetDimension(0) = {1, 4};
        ///
        /// @param i       Position of the dimension to retrieve.
        ///
        std::vector<uint32_t> const GetDimension( int32_t i ) const;

        /// @brief Adds a new dimension at position `aPosition` to the sTensorShape
        ///
        /// If i >= 0, this is the ordinary i-th dimension. If i < 0, then we insert the new dimension at the i-th
        /// position counted from the end of the shape vector.  For example, if the shape of a multi-tensor x is given
        /// by {{1, 2, 3}, {4, 5, 6}}, then InsertDimension(2, {11, 11}) --> {{1, 2, 11, 3}, {4, 5, 11, 6}}, whereas
        /// InsertDimension(-3, {11, 11}) --> {{1, 11, 2, 3}, {4, 11, 5, 6}}
        ///
        /// @param aPosition  Position at which ti insert the new dimension
        /// @param aDimension New dimension vector to insert .
        ///
        void InsertDimension( int32_t aPosition, std::vector<uint32_t> aDimension );

        /// @brief Retrieves the size and offset, in bytes of the i-th layer of the sTensorShape
        sBufferSizeInfo const &GetBufferSize( uint32_t i ) const
        {
            if( i >= CountLayers() )
                throw std::out_of_range(
                    fmt::format( "Attempted to access layer {}, but the stack only has {} layers", i + 1, CountLayers() ) );
            return mBufferSizes[i];
        }

        /// @brief Retrieves the size and offset, of the i-th layer of the sTensorShape
        template <typename _Ty>
        LTSE_CUDA_INLINE LTSE_CUDA_DEVICE_FUNCTION_DEF sBufferSizeInfo GetBufferSizeAs( uint32_t i ) const
        {
#ifdef __CUDACC__
            auto lData = mDeviceSideData.mBufferSizes.DataAs<sBufferSizeInfo>()[i];
            return sBufferSizeInfo{
                lData.mSize / static_cast<uint32_t>( sizeof( _Ty ) ), lData.mOffset / static_cast<uint32_t>( sizeof( _Ty ) ) };
#else
            if( i >= CountLayers() )
                throw std::out_of_range(
                    fmt::format( "Attempted to access layer {}, but the stack only has {} layers", i + 1, CountLayers() ) );
            return sBufferSizeInfo{ mBufferSizes[i].mSize / static_cast<uint32_t>( sizeof( _Ty ) ),
                mBufferSizes[i].mOffset / static_cast<uint32_t>( sizeof( _Ty ) ) };
#endif
        }

        template <typename _AsType>
        LTSE_CUDA_INLINE LTSE_CUDA_DEVICE_FUNCTION_DEF bool InBounds( uint32_t aLayer, uint32_t i ) const
        {
#ifdef __CUDACC__
            auto lData = mDeviceSideData.mBufferSizes.DataAs<sBufferSizeInfo>()[aLayer];
            return ( i * sizeof( _AsType ) ) < lData.mSize;
#else
            if( aLayer >= CountLayers() )
                throw std::out_of_range(
                    fmt::format( "Attempted to access layer {}, but the stack only has {} layers", i + 1, CountLayers() ) );
            auto lData = mBufferSizes[aLayer];
            return ( i * sizeof( _AsType ) ) < lData.mSize;
#endif
        }

        /// @brief Retrieves the size and offset vectors
        std::vector<sBufferSizeInfo> GetTypedBufferSizes() const
        {
            std::vector<sBufferSizeInfo> lReturn( mBufferSizes.begin(), mBufferSizes.end() );
            for( auto &x : lReturn )
            {
                x.mSize /= mElementSize;
                x.mOffset /= mElementSize;
            }
            return lReturn;
        }

        bool operator!=( const sTensorShape &aRhs );
        bool operator==( const sTensorShape &aRhs );

        /// @brief Upload the dimension data to the GPU.
        ///
        /// This function should be called before passing a sTensorShape object to the GPU, if GPU-side functions are to
        /// make use of the data.
        ///
        void SyncDeviceData();

      private:
        void UpdateMetadata();
    };

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
    class MultiTensor
    {
      public:
        MultiTensor()  = default;
        ~MultiTensor() = default;

        /// @brief Allocates a generalized tensor of the given shape from a memory pool
        ///
        /// @param aMemoryPool The memory pool from which to allocate the tensor
        /// @param aShape      The shape of the tensor to allocate
        ///
        MultiTensor( MemoryPool &aMemoryPool, const sTensorShape &aShape );

        /// @brief Create a generalized tensor of the given shape using a preallocated buffer from a memory pool
        ///
        /// @param aMemoryPool The memory pool from which to allocate the tensor
        /// @param aMemoryBuffer Preallocated buffer to hold data
        /// @param aShape      The shape of the tensor to allocate
        ///
        MultiTensor( MemoryPool &aMemoryPool, MemoryBuffer &aMemoryBuffer, const sTensorShape &aShape );

        /// @brief Retrieves the shape of the tensor
        LTSE_CUDA_INLINE LTSE_CUDA_HOST_DEVICE_FUNCTION_DEF sTensorShape &Shape() { return mShape; }

        /// @brief Retrieve a pointer to the i-th layer
        ///
        /// @param i The index of the stack layer to retrieve
        ///
        /// @return  A MemoryBuffer pointing to the layer
        ///
        template <typename _Ty>
        LTSE_CUDA_INLINE LTSE_CUDA_DEVICE_FUNCTION_DEF _Ty *DeviceBufferAt( uint32_t i ) const
        {
            sBufferSizeInfo lBufferSize = mShape.GetBufferSizeAs<_Ty>( i );
            return DataAs<_Ty>() + lBufferSize.mOffset;
        }

        /// @brief Retrieve a view into the i-th layer
        ///
        /// @param i The index of the stack layer to retrieve
        ///
        /// @return  A MemoryBuffer pointing to the layer
        ///
        MemoryBuffer BufferAt( uint32_t i ) const
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
        std::vector<_Tx> FetchBufferAt( uint32_t i ) const
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
        std::vector<_Tx> FetchFlattened() const
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
        void Upload( std::vector<_Tx> const &aArray ) const
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
        void Upload( std::vector<_Tx> const &aArray, uint32_t aLayer, uint32_t aOffset ) const
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
        void Upload( std::vector<_Tx> const &aArray, uint32_t aLayer ) const
        {
            Upload( aArray, aLayer, 0 );
        }

        /// @brief Size, in bytes, of the tensor.
        LTSE_CUDA_INLINE LTSE_CUDA_HOST_DEVICE_FUNCTION_DEF size_t Size() const { return mMemoryBuffer.Size(); }

        /// @brief Size of the tensor as elements of type `_Ty`.
        template <typename _Tx>
        LTSE_CUDA_INLINE LTSE_CUDA_HOST_DEVICE_FUNCTION_DEF size_t SizeAs() const
        {
            return mMemoryBuffer.SizeAs<_Tx>();
        }

        /// @brief Pointer to the underlying data as type `_Ty`.
        template <typename _Tx>
        LTSE_CUDA_INLINE LTSE_CUDA_HOST_DEVICE_FUNCTION_DEF _Tx *DataAs() const
        {
            return mMemoryBuffer.DataAs<_Tx>();
        }

        MemoryBuffer &GetMemoryBuffer() { return mMemoryBuffer; }

      private:
        sTensorShape mShape{};        //!< Shape of the tensor
        MemoryBuffer mMemoryBuffer{}; //!< Memory buffer assigned to the tensor
    };

} // namespace SE::Cuda
