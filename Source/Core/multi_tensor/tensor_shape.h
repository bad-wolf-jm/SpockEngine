#pragma once

#include <algorithm>
#include <numeric>

#include "Core/CUDA/Cuda.h"
#include "Core/CUDA/Array/MemoryPool.h"

namespace SE::Core
{

    /// @brief Buffer offset structure
    struct buffer_size_info_t
    {
        uint32_t Size   = 0; //!< Size of current buffer
        uint32_t Offset = 0; //!< Offset of current buffer

        buffer_size_info_t()                             = default;
        buffer_size_info_t( const buffer_size_info_t & ) = default;
    };

    bool operator==( const buffer_size_info_t &lhs, const buffer_size_info_t &rhs );

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
    struct tensor_shape_t
    {
        vector_t<vector_t<uint32_t>> Shape         = {}; //!< Shape
        vector_t<vector_t<uint32_t>> Strides       = {}; //!< Strides
        uint32_t                     Rank          = 0;  //!< Dimension of each element in the shape array
        uint32_t                     LayerCount    = 0;  //!< Number of layers
        uint32_t                     ElementSize   = 0;  //!< Size, in bytes, of each element in the tensor
        vector_t<uint32_t>           MaxDimensions = {}; //!< Pointwise maximum of the elements in the shape vector
        uint32_t                     MaxBufferSize = 0;  //!< Size, in bytes, of the largest tensor.
        size_t                       ByteSize      = 0;  //!< Size, in bytes, of the entire tensor
        vector_t<buffer_size_info_t> BufferSizes   = {}; //!< Size and offsets information of each layer in the tensor, in bytes.

        struct
        {
            Cuda::memory_buffer_t Shape{};
            Cuda::memory_buffer_t MaxDimensions{};
            Cuda::memory_buffer_t BufferSizes{};
        } DeviceSideData; //!< Data shared with GPU.

        tensor_shape_t()                         = default;
        tensor_shape_t( const tensor_shape_t & ) = default;

        ~tensor_shape_t() = default;

        /// @brief Constructs a tensor shape from the data provided.
        ///
        ///
        /// @param aShape       Vector of individual tensor dimensions.  All elements of `aShape` should have the same size.
        /// @param aElementSize Size, in bytes, of individual tensor elements.
        ///
        tensor_shape_t( vector_t<vector_t<uint32_t>> const &aShape, size_t aElementSize );

        /// @brief Constructs a tensor shape of rank 1 from the data provided.
        ///
        /// This is an overload provided for convenience. The passed-in shape will be converted to a vector of size one vectors
        /// and passed to the real constructor. Use this to build tensor shapes synamically.
        ///
        /// @param aShape       Vector of individual tensor dimensions.  All elements of `aShape` should have the same size.
        /// @param aElementSize Size, in bytes, of individual tensor elements.
        ///
        tensor_shape_t( vector_t<uint32_t> const &aShape, size_t aElementSize );

        /** @brief Returns the number of layers in the sTensorShape*/
        size_t CountLayers() const
        {
            return LayerCount;
        }

        /// @brief Retrieves the dimension of the i-th layer of the sTensorShape
        vector_t<uint32_t> const &GetShapeForLayer( uint32_t i ) const
        {
            if( i >= CountLayers() )
                throw std::out_of_range(
                    fmt::format( "Attempted to access layer {}, but the stack only has {} layers", i + 1, CountLayers() ) );

            return Shape[i];
        }

        /// @brief Retrieves the stride of the i-th layer of the sTensorShape
        vector_t<uint32_t> const &GetStridesForLayer( uint32_t i ) const
        {
            if( i >= CountLayers() )
                throw std::out_of_range(
                    fmt::format( "Attempted to access layer {}, but the stack only has {} layers", i + 1, CountLayers() ) );

            return Strides[i];
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
        vector_t<uint32_t> const GetDimension( int32_t i ) const;

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
        void InsertDimension( int32_t aPosition, vector_t<uint32_t> aDimension );

        /// @brief Retrieves the size and offset, in bytes of the i-th layer of the sTensorShape
        buffer_size_info_t const &GetBufferSize( uint32_t i ) const
        {
            if( i >= CountLayers() )
                throw std::out_of_range(
                    fmt::format( "Attempted to access layer {}, but the stack only has {} layers", i + 1, CountLayers() ) );
            return BufferSizes[i];
        }

        /// @brief Retrieves the size and offset, of the i-th layer of the sTensorShape
        template <typename _Ty>
        SE_CUDA_INLINE SE_CUDA_DEVICE_FUNCTION_DEF buffer_size_info_t GetBufferSizeAs( uint32_t i ) const
        {
#ifdef __CUDACC__
            auto lData = DeviceSideData.BufferSizes.DataAs<buffer_size_info_t>()[i];
            return buffer_size_info_t{ lData.Size / static_cast<uint32_t>( sizeof( _Ty ) ),
                                       lData.Offset / static_cast<uint32_t>( sizeof( _Ty ) ) };
#else
            if( i >= CountLayers() )
                throw std::out_of_range(
                    fmt::format( "Attempted to access layer {}, but the stack only has {} layers", i + 1, CountLayers() ) );
            return buffer_size_info_t{ BufferSizes[i].Size / static_cast<uint32_t>( sizeof( _Ty ) ),
                                       BufferSizes[i].Offset / static_cast<uint32_t>( sizeof( _Ty ) ) };
#endif
        }

        template <typename _AsType>
        SE_CUDA_INLINE SE_CUDA_DEVICE_FUNCTION_DEF bool InBounds( uint32_t aLayer, uint32_t i ) const
        {
#ifdef __CUDACC__
            auto lData = DeviceSideData.BufferSizes.DataAs<buffer_size_info_t>()[aLayer];
            return ( i * sizeof( _AsType ) ) < lData.Size;
#else
            if( aLayer >= CountLayers() )
                throw std::out_of_range(
                    fmt::format( "Attempted to access layer {}, but the stack only has {} layers", i + 1, CountLayers() ) );
            auto lData = mBufferSizes[aLayer];
            return ( i * sizeof( _AsType ) ) < lData.mSize;
#endif
        }

        /// @brief Retrieves the size and offset vectors
        vector_t<buffer_size_info_t> GetTypedBufferSizes() const
        {
            vector_t<buffer_size_info_t> lReturn( BufferSizes.begin(), BufferSizes.end() );
            for( auto &x : lReturn )
            {
                x.Size /= ElementSize;
                x.Offset /= ElementSize;
            }
            return lReturn;
        }

        bool operator!=( const tensor_shape_t &aRhs );
        bool operator==( const tensor_shape_t &aRhs );

        /// @brief Upload the dimension data to the GPU.
        ///
        /// This function should be called before passing a sTensorShape object to the GPU, if GPU-side functions are to
        /// make use of the data.
        ///
        void SyncDeviceData();

      private:
        void UpdateMetadata();
    };
} // namespace SE::Core