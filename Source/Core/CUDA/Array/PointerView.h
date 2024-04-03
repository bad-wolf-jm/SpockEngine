/// @file   PointerView.h
///
/// @brief  Wrapper class for cuda pointers
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert All rights reserved.

#pragma once

#include "Core/CUDA/Cuda.h"
#include <vector>

#include "Core/Definitions.h"

/** \namespace SE::Cuda::Internal
 *
 * @brief Internal definition
 *
 */
namespace numlua::cuda::Internal
{
    using namespace numlua::core;

    /// @struct sGPUDevicePointerView
    ///
    /// @brief Simple wrapper around a RawPointer
    ///
    /// Implements a thin abstraction layer around a raw @code{.cpp} RawPointer @endcode.
    /// The purpose of this class is to wrap existing Cuda device pointers, and the size of
    /// the memory allocated to them. All sizes are in bytes, and we provide a set of templated
    /// functions to transfer data between the host to the GPU using standard library containers.
    /// @code{.cpp} sGPUDevicePointerView @endcode does not perform any memory management.
    ///
    /// This class can be passed directly to Cuda kernels. However, it should be constructed and
    /// initialized on the host side.
    ///
    struct gpu_device_pointer_view_t
    {
        raw_pointer_t DevicePointer = 0; //!< Pointer to an area of GPU memory

        /// @brief Default constructor
        gpu_device_pointer_view_t() = default;

        /// @brief Copy constructor
        gpu_device_pointer_view_t( const gpu_device_pointer_view_t & ) = default;

        /// @brief View on the a portion of the buffer.
        ///
        /// This opens a view on an initial part of the buffer represented by `parent`. Note that opening a view
        /// that is larger than the original buffer will result in a runtime error.
        ///
        /// @exception  std::runtime_error the parent pointer is not large enough to accomodate the view
        ///
        /// @param size   Number of elements of the parent buffer to be included in the view.
        /// @param offset   Offset into the buffer where the view should start, in bytes.
        /// @param parent Reference to the underlying device pointer handle.
        ///
        gpu_device_pointer_view_t( size_t size, size_t offset, gpu_device_pointer_view_t const &parent )
            : _size{ size }
        {
            if( ( size + offset ) > parent.Size() )
                throw std::runtime_error(
                    fmt::format( "View upper boundary (offset) + (size) = ({}) + ({}) is greater than parent buffer boundary ({})",
                                 offset, size, parent.Size() ) );

            DevicePointer = ( parent.DevicePointer + offset );
        }

        /// @brief Wrap a non-owning memory view around an already existing CUDA pointer.
        ///
        /// This constructor wraps an already allocated CUDA buffer into a pointer view structure. Note that
        /// `parent` should point to an area at least @code{.cpp} size @endcode bytes in size
        ///
        /// @param size The presumed size of the memory buffer pointed to by `parent`, in bytes.
        /// @param parent Reference ot the parent buffer.
        ///
        gpu_device_pointer_view_t( size_t size, gpu_device_pointer_view_t const &parent )
            : gpu_device_pointer_view_t( size, 0, parent )
        {
        }

        /// @brief Wrap a non-owning memory view around an already existing CUDA pointer.
        ///
        /// This constructor wraps an already allocated CUDA buffer into a pointer view structure. Note that
        /// `devicePointer` should point to an area at least @code{.cpp} size @endcode bytes in size
        ///
        /// @param size The presumed size of the memory buffer pointed to by `devicePointer`, in bytes.
        /// @param devicePointer An already allocated pointer to device memory.
        ///
        gpu_device_pointer_view_t( size_t size, void *devicePointer )
            : _size{ size }
        {
            DevicePointer = (raw_pointer_t)devicePointer;
        }

        /// @brief Upload data to the device at a given offset.
        ///
        /// Uploads the contents of a vector of type `_Ty` to the device. The size of `array`, in bytes, should be less
        /// than the size of the underlying device buffer, or a runtime error will be raised. Nothing happens to the device
        /// data beyond `array.size()` if `array.size()` is less than the size of the buffer.
        ///
        /// @exception  std::runtime_error If trying to upload more data than there is space available
        ///
        /// @param array Array of data to upload to the device
        /// @param offset The offset at which to copy the array.
        ///
        template <typename _Ty>
        void Upload( vector_t<_Ty> &array, uint32_t offset ) const
        {
            if( ( array.size() + offset ) * sizeof( _Ty ) > _size )
                throw std::runtime_error(
                    fmt::format( "Upload upper boundary (offset) + (size) = ({}) + ({}) is greater than parent buffer boundary ({})",
                                 offset, array.size(), _size / sizeof( _Ty ) )
                        .c_str() );

            MemCopyHostToDevice( (void *)( DataAs<_Ty>() + offset ), (void *)array.data(), array.size() * sizeof( _Ty ) );
        }

        template <typename _Ty>
        void Upload( vector_t<_Ty> const &array, uint32_t offset ) const
        {
            if( ( array.size() + offset ) * sizeof( _Ty ) > _size )
                throw std::runtime_error(
                    fmt::format( "Upload upper boundary (offset) + (size) = ({}) + ({}) is greater than parent buffer boundary ({})",
                                 offset, array.size(), _size / sizeof( _Ty ) )
                        .c_str() );

            MemCopyHostToDevice( (void *)( DataAs<_Ty>() + offset ), (void *)array.data(), array.size() * sizeof( _Ty ) );
        }

        /// @brief Overloaded member provided for convenience
        ///
        /// Uploads the contents of the vector passed as parameter to the GPU with offset 0. This method has
        /// 4 overloads which can't seem to be avoided.
        ///
        /// @param array Array of data to upload to the device
        ///
        template <typename _Ty>
        void Upload( vector_t<_Ty> &array )
        {
            Upload<_Ty>( array, 0 );
        }
        template <typename _Ty>
        void Upload( vector_t<_Ty> &array ) const
        {
            Upload<_Ty>( array, 0 );
        }
        template <typename _Ty>
        void Upload( vector_t<_Ty> const &array )
        {
            Upload<_Ty>( array, 0 );
        }
        template <typename _Ty>
        void Upload( vector_t<_Ty> const &array ) const
        {
            Upload<_Ty>( array, 0 );
        }

        /// @brief Upload data to the device.
        ///
        /// Uploads the contents of a raw byte buffer to the GPU. The value of `byteSize`, should be less than
        /// the size of the underlying device buffer, or a runtime error will be raised. Nothing happens to the
        /// device data beyond `byteSize` if `byteSize` is less than the size of the buffer.
        ///
        /// @exception  std::runtime_error If trying to upload more data than there is space available
        ///
        /// @param data     Pointer to a buffer to upload to the GPU
        /// @param byteSize Size of the byffer pointed to by `data`, in bytes
        /// @param offset   Offset at which to upload the data
        ///
        void Upload( const uint8_t *data, size_t byteSize, size_t offset ) const
        {
            if( byteSize + offset > Size() )
                throw std::runtime_error(
                    fmt::format( "Upload upper boundary (offset) + (size) = ({}) + ({}) is greater than parent buffer boundary ({})",
                                 offset, byteSize, Size() )
                        .c_str() );

            MemCopyHostToDevice( (void *)( DataAs<uint8_t>() + offset ), (void *)data, byteSize );
        }

        /// @brief Overloaded member provided for convenience
        ///
        /// @param data     Pointer to the data to uploac
        /// @param byteSize Size of the byffer pointed to by `data`, in bytes
        ///
        void Upload( const uint8_t *data, size_t byteSize ) const
        {
            Upload( data, byteSize, 0 );
        }

        /// @brief Overloaded member provided for convenience
        ///
        /// @param data Element to upload
        ///
        template <typename _Ty>
        void Upload( _Ty &element ) const
        {
            Upload( reinterpret_cast<const uint8_t *>( &element ), sizeof( _Ty ) );
        }

        /// @brief Overloaded member provided for convenience
        ///
        /// @param data   Element to upload
        /// @param offset Position at which to upload the element
        ///
        template <typename _Ty>
        void Upload( _Ty &element, uint32_t offset ) const
        {
            Upload( reinterpret_cast<const uint8_t *>( &element ), sizeof( _Ty ), offset * sizeof( _Ty ) );
        }

        /// @brief Downloads data from the device.
        ///
        /// Downloads the contents of the device buffer into a newly allocated `vector_t` appropriate size and type.
        ///
        /// @exception  std::runtime_error If trying to fetch more data than there is space available
        ///
        /// @param offset Where the fetch starts
        /// @param size   Size of the buffer to fetch, in bytes
        ///
        /// @return newly allocated `vector_t` containing the data.
        ///
        template <typename _Ty>
        vector_t<_Ty> Fetch( size_t offset, size_t size ) const
        {
            if( ( size + offset ) * sizeof( _Ty ) > Size() )
                throw std::runtime_error(
                    fmt::format( "Attempted to fetch an array of size {} from a buffer of size {}", size, Size() ).c_str() );
            vector_t<_Ty> lHostArray( size );
            MemCopyDeviceToHost( reinterpret_cast<void *>( lHostArray.data() ), reinterpret_cast<void *>( DataAs<_Ty>() + offset ),
                                 size * sizeof( _Ty ) );
            return lHostArray;
        }

        /// @brief Overloaded member provided for convenience
        ///
        /// Retrieve the entire buffer into a newly allocated vector
        ///
        /// @exception  std::runtime_error If trying to fetch more data than there is space available
        ///
        /// @return newly allocated `vector_t` containing the data.
        ///
        template <typename _Ty>
        vector_t<_Ty> Fetch() const
        {
            return Fetch<_Ty>( _size / sizeof( _Ty ) );
        }

        /// @brief Overloaded member provided for convenience
        ///
        /// Retrieve an initial segment of the buffer into a newly allocated vector.
        ///
        /// @exception  std::runtime_error If trying to fetch more data than there is space available
        ///
        /// @param size   Size of the buffer to fetch, in bytes
        ///
        /// @return newly allocated `vector_t` containing the data.
        ///
        template <typename _Ty>
        vector_t<_Ty> Fetch( size_t size ) const
        {
            return Fetch<_Ty>( 0, size );
        }

        /// @brief Set the content of the buffer to 0
        ///
        /// This is roughly equivalent to @code{.cpp} cudaMemset(ptr, 0, this->size()); @endcode As of now there is
        /// no semantic initialization of elements of type `T`.
        ///
        void Zero() const
        {
            CUDA_ASSERT( cudaMemset( (void *)DevicePointer, 0, _size ) );
        }

        /// @brief Size of the allocated buffer, in bytes.
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF size_t Size() const
        {
            return _size;
        }

        /// @brief Size of the allocated buffer, in elements of type `_Ty`.
        template <typename _Ty>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF size_t SizeAs() const
        {
            return _size / sizeof( _Ty );
        }

        /// @brief Return the underlying device pointer as a pointer to an array of type `_Ty`.
        template <typename _Ty>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF _Ty *DataAs() const
        {
            return (_Ty *)DevicePointer;
        }

        /// @brief Number of elements in the buffer.
        raw_pointer_t RawDevicePtr() const
        {
            return DevicePointer;
        }

      protected:
        size_t _size = 0;

        /** @brief Trivial constructor. The device pointer member should be set in a subclass */
        gpu_device_pointer_view_t( size_t size )
            : _size{ size }
        {
        }
    };

    /// @struct sGPUDevicePointer
    ///
    /// @brief Simple wrapper around a RawPointer which can allocate memory on the device.
    ///
    struct gpu_device_pointer_t : public gpu_device_pointer_view_t
    {
        gpu_device_pointer_t()                               = default;
        gpu_device_pointer_t( const gpu_device_pointer_t & ) = default;

        gpu_device_pointer_t( size_t size )
        {
            _size = size;
            CUDA_ASSERT( cudaMalloc( (void **)&DevicePointer, size ) );
        }

        ~gpu_device_pointer_t() = default;

        /// @brief Free the allocated memory.
        void Dispose()
        {
            if( DevicePointer != 0 )
                CUDA_ASSERT( cudaFree( (void *)DevicePointer ) );

            DevicePointer = 0;
        }

        void Resize( uint32_t newSize )
        {
            Dispose();

            _size = newSize;
            CUDA_ASSERT( cudaMalloc( (void **)&DevicePointer, _size ) );
        }
    };

} // namespace SE::Cuda::Internal