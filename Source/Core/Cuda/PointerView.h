/// @file   PointerView.h
///
/// @brief  Wrapper class for cuda pointers
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>

#include "CudaAssert.h"

/** \namespace SE::Cuda::Internal
 *
 * @brief Internal definition
 *
 */
namespace SE::Cuda::Internal
{

    /// @struct sGPUDevicePointerView
    ///
    /// @brief Simple wrapper around a CUdeviceptr
    ///
    /// Implements a thin abstraction layer around a raw @code{.cpp} CUdeviceptr @endcode.
    /// The purpose of this class is to wrap existing Cuda device pointers, and the size of
    /// the memory allocated to them. All sizes are in bytes, and we provide a set of templated
    /// functions to transfer data between the host to the GPU using standard library containers.
    /// @code{.cpp} sGPUDevicePointerView @endcode does not perform any memory management.
    ///
    /// This class can be passed directly to Cuda kernels. However, it should be constructed and
    /// initialized on the host side.
    ///
    struct sGPUDevicePointerView
    {
        CUdeviceptr mDevicePointer = 0; //!< Pointer to an area of GPU memory

        /// @brief Default constructor
        sGPUDevicePointerView() = default;

        /// @brief Copy constructor
        sGPUDevicePointerView( const sGPUDevicePointerView & ) = default;

        /// @brief View on the a portion of the buffer.
        ///
        /// This opens a view on an initial part of the buffer represented by `aParent`. Note that opening a view
        /// that is larger than the original buffer will result in a runtime error.
        ///
        /// @exception  std::runtime_error the parent pointer is not large enough to accomodate the view
        ///
        /// @param aSize   Number of elements of the parent buffer to be included in the view.
        /// @param aOffset   Offset into the buffer where the view should start, in bytes.
        /// @param aParent Reference to the underlying device pointer handle.
        ///
        sGPUDevicePointerView( size_t aSize, size_t aOffset, sGPUDevicePointerView const &aParent )
            : mSize{ aSize }
        {
            if( ( aSize + aOffset ) > aParent.Size() )
                throw std::runtime_error(
                    fmt::format( "View upper boundary (offset) + (size) = ({}) + ({}) is greater than parent buffer boundary ({})",
                        aOffset, aSize, aParent.Size() ) );

            mDevicePointer = ( aParent.mDevicePointer + aOffset );
        }

        /// @brief Wrap a non-owning memory view around an already existing CUDA pointer.
        ///
        /// This constructor wraps an already allocated CUDA buffer into a pointer view structure. Note that
        /// `aParent` should point to an area at least @code{.cpp} aSize @endcode bytes in size
        ///
        /// @param aSize The presumed size of the memory buffer pointed to by `aParent`, in bytes.
        /// @param aParent Reference ot the parent buffer.
        ///
        sGPUDevicePointerView( size_t aSize, sGPUDevicePointerView const &aParent )
            : sGPUDevicePointerView( aSize, 0, aParent )
        {
        }

        /// @brief Wrap a non-owning memory view around an already existing CUDA pointer.
        ///
        /// This constructor wraps an already allocated CUDA buffer into a pointer view structure. Note that
        /// `aDevicePointer` should point to an area at least @code{.cpp} aSize @endcode bytes in size
        ///
        /// @param aSize The presumed size of the memory buffer pointed to by `aDevicePointer`, in bytes.
        /// @param aDevicePointer An already allocated pointer to device memory.
        ///
        sGPUDevicePointerView( size_t aSize, void *aDevicePointer )
            : mSize{ aSize }
        {
            mDevicePointer = (CUdeviceptr)aDevicePointer;
        }

        /// @brief Upload data to the device at a given offset.
        ///
        /// Uploads the contents of a vector of type `_Ty` to the device. The size of `aArray`, in bytes, should be less
        /// than the size of the underlying device buffer, or a runtime error will be raised. Nothing happens to the device
        /// data beyond `aArray.size()` if `aArray.size()` is less than the size of the buffer.
        ///
        /// @exception  std::runtime_error If trying to upload more data than there is space available
        ///
        /// @param aArray Array of data to upload to the device
        /// @param aOffset The offset at which to copy the array.
        ///
        template <typename _Ty>
        void Upload( std::vector<_Ty> &aArray, uint32_t aOffset ) const
        {
            if( ( aArray.size() + aOffset ) * sizeof( _Ty ) > mSize )
                throw std::runtime_error(
                    fmt::format( "Upload upper boundary (offset) + (size) = ({}) + ({}) is greater than parent buffer boundary ({})",
                        aOffset, aArray.size(), mSize / sizeof( _Ty ) )
                        .c_str() );

            CUDA_ASSERT( cudaMemcpy(
                (void *)( DataAs<_Ty>() + aOffset ), (void *)aArray.data(), aArray.size() * sizeof( _Ty ), cudaMemcpyHostToDevice ) );
        }

        template <typename _Ty>
        void Upload( std::vector<_Ty> const &aArray, uint32_t aOffset ) const
        {
            if( ( aArray.size() + aOffset ) * sizeof( _Ty ) > mSize )
                throw std::runtime_error(
                    fmt::format( "Upload upper boundary (offset) + (size) = ({}) + ({}) is greater than parent buffer boundary ({})",
                        aOffset, aArray.size(), mSize / sizeof( _Ty ) )
                        .c_str() );

            CUDA_ASSERT( cudaMemcpy(
                (void *)( DataAs<_Ty>() + aOffset ), (void *)aArray.data(), aArray.size() * sizeof( _Ty ), cudaMemcpyHostToDevice ) );
        }

        /// @brief Overloaded member provided for convenience
        ///
        /// Uploads the contents of the vector passed as parameter to the GPU with offset 0. This method has
        /// 4 overloads which can't seem to be avoided.
        ///
        /// @param aArray Array of data to upload to the device
        ///
        template <typename _Ty>
        void Upload( std::vector<_Ty> &aArray )
        {
            Upload<_Ty>( aArray, 0 );
        }
        template <typename _Ty>
        void Upload( std::vector<_Ty> &aArray ) const
        {
            Upload<_Ty>( aArray, 0 );
        }
        template <typename _Ty>
        void Upload( std::vector<_Ty> const &aArray )
        {
            Upload<_Ty>( aArray, 0 );
        }
        template <typename _Ty>
        void Upload( std::vector<_Ty> const &aArray ) const
        {
            Upload<_Ty>( aArray, 0 );
        }

        /// @brief Upload data to the device.
        ///
        /// Uploads the contents of a raw byte buffer to the GPU. The value of `aByteSize`, should be less than
        /// the size of the underlying device buffer, or a runtime error will be raised. Nothing happens to the
        /// device data beyond `aByteSize` if `aByteSize` is less than the size of the buffer.
        ///
        /// @exception  std::runtime_error If trying to upload more data than there is space available
        ///
        /// @param aData     Pointer to a buffer to upload to the GPU
        /// @param aByteSize Size of the byffer pointed to by `aData`, in bytes
        /// @param aOffset   Offset at which to upload the data
        ///
        void Upload( const uint8_t *aData, size_t aByteSize, size_t aOffset ) const
        {
            if( aByteSize + aOffset > Size() )
                throw std::runtime_error(
                    fmt::format( "Upload upper boundary (offset) + (size) = ({}) + ({}) is greater than parent buffer boundary ({})",
                        aOffset, aByteSize, Size() )
                        .c_str() );
            CUDA_ASSERT( cudaMemcpy( (void *)( DataAs<uint8_t>() + aOffset ), aData, aByteSize, cudaMemcpyHostToDevice ) );
        }

        /// @brief Overloaded member provided for convenience
        ///
        /// @param aData     Pointer to the data to uploac
        /// @param aByteSize Size of the byffer pointed to by `aData`, in bytes
        ///
        void Upload( const uint8_t *aData, size_t aByteSize ) const { Upload( aData, aByteSize, 0 ); }

        /// @brief Overloaded member provided for convenience
        ///
        /// @param aData Element to upload
        ///
        template <typename _Ty>
        void Upload( _Ty &aElement ) const
        {
            Upload( reinterpret_cast<const uint8_t *>( &aElement ), sizeof( _Ty ) );
        }

        /// @brief Overloaded member provided for convenience
        ///
        /// @param aData   Element to upload
        /// @param aOffset Position at which to upload the element
        ///
        template <typename _Ty>
        void Upload( _Ty &aElement, uint32_t aOffset ) const
        {
            Upload( reinterpret_cast<const uint8_t *>( &aElement ), sizeof( _Ty ), aOffset * sizeof( _Ty ) );
        }

        /// @brief Downloads data from the device.
        ///
        /// Downloads the contents of the device buffer into a newly allocated `std::vector` appropriate size and type.
        ///
        /// @exception  std::runtime_error If trying to fetch more data than there is space available
        ///
        /// @param aOffset Where the fetch starts
        /// @param aSize   Size of the buffer to fetch, in bytes
        ///
        /// @return newly allocated `std::vector` containing the data.
        ///
        template <typename _Ty>
        std::vector<_Ty> Fetch( size_t aOffset, size_t aSize ) const
        {
            if( ( aSize + aOffset ) * sizeof( _Ty ) > Size() )
                throw std::runtime_error(
                    fmt::format( "Attempted to fetch an array of size {} from a buffer of size {}", aSize, Size() ).c_str() );
            std::vector<_Ty> lHostArray( aSize );
            CUDA_ASSERT( cudaMemcpy( reinterpret_cast<void *>( lHostArray.data() ),
                reinterpret_cast<void *>( DataAs<_Ty>() + aOffset ), aSize * sizeof( _Ty ), cudaMemcpyDeviceToHost ) );
            return lHostArray;
        }

        /// @brief Overloaded member provided for convenience
        ///
        /// Retrieve the entire buffer into a newly allocated vector
        ///
        /// @exception  std::runtime_error If trying to fetch more data than there is space available
        ///
        /// @return newly allocated `std::vector` containing the data.
        ///
        template <typename _Ty>
        std::vector<_Ty> Fetch() const
        {
            return Fetch<_Ty>( mSize / sizeof( _Ty ) );
        }

        /// @brief Overloaded member provided for convenience
        ///
        /// Retrieve an initial segment of the buffer into a newly allocated vector.
        ///
        /// @exception  std::runtime_error If trying to fetch more data than there is space available
        ///
        /// @param aSize   Size of the buffer to fetch, in bytes
        ///
        /// @return newly allocated `std::vector` containing the data.
        ///
        template <typename _Ty>
        std::vector<_Ty> Fetch( size_t aSize ) const
        {
            return Fetch<_Ty>( 0, aSize );
        }

        /// @brief Set the content of the buffer to 0
        ///
        /// This is roughly equivalent to @code{.cpp} cudaMemset(ptr, 0, this->size()); @endcode As of now there is
        /// no semantic initialization of elements of type `T`.
        ///
        void Zero() const { CUDA_ASSERT( cudaMemset( (void *)mDevicePointer, 0, mSize ) ); }

        /// @brief Size of the allocated buffer, in bytes.
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF size_t Size() const { return mSize; }

        /// @brief Size of the allocated buffer, in elements of type `_Ty`.
        template <typename _Ty>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF size_t SizeAs() const
        {
            return mSize / sizeof( _Ty );
        }

        /// @brief Return the underlying device pointer as a pointer to an array of type `_Ty`.
        template <typename _Ty>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF _Ty *DataAs() const
        {
            return (_Ty *)mDevicePointer;
        }

        /// @brief Number of elements in the buffer.
        CUdeviceptr RawDevicePtr() const { return mDevicePointer; }

      protected:
        size_t mSize = 0;

        /** @brief Trivial constructor. The device pointer member should be set in a subclass */
        sGPUDevicePointerView( size_t aSize )
            : mSize{ aSize }
        {
        }
    };

    /// @struct sGPUDevicePointer
    ///
    /// @brief Simple wrapper around a CUdeviceptr which can allocate memory on the device.
    ///
    struct sGPUDevicePointer : public sGPUDevicePointerView
    {
        sGPUDevicePointer()                            = default;
        sGPUDevicePointer( const sGPUDevicePointer & ) = default;

        sGPUDevicePointer( size_t aSize )
        {
            mSize = aSize;
            CUDA_ASSERT( cudaMalloc( (void **)&mDevicePointer, aSize ) );
        }

        ~sGPUDevicePointer() = default;

        /// @brief Free the allocated memory.
        void Dispose()
        {
            if( mDevicePointer != 0 ) CUDA_ASSERT( cudaFree( (void *)mDevicePointer ) );
            mDevicePointer = 0;
        }
    };

} // namespace SE::Cuda::Internal