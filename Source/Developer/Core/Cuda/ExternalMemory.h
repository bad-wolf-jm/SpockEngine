/** @file */

#pragma once

#include <cuda.h>
#include <vulkan/vulkan.h>
#ifdef APIENTRY
#    undef APIENTRY
#endif

// clang-format off
#include <windows.h>
#include <vulkan/vulkan_win32.h>
// clang-format on

#include "Cuda/CudaAssert.h"
#include "Cuda/CudaBuffer.h"

#include "Developer/GraphicContext/Buffer.h"

namespace LTSE::Cuda
{

    /** @brief Map a graphic resource for use by CUDA
     *
     */
    class GPUExternalMemory
    {
      public:
        GPUExternalMemory() = default;

        /** @brief Constructor
         *
         * Registers the graphics buffer `aBuffer` for use with CUDA. The registered buffer
         * is then mapped and its corresponding device pointer is saved as per RAII.
         */
        GPUExternalMemory( LTSE::Graphics::Buffer &aBuffer, size_t aSize, size_t aOffset = 0 )
            : mSize{ aSize }
        {
            cudaExternalMemoryHandleDesc lExternalMemoryHandleDesc{};
            lExternalMemoryHandleDesc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
            lExternalMemoryHandleDesc.size                = mSize;
            lExternalMemoryHandleDesc.handle.win32.handle = (HANDLE)aBuffer.GetMemoryHandle();
            CUDA_ASSERT( cudaImportExternalMemory( &mExternalMemoryHandle, &lExternalMemoryHandleDesc ) );

            cudaExternalMemoryBufferDesc lExternalMemBufferDesc{};
            lExternalMemBufferDesc.offset = aOffset;
            lExternalMemBufferDesc.flags  = 0;
            lExternalMemBufferDesc.size   = mSize;
            CUDA_ASSERT( cudaExternalMemoryGetMappedBuffer( &mDevicePointer, mExternalMemoryHandle, &lExternalMemBufferDesc ) );
        }

        /** @brief Destructor
         *
         * Unmaps and unregisters the CUDA buffer. The underlying graphics buffer is left
         * untouched, and can be used by shaders.
         */
        ~GPUExternalMemory() = default;

        void Dispose()
        {
            if( mDevicePointer != nullptr )
                CUDA_ASSERT( cudaFree( mDevicePointer ) );
            mDevicePointer = nullptr;

            if( mExternalMemoryHandle )
                CUDA_ASSERT( cudaDestroyExternalMemory( mExternalMemoryHandle ) );
            mExternalMemoryHandle = 0;
        }

        /** @brief Retrieve the device pointer */
        template <typename _Ty> LTSE_CUDA_INLINE LTSE_CUDA_HOST_DEVICE_FUNCTION_DEF _Ty *DataAs() { return (_Ty *)mDevicePointer; }

        template <typename _Ty> LTSE_CUDA_INLINE LTSE_CUDA_HOST_DEVICE_FUNCTION_DEF size_t SizeAs() { return mSize / sizeof( _Ty ); }

        /** @brief Upload data to the device.
         *
         * Uploads the contents of a vector of type `_Ty` to the device. The size of `aArray` should
         * be less than the size of the underlying device buffer, or a runtime error will be raised.
         * Nothing happens to the device data beyond `aArray.size()` if `aArray.size()` is less
         * than the size of the buffer.
         *
         * @param aArray Array of data to upload to the device
         */
        template <typename _Ty> void Upload( std::vector<_Ty> aArray )
        {
            if( aArray.size() > mSize )
                std::runtime_error( fmt::format( "Attemp to copy an array of size {} into a buffer if size {}", aArray.size(), mSize ).c_str() );
            CUDA_ASSERT( cudaMemcpy( (void *)mDevicePointer, aArray.data(), aArray.size() * sizeof( _Ty ), cudaMemcpyHostToDevice ) );
        }

        /** @brief Set the content of the buffer to 0 */
        void Zero() { CUDA_ASSERT( cudaMemset( (void *)mDevicePointer, 0, mSize ) ); }

        /** @brief Set the content of the buffer to 0 */
        void CopyFrom( GPUMemory aFromBuffer )
        {
            CUDA_ASSERT( cudaMemcpy( (void *)mDevicePointer, aFromBuffer.DataAs<uint8_t>(), aFromBuffer.SizeAs<uint8_t>(), cudaMemcpyDeviceToDevice ) );
        }

        /** @brief Set the content of the buffer to 0 */
        void CopyTo( GPUMemory aToBuffer )
        {
            CUDA_ASSERT( cudaMemcpy( (void *)aToBuffer.DataAs<uint8_t>(), (void *)mDevicePointer, SizeAs<uint8_t>(), cudaMemcpyDeviceToDevice ) );
        }

        /** @brief Downloads data from the device.
         *
         * Downloads the contents of the device buffer into a newly allocated `std::vector` appropriate type.
         *
         * @return newly allocated `std::vector` containing the data.
         */
        template <typename _Ty> std::vector<_Ty> Fetch()
        {
            std::vector<_Ty> l_HostArray( mSize );
            CUDA_ASSERT( cudaMemcpy( (void *)l_HostArray.data(), (void *)mDevicePointer, mSize, cudaMemcpyDeviceToHost ) );
            return l_HostArray;
        }

        CUdeviceptr RawDevicePtr() { return (CUdeviceptr)mDevicePointer; }
        CUdeviceptr *RawDevicePtrP() { return (CUdeviceptr *)&mDevicePointer; }

      private:
        void *mDevicePointer                       = nullptr;
        cudaExternalMemory_t mExternalMemoryHandle = 0;
        size_t mSize                               = 0;
    };

} // namespace LTSE::Cuda
