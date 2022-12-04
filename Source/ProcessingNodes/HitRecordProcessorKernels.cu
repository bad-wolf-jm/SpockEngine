/// @file   ReturnDataDEstructerKernels.cu
///
/// @brief  Kernel implementation.
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#include "HitRecordProcessorKernels.h"

#include "Scene/EnvironmentSampler/LaunchParams.h"

#include "Core/GPUResource/CudaAssert.h"
#include "TensorOps/Implementation/HelperMacros.h"

namespace SE::SensorModel
{
    namespace Kernel
    {
        CUDA_KERNEL_DEFINITION void ExtractReflectivity( MultiTensor aOut, MultiTensor aReturns )
        {
            int32_t i = blockIdx.x * SE::TensorOps::Private::ThreadsPerBlock + threadIdx.x;

            RETURN_UNLESS( i < aReturns.SizeAs<Dev::sHitRecord>() );
            RETURN_UNLESS( i < aOut.SizeAs<float>() );

            auto *lData = aReturns.DataAs<Dev::sHitRecord>();
            auto *lOut  = aOut.DataAs<float>();

            lOut[i] = lData[i].mIntensity;
        }

        CUDA_KERNEL_DEFINITION void ExtractDistance( MultiTensor aOut, MultiTensor aReturns )
        {
            int32_t i = blockIdx.x * SE::TensorOps::Private::ThreadsPerBlock + threadIdx.x;

            RETURN_UNLESS( i < aReturns.SizeAs<Dev::sHitRecord>() );
            RETURN_UNLESS( i < aOut.SizeAs<float>() );

            auto *lData = aReturns.DataAs<Dev::sHitRecord>();
            auto *lOut  = aOut.DataAs<float>();

            lOut[i] = lData[i].mDistance;
        }
    } // namespace Kernel

    void ExtractReflectivityOp( MultiTensor &aOut, MultiTensor &aReturns )
    {
        int lBlockCount = ( aReturns.SizeAs<Dev::sHitRecord>() / SE::TensorOps::Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( lBlockCount, 1, 1 );
        dim3 lBlockDim( SE::TensorOps::Private::ThreadsPerBlock );

        Kernel::ExtractReflectivity<<<lGridDim, lBlockDim>>>( aOut, aReturns );
    }

    void ExtractDistanceOp( MultiTensor &aOut, MultiTensor &aReturns )
    {
        int lBlockCount = ( aReturns.SizeAs<Dev::sHitRecord>() / SE::TensorOps::Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( lBlockCount, 1, 1 );
        dim3 lBlockDim( SE::TensorOps::Private::ThreadsPerBlock );

        Kernel::ExtractDistance<<<lGridDim, lBlockDim>>>( aOut, aReturns );
    }

} // namespace SE::SensorModel