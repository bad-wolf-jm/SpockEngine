/// @file   ReturnDataDEstructerKernels.cu
///
/// @brief  Kernel implementation.
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#include "HitRecordProcessorKernels.h"

#include "Developer/EnvironmentSampler/LaunchParams.h"

#include "Cuda/CudaAssert.h"
#include "TensorOps/Implementation/HelperMacros.h"

namespace LTSE::SensorModel
{
    namespace Kernel
    {
        CUDA_KERNEL_DEFINITION void ExtractReflectivity( MultiTensor aOut, MultiTensor aReturns )
        {
            int32_t i = blockIdx.x * LTSE::TensorOps::Private::ThreadsPerBlock + threadIdx.x;

            RETURN_UNLESS( i < aReturns.SizeAs<Dev::HitRecord>() );
            RETURN_UNLESS( i < aOut.SizeAs<float>() );

            auto *lData = aReturns.DataAs<Dev::HitRecord>();
            auto *lOut  = aOut.DataAs<float>();

            lOut[i] = lData[i].Intensity;
        }

        CUDA_KERNEL_DEFINITION void ExtractDistance( MultiTensor aOut, MultiTensor aReturns )
        {
            int32_t i = blockIdx.x * LTSE::TensorOps::Private::ThreadsPerBlock + threadIdx.x;

            RETURN_UNLESS( i < aReturns.SizeAs<Dev::HitRecord>() );
            RETURN_UNLESS( i < aOut.SizeAs<float>() );

            auto *lData = aReturns.DataAs<Dev::HitRecord>();
            auto *lOut  = aOut.DataAs<float>();

            lOut[i] = lData[i].Distance;
        }
    } // namespace Kernel

    void ExtractReflectivityOp( MultiTensor &aOut, MultiTensor &aReturns )
    {
        int lBlockCount = ( aReturns.SizeAs<Dev::HitRecord>() / LTSE::TensorOps::Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( lBlockCount, 1, 1 );
        dim3 lBlockDim( LTSE::TensorOps::Private::ThreadsPerBlock );

        Kernel::ExtractReflectivity<<<lGridDim, lBlockDim>>>( aOut, aReturns );
    }

    void ExtractDistanceOp( MultiTensor &aOut, MultiTensor &aReturns )
    {
        int lBlockCount = ( aReturns.SizeAs<Dev::HitRecord>() / LTSE::TensorOps::Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( lBlockCount, 1, 1 );
        dim3 lBlockDim( LTSE::TensorOps::Private::ThreadsPerBlock );

        Kernel::ExtractDistance<<<lGridDim, lBlockDim>>>( aOut, aReturns );
    }

} // namespace LTSE::SensorModel