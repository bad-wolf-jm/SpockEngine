/// @file   ReturnDataDEstructerKernels.cu
///
/// @brief  Kernel implementation.
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#include "ReturnDataProcessorKernels.h"

#include "Core/Cuda/CudaAssert.h"
#include "LidarRaytracingPoint.h"
#include "LidarSensorConfig.h"
#include "TensorOps/Implementation/HelperMacros.h"

namespace LTSE::dSpaceCompatibility
{
    using namespace OptixSensorLidar;

    namespace Kernel
    {
        CUDA_KERNEL_DEFINITION void ExtractReflectivity( MultiTensor aOut, MemoryBuffer aReturns )
        {
            int32_t i = blockIdx.x * LTSE::TensorOps::Private::ThreadsPerBlock + threadIdx.x;

            RETURN_UNLESS( i < aReturns.SizeAs<LidarRaytracingPoint>() );

            auto *lData = aReturns.DataAs<LidarRaytracingPoint>();
            RETURN_UNLESS( lData[i].rayID < aOut.SizeAs<float>() );

            auto *lOut  = aOut.DataAs<float>();

            lOut[lData[i].rayID] = lData[i].reflectivity;
        }

        CUDA_KERNEL_DEFINITION void ExtractDistance( MultiTensor aOut, MemoryBuffer aReturns )
        {
            int32_t i = blockIdx.x * LTSE::TensorOps::Private::ThreadsPerBlock + threadIdx.x;

            RETURN_UNLESS( i < aReturns.SizeAs<LidarRaytracingPoint>() );

            auto *lData = aReturns.DataAs<LidarRaytracingPoint>();
            RETURN_UNLESS( lData[i].rayID < aOut.SizeAs<float>() );

            auto *lOut  = aOut.DataAs<float>();

            lOut[lData[i].rayID] = lData[i].distance;
        }
    } // namespace Kernel

    void ExtractReflectivityOp( MultiTensor &aOut, MemoryBuffer &aReturns )
    {
        int lBlockCount = ( aReturns.SizeAs<LidarRaytracingPoint>() / LTSE::TensorOps::Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( lBlockCount, 1, 1 );
        dim3 lBlockDim( LTSE::TensorOps::Private::ThreadsPerBlock );

        Kernel::ExtractReflectivity<<<lGridDim, lBlockDim>>>( aOut, aReturns );
    }

    void ExtractDistanceOp( MultiTensor &aOut, MemoryBuffer &aReturns )
    {
        int lBlockCount = ( aReturns.SizeAs<LidarRaytracingPoint>() / LTSE::TensorOps::Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( lBlockCount, 1, 1 );
        dim3 lBlockDim( LTSE::TensorOps::Private::ThreadsPerBlock );

        Kernel::ExtractDistance<<<lGridDim, lBlockDim>>>( aOut, aReturns );
    }

} // namespace LTSE::dSpaceCompatibility