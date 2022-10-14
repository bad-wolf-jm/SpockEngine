/// @file   StartRayStructBuilderKernels.cu
///
/// @brief  Kernel implementation.
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#include "StartRayStructBuilderKernels.h"

#include "LidarSensorConfig.h"
#include "Core/Cuda/CudaAssert.h"
#include "TensorOps/Implementation/HelperMacros.h"

namespace LTSE::dSpaceCompatibility
{

#define RETURN_UNLESS( condition )                                                                                                                                                 \
    do                                                                                                                                                                             \
    {                                                                                                                                                                              \
        if( !( condition ) )                                                                                                                                                       \
            return;                                                                                                                                                                \
    } while( 0 )

    namespace Kernel
    {
        static CUDA_KERNEL_DEFINITION void BuildStartRayStructure( MultiTensor aOut, MultiTensor aAzimuths, MultiTensor aElevations, MultiTensor aIntensities, MultiTensor aTimestamps )
        {
            uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
            int32_t i       = blockIdx.y * LTSE::TensorOps::Private::ThreadsPerBlock + threadIdx.x;

            RETURN_UNLESS( aOut.Shape().InBounds<OptixSensorLidar::StartRay>( lLayer, i ) );

            OptixSensorLidar::StartRay *lOut = aOut.DeviceBufferAt<OptixSensorLidar::StartRay>( lLayer );

            float *lAzimuths    = aAzimuths.DeviceBufferAt<float>( lLayer );
            float *lElevations  = aElevations.DeviceBufferAt<float>( lLayer );
            float *lIntensities = aIntensities.DeviceBufferAt<float>( lLayer );
            float *lTimestamps  = aTimestamps.DeviceBufferAt<float>( lLayer );

            lOut[i].azimuth        = lAzimuths[i];
            lOut[i].elevation      = lElevations[i];
            lOut[i].time_offset_ms = lTimestamps[i];
            lOut[i].beam_intensity = lIntensities[i];
        }
    } // namespace Kernel

    void BuildStartRayStructureOp( MultiTensor &aOut, MultiTensor &aAzimuths, MultiTensor &aElevations, MultiTensor &aIntensities, MultiTensor &aTimestamps )
    {
        int lBlockCount = ( aOut.Shape().mMaxBufferSize / LTSE::TensorOps::Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aOut.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( LTSE::TensorOps::Private::ThreadsPerBlock );

        Kernel::BuildStartRayStructure<<<lGridDim, lBlockDim>>>( aOut, aAzimuths, aElevations, aIntensities, aTimestamps );
    }

} // namespace LTSE::dSpaceCompatibility