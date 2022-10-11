#include "WaveformGeneratorKernels.h"

#include "Cuda/Texture2D.h"

#include "Cuda/CudaAssert.h"

#include "TensorOps/Implementation/HelperMacros.h"

namespace LTSE::SensorModel::Kernels
{
    using namespace LTSE::TensorOps;

    static CUDA_KERNEL_DEFINITION void ResolveElectronicCrosstalkWaveforms( MultiTensor aOut, MultiTensor aReturnIntensities, MultiTensor aReturnTimes, MemoryBuffer aSummandCount,
                                                                            MemoryBuffer aAPDSizes, MemoryBuffer aAPDTemplatePositions, MemoryBuffer aSamplingLength,
                                                                            MemoryBuffer aSamplingInterval, MemoryBuffer aPulseTemplates )
    {
        auto lSamplingLength = aSamplingLength.DataAs<uint32_t>()[blockIdx.x];
        uint32_t lAPDCount   = aAPDSizes.DataAs<uint32_t>()[blockIdx.x];

        int i = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < aAPDSizes.DataAs<uint32_t>()[blockIdx.x] );
        RETURN_UNLESS( i < lSamplingLength );

        auto lSamplingInterval = aSamplingInterval.DataAs<float>()[blockIdx.x];
        auto lSummandCount     = aSummandCount.DataAs<uint32_t>()[blockIdx.x];

        auto *lOut = &( aOut.DeviceBufferAt<float>( blockIdx.x )[lSamplingLength * blockIdx.y] );

        auto *lPulseTemplate0  = aPulseTemplates.DataAs<Cuda::TextureSampler2D::DeviceData>();
        auto *lIntensities0    = aReturnIntensities.DeviceBufferAt<float>( blockIdx.x );
        auto *lDetectionTimes0 = aReturnTimes.DeviceBufferAt<float>( blockIdx.x );

        float lTSample = static_cast<float>( i ) * lSamplingInterval;

        for( uint32_t lAggIdx = 0; lAggIdx < lAPDCount; lAggIdx++ )
        {
            if( lAggIdx == blockIdx.y )
                continue;

            auto *lIntensities    = &( lIntensities0[lSummandCount * lAggIdx] );
            auto *lDetectionTimes = &( lDetectionTimes0[lSummandCount * lAggIdx] );
            auto lPulseTemplate   = lPulseTemplate0[lAPDCount * lAggIdx + blockIdx.y];

            for( int j = 0; j < lSummandCount; j++ )
            {
                if( lIntensities[j] == 0.0f )
                    continue;

                float lTTemplate = ( lTSample - lDetectionTimes[j] );

                lOut[i] += lPulseTemplate.Fetch<float>( lTTemplate, lIntensities[j] );
            }
        }
    }

    void ResolveElectronicCrosstalkWaveformsOp( MultiTensor &aOut, MultiTensor &aReturnIntensities, MultiTensor &aReturnTimes, MemoryBuffer &aSummandCount, MemoryBuffer &aAPDSizes,
                                                MemoryBuffer &aAPDTemplatePositions, MemoryBuffer &aSamplingLength, MemoryBuffer &aSamplingInterval, MemoryBuffer &aPulseTemplates,
                                                int32_t lMaxAPDSize, uint32_t lMaxWaveformLength )
    {
        int lBlockCount = ( lMaxWaveformLength / Private::ThreadsPerBlock ) + 1;
        dim3 lGridDim( aReturnIntensities.Shape().CountLayers(), lMaxAPDSize, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        ResolveElectronicCrosstalkWaveforms<<<lGridDim, lBlockDim>>>( aOut, aReturnIntensities, aReturnTimes, aSummandCount, aAPDSizes, aAPDTemplatePositions, aSamplingLength,
                                                                      aSamplingInterval, aPulseTemplates );
    }

} // namespace LTSE::SensorModel::Kernels