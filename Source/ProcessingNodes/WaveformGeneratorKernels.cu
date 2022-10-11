#include "WaveformGeneratorKernels.h"

#include "Cuda/Texture2D.h"

#include "Cuda/CudaAssert.h"

#include "TensorOps/Implementation/HelperMacros.h"

namespace LTSE::SensorModel::Kernels
{
    using namespace LTSE::TensorOps;

    static CUDA_KERNEL_DEFINITION void GenerateWaveforms( MultiTensor aOut, MultiTensor aReturnIntensities, MultiTensor aReturnTimes, MemoryBuffer aSummandCount,
                                                          MemoryBuffer aAPDSizes, MemoryBuffer aAPDTemplatePositions, MemoryBuffer aSamplingLength, MemoryBuffer aSamplingInterval,
                                                          MemoryBuffer aPulseTemplates )
    {
        auto lSamplingLength = aSamplingLength.DataAs<uint32_t>()[blockIdx.x];
        int i                = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

        RETURN_UNLESS( blockIdx.y < aAPDSizes.DataAs<uint32_t>()[blockIdx.x] );
        RETURN_UNLESS( i < lSamplingLength );

        auto lSamplingInterval = aSamplingInterval.DataAs<float>()[blockIdx.x];
        auto lSummandCount     = aSummandCount.DataAs<uint32_t>()[blockIdx.x];

        Cuda::TextureSampler2D::DeviceData lPulseTemplate = aPulseTemplates.DataAs<Cuda::TextureSampler2D::DeviceData>()[blockIdx.x];

        uint32_t lBlockIdxX = static_cast<uint32_t>( blockIdx.x );
        auto *lIntensities  = aReturnIntensities.DeviceBufferAt<float>( lBlockIdxX );
        auto *lTimes        = aReturnTimes.DeviceBufferAt<float>( lBlockIdxX );
        auto *lOut          = aOut.DeviceBufferAt<float>( lBlockIdxX );

        int lInputBufferStart  = lSummandCount * blockIdx.y;
        int lOutputBufferStart = lSamplingLength * blockIdx.y;

        lIntensities += lInputBufferStart;
        lTimes += lInputBufferStart;
        lOut += lOutputBufferStart;

        for( int j = 0; j < lSummandCount; j++ )
        {
            if( lIntensities[j] == 0.0f )
                continue;

            float lTSample   = static_cast<float>( i ) * lSamplingInterval;
            float lTTemplate = ( lTSample - lTimes[j] );

            lOut[i] += lPulseTemplate.Fetch<float>( lTTemplate, lIntensities[j] );
        }
    }

    void ResolveAbstractWaveformsOp( MultiTensor &aOut, MultiTensor &aReturnIntensities, MultiTensor &aReturnTimes, MemoryBuffer &aSummandCount, MemoryBuffer &aAPDSizes,
                                     MemoryBuffer &aAPDTemplatePositions, MemoryBuffer &aSamplingLength, MemoryBuffer &aSamplingInterval, MemoryBuffer &aPulseTemplates,
                                     int32_t lMaxAPDSize, uint32_t lMaxWaveformLength )
    {
        int lBlockCount = ( lMaxWaveformLength / Private::ThreadsPerBlock ) + 1;
        dim3 lGridDim( aReturnIntensities.Shape().CountLayers(), lMaxAPDSize, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        GenerateWaveforms<<<lGridDim, lBlockDim>>>( aOut, aReturnIntensities, aReturnTimes, aSummandCount, aAPDSizes, aAPDTemplatePositions, aSamplingLength, aSamplingInterval,
                                                    aPulseTemplates );
    }

} // namespace LTSE::SensorModel::Kernels