/// @file   FPGAKernels.cu
///
/// @brief  Kernel definitions for the FPGA.
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2022 LeddarTech Inc. All rights reserved.

#include "FPGAKernels.h"

#include "Cuda/CudaAssert.h"
#include "Cuda/Texture2D.h"

#include "TensorOps/Implementation/HelperMacros.h"

#include "WaveformType.h"

namespace LTSE::SensorModel::Kernels
{
    using namespace LTSE::TensorOps;

    LTSE_CUDA_DEVICE_FUNCTION_DEF float InterpolatePeak( float Y0, float Y1, float Y2 )
    {
        float lDistance = 0.0f;
        if( ( Y0 <= Y1 ) && ( Y2 <= Y1 ) )
        {
            float lDenom = ( Y1 - Y0 ) + ( Y1 - Y2 );
            if( lDenom > 0 )
                lDistance += ( ( Y1 - Y0 ) / lDenom ) - 0.5f;
        }

        return lDistance;
    }

    CUDA_KERNEL_DEFINITION void FPGAProcess( MultiTensor aOut, sFPGAConfiguration aConfig, MultiTensor aWaveforms, MemoryBuffer aSegmentCounts, MemoryBuffer aWaveformLengths )
    {
        uint32_t lWaveformLength = aWaveformLengths.DataAs<uint32_t>()[blockIdx.x];
        uint32_t lSegmentCount   = aSegmentCounts.DataAs<uint32_t>()[blockIdx.x];

        int32_t i = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;
        RETURN_UNLESS( i < lSegmentCount );

        uint32_t lBlockIdxX = static_cast<uint32_t>( blockIdx.x );
        auto *lWaveform     = aWaveforms.DeviceBufferAt<uint16_t>( lBlockIdxX );
        auto *lOut          = aOut.DeviceBufferAt<sWaveformPacket>( lBlockIdxX );

        lWaveform += ( i * lWaveformLength );
        lOut += i;

        // Compute the base level for the current waveform.
        float lMean = 0.0f;
        for( uint16_t i = 0; i < aConfig.mBaseLevelSampleCount; i++ )
            lMean += lWaveform[i];
        lMean /= static_cast<float>( aConfig.mBaseLevelSampleCount );

        // Compute the noise level for the current waveform.
        float lStd = 0.0f;
        for( uint16_t i = 0; i < aConfig.mBaseLevelSampleCount; i++ )
            lStd += ( lWaveform[i] - lMean ) * ( lWaveform[i] - lMean );
        lStd /= static_cast<float>( aConfig.mBaseLevelSampleCount );
        lStd = sqrt( lStd );

        // Look for the maximal amplitude in the current trace. For now we only support one detection per segment
        // This is used in the short waveform structure.
        uint16_t lMaxAmplitudeIndex = 0;
        float lMaxAmplitude         = lWaveform[lMaxAmplitudeIndex];
        for( uint16_t i = 0; i < lWaveformLength; i++ )
        {
            if( lWaveform[i] > lMaxAmplitude )
            {
                lMaxAmplitudeIndex = i;
                lMaxAmplitude      = lWaveform[i];
            }
        }

        lOut->mHeader.mFrameNumber     = 1;
        lOut->mHeader.mTraceBaseLevel  = static_cast<int16_t>( lMean );
        lOut->mHeader.mTraceNoiseLevel = static_cast<int16_t>( lStd );

        uint32_t aID             = 1;
        uint32_t aVersion        = 1;
        uint32_t aDetectionCount = 1;

        if( lMaxAmplitudeIndex < ( SAMPLES_PER_CHANNEL - 1 ) / 2 )
        {
            lOut->mHeader.PackHeader( aID, aVersion, 0, ECHOES_PER_CHANNEL, SAMPLES_PER_CHANNEL );

            lOut->mWaveform[0].mPulse.mInterpolatedDistance  = 0;
            lOut->mWaveform[0].mPulse.mLastUnsaturatedSample = 0;
            lOut->mWaveform[0].mPulse.mPulseBaseLevel        = 0;
            lOut->mWaveform[0].mPulse.mMaxIndex              = 0;
            lOut->mWaveform[0].mPulse.mAmplitude             = 0;
            lOut->mWaveform[0].mPulse.mOffset                = 0;

            for( uint32_t i = 0; i < SAMPLES_PER_CHANNEL; i++ )
            {
                lOut->mWaveform[0].mRawTrace[i]       = 0;
                lOut->mWaveform[0].mProcessedTrace[i] = 0;
            }
        }
        else
        {
            lOut->mHeader.PackHeader( aID, aVersion, 1, ECHOES_PER_CHANNEL, SAMPLES_PER_CHANNEL );

            auto lPeakDistance =
                static_cast<float>( lMaxAmplitudeIndex ) + InterpolatePeak( lWaveform[lMaxAmplitudeIndex - 1], lWaveform[lMaxAmplitudeIndex], lWaveform[lMaxAmplitudeIndex + 1] );

            lOut->mWaveform[0].mPulse.mInterpolatedDistance  = static_cast<uint16_t>( lPeakDistance * static_cast<float>( 1 << 6 ) );
            lOut->mWaveform[0].mPulse.mLastUnsaturatedSample = 0;
            lOut->mWaveform[0].mPulse.mPulseBaseLevel        = 0;
            lOut->mWaveform[0].mPulse.mMaxIndex              = lMaxAmplitudeIndex;
            lOut->mWaveform[0].mPulse.mAmplitude             = static_cast<int32_t>( lMaxAmplitude );
            lOut->mWaveform[0].mPulse.mOffset                = lMaxAmplitudeIndex - ( SAMPLES_PER_CHANNEL - 1 ) / 2;

            for( uint32_t i = 0; i < SAMPLES_PER_CHANNEL; i++ )
            {
                lOut->mWaveform[0].mRawTrace[i]       = static_cast<int16_t>( lWaveform[i + lOut->mWaveform[0].mPulse.mOffset] );
                lOut->mWaveform[0].mProcessedTrace[i] = static_cast<int16_t>( lWaveform[i + lOut->mWaveform[0].mPulse.mOffset] );
            }
        }
    }
} // namespace LTSE::SensorModel::Kernels

namespace LTSE::SensorModel
{
    using namespace LTSE::TensorOps;

    void FPGAProcessOp( MultiTensor &aOut, sFPGAConfiguration &aConfig, MultiTensor &aWaveforms, MemoryBuffer &aSegmentCounts, MemoryBuffer &aWaveformLengths,
                        uint32_t aMaxSegmentCount )
    {
        int lBlockCount = ( aMaxSegmentCount / Private::ThreadsPerBlock ) + 1;
        dim3 lGridDim( aWaveforms.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::FPGAProcess<<<lGridDim, lBlockDim>>>( aOut, aConfig, aWaveforms, aSegmentCounts, aWaveformLengths );
    }
} // namespace LTSE::SensorModel