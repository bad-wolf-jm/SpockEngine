/// @file   FPGAModelKernels.cu
///
/// @brief  Kernel definitions for FPGA processing model.
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2022 LeddarTech Inc. All rights reserved.

#include "FPGAModelKernels.h"

#include "Configuration.h"
#include "TensorOps/Implementation/HelperMacros.h"

namespace LTSE::SensorModel
{
    using namespace LTSE::TensorOps;

    namespace Kernels
    {

        LTSE_CUDA_INLINE LTSE_CUDA_HOST_DEVICE_FUNCTION_DEF uint32_t ComputeSaturatedPulseLength( float *aWaveform, float aThreshold, uint32_t aMaxGapSize, uint32_t aStart,
                                                                                                  uint32_t aLength )
        {
            int32_t lPulseStart = aStart;

            int32_t i = aStart;
            while( i < aLength )
            {
                while( ( i < aLength ) && ( aWaveform[i] > aThreshold ) )
                    i++;

                if( i < aLength )
                {
                    int32_t lGapStart = i;
                    while( ( i < aLength ) && ( ( i - lGapStart ) <= aMaxGapSize ) && ( aWaveform[i] <= aThreshold ) )
                        i++;

                    if( ( i < aLength ) && ( i - lGapStart ) > aMaxGapSize )
                    {
                        return lGapStart - lPulseStart;
                    }
                    else if( i >= aLength )
                    {
                        return aLength - lPulseStart;
                    }
                    else
                    {
                        // Go back to the top, the next saturated plateau is merged with the current one
                    }
                }
            }

            return i - aStart;
        }

        LTSE_CUDA_INLINE LTSE_CUDA_HOST_DEVICE_FUNCTION_DEF float SpecialLeftMovingAverage( float *aWaveform, uint32_t aWindowSize, uint32_t aStart, uint32_t aLength )
        {
            while( aWindowSize > aStart )
                aWindowSize /= 2;

            float lAccumulator = 0;
            for( int32_t i = aStart - aWindowSize; i < aStart; i++ )
                lAccumulator += aWaveform[i];

            return lAccumulator / aWindowSize;
        }

        LTSE_CUDA_INLINE LTSE_CUDA_HOST_DEVICE_FUNCTION_DEF float SpecialRightMovingAverage( float *aWaveform, uint32_t aWindowSize, uint32_t aStart, uint32_t aLength )
        {
            while( aWindowSize > ( aLength - aStart ) )
                aWindowSize /= 2;

            float lAccumulator = 0;
            for( int32_t i = aStart; i < aStart + aWindowSize; i++ )
                lAccumulator += aWaveform[i];

            return lAccumulator / aWindowSize;
        }

        template <typename _Ty> LTSE_CUDA_INLINE LTSE_CUDA_HOST_DEVICE_FUNCTION_DEF void UnsafeArraySet( _Ty *aWaveform, uint32_t aStart, uint32_t aEnd, _Ty aValue )
        {
            for( uint32_t i = aStart; i < aEnd; i++ )
                aWaveform[i] = aValue;
        }

        CUDA_KERNEL_DEFINITION void Blinder( MultiTensor aOut, MultiTensor aPulseIsSaturated, MultiTensor aPreSaturationBaseline, MultiTensor aPostSaturationBaseline,
                                             MultiTensor aSaturatedPulseLength, MultiTensor aLastUnsaturatedSample, sFPGAConfiguration::Blinder aConfiguration,
                                             MultiTensor aInputWaveforms, MemoryBuffer aBlockSizes, MemoryBuffer aElementCount )
        {
            uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
            int32_t i       = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

            RETURN_UNLESS( i < aBlockSizes.DataAs<uint32_t>()[lLayer] );

            auto lWaveformLength = aElementCount.DataAs<uint32_t>()[lLayer];

            auto *lWaveform = aInputWaveforms.DeviceBufferAt<float>( lLayer ) + i * lWaveformLength;

            auto *lOut                    = aOut.DeviceBufferAt<float>( lLayer ) + i * lWaveformLength;
            auto *lPulseIsSaturated       = aPulseIsSaturated.DeviceBufferAt<uint8_t>( lLayer ) + i * lWaveformLength;
            auto *lPreSaturationBaseline  = aPreSaturationBaseline.DeviceBufferAt<float>( lLayer ) + i * lWaveformLength;
            auto *lPostSaturationBaseline = aPostSaturationBaseline.DeviceBufferAt<float>( lLayer ) + i * lWaveformLength;
            auto *lSaturatedPulseLength   = aSaturatedPulseLength.DeviceBufferAt<uint32_t>( lLayer ) + i * lWaveformLength;
            auto *lLastUnsaturatedSample  = aLastUnsaturatedSample.DeviceBufferAt<float>( lLayer ) + i * lWaveformLength;

            for( uint32_t k = 0; k < lWaveformLength; k++ )
                lOut[k] = lWaveform[k];

            int32_t l = 0;
            while( l < ( lWaveformLength - 1 ) )
            {
                if( lWaveform[l] > aConfiguration.mThreshold )
                {
                    uint32_t lPulseLength =
                        ComputeSaturatedPulseLength( lWaveform, aConfiguration.mThreshold, aConfiguration.mClipPeriod + aConfiguration.mGuardLength1, l, lWaveformLength );

                    if( lPulseLength >= aConfiguration.mClipPeriod )
                    {
                        lPulseIsSaturated[l + 1] = lPulseIsSaturated[l] = static_cast<uint8_t>( 1 );
                        lLastUnsaturatedSample[l + 1] = lLastUnsaturatedSample[l] = lWaveform[l - 1];
                        lSaturatedPulseLength[l + 1] = lSaturatedPulseLength[l] = lPulseLength;

                        if( l - aConfiguration.mGuardLength0 > 0 )
                            lPreSaturationBaseline[l + 1] = lPreSaturationBaseline[l] =
                                floor( SpecialLeftMovingAverage( lWaveform, aConfiguration.mWindowSize, l - aConfiguration.mGuardLength0, lWaveformLength ) );

                        if( aConfiguration.mBaselineDelay > 0 )
                        {
                            if( l + aConfiguration.mClipPeriod + aConfiguration.mBaselineDelay < lWaveformLength )
                                lPostSaturationBaseline[l + 1] = lPostSaturationBaseline[l] = SpecialRightMovingAverage(
                                    lWaveform, aConfiguration.mWindowSize, l + aConfiguration.mClipPeriod + aConfiguration.mBaselineDelay, lWaveformLength );
                            else
                                lPostSaturationBaseline[l + 1] = lPostSaturationBaseline[l] = static_cast<float>( -( 1 << 15 ) );
                        }
                        else
                        {
                            if( l + lPulseLength + aConfiguration.mGuardLength1 < lWaveformLength )
                                lPostSaturationBaseline[l + 1] = lPostSaturationBaseline[l] =
                                    floor( SpecialRightMovingAverage( lWaveform, aConfiguration.mWindowSize, l + lPulseLength + aConfiguration.mGuardLength1, lWaveformLength ) );
                            else
                                lPostSaturationBaseline[l + 1] = lPostSaturationBaseline[l] = static_cast<float>( -( 1 << 15 ) );
                        }

                        uint32_t lBlindValueStart = l + aConfiguration.mClipPeriod;
                        uint32_t lBlindValueEnd   = l + lPulseLength + aConfiguration.mGuardLength1;
                        float lBlindValue         = 0.0f;
                        if( l + lPulseLength + aConfiguration.mGuardLength1 < lWaveformLength )
                            lBlindValue =
                                floor( SpecialRightMovingAverage( lWaveform, aConfiguration.mWindowSize, l + lPulseLength + aConfiguration.mGuardLength1, lWaveformLength ) );
                        else
                            lBlindValue = floor( SpecialLeftMovingAverage( lWaveform, aConfiguration.mWindowSize, l - aConfiguration.mGuardLength0, lWaveformLength ) );

                        UnsafeArraySet( lOut, lBlindValueStart, min( lBlindValueEnd, lWaveformLength ), lBlindValue );
                        l += lPulseLength;
                    }
                    else
                    {
                        l += 1;
                    }
                }
                else
                {
                    l += 1;
                }
            }
        }

        LTSE_CUDA_INLINE LTSE_CUDA_HOST_DEVICE_FUNCTION_DEF uint32_t PackHeader( uint32_t aID, uint32_t aVersion, uint32_t aDetectionCount, uint32_t aMaxDetectionCount,
                                                                                 uint32_t aSampleCount )
        {
            // clang-format off
            return static_cast<uint32_t>(
                ( ( aDetectionCount    & 0x0000003F ) << 26 ) |
                ( ( aSampleCount       & 0x00000FFF ) << 14 ) |
                ( ( aMaxDetectionCount & 0x0000003F ) <<  8 ) |
                ( ( aVersion           & 0x0000000F ) <<  4 ) |
                ( ( aID                & 0x0000000F ) <<  0 )
            );
            // clang-format on
        }

        LTSE_CUDA_INLINE LTSE_CUDA_HOST_DEVICE_FUNCTION_DEF inline uint32_t PackPDLaserAngle( uint32_t aPD, uint32_t aLaserAngle, uint32_t aIsRemoved, uint32_t aFrameNumber )
        {
            // clang-format off
            return static_cast<uint32_t>(
                ( ( aIsRemoved   & 0x00000001 ) << 31 ) |
                ( ( aLaserAngle  & 0x000001FF ) << 22 ) |
                ( ( aPD          & 0x0000003F ) << 16 ) |
                ( ( aFrameNumber & 0x0000FFFF ) <<  0 )
            );
            // clang-format on
        }

        LTSE_CUDA_INLINE LTSE_CUDA_HOST_DEVICE_FUNCTION_DEF inline uint32_t PackAcquisitioninfo( uint32_t mConfigurationID, uint32_t mFrameID, uint32_t mOpticalID,
                                                                                                 uint32_t mAcquisitionID )
        {
            // clang-format off
            return static_cast<uint32_t>(
                ( ( mConfigurationID & 0x000000FF ) << 24 ) |
                ( ( mFrameID         & 0x000000FF ) << 16 ) |
                ( ( mOpticalID       & 0x000000FF ) <<  8 ) |
                ( ( mAcquisitionID   & 0x000000FF ) <<  0 )
            );
            // clang-format on
        }

        LTSE_CUDA_INLINE LTSE_CUDA_HOST_DEVICE_FUNCTION_DEF inline uint32_t PackWaveformStatistics( float mTraceNoise, float mBaseline )
        {
            uint16_t lNoise    = static_cast<uint16_t>( mTraceNoise );
            int16_t lBaseline0 = static_cast<int16_t>( mBaseline );

            // We want to be sure that we access the binary representation of the baseline in representing
            // the signed value as an unsigned one.
            uint16_t lBaseline = *(uint16_t *)&lBaseline0;

            // clang-format off
            return static_cast<uint32_t>(
                ( ( lBaseline & 0x0000FFFF ) << 16 ) |
                ( ( lNoise    & 0x0000FFFF ) <<  0 )
            );
            // clang-format on
        }

        CUDA_KERNEL_DEFINITION void GeneratePacketHeader( MultiTensor aOut, MemoryBuffer aVersion, MemoryBuffer aID, MemoryBuffer aSampleCount, MemoryBuffer aMaxDetectionCount,
                                                          MultiTensor aDetectionCount, MultiTensor aPDNumber, MultiTensor aLaserAngle, MultiTensor aWaveformNoise,
                                                          MultiTensor aWaveformBaseline, MemoryBuffer aBufferSizes, MemoryBuffer aHeaderSizes )
        {
            uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
            int32_t i       = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

            RETURN_UNLESS( i < aBufferSizes.DataAs<uint32_t>()[lLayer] );

            uint32_t lVersion           = aVersion.DataAs<uint32_t>()[lLayer];
            uint32_t lID                = aID.DataAs<uint32_t>()[lLayer];
            uint32_t lSampleCount       = aSampleCount.DataAs<uint32_t>()[lLayer];
            uint32_t lMaxDetectionCount = aMaxDetectionCount.DataAs<uint32_t>()[lLayer];
            uint32_t lHeaderSize        = aHeaderSizes.DataAs<uint32_t>()[lLayer];

            uint32_t lPDNumber       = aPDNumber.DeviceBufferAt<uint32_t>( lLayer )[i];
            uint32_t lLaserAngle     = aLaserAngle.DeviceBufferAt<uint32_t>( lLayer )[i];
            float lWaveformNoise     = aWaveformNoise.DeviceBufferAt<float>( lLayer )[i];
            float lWaveformBaseline  = aWaveformBaseline.DeviceBufferAt<float>( lLayer )[i];
            uint32_t lDetectionCount = aDetectionCount.DeviceBufferAt<uint32_t>( lLayer )[i];

            auto *lOut = aOut.DeviceBufferAt<uint32_t>( lLayer ) + i * lHeaderSize;
            lOut[0]    = PackHeader( lID, lVersion, lDetectionCount, lMaxDetectionCount, lSampleCount );
            lOut[1]    = PackPDLaserAngle( lPDNumber, lLaserAngle, 0, 0 );
            lOut[2]    = 0;
            lOut[3]    = PackAcquisitioninfo( 0, 0, 0, 0 );
            lOut[4]    = PackWaveformStatistics( lWaveformNoise, lWaveformBaseline );
        }

        LTSE_CUDA_INLINE LTSE_CUDA_HOST_DEVICE_FUNCTION_DEF uint8_t ComputeLocalStatistics( float *aWaveform, int32_t aStart, int32_t aEnd, int32_t aWaveformLength, float *aMean,
                                                                                            float *aStd )
        {
            if( ( aStart < 0 ) || ( aEnd <= aStart ) )
            {
                *aMean = 0;
                *aStd  = 0;

                return 0;
            }

            if( aEnd >= aWaveformLength )
            {
                *aMean = 0;
                *aStd  = 0;

                return 0;
            }

            int32_t lWindowLength = aEnd - aStart;
            if( abs( lWindowLength ) < 2 )
            {
                *aMean = 0;
                *aStd  = 0;

                return 0;
            }

            float lMean = 0;
            float lStd  = 0;
            for( int32_t i = 0; i < lWindowLength; i++ )
                lMean += aWaveform[aStart + i];
            for( int32_t i = 0; i < lWindowLength; i++ )
                lStd += abs( aWaveform[aStart + i] * static_cast<float>( lWindowLength ) - lMean );

            *aMean = lMean;
            *aStd  = lStd;

            return 1;
        }

        template <typename _Ty> LTSE_CUDA_INLINE LTSE_CUDA_HOST_DEVICE_FUNCTION_DEF void Swap( _Ty *aV1, _Ty *aV2 )
        {
            _Ty lX = *aV2;

            *aV2 = *aV1;
            *aV1 = lX;
        }

        template <typename _Ty> LTSE_CUDA_INLINE LTSE_CUDA_HOST_DEVICE_FUNCTION_DEF void DecSort( _Ty *aV1, _Ty *aV2, _Ty *aV3 )
        {
            if( *aV1 < *aV2 )
                Swap( aV1, aV2 );
            if( *aV2 < *aV3 )
                Swap( aV2, aV3 );
            if( *aV1 < *aV2 )
                Swap( aV1, aV2 );
        }

        template <typename _Ty> LTSE_CUDA_INLINE LTSE_CUDA_HOST_DEVICE_FUNCTION_DEF void DecSort( _Ty *aV1, _Ty *aV2, _Ty *aV3, _Ty *aV4 )
        {
            if( *aV1 < *aV2 )
                Swap( aV1, aV2 );
            if( *aV3 < *aV4 )
                Swap( aV3, aV4 );
            if( *aV1 < *aV3 )
                Swap( aV1, aV3 );
            if( *aV2 < *aV4 )
                Swap( aV2, aV4 );
            if( *aV2 < *aV3 )
                Swap( aV2, aV3 );
        }

        CUDA_KERNEL_DEFINITION void Cfar( MultiTensor aOut, MultiTensor aInputTrace, float aThresholdFactor, float aMinStd, uint32_t aSkip, uint32_t aReferenceLength,
                                          uint32_t aGuardLength, MemoryBuffer aBlockSizes, MemoryBuffer aElementCount )
        {

            uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
            int32_t i       = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x;

            RETURN_UNLESS( blockIdx.y < aBlockSizes.DataAs<uint32_t>()[lLayer] );

            auto lWaveformLength = aElementCount.DataAs<uint32_t>()[lLayer];
            RETURN_UNLESS( i < lWaveformLength );

            auto *lWaveform = aInputTrace.DeviceBufferAt<float>( lLayer ) + blockIdx.y * lWaveformLength;
            auto *lOut      = aOut.DeviceBufferAt<float>( lLayer ) + blockIdx.y * lWaveformLength;

            uint8_t lValid       = 0;
            uint8_t lValidCount  = 0;
            float lMean[4]       = { 0.0f };
            float lStd[4]        = { 0.0f };
            uint32_t lWindowSize = aReferenceLength + aGuardLength;
            lValid =
                ComputeLocalStatistics( lWaveform, i - 2 * lWindowSize, i - 2 * lWindowSize + aReferenceLength, lWaveformLength, &( lMean[lValidCount] ), &( lStd[lValidCount] ) );
            lValidCount += lValid;

            lValid = ComputeLocalStatistics( lWaveform, i - lWindowSize, i - lWindowSize + aReferenceLength, lWaveformLength, &( lMean[lValidCount] ), &( lStd[lValidCount] ) );
            lValidCount += lValid;

            lValid = ComputeLocalStatistics( lWaveform, ( i + 1 ) + aGuardLength, ( i + 1 ) + aGuardLength + aReferenceLength, lWaveformLength, &( lMean[lValidCount] ),
                                             &( lStd[lValidCount] ) );
            lValidCount += lValid;

            lValid = ComputeLocalStatistics( lWaveform, ( i + 1 ) + lWindowSize + aGuardLength, ( i + 1 ) + lWindowSize + aGuardLength + aReferenceLength, lWaveformLength,
                                             &( lMean[lValidCount] ), &( lStd[lValidCount] ) );
            lValidCount += lValid;

            auto lReferenceLengthSquared = static_cast<float>( aReferenceLength * aReferenceLength );

            switch( lValidCount )
            {
            case 2:
            {
                float lMean0 = min( lMean[0], lMean[1] );
                float lStd0  = max( ( lStd[0] + lStd[1] ) / ( 2 * lReferenceLengthSquared ), aMinStd );

                lOut[i] = round( aThresholdFactor * lStd0 + ( lMean0 / static_cast<float>( aReferenceLength ) ) );
                break;
            }
            case 3:
            {
                DecSort( &lMean[0], &lMean[1], &lMean[2] );
                DecSort( &lStd[0], &lStd[1], &lStd[2] );

                float lMean0 = lMean[aSkip];
                float lStd0  = max( lStd[aSkip] / lReferenceLengthSquared, aMinStd );

                lOut[i] = round( aThresholdFactor * lStd0 + ( lMean0 / static_cast<float>( aReferenceLength ) ) );
                break;
            }
            case 4:
            {
                DecSort( &lMean[0], &lMean[1], &lMean[2], &lMean[3] );
                DecSort( &lStd[0], &lStd[1], &lStd[2], &lStd[3] );

                float lMean0 = lMean[aSkip];
                float lStd0  = max( lStd[aSkip] / lReferenceLengthSquared, aMinStd );

                lOut[i] = round( aThresholdFactor * lStd0 + ( lMean0 / static_cast<float>( aReferenceLength ) ) );
                break;
            }
            default:
                break;
            }
        }

        CUDA_KERNEL_DEFINITION void PeakDetector( MultiTensor aOutIsPeak, MultiTensor aWaveforms, uint32_t aMarginLeft, uint32_t aMarginRight, MultiTensor aThresholds,
                                                  MemoryBuffer aWaveformLength, MemoryBuffer aBlockSizes )
        {
            uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
            int32_t i       = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x + 1;

            RETURN_UNLESS( blockIdx.y < aBlockSizes.DataAs<uint32_t>()[lLayer] );

            auto lWaveformLength = aWaveformLength.DataAs<uint32_t>()[lLayer];
            RETURN_UNLESS( ( i > aMarginLeft ) && ( i < ( lWaveformLength - aMarginRight ) ) );

            auto *lWaveforms  = aWaveforms.DeviceBufferAt<float>( lLayer ) + blockIdx.y * lWaveformLength;
            auto *lThresholds = aThresholds.DeviceBufferAt<float>( lLayer ) + blockIdx.y * lWaveformLength;
            auto *lOutIsPeak  = aOutIsPeak.DeviceBufferAt<bool>( lLayer ) + blockIdx.y * lWaveformLength;

            lOutIsPeak[i] = ( lWaveforms[i - 1] < lWaveforms[i] ) && ( lWaveforms[i] >= lWaveforms[i + 1] ) && ( lWaveforms[i] >= lThresholds[i] );
        }

        CUDA_KERNEL_DEFINITION void QuadraticInterpolation( MultiTensor aOutX, MultiTensor aOutY, MultiTensor aWaveforms, MemoryBuffer aWaveformLength, MemoryBuffer aBlockSizes )
        {
            uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
            int32_t i       = blockIdx.z * Private::ThreadsPerBlock + threadIdx.x + 1;

            RETURN_UNLESS( blockIdx.y < aBlockSizes.DataAs<uint32_t>()[lLayer] );

            auto lWaveformLength = aWaveformLength.DataAs<uint32_t>()[lLayer];
            RETURN_UNLESS( ( i + 1 ) < lWaveformLength );

            auto *lY = aWaveforms.DeviceBufferAt<float>( lLayer ) + blockIdx.y * lWaveformLength;

            auto *lOutX = aOutX.DeviceBufferAt<float>( lLayer ) + blockIdx.y * lWaveformLength;
            auto *lOutY = aOutY.DeviceBufferAt<float>( lLayer ) + blockIdx.y * lWaveformLength;

            float lY0 = lY[i - 1];
            float lY1 = lY[i + 0];
            float lY2 = lY[i + 1];

            float lDx = 0.5f * ( lY2 - lY0 ) / ( 2.0f * lY1 - lY0 - lY2 );
            lOutX[i]  = static_cast<float>( i ) + lDx;
            lOutY[i]  = lY1 - 0.25f * lDx * ( lY0 - lY2 );
        }

        CUDA_KERNEL_DEFINITION void FilterDetections( MultiTensor aOut, MultiTensor aDetections, uint32_t aMaskLength, MemoryBuffer aBlockSizes, MemoryBuffer aElementCount )
        {
            uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
            int32_t i       = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

            RETURN_UNLESS( i < aBlockSizes.DataAs<uint32_t>()[lLayer] );

            auto lWaveformLength = aElementCount.DataAs<uint32_t>()[lLayer];

            auto *lY   = aDetections.DeviceBufferAt<uint8_t>( lLayer ) + blockIdx.y * lWaveformLength;
            auto *lOut = aOut.DeviceBufferAt<uint8_t>( lLayer ) + blockIdx.y * lWaveformLength;

            uint32_t k = 0;

            while( k < lWaveformLength )
            {
                if( lY[k] != 0 )
                {
                    lOut[k] = lY[k];
                    k++;
                    uint32_t lMaskStart = k;
                    while( ( k < lWaveformLength ) && ( k - lMaskStart < aMaskLength ) )
                        k++;
                }
            }
        }

        CUDA_KERNEL_DEFINITION void GenerateShortWaveformPackets( MultiTensor aOut, MultiTensor aWaveforms, MultiTensor aThresholds, MultiTensor aDistance, MultiTensor aAmplitude,
                                                                  MultiTensor aPulseIsSaturated, MultiTensor aPreSaturationBaseline, MultiTensor aPostSaturationBaseline,
                                                                  MultiTensor aSaturatedPulseLength, MultiTensor aLastUnsaturatedSample, MultiTensor aValidDetections,
                                                                  MultiTensor aValidDetectionCount, uint32_t aMaxDetectionCount, uint32_t aNeighbourCount, MemoryBuffer aBlockSizes,
                                                                  MemoryBuffer aElementCount, MemoryBuffer aWaveformPacketLength )
        {
            uint32_t lLayer = static_cast<uint32_t>( blockIdx.x );
            int32_t i       = blockIdx.y * Private::ThreadsPerBlock + threadIdx.x;

            RETURN_UNLESS( i < aBlockSizes.DataAs<uint32_t>()[lLayer] );

            auto lWaveformLength        = aElementCount.DataAs<uint32_t>()[lLayer];
            auto lWaveformPacketLength  = aWaveformPacketLength.DataAs<uint32_t>()[lLayer];
            auto lDetectionPacketLength = lWaveformPacketLength / aMaxDetectionCount;

            auto *lOut = aOut.DeviceBufferAt<uint32_t>( lLayer ) + i * lWaveformPacketLength;

            auto *lDistance               = aDistance.DeviceBufferAt<float>( lLayer ) + i * lWaveformLength;
            auto *lAmplitude              = aAmplitude.DeviceBufferAt<float>( lLayer ) + i * lWaveformLength;
            auto *lPulseIsSaturated       = aPulseIsSaturated.DeviceBufferAt<uint8_t>( lLayer ) + i * lWaveformLength;
            auto *lPreSaturationBaseline  = aPreSaturationBaseline.DeviceBufferAt<float>( lLayer ) + i * lWaveformLength;
            auto *lPostSaturationBaseline = aPostSaturationBaseline.DeviceBufferAt<float>( lLayer ) + i * lWaveformLength;
            auto *lSaturatedPulseLength   = aSaturatedPulseLength.DeviceBufferAt<uint32_t>( lLayer ) + i * lWaveformLength;
            auto *lLastUnsaturatedSample  = aLastUnsaturatedSample.DeviceBufferAt<float>( lLayer ) + i * lWaveformLength;
            auto *lValidDetections        = aValidDetections.DeviceBufferAt<uint8_t>( lLayer ) + i * lWaveformLength;

            auto *lWaveform   = aWaveforms.DeviceBufferAt<float>( lLayer ) + i * lWaveformLength;
            auto *lThresholds = aThresholds.DeviceBufferAt<float>( lLayer ) + i * lWaveformLength;

            uint32_t lDetectionCount     = 0;
            int32_t lShortWaveformLength = 2 * aNeighbourCount + 1;
            for( uint32_t k = 0; k < lWaveformLength; k++ )
            {

                if( lValidDetections[k] == 1 )
                {
                    const float lFixedPointFactor = 64.0f; // This will be part of the configuration structure at some point
                    auto *lCurrentDetection       = &( lOut[lDetectionCount * lDetectionPacketLength] );
                    lCurrentDetection[0] = ( static_cast<int32_t>( lDistance[k] * lFixedPointFactor ) & 0x7FFF ) | ( static_cast<int32_t>( lLastUnsaturatedSample[k] ) << 16 );

                    lCurrentDetection[1] = ( static_cast<int32_t>( lPreSaturationBaseline[k] ) & 0xffff ) | ( static_cast<int32_t>( lPostSaturationBaseline[k] ) << 16 );

                    if( lPulseIsSaturated[k] == 1 )
                    {
                        lCurrentDetection[2] = lSaturatedPulseLength[k];
                    }
                    else
                    {
                        lCurrentDetection[2] = static_cast<int32_t>( lAmplitude[k] );
                    }

                    int32_t lBin    = ( static_cast<int32_t>( lDistance[k] ) );
                    int32_t lOffset = max( 0, min( static_cast<int32_t>( lWaveformLength ) - lShortWaveformLength, lBin - static_cast<int32_t>( aNeighbourCount ) ) );

                    lCurrentDetection[3] = ( static_cast<int32_t>( lPulseIsSaturated[k] ) << 15 ) | ( lBin & 0x7FFF ) | ( lOffset << 16 );

                    auto *lOutShortWf = (int16_t *)( &lCurrentDetection[4] );
                    for( uint32_t l = 0; l < lShortWaveformLength; l++ )
                        lOutShortWf[l] = static_cast<int16_t>( lWaveform[lOffset + l] );

                    auto *lOutThresholds = lOutShortWf + lShortWaveformLength;
                    for( uint32_t l = 0; l < lShortWaveformLength; l++ )
                        lOutThresholds[l] = static_cast<int16_t>( lThresholds[lOffset + l] );

                    lDetectionCount++;

                    if( lDetectionCount > aMaxDetectionCount )
                        break;
                }
            }
        }
    }; // namespace Kernels

    void BlinderOp( MultiTensor &aOut, MultiTensor &aPulseIsSaturated, MultiTensor &aPreSaturationBaseline, MultiTensor &aPostSaturationBaseline,
                    MultiTensor &aSaturatedPulseLength, MultiTensor &aLastUnsaturatedSample, sFPGAConfiguration::Blinder const &aConfiguration, MultiTensor &aInputWaveforms,
                    MemoryBuffer &aBlockSizes, MemoryBuffer &aElementCount, uint32_t aMaxBlockSize )
    {
        int lBlockCount = ( aMaxBlockSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aInputWaveforms.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Blinder<<<lGridDim, lBlockDim>>>( aOut, aPulseIsSaturated, aPreSaturationBaseline, aPostSaturationBaseline, aSaturatedPulseLength, aLastUnsaturatedSample,
                                                   aConfiguration, aInputWaveforms, aBlockSizes, aElementCount );
    }

    void CfarOp( MultiTensor &aOut, MultiTensor &aWaveforms, float aThresholdFactor, float aMinStd, uint32_t aSkip, uint32_t aReferenceLength, uint32_t aGuardLength,
                 MemoryBuffer &aBlockSizes, MemoryBuffer &aElementCount, uint32_t aMaxWaveformLength, uint32_t aMaxBlockSize )
    {
        int lBlockCount = ( aMaxWaveformLength / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aWaveforms.Shape().CountLayers(), aMaxBlockSize, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::Cfar<<<lGridDim, lBlockDim>>>( aOut, aWaveforms, aThresholdFactor, aMinStd, aSkip, aReferenceLength, aGuardLength, aBlockSizes, aElementCount );
    }

    void PeakDetectorOp( MultiTensor &aOutIsPeak, MultiTensor &aWaveforms, uint32_t aMarginLeft, uint32_t aMarginRight, MultiTensor &aThresholds, MemoryBuffer &aWaveformLength,
                         MemoryBuffer &aBlockSizes, uint32_t aMaxWaveformLength, uint32_t aMaxBlockSize )
    {
        int lBlockCount = ( aMaxWaveformLength / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aWaveforms.Shape().CountLayers(), aMaxBlockSize, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::PeakDetector<<<lGridDim, lBlockDim>>>( aOutIsPeak, aWaveforms, aMarginLeft, aMarginRight, aThresholds, aWaveformLength, aBlockSizes );
    }

    void QuadraticInterpolationOp( MultiTensor &aOutX, MultiTensor &aOutY, MultiTensor &aWaveforms, MemoryBuffer &aWaveformLength, MemoryBuffer &aBlockSizes,
                                   uint32_t aMaxWaveformLength, uint32_t aMaxBlockSize )
    {
        int lBlockCount = ( aMaxWaveformLength / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aWaveforms.Shape().CountLayers(), aMaxBlockSize, lBlockCount );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::QuadraticInterpolation<<<lGridDim, lBlockDim>>>( aOutX, aOutY, aWaveforms, aWaveformLength, aBlockSizes );
    }

    void FilterDetectionOp( MultiTensor &aOut, MultiTensor &aDetections, uint32_t aMaskLength, MemoryBuffer &aBlockSizes, MemoryBuffer &aElementCount, uint32_t aMaxBlockSize )
    {
        int lBlockCount = ( aMaxBlockSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aDetections.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::FilterDetections<<<lGridDim, lBlockDim>>>( aOut, aDetections, aMaskLength, aBlockSizes, aElementCount );
    }

    void GeneratePacketHeaderOp( MultiTensor &aOut, MemoryBuffer &aVersion, MemoryBuffer &aID, MemoryBuffer &aSampleCount, MemoryBuffer &aMaxDetectionCount,
                                 MultiTensor &aDetectionCount, MultiTensor &aPDNumber, MultiTensor &aLaserAngle, MultiTensor &aWaveformNoise, MultiTensor &aWaveformBaseline,
                                 MemoryBuffer &aBufferSizes, uint32_t aMaxBufferSize, MemoryBuffer &aHeaderSizes )
    {
        int lBlockCount = ( aMaxBufferSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aPDNumber.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::GeneratePacketHeader<<<lGridDim, lBlockDim>>>( aOut, aVersion, aID, aSampleCount, aMaxDetectionCount, aDetectionCount, aPDNumber, aLaserAngle, aWaveformNoise,
                                                                aWaveformBaseline, aBufferSizes, aHeaderSizes );
    }

    void GenerateShortWaveformPacketsOp( MultiTensor &aOut, MultiTensor &aWaveforms, MultiTensor &aThresholds, MultiTensor &aDistance, MultiTensor &aAmplitude,
                                         MultiTensor &aPulseIsSaturated, MultiTensor &aPreSaturationBaseline, MultiTensor &aPostSaturationBaseline,
                                         MultiTensor &aSaturatedPulseLength, MultiTensor &aLastUnsaturatedSample, MultiTensor &aValidDetections, MultiTensor &aValidDetectionCount,
                                         uint32_t aMaxDetectionCount, uint32_t aNeighbourCount, MemoryBuffer &aBlockSizes, MemoryBuffer &aElementCount, MemoryBuffer &aPacketSizes,
                                         uint32_t aMaxBlockSize, uint32_t aMaxWaveformLength )
    {
        int lBlockCount = ( aMaxBlockSize / Private::ThreadsPerBlock ) + 1;

        dim3 lGridDim( aWaveforms.Shape().CountLayers(), lBlockCount, 1 );
        dim3 lBlockDim( Private::ThreadsPerBlock );

        Kernels::GenerateShortWaveformPackets<<<lGridDim, lBlockDim>>>( aOut, aWaveforms, aThresholds, aDistance, aAmplitude, aPulseIsSaturated, aPreSaturationBaseline,
                                                                        aPostSaturationBaseline, aSaturatedPulseLength, aLastUnsaturatedSample, aValidDetections,
                                                                        aValidDetectionCount, aMaxDetectionCount, aNeighbourCount, aBlockSizes, aElementCount, aPacketSizes );
    }

} // namespace LTSE::SensorModel
