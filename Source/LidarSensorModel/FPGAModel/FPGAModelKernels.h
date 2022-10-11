/// @file   FPGAModelKernels.h
///
/// @brief  Kernel definitions for FPGA processing model.
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2022 LeddarTech Inc. All rights reserved.

#pragma once
#include "Configuration.h"
#include "Cuda/CudaAssert.h"
#include "Cuda/MultiTensor.h"

using namespace LTSE::Cuda;

namespace LTSE::SensorModel
{
    /// @brief Blinder algorithm abstraction
    ///
    /// @param aOut Output tensor
    /// @param aPulseIsSaturated  Determine whether the pulse is saturated
    /// @param aPreSaturationBaseline  Pulse baseline before saturation
    /// @param aPostSaturationBaseline Pulse baseline after saturation
    /// @param aSaturatedPulseLength Length of the saturation plateau
    /// @param aLastUnsaturatedSample Amplitude of the last sample before the saturation plateau
    /// @param aConfiguration Algorithm configuratiion
    /// @param aInputWaveforms Input waveform buffer
    /// @param aBlockSizes Product of the lengths of the first rank-1 dimensions of `aInputWaveforms`
    /// @param aElementCount Trace length
    /// @param aMaxBlockSize Maximum value of the `aBlockSizes` parameter
    ///
    void BlinderOp( MultiTensor &aOut, MultiTensor &aPulseIsSaturated, MultiTensor &aPreSaturationBaseline, MultiTensor &aPostSaturationBaseline,
                    MultiTensor &aSaturatedPulseLength, MultiTensor &aLastUnsaturatedSample, sFPGAConfiguration::Blinder const &aConfiguration, MultiTensor &aInputWaveforms,
                    MemoryBuffer &aBlockSizes, MemoryBuffer &aElementCount, uint32_t aMaxBlockSize );

    /// @brief Cfar algorithm abstraction
    ///
    /// @param aOut Output tensor
    /// @param aWaveforms Input waveform buffer
    /// @param aThresholdFactor Factor to be applied to calculated threshold
    /// @param aMinStd Minimum noise standard deviation for threshold calculation
    /// @param aSkip Number of values to skip in sorted list when selecting reference statistics
    /// @param aReferenceLength Number of samples to consider for the calculation on each side of the sample under investigation, not including the guard interval
    /// @param aGuardLength Number of samples on each side of the sample under investigation ignored, not part of the reference window
    /// @param aBlockSizes Product of the lengths of the first rank-1 dimensions of `aWaveforms`
    /// @param aElementCount Product of the lengths of the first rank-1 dimensions of `aWaveforms`
    /// @param aMaxWaveformLength Maximum value of the `aElementCount` parameter
    /// @param aMaxBlockSize Maximum value of the `aBlockSizes` parameter
    ///
    void CfarOp( MultiTensor &aOut, MultiTensor &aWaveforms, float aThresholdFactor, float aMinStd, uint32_t aSkip, uint32_t aReferenceLength, uint32_t aGuardLength,
                 MemoryBuffer &aBlockSizes, MemoryBuffer &aElementCount, uint32_t aMaxWaveformLength, uint32_t aMaxBlockSize );

    /// @brief Peak detector algorithm abstraction
    ///
    /// @param aOutIsPeak Output tensor
    /// @param aWaveforms Input waveforms
    /// @param aMarginLeft Number of samples in each trace to skip before looking for peaks
    /// @param aMarginRight Number of samples to leave out at the end of each trace when looking for peaks
    /// @param aThresholds Threshold values
    /// @param aWaveformLength Product of the lengths of the first rank-1 dimensions of `aWaveforms`
    /// @param aBlockSizes Product of the lengths of the first rank-1 dimensions of `aWaveforms`
    /// @param aMaxWaveformLength Maximum value of the `aWaveformLength` parameter
    /// @param aMaxBlockSize Maximum value of the `aBlockSizes` parameter
    ///
    void PeakDetectorOp( MultiTensor &aOutIsPeak, MultiTensor &aWaveforms, uint32_t aMarginLeft, uint32_t aMarginRight, MultiTensor &aThresholds, MemoryBuffer &aWaveformLength,
                         MemoryBuffer &aBlockSizes, uint32_t aMaxWaveformLength, uint32_t aMaxBlockSize );

    /// @brief Quadratic interpolation algorithm abstraction
    ///
    /// @param aOutX Output tensor for the x-axis value of the interpolated peak
    /// @param aOutY Output tensor for the xyaxis value of the interpolated peak
    /// @param aWaveforms Input waveforms
    /// @param aWaveformLength Product of the lengths of the first rank-1 dimensions of `aWaveforms`
    /// @param aBlockSizes Product of the lengths of the first rank-1 dimensions of `aWaveforms`
    /// @param aMaxWaveformLength Maximum value of the `aWaveformLength` parameter
    /// @param aMaxBlockSize Maximum value of the `aBlockSizes` parameter
    ///
    void QuadraticInterpolationOp( MultiTensor &aOutX, MultiTensor &aOutY, MultiTensor &aWaveforms, MemoryBuffer &aWaveformLength, MemoryBuffer &aBlockSizes,
                                   uint32_t aMaxWaveformLength, uint32_t aMaxBlockSize );

    /// @brief Filter detections by removing detections that are too close
    ///
    /// @param aOut Output tensor
    /// @param aDetections Input detections buffer
    /// @param aMaskLength Minimum distance between two consecutive detections
    /// @param aBlockSizes Product of the lengths of the first rank-1 dimensions of `aWaveforms`
    /// @param aElementCount Trace length
    /// @param aMaxBlockSize Maximum value of the `aBlockSizes` parameter
    ///
    void FilterDetectionOp( MultiTensor &aOut, MultiTensor &aDetections, uint32_t aMaskLength, MemoryBuffer &aBlockSizes, MemoryBuffer &aElementCount, uint32_t aMaxBlockSize );

    /// @brief Packet header generation abstraction
    ///
    /// @param aOut Output tensor
    /// @param aVersion  Format version for the specific ID
    /// @param aID  Packet identifier
    /// @param aSampleCount Configured number of sample per short waveform
    /// @param aDetectionCount Number of detections per waveform
    /// @param aMaxDetectionCount Configured number of detections per waveform
    /// @param aPDNumber Channel (According to configured Partition and LAU)
    /// @param aLaserAngle Angle or Laser identification
    /// @param aWaveformNoise Standard deviation of the noise present in the traces
    /// @param aWaveformBaseline Average level of the traces at the beginning of the traces
    /// @param aBufferSizes Product of the lengths of the first rank-1 dimensions of `mPDNumber`
    /// @param aMaxBufferSize Maximum value of the `aBufferSizes` parameter
    /// @param aHeaderSizes Length of the header
    ///
    void GeneratePacketHeaderOp( MultiTensor &aOut, MemoryBuffer &aVersion, MemoryBuffer &aID, MemoryBuffer &aSampleCount, MemoryBuffer &aMaxDetectionCount,
                                 MultiTensor &aDetectionCount, MultiTensor &aPDNumber, MultiTensor &aLaserAngle, MultiTensor &aWaveformNoise, MultiTensor &aWaveformBaseline,
                                 MemoryBuffer &aBufferSizes, uint32_t aMaxBufferSize, MemoryBuffer &aHeaderSizes );

    /// @brief Detection structure abstraction
    ///
    /// @param aOut Output tensor
    /// @param aWaveforms Input waveform buffer
    /// @param aThresholds Threshold values
    /// @param aDistance Interpolated distance
    /// @param aAmplitude Interpolated amplitude
    /// @param aPulseIsSaturated  Determine whether the pulse is saturated
    /// @param aPreSaturationBaseline  Pulse baseline before saturation
    /// @param aPostSaturationBaseline Pulse baseline after saturation
    /// @param aSaturatedPulseLength Length of the saturation plateau
    /// @param aLastUnsaturatedSample Amplitude of the last sample before the saturation plateau
    /// @param aValidDetections List of positions representing valid detections
    /// @param aValidDetectionCount Number of valid detections per waveform
    /// @param aMaxDetectionCount Maximum number of valid detections per waveform
    /// @param mNeighbourCount Number of samples to include on the left and right of the peak
    /// @param aBlockSizes Product of the lengths of the first rank-1 dimensions of `aWaveforms`
    /// @param aElementCount Trace length
    /// @param aPacketSizes Full size of short waveforms
    /// @param aMaxBlockSize Maximum value of the `aBlockSizes` parameter
    ///
    void GenerateShortWaveformPacketsOp( MultiTensor &aOut, MultiTensor &aWaveforms, MultiTensor &aThresholds, MultiTensor &aDistance, MultiTensor &aAmplitude,
                                         MultiTensor &aPulseIsSaturated, MultiTensor &aPreSaturationBaseline, MultiTensor &aPostSaturationBaseline,
                                         MultiTensor &aSaturatedPulseLength, MultiTensor &aLastUnsaturatedSample, MultiTensor &aValidDetections, MultiTensor &aValidDetectionCount,
                                         uint32_t aMaxDetectionCount, uint32_t aNeighbourCount, MemoryBuffer &aBlockSizes, MemoryBuffer &aElementCount, MemoryBuffer &aPacketSizes,
                                         uint32_t aMaxBlockSize, uint32_t aMaxWaveformLength );
} // namespace LTSE::SensorModel
