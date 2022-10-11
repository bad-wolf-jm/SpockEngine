#pragma once

#include "Cuda/CudaAssert.h"
#include <cstdint>

const uint32_t SAMPLES_PER_CHANNEL = 11;
const uint32_t ECHOES_PER_CHANNEL  = 5;

/// @struct sWaveformPacketID
///
/// Documentation: http://svleddarapp05/cb/tracker/131614?view_id=-11&subtreeRoot=47919
///
struct sWaveformPacketID
{
    uint32_t mHeader             = 0x00000000; //!< Header, packed as follows: ID(4), VERSION(4), MAX_DETECTIONS(6), SAMPLE_COUNT(12), DETECTION_COUNT(6)
    uint16_t mFrameNumber        = 0x0000;     //!< Frame number from the CSI2 data
    uint16_t mPackedPDLaserAngle = 0x0000;     //!< Coordinates, packed as follows: PD(6), LASER_ANGLE(9), REMOVED_DETECTION(1)
    uint32_t mTimestamp          = 0x00000000; //!< Free running counter (1us granularity)
    uint8_t mConfigurationID     = 0x00;       //!< Unused by LibSPP
    uint8_t mFrameID             = 0x00;       //!< Unused by LibSPP
    uint8_t mOpticalID           = 0x00;       //!< Unused by LibSPP
    uint8_t mAcquisitionID       = 0x00;       //!< Unused by LibSPP
    uint16_t mTraceNoiseLevel    = 0x0000;     //!< Std of the noise present in the traces
    int16_t mTraceBaseLevel      = 0;          //!< Average level of the traces at the beginning of the traces

    void LTSE_CUDA_HOST_DEVICE_FUNCTION_DEF inline UnpackHeader( uint32_t &aID, uint32_t &aVersion, uint32_t &aDetectionCount, uint32_t &aMaxDetectionCount,
                                                                 uint32_t &aSampleCount )
    {
        aID                = ( mHeader & 0xF0000000 ) >> 28; // Packet identifier (4 bits)
        aVersion           = ( mHeader & 0x0F000000 ) >> 24; // Format version for the specific ID (4 bits)
        aMaxDetectionCount = ( mHeader & 0x00FC0000 ) >> 18; // Configured number of detections per waveform (6 bits)
        aSampleCount       = ( mHeader & 0x0003FFC0 ) >> 6;  // Configured number of sample per short waveform (12 bits)
        aDetectionCount    = ( mHeader & 0x0000003F ) >> 0;  // Actual number of valid detections in the current waveform packet (6 bits)
    }

    void LTSE_CUDA_HOST_DEVICE_FUNCTION_DEF inline PackHeader( uint32_t aID, uint32_t aVersion, uint32_t aDetectionCount, uint32_t aMaxDetectionCount, uint32_t aSampleCount )
    {
        mHeader = static_cast<uint32_t>( ( ( aID & 0x00000000F ) << 28 ) | ( ( aVersion & 0x00000000F ) << 24 ) | ( ( aMaxDetectionCount & 0x00000003F ) << 18 ) |
                                         ( ( aSampleCount & 0x00000FFF ) << 6 ) | ( ( aDetectionCount & 0x000003F ) << 0 ) );
    }

    void LTSE_CUDA_HOST_DEVICE_FUNCTION_DEF inline UnpackPDLaserAngle( uint32_t &aPD, uint32_t &aLaserAngle, uint32_t &aIsRemoved )
    {
        aPD         = ( mPackedPDLaserAngle & 0xFC00 ) >> 10; // Channel (According to configured Partition and LAU) (6 bits)
        aLaserAngle = ( mPackedPDLaserAngle & 0x03FE ) >> 1;  // Angle or Laser identification (9 bits)
        aIsRemoved  = ( mPackedPDLaserAngle & 0x0001 ) >> 0;  // Angle or Laser identification (1 bit)
    }

    void LTSE_CUDA_HOST_DEVICE_FUNCTION_DEF inline PackPDLaserAngle( uint32_t aPD, uint32_t aLaserAngle, uint32_t aIsRemoved )
    {
        mPackedPDLaserAngle = static_cast<uint16_t>( ( ( aPD & 0x00000003F ) << 10 ) | ( ( aLaserAngle & 0x0000001FF ) << 1 ) | ( ( aIsRemoved & 0x00000001 ) << 0 ) );
    }
};

/// \struct sPulse_t
/// \brief  pulse of the sWaveformPacket_t structure
struct sPulse
{
    uint16_t mInterpolatedDistance;   //!< (10.6) Interpolate Scaled distance of the pulse
    int16_t mLastUnsaturatedSample;   //!< Last unsaturated sample before a saturated pulse
    int16_t mPulseBaseLevel;          //!< Baseline of the waveform (before a saturated pulse)
    int16_t mBaselineAfterSaturation; //!< Baseline of the waveform after a saturated pulse
    int32_t mAmplitude;               //!< Extrapolated Amplitude of the peak by the saturation
    uint16_t mMaxIndex;               //!< Index of the maximum point of the pulse (the 16 bit represent the saturated stated of the pulse 1:saturated 0:Not Saturated)
    uint16_t mOffset;                 //!< Offset between the starting point of the traces and the beginning of the small traces
};

/// \struct sWaveform_t
/// \brief  waveform of the sWaveformPacket_t structure
struct sWaveform
{
    sPulse mPulse;                                //!< Pulses
    int16_t mProcessedTrace[SAMPLES_PER_CHANNEL]; //!< Processed traces
    int16_t mRawTrace[SAMPLES_PER_CHANNEL];       //!< Threshold, Raw Traces or post static noise Traces
};

struct sWaveformPacket
{
    sWaveformPacketID mHeader;               //!> Current header
    sWaveform mWaveform[ECHOES_PER_CHANNEL]; //!> Processed traces
};
