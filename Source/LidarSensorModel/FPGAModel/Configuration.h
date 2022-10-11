#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "Cuda/MultiTensor.h"

namespace fs = std::filesystem;
using namespace LTSE::Cuda;

namespace LTSE::SensorModel
{
    struct sTileConfiguration
    {

        std::vector<uint32_t> mLCA3PhotodetectorIndices = {};

        bool mEnableStaticNoiseRemoval    = false; //!< Enable or disable static noise removal
        uint32_t mStaticNoiseDatasetIndex = 0;
        uint32_t mStaticGlobalOffset      = 0;
        std::vector<int16_t> mStaticNoiseTemplateData{};

        bool mEnableBlinder = false; //!< Enable or disable the blinder

        enum class PeakDetectorMode : uint8_t
        {
            SURFACE_CLOUD       = 0,
            COMPRESSED_RAW_DATA = 1
        };

        PeakDetectorMode mPeakDetectorMode = PeakDetectorMode::COMPRESSED_RAW_DATA; //!< selects transmission mode: 0) surface cloud, 1) compressed raw data
        uint32_t mDetectionCount           = 0; //!< number of detections to transmit; the closest and n_detections-1 largest detections are selected; 0=no restriction
        uint32_t mNeighbourCount           = 0; //!< Number of raw samples on each side of the detection to transmit with the detection info

        sTileConfiguration() = default;

        ~sTileConfiguration() = default;

        uint32_t GetSampleCount() const { return 2 * mNeighbourCount + 1; }
    };

    struct sFPGAConfiguration
    {
        std::string mVersion = "";
        std::string mName    = "";
        std::string mDate    = "";

        std::vector<uint32_t> mPhotodetectorIndices     = {};
        std::vector<uint32_t> mLCA3PhotodetectorIndices = {};

        struct Statistics
        {
            uint32_t mWindowSize = 0; //!< Number of samples that are considered for the calculation of the trace statistics. This value should be set to (2 ^ win_size) in the
                                      //!< configuration file. The samples are picked at the beginning of the trace
        } mStatistics;

        struct StaticNoiseRemoval
        {
            static constexpr uint32_t CHANNEL_COUNT = 64;

            uint32_t mGlobalOffset    = 0;
            uint32_t mDatasetSelector = 0;
            std::vector<uint32_t> mTemplateOffsets{};
            std::vector<uint32_t> mPhotoDetectorMapping{};
        } mStaticNoiseRemoval;

        struct Blinder
        {
            float mThreshold     = 0;    //!< Input threshold, above which blind mode is triggered. Should be set close to close to ADC clipping level of 8191.
            uint32_t mClipPeriod = 0;    //!< Number of consecutive samples above threshold necessary to trigger blind mode
            uint32_t mWindowSize = 0;    //!< Number of samples to consider for the calculation of the blind value and baseline levels. This should be set to a power of 2, and
                                         //!< corresponds to 2 ^ win_size) in the configuration file.
            uint32_t mBaselineDelay = 0; //!< Number of samples to skip before computing the baseline after regardless the length of th e saturation plateau.
            int32_t mGuardLength0   = 0; //!< Number of samples between the first clipped sample observed (i.e. above `mThreshold`) and the last sample of the averaging window used
                                         //!< to calculate `bl_before`
            int32_t mGuardLength1 = 0;   //!< Number of samples between the last clipped sample observed (i.e. above th_blind_on) and the end of the blind period.
        } mBlinder;

        struct Filter
        {
            std::vector<float> mCoefficients = {}; //!< Filter coefficients.

            uint32_t mOutputDelay = 0; //!< Number of clock cycles to delay the sideinfo between its input and output (in addition to any intrinsic processing delay to keep it
                                       //!< aligned with the output data)
        } mFilter;

        struct Cfar
        {
            uint32_t mGuardLength  = 0; //!< Number of samples on each side of the sample under investigation ignored, not part of the reference window
            uint32_t mWindowLength = 0; //!< NUmber of samples to consider for the calaulation of local statistics. This should be a power of 2. The value should be set to
                                        //!< (2^ref_length) from the configuration file. This value does not include the guard interval.
            float mThresholdFactor = 0; //!< Factor to be applied to calculated threshold
            float mMinStd          = 0; //!< Minimum noise standard deviation for threshold calculation
            uint32_t mNSkip        = 0; //!< Number of values to skip in sorted list when selecting reference statistics
        } mCfar;

        struct PeakDetector
        {
            enum class Mode : uint8_t
            {
                SURFACE_CLOUD       = 0,
                COMPRESSED_RAW_DATA = 1
            };

            Mode mMode = Mode::COMPRESSED_RAW_DATA; //!< selects transmission mode: 0) surface cloud, 1) compressed raw data

            bool mEnableInterpolation = false; //!< enable quadratic interpolation of detection results
            bool mIgnoreCfarData      = 0;     //!< Ignore thresholding information when detecting peaks.
            uint32_t mMaskLength      = 0;     //!< Number of samples to mask before and after the detected bin
            uint32_t mMarginStart     = 0;     //!< Ignore all peaks whise index is in [0, mMarginStart].
            uint32_t mMarginEnd       = 0;     //!< Ignore all peaks whise index is to [Trace_length - mMarginEnd, TraceLength].
        } mPeakDetector;

        sFPGAConfiguration() = default;

        ~sFPGAConfiguration() = default;

        std::vector<std::vector<float>> GetStaticNoiseTemplates( sTileConfiguration const &aTileConfiguration, uint32_t aTraceLength ) const;
        uint32_t GetHeaderSize( sTileConfiguration const &aTileConfiguration ) const;
        uint32_t GetPacketLength( sTileConfiguration const &aTileConfiguration ) const;
    };

    // struct sDetection
    // {
    //     bool mPulseIsSaturated            = false; //!< Saturation flag
    //     uint16_t mInterpolatedDistance    = 0;     //!< (10.6) Interpolate Scaled distance of the pulse
    //     int16_t mLastUnsaturatedSample    = 0;     //!< Last unsaturated sample before a saturated pulse
    //     int16_t mBaselineBeforeSaturation = 0;     //!< Baseline of the waveform (before a saturated pulse)
    //     int16_t mBaselineAfterSaturation  = 0;     //!< Baseline of the waveform after a saturated pulse
    //     int32_t mSaturationLength         = 0;     //!< Extrapolated Amplitude of the peak by the saturation
    //     int32_t mAmplitude                = 0;     //!< Extrapolated Amplitude of the peak by the saturation
    //     uint16_t mBin                     = 0;     //!< Rounded mInterpolatedDistance
    //     uint16_t mOffset                  = 0;     //!< Offset between the starting point of the traces and the beginning of the small traces
    // };

} // namespace LTSE::SensorModel