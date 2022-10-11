/// @file   FPGAModel.h
///
/// @brief  FPGA processing.
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2022 LeddarTech Inc. All rights reserved.

#pragma once

#include "Core/Math/Types.h"

#include "Core/EntityRegistry/Registry.h"

#include "Configuration.h"
#include "TensorOps/NodeComponents.h"
#include "TensorOps/Scope.h"

namespace LTSE::SensorModel
{
    using namespace LTSE::TensorOps;

    /// @brief Packet header generator.
    ///
    /// Generate the header portion of the waveform packets sent by the FPGA to the software signal processing library.
    /// All input data uses uint32_t as the underlying type. In cases where the actual data contained in a given field
    /// is to be packed using fewer bits, the information is included.
    ///
    struct sPacketHeaderNodeComponent
    {
        OpNode mID{};                //!< Packet identifier (4 bits)
        OpNode mVersion{};           //!< Format version for the specific ID (4 bits)
        OpNode mSampleCount{};       //!< Configured number of sample per short waveform (12 bits)
        OpNode mMaxDetectionCount{}; //!< Configured number of detections per waveform (6 bits)
        OpNode mDetectionCount{};    //!< Number of detection in this packet
        OpNode mPDNumber{};          //!< Channel (According to configured Partition and LAU) (6 bits)
        OpNode mLaserAngle{};        //!< Angle or Laser identification (9 bits)
        OpNode mFrameNumber{};       //!< Frame number from the CSI2 data
        OpNode mWaveformNoise{};     //!< Standard deviation of the noise present in the traces
        OpNode mWaveformBaseline{};  //!< Average level of the traces at the beginning of the traces

        uint32_t mMaxBlockSize = 0; //!< Maximum value of `mBlockSizes`
        OpNode mBlockSizes{};       //!< Product of the lengths of the first rank-1 dimensions of `mPDNumber`
        OpNode mHeaderSizes{};      //!< Length of the header

        sPacketHeaderNodeComponent()  = default;
        ~sPacketHeaderNodeComponent() = default;
    };

    /// @brief Operation controller for the packet header generator.
    struct sPacketHeaderOperationController : public LTSE::TensorOps::sGraphOperationController
    {
        void Run();
    };

    struct sBlinderNodeComponent
    {
        sFPGAConfiguration::Blinder mConfiguration{};

        OpNode mInputWaveforms{};
        OpNode mPulseIsSaturated{};
        OpNode mPreSaturationBaseline{};
        OpNode mPostSaturationBaseline{};
        OpNode mSaturatedPulseLength{};
        OpNode mLastUnsaturatedSample{};

        uint32_t mMaxBlockSize = 0;
        OpNode mBlockSizes{};
        OpNode mElementCount{};

        sBlinderNodeComponent()  = default;
        ~sBlinderNodeComponent() = default;
    };

    struct sBlinderOperationController : public LTSE::TensorOps::sGraphOperationController
    {
        void Run();
    };

    /// @brief Cfar node component
    struct sCfarNodeComponent
    {
        sFPGAConfiguration::Cfar mConfiguration{}; //!< Cfar configuration

        OpNode mInputWaveforms{}; //!< Input waveforms.

        OpNode mBlockSizes{};     //!< Product of the lengths of the first rank-1 dimensions of `mInputWaveforms`
        uint32_t mMaxBlockSize{}; //!< Maximum value of `mBlockSizes`

        OpNode mElementCount{};        //!< Waveform length
        uint32_t mMaxWaveformLength{}; //!< Maximum value of `mElementCount`

        sCfarNodeComponent()  = default;
        ~sCfarNodeComponent() = default;
    };

    struct sCfarOperationController : public LTSE::TensorOps::sGraphOperationController
    {
        void Run();
    };

    struct sPeakDetectorNodeComponent
    {
        sFPGAConfiguration::PeakDetector mConfiguration{}; //!< peak detector configuration

        OpNode mInputWaveforms{}; //!< Input waveforms.
        OpNode mCfarThresholds{}; //!< Peak thresholds, usually the output of a cfar-like algorithm.

        OpNode mBlockSizes{};     //!< Product of the lengths of the first rank-1 dimensions of `mInputWaveforms`
        uint32_t mMaxBlockSize{}; //!< Maximum value of `mBlockSizes`

        OpNode mElementCount{};        //!< Waveform length
        uint32_t mMaxWaveformLength{}; //!< Maximum value of `mElementCount`

        sPeakDetectorNodeComponent()  = default;
        ~sPeakDetectorNodeComponent() = default;
    };

    struct sPeakDetectorOperationController : public LTSE::TensorOps::sGraphOperationController
    {
        void Run();
    };

    struct sQuadraticInterpolationNodeComponent
    {
        OpNode mInputWaveforms{}; //!< Input waveforms.

        OpNode mDistances{};  //!< Input waveforms.
        OpNode mAmplitudes{}; //!< Input waveforms.

        OpNode mBlockSizes{};     //!< Product of the lengths of the first rank-1 dimensions of `mInputWaveforms`
        uint32_t mMaxBlockSize{}; //!< Maximum value of `mBlockSizes`

        OpNode mElementCount{};        //!< Waveform length
        uint32_t mMaxWaveformLength{}; //!< Maximum value of `mElementCount`

        sQuadraticInterpolationNodeComponent()  = default;
        ~sQuadraticInterpolationNodeComponent() = default;
    };

    struct sQuadraticInterpolationOperationController : public LTSE::TensorOps::sGraphOperationController
    {
        void Run();
    };

    struct sFilterDetectionsNodeComponent
    {
        sFPGAConfiguration::PeakDetector mConfiguration{}; //!< peak detector configuration
        OpNode mDetections{};                              //!< Input waveforms.

        OpNode mBlockSizes{};     //!< Product of the lengths of the first rank-1 dimensions of `mInputWaveforms`
        uint32_t mMaxBlockSize{}; //!< Maximum value of `mBlockSizes`

        OpNode mElementCount{}; //!< Waveform length

        sFilterDetectionsNodeComponent()  = default;
        ~sFilterDetectionsNodeComponent() = default;
    };

    struct sFilterDetectionsOperationController : public LTSE::TensorOps::sGraphOperationController
    {
        void Run();
    };

    struct sStructureDetectionsNodeComponent
    {
        OpNode mInputWaveforms{};         //!< Input waveforms.
        OpNode mThresholds{};             //!< Threshold values.
        OpNode mDistance{};               //!< Distance from peak detector
        OpNode mAmplitude{};              //!< Amplitude from peak detector
        OpNode mPreSaturationBaseline{};  //!< Baseline before saturation plateau (from blinder)
        OpNode mPostSaturationBaseline{}; //!< Baseline after saturation plateau (from blinder)
        OpNode mSaturatedPulseLength{};   //!< Length of saturation plateau (from blinder)
        OpNode mLastUnsaturatedSample{};  //!< Amplitude of last sample before saturation plateau (from blinder)
        OpNode mPulseIsSaturated{};       //!< Saturation flag (from blinder)
        OpNode mValidDetections{};        //!< Valid detection flag (from peak detector)
        OpNode mValidDetectionCount{};    //!< Valid detection count (from peak detector)
        uint32_t mMaxDetectionCount{};    //!< Maximum number of detections
        uint32_t mNeighbourCount{};       //!< Number of samples to include on the left and right of the peak

        OpNode mBlockSizes{};     //!< Product of the lengths of the first rank-1 dimensions of `mInputWaveforms`
        uint32_t mMaxBlockSize{}; //!< Maximum value of `mBlockSizes`

        OpNode mElementCount{};        //!< Waveform length
        OpNode mPacketSizes{};         //!< Waveform length
        uint32_t mMaxWaveformLength{}; //!< Maximum value of `mElementCount`

        sStructureDetectionsNodeComponent()  = default;
        ~sStructureDetectionsNodeComponent() = default;
    };

    struct sStructureDetectionsOperationController : public LTSE::TensorOps::sGraphOperationController
    {
        void Run();
    };

    struct sPackageDetectionsNodeComponent
    {
        uint32_t mNeighbourCount{}; //!< Number of samples to include on the left and the right of the peak

        OpNode mInputWaveforms{}; //!< Input waveforms
        OpNode mThresholds{};     //!< Cfar thresholds
        OpNode mDetections{};     //!< Detection structures

        OpNode mBlockSizes{};     //!< Product of the lengths of the first rank-1 dimensions of `mInputWaveforms`
        uint32_t mMaxBlockSize{}; //!< Maximum value of `mBlockSizes`

        OpNode mElementCount{};        //!< Waveform length
        uint32_t mMaxWaveformLength{}; //!< Maximum value of `mElementCount`

        sPackageDetectionsNodeComponent()  = default;
        ~sPackageDetectionsNodeComponent() = default;
    };

    struct sPackageDetectionsOperationController : public LTSE::TensorOps::sGraphOperationController
    {
        void Run();
    };

    /// @brief Generate the positional metadata for the given tile configuration
    ///
    /// This operation consists in generating the laser angle indices, as well as the photodetector channel indices
    /// and the frame index for the given tile configuration. This node assumes that the waveform buffer has one layer
    /// for every submitted tile, and that the shape of each layer is (LASER_ANGLE_COUNT, PD_COUNT, WAVEFORM_LENGTH).
    ///
    /// @param aScope Parent computation scope
    /// @param aConfiguration Global FPGA configuration
    /// @param aTileConfiguration Per-tile configuration values
    /// @param aWaveforms Waveform buffer.
    ///
    std::tuple<OpNode, OpNode, OpNode> TileMetadata( Scope &aScope, sFPGAConfiguration const &aConfiguration, sTileConfiguration const &aTileConfiguration,
                                                     OpNode const &aWaveforms );

    /// @brief Compute the baseline and noise value for the given waveform buffer
    ///
    /// This node assumes that the waveform buffer has one layer for every submitted tile, and that the shape of each layer
    /// is (LASER_ANGLE_COUNT, PD_COUNT, WAVEFORM_LENGTH).
    ///
    /// @param aScope Parent computation scope
    /// @param aConfiguration Global FPGA configuration
    /// @param aTileConfiguration Per-tile configuration values
    /// @param aWaveforms Waveform buffer.
    /// @param aFixed Fixed point precision.
    ///
    std::tuple<OpNode, OpNode> ComputeWaveformStatistics( Scope &aScope, sFPGAConfiguration const &aConfiguration, sTileConfiguration const &aTileConfiguration,
                                                          OpNode const &aWaveforms, uint32_t aFixed );

    /// @brief Remove the static noise from the given waveform buffer
    ///
    /// This node assumes that the waveform buffer has one layer for every submitted tile, and that the shape of each layer
    /// is (LASER_ANGLE_COUNT, PD_COUNT, WAVEFORM_LENGTH).
    ///
    /// @param aScope Parent computation scope
    /// @param aConfiguration Global FPGA configuration
    /// @param aTileConfiguration Per-tile configuration values
    /// @param aWaveforms Waveform buffer.
    ///
    OpNode RemoveStaticNoise( Scope &aScope, sFPGAConfiguration const &aConfiguration, sTileConfiguration const &aTileConfiguration, OpNode const &aWaveforms );

    /// @brief Apply the blinder algorithm to the given waveform buffer
    ///
    /// This node assumes that the waveform buffer has one layer for every submitted tile, and that the shape of each layer
    /// is (LASER_ANGLE_COUNT, PD_COUNT, WAVEFORM_LENGTH). This nodes produces a few artefacts which can be retrieved by
    /// calling `aBlinderNode.Get<sBlinderNodeComponent>()` in order to use them in further processing.
    ///
    /// @param aScope Parent computation scope
    /// @param aConfiguration Global FPGA configuration
    /// @param aTileConfiguration Per-tile configuration values
    /// @param aWaveforms Waveform buffer.
    ///
    OpNode Blinder( Scope &aScope, sFPGAConfiguration const &aConfiguration, sTileConfiguration const &aTileConfiguration, OpNode const &aWaveforms );

    /// @brief Apply the matched filter algorithm to the given waveform buffer
    ///
    /// This node assumes that the waveform buffer has one layer for every submited tile, and that the shape of each layer
    /// is (LASER_ANGLE_COUNT, PD_COUNT, WAVEFORM_LENGTH).
    ///
    /// @param aScope Parent computation scope
    /// @param aConfiguration Global FPGA configuration
    /// @param aTileConfiguration Per-tile configuration values
    /// @param aWaveforms Waveform buffer.
    ///
    OpNode Filter( Scope &aScope, sFPGAConfiguration const &aConfiguration, sTileConfiguration const &aTileConfiguration, OpNode const &aWaveforms );

    /// @brief Apply the Cfar algorithm to the given waveform buffer
    ///
    /// This node assumes that the waveform buffer has one layer for every submited tile, and that the shape of each layer
    /// is (LASER_ANGLE_COUNT, PD_COUNT, WAVEFORM_LENGTH).
    ///
    /// @param aScope Parent computation scope
    /// @param aConfiguration Global FPGA configuration
    /// @param aTileConfiguration Per-tile configuration values
    /// @param aFilteredWaveforms Waveform buffer.
    ///
    OpNode Cfar( Scope &aScope, sFPGAConfiguration const &aConfiguration, sTileConfiguration const &aTileConfiguration, OpNode const &aFilteredWaveforms );

    /// @brief Apply the peak detector algorithm to the given waveform buffer
    ///
    /// This node assumes that the waveform buffer has one layer for every submitted tile, and that the shape of each layer
    /// is (LASER_ANGLE_COUNT, PD_COUNT, WAVEFORM_LENGTH).
    ///
    /// @param aScope Parent computation scope
    /// @param aConfiguration Global FPGA configuration
    /// @param aTileConfiguration Per-tile configuration values
    /// @param aWaveforms Waveform buffer.
    /// @param aThresholds Thresholds buffer.
    ///
    OpNode DetectPeaks( Scope &aScope, sFPGAConfiguration const &aConfiguration, sTileConfiguration const &aTileConfiguration, OpNode const &aWaveforms,
                        OpNode const &aThresholds );

    /// @brief Generate the waveform packet header
    ///
    /// This node assumes that the waveform buffer has one layer for every submitted tile, and that the shape of each
    /// layer is (LASER_ANGLE_COUNT, PD_COUNT). The waveform buffer should have shape (LASER_ANGLE_COUNT, PD_COUNT, WAVEFORM_LENGTH).
    ///
    /// @param aScope Parent computation scope
    /// @param aConfiguration Global FPGA configuration
    /// @param aTileConfiguration Per-tile configuration values
    /// @param aPDNumber Buffer containing the appropriate photodetector numbers.
    /// @param aLaserAngle Buffer containing the appropriate laser angle numbers.
    /// @param aFrameNumber Buffer containing the appropriate frame numbers.
    /// @param aWaveformNoise Buffer containing the appropriate noise values
    /// @param aWaveformBaseline Buffer containing the appropriate baseline values
    /// @param aWaveforms Waveform buffer.
    ///
    OpNode GeneratePacketHeaders( Scope &aScope, sFPGAConfiguration const &aConfiguration, sTileConfiguration const &aTileConfiguration, OpNode const &aDetectionCount,
                                  OpNode const &aPDNumber, OpNode const &aLaserAngle, OpNode const &aFrameNumber, OpNode const &aWaveformNoise, OpNode const &aWaveformBaseline,
                                  OpNode const &aWaveforms );

    /// @brief Apply quadratic niterpolation algorithm to the given waveform buffer
    ///
    /// This node assumes that the waveform buffer has one layer for every submitted tile, and that the shape of each layer
    /// is (LASER_ANGLE_COUNT, PD_COUNT, WAVEFORM_LENGTH).
    ///
    /// @param aScope Parent computation scope
    /// @param aConfiguration Global FPGA configuration
    /// @param aTileConfiguration Per-tile configuration values
    /// @param aWaveforms Waveform buffer.
    ///
    std::tuple<OpNode, OpNode> QuadraticInterpolation( Scope &aScope, sFPGAConfiguration const &aConfiguration, sTileConfiguration const &aTileConfiguration,
                                                       OpNode const &aWaveforms );

    /// @brief Generate detection structures from data collected in previous passes
    ///
    /// This node assumes that the waveform buffer has one layer for every submitted tile, and that the shape of each
    /// layer is (LASER_ANGLE_COUNT, PD_COUNT). The waveform buffer should have shape (LASER_ANGLE_COUNT, PD_COUNT, WAVEFORM_LENGTH).
    ///
    /// @param aScope Parent computation scope
    /// @param aConfiguration Global FPGA configuration
    /// @param aTileConfiguration Per-tile configuration values
    /// @param aWaveforms Input waveforms.
    /// @param aDistance Detection distance
    /// @param aAmplitude Detection amplitude
    /// @param aBaselineBeforeSaturation Baseline before the saturation plateau
    /// @param aBaselineAfterSaturation Baseline after the saturation plateau
    /// @param aPulseLength Length of saturation plateau.
    /// @param aLastUnsaturatedSample Value of last sample before saturation plateau.
    /// @param aIsSaturated Saturation flag.
    /// @param aValidDetections Valid detection flag.
    /// @param aValidDetectionCount Number of valid detections.
    ///
    OpNode GenerateShortWaveforms( Scope &aScope, sFPGAConfiguration const &aConfiguration, sTileConfiguration const &aTileConfiguration, OpNode const &aWaveforms,
                                   OpNode const &aThresholds, OpNode const &aDistance, OpNode const &aAmplitude, OpNode const &aBaselineBeforeSaturation,
                                   OpNode const &aBaselineAfterSaturation, OpNode const &aPulseLength, OpNode const &aLastUnsaturatedSample, OpNode const &aIsSaturated,
                                   OpNode const &aValidDetections, OpNode const &aValidDetectionCount );

    /// @brief Main FPGA process node
    ///
    /// Here we run the entire FPGA signal processing pipeline for the given tiile configutaions. This node assumes
    /// that the waveform buffer has one layer for every submitted tile, and that the shape of each layer is
    /// (LASER_ANGLE_COUNT, PD_COUNT, WAVEFORM_LENGTH).
    ///
    /// @param aScope Parent computation scope
    /// @param aConfiguration Global FPGA configuration
    /// @param aTileConfiguration Per-tile configuration values
    /// @param aWaveforms Waveform buffer.
    ///
    OpNode FPGAProcess( Scope &aScope, sFPGAConfiguration const &aConfiguration, sTileConfiguration const &aTileConfiguration, OpNode const &aWaveforms );
} // namespace LTSE::SensorModel