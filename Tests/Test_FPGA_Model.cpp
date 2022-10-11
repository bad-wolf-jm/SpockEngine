#include <catch2/catch_test_macros.hpp>

#include <array>
#include <filesystem>
#include <iostream>
#include <numeric>

#include "TestUtils.h"

#include "Cuda/MemoryPool.h"
#include "Cuda/MultiTensor.h"

#include "TensorOps/NodeComponents.h"
#include "TensorOps/Scope.h"

#include "LidarSensorModel/FPGAModel/Configuration.h"
#include "LidarSensorModel/FPGAModel/FPGAModel.h"

using namespace LTSE::Core;
using namespace LTSE::SensorModel;
using namespace LTSE::TensorOps;
using namespace TestUtils;

#include "Include/TestDataFunctions.h"

static sTileConfiguration CreateTileConfiguration()
{
    sTileConfiguration lNewConfiguration{};

    lNewConfiguration.mEnableStaticNoiseRemoval = true;
    lNewConfiguration.mStaticGlobalOffset       = 0u;
    lNewConfiguration.mStaticNoiseDatasetIndex  = 0u;

    std::string lStaticNoiseTemplateFilename = "sn_rem.bin";
    auto lStaticNoiseTemplatePath            = fs::path( "Tests/Data/FPGAConfiguration/" ) / lStaticNoiseTemplateFilename;
    auto lStaticNoiseTemplateDataStream      = std::basic_ifstream<uint8_t>( lStaticNoiseTemplatePath, std::ios::binary );
    auto lStaticNoiseTemplateData            = std::vector<uint8_t>( ( std::istreambuf_iterator<uint8_t>( lStaticNoiseTemplateDataStream ) ), std::istreambuf_iterator<uint8_t>() );
    lNewConfiguration.mStaticNoiseTemplateData = std::vector<int16_t>( lStaticNoiseTemplateData.size() / sizeof( int16_t ) );
    std::memcpy( lNewConfiguration.mStaticNoiseTemplateData.data(), lStaticNoiseTemplateData.data(), lStaticNoiseTemplateData.size() );

    lNewConfiguration.mEnableBlinder = true;

    lNewConfiguration.mPeakDetectorMode = sTileConfiguration::PeakDetectorMode::COMPRESSED_RAW_DATA;

    lNewConfiguration.mDetectionCount = 5u;
    lNewConfiguration.mNeighbourCount = 5u;

    std::vector<bool> lEnabledLCA3Partition{ true, false, true, false };
    uint32_t lPartition = 0;
    uint32_t lIndex     = 0;
    for( uint32_t i = 0; i < 32; i++ )
    {
        if( lIndex >= 64 )
        {
            bool lPartitionFound = false;
            while( !lPartitionFound )
            {
                lPartition      = ( lPartition + 1 ) % 4;
                lPartitionFound = lEnabledLCA3Partition[lPartition];
            }
            lIndex = lPartition;
        }
        lNewConfiguration.mLCA3PhotodetectorIndices.push_back( lIndex );
        lIndex += 4;
    }

    return lNewConfiguration;
}

static sFPGAConfiguration CreateFPGAConfiguration()
{
    sFPGAConfiguration lNewConfiguration{};

    lNewConfiguration.mName    = "Cyclops";
    lNewConfiguration.mDate    = "2022-04-25";
    lNewConfiguration.mVersion = "0xB0400400";

    lNewConfiguration.mStatistics.mWindowSize = 8;

    lNewConfiguration.mStaticNoiseRemoval.mPhotoDetectorMapping =
        std::vector<uint32_t>{ 0,  0, 1,  0, 2,  0, 3,  0, 4,  0, 5,  0, 6,  0, 7,  0, 8,  0, 9,  0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0,
                               16, 0, 17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23, 0, 24, 0, 25, 0, 26, 0, 27, 0, 28, 0, 29, 0, 30, 0, 31, 0 };

    lNewConfiguration.mStaticNoiseRemoval.mTemplateOffsets =
        std::vector<uint32_t>{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    lNewConfiguration.mBlinder.mThreshold     = 8050.0f;
    lNewConfiguration.mBlinder.mClipPeriod    = 1u;
    lNewConfiguration.mBlinder.mWindowSize    = 8u;
    lNewConfiguration.mBlinder.mBaselineDelay = 0u;
    lNewConfiguration.mBlinder.mGuardLength0  = 3u;
    lNewConfiguration.mBlinder.mGuardLength1  = 25u;

    auto lFilterCoefficients = std::vector<float>{ -113.0f, 2121.0f, 6579.0f, 14854.0f, 10966.0f, 520.0f, -2161.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    for( auto &lCoefficient : lFilterCoefficients )
        lNewConfiguration.mFilter.mCoefficients.push_back( lCoefficient / std::powf( 2.0f, 15.0f ) );

    lNewConfiguration.mCfar.mGuardLength     = 2u;
    lNewConfiguration.mCfar.mWindowLength    = 8u;
    lNewConfiguration.mCfar.mThresholdFactor = 6.0f;
    lNewConfiguration.mCfar.mMinStd          = 7.0f;
    lNewConfiguration.mCfar.mNSkip           = 1u;

    lNewConfiguration.mPeakDetector.mEnableInterpolation = true;
    lNewConfiguration.mPeakDetector.mIgnoreCfarData      = false;
    lNewConfiguration.mPeakDetector.mMaskLength          = 2u;
    lNewConfiguration.mPeakDetector.mMarginStart         = 21u;
    lNewConfiguration.mPeakDetector.mMarginEnd           = 8u;

    return lNewConfiguration;
}

TEST_CASE( "Tile Metadata", "[CORE_SENSOR_MODEL_FPGA]" )
{
    size_t lPoolSize = 1024 * 1024;
    Scope lScope( lPoolSize );

    std::vector<uint32_t> lDim1{ 32, 32, 100 };
    std::vector<uint32_t> lDim2{ 32, 32, 150 };
    sConstantValueInitializerComponent lInitializer{};
    lInitializer.mValue = (uint16_t)0;
    auto lWaveforms     = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( uint16_t ) ) );

    sFPGAConfiguration lConfiguration     = CreateFPGAConfiguration();
    sTileConfiguration lTileConfiguration = CreateTileConfiguration();

    auto [lPDNumbersNode, lLaserAnglesNode, lFrameNumberNode] = TileMetadata( lScope, lConfiguration, lTileConfiguration, lWaveforms );

    REQUIRE( lPDNumbersNode.Get<sMultiTensorComponent>().mValue.Shape().CountLayers() == 2 );
    REQUIRE( lLaserAnglesNode.Get<sMultiTensorComponent>().mValue.Shape().CountLayers() == 2 );
    REQUIRE( lFrameNumberNode.Get<sMultiTensorComponent>().mValue.Shape().CountLayers() == 2 );

    lScope.Run( { lPDNumbersNode, lLaserAnglesNode, lFrameNumberNode } );

    std::vector<uint32_t> lExpectedLaserAngles{};
    for( uint32_t i = 0; i < 32; i++ )
    {
        auto lValues = std::vector<uint32_t>( lTileConfiguration.mLCA3PhotodetectorIndices.size(), i );
        lExpectedLaserAngles.insert( lExpectedLaserAngles.end(), lValues.begin(), lValues.end() );
    }

    std::vector<uint32_t> lExpectedPDNumbers{};
    for( uint32_t i = 0; i < 32; i++ )
        lExpectedPDNumbers.insert( lExpectedPDNumbers.end(), lTileConfiguration.mLCA3PhotodetectorIndices.begin(), lTileConfiguration.mLCA3PhotodetectorIndices.end() );

    auto lPDNumbers0 = lPDNumbersNode.Get<sMultiTensorComponent>().mValue.FetchBufferAt<uint32_t>( 0 );
    REQUIRE( lExpectedPDNumbers == lPDNumbers0 );

    auto lLaserAngles0 = lLaserAnglesNode.Get<sMultiTensorComponent>().mValue.FetchBufferAt<uint32_t>( 0 );
    REQUIRE( lExpectedLaserAngles == lLaserAngles0 );

    auto lFrameNumberNode0 = lFrameNumberNode.Get<sMultiTensorComponent>().mValue.FetchBufferAt<uint32_t>( 0 );

    auto lPDNumbers1 = lPDNumbersNode.Get<sMultiTensorComponent>().mValue.FetchBufferAt<uint32_t>( 1 );
    REQUIRE( lExpectedPDNumbers == lPDNumbers1 );

    auto lLaserAngles1 = lLaserAnglesNode.Get<sMultiTensorComponent>().mValue.FetchBufferAt<uint32_t>( 1 );
    REQUIRE( lExpectedLaserAngles == lLaserAngles1 );

    auto lFrameNumberNode1 = lFrameNumberNode.Get<sMultiTensorComponent>().mValue.FetchBufferAt<uint32_t>( 1 );
    REQUIRE( true );
}

TEST_CASE( "Packet headers", "[CORE_SENSOR_MODEL_FPGA]" )
{
    size_t lPoolSize = 1024 * 1024;
    Scope lScope( lPoolSize );

    std::vector<uint32_t> lDim1{ 32, 32, 100 };
    std::vector<uint32_t> lDim2{ 32, 32, 150 };
    sConstantValueInitializerComponent lInitializer{};
    lInitializer.mValue = (uint16_t)0;
    auto lWaveforms     = MultiTensorValue( lScope, lInitializer, sTensorShape( { lDim1, lDim2 }, sizeof( uint16_t ) ) );

    sFPGAConfiguration lConfiguration     = CreateFPGAConfiguration();
    sTileConfiguration lTileConfiguration = CreateTileConfiguration();

    auto [lPDNumbersNode, lLaserAnglesNode, lFrameNumberNode] = TileMetadata( lScope, lConfiguration, lTileConfiguration, lWaveforms );

    sConstantValueInitializerComponent lInitializer1{};
    lInitializer1.mValue   = (float)0;
    auto lWaveformNoise    = MultiTensorValue( lScope, lInitializer1, sTensorShape( lPDNumbersNode.Get<sMultiTensorComponent>().mValue.Shape().mShape, sizeof( float ) ) );
    auto lWaveformBaseline = MultiTensorValue( lScope, lInitializer1, sTensorShape( lPDNumbersNode.Get<sMultiTensorComponent>().mValue.Shape().mShape, sizeof( float ) ) );

    sConstantValueInitializerComponent lInitializer2{};
    lInitializer1.mValue = (uint32_t)1;
    auto lDetectionCount = MultiTensorValue( lScope, lInitializer2, sTensorShape( lPDNumbersNode.Get<sMultiTensorComponent>().mValue.Shape().mShape, sizeof( uint32_t ) ) );

    auto lHeadersNode = GeneratePacketHeaders( lScope, lConfiguration, lTileConfiguration, lDetectionCount, lPDNumbersNode, lLaserAnglesNode, lFrameNumberNode, lWaveformNoise,
                                               lWaveformBaseline, lWaveforms );

    lScope.Run( lHeadersNode );

    std::vector<uint32_t> lExpectedLaserAngles{};
    for( uint32_t i = 0; i < 32; i++ )
    {
        auto lValues = std::vector<uint32_t>( lTileConfiguration.mLCA3PhotodetectorIndices.size(), i );

        lExpectedLaserAngles.insert( lExpectedLaserAngles.end(), lValues.begin(), lValues.end() );
    }

    std::vector<uint32_t> lExpectedPDNumbers{};
    for( uint32_t i = 0; i < 32; i++ )
        lExpectedPDNumbers.insert( lExpectedPDNumbers.end(), lTileConfiguration.mLCA3PhotodetectorIndices.begin(), lTileConfiguration.mLCA3PhotodetectorIndices.end() );

    {
        auto lHeaders0 = lHeadersNode.Get<sMultiTensorComponent>().mValue.FetchBufferAt<uint32_t[5]>( 0 );
        std::vector<uint32_t> lReceivedLaserAngles{};
        std::vector<uint32_t> lReceivedPDNumbers{};
        for( auto lH : lHeaders0 )
        {
            uint32_t lPacked = lH[1];
            lReceivedLaserAngles.push_back( static_cast<uint32_t>( ( lPacked >> 22 ) & 0x1FF ) );
            lReceivedPDNumbers.push_back( static_cast<uint32_t>( ( lPacked >> 16 ) & 0x3F ) );
        }

        REQUIRE( lReceivedLaserAngles == lExpectedLaserAngles );
        REQUIRE( lReceivedPDNumbers == lExpectedPDNumbers );
    }

    {
        auto lHeaders1 = lHeadersNode.Get<sMultiTensorComponent>().mValue.FetchBufferAt<uint32_t[5]>( 1 );
        std::vector<uint32_t> lReceivedLaserAngles{};
        std::vector<uint32_t> lReceivedPDNumbers{};

        for( auto lH : lHeaders1 )
        {
            uint32_t lPacked = lH[1];
            lReceivedLaserAngles.push_back( static_cast<uint32_t>( ( lPacked >> 22 ) & 0x1FF ) );
            lReceivedPDNumbers.push_back( static_cast<uint32_t>( ( lPacked >> 16 ) & 0x3F ) );
        }

        REQUIRE( lReceivedLaserAngles == lExpectedLaserAngles );
        REQUIRE( lReceivedPDNumbers == lExpectedPDNumbers );
    }
}

TEST_CASE( "Waveform statistics", "[CORE_SENSOR_MODEL_FPGA]" )
{
    size_t lPoolSize = 16 * 1024 * 1024;
    Scope lScope( lPoolSize );

    sFPGAConfiguration lConfiguration     = CreateFPGAConfiguration();
    sTileConfiguration lTileConfiguration = CreateTileConfiguration();

    auto lWaveforms          = GetWaveformNode( lScope );
    auto lBaselineOutputData = GetWaveformBaselineData();
    auto lNoiseOutputData    = GetWaveformNoiseData();

    auto [lBaseline, lNoiseStd] = ComputeWaveformStatistics( lScope, lConfiguration, lTileConfiguration, lWaveforms, 2 );
    lScope.Run( { lBaseline, lNoiseStd } );

    auto lBaseline0 = lBaseline.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    REQUIRE( VectorEqual( lBaseline0, lBaselineOutputData, 1.1f ) );

    auto lNoiseStd0 = lNoiseStd.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    REQUIRE( VectorEqual( lNoiseStd0, lNoiseOutputData, 1.1f ) );
}

TEST_CASE( "Static noise templates", "[CORE_SENSOR_MODEL_FPGA]" )
{
    sFPGAConfiguration lConfiguration     = CreateFPGAConfiguration();
    sTileConfiguration lTileConfiguration = CreateTileConfiguration();

    auto lStaticNoiseTemplates = lConfiguration.GetStaticNoiseTemplates( lTileConfiguration, 150 );
    REQUIRE( lStaticNoiseTemplates.size() == 32 );
    bool lCondition = true;
    for( auto &lT : lStaticNoiseTemplates )
        lCondition = lCondition && ( lT.size() == 150 );
    REQUIRE( lCondition );
}

TEST_CASE( "Static Noise Removal", "[CORE_SENSOR_MODEL_FPGA]" )
{
    size_t lPoolSize = 16 * 1024 * 1024;
    Scope lScope( lPoolSize );

    sFPGAConfiguration lConfiguration     = CreateFPGAConfiguration();
    sTileConfiguration lTileConfiguration = CreateTileConfiguration();

    auto lWaveforms  = GetWaveformNode( lScope );
    auto lOutputData = GetPostStaticNoiseWaveformData();

    auto lPostStaticNoise = RemoveStaticNoise( lScope, lConfiguration, lTileConfiguration, lWaveforms );
    lScope.Run( lPostStaticNoise );

    auto lPostStaticNoise0 = lPostStaticNoise.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();
    REQUIRE( lPostStaticNoise0 == lOutputData );
}

TEST_CASE( "Matched filter", "[CORE_SENSOR_MODEL_FPGA]" )
{
    size_t lPoolSize = 16 * 1024 * 1024;
    Scope lScope( lPoolSize );

    sFPGAConfiguration lConfiguration     = CreateFPGAConfiguration();
    sTileConfiguration lTileConfiguration = CreateTileConfiguration();

    auto lWaveforms  = GetBlinderOutputNode( lScope );
    auto lOutputData = GetFilteredWaveformData();

    auto lFilteredWaveforms = Filter( lScope, lConfiguration, lTileConfiguration, lWaveforms );
    lScope.Run( lFilteredWaveforms );

    auto lFilteredWaveforms0 = lFilteredWaveforms.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();

    REQUIRE( VectorEqual( lFilteredWaveforms0, lOutputData, 1.1f ) );
}

TEST_CASE( "Blinder - PulseIsSaturated", "[CORE_SENSOR_MODEL_FPGA]" )
{
    size_t lPoolSize = 16 * 1024 * 1024;
    Scope lScope( lPoolSize );

    sFPGAConfiguration lConfiguration     = CreateFPGAConfiguration();
    sTileConfiguration lTileConfiguration = CreateTileConfiguration();

    auto lWaveforms  = GetPostStaticNoiseWaveformNode( lScope );
    auto lOutputData = GetPulseIsSaturatedData();

    auto lBlinderOutput = Blinder( lScope, lConfiguration, lTileConfiguration, lWaveforms );
    lScope.Run( lBlinderOutput );

    auto lPulseIsSaturated0 = lBlinderOutput.Get<sBlinderNodeComponent>().mPulseIsSaturated.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();
    REQUIRE( lPulseIsSaturated0 == lOutputData );
}

TEST_CASE( "Blinder - BaselineBeforeSaturation", "[CORE_SENSOR_MODEL_FPGA]" )
{
    size_t lPoolSize = 16 * 1024 * 1024;
    Scope lScope( lPoolSize );

    sFPGAConfiguration lConfiguration     = CreateFPGAConfiguration();
    sTileConfiguration lTileConfiguration = CreateTileConfiguration();

    auto lWaveforms  = GetPostStaticNoiseWaveformNode( lScope );
    auto lOutputData = GetPreSaturationBaselineData();

    auto lBlinderOutput = Blinder( lScope, lConfiguration, lTileConfiguration, lWaveforms );
    lScope.Run( lBlinderOutput );

    auto lBaselineBeforeSaturation = lBlinderOutput.Get<sBlinderNodeComponent>().mPreSaturationBaseline.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();

    REQUIRE( lBaselineBeforeSaturation == lOutputData );
}

TEST_CASE( "Blinder - BaselineAfterSaturation", "[CORE_SENSOR_MODEL_FPGA]" )
{
    size_t lPoolSize = 16 * 1024 * 1024;
    Scope lScope( lPoolSize );

    sFPGAConfiguration lConfiguration     = CreateFPGAConfiguration();
    sTileConfiguration lTileConfiguration = CreateTileConfiguration();

    auto lWaveforms  = GetPostStaticNoiseWaveformNode( lScope );
    auto lOutputData = GetPostSaturationBaselineData();

    auto lBlinderOutput = Blinder( lScope, lConfiguration, lTileConfiguration, lWaveforms );
    lScope.Run( lBlinderOutput );

    auto lBaselineAfterSaturation = lBlinderOutput.Get<sBlinderNodeComponent>().mPostSaturationBaseline.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();

    REQUIRE( lBaselineAfterSaturation == lOutputData );
}

TEST_CASE( "Blinder - LastUnsaturatedSample", "[CORE_SENSOR_MODEL_FPGA]" )
{
    size_t lPoolSize = 16 * 1024 * 1024;
    Scope lScope( lPoolSize );

    sFPGAConfiguration lConfiguration     = CreateFPGAConfiguration();
    sTileConfiguration lTileConfiguration = CreateTileConfiguration();

    auto lWaveforms  = GetPostStaticNoiseWaveformNode( lScope );
    auto lOutputData = GetLastUnsaturatedSampleData();

    auto lBlinderOutput = Blinder( lScope, lConfiguration, lTileConfiguration, lWaveforms );
    lScope.Run( lBlinderOutput );

    auto lLastUnsaturatedSample = lBlinderOutput.Get<sBlinderNodeComponent>().mLastUnsaturatedSample.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();

    REQUIRE( lLastUnsaturatedSample == lOutputData );
}

TEST_CASE( "Blinder - SaturatedPulseLength", "[CORE_SENSOR_MODEL_FPGA]" )
{
    size_t lPoolSize = 16 * 1024 * 1024;
    Scope lScope( lPoolSize );

    sFPGAConfiguration lConfiguration     = CreateFPGAConfiguration();
    sTileConfiguration lTileConfiguration = CreateTileConfiguration();

    auto lWaveforms  = GetPostStaticNoiseWaveformNode( lScope );
    auto lOutputData = GetSaturatedPulseLengthData();

    auto lBlinderOutput = Blinder( lScope, lConfiguration, lTileConfiguration, lWaveforms );
    lScope.Run( lBlinderOutput );

    auto lSaturatedPulseLength = lBlinderOutput.Get<sBlinderNodeComponent>().mSaturatedPulseLength.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint32_t>();

    REQUIRE( lSaturatedPulseLength == lOutputData );
}

TEST_CASE( "Blinder - Output", "[CORE_SENSOR_MODEL_FPGA]" )
{
    size_t lPoolSize = 16 * 1024 * 1024;
    Scope lScope( lPoolSize );

    sFPGAConfiguration lConfiguration     = CreateFPGAConfiguration();
    sTileConfiguration lTileConfiguration = CreateTileConfiguration();

    auto lWaveforms  = GetPostStaticNoiseWaveformNode( lScope );
    auto lOutputData = GetBlinderOutputData();

    auto lBlinderOutput = Blinder( lScope, lConfiguration, lTileConfiguration, lWaveforms );
    lScope.Run( lBlinderOutput );

    auto lSaturatedPulseLength = lBlinderOutput.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();

    REQUIRE( lSaturatedPulseLength == lOutputData );
}

TEST_CASE( "Cfar thresholds", "[CORE_SENSOR_MODEL_FPGA]" )
{
    size_t lPoolSize = 16 * 1024 * 1024;
    Scope lScope( lPoolSize );

    sFPGAConfiguration lConfiguration     = CreateFPGAConfiguration();
    sTileConfiguration lTileConfiguration = CreateTileConfiguration();

    auto lWaveforms  = GetFilteredWaveformNode( lScope );
    auto lOutputData = GetThresholdData();

    auto lThresholds = Cfar( lScope, lConfiguration, lTileConfiguration, lWaveforms );
    lScope.Run( lThresholds );

    auto lThresholds0 = lThresholds.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();

    auto lCorrectThresholdCount = 0.0f;
    for( uint32_t i = 0; i < lThresholds0.size(); i++ )
        if( std::abs( lThresholds0[i] - lOutputData[i] ) <= 1.0f )
            lCorrectThresholdCount++;

    REQUIRE( ( lCorrectThresholdCount / lThresholds0.size() ) > 0.99f );
}

TEST_CASE( "Peak detector (valid detections)", "[CORE_SENSOR_MODEL_FPGA]" )
{
    size_t lPoolSize = 16 * 1024 * 1024;
    Scope lScope( lPoolSize );

    sFPGAConfiguration lConfiguration     = CreateFPGAConfiguration();
    sTileConfiguration lTileConfiguration = CreateTileConfiguration();

    auto lWaveforms  = GetFilteredWaveformNode( lScope );
    auto lThresholds = GetThresholdNode( lScope );
    auto lOutputData = GetValidDetectionData();

    auto lValidDetections = DetectPeaks( lScope, lConfiguration, lTileConfiguration, lWaveforms, lThresholds );
    lScope.Run( lValidDetections );

    auto lValidDetections0 = lValidDetections.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint8_t>();

    auto lCorrectValidDetectionCount = 0.0f;
    for( uint32_t i = 0; i < lValidDetections0.size(); i++ )
        if( std::abs( lValidDetections0[i] - lOutputData[i] ) <= 1.0f )
            lCorrectValidDetectionCount++;

    REQUIRE( ( lCorrectValidDetectionCount / lValidDetections0.size() ) > 0.99f );
}

TEST_CASE( "Peak detector (detections count)", "[CORE_SENSOR_MODEL_FPGA]" )
{
    size_t lPoolSize = 16 * 1024 * 1024;
    Scope lScope( lPoolSize );

    sFPGAConfiguration lConfiguration     = CreateFPGAConfiguration();
    sTileConfiguration lTileConfiguration = CreateTileConfiguration();

    auto lWaveforms  = GetFilteredWaveformNode( lScope );
    auto lThresholds = GetThresholdNode( lScope );
    auto lOutputData = GetValidDetectionCountData();

    auto lValidDetections = DetectPeaks( lScope, lConfiguration, lTileConfiguration, lWaveforms, lThresholds );
    auto lDetectionCount  = CountTrue( lScope, lValidDetections );
    lScope.Run( lDetectionCount );

    auto lDetectionCount0 = lDetectionCount.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint32_t>();

    auto lCorrectDetectionCount = 0.0f;
    for( uint32_t i = 0; i < lDetectionCount0.size(); i++ )
        if( std::abs( static_cast<int32_t>( lDetectionCount0[i] ) - static_cast<int32_t>( lOutputData[i] ) ) <= 1 )
            lCorrectDetectionCount++;

    REQUIRE( ( lCorrectDetectionCount / lDetectionCount0.size() ) > 0.99f );
}

TEST_CASE( "Peak detector (interpolated distance)", "[CORE_SENSOR_MODEL_FPGA]" )
{
    size_t lPoolSize = 16 * 1024 * 1024;
    Scope lScope( lPoolSize );

    sFPGAConfiguration lConfiguration     = CreateFPGAConfiguration();
    sTileConfiguration lTileConfiguration = CreateTileConfiguration();

    auto lWaveforms  = GetFilteredWaveformNode( lScope );
    auto lThresholds = GetThresholdNode( lScope );
    auto lOutputData = GetDistanceData();

    auto lValidDetections          = DetectPeaks( lScope, lConfiguration, lTileConfiguration, lWaveforms, lThresholds );
    auto [lDistances, lAmplitudes] = QuadraticInterpolation( lScope, lConfiguration, lTileConfiguration, lWaveforms );

    auto lZero = ConstantScalarValue( lScope, 0.0f );
    lDistances = Where( lScope, lValidDetections, lDistances, lZero );
    lScope.Run( lDistances );

    auto lDistances0 = lDistances.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();

    auto lCorrectDistance = 0.0f;
    for( uint32_t i = 0; i < lDistances0.size(); i++ )
        if( std::abs( lDistances0[i] - lOutputData[i] ) <= 1.0f )
            lCorrectDistance++;

    REQUIRE( ( lCorrectDistance / lDistances0.size() ) > 0.99f );
}

TEST_CASE( "Peak detector (interpolated amplitudes)", "[CORE_SENSOR_MODEL_FPGA]" )
{
    size_t lPoolSize = 16 * 1024 * 1024;
    Scope lScope( lPoolSize );

    sFPGAConfiguration lConfiguration     = CreateFPGAConfiguration();
    sTileConfiguration lTileConfiguration = CreateTileConfiguration();

    auto lWaveforms  = GetFilteredWaveformNode( lScope );
    auto lThresholds = GetThresholdNode( lScope );
    auto lOutputData = GetAmplitudeData();

    auto lValidDetections          = DetectPeaks( lScope, lConfiguration, lTileConfiguration, lWaveforms, lThresholds );
    auto [lDistances, lAmplitudes] = QuadraticInterpolation( lScope, lConfiguration, lTileConfiguration, lWaveforms );

    auto lZero  = ConstantScalarValue( lScope, 0.0f );
    lAmplitudes = Where( lScope, lValidDetections, lAmplitudes, lZero );
    lScope.Run( lAmplitudes );

    auto lAmplitudes0 = lAmplitudes.Get<sMultiTensorComponent>().mValue.FetchFlattened<float>();

    auto lCorrectAmplitude = 0.0f;
    for( uint32_t i = 0; i < lAmplitudes0.size(); i++ )
        if( std::abs( lAmplitudes0[i] - lOutputData[i] ) <= 1.0f )
            lCorrectAmplitude++;

    REQUIRE( ( lCorrectAmplitude / lAmplitudes0.size() ) > 0.99f );
}

TEST_CASE( "Packet header reference", "[CORE_SENSOR_MODEL_FPGA]" )
{
    size_t lPoolSize = 16 * 1024 * 1024;
    Scope lScope( lPoolSize );

    sFPGAConfiguration lConfiguration     = CreateFPGAConfiguration();
    sTileConfiguration lTileConfiguration = CreateTileConfiguration();

    auto lWaveforms      = GetWaveformNode( lScope );
    auto lDetectionCount = GetValidDetectionCountNode( lScope );
    auto lOutputData     = GetPacketHeaderData();

    auto [lPDNumber, lLaserAngle, lFrameNumber] = TileMetadata( lScope, lConfiguration, lTileConfiguration, lWaveforms );
    auto [lWaveformBaseline, lWaveformNoise]    = ComputeWaveformStatistics( lScope, lConfiguration, lTileConfiguration, lWaveforms, 2 );

    auto lHeaders =
        GeneratePacketHeaders( lScope, lConfiguration, lTileConfiguration, lDetectionCount, lPDNumber, lLaserAngle, lFrameNumber, lWaveformNoise, lWaveformBaseline, lWaveforms );
    lScope.Run( lHeaders );

    auto lHeaders0 = lHeaders.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint32_t>();

    auto lIsolate = []( std::vector<uint32_t> aVector, uint32_t aSkip, uint32_t aStart )
    {
        std::vector<uint32_t> lResult;
        for( uint32_t i = aStart; i < aVector.size(); i += aSkip )
            lResult.push_back( aVector[i] );

        return lResult;
    };

    std::vector<uint32_t> lComputedFirst = lIsolate( lHeaders0, 5, 0 );
    std::vector<uint32_t> lExpectedFirst = lIsolate( lOutputData, 5, 0 );
    REQUIRE( lComputedFirst == lExpectedFirst );

    std::vector<uint32_t> lComputedSecond = lIsolate( lHeaders0, 5, 1 );
    std::vector<uint32_t> lExpectedSecond = lIsolate( lOutputData, 5, 1 );
    REQUIRE( lComputedSecond == lExpectedSecond );

    std::vector<uint32_t> lComputedFourth = lIsolate( lHeaders0, 5, 3 );
    std::vector<uint32_t> lExpectedFourth = lIsolate( lOutputData, 5, 3 );
    REQUIRE( lComputedFourth == lExpectedFourth );

    std::vector<float> lNoise;
    std::vector<float> lENoise;
    for( uint32_t i = 4; i < lHeaders0.size(); i += 5 )
    {
        lNoise.push_back( static_cast<float>( static_cast<int16_t>( lHeaders0[i] & 0xFFFF ) ) );
        lENoise.push_back( static_cast<float>( static_cast<int16_t>( lOutputData[i] & 0xFFFF ) ) );
    }
    REQUIRE( VectorEqual( lNoise, lENoise, 1.1f ) );

    std::vector<float> lBaselines;
    std::vector<float> lEBaselines;
    for( uint32_t i = 4; i < lHeaders0.size(); i += 5 )
    {
        lBaselines.push_back( static_cast<float>( static_cast<int16_t>( ( lHeaders0[i] >> 16 ) & 0xFFFF ) ) );
        lEBaselines.push_back( static_cast<float>( static_cast<int16_t>( ( lOutputData[i] >> 16 ) & 0xFFFF ) ) );
    }
    REQUIRE( VectorEqual( lBaselines, lEBaselines, 1.1f ) );
}

TEST_CASE( "Detection Metadata", "[CORE_SENSOR_MODEL_FPGA]" )
{
    size_t lPoolSize = 16 * 1024 * 1024;
    Scope lScope( lPoolSize );

    sFPGAConfiguration lConfiguration     = CreateFPGAConfiguration();
    sTileConfiguration lTileConfiguration = CreateTileConfiguration();

    auto lZero    = ConstantScalarValue( lScope, 0.0f );
    auto lZeroU32 = ConstantScalarValue( lScope, 0u );
    auto lZeroU8  = ConstantScalarValue( lScope, static_cast<uint8_t>( 0 ) );

    auto lInputWaveforms         = GetFilteredWaveformNode( lScope );
    auto lThresholds             = GetThresholdNode( lScope );
    auto lDistances              = GetDistanceNode( lScope );
    auto lAmplitudes             = GetAmplitudeNode( lScope );
    auto lPreSaturationBaseline  = GetPreSaturationBaselineNode( lScope );
    auto lPostSaturationBaseline = GetPostSaturationBaselineNode( lScope );
    auto lSaturatedPulseLength   = GetSaturatedPulseLengthNode( lScope );
    auto lLastUnsaturatedSample  = GetLastUnsaturatedSampleNode( lScope );
    auto lPulseIsSaturated       = GetPulseIsSaturatedNode( lScope );
    auto lValidDetections        = GetValidDetectionNode( lScope );
    auto lValidDetectionCount    = GetValidDetectionCountNode( lScope );

    // Account for the shift induced by the matched filter
    lPreSaturationBaseline  = Shift( lScope, lPreSaturationBaseline, 3, lZero );
    lPostSaturationBaseline = Shift( lScope, lPostSaturationBaseline, 3, lZero );
    lSaturatedPulseLength   = Shift( lScope, lSaturatedPulseLength, 3, lZeroU32 );
    lLastUnsaturatedSample  = Shift( lScope, lLastUnsaturatedSample, 3, lZero );
    lPulseIsSaturated       = Shift( lScope, lPulseIsSaturated, 3, lZeroU8 );

    auto lOutputData = GetDetectionMetadata();

    auto lShortWaveforms =
        GenerateShortWaveforms( lScope, lConfiguration, lTileConfiguration, lInputWaveforms, lThresholds, lDistances, lAmplitudes, lPreSaturationBaseline, lPostSaturationBaseline,
                                lSaturatedPulseLength, lLastUnsaturatedSample, lPulseIsSaturated, lValidDetections, lValidDetectionCount );
    lScope.Run( lShortWaveforms );
    auto x = lPulseIsSaturated.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint32_t>();

    auto lShortWaveforms0 = lShortWaveforms.Get<sMultiTensorComponent>().mValue.FetchFlattened<uint32_t>();

    auto lIsolate = []( std::vector<uint32_t> aVector, uint32_t aStride, uint32_t aSkip, uint32_t aStart )
    {
        std::vector<uint32_t> lResult;
        for( uint32_t i = aStart; i < aVector.size(); i += aStride )
            lResult.push_back( aVector[i] );

        return lResult;
    };

    auto lHeader1 = lIsolate( lShortWaveforms0, 75, 0, 0 );
    std::vector<int16_t> lSampleBefore;
    std::vector<int16_t> lDistance;
    for( auto &v : lHeader1 )
    {
        lSampleBefore.push_back( static_cast<int16_t>( ( v >> 16 ) & 0xffff ) );
        lDistance.push_back( static_cast<int16_t>( v & 0x7fff ) );
    }

    auto lEHeader1 = lIsolate( lOutputData, 4, 0, 0 );
    std::vector<int16_t> lESampleBefore;
    std::vector<int16_t> lEDistance;
    for( auto &v : lEHeader1 )
    {
        lESampleBefore.push_back( static_cast<int16_t>( ( v >> 16 ) & 0xffff ) );
        lEDistance.push_back( static_cast<int16_t>( v & 0x7fff ) );
    }
    REQUIRE( VectorEqual( lSampleBefore, lESampleBefore, static_cast<int16_t>( 2 ) ) );
    REQUIRE( VectorEqual( lDistance, lEDistance, static_cast<int16_t>( 2 ) ) );

    auto lHeader2 = lIsolate( lShortWaveforms0, 75, 1, 1 );
    std::vector<int16_t> lBLBefore;
    std::vector<int16_t> lBLAfter;
    for( auto &v : lHeader2 )
    {
        lBLBefore.push_back( static_cast<int16_t>( v & 0xffff ) );
        lBLAfter.push_back( static_cast<int16_t>( ( v >> 16 ) & 0xffff ) );
    }
    auto lEHeader2 = lIsolate( lOutputData, 4, 1, 1 );
    std::vector<int16_t> lEBLBefore;
    std::vector<int16_t> lEBLAfter;
    for( auto &v : lEHeader2 )
    {
        lEBLBefore.push_back( static_cast<int16_t>( v & 0xffff ) );
        lEBLAfter.push_back( static_cast<int16_t>( ( v >> 16 ) & 0xffff ) );
    }
    REQUIRE( lBLBefore == lEBLBefore );
    REQUIRE( lBLAfter == lEBLAfter );

    auto lHeader3  = lIsolate( lShortWaveforms0, 75, 2, 2 );
    auto lEHeader3 = lIsolate( lOutputData, 4, 2, 2 );
    REQUIRE( VectorEqual( lHeader3, lEHeader3, static_cast<uint32_t>( 2 ) ) );

    auto lHeader4 = lIsolate( lShortWaveforms0, 75, 3, 3 );
    std::vector<int16_t> lIsSaturated;
    for( auto &v : lHeader4 )
    {
        lIsSaturated.push_back( static_cast<int16_t>( ( v & 0x8000 ) >> 15 ) );
    }
    std::vector<int16_t> lBin;
    for( auto &v : lHeader4 )
    {
        lBin.push_back( static_cast<int16_t>( v & 0x7fff ) );
    }
    std::vector<int16_t> lOffset;
    for( auto &v : lHeader4 )
    {
        lOffset.push_back( static_cast<int16_t>( v >> 16 ) );
    }

    auto lEHeader4 = lIsolate( lOutputData, 4, 3, 3 );
    std::vector<int16_t> lEIsSaturated;
    for( auto &v : lEHeader4 )
    {
        lEIsSaturated.push_back( static_cast<int16_t>( ( v & 0x8000 ) >> 15 ) );
    }
    std::vector<int16_t> lEBin;
    for( auto &v : lEHeader4 )
    {
        lEBin.push_back( static_cast<int16_t>( v & 0x7fff ) );
    }
    std::vector<int16_t> lEOffset;
    for( auto &v : lEHeader4 )
    {
        lEOffset.push_back( static_cast<int16_t>( v >> 16 ) );
    }
    REQUIRE( lIsSaturated == lEIsSaturated );
    REQUIRE( VectorEqual( lBin, lEBin, static_cast<int16_t>( 2 ) ) );
    REQUIRE( VectorEqual( lOffset, lEOffset, static_cast<int16_t>( 2 ) ) );

    auto lIsolateTraces = []( std::vector<uint32_t> aVector, uint32_t aStride, uint32_t aStart, uint32_t aLength )
    {
        std::vector<uint32_t> lResult;
        for( uint32_t i = aStart; i < aVector.size(); i += aStride )
            for( uint32_t j = 0; j < aLength; j++ )
                lResult.push_back( aVector[i + j] );

        return lResult;
    };

    auto lShortTraces    = lIsolateTraces( lShortWaveforms0, 75, 4, 11 );
    int16_t *lTraceData0 = (int16_t *)lShortTraces.data();
    std::vector<int16_t> lTraceData{};
    for( uint32_t i = 0; i < 11; i++ )
        lTraceData.push_back( lTraceData0[i] );
    std::vector<int16_t> lThresholdData{};
    for( uint32_t i = 0; i < 11; i++ )
        lThresholdData.push_back( lTraceData0[i + 11] );

    auto lEShortTraces    = GetPacketTraceData();
    int16_t *lETraceData0 = (int16_t *)lEShortTraces.data();
    std::vector<int16_t> lETraceData{};
    for( uint32_t i = 0; i < 11; i++ )
        lETraceData.push_back( lETraceData0[i] );
    std::vector<int16_t> lEThresholdData{};
    for( uint32_t i = 0; i < 11; i++ )
        lEThresholdData.push_back( lETraceData0[i + 11] );

    REQUIRE( VectorEqual( lTraceData, lETraceData, static_cast<int16_t>( 2 ) ) );
    REQUIRE( VectorEqual( lThresholdData, lEThresholdData, static_cast<int16_t>( 2 ) ) );
}
