/// @file   FPGAModel.cpp
///
/// @brief  FPGA processing.
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2022 LeddarTech Inc. All rights reserved.

#include "FPGAModel.h"
#include "FPGAModelKernels.h"

#include "Cuda/MultiTensor.h"

using namespace LTSE::Cuda;
using namespace LTSE::Core;

namespace LTSE::SensorModel
{
    std::tuple<OpNode, OpNode, OpNode> TileMetadata( Scope &aScope, sFPGAConfiguration const &aConfiguration, sTileConfiguration const &aTileConfiguration,
                                                     OpNode const &aWaveforms )
    {
        auto lLaserAngleCount       = aWaveforms.Get<sMultiTensorComponent>().mValue.Shape().GetDimension( 1 );
        auto lLaserAngleCountVector = VectorValue( aScope, lLaserAngleCount );

        std::vector<uint32_t> lPDNumberValues;
        for( auto &x : lLaserAngleCount )
            lPDNumberValues.insert( lPDNumberValues.end(), aTileConfiguration.mLCA3PhotodetectorIndices.begin(), aTileConfiguration.mLCA3PhotodetectorIndices.end() );

        std::vector<std::vector<uint32_t>> lPDNumberShape;
        for( auto &x : lLaserAngleCount )
            lPDNumberShape.push_back( std::vector<uint32_t>{ static_cast<uint32_t>( aTileConfiguration.mLCA3PhotodetectorIndices.size() ) } );
        sDataInitializerComponent lPDNumbersInitializer( lPDNumberValues );
        auto lPDNumbers = MultiTensorValue( aScope, lPDNumbersInitializer, sTensorShape( lPDNumberShape, sizeof( uint32_t ) ) );

        lPDNumbers = Tile( aScope, lPDNumbers, lLaserAngleCountVector );

        std::vector<std::vector<uint32_t>> lLaserAngleShape;
        for( auto &x : lLaserAngleCount )
            lLaserAngleShape.push_back( { x } );

        std::vector<uint32_t> lLaserAngleValues;
        for( auto &x : lLaserAngleCount )
        {
            for( uint32_t i = 0; i < x; i++ )
                lLaserAngleValues.push_back( i );
        }

        std::vector<uint32_t> lPDNumberCountValues;
        for( auto &x : lLaserAngleCount )
            lPDNumberCountValues.push_back( aTileConfiguration.mLCA3PhotodetectorIndices.size() );
        auto lPDNumberCountVector = VectorValue( aScope, lPDNumberCountValues );

        sDataInitializerComponent lLaserAngleInitializer( lLaserAngleValues );
        auto lLaserAngles = MultiTensorValue( aScope, lLaserAngleInitializer, sTensorShape( lLaserAngleShape, sizeof( uint32_t ) ) );
        lLaserAngles      = Repeat( aScope, lLaserAngles, lPDNumberCountVector );

        sConstantValueInitializerComponent lFrameNumberInitializer{};
        lFrameNumberInitializer.mValue = (uint32_t)0;
        auto lFrameNumber              = MultiTensorValue( aScope, lFrameNumberInitializer, lLaserAngles.Get<sMultiTensorComponent>().mValue.Shape() );

        return { lPDNumbers, lLaserAngles, lFrameNumber };
    }

    std::tuple<OpNode, OpNode> ComputeWaveformStatistics( Scope &aScope, sFPGAConfiguration const &aConfiguration, sTileConfiguration const &aTileConfiguration,
                                                          OpNode const &aWaveforms, uint32_t aFixed )
    {
        auto lZero = ConstantScalarValue( aScope, static_cast<uint32_t>( 0 ) );

        auto lStatsWindowSize0 = ConstantScalarValue( aScope, aConfiguration.mStatistics.mWindowSize - 1 );
        auto lInitialSegment   = Slice( aScope, aWaveforms, lZero, lStatsWindowSize0 );

        auto lWindowSize = ConstantScalarValue( aScope, static_cast<float>( aConfiguration.mStatistics.mWindowSize ) );
        auto lSum        = Summation( aScope, lInitialSegment );
        auto lBaseline   = Divide( aScope, lSum, lWindowSize );
        lBaseline        = Round( aScope, lBaseline );

        auto lSquaredSum       = Multiply( aScope, lSum, lSum );
        auto lSquaredWaveforms = Multiply( aScope, lInitialSegment, lInitialSegment );
        auto lSumOfSquares     = Summation( aScope, lSquaredWaveforms );

        auto lNoiseFactor      = ConstantScalarValue( aScope, 4.0f );
        auto lNoiseDenominator = ConstantScalarValue( aScope, 1.0f / ( 2.0f * static_cast<float>( aConfiguration.mStatistics.mWindowSize ) ) );

        auto lVariance0 = Multiply( aScope, lSumOfSquares, lWindowSize );
        auto lVariance1 = Subtract( aScope, lVariance0, lSquaredSum );
        auto lVariance2 = Abs( aScope, lVariance1 );
        auto lVariance3 = Multiply( aScope, lVariance2, lNoiseFactor );
        auto lVariance4 = Sqrt( aScope, lVariance3 );
        auto lVariance5 = Floor( aScope, lVariance4 );
        auto lNoiseStd  = Multiply( aScope, lVariance5, lNoiseDenominator );
        lNoiseStd       = Round( aScope, lNoiseStd );

        return { lBaseline, lNoiseStd };
    }

    template <typename _Ty> std::vector<_Ty> Concatenate( std::vector<std::vector<_Ty>> aVectors )
    {
        std::vector<_Ty> lResult;

        for( auto &lVector : aVectors )
            lResult.insert( lResult.end(), lVector.begin(), lVector.end() );

        return lResult;
    }

    OpNode RemoveStaticNoise( Scope &aScope, sFPGAConfiguration const &aConfiguration, sTileConfiguration const &aTileConfiguration, OpNode const &aWaveforms )
    {
        auto lTraceLengths = aWaveforms.Get<sMultiTensorComponent>().mValue.Shape().GetDimension( -1 );

        std::vector<std::vector<float>> lTemplateData;
        std::vector<std::vector<uint32_t>> lTemplateShape;
        for( auto lTraceLength : lTraceLengths )
        {
            auto lConfiguredTemplates = aConfiguration.GetStaticNoiseTemplates( aTileConfiguration, lTraceLength );
            auto lPDCount             = static_cast<uint32_t>( lConfiguredTemplates.size() );

            lTemplateShape.push_back( std::vector<uint32_t>{ lPDCount, lTraceLength } );
            lTemplateData.push_back( Concatenate( lConfiguredTemplates ) );
        }

        auto lTemplateInitializerData = Concatenate( lTemplateData );
        sDataInitializerComponent lLaserAngleInitializer( lTemplateInitializerData );
        auto lStaticNoiseTemplates = MultiTensorValue( aScope, lLaserAngleInitializer, sTensorShape( lTemplateShape, sizeof( float ) ) );

        auto lTilingRepetitions = VectorValue( aScope, aWaveforms.Get<sMultiTensorComponent>().mValue.Shape().GetDimension( 0 ) );
        lStaticNoiseTemplates   = Tile( aScope, lStaticNoiseTemplates, lTilingRepetitions );

        return Subtract( aScope, aWaveforms, lStaticNoiseTemplates );
    }

    void sBlinderOperationController::Run()
    {
        MultiTensor &lPostBlinderWaveforms = Get<sMultiTensorComponent>().mValue;

        auto &lOperandData                   = Get<sBlinderNodeComponent>();
        MultiTensor &lInputWaveforms         = lOperandData.mInputWaveforms.Get<sMultiTensorComponent>().mValue;
        MultiTensor &lPulseIsSaturated       = lOperandData.mPulseIsSaturated.Get<sMultiTensorComponent>().mValue;
        MultiTensor &lPreSaturationBaseline  = lOperandData.mPreSaturationBaseline.Get<sMultiTensorComponent>().mValue;
        MultiTensor &lPostSaturationBaseline = lOperandData.mPostSaturationBaseline.Get<sMultiTensorComponent>().mValue;
        MultiTensor &lSaturatedPulseLength   = lOperandData.mSaturatedPulseLength.Get<sMultiTensorComponent>().mValue;
        MultiTensor &lLastUnsaturatedSample  = lOperandData.mLastUnsaturatedSample.Get<sMultiTensorComponent>().mValue;

        MemoryBuffer &lBlockSizes   = lOperandData.mBlockSizes.Get<sVectorComponent<uint32_t>>().mData;
        MemoryBuffer &lElementCount = lOperandData.mElementCount.Get<sVectorComponent<uint32_t>>().mData;

        BlinderOp( lPostBlinderWaveforms, lPulseIsSaturated, lPreSaturationBaseline, lPostSaturationBaseline, lSaturatedPulseLength, lLastUnsaturatedSample,
                   lOperandData.mConfiguration, lInputWaveforms, lBlockSizes, lElementCount, lOperandData.mMaxBlockSize );
    }

    OpNode Blinder( Scope &aScope, sFPGAConfiguration const &aConfiguration, sTileConfiguration const &aTileConfiguration, OpNode const &aWaveforms )
    {
        sTensorShape lWaveformTensorShape( aWaveforms.Get<sMultiTensorComponent>().mValue.Shape().mShape, sizeof( float ) );

        auto lNewEntity = aScope.CreateNode();
        lNewEntity.Add<sTypeComponent>( aWaveforms.Get<sTypeComponent>() );
        auto &lOperandData           = lNewEntity.Add<sBlinderNodeComponent>();
        lOperandData.mConfiguration  = aConfiguration.mBlinder;
        lOperandData.mInputWaveforms = aWaveforms;

        sConstantValueInitializerComponent lZeroF( 0.0f );
        lOperandData.mPreSaturationBaseline  = MultiTensorValue( aScope, lZeroF, sTensorShape( lWaveformTensorShape.mShape, sizeof( float ) ) );
        lOperandData.mPostSaturationBaseline = MultiTensorValue( aScope, lZeroF, sTensorShape( lWaveformTensorShape.mShape, sizeof( float ) ) );
        lOperandData.mLastUnsaturatedSample  = MultiTensorValue( aScope, lZeroF, sTensorShape( lWaveformTensorShape.mShape, sizeof( float ) ) );

        sConstantValueInitializerComponent lZeroU8( static_cast<uint8_t>( 0 ) );
        lOperandData.mPulseIsSaturated = MultiTensorValue( aScope, lZeroU8, sTensorShape( lWaveformTensorShape.mShape, sizeof( uint8_t ) ) );

        sConstantValueInitializerComponent lZeroU32( static_cast<uint32_t>( 0 ) );
        lOperandData.mSaturatedPulseLength = MultiTensorValue( aScope, lZeroU32, sTensorShape( lWaveformTensorShape.mShape, sizeof( uint32_t ) ) );

        sTensorShape lBlockShape = aWaveforms.Get<sMultiTensorComponent>().mValue.Shape();
        lBlockShape.Flatten( -1 );
        lOperandData.mMaxBlockSize = lBlockShape.mMaxDimensions[0];
        lOperandData.mBlockSizes   = VectorValue( aScope, lBlockShape.GetDimension( 0 ) );
        lOperandData.mElementCount = VectorValue( aScope, lBlockShape.GetDimension( -1 ) );

        lNewEntity.Add<sOperandComponent>( std::vector{ lOperandData.mInputWaveforms, lOperandData.mPulseIsSaturated, lOperandData.mPreSaturationBaseline,
                                                        lOperandData.mPostSaturationBaseline, lOperandData.mSaturatedPulseLength, lOperandData.mLastUnsaturatedSample,
                                                        lOperandData.mBlockSizes, lOperandData.mElementCount } );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, lWaveformTensorShape );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sBlinderOperationController>();

        return lNewEntity;
    }

    void sCfarOperationController::Run()
    {
        MultiTensor &lPostBlinderWaveforms = Get<sMultiTensorComponent>().mValue;

        auto &lOperandData           = Get<sCfarNodeComponent>();
        MultiTensor &lInputWaveforms = lOperandData.mInputWaveforms.Get<sMultiTensorComponent>().mValue;

        MemoryBuffer &lBlockSizes   = lOperandData.mBlockSizes.Get<sVectorComponent<uint32_t>>().mData;
        MemoryBuffer &lElementCount = lOperandData.mElementCount.Get<sVectorComponent<uint32_t>>().mData;

        CfarOp( lPostBlinderWaveforms, lInputWaveforms, lOperandData.mConfiguration.mThresholdFactor, lOperandData.mConfiguration.mMinStd, lOperandData.mConfiguration.mNSkip,
                lOperandData.mConfiguration.mWindowLength, lOperandData.mConfiguration.mGuardLength, lBlockSizes, lElementCount, lOperandData.mMaxWaveformLength,
                lOperandData.mMaxBlockSize );
    }

    OpNode Cfar( Scope &aScope, sFPGAConfiguration const &aConfiguration, sTileConfiguration const &aTileConfiguration, OpNode const &aFilteredWaveforms )
    {
        auto lNewEntity = aScope.CreateNode();

        auto &lOperandData           = lNewEntity.Add<sCfarNodeComponent>();
        lOperandData.mConfiguration  = aConfiguration.mCfar;
        lOperandData.mInputWaveforms = aFilteredWaveforms;

        sTensorShape lBlockShape = aFilteredWaveforms.Get<sMultiTensorComponent>().mValue.Shape();
        lBlockShape.Flatten( -1 );
        lOperandData.mMaxBlockSize      = lBlockShape.mMaxDimensions[0];
        lOperandData.mMaxWaveformLength = lBlockShape.mMaxDimensions[1];
        lOperandData.mBlockSizes        = VectorValue( aScope, lBlockShape.GetDimension( 0 ) );
        lOperandData.mElementCount      = VectorValue( aScope, lBlockShape.GetDimension( -1 ) );

        lNewEntity.Add<sTypeComponent>( aFilteredWaveforms.Get<sTypeComponent>().mValue );
        lNewEntity.Add<sOperandComponent>( std::vector{ lOperandData.mInputWaveforms, lOperandData.mBlockSizes, lOperandData.mElementCount } );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, aFilteredWaveforms.Get<sMultiTensorComponent>().mValue.Shape() );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sCfarOperationController>();

        return lNewEntity;
    }

    void sPeakDetectorOperationController::Run()
    {
        MultiTensor &lOutputPeaks = Get<sMultiTensorComponent>().mValue;

        auto &lOperandData           = Get<sPeakDetectorNodeComponent>();
        MultiTensor &lInputWaveforms = lOperandData.mInputWaveforms.Get<sMultiTensorComponent>().mValue;
        MultiTensor &lCfarThresholds = lOperandData.mCfarThresholds.Get<sMultiTensorComponent>().mValue;

        MemoryBuffer &lBlockSizes   = lOperandData.mBlockSizes.Get<sVectorComponent<uint32_t>>().mData;
        MemoryBuffer &lElementCount = lOperandData.mElementCount.Get<sVectorComponent<uint32_t>>().mData;

        PeakDetectorOp( lOutputPeaks, lInputWaveforms, lOperandData.mConfiguration.mMarginStart, lOperandData.mConfiguration.mMarginEnd, lCfarThresholds, lElementCount,
                        lBlockSizes, lOperandData.mMaxWaveformLength, lOperandData.mMaxBlockSize );
    }

    OpNode DetectPeaks( Scope &aScope, sFPGAConfiguration const &aConfiguration, sTileConfiguration const &aTileConfiguration, OpNode const &aWaveforms, OpNode const &aThresholds )
    {
        auto lNewEntity = aScope.CreateNode();

        auto &lOperandData           = lNewEntity.Add<sPeakDetectorNodeComponent>();
        lOperandData.mConfiguration  = aConfiguration.mPeakDetector;
        lOperandData.mInputWaveforms = aWaveforms;
        lOperandData.mCfarThresholds = aThresholds;

        sTensorShape lBlockShape = aWaveforms.Get<sMultiTensorComponent>().mValue.Shape();
        lBlockShape.Flatten( -1 );
        lOperandData.mMaxBlockSize      = lBlockShape.mMaxDimensions[0];
        lOperandData.mMaxWaveformLength = lBlockShape.mMaxDimensions[1];
        lOperandData.mBlockSizes        = VectorValue( aScope, lBlockShape.GetDimension( 0 ) );
        lOperandData.mElementCount      = VectorValue( aScope, lBlockShape.GetDimension( -1 ) );

        lNewEntity.Add<sTypeComponent>( eScalarType::UINT8 );
        lNewEntity.Add<sOperandComponent>( std::vector{ lOperandData.mInputWaveforms, lOperandData.mCfarThresholds, lOperandData.mBlockSizes, lOperandData.mElementCount } );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, sTensorShape( aWaveforms.Get<sMultiTensorComponent>().mValue.Shape().mShape, sizeof( uint8_t ) ) );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sPeakDetectorOperationController>();

        return lNewEntity;
    }

    void sQuadraticInterpolationOperationController::Run()
    {
        auto &lOperandData           = Get<sQuadraticInterpolationNodeComponent>();
        MultiTensor &lInputWaveforms = lOperandData.mInputWaveforms.Get<sMultiTensorComponent>().mValue;

        MultiTensor &lDistances  = lOperandData.mDistances.Get<sMultiTensorComponent>().mValue;
        MultiTensor &lAmplitudes = lOperandData.mAmplitudes.Get<sMultiTensorComponent>().mValue;

        MemoryBuffer &lBlockSizes   = lOperandData.mBlockSizes.Get<sVectorComponent<uint32_t>>().mData;
        MemoryBuffer &lElementCount = lOperandData.mElementCount.Get<sVectorComponent<uint32_t>>().mData;

        QuadraticInterpolationOp( lDistances, lAmplitudes, lInputWaveforms, lElementCount, lBlockSizes, lOperandData.mMaxWaveformLength, lOperandData.mMaxBlockSize );
    }

    std::tuple<OpNode, OpNode> QuadraticInterpolation( Scope &aScope, sFPGAConfiguration const &aConfiguration, sTileConfiguration const &aTileConfiguration,
                                                       OpNode const &aWaveforms )
    {
        auto lNewEntity = aScope.CreateNode();

        auto &lOperandData = lNewEntity.Add<sQuadraticInterpolationNodeComponent>();

        lOperandData.mInputWaveforms = aWaveforms;

        sTensorShape lWaveformTensorShape = aWaveforms.Get<sMultiTensorComponent>().mValue.Shape();

        sTensorShape lBlockShape = aWaveforms.Get<sMultiTensorComponent>().mValue.Shape();
        lBlockShape.Flatten( -1 );
        lOperandData.mMaxBlockSize      = lBlockShape.mMaxDimensions[0];
        lOperandData.mMaxWaveformLength = lBlockShape.mMaxDimensions[1];
        lOperandData.mBlockSizes        = VectorValue( aScope, lBlockShape.GetDimension( 0 ) );
        lOperandData.mElementCount      = VectorValue( aScope, lBlockShape.GetDimension( -1 ) );

        lNewEntity.Add<sOperandComponent>( std::vector{ lOperandData.mInputWaveforms, lOperandData.mBlockSizes, lOperandData.mElementCount } );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sQuadraticInterpolationOperationController>();

        auto lDistances = aScope.CreateNode();
        lDistances.Add<sTypeComponent>( aWaveforms.Get<sTypeComponent>().mValue );
        lDistances.Add<sOperandComponent>( std::vector{ lNewEntity } );
        lDistances.Add<sMultiTensorComponent>( aScope.mPool, lWaveformTensorShape );
        lOperandData.mDistances = lDistances;

        auto lAmplitudes = aScope.CreateNode();
        lAmplitudes.Add<sTypeComponent>( aWaveforms.Get<sTypeComponent>().mValue );
        lAmplitudes.Add<sOperandComponent>( std::vector{ lNewEntity } );
        lAmplitudes.Add<sMultiTensorComponent>( aScope.mPool, lWaveformTensorShape );
        lOperandData.mAmplitudes = lAmplitudes;

        return { lDistances, lAmplitudes };
    }

    void sPacketHeaderOperationController::Run()
    {
        auto &lValue = Get<sMultiTensorComponent>().mValue;
        auto &lRands = Get<sPacketHeaderNodeComponent>();

        GeneratePacketHeaderOp( lValue, lRands.mVersion.Get<sVectorComponent<uint32_t>>().mData, lRands.mID.Get<sVectorComponent<uint32_t>>().mData,
                                lRands.mSampleCount.Get<sVectorComponent<uint32_t>>().mData, lRands.mMaxDetectionCount.Get<sVectorComponent<uint32_t>>().mData,
                                lRands.mDetectionCount.Get<sMultiTensorComponent>().mValue, lRands.mPDNumber.Get<sMultiTensorComponent>().mValue,
                                lRands.mLaserAngle.Get<sMultiTensorComponent>().mValue, lRands.mWaveformNoise.Get<sMultiTensorComponent>().mValue,
                                lRands.mWaveformBaseline.Get<sMultiTensorComponent>().mValue, lRands.mBlockSizes.Get<sVectorComponent<uint32_t>>().mData, lRands.mMaxBlockSize,
                                lRands.mHeaderSizes.Get<sVectorComponent<uint32_t>>().mData );
    }

    OpNode GeneratePacketHeaders( Scope &aScope, sFPGAConfiguration const &aConfiguration, sTileConfiguration const &aTileConfiguration, OpNode const &aDetectionCount,
                                  OpNode const &aPDNumber, OpNode const &aLaserAngle, OpNode const &aFrameNumber, OpNode const &aWaveformNoise, OpNode const &aWaveformBaseline,
                                  OpNode const &aWaveforms )
    {
        uint32_t lHeaderSize = aConfiguration.GetHeaderSize( aTileConfiguration );

        sTensorShape lOutputShape = aPDNumber.Get<sMultiTensorComponent>().mValue.Shape();
        std::vector<uint32_t> lHeaderSizeDimension( lOutputShape.CountLayers(), lHeaderSize );
        lOutputShape.InsertDimension( -1, lHeaderSizeDimension );

        auto lPeakDetectorMode = aTileConfiguration.mPeakDetectorMode;

        constexpr uint32_t ID_COMPRESS  = 4;
        constexpr uint32_t ID_CLOUD     = 5;
        constexpr uint32_t ID_CLOUD_EXT = 6;
        OpNode lIDNode{};
        OpNode lSampleCountNode{};
        switch( aTileConfiguration.mPeakDetectorMode )
        {
        case sTileConfiguration::PeakDetectorMode::COMPRESSED_RAW_DATA:
            lIDNode          = VectorValue( aScope, std::vector<uint32_t>( lOutputShape.CountLayers(), ID_COMPRESS ) );
            lSampleCountNode = VectorValue( aScope, std::vector<uint32_t>( lOutputShape.CountLayers(), aTileConfiguration.GetSampleCount() ) );
            break;
        case sTileConfiguration::PeakDetectorMode::SURFACE_CLOUD:
            lIDNode          = VectorValue( aScope, std::vector<uint32_t>( lOutputShape.CountLayers(), ID_CLOUD ) );
            lSampleCountNode = VectorValue( aScope, aWaveforms.Get<sMultiTensorComponent>().mValue.Shape().GetDimension( -1 ) );
            break;
        default:
            lIDNode          = VectorValue( aScope, std::vector<uint32_t>( lOutputShape.CountLayers(), ID_CLOUD ) );
            lSampleCountNode = VectorValue( aScope, std::vector<uint32_t>( lOutputShape.CountLayers(), 0 ) );
            break;
        }

        auto lVersionNode       = VectorValue( aScope, std::vector<uint32_t>( lOutputShape.CountLayers(), 1 ) );
        auto lMaxDetectionCount = VectorValue( aScope, std::vector<uint32_t>( lOutputShape.CountLayers(), aTileConfiguration.mDetectionCount ) );

        auto lNewEntity = aScope.CreateNode();
        auto &lRands    = lNewEntity.Add<sPacketHeaderNodeComponent>( sPacketHeaderNodeComponent{ lIDNode, lVersionNode, lSampleCountNode, lMaxDetectionCount, aDetectionCount,
                                                                                               aPDNumber, aLaserAngle, aFrameNumber, aWaveformNoise, aWaveformBaseline } );

        auto lKernelShape = aPDNumber.Get<sMultiTensorComponent>().mValue.Shape();
        lKernelShape.Flatten( 0 );
        lRands.mMaxBlockSize = lKernelShape.mMaxDimensions[0];
        lRands.mBlockSizes   = VectorValue( aScope, lKernelShape.GetDimension( 0 ) );
        lRands.mHeaderSizes  = VectorValue( aScope, lOutputShape.GetDimension( -1 ) );

        lNewEntity.Add<sOperandComponent>( std::vector{ lIDNode, lVersionNode, lSampleCountNode, lMaxDetectionCount, aDetectionCount, aPDNumber, aLaserAngle, aFrameNumber,
                                                        aWaveformNoise, aWaveformBaseline, lRands.mBlockSizes, lRands.mHeaderSizes } );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, lOutputShape );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sPacketHeaderOperationController>();

        return lNewEntity;
    }

    OpNode Filter( Scope &aScope, sFPGAConfiguration const &aConfiguration, sTileConfiguration const &aTileConfiguration, OpNode const &aWaveforms )
    {
        std::vector<uint32_t> lScanCount    = aWaveforms.Get<sMultiTensorComponent>().mValue.Shape().GetDimension( 0 );
        std::vector<uint32_t> lSegmentCount = aWaveforms.Get<sMultiTensorComponent>().mValue.Shape().GetDimension( 1 );

        std::vector<uint32_t> lPixelCountValues{};
        for( uint32_t i = 0; i < lSegmentCount.size(); i++ )
            lPixelCountValues.push_back( lSegmentCount[i] * lScanCount[i] );
        auto lPixelCount = VectorValue( aScope, lPixelCountValues );

        auto lFilterCoefficientValues = sDataInitializerComponent( aConfiguration.mFilter.mCoefficients );
        uint32_t lFilterLength        = static_cast<uint32_t>( aConfiguration.mFilter.mCoefficients.size() );
        auto lFilterCoefficients =
            MultiTensorValue( aScope, lFilterCoefficientValues, sTensorShape( std::vector<std::vector<uint32_t>>{ std::vector<uint32_t>{ lFilterLength } }, sizeof( float ) ) );

        sTensorShape lFilterKernelShape = aWaveforms.Get<sMultiTensorComponent>().mValue.Shape();
        lFilterKernelShape.Trim( -1 );
        lFilterKernelShape.InsertDimension( -1, std::vector<uint32_t>{ lFilterLength } );
        lFilterCoefficients = Tile( aScope, lFilterCoefficients, lPixelCount );
        lFilterCoefficients = Relayout( aScope, lFilterCoefficients, lFilterKernelShape );

        auto lPrechargeShape  = aWaveforms.Get<sMultiTensorComponent>().mValue.Shape();
        auto lFilterLengthDim = std::vector<uint32_t>( lPrechargeShape.CountLayers(), aConfiguration.mFilter.mCoefficients.size() );
        lPrechargeShape.Trim( -1 );
        lPrechargeShape.InsertDimension( -1, lFilterLengthDim );
        auto lOne       = sConstantValueInitializerComponent( 1.0f );
        auto lPrecharge = MultiTensorValue( aScope, lOne, lPrechargeShape );

        auto lZero         = ConstantScalarValue( aScope, static_cast<uint32_t>( 0 ) );
        auto lSample0      = Slice( aScope, aWaveforms, lZero, lZero );
        auto lSample0Shape = lSample0.Get<sMultiTensorComponent>().mValue.Shape();
        lSample0Shape.Trim( -1 );
        lSample0   = Reshape( aScope, lSample0, lSample0Shape );
        lPrecharge = Multiply( aScope, lPrecharge, lSample0 );

        auto lWaveformsWithPrecharge = HCat( aScope, lPrecharge, aWaveforms );
        auto lFilteredWaveforms      = Conv1D( aScope, lWaveformsWithPrecharge, lFilterCoefficients );
        lFilteredWaveforms           = Round( aScope, lFilteredWaveforms );

        auto lFilteredWaveformBegin = ConstantScalarValue( aScope, lFilterLength );

        auto lLastIndex = lFilteredWaveforms.Get<sMultiTensorComponent>().mValue.Shape().GetDimension( -1 );
        for( auto &x : lLastIndex )
            x -= 1;

        auto lFilteredWaveformEnd = VectorValue( aScope, lLastIndex );
        lFilteredWaveforms        = Slice( aScope, lFilteredWaveforms, lFilteredWaveformBegin, lFilteredWaveformEnd );

        auto x = lFilteredWaveforms.Get<sMultiTensorComponent>().mValue.Shape();

        return lFilteredWaveforms;
    }

    void sFilterDetectionsOperationController::Run()
    {
        MultiTensor &lFilteredDetections = Get<sMultiTensorComponent>().mValue;

        auto &lOperandData       = Get<sFilterDetectionsNodeComponent>();
        MultiTensor &lDetections = lOperandData.mDetections.Get<sMultiTensorComponent>().mValue;

        MemoryBuffer &lBlockSizes   = lOperandData.mBlockSizes.Get<sVectorComponent<uint32_t>>().mData;
        MemoryBuffer &lElementCount = lOperandData.mElementCount.Get<sVectorComponent<uint32_t>>().mData;

        FilterDetectionOp( lFilteredDetections, lDetections, lOperandData.mConfiguration.mMaskLength, lBlockSizes, lElementCount, lOperandData.mMaxBlockSize );
    }

    OpNode FilterPeaks( Scope &aScope, sFPGAConfiguration const &aConfiguration, sTileConfiguration const &aTileConfiguration, OpNode const &aDetections )
    {
        sTensorShape lWaveformTensorShape( aDetections.Get<sMultiTensorComponent>().mValue.Shape().mShape, sizeof( float ) );
        auto lZero = ConstantScalarValue( aScope, 0.0f );

        auto lNewEntity = aScope.CreateNode();
        lNewEntity.Add<sTypeComponent>( aDetections.Get<sTypeComponent>() );
        auto &lOperandData       = lNewEntity.Add<sFilterDetectionsNodeComponent>();
        lOperandData.mDetections = aDetections;

        sTensorShape lBlockShape = aDetections.Get<sMultiTensorComponent>().mValue.Shape();
        lBlockShape.Flatten( -1 );
        lOperandData.mBlockSizes   = VectorValue( aScope, lBlockShape.GetDimension( 0 ) );
        lOperandData.mElementCount = VectorValue( aScope, lBlockShape.GetDimension( -1 ) );

        lOperandData.mMaxBlockSize = lBlockShape.mMaxDimensions[0];

        lNewEntity.Add<sOperandComponent>( std::vector{ lOperandData.mDetections } );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, sTensorShape( lWaveformTensorShape.mShape, sizeof( uint8_t ) ) );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sStructureDetectionsOperationController>();

        return lNewEntity;
    }

    void sStructureDetectionsOperationController::Run()
    {
        MultiTensor &lDetectionPackets = Get<sMultiTensorComponent>().mValue;

        auto &lOperandData                   = Get<sStructureDetectionsNodeComponent>();
        MultiTensor &lInputWaveforms         = lOperandData.mInputWaveforms.Get<sMultiTensorComponent>().mValue;
        MultiTensor &lThresholds             = lOperandData.mThresholds.Get<sMultiTensorComponent>().mValue;
        MultiTensor &lDistance               = lOperandData.mDistance.Get<sMultiTensorComponent>().mValue;
        MultiTensor &lAmplitude              = lOperandData.mAmplitude.Get<sMultiTensorComponent>().mValue;
        MultiTensor &lPulseIsSaturated       = lOperandData.mPulseIsSaturated.Get<sMultiTensorComponent>().mValue;
        MultiTensor &lPreSaturationBaseline  = lOperandData.mPreSaturationBaseline.Get<sMultiTensorComponent>().mValue;
        MultiTensor &lPostSaturationBaseline = lOperandData.mPostSaturationBaseline.Get<sMultiTensorComponent>().mValue;
        MultiTensor &lSaturatedPulseLength   = lOperandData.mSaturatedPulseLength.Get<sMultiTensorComponent>().mValue;
        MultiTensor &lLastUnsaturatedSample  = lOperandData.mLastUnsaturatedSample.Get<sMultiTensorComponent>().mValue;
        MultiTensor &lValidDetections        = lOperandData.mValidDetections.Get<sMultiTensorComponent>().mValue;
        MultiTensor &lValidDetectionCount    = lOperandData.mValidDetectionCount.Get<sMultiTensorComponent>().mValue;

        MemoryBuffer &lBlockSizes   = lOperandData.mBlockSizes.Get<sVectorComponent<uint32_t>>().mData;
        MemoryBuffer &lElementCount = lOperandData.mElementCount.Get<sVectorComponent<uint32_t>>().mData;
        MemoryBuffer &lPacketSizes  = lOperandData.mPacketSizes.Get<sVectorComponent<uint32_t>>().mData;

        GenerateShortWaveformPacketsOp( lDetectionPackets, lInputWaveforms, lThresholds, lDistance, lAmplitude, lPulseIsSaturated, lPreSaturationBaseline, lPostSaturationBaseline,
                                        lSaturatedPulseLength, lLastUnsaturatedSample, lValidDetections, lValidDetectionCount, lOperandData.mMaxDetectionCount,
                                        lOperandData.mNeighbourCount, lBlockSizes, lElementCount, lPacketSizes, lOperandData.mMaxBlockSize, lOperandData.mMaxWaveformLength );
    }

    OpNode GenerateShortWaveforms( Scope &aScope, sFPGAConfiguration const &aConfiguration, sTileConfiguration const &aTileConfiguration, OpNode const &aWaveforms,
                                   OpNode const &aThresholds, OpNode const &aDistance, OpNode const &aAmplitude, OpNode const &aBaselineBeforeSaturation,
                                   OpNode const &aBaselineAfterSaturation, OpNode const &aPulseLength, OpNode const &aLastUnsaturatedSample, OpNode const &aIsSaturated,
                                   OpNode const &aValidDetections, OpNode const &aValidDetectionCount )
    {
        auto lNewEntity = aScope.CreateNode();
        lNewEntity.Add<sTypeComponent>( eScalarType::UINT32 );
        auto &lOperandData                   = lNewEntity.Add<sStructureDetectionsNodeComponent>();
        lOperandData.mInputWaveforms         = aWaveforms;
        lOperandData.mThresholds             = aThresholds;
        lOperandData.mDistance               = aDistance;
        lOperandData.mAmplitude              = aAmplitude;
        lOperandData.mPulseIsSaturated       = aIsSaturated;
        lOperandData.mPreSaturationBaseline  = aBaselineBeforeSaturation;
        lOperandData.mPostSaturationBaseline = aBaselineAfterSaturation;
        lOperandData.mSaturatedPulseLength   = aPulseLength;
        lOperandData.mLastUnsaturatedSample  = aLastUnsaturatedSample;
        lOperandData.mValidDetections        = aValidDetections;
        lOperandData.mValidDetectionCount    = aValidDetectionCount;
        lOperandData.mMaxDetectionCount      = aTileConfiguration.mDetectionCount;
        lOperandData.mNeighbourCount         = aTileConfiguration.mNeighbourCount;

        sTensorShape lBlockShape = aWaveforms.Get<sMultiTensorComponent>().mValue.Shape();
        lBlockShape.Flatten( -1 );
        lOperandData.mBlockSizes   = VectorValue( aScope, lBlockShape.GetDimension( 0 ) );
        lOperandData.mElementCount = VectorValue( aScope, lBlockShape.GetDimension( -1 ) );

        lOperandData.mMaxBlockSize = lBlockShape.mMaxDimensions[0];

        sTensorShape lWaveformPacketShape = aWaveforms.Get<sMultiTensorComponent>().mValue.Shape();
        lWaveformPacketShape.Trim( -1 );
        std::vector<uint32_t> lPacketSize( lWaveformPacketShape.CountLayers(), aConfiguration.GetPacketLength( aTileConfiguration ) * aTileConfiguration.mDetectionCount );
        lWaveformPacketShape.InsertDimension( -1, lPacketSize );
        lOperandData.mPacketSizes = VectorValue( aScope, lWaveformPacketShape.GetDimension( -1 ) );

        lNewEntity.Add<sOperandComponent>( std::vector{ lOperandData.mInputWaveforms, lOperandData.mThresholds, lOperandData.mDistance, lOperandData.mAmplitude,
                                                        lOperandData.mPulseIsSaturated, lOperandData.mPreSaturationBaseline, lOperandData.mPostSaturationBaseline,
                                                        lOperandData.mSaturatedPulseLength, lOperandData.mLastUnsaturatedSample, lOperandData.mValidDetections,
                                                        lOperandData.mValidDetectionCount, lOperandData.mBlockSizes, lOperandData.mElementCount, lOperandData.mPacketSizes } );
        lNewEntity.Add<sMultiTensorComponent>( aScope.mPool, sTensorShape( lWaveformPacketShape.mShape, sizeof( uint32_t ) ) );
        lNewEntity.Add<sGraphOperationComponent>().Bind<sStructureDetectionsOperationController>();

        return lNewEntity;
    }

    OpNode FPGAProcess( Scope &aScope, sFPGAConfiguration const &aConfiguration, sTileConfiguration const &aTileConfiguration, OpNode const &aWaveforms )
    {
        auto [lPDNumber, lLaserAngle, lFrameNumber] = TileMetadata( aScope, aConfiguration, aTileConfiguration, aWaveforms );

        auto [lWaveformBaseline, lWaveformNoise] = ComputeWaveformStatistics( aScope, aConfiguration, aTileConfiguration, aWaveforms, 2 );

        auto lPostStaticNoiseWaveforms = RemoveStaticNoise( aScope, aConfiguration, aTileConfiguration, aWaveforms );

        auto lBlinderData      = Blinder( aScope, aConfiguration, aTileConfiguration, lPostStaticNoiseWaveforms );
        auto lBlinderArtefacts = lBlinderData.Get<sBlinderNodeComponent>();

        auto lFilteredWaveform = Filter( aScope, aConfiguration, aTileConfiguration, lPostStaticNoiseWaveforms );

        auto lThresholds = Cfar( aScope, aConfiguration, aTileConfiguration, lFilteredWaveform );

        auto lValidPeaks = DetectPeaks( aScope, aConfiguration, aTileConfiguration, aWaveforms, lThresholds );
        auto lPeakCount  = CountTrue( aScope, lValidPeaks );

        auto [lDistances, lAmplitudes] = QuadraticInterpolation( aScope, aConfiguration, aTileConfiguration, aWaveforms );

        auto lZero  = ConstantScalarValue( aScope, 0.0f );
        lValidPeaks = FilterPeaks( aScope, aConfiguration, aTileConfiguration, lValidPeaks );
        lDistances  = Where( aScope, lValidPeaks, lDistances, lZero );
        lAmplitudes = Where( aScope, lValidPeaks, lAmplitudes, lZero );

        // clang-format off

        auto lHeaders = GeneratePacketHeaders(
            aScope, aConfiguration, aTileConfiguration, lPeakCount, lPDNumber, lLaserAngle, lFrameNumber,
            lWaveformNoise, lWaveformBaseline, aWaveforms );

        auto lShortWaveforms = GenerateShortWaveforms(
            aScope, aConfiguration, aTileConfiguration, aWaveforms, lThresholds, lDistances, lAmplitudes,
            lBlinderArtefacts.mPreSaturationBaseline, lBlinderArtefacts.mPostSaturationBaseline, lBlinderArtefacts.mSaturatedPulseLength,
            lBlinderArtefacts.mLastUnsaturatedSample, lBlinderArtefacts.mPulseIsSaturated, lValidPeaks, lPeakCount );

        // clang-format on

        return HCat( aScope, lHeaders, lShortWaveforms );
    }
} // namespace LTSE::SensorModel
