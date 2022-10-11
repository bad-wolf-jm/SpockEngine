/// @file   EnvironmentSampler.cpp
///
/// @brief  Implmeentation file for environment sampler
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#include "EnvironmentSampler.h"
#include "DeveloperTools/Profiling/BlockTimer.h"

/** @brief */
namespace LTSE::SensorModel
{

    OpNode EnvironmentSampler::operator[]( std::string aNodeName ) { return ( *mScope )[aNodeName]; }

    OpNode EnvironmentSampler::CreateRangeNode( Scope &aScope, std::vector<float> const &aStart, std::vector<float> const &aEnd, float aDelta )
    {
        uint32_t lOutputRangeLayers = aStart.size();

        auto lStartNode = ScalarVectorValue( aScope, eScalarType::FLOAT32, aStart );
        auto lEndNode   = ScalarVectorValue( aScope, eScalarType::FLOAT32, aEnd );

        std::vector<ScalarValue> lDeltaInit( lOutputRangeLayers );
        for( uint32_t i = 0; i < lOutputRangeLayers; i++ )
        {
            lDeltaInit[i] = aDelta;
        }

        auto lDeltaNode = VectorValue( aScope, lDeltaInit );
        return ARange( aScope, lStartNode, lEndNode, lDeltaNode );
    }

    std::tuple<OpNode, OpNode> EnvironmentSampler::Prod( Scope &aScope, OpNode const &aLeft, OpNode const &aRight )
    {
        uint32_t lLayers = aLeft.Get<sMultiTensorComponent>().mValue.Shape().CountLayers();

        sTensorShape &lXShape = aLeft.Get<sMultiTensorComponent>().mValue.Shape();
        sTensorShape &lYShape = aRight.Get<sMultiTensorComponent>().mValue.Shape();

        std::vector<uint32_t> lXRepetitionsInit( lLayers );
        for( uint32_t i = 0; i < lLayers; i++ )
        {
            lXRepetitionsInit[i] = lYShape.mShape[i][lXShape.mRank - 1];
        }

        std::vector<uint32_t> lYRepetitionsInit( lLayers );
        for( uint32_t i = 0; i < lLayers; i++ )
        {
            lYRepetitionsInit[i] = lXShape.mShape[i][lXShape.mRank - 1];
        }

        auto lXRepetitionsNode = VectorValue( aScope, lXRepetitionsInit );
        auto lYRepetitionsNode = VectorValue( aScope, lYRepetitionsInit );

        auto lProdX = Repeat( aScope, aLeft, lXRepetitionsNode );
        auto lProdY = Tile( aScope, aRight, lYRepetitionsNode );

        return std::tuple<OpNode, OpNode>{ lProdX, lProdY };
    }

    EnvironmentSampler::EnvironmentSampler( sCreateInfo const &aSpec, Ref<Scope> aScope, AcquisitionContext const &aFlashList )
        : mSpec{ aSpec }
        , mScope{ aScope }
        , mFlashList{ aFlashList }
    {
        mScheduledFlashCount = aFlashList.mScheduledFlashEntities.size();

        mWorldAzimuth.mMin = aFlashList.mEnvironmentSampling.mWorldAzimuth.mMin;
        mWorldAzimuth.mMax = aFlashList.mEnvironmentSampling.mWorldAzimuth.mMax;

        mWorldElevation.mMin = aFlashList.mEnvironmentSampling.mWorldElevation.mMin;
        mWorldElevation.mMax = aFlashList.mEnvironmentSampling.mWorldElevation.mMax;

        mTimestamp = aFlashList.mEnvironmentSampling.mTimestamp;
    }

    EnvironmentSampler::EnvironmentSampler( sCreateInfo const &aSpec, uint32_t aPoolSize, AcquisitionContext const &aFlashList )
        : EnvironmentSampler( aSpec, New<Scope>( aPoolSize ), aFlashList )
    {
    }

    void EnvironmentSampler::CreateGraph()
    {
        LTSE_PROFILE_FUNCTION();

        auto lResolutionX   = ConstantScalarValue( *mScope, mSpec.mSamplingResolution.x );
        auto lAzimuthRange0 = CreateRangeNode( *mScope, mWorldAzimuth.mMin, mWorldAzimuth.mMax, mSpec.mSamplingResolution.x );
        auto lAzimuthRange1 = Add( *mScope, lAzimuthRange0, lResolutionX );

        auto lResolutionY     = ConstantScalarValue( *mScope, mSpec.mSamplingResolution.y );
        auto lElevationRange0 = CreateRangeNode( *mScope, mWorldElevation.mMin, mWorldElevation.mMax, mSpec.mSamplingResolution.y );
        auto lElevationRange1 = Add( *mScope, lElevationRange0, lResolutionY );

        auto [lElevationXY0, lAzimuthXY0] = Prod( *mScope, lElevationRange0, lAzimuthRange0 );
        auto [lElevationXY1, lAzimuthXY1] = Prod( *mScope, lElevationRange1, lAzimuthRange1 );

        if( mSpec.mUseRegularMultiSampling )
        {
            throw std::runtime_error( "This is not yet implemented!!" );
        }
        else if( mSpec.mMultiSamplingFactor > 1 )
        {
            std::vector<uint32_t> lMultiSamplingRepetitionsInit( mScheduledFlashCount );
            for( uint32_t i = 0; i < mScheduledFlashCount; i++ )
            {
                lMultiSamplingRepetitionsInit[i] = mSpec.mMultiSamplingFactor;
            }

            auto lMultiSamplingRepetitionsNode = VectorValue( *mScope, lMultiSamplingRepetitionsInit );
            auto lAzimuthMultiSampledXY0       = Repeat( *mScope, lAzimuthXY0, lMultiSamplingRepetitionsNode );
            auto lAzimuthMultiSampledXY1       = Repeat( *mScope, lAzimuthXY1, lMultiSamplingRepetitionsNode );
            auto lElevationMultiSampledXY0     = Repeat( *mScope, lElevationXY0, lMultiSamplingRepetitionsNode );
            auto lElevationMultiSampledXY1     = Repeat( *mScope, lElevationXY1, lMultiSamplingRepetitionsNode );

            sRandomUniformInitializerComponent lInitializer{};
            lInitializer.mType                  = eScalarType::FLOAT32;
            auto lAzimuthSamplingCoefficients   = MultiTensorValue( *mScope, lInitializer, lAzimuthMultiSampledXY0.Get<sMultiTensorComponent>().mValue.Shape() );
            auto lElevationSamplingCoefficients = MultiTensorValue( *mScope, lInitializer, lElevationMultiSampledXY0.Get<sMultiTensorComponent>().mValue.Shape() );

            mSampledAzimuths   = Mix( mScope->WithOpName( "Azimuth" ), lAzimuthMultiSampledXY0, lAzimuthMultiSampledXY1, lAzimuthSamplingCoefficients );
            mSampledElevations = Mix( mScope->WithOpName( "Elevation" ), lElevationMultiSampledXY0, lElevationMultiSampledXY1, lElevationSamplingCoefficients );
        }
        else
        {
            sConstantValueInitializerComponent lInitializer{};
            lInitializer.mValue = 0.5f;

            auto lSamplingCoefficients = MultiTensorValue( *mScope, lInitializer, lAzimuthXY0.Get<sMultiTensorComponent>().mValue.Shape() );
            mSampledAzimuths           = Mix( *mScope, lAzimuthXY0, lAzimuthXY1, lSamplingCoefficients );
            mSampledElevations         = Mix( *mScope, lElevationXY0, lElevationXY1, lSamplingCoefficients );
        }

        std::vector<uint32_t> lFlashIdLUTInitialiserValues( mScheduledFlashCount );
        for( uint32_t i = 0; i < mScheduledFlashCount; i++ )
        {
            lFlashIdLUTInitialiserValues[i] = i;
        }

        mSamplingShape = mSampledAzimuths.Get<sMultiTensorComponent>().mValue.Shape();

        sVectorInitializerComponent lFlashIdLUTInitialiser( lFlashIdLUTInitialiserValues );
        mFlashIdLUT = MultiTensorValue( *mScope, lFlashIdLUTInitialiser, mSamplingShape );

        sVectorInitializerComponent lTimestampInitialiser( mTimestamp );
        mTimestamps = MultiTensorValue( *mScope, lTimestampInitialiser, mSamplingShape );

        sConstantValueInitializerComponent lIntensityInitializer( mSpec.mLaserPower );
        mSampleRayIntensities = MultiTensorValue( *mScope, lIntensityInitializer, mSamplingShape );

        // At this point all OpNodes have dimensions that reflect the way the sampling was constructed. These dimensions are no
        // longer needed. We therefore flatten the tensors so that each layer contains a one-dimensional list of all sampled
        // coordinates in the area of that flash. This way all future modifications to the tensors that happen wil better reflect
        // the desired output.
        mSampledAzimuths      = Flatten( mScope->WithOpName( "Azimuth" ), mSampledAzimuths );
        mSampledElevations    = Flatten( mScope->WithOpName( "Elevation" ), mSampledElevations );
        mSampleRayIntensities = Flatten( *mScope, mSampleRayIntensities );

        // Compute the azimuth of each return relative to the center of the configured flash
        auto lAzimuthOffset = ScalarVectorValue( *mScope, eScalarType::FLOAT32, mFlashList.mEnvironmentSampling.mWorldPosition.mX );
        mRelativeAzimuths   = Subtract( mScope->WithOpName( "RelativeAzimuth" ), mSampledAzimuths, lAzimuthOffset );

        // Compute the elevation of each return relative to the center of the configured flash
        auto lElevationOffset = ScalarVectorValue( *mScope, eScalarType::FLOAT32, mFlashList.mEnvironmentSampling.mWorldPosition.mY );
        mRelativeElevations   = Subtract( mScope->WithOpName( "RelativeElevation" ), mSampledElevations, lElevationOffset );

        auto lReductionInterpolatorNode = VectorValue( *mScope, mFlashList.mEnvironmentSampling.mDiffusion );
        auto lDiffusionCoefficient      = Sample2D( *mScope, mRelativeAzimuths, mRelativeElevations, lReductionInterpolatorNode );

        // The return intensities need to be multyplied by the diffuser value.
        mSampleRayIntensities = Multiply( mScope->WithOpName( "Intensity" ), mSampleRayIntensities, lDiffusionCoefficient );

        mFlashIdLUT = Flatten( mScope->WithOpName( "FlashId" ), mSampledAzimuths );
        mTimestamps = Flatten( mScope->WithOpName( "Timestamp" ), mSampledElevations );
    }

    void EnvironmentSampler::Run()
    {
        mScope->Reset();
        CreateGraph();
        {
            BlockTimer lPr( "EnvironmentSampler::Run()" );
            mScope->Run( { mSampledAzimuths, mSampledElevations, mSampleRayIntensities, mTimestamps, mFlashIdLUT } );
        }
        mScope->GetNodesRegistry().ForEach<sGraphOperationComponent>( [&]( auto aNode, auto &aComp ) { aNode.Tag<sDoNotExpand>(); } );
    }

} // namespace LTSE::SensorModel
