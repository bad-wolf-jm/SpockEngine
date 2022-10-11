/// @file   AcquisitionContext.cpp
///
/// @brief  Implementation file for AcquisitionContext
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#include "AcquisitionContext.h"
#include "DeveloperTools/Profiling/BlockTimer.h"

#include <tuple>

namespace LTSE::SensorModel
{
    using namespace math;

    namespace
    {
        std::tuple<uint32_t, uint32_t, uint32_t> CountFlashes( Entity const &aTile )
        {
            uint32_t lFlashCount = static_cast<uint32_t>( aTile.Get<sRelationshipComponent>().mChildren.size() );

            uint32_t lPhotodetectorCellCount        = 0;
            uint32_t lElectronicCrosstalkMatrixSize = 0;
            for( auto &lFlash : aTile.Get<sRelationshipComponent>().mChildren )
            {
                if( lFlash.Has<sJoinComponent<sPhotoDetector>>() )
                {
                    auto lCellCount = lFlash.Get<sJoinComponent<sPhotoDetector>>().JoinedComponent().mCellPositions.size();
                    lPhotodetectorCellCount += lCellCount;
                    lElectronicCrosstalkMatrixSize += ( lCellCount * lCellCount );
                }
                else
                {
                    lPhotodetectorCellCount += 1;
                    lElectronicCrosstalkMatrixSize += 1;
                };
            }

            return { lFlashCount, lPhotodetectorCellCount, lElectronicCrosstalkMatrixSize };
        }

        std::tuple<uint32_t, uint32_t, uint32_t> CountFlashes( std::vector<Entity> const &aTileSequence )
        {
            uint32_t lFlashCount                    = 0;
            uint32_t lPhotodetectorCellCount        = 0;
            uint32_t lElectronicCrosstalkMatrixSize = 0;
            for( auto &lTile : aTileSequence )
            {
                auto [lTileFlashCount, lTilePhotoDetectorCellCount, lTileElectronicCrosstalkMatrixSize] = CountFlashes( lTile );
                lFlashCount += lTileFlashCount;
                lPhotodetectorCellCount += lTilePhotoDetectorCellCount;
                lElectronicCrosstalkMatrixSize += lTileElectronicCrosstalkMatrixSize;
            }

            return { lFlashCount, lPhotodetectorCellCount, lElectronicCrosstalkMatrixSize };
        }

        float PolyEval( vec4 aCoeffs, float x ) { return x * ( x * ( x * aCoeffs.w + aCoeffs.z ) + aCoeffs.y ) + aCoeffs.x; };

    } // namespace

    void sPositionArray::Reserve( size_t aNewSize )
    {
        mX.Reserve( aNewSize );
        mY.Reserve( aNewSize );
    }

    void sPositionArray::Append( vec2 aPoint )
    {
        mX.Append( aPoint.x );
        mY.Append( aPoint.y );
    }

    size_t sPositionArray::Size() { return mX.Size(); }

    void sSizeArray::Reserve( size_t aNewSize )
    {
        mWidth.Reserve( aNewSize );
        mHeight.Reserve( aNewSize );
    }

    void sSizeArray::Append( vec2 aPoint )
    {
        mWidth.Append( aPoint.x );
        mHeight.Append( aPoint.y );
    }

    void sSizeArray::Append( float aWidth, float aHeight )
    {
        mWidth.Append( aWidth );
        mHeight.Append( aHeight );
    }

    size_t sSizeArray::Size() { return mWidth.Size(); }

    void sIntervalArray::Reserve( size_t aNewSize )
    {
        mMin.Reserve( aNewSize );
        mMax.Reserve( aNewSize );
    }

    void sIntervalArray::Append( vec2 aPoint )
    {
        mMin.Append( aPoint.x );
        mMax.Append( aPoint.y );
    }

    void sIntervalArray::Append( float aMin, float aMax )
    {
        mMin.Append( aMin );
        mMax.Append( aMax );
    }

    size_t sIntervalArray::Size() { return mMin.Size(); }

    AcquisitionContext::AcquisitionContext( AcquisitionSpecification const &aSpec, Entity const &aTile, vec2 const &aPosition, float aTimestamp )
        : mSpec{ aSpec }
    {
        LTSE_PROFILE_FUNCTION();

        auto [lFlashCount, lTotalPhotodetectorCellCount, lTotalElectronicCrosstalkMatrixSize] = CountFlashes( aTile );

        ResetInternalStructures( lFlashCount, lTotalPhotodetectorCellCount, lTotalElectronicCrosstalkMatrixSize );

        float lCurrentFlashTimestamp = aTimestamp;
        for( auto &lFlash : aTile.Get<sRelationshipComponent>().mChildren )
        {
            auto lFlashTime = 0.0f;
            if( lFlash.Has<sJoinComponent<sLaserAssembly>>() )
            {
                auto &lComponent = lFlash.Get<sJoinComponent<sLaserAssembly>>().mJoinEntity.Get<sLaserAssembly>();
                lFlashTime       = PolyEval( lComponent.mFlashTime, mSpec.mTemperature );
                mEnvironmentSampling.mLaser.mFlashTime.Append( lFlashTime );
            }
            else
            {
                mEnvironmentSampling.mLaser.mFlashTime.Append( 0.0f );
            }

            AppendLaserFlash( lFlash, aPosition, lCurrentFlashTimestamp );
            lCurrentFlashTimestamp += lFlashTime;
        }
    }

    AcquisitionContext::AcquisitionContext( AcquisitionSpecification const &aSpec, Entity const &aTile, std::vector<vec2> const &aPositions, std::vector<float> const &aTimestamp )
        : mSpec{ aSpec }
    {
        LTSE_PROFILE_FUNCTION();

        auto [lFlashCount, lTotalPhotodetectorCellCount, lTotalElectronicCrosstalkMatrixSize] = CountFlashes( aTile );

        ResetInternalStructures( lFlashCount, lTotalPhotodetectorCellCount, lTotalElectronicCrosstalkMatrixSize );

        for( uint32_t i = 0; i < aPositions.size(); i++ )
        {
            float lCurrentFlashTimestamp = aTimestamp[i];
            for( auto &lFlash : aTile.Get<sRelationshipComponent>().mChildren )
            {
                auto lFlashTime = 0.0f;
                if( lFlash.Has<sJoinComponent<sLaserAssembly>>() )
                {
                    auto &lComponent = lFlash.Get<sJoinComponent<sLaserAssembly>>().mJoinEntity.Get<sLaserAssembly>();
                    lFlashTime       = PolyEval( lComponent.mFlashTime, mSpec.mTemperature );
                    mEnvironmentSampling.mLaser.mFlashTime.Append( lFlashTime );
                }
                else
                {
                    mEnvironmentSampling.mLaser.mFlashTime.Append( 0.0f );
                }

                AppendLaserFlash( lFlash, aPositions[i], lCurrentFlashTimestamp );
                lCurrentFlashTimestamp += lFlashTime;
            }
        }
    }

    AcquisitionContext::AcquisitionContext( AcquisitionSpecification const &aSpec, std::vector<Entity> const &aTileSequence, std::vector<vec2> const &aPositions,
                                            std::vector<float> const &aTimestamp )
        : mSpec{ aSpec }
    {
        LTSE_PROFILE_FUNCTION();

        auto [lFlashCount, lTotalPhotodetectorCellCount, lTotalElectronicCrosstalkMatrixSize] = CountFlashes( aTileSequence );

        ResetInternalStructures( lFlashCount, lTotalPhotodetectorCellCount, lTotalElectronicCrosstalkMatrixSize );

        for( uint32_t i = 0; i < aTileSequence.size(); i++ )
        {
            float lCurrentFlashTimestamp = aTimestamp[i];
            for( auto &lFlash : aTileSequence[i].Get<sRelationshipComponent>().mChildren )
            {
                auto lFlashTime = 0.0f;
                if( lFlash.Has<sJoinComponent<sLaserAssembly>>() )
                {
                    auto &lComponent = lFlash.Get<sJoinComponent<sLaserAssembly>>().mJoinEntity.Get<sLaserAssembly>();
                    lFlashTime       = PolyEval( lComponent.mFlashTime, mSpec.mTemperature );
                    mEnvironmentSampling.mLaser.mFlashTime.Append( lFlashTime );
                }
                else
                {
                    mEnvironmentSampling.mLaser.mFlashTime.Append( 0.0f );
                }

                AppendLaserFlash( lFlash, aPositions[i], lCurrentFlashTimestamp );
                lCurrentFlashTimestamp += lFlashTime;
            }
        }
    }

    void AcquisitionContext::ResetInternalStructures( uint32_t aFlashCount, uint32_t aTotalPhotodetectorCellCount, uint32_t aTotalElectronicCrosstalkMatrixSize )
    {
        mScheduledFlashEntities.Reserve( aFlashCount );

        mEnvironmentSampling.mTileID.Reserve( aFlashCount );
        mEnvironmentSampling.mFlashID.Reserve( aFlashCount );

        mEnvironmentSampling.mTilePosition.Reserve( aFlashCount );
        mEnvironmentSampling.mLocalPosition.Reserve( aFlashCount );
        mEnvironmentSampling.mWorldPosition.Reserve( aFlashCount );
        mEnvironmentSampling.mWorldAzimuth.Reserve( aFlashCount );
        mEnvironmentSampling.mWorldElevation.Reserve( aFlashCount );

        mEnvironmentSampling.mFlashSize.Reserve( aFlashCount );

        mEnvironmentSampling.mTimestamp.Reserve( aFlashCount );
        mEnvironmentSampling.mDiffusion.Reserve( aFlashCount );

        mEnvironmentSampling.mSampling.mLength.Reserve( aFlashCount );
        mEnvironmentSampling.mSampling.mInterval.Reserve( aFlashCount );
        mEnvironmentSampling.mSampling.mFrequency.Reserve( aFlashCount );

        mEnvironmentSampling.mLaser.mPulseTemplate.Reserve( aFlashCount );
        mEnvironmentSampling.mLaser.mTimebaseDelay.Reserve( aFlashCount );
        mEnvironmentSampling.mLaser.mFlashTime.Reserve( aFlashCount );

        mPulseSampling.mPhotoDetectorCellCount.Reserve( aFlashCount );

        mPulseSampling.mPhotoDetectorData.mCellPositions.Reserve( aTotalPhotodetectorCellCount );
        mPulseSampling.mPhotoDetectorData.mCellWorldPositions.Reserve( aTotalPhotodetectorCellCount );
        mPulseSampling.mPhotoDetectorData.mCellTilePositions.Reserve( aTotalPhotodetectorCellCount );

        mPulseSampling.mPhotoDetectorData.mCellWorldAzimuthBounds.Reserve( aTotalPhotodetectorCellCount );
        mPulseSampling.mPhotoDetectorData.mCellWorldElevationBounds.Reserve( aTotalPhotodetectorCellCount );

        mPulseSampling.mPhotoDetectorData.mCellSizes.Reserve( aTotalPhotodetectorCellCount );
        mPulseSampling.mPhotoDetectorData.mBaseline.Reserve( aTotalPhotodetectorCellCount );
        mPulseSampling.mPhotoDetectorData.mGain.Reserve( aTotalPhotodetectorCellCount );
        mPulseSampling.mPhotoDetectorData.mStaticNoiseShift.Reserve( aTotalPhotodetectorCellCount );
        mPulseSampling.mPhotoDetectorData.mStaticNoise.Reserve( aTotalPhotodetectorCellCount );

        mPulseSampling.mPhotoDetectorData.mElectronicCrosstalk.Reserve( aTotalElectronicCrosstalkMatrixSize );
    }

    void AcquisitionContext::AppendLaserFlash( Entity const &aFlash, vec2 aTilePosition, float aTimestamp )
    {
        auto &lTile                    = aFlash.Get<sRelationshipComponent>().mParent;
        auto &lLaserFlashSpecification = aFlash.Get<sLaserFlashSpecificationComponent>();

        mScheduledFlashEntities.Append( aFlash );

        mEnvironmentSampling.mTileID.Append( lTile.Get<sTileSpecificationComponent>().mID );
        mEnvironmentSampling.mFlashID.Append( lLaserFlashSpecification.mFlashID );

        auto lWorldPosition = lLaserFlashSpecification.mPosition + aTilePosition;
        mEnvironmentSampling.mTilePosition.Append( aTilePosition );
        mEnvironmentSampling.mLocalPosition.Append( lLaserFlashSpecification.mPosition );
        mEnvironmentSampling.mWorldPosition.Append( lWorldPosition );
        mEnvironmentSampling.mTimestamp.Append( aTimestamp );

        if( lTile.Has<sJoinComponent<sSampler>>() && lTile.Get<sJoinComponent<sSampler>>().mJoinEntity )
        {
            auto &lSamplerComponent = lTile.Get<sJoinComponent<sSampler>>().JoinedComponent();

            // The sampling length is the number of points to sample to produce a raw waveform. It is given
            // by the basepoint value multiplied by the oversampling value.
            mEnvironmentSampling.mSampling.mLength.Append( mSpec.mBasePoints * mSpec.mOversampling );

            // Calculate the effective sampling interval. The effective sampling frequency is given by
            // F x OVR, where F is the ADC sampling frequency, and OVR is the oversampling value. For
            // example, an ADC sampling at 100MHz oversampled by a factored of 4 will give an effective
            // frequency of 400Mhz. The sampling interval is the multiplicative inverse of the sampling
            // frequency. For convenience we represent the sampling interval in nanoseconds.
            mEnvironmentSampling.mSampling.mInterval.Append( ( 1.0f / ( lSamplerComponent.mFrequency * mSpec.mOversampling ) ) * 1000000000.0f );

            // Sampling frequency remains in Hertz
            mEnvironmentSampling.mSampling.mFrequency.Append( lSamplerComponent.mFrequency * mSpec.mOversampling );
        }
        else
        {
            mEnvironmentSampling.mSampling.mLength.Append( 0 );
            mEnvironmentSampling.mSampling.mInterval.Append( 1.0f );
            mEnvironmentSampling.mSampling.mFrequency.Append( 1.0f );
        }

        if( aFlash.Has<sJoinComponent<sLaserAssembly>>() )
        {
            auto &lComponent = aFlash.Get<sJoinComponent<sLaserAssembly>>().mJoinEntity.Get<sLaserAssembly>();

            if( lComponent.mDiffuserData.Has<sDiffusionPattern>() )
            {
                auto &lDiffusionPattern = lComponent.mDiffuserData.Get<sDiffusionPattern>();

                mEnvironmentSampling.mFlashSize.Append( lDiffusionPattern.mInterpolator->mDeviceData.mScaling );

                vec2 lActualExtent        = lDiffusionPattern.mInterpolator->mDeviceData.mScaling / 2.0f;
                vec2 lFieldOfViewRangeMin = lWorldPosition - lActualExtent;
                vec2 lFieldOfViewRangeMax = lWorldPosition + lActualExtent;
                mEnvironmentSampling.mWorldAzimuth.Append( lFieldOfViewRangeMin.x, lFieldOfViewRangeMax.x );
                mEnvironmentSampling.mWorldElevation.Append( lFieldOfViewRangeMin.y, lFieldOfViewRangeMax.y );
                mEnvironmentSampling.mDiffusion.Append( lDiffusionPattern.mInterpolator->mDeviceData );
            }
            else
            {
                mEnvironmentSampling.mFlashSize.Append( lLaserFlashSpecification.mExtent * 2.0f );

                vec2 lFieldOfViewRangeMin = lWorldPosition - lLaserFlashSpecification.mExtent;
                vec2 lFieldOfViewRangeMax = lWorldPosition + lLaserFlashSpecification.mExtent;
                mEnvironmentSampling.mWorldAzimuth.Append( lFieldOfViewRangeMin.x, lFieldOfViewRangeMax.x );
                mEnvironmentSampling.mWorldElevation.Append( lFieldOfViewRangeMin.y, lFieldOfViewRangeMax.y );
                mEnvironmentSampling.mDiffusion.Append( TextureSampler{} );
            }

            if( lComponent.mWaveformData.Has<sPulseTemplate>() )
                mEnvironmentSampling.mLaser.mPulseTemplate.Append( lComponent.mWaveformData.Get<sPulseTemplate>().mInterpolator->mDeviceData );
            else
                mEnvironmentSampling.mLaser.mPulseTemplate.Append( TextureSampler{} );

            mEnvironmentSampling.mLaser.mTimebaseDelay.Append( PolyEval( lComponent.mTimebaseDelay, mSpec.mTemperature ) );
        }
        else
        {
            mEnvironmentSampling.mLaser.mPulseTemplate.Append( TextureSampler{} );
            mEnvironmentSampling.mLaser.mTimebaseDelay.Append( 0.0f );

            mEnvironmentSampling.mFlashSize.Append( lLaserFlashSpecification.mExtent * 2.0f );

            vec2 lFieldOfViewRangeMin = lWorldPosition - lLaserFlashSpecification.mExtent;
            vec2 lFieldOfViewRangeMax = lWorldPosition + lLaserFlashSpecification.mExtent;
            mEnvironmentSampling.mWorldAzimuth.Append( lFieldOfViewRangeMin.x, lFieldOfViewRangeMax.x );
            mEnvironmentSampling.mWorldElevation.Append( lFieldOfViewRangeMin.y, lFieldOfViewRangeMax.y );
            mEnvironmentSampling.mDiffusion.Append( TextureSampler{} );
        }

        if( aFlash.Has<sJoinComponent<sPhotoDetector>>() )
        {
            auto &lComponent = aFlash.Get<sJoinComponent<sPhotoDetector>>().JoinedComponent();

            AppendPhotoDetectorData( lComponent, lLaserFlashSpecification.mPosition, aTilePosition );
        }
        else
        {
            AppendPhotoDetectorData( lLaserFlashSpecification.mPosition, aTilePosition );
        }
    }

    void AcquisitionContext::AppendCellPosition( vec4 const &aCellPosition, vec2 aFlashPosition, vec2 aTilePosition )
    {
        vec2 lCellPosition = { aCellPosition.x, aCellPosition.y };
        vec2 lCellSize     = { aCellPosition.z, aCellPosition.w };
        mPulseSampling.mPhotoDetectorData.mFlashIndex.Append( 0 );
        mPulseSampling.mPhotoDetectorData.mCellPositions.Append( lCellPosition );
        mPulseSampling.mPhotoDetectorData.mCellTilePositions.Append( lCellPosition + aFlashPosition );
        mPulseSampling.mPhotoDetectorData.mCellWorldPositions.Append( lCellPosition + aFlashPosition + aTilePosition );

        vec2 lCellWorldPosition         = lCellPosition + aFlashPosition + aTilePosition;
        vec2 lPhotodetectorCellWorldMin = lCellWorldPosition - lCellSize;
        vec2 lPhotodetectorCellWorldMax = lCellWorldPosition + lCellSize;
        mPulseSampling.mPhotoDetectorData.mCellWorldAzimuthBounds.Append( lPhotodetectorCellWorldMin.x, lPhotodetectorCellWorldMax.x );
        mPulseSampling.mPhotoDetectorData.mCellWorldElevationBounds.Append( lPhotodetectorCellWorldMin.y, lPhotodetectorCellWorldMax.y );

        mPulseSampling.mPhotoDetectorData.mCellSizes.Append( aCellPosition.z, aCellPosition.w );
    }

    void AcquisitionContext::AppendPhotoDetectorData( sPhotoDetector const &aPhotoDetector, vec2 aFlashPosition, vec2 aTilePosition )
    {
        mPulseSampling.mPhotoDetectorCellCount.Append( aPhotoDetector.mCellPositions.size() );

        for( uint32_t j = 0; j < aPhotoDetector.mCellPositions.size(); j++ )
        {
            AppendCellPosition( aPhotoDetector.mCellPositions[j], aFlashPosition, aTilePosition );

            mPulseSampling.mPhotoDetectorData.mBaseline.Append( PolyEval( aPhotoDetector.mBaseline[j], mSpec.mTemperature ) );
            mPulseSampling.mPhotoDetectorData.mGain.Append( PolyEval( aPhotoDetector.mGain[j], mSpec.mTemperature ) );

            auto lStaticNoise = aPhotoDetector.mStaticNoise[j];
            if( lStaticNoise.Has<sStaticNoise>() && lStaticNoise.Get<sStaticNoise>().mInterpolator )
            {
                mPulseSampling.mPhotoDetectorData.mStaticNoiseShift.Append( PolyEval( aPhotoDetector.mStaticNoiseShift[j], mSpec.mTemperature ) );
                mPulseSampling.mPhotoDetectorData.mStaticNoise.Append( lStaticNoise.Get<sStaticNoise>().mInterpolator->mDeviceData );
            }
            else
            {
                mPulseSampling.mPhotoDetectorData.mStaticNoiseShift.Append( 0.0f );
                mPulseSampling.mPhotoDetectorData.mStaticNoise.Append( Cuda::TextureSampler2D::DeviceData{} );
            }

        }

        for( uint32_t j = 0; j < aPhotoDetector.mElectronicCrosstalk.size(); j++ )
        {
            auto lElectronicCrosstalk = aPhotoDetector.mElectronicCrosstalk[j];
            if( lElectronicCrosstalk.Has<sElectronicCrosstalk>() && lElectronicCrosstalk.Get<sElectronicCrosstalk>().mInterpolator )
                mPulseSampling.mPhotoDetectorData.mElectronicCrosstalk.Append( lElectronicCrosstalk.Get<sElectronicCrosstalk>().mInterpolator->mDeviceData );
            else
                mPulseSampling.mPhotoDetectorData.mElectronicCrosstalk.Append( Cuda::TextureSampler2D::DeviceData{} );
        }
    }

    void AcquisitionContext::AppendPhotoDetectorData( vec2 aFlashPosition, vec2 aTilePosition )
    {
        AppendCellPosition( vec4{ 0.0f, 0.0f, 0.0f, 0.0f }, vec2{ 0.0f, 0.0f }, vec2{ 0.0f, 0.0f } );

        mPulseSampling.mPhotoDetectorData.mBaseline.Append( 0.0f );
        mPulseSampling.mPhotoDetectorData.mGain.Append( 0.0f );
        mPulseSampling.mPhotoDetectorData.mStaticNoise.Append( Cuda::TextureSampler2D::DeviceData{} );
        mPulseSampling.mPhotoDetectorData.mStaticNoiseShift.Append( 0.0f );
        mPulseSampling.mPhotoDetectorCellCount.Append( 1 );
    }
} // namespace LTSE::SensorModel