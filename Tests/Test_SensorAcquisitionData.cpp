#include <catch2/catch_test_macros.hpp>

#include "LidarSensorModel/AcquisitionContext/AcquisitionContext.h"
#include "LidarSensorModel/Components.h"
#include "LidarSensorModel/SensorModelBase.h"

#include "TestUtils.h"

using namespace math;
using namespace TestUtils;
using namespace LTSE::Core;
using namespace LTSE::SensorModel;

#define CHECK_COLUMN_SIZE( lLaserFlashList, lExpectedColumnSize )                                                                                                                  \
    {                                                                                                                                                                              \
        auto lEnvSampling = lLaserFlashList.mEnvironmentSampling;                                                                                                                  \
        REQUIRE( lLaserFlashList.mScheduledFlashEntities.Size() == lExpectedColumnSize );                                                                                          \
        REQUIRE( lEnvSampling.mTileID.Size() == lExpectedColumnSize );                                                                                                             \
        REQUIRE( lEnvSampling.mFlashID.Size() == lExpectedColumnSize );                                                                                                            \
        REQUIRE( lEnvSampling.mTilePosition.Size() == lExpectedColumnSize );                                                                                                       \
        REQUIRE( lEnvSampling.mWorldPosition.Size() == lExpectedColumnSize );                                                                                                      \
        REQUIRE( lEnvSampling.mLocalPosition.Size() == lExpectedColumnSize );                                                                                                      \
        REQUIRE( lEnvSampling.mWorldAzimuth.Size() == lExpectedColumnSize );                                                                                                       \
        REQUIRE( lEnvSampling.mWorldElevation.Size() == lExpectedColumnSize );                                                                                                     \
        REQUIRE( lEnvSampling.mFlashSize.Size() == lExpectedColumnSize );                                                                                                          \
        REQUIRE( lEnvSampling.mTimestamp.Size() == lExpectedColumnSize );                                                                                                          \
        REQUIRE( lEnvSampling.mDiffusion.Size() == lExpectedColumnSize );                                                                                                          \
        REQUIRE( lEnvSampling.mSampling.mLength.Size() == lExpectedColumnSize );                                                                                                   \
        REQUIRE( lEnvSampling.mSampling.mInterval.Size() == lExpectedColumnSize );                                                                                                 \
        REQUIRE( lEnvSampling.mLaser.mPulseTemplate.Size() == lExpectedColumnSize );                                                                                               \
        REQUIRE( lEnvSampling.mLaser.mTimebaseDelay.Size() == lExpectedColumnSize );                                                                                               \
        REQUIRE( lEnvSampling.mLaser.mFlashTime.Size() == lExpectedColumnSize );                                                                                                   \
        REQUIRE( lLaserFlashList.mPulseSampling.mPhotoDetectorCellCount.Size() == lExpectedColumnSize );                                                                           \
    }

TEST_CASE( "Schedule single tile", "[CORE_SENSOR_MODEL]" )
{
    SensorModelBase lSensorModelBase{};

    Entity lTile1   = lSensorModelBase.CreateTile( "1", vec2{ 1.0f, 0.0f } );
    Entity lFlash10 = lSensorModelBase.CreateFlash( lTile1, "0", vec2{ -1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );
    Entity lFlash11 = lSensorModelBase.CreateFlash( lTile1, "1", vec2{ 1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );
    Entity lFlash12 = lSensorModelBase.CreateFlash( lTile1, "2", vec2{ 2.0f, 0.0f }, vec2{ 10.0f, 10.0f } );

    AcquisitionSpecification lAcqCreateInfo{};
    AcquisitionContext lLaserFlashList( lAcqCreateInfo, lTile1, math::vec2{ 2.0f, 2.0f }, 0.0f );

    constexpr uint32_t lExpectedColumnSize = 3;
    CHECK_COLUMN_SIZE( lLaserFlashList, lExpectedColumnSize );
}

TEST_CASE( "Schedule single tile multiple times", "[CORE_SENSOR_MODEL]" )
{
    SensorModelBase lSensorModelBase{};

    Entity lTile1 = lSensorModelBase.CreateTile( "0", vec2{ 1.0f, 0.0f } );
    lSensorModelBase.CreateFlash( lTile1, "0", vec2{ -1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );
    lSensorModelBase.CreateFlash( lTile1, "1", vec2{ 1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );
    lSensorModelBase.CreateFlash( lTile1, "2", vec2{ 2.0f, 0.0f }, vec2{ 10.0f, 10.0f } );

    std::vector<vec2> lTilesPositionsToSchedule = { vec2{ 0.0f, 0.0f }, vec2{ 1.0f, 1.0f }, vec2{ 2.0f, 2.0f } };
    std::vector<float> lTimings                 = { 0.0f, 1.0f, 2.0f };

    AcquisitionSpecification lAcqCreateInfo{};
    AcquisitionContext lLaserFlashList( lAcqCreateInfo, lTile1, lTilesPositionsToSchedule, lTimings );

    constexpr uint32_t lExpectedColumnSize = 9;
    CHECK_COLUMN_SIZE( lLaserFlashList, lExpectedColumnSize );
}

TEST_CASE( "Schedule multiple tiles", "[CORE_SENSOR_MODEL]" )
{
    SensorModelBase lSensorModelBase{};

    Entity lTile1 = lSensorModelBase.CreateTile( "0", vec2{ 1.0f, 0.0f } );
    lSensorModelBase.CreateFlash( lTile1, "0", vec2{ -1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );
    lSensorModelBase.CreateFlash( lTile1, "1", vec2{ 1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );

    Entity lTile2 = lSensorModelBase.CreateTile( "1", vec2{ 1.0f, 0.0f } );
    lSensorModelBase.CreateFlash( lTile2, "0", vec2{ -1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );
    lSensorModelBase.CreateFlash( lTile2, "1", vec2{ 1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );
    lSensorModelBase.CreateFlash( lTile2, "2", vec2{ 2.0f, 0.0f }, vec2{ 10.0f, 10.0f } );

    std::vector<Entity> lTilesToSchedule        = { lTile1, lTile1, lTile2 };
    std::vector<vec2> lTilesPositionsToSchedule = { vec2{ 0.0f, 0.0f }, vec2{ 1.0f, 1.0f }, vec2{ 2.0f, 2.0f } };
    std::vector<float> lTimings                 = { 0.0f, 1.0f, 2.0f };

    AcquisitionSpecification lAcqCreateInfo{};
    AcquisitionContext lLaserFlashList( lAcqCreateInfo, lTilesToSchedule, lTilesPositionsToSchedule, lTimings );

    constexpr uint32_t lExpectedColumnSize = 7;
    CHECK_COLUMN_SIZE( lLaserFlashList, lExpectedColumnSize );
}

TEST_CASE( "Check that flashes have the correct timestamps (single tile)", "[CORE_SENSOR_MODEL]" )
{
    SensorModelBase lSensorModelBase{};

    Entity lTile1  = lSensorModelBase.CreateTile( "0", vec2{ 1.0f, 0.0f } );
    Entity lFlash0 = lSensorModelBase.CreateFlash( lTile1, "0", vec2{ -1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );
    Entity lFlash1 = lSensorModelBase.CreateFlash( lTile1, "1", vec2{ 1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );
    Entity lFlash2 = lSensorModelBase.CreateFlash( lTile1, "2", vec2{ 1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );

    sLaserAssembly lLaserAssembly{};
    lLaserAssembly.mFlashTime = { 0.23245f, 0.1345f, 0.332f, 0.123f };

    auto lLaser = lSensorModelBase.CreateElement( "laser", "laser", lLaserAssembly );
    lFlash0.Adjoin<sLaserAssembly>( lLaser );
    lFlash1.Adjoin<sLaserAssembly>( lLaser );
    lFlash2.Adjoin<sLaserAssembly>( lLaser );

    AcquisitionSpecification lAcqCreateInfo{};
    lAcqCreateInfo.mTemperature = 3.0f;

    float lFlashTime = lLaserAssembly.mFlashTime.w * std::pow( lAcqCreateInfo.mTemperature, 3.0f ) + lLaserAssembly.mFlashTime.z * std::pow( lAcqCreateInfo.mTemperature, 2.0f ) +
                       lLaserAssembly.mFlashTime.y * lAcqCreateInfo.mTemperature + lLaserAssembly.mFlashTime.x;

    AcquisitionContext lLaserFlashList( lAcqCreateInfo, lTile1, math::vec2{ 1.0f, 1.0f }, 2.0f );

    std::vector<float> lExpectedFlashTimes = { lFlashTime, lFlashTime, lFlashTime };
    REQUIRE( VectorEqual( lLaserFlashList.mEnvironmentSampling.mLaser.mFlashTime, lExpectedFlashTimes ) );

    std::vector<float> lExpectedTimestamps = { 2.0f, 2.0f + lFlashTime, 2.0f + 2 * lFlashTime };
    REQUIRE( lLaserFlashList.mEnvironmentSampling.mTimestamp == lExpectedTimestamps );
}

TEST_CASE( "Check that flashes have the correct timestamps (single tile submitted multiple times)", "[CORE_SENSOR_MODEL]" )
{
    SensorModelBase lSensorModelBase{};

    Entity lTile1  = lSensorModelBase.CreateTile( "1", vec2{ 1.0f, 0.0f } );
    Entity lFlash0 = lSensorModelBase.CreateFlash( lTile1, "0", vec2{ -1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );
    Entity lFlash1 = lSensorModelBase.CreateFlash( lTile1, "1", vec2{ 1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );
    Entity lFlash2 = lSensorModelBase.CreateFlash( lTile1, "2", vec2{ 1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );

    sLaserAssembly lLaserAssembly{};
    lLaserAssembly.mFlashTime = { 0.23245f, 0.1345f, 0.332f, 0.123f };

    auto lLaser = lSensorModelBase.CreateElement( "laser", "laser", lLaserAssembly );
    lFlash0.Adjoin<sLaserAssembly>( lLaser );
    lFlash1.Adjoin<sLaserAssembly>( lLaser );
    lFlash2.Adjoin<sLaserAssembly>( lLaser );

    AcquisitionSpecification lAcqCreateInfo{};
    lAcqCreateInfo.mTemperature = 3.0f;

    float lFlashTime = lLaserAssembly.mFlashTime.w * std::pow( lAcqCreateInfo.mTemperature, 3.0f ) + lLaserAssembly.mFlashTime.z * std::pow( lAcqCreateInfo.mTemperature, 2.0f ) +
                       lLaserAssembly.mFlashTime.y * lAcqCreateInfo.mTemperature + lLaserAssembly.mFlashTime.x;

    std::vector<vec2> lTilesPositionsToSchedule = { vec2{ 0.0f, 0.0f }, vec2{ 1.0f, 1.0f }, vec2{ 2.0f, 2.0f } };
    std::vector<float> lTimings                 = { 0.0f, 1.0f, 2.0f };

    AcquisitionContext lLaserFlashList( lAcqCreateInfo, lTile1, lTilesPositionsToSchedule, lTimings );

    std::vector<float> lExpectedFlashTimes = { lFlashTime, lFlashTime, lFlashTime };
    REQUIRE( VectorEqual( lLaserFlashList.mEnvironmentSampling.mLaser.mFlashTime, lExpectedFlashTimes ) );

    std::vector<float> lExpectedTimestamps = { 0.0f, 0.0f + lFlashTime, 0.0f + 2 * lFlashTime, 1.0f, 1.0f + lFlashTime, 1.0f + 2 * lFlashTime,
                                               2.0f, 2.0f + lFlashTime, 2.0f + 2 * lFlashTime };
    REQUIRE( VectorEqual( lLaserFlashList.mEnvironmentSampling.mTimestamp, lExpectedTimestamps ) );
}

TEST_CASE( "Check that flashes have the correct timestamps (multiple tiles)", "[CORE_SENSOR_MODEL]" )
{
    SensorModelBase lSensorModelBase{};

    Entity lTile1   = lSensorModelBase.CreateTile( "0", vec2{ 1.0f, 0.0f } );
    Entity lFlash10 = lSensorModelBase.CreateFlash( lTile1, "0", vec2{ -1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );
    Entity lFlash11 = lSensorModelBase.CreateFlash( lTile1, "1", vec2{ 1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );

    Entity lTile2   = lSensorModelBase.CreateTile( "1", vec2{ 1.0f, 0.0f } );
    Entity lFlash20 = lSensorModelBase.CreateFlash( lTile2, "0", vec2{ -1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );
    Entity lFlash21 = lSensorModelBase.CreateFlash( lTile2, "1", vec2{ 1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );
    Entity lFlash22 = lSensorModelBase.CreateFlash( lTile2, "2", vec2{ 1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );

    sLaserAssembly lLaserAssembly{};
    lLaserAssembly.mFlashTime = { 0.23245f, 0.1345f, 0.332f, 0.123f };

    auto lLaser = lSensorModelBase.CreateElement( "laser", "laser", lLaserAssembly );
    lFlash10.Adjoin<sLaserAssembly>( lLaser );
    lFlash11.Adjoin<sLaserAssembly>( lLaser );
    lFlash20.Adjoin<sLaserAssembly>( lLaser );
    lFlash21.Adjoin<sLaserAssembly>( lLaser );
    lFlash22.Adjoin<sLaserAssembly>( lLaser );

    AcquisitionSpecification lAcqCreateInfo{};
    lAcqCreateInfo.mTemperature = 3.0f;

    float lFlashTime = lLaserAssembly.mFlashTime.w * std::pow( lAcqCreateInfo.mTemperature, 3.0f ) + lLaserAssembly.mFlashTime.z * std::pow( lAcqCreateInfo.mTemperature, 2.0f ) +
                       lLaserAssembly.mFlashTime.y * lAcqCreateInfo.mTemperature + lLaserAssembly.mFlashTime.x;

    std::vector<Entity> lTilesToSchedule        = { lTile1, lTile1, lTile2 };
    std::vector<vec2> lTilesPositionsToSchedule = { vec2{ 0.0f, 0.0f }, vec2{ 1.0f, 1.0f }, vec2{ 2.0f, 2.0f } };
    std::vector<float> lTimings                 = { 0.0f, 1.0f, 2.0f };

    AcquisitionContext lLaserFlashList( lAcqCreateInfo, lTilesToSchedule, lTilesPositionsToSchedule, lTimings );

    std::vector<float> lExpectedFlashTimes = { lFlashTime, lFlashTime, lFlashTime };
    REQUIRE( VectorEqual( lLaserFlashList.mEnvironmentSampling.mLaser.mFlashTime, lExpectedFlashTimes ) );

    std::vector<float> lExpectedTimestamps = { 0.0f, 0.0f + lFlashTime, 1.0f, 1.0f + lFlashTime, 2.0f, 2.0f + lFlashTime, 2.0f + 2 * lFlashTime };
    REQUIRE( VectorEqual( lLaserFlashList.mEnvironmentSampling.mTimestamp, lExpectedTimestamps ) );
}

#define CHECK_PD_COLUMN_SIZE( lLaserFlashList, lExpectedColumnSize )                                                                                                               \
    {                                                                                                                                                                              \
        auto lPulseSampling = lLaserFlashList.mPulseSampling.mPhotoDetectorData;                                                                                                   \
        REQUIRE( lPulseSampling.mFlashIndex.Size() == lExpectedColumnSize );                                                                                                       \
        REQUIRE( lPulseSampling.mCellPositions.Size() == lExpectedColumnSize );                                                                                                    \
        REQUIRE( lPulseSampling.mCellWorldPositions.Size() == lExpectedColumnSize );                                                                                               \
        REQUIRE( lPulseSampling.mCellTilePositions.Size() == lExpectedColumnSize );                                                                                                \
        REQUIRE( lPulseSampling.mCellWorldAzimuthBounds.Size() == lExpectedColumnSize );                                                                                           \
        REQUIRE( lPulseSampling.mCellWorldElevationBounds.Size() == lExpectedColumnSize );                                                                                         \
        REQUIRE( lPulseSampling.mCellSizes.Size() == lExpectedColumnSize );                                                                                                        \
        REQUIRE( lPulseSampling.mBaseline.Size() == lExpectedColumnSize );                                                                                                         \
        REQUIRE( lPulseSampling.mGain.Size() == lExpectedColumnSize );                                                                                                             \
        REQUIRE( lPulseSampling.mStaticNoise.Size() == lExpectedColumnSize );                                                                                                      \
        REQUIRE( lPulseSampling.mStaticNoiseShift.Size() == lExpectedColumnSize );                                                                                                 \
    }

template <typename _Ty> static void Repeat( std::vector<_Ty> &aResult, std::vector<_Ty> const &aArray, uint32_t aTimes )
{
    for( uint32_t i = 0; i < aTimes; i++ )
        aResult.insert( aResult.end(), aArray.begin(), aArray.end() );
}

static void Repeat( sFloat32Array &aResult, sFloat32Array const &aArray, uint32_t aTimes )
{
    for( uint32_t i = 0; i < aTimes; i++ )
        aResult.insert( aResult.end(), aArray.begin(), aArray.end() );
}

static float EvaluatePolynomial( math::vec4 aCoeffs, float x ) { return aCoeffs.w * x * x * x + aCoeffs.z * x * x + aCoeffs.y * x + aCoeffs.x; }

TEST_CASE( "Check that photodetector data has the correct length (single tile)", "[CORE_SENSOR_MODEL]" )
{
    SensorModelBase lSensorModelBase{};

    Entity lTile1  = lSensorModelBase.CreateTile( "0", vec2{ 1.0f, 0.0f } );
    Entity lFlash0 = lSensorModelBase.CreateFlash( lTile1, "0", vec2{ -1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );
    Entity lFlash1 = lSensorModelBase.CreateFlash( lTile1, "1", vec2{ 1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );
    Entity lFlash2 = lSensorModelBase.CreateFlash( lTile1, "2", vec2{ 1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );

    sLaserAssembly lLaserAssembly{};
    lLaserAssembly.mFlashTime = { 0.23245f, 0.1345f, 0.332f, 0.123f };

    auto lLaser = lSensorModelBase.CreateElement( "laser", "laser", lLaserAssembly );
    lFlash0.Adjoin<sLaserAssembly>( lLaser );
    lFlash1.Adjoin<sLaserAssembly>( lLaser );
    lFlash2.Adjoin<sLaserAssembly>( lLaser );

    sPhotoDetector lPhotoDetectorComponent{};
    lPhotoDetectorComponent.mCellPositions = { math::vec4{ 0.1234f, 1.3456f, 0.8234f, 0.56798f }, math::vec4{ 0.413275986f, 5.09487152f, 0.310279485f, 0.70198432f },
                                               math::vec4{ 0.163279845f, 0.498675213f, 0.31924078f, 0.17983402f } };

    lPhotoDetectorComponent.mGain = { math::vec4{ 0.1234f, 1.3456f, 0.8234f, 0.56798f }, math::vec4{ 0.413275986f, 5.09487152f, 0.310279485f, 0.70198432f },
                                      math::vec4{ 0.163279845f, 0.498675213f, 0.31924078f, 0.17983402f } };

    lPhotoDetectorComponent.mBaseline = { math::vec4{ 0.91730842f, 0.10923478f, 0.13470892f, 0.13427098f }, math::vec4{ 0.13097248f, 0.19347082f, 0.30172498f, 0.7092831f },
                                          math::vec4{ 0.2137098f, 0.13624978f, 0.12347908f, 0.13246798f } };

    lPhotoDetectorComponent.mStaticNoise      = { Entity{}, Entity{}, Entity{} };
    lPhotoDetectorComponent.mStaticNoiseShift = { math::vec4{ 0.91730842f, 0.10923478f, 0.13470892f, 0.13427098f }, math::vec4{ 0.13097248f, 0.19347082f, 0.30172498f, 0.7092831f },
                                                  math::vec4{ 0.2137098f, 0.13624978f, 0.12347908f, 0.13246798f } };

    auto lPhotoDetector = lSensorModelBase.CreateElement( "PD", "apd_0", lPhotoDetectorComponent );
    lFlash0.Adjoin<sPhotoDetector>( lPhotoDetector );
    lFlash1.Adjoin<sPhotoDetector>( lPhotoDetector );
    lFlash2.Adjoin<sPhotoDetector>( lPhotoDetector );

    AcquisitionSpecification lAcqCreateInfo{};
    lAcqCreateInfo.mTemperature = 3.0f;

    float lFlashTime = EvaluatePolynomial( lLaserAssembly.mFlashTime, lAcqCreateInfo.mTemperature );

    AcquisitionContext lLaserFlashList( lAcqCreateInfo, lTile1, math::vec2{ 1.17023948f, 11.0347298f }, 2.0f );

    CHECK_PD_COLUMN_SIZE( lLaserFlashList, 9 );
    REQUIRE( ( lLaserFlashList.mPulseSampling.mPhotoDetectorCellCount == std::vector<uint32_t>{ 3, 3, 3 } ) );

    sPositionArray lExpectedCellPositions0{};
    for( auto &lCellPos : lPhotoDetectorComponent.mCellPositions )
        lExpectedCellPositions0.Append( math::vec2{ lCellPos.x, lCellPos.y } );

    sPositionArray lExpectedCellPositions{};
    Repeat( lExpectedCellPositions.mX, lExpectedCellPositions0.mX, 3 );
    REQUIRE( lLaserFlashList.mPulseSampling.mPhotoDetectorData.mCellPositions.mX == lExpectedCellPositions.mX );

    Repeat( lExpectedCellPositions.mY, lExpectedCellPositions0.mY, 3 );
    REQUIRE( lLaserFlashList.mPulseSampling.mPhotoDetectorData.mCellPositions.mY == lExpectedCellPositions.mY );

    sPositionArray lExpectedCellWorldPositions0{};
    sPositionArray lExpectedCellTilePositions0{};
    for( auto &lFlash : lTile1.Get<sRelationshipComponent>().mChildren )
    {
        math::vec2 lFlashPosition = lFlash.Get<sLaserFlashSpecificationComponent>().mPosition;
        for( auto &lCellPos : lPhotoDetectorComponent.mCellPositions )
        {
            lExpectedCellWorldPositions0.Append( math::vec2{ lCellPos.x, lCellPos.y } + lFlashPosition + math::vec2{ 1.17023948f, 11.0347298f } );
            lExpectedCellTilePositions0.Append( math::vec2{ lCellPos.x, lCellPos.y } + lFlashPosition );
        }
    }

    REQUIRE( lLaserFlashList.mPulseSampling.mPhotoDetectorData.mCellWorldPositions.mX == lExpectedCellWorldPositions0.mX );
    REQUIRE( lLaserFlashList.mPulseSampling.mPhotoDetectorData.mCellWorldPositions.mY == lExpectedCellWorldPositions0.mY );

    REQUIRE( lLaserFlashList.mPulseSampling.mPhotoDetectorData.mCellTilePositions.mX == lExpectedCellTilePositions0.mX );
    REQUIRE( lLaserFlashList.mPulseSampling.mPhotoDetectorData.mCellTilePositions.mY == lExpectedCellTilePositions0.mY );

    sSizeArray lExpectedCellSizes0{};
    for( auto &lCellSize : lPhotoDetectorComponent.mCellPositions )
        lExpectedCellSizes0.Append( math::vec2{ lCellSize.z, lCellSize.w } );

    sSizeArray lExpectedCellSizes{};
    Repeat( lExpectedCellSizes.mWidth, lExpectedCellSizes0.mWidth, 3 );
    REQUIRE( lLaserFlashList.mPulseSampling.mPhotoDetectorData.mCellSizes.mWidth == lExpectedCellSizes.mWidth );

    Repeat( lExpectedCellSizes.mHeight, lExpectedCellSizes0.mHeight, 3 );
    REQUIRE( lLaserFlashList.mPulseSampling.mPhotoDetectorData.mCellSizes.mHeight == lExpectedCellSizes.mHeight );

    sFloat32Array lExpectedBaseline0{};
    for( auto &lBaseline : lPhotoDetectorComponent.mBaseline )
        lExpectedBaseline0.Append( EvaluatePolynomial( lBaseline, lAcqCreateInfo.mTemperature ) );
    sFloat32Array lExpectedBaseline{};
    Repeat( lExpectedBaseline, lExpectedBaseline0, 3 );
    REQUIRE( VectorEqual( lLaserFlashList.mPulseSampling.mPhotoDetectorData.mBaseline, lExpectedBaseline, 0.0001f ) );
}

TEST_CASE( "Check that photodetector data has the correct length (single tile submitted multiple times)", "[CORE_SENSOR_MODEL]" )
{
    SensorModelBase lSensorModelBase{};

    Entity lTile1  = lSensorModelBase.CreateTile( "0", vec2{ 1.0f, 0.0f } );
    Entity lFlash0 = lSensorModelBase.CreateFlash( lTile1, "0", vec2{ -1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );
    Entity lFlash1 = lSensorModelBase.CreateFlash( lTile1, "1", vec2{ 1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );
    Entity lFlash2 = lSensorModelBase.CreateFlash( lTile1, "2", vec2{ 1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );

    sLaserAssembly lLaserAssembly{};
    lLaserAssembly.mFlashTime = { 0.23245f, 0.1345f, 0.332f, 0.123f };

    auto lLaser = lSensorModelBase.CreateElement( "laser", "laser", lLaserAssembly );
    lFlash0.Adjoin<sLaserAssembly>( lLaser );
    lFlash1.Adjoin<sLaserAssembly>( lLaser );
    lFlash2.Adjoin<sLaserAssembly>( lLaser );

    sPhotoDetector lPhotoDetectorComponent{};
    lPhotoDetectorComponent.mCellPositions = { math::vec4{ 0.1234f, 1.3456f, 0.8234f, 0.56798f }, math::vec4{ 0.413275986f, 5.09487152f, 0.310279485f, 0.70198432f },
                                               math::vec4{ 0.163279845f, 0.498675213f, 0.31924078f, 0.17983402f } };

    lPhotoDetectorComponent.mGain = { math::vec4{ 0.1234f, 1.3456f, 0.8234f, 0.56798f }, math::vec4{ 0.413275986f, 5.09487152f, 0.310279485f, 0.70198432f },
                                      math::vec4{ 0.163279845f, 0.498675213f, 0.31924078f, 0.17983402f } };

    lPhotoDetectorComponent.mBaseline = { math::vec4{ 0.91730842f, 0.10923478f, 0.13470892f, 0.13427098f }, math::vec4{ 0.13097248f, 0.19347082f, 0.30172498f, 0.7092831f },
                                          math::vec4{ 0.2137098f, 0.13624978f, 0.12347908f, 0.13246798f } };

    lPhotoDetectorComponent.mStaticNoise      = { Entity{}, Entity{}, Entity{} };
    lPhotoDetectorComponent.mStaticNoiseShift = { math::vec4{ 0.91730842f, 0.10923478f, 0.13470892f, 0.13427098f }, math::vec4{ 0.13097248f, 0.19347082f, 0.30172498f, 0.7092831f },
                                                  math::vec4{ 0.2137098f, 0.13624978f, 0.12347908f, 0.13246798f } };

    auto lPhotoDetector = lSensorModelBase.CreateElement( "PD", "apd_0", lPhotoDetectorComponent );
    lFlash0.Adjoin<sPhotoDetector>( lPhotoDetector );
    lFlash1.Adjoin<sPhotoDetector>( lPhotoDetector );
    lFlash2.Adjoin<sPhotoDetector>( lPhotoDetector );

    AcquisitionSpecification lAcqCreateInfo{};
    lAcqCreateInfo.mTemperature = 3.0f;

    float lFlashTime = EvaluatePolynomial( lLaserAssembly.mFlashTime, lAcqCreateInfo.mTemperature );

    std::vector<vec2> lTilesPositionsToSchedule = { vec2{ 0.0f, 0.0f }, vec2{ 1.0f, 1.0f }, vec2{ 2.0f, 2.0f } };
    std::vector<float> lTimings                 = { 0.0f, 1.0f, 2.0f };

    AcquisitionContext lLaserFlashList( lAcqCreateInfo, lTile1, lTilesPositionsToSchedule, lTimings );

    CHECK_PD_COLUMN_SIZE( lLaserFlashList, 27 );

    std::vector<uint32_t> lExpectedPDCellCounts( 9 );
    std::fill( lExpectedPDCellCounts.begin(), lExpectedPDCellCounts.end(), 3 );
    REQUIRE( ( lLaserFlashList.mPulseSampling.mPhotoDetectorCellCount == lExpectedPDCellCounts ) );

    sPositionArray lExpectedCellPositions0{};
    for( auto &lCellPos : lPhotoDetectorComponent.mCellPositions )
        lExpectedCellPositions0.Append( math::vec2{ lCellPos.x, lCellPos.y } );

    sPositionArray lExpectedCellPositions{};
    Repeat( lExpectedCellPositions.mX, lExpectedCellPositions0.mX, 9 );
    REQUIRE( lLaserFlashList.mPulseSampling.mPhotoDetectorData.mCellPositions.mX == lExpectedCellPositions.mX );

    Repeat( lExpectedCellPositions.mY, lExpectedCellPositions0.mY, 9 );
    REQUIRE( lLaserFlashList.mPulseSampling.mPhotoDetectorData.mCellPositions.mY == lExpectedCellPositions.mY );

    sPositionArray lExpectedCellWorldPositions0{};
    sPositionArray lExpectedCellTilePositions0{};
    for( auto &lTilePosition : lTilesPositionsToSchedule )
    {
        for( auto &lFlash : lTile1.Get<sRelationshipComponent>().mChildren )
        {
            math::vec2 lFlashPosition = lFlash.Get<sLaserFlashSpecificationComponent>().mPosition;
            for( auto &lCellPos : lPhotoDetectorComponent.mCellPositions )
            {
                lExpectedCellWorldPositions0.Append( math::vec2{ lCellPos.x, lCellPos.y } + lFlashPosition + lTilePosition );
                lExpectedCellTilePositions0.Append( math::vec2{ lCellPos.x, lCellPos.y } + lFlashPosition );
            }
        }
    }
    REQUIRE( lLaserFlashList.mPulseSampling.mPhotoDetectorData.mCellWorldPositions.mX == lExpectedCellWorldPositions0.mX );
    REQUIRE( lLaserFlashList.mPulseSampling.mPhotoDetectorData.mCellWorldPositions.mY == lExpectedCellWorldPositions0.mY );

    REQUIRE( lLaserFlashList.mPulseSampling.mPhotoDetectorData.mCellTilePositions.mX == lExpectedCellTilePositions0.mX );
    REQUIRE( lLaserFlashList.mPulseSampling.mPhotoDetectorData.mCellTilePositions.mY == lExpectedCellTilePositions0.mY );

    sSizeArray lExpectedCellSizes0{};
    for( auto &lCellSize : lPhotoDetectorComponent.mCellPositions )
        lExpectedCellSizes0.Append( math::vec2{ lCellSize.z, lCellSize.w } );

    sSizeArray lExpectedCellSizes{};
    Repeat( lExpectedCellSizes.mWidth, lExpectedCellSizes0.mWidth, 9 );
    REQUIRE( lLaserFlashList.mPulseSampling.mPhotoDetectorData.mCellSizes.mWidth == lExpectedCellSizes.mWidth );

    Repeat( lExpectedCellSizes.mHeight, lExpectedCellSizes0.mHeight, 9 );
    REQUIRE( lLaserFlashList.mPulseSampling.mPhotoDetectorData.mCellSizes.mHeight == lExpectedCellSizes.mHeight );

    sFloat32Array lExpectedBaseline0{};
    for( auto &lBaseline : lPhotoDetectorComponent.mBaseline )
        lExpectedBaseline0.Append( EvaluatePolynomial( lBaseline, lAcqCreateInfo.mTemperature ) );
    sFloat32Array lExpectedBaseline{};
    Repeat( lExpectedBaseline, lExpectedBaseline0, 9 );
    REQUIRE( VectorEqual( lLaserFlashList.mPulseSampling.mPhotoDetectorData.mBaseline, lExpectedBaseline, 0.0001f ) );
}

TEST_CASE( "Check that photodetector data has the correct length (multiple tiles)", "[CORE_SENSOR_MODEL]" )
{
    SensorModelBase lSensorModelBase{};

    Entity lTile1   = lSensorModelBase.CreateTile( "0", vec2{ 1.0f, 0.0f } );
    Entity lFlash10 = lSensorModelBase.CreateFlash( lTile1, "0", vec2{ -1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );
    Entity lFlash11 = lSensorModelBase.CreateFlash( lTile1, "1", vec2{ 1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );

    Entity lTile2   = lSensorModelBase.CreateTile( "1", vec2{ 1.0f, 0.0f } );
    Entity lFlash20 = lSensorModelBase.CreateFlash( lTile2, "0", vec2{ -1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );
    Entity lFlash21 = lSensorModelBase.CreateFlash( lTile2, "1", vec2{ 1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );
    Entity lFlash22 = lSensorModelBase.CreateFlash( lTile2, "2", vec2{ 1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );

    sLaserAssembly lLaserAssembly{};
    lLaserAssembly.mFlashTime = { 0.23245f, 0.1345f, 0.332f, 0.123f };

    auto lLaser = lSensorModelBase.CreateElement( "laser", "laser", lLaserAssembly );
    lFlash10.Adjoin<sLaserAssembly>( lLaser );
    lFlash11.Adjoin<sLaserAssembly>( lLaser );
    lFlash20.Adjoin<sLaserAssembly>( lLaser );
    lFlash21.Adjoin<sLaserAssembly>( lLaser );
    lFlash22.Adjoin<sLaserAssembly>( lLaser );

    sPhotoDetector lPhotoDetectorComponent{};
    lPhotoDetectorComponent.mCellPositions = { math::vec4{ 0.1234f, 1.3456f, 0.8234f, 0.56798f }, math::vec4{ 0.413275986f, 5.09487152f, 0.310279485f, 0.70198432f },
                                               math::vec4{ 0.163279845f, 0.498675213f, 0.31924078f, 0.17983402f } };

    lPhotoDetectorComponent.mGain = { math::vec4{ 0.1234f, 1.3456f, 0.8234f, 0.56798f }, math::vec4{ 0.413275986f, 5.09487152f, 0.310279485f, 0.70198432f },
                                      math::vec4{ 0.163279845f, 0.498675213f, 0.31924078f, 0.17983402f } };

    lPhotoDetectorComponent.mBaseline = { math::vec4{ 0.91730842f, 0.10923478f, 0.13470892f, 0.13427098f }, math::vec4{ 0.13097248f, 0.19347082f, 0.30172498f, 0.7092831f },
                                          math::vec4{ 0.2137098f, 0.13624978f, 0.12347908f, 0.13246798f } };

    lPhotoDetectorComponent.mStaticNoise      = { Entity{}, Entity{}, Entity{} };
    lPhotoDetectorComponent.mStaticNoiseShift = { math::vec4{ 0.91730842f, 0.10923478f, 0.13470892f, 0.13427098f }, math::vec4{ 0.13097248f, 0.19347082f, 0.30172498f, 0.7092831f },
                                                  math::vec4{ 0.2137098f, 0.13624978f, 0.12347908f, 0.13246798f } };

    auto lPhotoDetector = lSensorModelBase.CreateElement( "PD", "apd_0", lPhotoDetectorComponent );
    lFlash10.Adjoin<sPhotoDetector>( lPhotoDetector );
    lFlash11.Adjoin<sPhotoDetector>( lPhotoDetector );
    lFlash20.Adjoin<sPhotoDetector>( lPhotoDetector );
    lFlash21.Adjoin<sPhotoDetector>( lPhotoDetector );
    lFlash22.Adjoin<sPhotoDetector>( lPhotoDetector );

    AcquisitionSpecification lAcqCreateInfo{};
    lAcqCreateInfo.mTemperature = 3.0f;

    float lFlashTime = EvaluatePolynomial( lLaserAssembly.mFlashTime, lAcqCreateInfo.mTemperature );

    std::vector<Entity> lTilesToSchedule        = { lTile1, lTile1, lTile2 };
    std::vector<vec2> lTilesPositionsToSchedule = { vec2{ 0.0f, 0.0f }, vec2{ 1.0f, 1.0f }, vec2{ 2.0f, 2.0f } };
    std::vector<float> lTimings                 = { 0.0f, 1.0f, 2.0f };

    AcquisitionContext lLaserFlashList( lAcqCreateInfo, lTilesToSchedule, lTilesPositionsToSchedule, lTimings );

    CHECK_PD_COLUMN_SIZE( lLaserFlashList, 21 );

    std::vector<uint32_t> lExpectedPDCellCounts( 7 );
    std::fill( lExpectedPDCellCounts.begin(), lExpectedPDCellCounts.end(), 3 );
    REQUIRE( ( lLaserFlashList.mPulseSampling.mPhotoDetectorCellCount == lExpectedPDCellCounts ) );

    sPositionArray lExpectedCellPositions0{};
    for( auto &lCellPos : lPhotoDetectorComponent.mCellPositions )
        lExpectedCellPositions0.Append( math::vec2{ lCellPos.x, lCellPos.y } );

    sPositionArray lExpectedCellPositions{};
    Repeat( lExpectedCellPositions.mX, lExpectedCellPositions0.mX, 7 );
    REQUIRE( lLaserFlashList.mPulseSampling.mPhotoDetectorData.mCellPositions.mX == lExpectedCellPositions.mX );

    Repeat( lExpectedCellPositions.mY, lExpectedCellPositions0.mY, 7 );
    REQUIRE( lLaserFlashList.mPulseSampling.mPhotoDetectorData.mCellPositions.mY == lExpectedCellPositions.mY );

    sPositionArray lExpectedCellWorldPositions0{};
    sPositionArray lExpectedCellTilePositions0{};

    for( uint32_t i = 0; i < lTilesToSchedule.size(); i++ )
    {
        for( auto &lFlash : lTilesToSchedule[i].Get<sRelationshipComponent>().mChildren )
        {
            math::vec2 lTilePosition  = lTilesPositionsToSchedule[i];
            math::vec2 lFlashPosition = lFlash.Get<sLaserFlashSpecificationComponent>().mPosition;
            for( auto &lCellPos : lPhotoDetectorComponent.mCellPositions )
            {
                lExpectedCellWorldPositions0.Append( math::vec2{ lCellPos.x, lCellPos.y } + lFlashPosition + lTilePosition );
                lExpectedCellTilePositions0.Append( math::vec2{ lCellPos.x, lCellPos.y } + lFlashPosition );
            }
        }
    }
    REQUIRE( lLaserFlashList.mPulseSampling.mPhotoDetectorData.mCellWorldPositions.mX == lExpectedCellWorldPositions0.mX );
    REQUIRE( lLaserFlashList.mPulseSampling.mPhotoDetectorData.mCellWorldPositions.mY == lExpectedCellWorldPositions0.mY );

    REQUIRE( lLaserFlashList.mPulseSampling.mPhotoDetectorData.mCellTilePositions.mX == lExpectedCellTilePositions0.mX );
    REQUIRE( lLaserFlashList.mPulseSampling.mPhotoDetectorData.mCellTilePositions.mY == lExpectedCellTilePositions0.mY );

    sSizeArray lExpectedCellSizes0{};
    for( auto &lCellSize : lPhotoDetectorComponent.mCellPositions )
        lExpectedCellSizes0.Append( math::vec2{ lCellSize.z, lCellSize.w } );

    sSizeArray lExpectedCellSizes{};
    Repeat( lExpectedCellSizes.mWidth, lExpectedCellSizes0.mWidth, 7 );
    REQUIRE( lLaserFlashList.mPulseSampling.mPhotoDetectorData.mCellSizes.mWidth == lExpectedCellSizes.mWidth );

    Repeat( lExpectedCellSizes.mHeight, lExpectedCellSizes0.mHeight, 7 );
    REQUIRE( lLaserFlashList.mPulseSampling.mPhotoDetectorData.mCellSizes.mHeight == lExpectedCellSizes.mHeight );

    sFloat32Array lExpectedBaseline0{};
    for( auto &lBaseline : lPhotoDetectorComponent.mBaseline )
        lExpectedBaseline0.Append( EvaluatePolynomial( lBaseline, lAcqCreateInfo.mTemperature ) );
    sFloat32Array lExpectedBaseline{};
    Repeat( lExpectedBaseline, lExpectedBaseline0, 7 );
    REQUIRE( VectorEqual( lLaserFlashList.mPulseSampling.mPhotoDetectorData.mBaseline, lExpectedBaseline, 0.0001f ) );
}
