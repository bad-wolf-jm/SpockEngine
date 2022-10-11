#include <catch2/catch_test_macros.hpp>

#include "LidarSensorModel/Components.h"
#include "LidarSensorModel/ModelBuilder.h"
#include "LidarSensorModel/SensorModelBase.h"

#include "TestUtils.h"

using namespace math;
using namespace TestUtils;
using namespace LTSE::Core;
using namespace LTSE::SensorModel;

TEST_CASE( "Create laser element", "[CORE_SENSOR_MODEL]" )
{
    SensorModelBase lSensorModelBase{};

    sLaserAssembly lElementSpecification{};
    lElementSpecification.mWaveformData  = Entity{};
    lElementSpecification.mTimebaseDelay = math::vec4{ 150.10384283916842f, 0.0f, 0.0f, 0.0f };
    lElementSpecification.mFlashTime     = math::vec4{ 0.1f, 0.0f, 0.0f, 0.0f };

    auto lSensorElementEntity = lSensorModelBase.CreateElement( "Laser", "laser_0", lElementSpecification );

    REQUIRE( lSensorElementEntity.HasAll<sSensorComponent, sLaserAssembly>() );

    REQUIRE( lSensorElementEntity.Get<sSensorComponent>().mID == "laser_0" );
    REQUIRE( lSensorElementEntity.Get<sLaserAssembly>().mTimebaseDelay == lElementSpecification.mTimebaseDelay );
    REQUIRE( lSensorElementEntity.Get<sLaserAssembly>().mFlashTime == lElementSpecification.mFlashTime );
}

TEST_CASE( "Create photodetector element", "[CORE_SENSOR_MODEL]" )
{
    SensorModelBase lSensorModelBase{};

    std::vector<math::vec4> lCellPositions( 32 );
    std::vector<math::vec4> lBaseline( 32 );
    std::vector<Entity> lStaticNoise( 32 );
    for( uint32_t i = 0; i < lCellPositions.size(); i++ )
    {
        lCellPositions[i] = math::vec4{ 0.0f, i * 0.2 - ( 32.0f * 0.2f ) / 2.0f, 0.2f, 0.2f };
        lBaseline[i]      = math::vec4{ 0.0f };
        lStaticNoise[i]   = Entity{};
    }
    sPhotoDetector lElementSpecification{};
    lElementSpecification.mCellPositions = lCellPositions;
    lElementSpecification.mStaticNoise   = lStaticNoise;
    lElementSpecification.mBaseline      = lBaseline;

    auto lSensorElementEntity = lSensorModelBase.CreateElement( "PD", "apd_0", lElementSpecification );

    REQUIRE( lSensorElementEntity.HasAll<sSensorComponent, sPhotoDetector>() );

    REQUIRE( lSensorElementEntity.Get<sSensorComponent>().mID == "apd_0" );
    REQUIRE( lSensorElementEntity.Get<sPhotoDetector>().mCellPositions == lElementSpecification.mCellPositions );
    REQUIRE( lSensorElementEntity.Get<sPhotoDetector>().mBaseline == lElementSpecification.mBaseline );
    REQUIRE( lSensorElementEntity.Get<sPhotoDetector>().mStaticNoise == lElementSpecification.mStaticNoise );
}

TEST_CASE( "Create sampler element", "[CORE_SENSOR_MODEL]" )
{
    SensorModelBase lSensorModelBase{};

    sSampler lElementSpecification{};
    lElementSpecification.mLength    = 1234;
    lElementSpecification.mFrequency = 5678.0f;

    auto lSensorElementEntity = lSensorModelBase.CreateElement( "Sampler", "sampler_0", lElementSpecification );

    REQUIRE( lSensorElementEntity.HasAll<sSensorComponent, sSampler>() );

    REQUIRE( lSensorElementEntity.Get<sSensorComponent>().mID == "sampler_0" );
    REQUIRE( lSensorElementEntity.Get<sSampler>().mLength == lElementSpecification.mLength );
    REQUIRE( lSensorElementEntity.Get<sSampler>().mFrequency == lElementSpecification.mFrequency );
}

TEST_CASE( "Create tile layouts", "[CORE_SENSOR_MODEL]" )
{
    SensorModelBase lSensorModelBase{};
    Entity lTile0 = lSensorModelBase.CreateTile( "0", vec2{ 1.0f, 1.0f } );
    Entity lTile1 = lSensorModelBase.CreateTile( "1", vec2{ 1.0f, 0.0f } );
    Entity lTile2 = lSensorModelBase.CreateTile( "2", vec2{ 1.0f, 0.0f } );

    sTileLayoutComponent lTileLayoutComponent{};
    lTileLayoutComponent.mID          = "0";
    lTileLayoutComponent.mLayout["a"] = { "0", math::vec2{ 1.234f, 4.567f } };
    lTileLayoutComponent.mLayout["b"] = { "1", math::vec2{ 2.234f, 5.567f } };
    lTileLayoutComponent.mLayout["c"] = { "2", math::vec2{ 3.234f, 6.567f } };

    auto lTileLayout = lSensorModelBase.CreateTileLayout( "tile_layout_name", lTileLayoutComponent );

    REQUIRE( lSensorModelBase.mRootLayout.Get<sRelationshipComponent>().mChildren.size() == 1 );

    REQUIRE( lTileLayout.HasAll<sTag, sTileLayoutComponent>() );
    REQUIRE( lTileLayout.Get<sTag>().mValue == "tile_layout_name" );
    REQUIRE( lTileLayout.Get<sTileLayoutComponent>().mID == "0" );
    REQUIRE( lTileLayout.Get<sTileLayoutComponent>().mLayout.size() == 3 );
    REQUIRE( lTileLayout.Get<sTileLayoutComponent>().mLayout.find( "a" ) != lTileLayout.Get<sTileLayoutComponent>().mLayout.end() );
    REQUIRE( lTileLayout.Get<sTileLayoutComponent>().mLayout.find( "b" ) != lTileLayout.Get<sTileLayoutComponent>().mLayout.end() );
    REQUIRE( lTileLayout.Get<sTileLayoutComponent>().mLayout.find( "c" ) != lTileLayout.Get<sTileLayoutComponent>().mLayout.end() );

    REQUIRE( ( lTileLayout.Get<sTileLayoutComponent>().mLayout["a"] == sTileLayoutComponent::TileData{ "0", math::vec2{ 1.234f, 4.567f } } ) );
    REQUIRE( ( lTileLayout.Get<sTileLayoutComponent>().mLayout["b"] == sTileLayoutComponent::TileData{ "1", math::vec2{ 2.234f, 5.567f } } ) );
    REQUIRE( ( lTileLayout.Get<sTileLayoutComponent>().mLayout["c"] == sTileLayoutComponent::TileData{ "2", math::vec2{ 3.234f, 6.567f } } ) );
}

TEST_CASE( "Build sensor", "[CORE_SENSOR_MODEL]" )
{
    Ref<SensorModelBase> lSensorModel = Build<SensorModelBase>( "Tests\\Data\\TestSensor0", "TestSensor1.yaml" );

    uint32_t lAssetCount = 0;
    lSensorModel->ForEach<sAssetMetadata>( [&]( auto e, auto &c ) { lAssetCount++; } );
    REQUIRE( lAssetCount == 2 );

    uint32_t lElementCount = 0;
    lSensorModel->ForEach<sSensorComponent>( [&]( auto e, auto &c ) { lElementCount++; } );
    REQUIRE( lElementCount == 3 );

    uint32_t lTileCount = 0;
    lSensorModel->ForEach<sTileSpecificationComponent>( [&]( auto e, auto &c ) { lTileCount++; } );
    REQUIRE( lTileCount == 2 );

    uint32_t lFlashCount = 0;
    lSensorModel->ForEach<sLaserFlashSpecificationComponent>( [&]( auto e, auto &c ) { lFlashCount++; } );
    REQUIRE( lFlashCount == 3 );

    uint32_t lTileLayoutCount = 0;
    lSensorModel->ForEach<sTileLayoutComponent>( [&]( auto e, auto &c ) { lTileLayoutCount++; } );
    REQUIRE( lTileLayoutCount == 2 );

    auto lTile0 = lSensorModel->GetTileByID( "tile_0" );
    REQUIRE( lTile0.Has<sRelationshipComponent>() );
    REQUIRE( lTile0.Get<sRelationshipComponent>().mChildren.size() == 1 );
    auto lFlash0 = lTile0.Get<sRelationshipComponent>().mChildren[0];
    REQUIRE( lFlash0.HasAll<sJoinComponent<sLaserAssembly>, sJoinComponent<sPhotoDetector>>() );

    auto lLaserAssembly0 = lFlash0.Get<sJoinComponent<sLaserAssembly>>();
    REQUIRE( ( lLaserAssembly0.JoinedComponent().mWaveformData ) );
    REQUIRE( ( lLaserAssembly0.JoinedComponent().mTimebaseDelay == math::vec4{ 1.0f, 2.0f, 3.0f, 4.0f } ) );
    REQUIRE( ( lLaserAssembly0.JoinedComponent().mFlashTime == math::vec4{ 5.0f, 6.0f, 7.0f, 8.0f } ) );
    REQUIRE( ( lLaserAssembly0.mJoinEntity.Has<sSensorComponent>() ) );
    REQUIRE( ( lLaserAssembly0.mJoinEntity.Get<sSensorComponent>().mID == "laser_diode_1" ) );

    auto lPhotoDetector0 = lFlash0.Get<sJoinComponent<sPhotoDetector>>();
    REQUIRE( ( lPhotoDetector0.JoinedComponent().mCellPositions.size() == 3 ) );
    REQUIRE( ( lPhotoDetector0.JoinedComponent().mBaseline.size() == 3 ) );
    REQUIRE( ( lPhotoDetector0.JoinedComponent().mStaticNoise.size() == 3 ) );
    REQUIRE( ( lPhotoDetector0.mJoinEntity.Has<sSensorComponent>() ) );
    REQUIRE( ( lPhotoDetector0.mJoinEntity.Get<sSensorComponent>().mID == "photodetector_array_1" ) );

    auto lTile1 = lSensorModel->GetTileByID( "tile_1" );
    REQUIRE( lTile1.Has<sRelationshipComponent>() );
    REQUIRE( lTile1.Get<sRelationshipComponent>().mChildren.size() == 2 );
    auto lFlash1 = lTile1.Get<sRelationshipComponent>().mChildren[0];
    REQUIRE( lFlash1.HasAll<sJoinComponent<sLaserAssembly>, sJoinComponent<sPhotoDetector>>() );

    auto lLaserAssembly1 = lFlash1.Get<sJoinComponent<sLaserAssembly>>();
    REQUIRE( ( lLaserAssembly1.JoinedComponent().mWaveformData ) );
    REQUIRE( ( lLaserAssembly1.JoinedComponent().mTimebaseDelay == math::vec4{ 1.0f, 2.0f, 3.0f, 4.0f } ) );
    REQUIRE( ( lLaserAssembly1.JoinedComponent().mFlashTime == math::vec4{ 5.0f, 6.0f, 7.0f, 8.0f } ) );
    REQUIRE( ( lLaserAssembly1.mJoinEntity.Has<sSensorComponent>() ) );
    REQUIRE( ( lLaserAssembly1.mJoinEntity.Get<sSensorComponent>().mID == "laser_diode_1" ) );

    auto lPhotoDetector1 = lFlash1.Get<sJoinComponent<sPhotoDetector>>();
    REQUIRE( ( lPhotoDetector1.JoinedComponent().mCellPositions.size() == 3 ) );
    REQUIRE( ( lPhotoDetector1.JoinedComponent().mBaseline.size() == 3 ) );
    REQUIRE( ( lPhotoDetector1.JoinedComponent().mStaticNoise.size() == 3 ) );
    REQUIRE( ( lPhotoDetector1.mJoinEntity.Has<sSensorComponent>() ) );
    REQUIRE( ( lPhotoDetector1.mJoinEntity.Get<sSensorComponent>().mID == "photodetector_array_1" ) );

    auto lFlash2 = lTile1.Get<sRelationshipComponent>().mChildren[1];
    REQUIRE( lFlash2.HasAll<sJoinComponent<sLaserAssembly>, sJoinComponent<sPhotoDetector>>() );

    auto lLaserAssembly2 = lFlash2.Get<sJoinComponent<sLaserAssembly>>();
    REQUIRE( ( lLaserAssembly2.JoinedComponent().mWaveformData ) );
    REQUIRE( ( lLaserAssembly2.JoinedComponent().mTimebaseDelay == math::vec4{ 1.0f, 2.0f, 3.0f, 4.0f } ) );
    REQUIRE( ( lLaserAssembly2.JoinedComponent().mFlashTime == math::vec4{ 5.0f, 6.0f, 7.0f, 8.0f } ) );
    REQUIRE( ( lLaserAssembly2.mJoinEntity.Has<sSensorComponent>() ) );
    REQUIRE( ( lLaserAssembly2.mJoinEntity.Get<sSensorComponent>().mID == "laser_diode_1" ) );

    auto lPhotoDetector2 = lFlash2.Get<sJoinComponent<sPhotoDetector>>();
    REQUIRE( ( lPhotoDetector2.JoinedComponent().mCellPositions.size() == 3 ) );
    REQUIRE( ( lPhotoDetector2.JoinedComponent().mBaseline.size() == 3 ) );
    REQUIRE( ( lPhotoDetector2.JoinedComponent().mStaticNoise.size() == 3 ) );
    REQUIRE( ( lPhotoDetector2.mJoinEntity.Has<sSensorComponent>() ) );
    REQUIRE( ( lPhotoDetector2.mJoinEntity.Get<sSensorComponent>().mID == "photodetector_array_1" ) );
}
