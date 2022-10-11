#include <catch2/catch_test_macros.hpp>

#include "LidarSensorModel/Components.h"
#include "LidarSensorModel/SensorModelBase.h"

#include "TestUtils.h"

using namespace math;
using namespace TestUtils;
using namespace LTSE::Core;
using namespace LTSE::SensorModel;

TEST_CASE( "Create sensor", "[CORE_SENSOR_MODEL]" )
{
    SensorModelBase lSensorModelBase{};

    REQUIRE( true );
}

TEST_CASE( "Create tiles", "[CORE_SENSOR_MODEL]" )
{
    SensorModelBase lSensorModelBase{};

    SECTION( "Uniqueness of tile IDs" )
    {
        Entity lTile = lSensorModelBase.CreateTile( "0", vec2{ 0.0f, 1.0f } );
        REQUIRE_THROWS( lSensorModelBase.CreateTile( "0", vec2{ 0.0f, 1.0f } ) );
    }

    SECTION( "Tile with position" )
    {
        Entity lTile0 = lSensorModelBase.CreateTile( "1", vec2{ 1.0f, 1.0f } );
        Entity lTile1 = lSensorModelBase.CreateTile( "2", vec2{ 1.0f, 0.0f } );

        if( lTile0.Has<sTileSpecificationComponent>() )
        {
            auto &lTileSpecification = lTile0.Get<sTileSpecificationComponent>();
            REQUIRE( lTileSpecification.mID == "1" );
            REQUIRE( lTileSpecification.mPosition == vec2{ 1.0f, 1.0f } );
        }

        if( lTile1.Has<sTileSpecificationComponent>() )
        {
            auto &lTileSpecification = lTile1.Get<sTileSpecificationComponent>();
            REQUIRE( lTileSpecification.mID == "2" );
            REQUIRE( lTileSpecification.mPosition == vec2{ 1.0f, 0.0f } );
        }
    }

    SECTION( "Get tile by ID" )
    {
        Entity lTile0 = lSensorModelBase.CreateTile( "3", vec2{ 1.0f, 1.0f } );
        Entity lTile1 = lSensorModelBase.CreateTile( "4", vec2{ 1.0f, 0.0f } );

        REQUIRE( lSensorModelBase.GetTileByID( "3" ) == lTile0 );
        REQUIRE( lSensorModelBase.GetTileByID( "4" ) == lTile1 );
    }

    SECTION( "Tile count" )
    {
        Entity lTile0 = lSensorModelBase.CreateTile( "3", vec2{ 1.0f, 1.0f } );
        Entity lTile1 = lSensorModelBase.CreateTile( "4", vec2{ 1.0f, 0.0f } );

        REQUIRE( lSensorModelBase.GetAllTiles().size() == 2 );
    }
}

TEST_CASE( "Create flashes", "[CORE_SENSOR_MODEL]" )
{
    SensorModelBase lSensorModelBase{};

    SECTION( "Basic flash creation" )
    {
        Entity lTile  = lSensorModelBase.CreateTile( "0", vec2{ 1.0f, 1.0f } );
        Entity lFlash = lSensorModelBase.CreateFlash( lTile, "0", vec2{ -1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );

        REQUIRE( lFlash.Has<sRelationshipComponent>() );
        REQUIRE( lFlash.Has<sLaserFlashSpecificationComponent>() );

        if( lFlash.Has<sRelationshipComponent>() )
        {
            REQUIRE( lFlash.Get<sRelationshipComponent>().mParent == lTile );
        }

        if( lFlash.Has<sLaserFlashSpecificationComponent>() )
        {
            auto &lLaserFlashSpecificationComponent = lFlash.Get<sLaserFlashSpecificationComponent>();

            REQUIRE( lLaserFlashSpecificationComponent.mFlashID == "0" );
            REQUIRE( lLaserFlashSpecificationComponent.mPosition == vec2{ -1.0f, 0.0f } );
            REQUIRE( lLaserFlashSpecificationComponent.mExtent == vec2{ 10.0f, 10.0f } );
        }
    }

    SECTION( "Uniqueness of flash IDs" )
    {
        Entity lTile  = lSensorModelBase.CreateTile( "0", vec2{ 1.0f, 1.0f } );
        Entity lFlash = lSensorModelBase.CreateFlash( lTile, "0", vec2{ -1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );
        // REQUIRE_THROWS(lSensorModelBase.CreateFlash( lTile, "0", vec2{ -1.0f, 0.0f }, vec2{ 10.0f, 10.0f } ));
    }

    SECTION( "Flash/Tile parent relationship" )
    {
        Entity lTile0   = lSensorModelBase.CreateTile( "0", vec2{ 1.0f, 1.0f } );
        Entity lFlash00 = lSensorModelBase.CreateFlash( lTile0, "0", vec2{ -1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );
        Entity lFlash01 = lSensorModelBase.CreateFlash( lTile0, "1", vec2{ 1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );
        REQUIRE( lTile0.Has<sRelationshipComponent>() );

        if( lTile0.Has<sRelationshipComponent>() )
        {
            REQUIRE( lTile0.Get<sRelationshipComponent>().mChildren.size() == 2 );
        }

        Entity lTile1   = lSensorModelBase.CreateTile( "1", vec2{ 1.0f, 0.0f } );
        Entity lFlash10 = lSensorModelBase.CreateFlash( lTile1, "0", vec2{ -1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );
        Entity lFlash11 = lSensorModelBase.CreateFlash( lTile1, "1", vec2{ 1.0f, 0.0f }, vec2{ 10.0f, 10.0f } );
        Entity lFlash12 = lSensorModelBase.CreateFlash( lTile1, "2", vec2{ 2.0f, 0.0f }, vec2{ 10.0f, 10.0f } );

        if( lTile1.Has<sRelationshipComponent>() )
        {
            REQUIRE( lTile1.Get<sRelationshipComponent>().mChildren.size() == 3 );
        }
    }
}
