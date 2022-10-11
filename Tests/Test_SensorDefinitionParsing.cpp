#include <catch2/catch_test_macros.hpp>
#include <filesystem>

#include "yaml-cpp/yaml.h"

#include "TestUtils.h"

#include "Serialize/FileIO.h"
#include "Serialize/SensorAsset.h"
#include "Serialize/SensorComponents.h"
#include "Serialize/SensorDefinition.h"

using namespace math;
using namespace TestUtils;
using namespace LTSE::SensorModel;

namespace fs = std::filesystem;

TEST_CASE( "Parse sensor from file", "[CORE_SENSOR_MODEL]" )
{
    sSensorDefinition lSensorDefinition = ReadSensorDefinition( "Tests\\Data\\TestSensor0", "TestSensor0.yaml" );

    REQUIRE( lSensorDefinition.mName == "Test Sensor Model" );
    REQUIRE( lSensorDefinition.Assets.size() == 2 );
    REQUIRE( lSensorDefinition.Assets.find( "laser_diode_1" ) != lSensorDefinition.Assets.end() );
    REQUIRE( lSensorDefinition.Assets.find( "photodetector_array_1" ) != lSensorDefinition.Assets.end() );

    REQUIRE( lSensorDefinition.Components.size() == 2 );
    REQUIRE( lSensorDefinition.Components.find( "sampler_1" ) != lSensorDefinition.Components.end() );
    REQUIRE( lSensorDefinition.Components.find( "sampler_2" ) != lSensorDefinition.Components.end() );

    REQUIRE( lSensorDefinition.Tiles.size() == 2 );
    REQUIRE( lSensorDefinition.Tiles.find( "tile_0" ) != lSensorDefinition.Tiles.end() );
    REQUIRE( lSensorDefinition.Tiles.find( "tile_1" ) != lSensorDefinition.Tiles.end() );

    {
        auto &lTile = lSensorDefinition.Tiles["tile_0"];
        REQUIRE( lTile.Flashes.size() == 1 );
        REQUIRE( lTile.mID == "tile_0" );
        REQUIRE( lTile.FieldOfView == math::vec2{ 6.5f, 7.5f } );
        REQUIRE( lTile.mPosition == math::vec2{ 0.0f, 0.0f } );
        REQUIRE( lTile.SamplerComponentID == "sampler_1" );

        auto &lFlash = lTile.Flashes[0];
        REQUIRE( lFlash.Area == math::vec4{ -16.0f, 0.0f, 12.0f, 34.0f } );
        REQUIRE( lFlash.LaserDiodeComponentID == "laser_diode_1" );
        REQUIRE( lFlash.PhotodetectorComponentID == "photodetector_array_1" );
    }

    {
        auto &lTile = lSensorDefinition.Tiles["tile_1"];
        REQUIRE( lTile.Flashes.size() == 2 );
        REQUIRE( lTile.mID == "tile_1" );
        REQUIRE( lTile.FieldOfView == math::vec2{ 8.5f, 9.5f } );
        REQUIRE( lTile.mPosition == math::vec2{ 10.0f, 10.0f } );
        REQUIRE( lTile.SamplerComponentID == "sampler_2" );

        {
            auto &lFlash = lTile.Flashes[0];
            REQUIRE( lFlash.Area == math::vec4{ -16.0f, 0.0f, 2.5f, 16.0f } );
            REQUIRE( lFlash.LaserDiodeComponentID == "laser_diode_2" );
            REQUIRE( lFlash.PhotodetectorComponentID == "photodetector_array_2" );
        }

        {
            auto &lFlash = lTile.Flashes[1];
            REQUIRE( lFlash.Area == math::vec4{ -15.0f, 0.0f, 2.5f, 16.0f } );
            REQUIRE( lFlash.LaserDiodeComponentID == "laser_diode_3" );
            REQUIRE( lFlash.PhotodetectorComponentID == "photodetector_array_3" );
        }
    }

    REQUIRE( lSensorDefinition.Layouts.size() == 2 );
    REQUIRE( lSensorDefinition.Layouts.find( "tile_layout_0" ) != lSensorDefinition.Layouts.end() );
    {
        auto &lTileLayout = lSensorDefinition.Layouts["tile_layout_0"];
        REQUIRE( lTileLayout.mID == "tile_layout_0" );
        REQUIRE( lTileLayout.Elements.size() == 1 );
        REQUIRE( lTileLayout.Elements.find( "tile_0" ) != lTileLayout.Elements.end() );
        REQUIRE( lTileLayout.Elements["tile_0"].mTileID == "tile_1" );
        REQUIRE( lTileLayout.Elements["tile_0"].mPosition == math::vec2{ 0.0f, 0.0f } );
    }

    REQUIRE( lSensorDefinition.Layouts.find( "tile_layout_1" ) != lSensorDefinition.Layouts.end() );
    {
        auto &lTileLayout = lSensorDefinition.Layouts["tile_layout_1"];
        REQUIRE( lTileLayout.mID == "tile_layout_1" );
        REQUIRE( lTileLayout.Elements.size() == 3 );

        REQUIRE( lTileLayout.Elements.find( "tile_0" ) != lTileLayout.Elements.end() );
        REQUIRE( lTileLayout.Elements["tile_0"].mTileID == "tile_0" );
        REQUIRE( lTileLayout.Elements["tile_0"].mPosition == math::vec2{ -1.0f, -3.0f } );

        REQUIRE( lTileLayout.Elements.find( "tile_1" ) != lTileLayout.Elements.end() );
        REQUIRE( lTileLayout.Elements["tile_1"].mTileID == "tile_0" );
        REQUIRE( lTileLayout.Elements["tile_1"].mPosition == math::vec2{ 0.0f, -2.0f } );

        REQUIRE( lTileLayout.Elements.find( "tile_2" ) != lTileLayout.Elements.end() );
        REQUIRE( lTileLayout.Elements["tile_2"].mTileID == "tile_1" );
        REQUIRE( lTileLayout.Elements["tile_2"].mPosition == math::vec2{ 1.0f, 1.0f } );
    }
}

TEST_CASE( "SaveSensorDefinition sensor", "[CORE_SENSOR_MODEL]" )
{
    sSensorDefinition lSensorDefinition0 = ReadSensorDefinition( "Tests\\Data\\TestSensor0", "TestSensor0.yaml" );
    std::string lSavedVersion0           = ToString( lSensorDefinition0 );

    sSensorDefinition lSensorDefinition1 = ReadSensorDefinitionFromString( "Tests\\Data\\TestSensor0", lSavedVersion0 );
    std::string lSavedVersion1           = ToString( lSensorDefinition1 );

    REQUIRE( lSavedVersion0 == lSavedVersion1 );
}