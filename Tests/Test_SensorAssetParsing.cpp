#include <catch2/catch_test_macros.hpp>
#include <filesystem>

#include "TestUtils.h"

#include "Serialize/FileIO.h"
#include "Serialize/SensorAsset.h"
#include "Serialize/SensorComponents.h"

using namespace math;
using namespace TestUtils;
using namespace LTSE::SensorModel;

namespace fs = std::filesystem;

TEST_CASE( "Parse photodetector asset", "[CORE_SENSOR_MODEL]" )
{
    sSensorAssetData lSensorAsset = ReadAsset( "Tests\\Data\\TestSensor0", "photodetector/component.yaml", "pulse_asset_name" );

    REQUIRE( lSensorAsset.Type() == eAssetType::PHOTODETECTOR );
    REQUIRE_NOTHROW( ( lSensorAsset.Get<sPhotodetectorAssetData>() ) );

    auto &lPhotodetectorAsset = lSensorAsset.Get<sPhotodetectorAssetData>();

    REQUIRE( lPhotodetectorAsset.mCells.size() == 3 );
    // clang-format off
    REQUIRE( lPhotodetectorAsset.mCells[0].mId == 0 );
    REQUIRE( lPhotodetectorAsset.mCells[0].mPosition         == math::vec4{ 0.1f, 0.4f, 0.7f, 0.1f } );
    REQUIRE( lPhotodetectorAsset.mCells[0].mGain             == math::vec4{ 0.1f, 0.2f, 0.3f, 0.4f } );
    REQUIRE( lPhotodetectorAsset.mCells[0].mBaseline         == math::vec4{ 0.1f, 0.2f, 0.3f, 0.4f } );
    REQUIRE( lPhotodetectorAsset.mCells[0].mStaticNoiseShift == math::vec4{ 0.11f, 0.21f, 0.31f, 0.41f } );

    REQUIRE( lPhotodetectorAsset.mCells[1].mId == 1 );
    REQUIRE( lPhotodetectorAsset.mCells[1].mPosition         == math::vec4{ 0.2f, 0.5f, 0.8f, 0.11f } );
    REQUIRE( lPhotodetectorAsset.mCells[1].mGain             == math::vec4{ 0.5f, 0.6f, 0.7f, 0.8f } );
    REQUIRE( lPhotodetectorAsset.mCells[1].mBaseline         == math::vec4{ 0.5f, 0.6f, 0.7f, 0.8f } );
    REQUIRE( lPhotodetectorAsset.mCells[1].mStaticNoiseShift == math::vec4{ 0.51f, 0.61f, 0.71f, 0.81f } );

    REQUIRE( lPhotodetectorAsset.mCells[2].mId == 2 );
    REQUIRE( lPhotodetectorAsset.mCells[2].mPosition         == math::vec4{ 0.3f, 0.6f, 0.9f, 0.12f } );
    REQUIRE( lPhotodetectorAsset.mCells[2].mGain             == math::vec4{ 0.9f, 0.1f, 0.11f, 0.12f } );
    REQUIRE( lPhotodetectorAsset.mCells[2].mBaseline         == math::vec4{ 0.9f, 0.1f, 0.11f, 0.12f } );
    REQUIRE( lPhotodetectorAsset.mCells[2].mStaticNoiseShift == math::vec4{ 0.91f, 0.101f, 0.111f, 0.121f } );
    // clang-format on
}

TEST_CASE( "Parse laser asset", "[CORE_SENSOR_MODEL]" )
{
    sSensorAssetData lSensorAsset = ReadAsset( "Tests\\Data\\TestSensor0", "laser/asset.yaml", "laser_asset_name" );

    REQUIRE( lSensorAsset.Type() == eAssetType::LASER_ASSEMBLY );
    REQUIRE_NOTHROW( ( lSensorAsset.Get<sLaserAssetData>() ) );

    auto &lLaserAsset = lSensorAsset.Get<sLaserAssetData>();

    // clang-format off
    REQUIRE( lLaserAsset.mWaveformTemplate == "laser/waveform_template.lta" );
    REQUIRE( lLaserAsset.mDiffuser         == "laser/diffuser_data.lta" );
    REQUIRE( lLaserAsset.mTimebaseDelay    == math::vec4{ 1.0, 2.0, 3.0, 4.0 } );
    REQUIRE( lLaserAsset.mFlashTime        == math::vec4{ 5.0f, 6.0f, 7.0f, 8.0f } );
    // clang-format on
}
