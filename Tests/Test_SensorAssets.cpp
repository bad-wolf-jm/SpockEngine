#include <catch2/catch_test_macros.hpp>

#include "LidarSensorModel/Components.h"
#include "LidarSensorModel/EditorComponents.h"
#include "LidarSensorModel/LaserFlashList.h"
#include "LidarSensorModel/SensorModelBase.h"
#include "LidarSensorModel/SensorDeviceBase.h"

#include "AssetManager/ConfigurationFile.h"

#include "yaml-cpp/yaml.h"

#include "TestUtils.h"

using namespace math;
using namespace TestUtils;
using namespace LTSE::Core;
using namespace LTSE::SensorModel;

TEST_CASE( "Add attenuation asset from file", "[CORE_SENSOR_MODEL]" )
{
    SensorModelBase lSensorModelBase{};

    auto lNewAsset = lSensorModelBase.AddAsset( "attenuation_test_0", "C:\\GitLab\\LTSimulationEngine\\Tests\\Data\\TestSensor0", "attenuation/asset.yaml", "attenuation_asset_name" );

    REQUIRE( lSensorModelBase.mRootAsset.Get<sRelationshipComponent>().mChildren.size() == 1 );
    REQUIRE( lSensorModelBase.AssetsByID["attenuation_test_0"] == lNewAsset );
    REQUIRE( lSensorModelBase.IdsByAsset[lNewAsset] == "attenuation_test_0" );
    REQUIRE( lNewAsset.Has<sAssetMetadata>() );
    REQUIRE( lNewAsset.Has<sAttenuationAsset>() );
    REQUIRE( lNewAsset.Get<sRelationshipComponent>().mChildren.size() == 1 );
}

TEST_CASE( "Add reduction asset", "[CORE_SENSOR_MODEL]" )
{
    SensorModelBase lSensorModelBase{};

    auto lNewAsset = lSensorModelBase.AddAsset( "reduction_test_0", "C:\\GitLab\\LTSimulationEngine\\Tests\\Data\\TestSensor0", "reduction/asset.yaml", "reduction_asset_name" );

    REQUIRE( lSensorModelBase.mRootAsset.Get<sRelationshipComponent>().mChildren.size() == 1 );
    REQUIRE( lSensorModelBase.AssetsByID["reduction_test_0"] == lNewAsset );
    REQUIRE( lSensorModelBase.IdsByAsset[lNewAsset] == "reduction_test_0" );
    REQUIRE( lNewAsset.Has<sAssetMetadata>() );
    REQUIRE( lNewAsset.Has<sReductionMapAssetC>() );
    REQUIRE( lNewAsset.Get<sRelationshipComponent>().mChildren.size() == 1 );
}

TEST_CASE( "Add pulse template asset", "[CORE_SENSOR_MODEL]" )
{
    SensorModelBase lSensorModelBase{};

    auto lNewAsset = lSensorModelBase.AddAsset( "pulse_template_0", "C:\\GitLab\\LTSimulationEngine\\Tests\\Data\\TestSensor0", "pulse_template/asset.yaml", "pulse_template_name" );

    REQUIRE( lSensorModelBase.mRootAsset.Get<sRelationshipComponent>().mChildren.size() == 1 );
    REQUIRE( lSensorModelBase.AssetsByID["pulse_template_0"] == lNewAsset );
    REQUIRE( lSensorModelBase.IdsByAsset[lNewAsset] == "pulse_template_0" );
    REQUIRE( lNewAsset.Has<sAssetMetadata>() );
    REQUIRE( lNewAsset.Has<sPulseTemplateAssetC>() );
    REQUIRE( lNewAsset.Get<sRelationshipComponent>().mChildren.size() == 1 );
}

TEST_CASE( "Add more than one asset", "[CORE_SENSOR_MODEL]" )
{
    SensorModelBase lSensorModelBase{};

    auto lNewAsset0 = lSensorModelBase.AddAsset( "pulse_template_0", "C:\\GitLab\\LTSimulationEngine\\Tests\\Data\\TestSensor0", "pulse_template/asset.yaml", "pulse_template_name" );
    auto lNewAsset1 = lSensorModelBase.AddAsset( "reduction_test_0", "C:\\GitLab\\LTSimulationEngine\\Tests\\Data\\TestSensor0", "reduction/asset.yaml", "reduction_asset_name" );

    REQUIRE( lSensorModelBase.mRootAsset.Get<sRelationshipComponent>().mChildren.size() == 2 );
    REQUIRE( lSensorModelBase.AssetsByID["pulse_template_0"] == lNewAsset0 );
    REQUIRE( lSensorModelBase.IdsByAsset[lNewAsset0] == "pulse_template_0" );
    REQUIRE( lSensorModelBase.AssetsByID["reduction_test_0"] == lNewAsset1 );
    REQUIRE( lSensorModelBase.IdsByAsset[lNewAsset1] == "reduction_test_0" );
}

TEST_CASE( "Adding duplicate asset IDs raises an error", "[CORE_SENSOR_MODEL]" )
{
    SensorModelBase lSensorModelBase{};

    auto lNewAsset0 = lSensorModelBase.AddAsset( "asset_id", "C:\\GitLab\\LTSimulationEngine\\Tests\\Data\\TestSensor0", "pulse_template/asset.yaml", "pulse_template_name" );

    try
    {
        auto lNewAsset1 = lSensorModelBase.AddAsset( "asset_id", "C:\\GitLab\\LTSimulationEngine\\Tests\\Data\\TestSensor0", "reduction/asset.yaml", "reduction_asset_name" );
        REQUIRE( false );
    }
    catch( ... )
    {
        REQUIRE( true );
    }
}

TEST_CASE( "Create laser component", "[CORE_SENSOR_MODEL]" )
{
    SensorDeviceBase lSensorModelBase{};

    lSensorModelBase.mSensorDefinition.AddAsset( "pulse_template_0", "C:\\GitLab\\LTSimulationEngine\\Tests\\Data\\TestSensor0", "pulse_template/asset.yaml", "pulse_template_name" );
    const std::string lCode          = R""""(components:
  laser_diode_1:
    type: laser_assembly
    name: laser_0
    data:
      waveform_template: pulse_template_0
      timebase_delay:  { x: 0.00000015010384283916842, y: 0.0, z: 0.0, w: 0.0 }
      flash_time: { x: 0.100000001, y: 0.0, z: 0.0, w: 0.0 }
)"""";
    Dev::ConfigurationReader lReader = Dev::ConfigurationReader( YAML::Load( lCode ) );
    Dev::ConfigurationNode lRoot     = lReader.GetRoot()["components.laser_diode_1"];
    auto lNewAsset                   = lSensorModelBase.CreateComponent( lSensorModelBase.mRootComponent, "laser_0", lRoot );

    REQUIRE( lSensorModelBase.mRootComponent.Get<sRelationshipComponent>().mChildren.size() == 1 );
    REQUIRE( lSensorModelBase.ComponentsByID["laser_0"] == lNewAsset );
    REQUIRE( lSensorModelBase.IdsByComponent[lNewAsset] == "laser_0" );
    REQUIRE( lNewAsset.Has<sSensorComponent>() );
    REQUIRE( lNewAsset.Get<sRelationshipComponent>().mChildren.size() > 0 );

    auto lLaserComponent = lNewAsset.Get<sRelationshipComponent>().mChildren[0];
    REQUIRE( lLaserComponent.Has<sLaserAssembly>() );
}

TEST_CASE( "Create photodetector component", "[CORE_SENSOR_MODEL]" )
{
    SensorDeviceBase lSensorModelBase{};

    lSensorModelBase.mSensorDefinition.AddAsset( "pulse_template_0", "C:\\GitLab\\LTSimulationEngine\\Tests\\Data\\TestSensor0", "pulse_template/asset.yaml", "pulse_template_name" );
    const std::string lCode          = R""""(components:
  photodetector_array_1:
    type: photodetector
    name: pd_array_0
    data:
      cells:
        - { position: { x: 0.2, y: 0.2, z: 0.1, w: 0.1 }, baseline: { x: 0.0, y: 0.0, z: 0.0, w: 0.0 }, static_noise: { asset_id: static_noise_1, mai_id: 0} }
        - { position: { x: 0.2, y: 0.2, z: 0.1, w: 0.1 }, baseline: { x: 0.0, y: 0.0, z: 0.0, w: 0.0 }, static_noise: { asset_id: static_noise_1, mai_id: 0} }
        - { position: { x: 0.2, y: 0.2, z: 0.1, w: 0.1 }, baseline: { x: 0.0, y: 0.0, z: 0.0, w: 0.0 }, static_noise: { asset_id: static_noise_1, mai_id: 0} }
)"""";
    Dev::ConfigurationReader lReader = Dev::ConfigurationReader( YAML::Load( lCode ) );
    Dev::ConfigurationNode lRoot     = lReader.GetRoot()["components.photodetector_array_1"];
    auto lNewAsset                   = lSensorModelBase.CreateComponent( lSensorModelBase.mRootComponent, "pd_array_0", lRoot );

    REQUIRE( lSensorModelBase.mRootComponent.Get<sRelationshipComponent>().mChildren.size() == 1 );
    REQUIRE( lSensorModelBase.ComponentsByID["pd_array_0"] == lNewAsset );
    REQUIRE( lSensorModelBase.IdsByComponent[lNewAsset] == "pd_array_0" );
    REQUIRE( lNewAsset.Has<sSensorComponent>() );
    REQUIRE( lNewAsset.Get<sRelationshipComponent>().mChildren.size() > 0 );

    auto lPhotodetectorComponent = lNewAsset.Get<sRelationshipComponent>().mChildren[0];
    REQUIRE( lPhotodetectorComponent.Has<sPhotoDetector>() );
}

TEST_CASE( "Create sampler component", "[CORE_SENSOR_MODEL]" )
{
    SensorDeviceBase lSensorModelBase{};

    const std::string lCode          = R""""(components:
  sampler_1:
    type: sampler
    name: sampler_1
    data:
        length: 2345
        frequency: 12345678.9
)"""";
    Dev::ConfigurationReader lReader = Dev::ConfigurationReader( YAML::Load( lCode ) );
    Dev::ConfigurationNode lRoot     = lReader.GetRoot()["components.sampler_1"];
    auto lNewAsset                   = lSensorModelBase.CreateComponent( lSensorModelBase.mRootComponent, "sampler_1", lRoot );

    REQUIRE( lSensorModelBase.mRootComponent.Get<sRelationshipComponent>().mChildren.size() == 1 );
    REQUIRE( lSensorModelBase.ComponentsByID["sampler_1"] == lNewAsset );
    REQUIRE( lSensorModelBase.IdsByComponent[lNewAsset] == "sampler_1" );
    REQUIRE( lNewAsset.Has<sSensorComponent>() );
    REQUIRE( lNewAsset.Get<sRelationshipComponent>().mChildren.size() > 0 );

    auto lPhotodetectorComponent = lNewAsset.Get<sRelationshipComponent>().mChildren[0];
    REQUIRE( lPhotodetectorComponent.Has<sSampler>() );
}


TEST_CASE("Load sensor from file", "[CORE_SENSOR_MODEL]")
{
    SensorDeviceBase lSensorModelBase{};
    lSensorModelBase.Load("C:\\GitLab\\LTSimulationEngine\\Tests\\Data\\TestSensor0", "TestSensor0.yaml");

    REQUIRE( lSensorModelBase.mSensorDefinition.mRootTile.Get<sRelationshipComponent>().mChildren.size() == 2 );
    REQUIRE( lSensorModelBase.mRootComponent.Get<sRelationshipComponent>().mChildren.size() == 4 );
    REQUIRE( lSensorModelBase.mSensorDefinition.mRootAsset.Get<sRelationshipComponent>().mChildren.size() == 3 );
}

TEST_CASE( "Add more than one asset", "[CORE_SENSOR_MODEL]" )
{
    SensorModelBase lSensorModelBase{};

    auto lNewAsset0 = lSensorModelBase.AddAsset( "pulse_template_0", "C:\\GitLab\\LTSimulationEngine\\Tests\\Data", "pulse_template/asset.yaml", "pulse_template_name" );
    auto lNewAsset1 = lSensorModelBase.AddAsset( "reduction_test_0", "C:\\GitLab\\LTSimulationEngine\\Tests\\Data", "reduction/asset.yaml", "reduction_asset_name" );

    REQUIRE( lSensorModelBase.mRootAsset.Get<sRelationshipComponent>().mChildren.size() == 2 );
    REQUIRE( lSensorModelBase.AssetsByID["pulse_template_0"] == lNewAsset0 );
    REQUIRE( lSensorModelBase.IdsByAsset[lNewAsset0] == "pulse_template_0" );
    REQUIRE( lSensorModelBase.AssetsByID["reduction_test_0"] == lNewAsset1 );
    REQUIRE( lSensorModelBase.IdsByAsset[lNewAsset1] == "reduction_test_0" );
}

TEST_CASE( "Adding duplicate asset IDs raises an error", "[CORE_SENSOR_MODEL]" )
{
    SensorModelBase lSensorModelBase{};

    auto lNewAsset0 = lSensorModelBase.AddAsset( "asset_id", "C:\\GitLab\\LTSimulationEngine\\Tests\\Data\\TestSensor0", "pulse_template/asset.yaml", "pulse_template_name" );

    try
    {
        auto lNewAsset1 = lSensorModelBase.AddAsset( "asset_id", "C:\\GitLab\\LTSimulationEngine\\Tests\\Data\\TestSensor0", "reduction/asset.yaml", "reduction_asset_name" );
        REQUIRE( false );
    }
    catch( ... )
    {
        REQUIRE( true );
    }
}

TEST_CASE( "Create laser component" )
{
    SensorModelBase lSensorModelBase{};

    lSensorModelBase.mSensorDefinition.AddAsset( "pulse_template_0", "C:\\GitLab\\LTSimulationEngine\\Tests\\Data\\TestSensor0", "pulse_template/asset.yaml", "pulse_template_name" );
    const std::string lCode          = R""""(components:
  laser_diode_1:
    type: laser_assembly
    name: laser_0
    data:
      waveform_template: pulse_template_0
      timebase_delay:  { x: 0.00000015010384283916842, y: 0.0, z: 0.0, w: 0.0 }
      flash_time: { x: 0.100000001, y: 0.0, z: 0.0, w: 0.0 }
)"""";
    Dev::ConfigurationReader lReader = Dev::ConfigurationReader( YAML::Load( lCode ) );
    Dev::ConfigurationNode lRoot     = lReader.GetRoot()["components.laser_diode_1"];
    auto lNewAsset                   = lSensorModelBase.CreateComponent( lSensorModelBase.mRootComponent, "laser_0", lRoot );

    REQUIRE( lSensorModelBase.mRootComponent.Get<sRelationshipComponent>().mChildren.size() == 1 );
    REQUIRE( lSensorModelBase.ComponentsByID["laser_0"] == lNewAsset );
    REQUIRE( lSensorModelBase.IdsByComponent[lNewAsset] == "laser_0" );
    REQUIRE( lNewAsset.Has<SensorComponent>() );
    REQUIRE( lNewAsset.Get<sRelationshipComponent>().mChildren.size() > 0 );

    auto lLaserComponent = lNewAsset.Get<sRelationshipComponent>().mChildren[0];
    REQUIRE( lLaserComponent.Has<sLaserAssembly>() );
}

TEST_CASE( "Create photodetector component", "[CORE_SENSOR_MODEL]" )
{
    SensorDeviceBase lSensorModelBase{};

    lSensorModelBase.mSensorDefinition.AddAsset( "pulse_template_0", "C:\\GitLab\\LTSimulationEngine\\Tests\\Data\\TestSensor0", "pulse_template/asset.yaml", "pulse_template_name" );
    const std::string lCode          = R""""(components:
  photodetector_array_1:
    type: photodetector
    name: pd_array_0
    data:
      cells:
        - { position: { x: 0.2, y: 0.2, z: 0.1, w: 0.1 }, baseline: { x: 0.0, y: 0.0, z: 0.0, w: 0.0 }, static_noise: { asset_id: static_noise_1, mai_id: 0} }
        - { position: { x: 0.2, y: 0.2, z: 0.1, w: 0.1 }, baseline: { x: 0.0, y: 0.0, z: 0.0, w: 0.0 }, static_noise: { asset_id: static_noise_1, mai_id: 0} }
        - { position: { x: 0.2, y: 0.2, z: 0.1, w: 0.1 }, baseline: { x: 0.0, y: 0.0, z: 0.0, w: 0.0 }, static_noise: { asset_id: static_noise_1, mai_id: 0} }
)"""";
    Dev::ConfigurationReader lReader = Dev::ConfigurationReader( YAML::Load( lCode ) );
    Dev::ConfigurationNode lRoot     = lReader.GetRoot()["components.photodetector_array_1"];
    auto lNewAsset                   = lSensorModelBase.CreateComponent( lSensorModelBase.mRootComponent, "pd_array_0", lRoot );

    REQUIRE( lSensorModelBase.mRootComponent.Get<sRelationshipComponent>().mChildren.size() == 1 );
    REQUIRE( lSensorModelBase.ComponentsByID["pd_array_0"] == lNewAsset );
    REQUIRE( lSensorModelBase.IdsByComponent[lNewAsset] == "pd_array_0" );
    REQUIRE( lNewAsset.Has<sSensorComponent>() );
    REQUIRE( lNewAsset.Get<sRelationshipComponent>().mChildren.size() > 0 );

    auto lPhotodetectorComponent = lNewAsset.Get<sRelationshipComponent>().mChildren[0];
    REQUIRE( lPhotodetectorComponent.Has<sPhotoDetector>() );
}

TEST_CASE("Load sensor from file", "[CORE_SENSOR_MODEL]")
{
    SensorDeviceBase lSensorModelBase{};
    lSensorModelBase.Load("C:\\GitLab\\LTSimulationEngine\\Tests\\Data\\TestSensor0", "TestSensor0.yaml");

    REQUIRE( lSensorModelBase.mSensorDefinition.mRootTile.Get<sRelationshipComponent>().mChildren.size() == 2 );
    REQUIRE( lSensorModelBase.mRootComponent.Get<sRelationshipComponent>().mChildren.size() == 3 );
    REQUIRE( lSensorModelBase.mSensorDefinition.mRootAsset.Get<sRelationshipComponent>().mChildren.size() == 3 );
}