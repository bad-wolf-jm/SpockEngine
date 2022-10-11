/// @file   SensorDefinition.cpp
///
/// @brief  Implementation file for sensor serialization
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2022 LeddarTech Inc. All rights reserved.

#include "SensorDefinition.h"
#include "Core/Logging.h"

namespace LTSE::SensorModel
{

    static sFlashData LoadFlash( ConfigurationNode const &aFlashDefinition )
    {
        sFlashData lNewFlash{};
        lNewFlash.Area = aFlashDefinition["flash_area"].Vec( { "x", "y", "w", "h" }, { 0.0f, 0.0f, 0.0f, 0.0f } );

        lNewFlash.LaserDiodeComponentID    = aFlashDefinition["laser_diode_asset_id"].As<std::string>( "" );
        lNewFlash.PhotodetectorComponentID = aFlashDefinition["photodetector_asset_id"].As<std::string>( "" );

        return lNewFlash;
    }

    static sTileData LoadTile( std::string aID, ConfigurationNode const &aTileDefinition )
    {
        sTileData lNewTile{};

        lNewTile.mID         = aID;
        lNewTile.mPosition   = aTileDefinition["position"].Vec( { "x", "y" }, { 0.0f, 0.0f } );
        lNewTile.FieldOfView = aTileDefinition["field_of_view"].Vec( { "x", "y" }, { 0.0f, 0.0f } );

        lNewTile.SamplerComponentID = aTileDefinition["sampler_component_id"].As<std::string>( "" );

        aTileDefinition["flashes"].ForEach( [&]( ConfigurationNode &aValue ) { lNewTile.Flashes.push_back( LoadFlash( aValue ) ); } );

        return lNewTile;
    }

    static sTileLayoutData LoadLayout( std::string aKey, ConfigurationNode &aTileLayoutDefinition )
    {
        sTileLayoutData lNewTileLayout{};

        lNewTileLayout.mID   = aKey;
        lNewTileLayout.mName = aTileLayoutDefinition["name"].As<std::string>( "" );

        aTileLayoutDefinition["elements"].ForEach<std::string>(
            [&]( std::string aID, ConfigurationNode &aValue )
            {
                sTileLayoutData::sTileLayoutElement lElement{};
                lElement.mTileID   = aValue["tile_id"].As<std::string>( "" );
                lElement.mPosition = aValue["position"].Vec( { "x", "y" }, { 0.0f, 0.0f } );

                lNewTileLayout.Elements[aID] = lElement;
            } );

        return lNewTileLayout;
    }

    sSensorDefinition ReadSensorDefinition( fs::path const &aRoot, fs::path const &aPath )
    {
        sSensorDefinition lNewSensor{};
        ConfigurationReader lConfigFile( aRoot / aPath );
        ConfigurationNode lRootNode = lConfigFile.GetRoot();

        lNewSensor.mName = lRootNode["sensor.global.name"].As<std::string>( "" );

        lRootNode["sensor.assets"].ForEach<std::string>(
            [&]( std::string aKey, ConfigurationNode &aValue )
            {
                fs::path lAssetPath     = aPath.parent_path() / aValue["path"].As<std::string>( "" );
                std::string l_AssetName = aValue["name"].As<std::string>( "" );
                lNewSensor.Assets[aKey] = ReadAsset( aRoot, lAssetPath, l_AssetName );
            } );

        lRootNode["sensor.components"].ForEach<std::string>( [&]( std::string aKey, ConfigurationNode &aValue ) { lNewSensor.Components[aKey] = ReadComponent( aValue ); } );

        lRootNode["sensor.tiles"].ForEach<std::string>( [&]( std::string aKey, ConfigurationNode &aValue ) { lNewSensor.Tiles[aKey] = LoadTile( aKey, aValue ); } );

        lRootNode["sensor.layouts"].ForEach<std::string>( [&]( std::string aKey, ConfigurationNode &aValue ) { lNewSensor.Layouts[aKey] = LoadLayout( aKey, aValue ); } );

        return lNewSensor;
    }

    sSensorDefinition ReadSensorDefinitionFromString( fs::path const &aRoot, std::string const &aDefinition )
    {
        sSensorDefinition lNewSensor{};
        ConfigurationReader lConfigFile( aDefinition );
        ConfigurationNode lRootNode = lConfigFile.GetRoot();

        lNewSensor.mName = lRootNode["sensor.global.name"].As<std::string>( "" );

        lRootNode["sensor.assets"].ForEach<std::string>(
            [&]( std::string aKey, ConfigurationNode &aValue )
            {
                fs::path lAssetPath     = aValue["path"].As<std::string>( "" );
                std::string l_AssetName = aValue["name"].As<std::string>( "" );
                lNewSensor.Assets[aKey] = ReadAsset( aRoot, lAssetPath, l_AssetName );
            } );

        lRootNode["sensor.components"].ForEach<std::string>( [&]( std::string aKey, ConfigurationNode &aValue ) { lNewSensor.Components[aKey] = ReadComponent( aValue ); } );

        lRootNode["sensor.tiles"].ForEach<std::string>( [&]( std::string aKey, ConfigurationNode &aValue ) { lNewSensor.Tiles[aKey] = LoadTile( aKey, aValue ); } );

        lRootNode["sensor.layouts"].ForEach<std::string>( [&]( std::string aKey, ConfigurationNode &aValue ) { lNewSensor.Layouts[aKey] = LoadLayout( aKey, aValue ); } );

        return lNewSensor;
    }

    static void SaveAssetList( ConfigurationWriter &out, std::unordered_map<std::string, sSensorAssetData> const &aAssets, std::string const &aName )
    {
        out.WriteKey( aName );
        out.BeginMap();
        for( auto &lAssetEntity : aAssets )
        {
            out.WriteKey( lAssetEntity.first );
            out.BeginMap();
            out.WriteKey( "name", lAssetEntity.second.mName );
            out.WriteKey( "path", lAssetEntity.second.mFilePath.string() );
            out.EndMap();
        }
        out.EndMap();
    }

    static void SaveComponent( ConfigurationWriter &out, std::string const &aKey, sSensorComponentData const &aComponent )
    {
        out.WriteKey( aKey );
        out.BeginMap();
        switch( aComponent.Type() )
        {
        case eComponentType::SAMPLER:
        {
            out.WriteKey( "type", "sampler" );
            out.WriteKey( "name", aComponent.mName );
            out.WriteKey( "data" );
            out.BeginMap();
            out.WriteKey( "length", aComponent.Get<sSamplerComponentData>().mLength );
            out.WriteKey( "frequency", aComponent.Get<sSamplerComponentData>().mFrequency );
            out.EndMap();
            break;
        }
        }
        out.EndMap();
    }

    static void SaveComponentsList( ConfigurationWriter &out, std::unordered_map<std::string, sSensorComponentData> const &aComponents, std::string const &aName )
    {
        out.WriteKey( aName );
        out.BeginMap();
        for( auto &lComponent : aComponents )
            SaveComponent( out, lComponent.first, lComponent.second );
        out.EndMap();
    }

    static void SaveFlashList( ConfigurationWriter &out, std::vector<sFlashData> const &aFlashes, std::string const &aName )
    {
        out.WriteKey( "flashes" );
        out.BeginSequence();
        for( auto &lFlash : aFlashes )
        {
            out.BeginMap();
            out.WriteKey( "flash_area" );
            out.Write( math::vec4{ lFlash.Area.x, lFlash.Area.y, lFlash.Area.z, lFlash.Area.w }, { "x", "y", "w", "h" } );
            out.WriteKey( "laser_diode_asset_id", lFlash.LaserDiodeComponentID );
            out.WriteKey( "photodetector_asset_id", lFlash.PhotodetectorComponentID );
            out.EndMap();
        }
        out.EndSequence();
    }

    static void SaveTileList( ConfigurationWriter &out, std::unordered_map<std::string, sTileData> const &aTiles, std::string const &aName )
    {
        out.WriteKey( aName );
        out.BeginMap();
        {
            for( auto &lTile : aTiles )
            {
                out.WriteKey( lTile.first );
                out.BeginMap();
                out.WriteKey( "id", lTile.second.mID );

                out.WriteKey( "position" );
                out.Write( lTile.second.mPosition, { "x", "y" } );

                out.WriteKey( "field_of_view" );
                out.Write( lTile.second.FieldOfView, { "x", "y" } );
                out.WriteKey( "sampler_component_id", lTile.second.SamplerComponentID );
                SaveFlashList( out, lTile.second.Flashes, "flashes" );
                out.EndMap();
            }
        }
        out.EndMap();
    }

    static void SaveTileLayouts( ConfigurationWriter &out, std::unordered_map<std::string, sTileLayoutData> const &aTileLayouts, std::string const &aName )
    {
        out.WriteKey( aName );
        out.BeginMap();
        {
            for( auto &lTileLayout : aTileLayouts )
            {
                out.WriteKey( lTileLayout.first );
                out.BeginMap();
                out.WriteKey( "name", lTileLayout.second.mName );
                out.WriteKey( "elements" );
                out.BeginMap();
                {
                    for( auto &lLayoutElement : lTileLayout.second.Elements )
                    {
                        out.WriteKey( lLayoutElement.first );
                        out.BeginMap();
                        {
                            out.WriteKey( "tile_id", lLayoutElement.second.mTileID );
                            out.WriteKey( "position" );
                            out.Write( lLayoutElement.second.mPosition, { "x", "y" } );
                        }
                        out.EndMap();
                    }
                }
                out.EndMap();
                out.EndMap();
            }
        }
        out.EndMap();
    }

    void SaveSensorDefinition( sSensorDefinition const &aSensorDefinition, ConfigurationWriter &out )
    {
        out.BeginMap();
        {
            out.WriteKey( "sensor" );

            out.BeginMap();
            {
                out.WriteKey( "global" );
                out.BeginMap();
                out.WriteKey( "name", aSensorDefinition.mName );
                out.EndMap();

                out.WriteKey( "sensor" );
                out.BeginMap();
                {
                    out.WriteKey( "field_of_view" );
                    out.WriteNull();
                }
                out.EndMap();

                SaveAssetList( out, aSensorDefinition.Assets, "assets" );
                SaveComponentsList( out, aSensorDefinition.Components, "components" );
                SaveTileLayouts( out, aSensorDefinition.Layouts, "layouts" );
                SaveTileList( out, aSensorDefinition.Tiles, "tiles" );
            }
            out.EndMap();
        }
        out.EndMap();
    }

    void SaveSensorDefinition( sSensorDefinition const &aSensorDefinition, fs::path const &aRoot, fs::path const &aModelFilePath )
    {
        ConfigurationWriter out( aRoot / aModelFilePath );
        SaveSensorDefinition( aSensorDefinition, out );
    }

    std::string ToString( sSensorDefinition const &aSensorDefinition )
    {
        ConfigurationWriter out{};
        SaveSensorDefinition( aSensorDefinition, out );
        return out.GetString();
    }

} // namespace LTSE::SensorModel