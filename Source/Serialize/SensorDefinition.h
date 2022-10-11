/// @file   SensorDefinition.h
///
/// @brief  Definition for sensor serialization
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2022 LeddarTech Inc. All rights reserved.

#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "Core/Math/Types.h"

#include "FileIO.h"
#include "SensorAsset.h"
#include "SensorComponents.h"

namespace LTSE::SensorModel
{
    namespace fs = std::filesystem;

    /// @brief Reference to a list-type asset
    ///
    /// Since list-type assets are represented as a list of individual interpolated textures, a reference
    /// to such an asset should contain the name of the asset to reference, as well as the index of the texture
    /// to use.
    ///
    struct sAssetInternalReference
    {
        std::string mAssetID = ""; //!< Asset ID
        std::string mMapID   = ""; //<! Texture ID within the asset

        /// @brief Default constructor
        sAssetInternalReference() = default;

        /// @brief Copy constructor
        sAssetInternalReference( const sAssetInternalReference & ) = default;

        /// @brief Default destructor
        ~sAssetInternalReference() = default;
    };

    /// @brief Abstract laser flash data
    ///
    /// Contains data about individual laser flashes as described in the configuration. Laser flashes
    /// contain enough information to carry out most of the operations related to environment sampling
    /// and the front-end model.
    ///
    /// ```yaml
    /// id: 1
    /// flash_area: { x: 0.0, y: 0.0, w: 0.0, h: 0.0 }
    /// attenuation_map: { asset_id: diffusion_asset_id, map_id: 0 }
    /// laser_diode_asset_id: laser_diode_component_id
    /// photodetector_asset_id: laser_diode_component_id
    /// ```
    ///
    /// Note that the ID field is an arbitrary 32-bit integer, and does not need to correspond to
    /// the position of the flash in the flash sequence of its parent tile.
    ///
    struct sFlashData
    {
        // std::string mID = "";                         //!< ID of the laser flash
        math::vec4 Area = { 0.0f, 0.0f, 0.0f, 0.0f }; // Area covered by the laser flash in the absence of any diffusion.

        std::string LaserDiodeComponentID    = ""; //!< Reference to the laser that will produce the pulse for this flash
        std::string PhotodetectorComponentID = ""; //! Reference to the photodetector that will collect the data originating from thie laser flash

        /// @brief Default constructor
        sFlashData() = default;

        /// @brief Copy constructor
        sFlashData( const sFlashData & ) = default;

        /// @brief Default destructor
        ~sFlashData() = default;
    };

    /// @brief Abstract tile data
    ///
    /// In our case, a tile is just a position in space and a list of laser flashes which share a similar configuration.
    ///
    /// ```yaml
    /// id: 1
    /// position: { x: 0, y: 0 }
    /// field_of_view: { x: 0.0, y: 0.0 }
    /// sampler_component_id: sampler_component_id
    /// flashes:
    ///   - flash_data_0
    ///   - flash_data_1
    ///   -...
    /// ```
    ///
    /// Here the ID of a tile is an arbitrary 64-bit integer used for hashing.
    ///
    struct sTileData
    {
        std::string mID                 = "";             //!< ID for the tile
        math::vec2 mPosition            = { 0.0f, 0.0f }; //!< Default position for the tile
        math::vec2 FieldOfView          = { 0.0f, 0.0f }; //!< Field of view of the tile (for reference only)
        std::string SamplerComponentID  = "";             //!< Reference to the sampler to use for this tile
        std::vector<sFlashData> Flashes = {};             //!< List of configured flashes for this tile

        /// @brief Default constructor
        sTileData() = default;

        /// @brief Copy constructor
        sTileData( const sTileData & ) = default;

        /// @brief Default destructor
        ~sTileData() = default;
    };

    /// @brief Tile layout data
    ///
    /// Tile layouts provide a way to associate a position to a configured tile, and
    /// reference each the (position, tile) pair using a separate ID
    ///
    /// ```yaml
    ///  name: tile_layout_1_name
    ///  elements:
    ///    tile_layout_id_0: { tile_id: tile_ref_0, position: { x: 0, y: 0 }}
    ///    tile_layout_id_1: { tile_id: tile_ref_1, position: { x: 0, y: 0 }}
    ///    tile_layout_id_2: { tile_id: tile_ref_2, position: { x: 0, y: 0 }}
    /// ```
    ///
    struct sTileLayoutData
    {
        struct sTileLayoutElement
        {
            std::string mTileID  = "";             //!< Internal tile ID. This id must match one of the configured tiles in the sensor definition
            math::vec2 mPosition = { 0.0f, 0.0f }; //!< Tile position.
        };

        std::string mID                                              = ""; //!< ID for the tile layout
        std::string mName                                            = ""; //!< Name for the tile layout, for display purposes
        std::unordered_map<std::string, sTileLayoutElement> Elements = {}; //!< Mapping tile -> position

        /// @brief Default constructor
        sTileLayoutData() = default;

        /// @brief Copy constructor
        sTileLayoutData( const sTileLayoutData & ) = default;

        /// @brief Default destructor
        ~sTileLayoutData() = default;
    };

    /// @brief Abstract tile data
    ///
    /// Parsed version of a sensor as read from disk.
    ///
    /// ```yaml
    /// sensor:
    ///   global:
    ///     name: "Sensor name"
    ///   assets:
    ///     asset_id_0: { name: asset_name_0, path: asset_path_0}
    ///     asset_id_1: { name: asset_name_1, path: asset_path_1}
    ///     ...
    ///   components:
    ///     component_id_0: component_definition_0
    ///     component_id_1: component_definition_1
    ///     ...
    ///   tiles:
    ///     - tile_definition_0
    ///     - tile_definition_1
    ///     - ...
    /// ```
    ///
    struct sSensorDefinition
    {
        std::string mName = ""; //!< mName for the sensor, for display purposes

        std::unordered_map<std::string, sSensorAssetData> Assets         = {}; //!< Asset ID to asset mapping
        std::unordered_map<std::string, sSensorComponentData> Components = {}; //!< Component ID to component mapping
        std::unordered_map<std::string, sTileData> Tiles                 = {}; //!< List of configured tiles for the sensor
        std::unordered_map<std::string, sTileLayoutData> Layouts         = {}; //!< List of configured tile layouts

        /// @brief Default constructor
        sSensorDefinition() = default;

        /// @brief Copy constructor
        sSensorDefinition( const sSensorDefinition & ) = default;

        /// @brief Default destructor
        ~sSensorDefinition() = default;
    };

    /// @brief Load a sensor from file
    ///
    /// Load the file located at aRoot / aPath
    ///
    /// @param aRoot Root folder
    /// @param aPath path to sensor definition file, relative to the root
    ///
    /// @return the sensor definition.
    ///
    sSensorDefinition ReadSensorDefinition( fs::path const &aRoot, fs::path const &aPath );

    /// @brief Load a sensor from a string
    ///
    /// Load the sensor described in `aDefinition`, using `aRoot` as a root folder for assets.
    ///
    /// @param aRoot Root folder for asset path resolution
    /// @param aDefinition Sensor definition as a string
    ///
    /// @return the sensor definition.
    ///
    sSensorDefinition ReadSensorDefinitionFromString( fs::path const &aRoot, std::string const &aDefinition );

    /// @brief Save the given sensor configuration to disk
    ///
    /// The given sensor definition is serialized to the file located by aRoot / aModelFilePath.
    /// This is an overloaded funciton provided for convenience.
    ///
    void SaveSensorDefinition( sSensorDefinition const &aSensorDefinition, fs::path const &aRoot, fs::path const &aModelFilePath );

    /// @brief Save the given sensor configuration to the provided output writer
    void SaveSensorDefinition( sSensorDefinition const &aSensorDefinition, ConfigurationWriter &out );

    /// @brief Save the given sensor configuration to a string
    std::string ToString( sSensorDefinition const &aSensorDefinition );

} // namespace LTSE::SensorModel
