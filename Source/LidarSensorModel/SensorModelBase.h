/// @file   SensorModelBase.h
///
/// @brief  Base class for sensor model definitions
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#pragma once

#include <filesystem>
#include <unordered_map>

#include "Core/EntityRegistry/Registry.h"

#include "Core/TextureData.h"
#include "Cuda/Texture2D.h"

#include "Components.h"

#include "Serialize/FileIO.h"
#include "Serialize/SensorAsset.h"
#include "Serialize/SensorDefinition.h"

/** @brief */
namespace LTSE::SensorModel
{

    using namespace LTSE::Core;

    /// @class SensorModelBase
    ///
    /// This is the base class for all simulated sensor models.  Here we hold all information
    /// relevant to the configuration of a particular sensor, and implement the main processing
    /// pipeline. This class maintains an entity registry which contains all the configured tile
    /// information, as well as configured information about each laser flash belonging to each
    /// tile.
    ///
    /// Abstractly, a tile as we implement it is little more than a position in space. All positional
    /// data for flashes and possible other children of tiles is relative to the position of the tile.
    /// This allows to move a tile around and not have to update the positions of the individual flashes
    /// which compose it.
    ///
    /// By definition, a `tile` is an entity which has the `TileSpecification` attached to it,
    /// while a flash has the `FlashSpecificationComponent` attached to it.  It is therefore possible to
    /// iterate over all tiles, or over all flashes, by querying either component type.
    ///
    ///
    class SensorModelBase
    {
      public:
        std::string mName = "";

        Entity mRootAsset{};
        Entity mRootLayout{};
        Entity mRootComponent{};
        Entity mRootTile{};

        SensorModelBase();
        SensorModelBase( const SensorModelBase & ) = default;

        ~SensorModelBase() = default;

        void Clear();
        Entity CreateEntity();
        Entity CreateEntity( Entity const &aParent );
        Entity CreateEntity( std::string const &aName );
        Entity CreateEntity( std::string const &aName, Entity const &aParent );

        template <typename _DataComponent> Entity CreateEntity( std::string const &aName, _DataComponent const &aDataComponent )
        {
            auto lNewEntity = CreateEntity( aName );
            lNewEntity.Add<_DataComponent>( aDataComponent );

            return lNewEntity;
        }

        template <typename _DataComponent> Entity CreateEntity( std::string const &aName, Entity const &aParent, _DataComponent const &aDataComponent )
        {
            auto lNewEntity = CreateEntity( aName, aParent );
            lNewEntity.Add<_DataComponent>( aDataComponent );

            return lNewEntity;
        }

        template <typename _DataComponent> Entity CreateEntity( Entity const &aParent, _DataComponent const &aDataComponent )
        {
            auto lNewEntity = CreateEntity( aParent );
            lNewEntity.Add<_DataComponent>( aDataComponent );

            return lNewEntity;
        }

        /// @brief Create an asset
        ///
        /// Create an asset entity with the provided ID and name.
        ///
        /// @param aID ID for the new asset
        /// @param aAssetName Name for the new asset, for display purposes
        ///
        /// @returns The newly created entity
        ///
        Entity CreateAsset( std::string const &aID, std::string const &aAssetName );

        /// @brief Create an asset
        ///
        /// Create an asset entity with the provided ID and name, and populate the asset using data from aAssetRoot and aAssetPath
        ///
        /// @param aID ID for the new asset
        /// @param aAssetName Name for the new asset, for display purposes
        /// @param aAssetRoot Root folder for the asset
        /// @param aAssetPath Path for the asset, relative to the root
        ///
        /// @returns The newly created entity
        ///
        Entity CreateAsset( std::string const &aID, std::string const &aAssetName, fs::path const &aAssetRoot, fs::path const &aAssetPath );

        /// @brief Create an asset
        ///
        /// Create an asset entity with the provided ID, and populate the asset using data that has been loaded from the serializer
        ///
        /// @param aID ID for the new asset
        /// @param aAssetData Data for the new asset
        ///
        /// @returns The newly created entity
        ///
        Entity CreateAsset( std::string const &aID, sSensorAssetData const &aAssetData );

        /// @brief Create a sensor elements
        ///
        /// Create a sensor element entity with the provided ID and name.
        ///
        /// @param aID ID for the new asset
        /// @param aComponentName Name for the new element, for display purposes
        ///
        /// @returns The newly created entity
        ///
        Entity CreateElement( std::string const &aID, std::string const &aComponentName );

        /// @brief Create a sensor elements
        ///
        /// Create a sensor element entity with the provided ID and name, and populate the asset using data that has been loaded from the serializer
        ///
        /// @param aID ID for the new asset
        /// @param aAssetData Data for the new asset
        ///
        /// @returns The newly created entity
        ///
        Entity CreateElement( std::string const &aID, sSensorComponentData const &aComponent );

        /// @brief Create a sensor elements
        ///
        /// Create a sensor element entity with the provided ID and name, tags it with the provided data component
        ///
        /// @param aName Name for the new element, for display purposes
        /// @param aID ID for the new element
        /// @param aDataComponent Data component to add to the newly created entity
        ///
        /// @returns The newly created entity
        ///
        template <typename _DataComponent> Entity CreateElement( std::string const &aName, std::string const &aID, _DataComponent const &aDataComponent )
        {
            auto lNewComponent = CreateElement( aID, aName );
            lNewComponent.Add<_DataComponent>( aDataComponent );

            return lNewComponent;
        }

        /// @brief Creates a new tile
        ///
        /// @param aTileID   The ID of the tile to be created. This should be unique.
        /// @param aPosition The position of the tile in spherical coordinates. The coordinates are specified
        ///                  in degrees, but the signs are not the usual angle sizes: negative elevation angles
        ///                  point downwards and negative azimuth angles point to the left.
        ///
        /// @return The newly created entity with basic tile components.
        ///
        Entity CreateTile( std::string const &aTileID, math::vec2 const &aPosition );

        /// @brief Creates a new laser flash
        ///
        /// @param aTile          The parent tile
        /// @param aFlashID       The ID of the laser flash
        /// @param aRelativeAngle The angular position of the flash with respect to the center of the parent tile.
        /// @param aExtent        The extent of the laset flash. This is the hald-width and the half height of the flash rectangle.
        ///
        /// @return The newly created entity with basic laser flash components.
        ///
        Entity CreateFlash( Entity aTile, std::string const &aFlashID, math::vec2 const &aRelativeAngle, math::vec2 const &aExtent );

        /// @brief Create a new tile layout with the given name and ID
        ///
        /// @param aID   ID of the new tile layout
        /// @param aName Name of the new tile layout
        ///
        /// @return An entity representing the newly create tile layout
        ///
        Entity CreateTileLayout( std::string aID, std::string aName );

        /// @brief Create a new tile layout with the given name and ID
        ///
        /// @param aID   ID of the new tile layout
        /// @param aData Data for the new tile layout
        ///
        /// @return An entity representing the newly create tile layout
        ///
        Entity CreateTileLayout( std::string aID, sTileLayoutData aData );

        /// @brief Create a new tile layout with the given name and ID
        ///
        /// @param aName Name of the new tile layout
        /// @param aData Data of the new tile layout
        ///
        /// @return An entity representing the newly create tile layout
        ///
        Entity SensorModelBase::CreateTileLayout( std::string aName, sTileLayoutComponent aData );

        /// @brief Retrieves the entire collection of tiles
        std::vector<Entity> GetAllTiles();

        /// @brief Retrieves a tile using its ID
        ///
        /// @param aTileID ID of the tile to retrieve
        ///
        Entity GetTileByID( std::string const &aTileID );

        /// @brief Retrieves an asset from its ID
        ///
        /// @param aAssetID ID of the asset to retrieve
        ///
        Entity GetAssetByID( std::string const &aAssetID );

        /// @brief Retrieves alayout from its ID
        ///
        /// @param aLayoutID ID of the layout to retrieve
        ///
        Entity GetLayoutByID( std::string const &aLayoutID );

        /// @brief Retrieves a component from its ID
        ///
        /// @param aComponentID ID of the component to retrieve
        ///
        Entity GetComponentByID( std::string const &aComponentID );

        /// @brief Iterate over all entities containing the listed components.
        template <typename... Args> void ForEach( std::function<void( Entity, Args &... )> aApplyFunction ) { mRegistry.ForEach<Args...>( aApplyFunction ); }

        /// @brief Load an asset for disk
        ///
        /// @param aID ID for the new asset
        /// @param aAssetName Name for the asset, for display purposes
        /// @param aAssetRoot Root path of the asset
        /// @param aAssetPath Path of the asset, relative to the root
        ///
        Entity LoadAsset( std::string const &aID, std::string const &aAssetName, fs::path const &aAssetRoot, fs::path const &aAssetPath );

      protected:
        /// @brief Load a testure from disk
        Ref<LTSE::Cuda::TextureSampler2D> LoadTexture( fs::path const &aTexturePath, std::string const &aName );
        Ref<LTSE::Cuda::TextureSampler2D> LoadTexture( Core::TextureData2D &aTexture, Core::TextureSampler2D &aSampler );

        /// @brief Fill asset data
        Entity LoadAndFillAssetDefinition( Entity aNewAsset, sPhotodetectorAssetData const &aAsset );
        Entity LoadAndFillAssetDefinition( Entity aNewAsset, sLaserAssetData const &aAsset );

      protected:
        EntityRegistry mRegistry; //!< Entity registry

        std::unordered_map<std::string, Entity> mTileIDToTileLUT = {}; //!< Lookup tile entities
        std::unordered_map<std::string, Entity> mLayoutByID      = {}; //!< Lookup asset entities
        std::unordered_map<std::string, Entity> mAssetsByID      = {}; //!< Lookup asset entities
        std::unordered_map<std::string, Entity> mComponentsByID  = {}; //!< Lookup component entities
    };
} // namespace LTSE::SensorModel
