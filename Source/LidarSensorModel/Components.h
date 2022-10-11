/// @file   Components.h
///
/// @brief  Definition file for the different sensor model component types.
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#pragma once

#include <algorithm>
#include <filesystem>

#include "Core/Memory.h"

#include "Core/Math/Types.h"

#include "Core/EntityRegistry/Registry.h"

#include "Cuda/Texture2D.h"

/** @brief */
namespace LTSE::SensorModel
{

    using namespace LTSE::Core;
    using namespace math;
    using Interpolator = Ref<LTSE::Cuda::TextureSampler2D>;

    /// @struct sTileSpecificationComponent
    ///
    /// This component is added to an entity to tag it as a `Tile`
    ///
    struct sTileSpecificationComponent
    {
        std::string mID = "";             //!< mID for the tile. This should be unique.
        vec2 mPosition  = { 0.0f, 0.0f }; //!< Default position for the tile

        sTileSpecificationComponent()                                      = default;
        sTileSpecificationComponent( const sTileSpecificationComponent & ) = default;
    };

    /// @struct sLaserFlashSpecificationComponent
    ///
    /// This component is added to an entity to tag said entity as a `LaserFlash`. Laser flashes
    /// are generally parented to a specific tile, and all their position and extent components
    /// are relative to their parent tile.
    ///
    struct sLaserFlashSpecificationComponent
    {
        std::string mFlashID = "";             //!< mID for the flash. This should be unique for the flash.
        float mTimebaseDelay = 0.0f;           //!< Timebase delay hint for this flash
        vec2 mPosition       = { 0.0f, 0.0f }; //<! Position of the flash relative to the tile's position.
        vec2 mExtent         = { 0.0f, 0.0f }; //<! Extent (radius) of the flash. We can get the full size of the tile via Extent * 2.0f

        sLaserFlashSpecificationComponent()                                            = default;
        sLaserFlashSpecificationComponent( const sLaserFlashSpecificationComponent & ) = default;
    };

    /// @struct sAssetMetadata
    ///
    /// Tags an entity as containing an asset. Asset entities are containers that contain actual asset data in
    /// the form of a hash table mapping string IDs to other entities in the registry. For example, a diffusion
    /// asset will contain references to the various textures to be used in computation. All asset entities have
    /// at least one child/
    ///
    struct sAssetMetadata
    {
        std::string mID                                        = ""; //!< ID for the asset
        std::unordered_map<std::string, Entity> mChildEntities = {}; //!< Mapping to the entities that contain the usable asset data.

        sAssetMetadata()                         = default;
        sAssetMetadata( const sAssetMetadata & ) = default;
    };

    /// @struct sAssetLocation
    ///
    /// Location of the asset data on disk. This is used by the archiving method to produce a reference ot the asset.
    ///
    struct sAssetLocation
    {
        fs::path mRoot     = ""; //!< Location of the asset definition file. All asset data is relative to this path
        fs::path mFilePath = ""; //!< Name of the asset definition file, typically "asset.yaml"

        sAssetLocation()                         = default;
        sAssetLocation( const sAssetLocation & ) = default;
    };

    /// @struct sInternalAssetReference
    ///
    /// This component stores a reference to the asset as a child of the asset container. It is used when saving sensor
    /// configurations to disk.
    ///
    struct sInternalAssetReference
    {
        std::string mParentID = "";
        std::string mID       = "";

        sInternalAssetReference()                                  = default;
        sInternalAssetReference( const sInternalAssetReference & ) = default;
    };

    /// @struct sDiffusionAssetTag
    ///
    /// Tag that indicates that the entity holds a diffusion pattern asset. This tag should be applied to the same
    /// entity that has the sAssetMetadata component.
    ///
    struct sDiffusionAssetTag
    {
        // Empty structure, this is on purpose
    };

    /// @struct sReductionMapAssetTag
    ///
    /// Tag that indicates that the entity holds a reduction pattern asset. This tag should be applied to the same
    /// entity that has the sAssetMetadata component.
    ///
    struct sReductionMapAssetTag
    {
        // Empty structure, this is on purpose
    };

    /// @struct sPulseTemplateAssetTag
    ///
    /// Tag that indicates that the entity holds a pulse template asset. This tag should be applied to the same
    /// entity that has the sAssetMetadata component.
    ///
    struct sPulseTemplateAssetTag
    {
        // Empty structure, this is on purpose
    };

    /// @struct sStaticNoiseAssetTag
    ///
    /// Tag that indicates that the entity holds a static noise asset. This tag should be applied to the same
    /// entity that has the sAssetMetadata component.
    ///
    struct sStaticNoiseAssetTag
    {
        // Empty structure, this is on purpose
    };

    /// @struct sPhotoDetectorAssetTag
    ///
    /// Tag that indicates that the entity holds a photodetector asset. This tag should be applied to the same
    /// entity that has the sAssetMetadata component.
    ///
    struct sPhotodetectorAssetTag
    {
        // Empty structure, this is on purpose
    };

    /// @struct sLaserAssemblyAssetTag
    ///
    /// Tag that indicates that the entity holds a laser assembly asset. This tag should be applied to the same
    /// entity that has the sAssetMetadata component.
    ///
    struct sLaserAssemblyAssetTag
    {
        // Empty structure, this is on purpose
    };

    /// @struct sDiffusionPattern
    ///
    /// Indicates that the entity holds a diffusion pattern to be used as coefficients for laser beam intensities.
    ///
    struct sDiffusionPattern
    {
        Interpolator mInterpolator = nullptr; //!< Interpolator structure

        sDiffusionPattern()                            = default;
        sDiffusionPattern( const sDiffusionPattern & ) = default;
    };

    /// @struct sPulseTemplate
    ///
    /// Indicates that the entity holds a pulse template to be used when creating waveforms.
    ///
    struct sPulseTemplate
    {
        Interpolator mInterpolator = nullptr; //!< Interpolator structure

        sPulseTemplate()                         = default;
        sPulseTemplate( const sPulseTemplate & ) = default;
    };

    /// @struct sStaticNoise
    ///
    /// Indicates that the entity holds static noise data to be added to waveforms.
    ///
    struct sStaticNoise
    {
        Interpolator mInterpolator = nullptr; //!< Interpolator structure

        sStaticNoise()                       = default;
        sStaticNoise( const sStaticNoise & ) = default;
    };

    /// @struct sElectronicCrosstalk
    ///
    /// Indicates that the entity holds electronic crosstalk data to be added to waveforms.
    ///
    struct sElectronicCrosstalk
    {
        Interpolator mInterpolator = nullptr; //!< Interpolator structure

        sElectronicCrosstalk()                               = default;
        sElectronicCrosstalk( const sElectronicCrosstalk & ) = default;
    };

    /// @struct sSensorComponent
    ///
    /// Indidates that the entity represents a sensor element. Sensor elements typically
    /// group together asset entities and other data used to simulate a sensor.
    ///
    struct sSensorComponent
    {
        std::string mID = ""; //!< ID for the component

        sSensorComponent()                           = default;
        sSensorComponent( const sSensorComponent & ) = default;
    };

    /// @struct sPhotoDetector
    ///
    /// Indidates that the entity represents a photodetector element. This should be added to the
    /// same entity that contains the `sSensorComponent` component. Photodetector elements contain
    /// the data necessary to represent the reception side of the sensor: APD cell position and size,
    /// as well a handle to an entity containing static noise data, and the base level to use for
    /// sampling waveforms. Photodetector elements are typically adjoined to laser flashes
    ///
    struct sPhotoDetector
    {
        std::vector<math::vec4> mCellPositions    = {}; //!< Cell position and size, encoded as { x, y, w, h }, there x and y represent the center of the cell. In degrees.
        std::vector<math::vec4> mGain             = {}; //!< Cell gain factor
        std::vector<math::vec4> mBaseline         = {}; //!< Cubic polynomial representing the baseline shift for each photodetector cell as a function of temperature.
        std::vector<math::vec4> mStaticNoiseShift = {}; //!< Cubic polynomial representing the x-axis shift for the static noise as a function of temperature.
        std::vector<Entity> mStaticNoise          = {}; //!< Handle to an external entity containing a static noise interpolator for each photodetector cell.
        std::vector<Entity> mElectronicCrosstalk  = {}; //!< Pulse template interpolator to use for this photodetector cell.

        sPhotoDetector()                         = default;
        sPhotoDetector( const sPhotoDetector & ) = default;
    };

    /// @struct sLaserAssembly
    ///
    /// Indicates that the entity represents a laser diode. This should be added to the same entity
    /// that contains the `sSensorComponent` component. Laser elements hold all data necessary to
    /// generate a laser pulse at a given wavelength for a given duration in time. Laser elements are
    /// typically adjoined to configured laser flashes
    ///
    struct sLaserAssembly
    {
        Entity mWaveformData{}; //!< Pulse template interpolator to use for this laser.
        Entity mDiffuserData{}; //!< Pulse template interpolator to use for this laser.

        math::vec4 mTimebaseDelay = { 0.0f, 0.0f, 0.0f, 0.0f }; //!< Cubic polynomial representing the timebase delay of the laser as a function of temperature.
        math::vec4 mFlashTime     = { 0.1f, 0.0f, 0.0f, 0.0f }; //!< Cubic polynomial representing flast time of the laser as a function of temperature.

        sLaserAssembly()                         = default;
        sLaserAssembly( const sLaserAssembly & ) = default;
    };

    /// @struct sSampler
    ///
    /// Indicates that the entity represents a sampler. This should be added to the same entity
    /// that contains the `sSensorComponent` component. The sampler are responsible for producing
    /// waveform data from raw processed detections. Typically the sampler is adjoined to a tile,
    /// in order for the sampling to be consistent throughout flashes.
    ///
    struct sSampler
    {
        uint32_t mLength = 1024;         //!< Number of samples to collect, starting at time t=0.
        float mFrequency = 800000000.0f; //!< Sampling frequency.

        sSampler()                   = default;
        sSampler( const sSampler & ) = default;
    };

    /// @struct sTileLayoutComponent
    ///
    /// Indicates that the entity represents a tile layout. Each tile layouts provides a mapping
    /// betweeen a set of external tile IDs and the data necessary to retrieve the tile definition
    /// in the internal database and the position where the tile should be sampled. This allows for
    /// tile IDs to be procedurally generated and associated to actual tile definitions even after
    /// the sensor configuration has been loaded.
    ///
    struct sTileLayoutComponent
    {
        using TileData = std::pair<std::string, math::vec2>;

        std::string mID                                   = ""; //!< ID for the tile layout
        std::unordered_map<std::string, TileData> mLayout = {}; //!< Layout data

        sTileLayoutComponent()                               = default;
        sTileLayoutComponent( const sTileLayoutComponent & ) = default;
    };

} // namespace LTSE::SensorModel