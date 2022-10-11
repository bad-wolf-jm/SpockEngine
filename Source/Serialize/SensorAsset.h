/// @file   SensorAsset.h
///
/// @brief  Definitions for sensor asset types
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2022 LeddarTech Inc. All rights reserved.

#pragma once

#include <filesystem>
#include <variant>
#include <vector>

#include "FileIO.h"

namespace LTSE::SensorModel
{
    namespace fs = std::filesystem;

    /// @brief Laser component
    ///
    /// This provides information about laser diodes, the shape of pulse that they emit, the time
    /// interval between two pulses, as well as the delay between triggering the pulse, and the actual
    /// firing of the laser.
    ///
    struct sLaserAssetData
    {
        fs::path mWaveformTemplate = ""; //!< Path of the texture holding waveform data
        fs::path mDiffuser         = ""; //!< Path of the texture holding diffuser data

        math::vec4 mTimebaseDelay = { 0.0f, 0.0f, 0.0f, 0.0f }; //!< Timebase delay as a cubic polynomial with respect to temperature
        math::vec4 mFlashTime     = { 0.0f, 0.0f, 0.0f, 0.0f }; //!< Flash time as a cubic polynomial with respect to temperature

        /// @brief Default constructor
        sLaserAssetData() = default;

        /// @brief Copy constructor
        sLaserAssetData( const sLaserAssetData & ) = default;

        /// @brief Default destructor
        ~sLaserAssetData() = default;
    };

    /// @brief Photodetector
    ///
    /// ```yaml
    /// asset:
    ///   type: photodetector
    ///   data:
    ///     static_noise: file_name_for_texture
    ///     extalk_matrix: file_name_for_texture
    ///     cells:
    ///       - { id: 0, position: { x: 0.0, y: 0.0, z: 0.0, w: 0.0 }, baseline: { x: 0.0, y: 0.0, z: 0.0, w: 0.0 }, static_noise_shift: { x: 0.0, y: 0.0, z: 0., w: 0.0 } }
    ///       - { id: 1, position: { x: 0.0, y: 0.0, z: 0.0, w: 0.0 }, baseline: { x: 0.0, y: 0.0, z: 0.0, w: 0.0 }, static_noise_shift: { x: 0.0, y: 0.0, z: 0., w: 0.0 } }
    ///       - { id: 2, position: { x: 0.0, y: 0.0, z: 0.0, w: 0.0 }, baseline: { x: 0.0, y: 0.0, z: 0.0, w: 0.0 }, static_noise_shift: { x: 0.0, y: 0.0, z: 0., w: 0.0 } }
    ///       - ...
    /// ```
    ///
    struct sPhotodetectorAssetData
    {
        fs::path mStaticNoiseData      = ""; //!< Path to file containing texture data for static noise
        fs::path mElectronicXtalkData  = ""; //!< Path to file containing texture data for electronic xtalk

        struct CellData
        {
            uint32_t mId                 = std::numeric_limits<uint32_t>::max(); //!< ID of the individual cell
            math::vec4 mPosition         = { 0.0f, 0.0f, 0.0f, 0.0f };           //!< Position of the cell, relative to the origin.
            math::vec4 mGain             = { 0.0f, 0.0f, 0.0f, 0.0f };           //!< Gain for the cell, represented as a cubic polynomial function of temperature
            math::vec4 mBaseline         = { 0.0f, 0.0f, 0.0f, 0.0f };           //!< Baseline, represented as a cubic polynomial function of temperature
            math::vec4 mStaticNoiseShift = { 0.0f, 0.0f, 0.0f, 0.0f };           //!< Static noise shift, represented as a cubic polynomial function of temperature
        };

        std::vector<CellData> mCells; //!< Photodetector cell data

        /// @brief Default constructor
        sPhotodetectorAssetData() = default;

        /// @brief Copy constructor
        sPhotodetectorAssetData( const sPhotodetectorAssetData & ) = default;

        /// @brief Default destructor
        ~sPhotodetectorAssetData() = default;
    };

    /// @brief Supported asset types
    ///
    /// To define a new asset type, one should extend this enumeration with a new index.
    ///
    enum class eAssetType : uint32_t
    {
        UNKNOWN        = 0,
        LASER_ASSEMBLY = 1,
        PHOTODETECTOR  = 2
    };

    /// @brief Main asset value
    ///
    /// To define a new asset type, the structure corresponding to the asset should be added at the end of the
    /// variant definition. Note that the ordering of the types in the variant definition should match the
    /// ordering in the enumeration above
    ///
    using SensorAssetValue = std::variant<std::monostate, sLaserAssetData, sPhotodetectorAssetData>;

    /// @brief Abstract sensor asset
    ///
    /// Holds the asset data that has been parsed from disk.  All file paths inside the asset definition are
    /// assumed to be relative to the provided asset root.
    ///
    struct sSensorAssetData
    {
        std::string mName  = ""; //!< Name for the asset for display purposes
        fs::path mRoot     = ""; //!< Root folder for the asset.
        fs::path mFilePath = ""; //!< Path for the asset, relative to the root.

        SensorAssetValue mValue{}; //!< Parsed asset data.

        /// @brief Default constructor
        sSensorAssetData() = default;

        /// @brief Copy constructor
        sSensorAssetData( const sSensorAssetData & ) = default;

        /// @brief Default destructor
        ~sSensorAssetData() = default;

        /// @brief Retrieve the component stored at the type index corresponding to _Ty
        ///
        /// @tparam _Ty Type to retrieve.
        ///
        /// @return The relevant component.
        template <typename _Ty> _Ty Get() { return std::get<_Ty>( mValue ); }

        /// @brief Retrieve the component stored at the type index corresponding to _Ty (const version)
        ///
        /// @tparam _Ty Type to retrieve.
        ///
        /// @return The relevant component.
        template <typename _Ty> _Ty Get() const { return std::get<_Ty>( mValue ); }

        /// @brief Retrieve the type of the currently stored asset
        ///
        /// @return The type of the currently stored asset.
        ///
        eAssetType Type() { return static_cast<eAssetType>( mValue.index() ); }

        /// @brief Retrieve the type of the currently stored asset (const version)
        ///
        /// @return The type of the currently stored asset.
        ///
        eAssetType Type() const { return static_cast<eAssetType>( mValue.index() ); }
    };

    /// @brief Read an asset from disk
    ///
    /// The asset is read from disk, and parsed into the appropriate `sSensorAssetData` structure. Note that
    /// the individual textures and other external dependencies of the asset are not loaded at this stage.
    ///
    /// @param aAssetRoot Root folder where the asset is located
    /// @param aAssetPath Path of the `asset.yaml` file, relative to the provided root folder
    /// @param aAssetName Name to give the asset, mostly for display purposes.
    ///
    /// @return The parsed component
    ///
    sSensorAssetData ReadAsset( fs::path const &aAssetRoot, fs::path const &aAssetPath, std::string const &aAssetName );

} // namespace LTSE::SensorModel
