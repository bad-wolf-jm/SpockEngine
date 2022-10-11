/// @file   SensorAsset.cpp
///
/// @brief  Implementation file for sensor assets
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2022 LeddarTech Inc. All rights reserved.

#include "SensorAsset.h"
#include <fmt/core.h>

namespace LTSE::SensorModel
{
    namespace
    {
        static ConfigurationNode ReadAssetInfo( fs::path const &aAssetRoot, fs::path const &aAssetPath, std::string const &aAssetName, sSensorAssetData &aNewAsset )
        {
            ConfigurationReader lRootReader( ( aAssetRoot / aAssetPath ) );
            ConfigurationNode lRootNode = lRootReader.GetRoot();

            std::string lAssetType = lRootNode["asset.type"].As<std::string>( "UNKNOWN" );

            aNewAsset.mName     = aAssetName.empty() ? lRootNode["asset.name"].As<std::string>( aAssetName ) : aAssetName;
            aNewAsset.mFilePath = aAssetPath;
            aNewAsset.mRoot     = aAssetRoot;

            return lRootNode;
        }

        static void ReadPhotodetectorAsset( ConfigurationNode &aAssetData, fs::path const &aAssetRoot, fs::path const &aAssetPath, std::string const &aAssetName,
                                            sSensorAssetData &aNewAsset )
        {
            auto lAssetRoot = aAssetPath.parent_path();

            sPhotodetectorAssetData lAssetData{};

            lAssetData.mStaticNoiseData      = lAssetRoot / fs::path( aAssetData["static_noise"].As<std::string>( "" ) );
            lAssetData.mElectronicXtalkData  = lAssetRoot / fs::path( aAssetData["extalk_matrix"].As<std::string>( "" ) );

            aAssetData["cells"].ForEach(
                [&]( ConfigurationNode &lCell )
                {
                    sPhotodetectorAssetData::CellData lCellData{};
                    lCellData.mId               = lCell["id"].As<uint32_t>( std::numeric_limits<uint32_t>::max() );
                    lCellData.mPosition         = lCell["position"].Vec( { "x", "y", "z", "w" }, { 0.0f, 0.0f, 0.0f, 0.0f } );
                    lCellData.mGain             = lCell["gain"].Vec( { "x", "y", "z", "w" }, { 0.0f, 0.0f, 0.0f, 0.0f } );
                    lCellData.mBaseline         = lCell["baseline"].Vec( { "x", "y", "z", "w" }, { 0.0f, 0.0f, 0.0f, 0.0f } );
                    lCellData.mStaticNoiseShift = lCell["static_noise_shift"].Vec( { "x", "y", "z", "w" }, { 0.0f, 0.0f, 0.0f, 0.0f } );

                    lAssetData.mCells.push_back( lCellData );
                } );

            aNewAsset.mValue = lAssetData;
        }

        static void ReadLaserAssemblyAsset( ConfigurationNode &aAssetData, fs::path const &aAssetRoot, fs::path const &aAssetPath, std::string const &aAssetName,
                                            sSensorAssetData &aNewAsset )
        {
            auto lAssetRoot = aAssetPath.parent_path();

            sLaserAssetData lAssetData{};

            lAssetData.mWaveformTemplate = lAssetRoot / fs::path( aAssetData["waveform_template"].As<std::string>( "" ) );
            lAssetData.mDiffuser         = lAssetRoot / fs::path( aAssetData["diffuser_data"].As<std::string>( "" ) );
            lAssetData.mTimebaseDelay    = aAssetData["timebase_delay"].Vec( { "x", "y", "z", "w" }, { 0.0f, 0.0f, 0.0f, 0.0f } );
            lAssetData.mFlashTime        = aAssetData["flash_time"].Vec( { "x", "y", "z", "w" }, { 0.0f, 0.0f, 0.0f, 0.0f } );

            aNewAsset.mValue = lAssetData;
        }

    } // namespace

    sSensorAssetData ReadAsset( fs::path const &aAssetRoot, fs::path const &aAssetPath, std::string const &aAssetName )
    {
        sSensorAssetData lNewAsset{};

        ConfigurationNode lRootNode = ReadAssetInfo( aAssetRoot, aAssetPath, aAssetName, lNewAsset );

        ConfigurationNode &lAssetData = lRootNode["asset.data"];
        std::string lAssetType        = lRootNode["asset.type"].As<std::string>( "UNKNOWN" );

        if( lAssetType == "laser_assembly" )
            ReadLaserAssemblyAsset( lAssetData, aAssetRoot, aAssetPath, aAssetName, lNewAsset );
        else if( lAssetType == "photodetector" )
            ReadPhotodetectorAsset( lAssetData, aAssetRoot, aAssetPath, aAssetName, lNewAsset );
        else
            throw std::runtime_error( "Unrecognized asset type" );

        return lNewAsset;
    }

} // namespace LTSE::SensorModel
