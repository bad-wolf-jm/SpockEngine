/// @file   SensorComponents.cpp
///
/// @brief  Implementation file for sensor components
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2022 LeddarTech Inc. All rights reserved.

#include "SensorComponents.h"
#include <fmt/core.h>

namespace LTSE::SensorModel
{
    namespace
    {
        static void ReadSamplerComponent( ConfigurationNode const &aAssetData, std::string const &aAssetName, sSensorComponentData &aNewAsset )
        {
            sSamplerComponentData lComponentData{};

            lComponentData.mLength    = aAssetData["length"].As<uint32_t>( 0 );
            lComponentData.mFrequency = aAssetData["frequency"].As<float>( 1.0f );

            aNewAsset.mValue = lComponentData;
        }
    } // namespace

    sSensorComponentData ReadComponent( ConfigurationNode const &aAssetData )
    {
        sSensorComponentData lNewComponent{};

        ConfigurationNode &lAssetData = aAssetData["data"];
        std::string lAssetName        = aAssetData["name"].As<std::string>( "" );
        std::string lAssetType        = aAssetData["type"].As<std::string>( "" );

        lNewComponent.mName = lAssetName;

        if( lAssetType == "sampler" )
            ReadSamplerComponent( lAssetData, lAssetName, lNewComponent );
        else
            throw std::runtime_error( "Unrecognized asset type" );

        return lNewComponent;
    }

} // namespace LTSE::SensorModel