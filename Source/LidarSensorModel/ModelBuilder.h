/// @file   ModelBuilder.cpp
///
/// @brief  Build a sensor model from confitguration stored on disk or in a string
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2022 LeddarTech Inc. All rights reserved.

#pragma once

#include "Core/Memory.h"
#include "SensorModelBase.h"

#include "Components.h"

#include "Serialize/SensorDefinition.h"

#include <fstream>
#include <iostream>

namespace LTSE::SensorModel
{
    /// @brief Build a sensor model from a definition structure
    ///
    /// The required sensor definition is typically obtained by parsing a YAML file using `ReadSensorDefinition`.
    /// This function does the heavy lifting of actually creating and configuring the sensor model.
    ///
    /// @tparam _SensorType Class to instantiate. This should be a subclass of `SensorModelBase`
    ///
    /// @param aDefinition Sensor definition from which to build the model
    ///
    /// @returns A shared reference to the newly created sensor model
    ///
    template <typename _SensorType> Ref<_SensorType> Build( sSensorDefinition const &aDefinition )
    {
        Ref<_SensorType> lNewSensor = New<_SensorType>();

        lNewSensor->mName = aDefinition.mName;

        for( auto &lAsset : aDefinition.Assets )
            lNewSensor->CreateAsset( lAsset.first, lAsset.second );

        for( auto &lComponent : aDefinition.Components )
            lNewSensor->CreateElement( lComponent.first, lComponent.second );

        for( auto &lLayout : aDefinition.Layouts )
            lNewSensor->CreateTileLayout( lLayout.first, lLayout.second );

        for( auto &lTileDefinition : aDefinition.Tiles )
        {
            Entity lTileEntity = lNewSensor->CreateTile( lTileDefinition.first, lTileDefinition.second.mPosition );

            lTileEntity.AddOrReplace<sJoinComponent<sSampler>>( lNewSensor->GetComponentByID( lTileDefinition.second.SamplerComponentID ) );

            for( auto &lFlashDefinition : lTileDefinition.second.Flashes )
            {
                math::vec2 lFlashPosition{ lFlashDefinition.Area.x, lFlashDefinition.Area.y };
                math::vec2 lFlashArea{ lFlashDefinition.Area.z, lFlashDefinition.Area.w };

                auto lNewFlash = lNewSensor->CreateFlash( lTileEntity, "", lFlashPosition, lFlashArea );

                lNewFlash.Adjoin<sLaserAssembly>( lNewSensor->GetComponentByID( lFlashDefinition.LaserDiodeComponentID ) );
                lNewFlash.Adjoin<sPhotoDetector>( lNewSensor->GetAssetByID( lFlashDefinition.PhotodetectorComponentID ) );
            }
        }

        return lNewSensor;
    }

    /// @brief Build a sensor model from a definition file
    ///
    /// This function reads the sensor definition from the provided yaml source file, and generates a new sensor
    /// model from it.
    ///
    /// @tparam _SensorType Class to instantiate. This should be a subclass of `SensorModelBase`
    ///
    /// @param aRoot Root folder for the sensor model. It will be used as the source folder for loading assets.
    /// @param aDefinitionFile Path of the sensor definition.
    ///
    /// @returns A shared reference to the newly created sensor model
    ///
    template <typename _SensorType> Ref<_SensorType> Build( fs::path const &aRoot, fs::path const &aDefinitionFile )
    {
        LTSE::Logging::Info( "Opening sensor configuration file: {}", aRoot.string() );

        sSensorDefinition lSensorDefinition = ReadSensorDefinition( aRoot, aDefinitionFile );
        return Build<_SensorType>( lSensorDefinition );
    }
} // namespace LTSE::SensorModel