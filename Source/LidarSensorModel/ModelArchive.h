/// @file   ModelArchive.h
///
/// @brief  Save a sensor model to disk
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2022 LeddarTech Inc. All rights reserved.

#pragma once

#include "Core/Memory.h"
#include "SensorModelBase.h"

#include "Components.h"

#include "Serialize/SensorDefinition.h"

namespace LTSE::SensorModel
{
    /// @brief Save the provided sensor model to a definition file
    ///
    /// The definition file can then be loaded using `Build` at a later time. Note that for this function to work
    /// properly, sensor elements and assets should reference one another properly, and asset references to on-disk
    /// resources must be valid. Otherwise, though the function will not fail, the resulting sensor configuration
    /// will not be readable.
    ///
    /// @param aSensorDefinition Sensor model definition to save.
    /// @param aRoot Root folder where to save the model
    /// @param aModelFilePath Name of the definition file to generate.
    ///
    void Save( Ref<SensorModelBase> aSensorDefinition, fs::path const &aRoot, fs::path const &aModelFilePath );
} // namespace LTSE::SensorModel
