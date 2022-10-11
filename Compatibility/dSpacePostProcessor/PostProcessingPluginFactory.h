/// @file   PostProcessingPluginFactory.h
///
/// @brief  Construct and initialize a new post processing plugin
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#pragma once

#include "LidarSensorConfig.h"

/// @brief Create a new post processing plugin of the given type
///
/// @tparam _PPType class to instanciate.
///
template <typename _PPType> IPostProcessor *New()
{
    _PPType *lProcessor = new _PPType();
    lProcessor->Initialize();

    return reinterpret_cast<IPostProcessor *>( lProcessor );
}
