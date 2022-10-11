/// @file   StartRayStructBuilderKernels.h
///
/// @brief  Kernel interface definition.
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#pragma once

#include "Cuda/MemoryPool.h"
#include "Cuda/MultiTensor.h"

#include "TensorOps/ScalarTypes.h"

namespace LTSE::dSpaceCompatibility
{
    using MultiTensor = LTSE::Cuda::MultiTensor;

    /// @brief Build start ray structures based on the values of the individual components.
    ///
    /// All parameters should be multitensors of the same shape
    ///
    /// @param aOut Output tensor.
    /// @param aAzimuths Azimuth values.
    /// @param aElevations Elevation values.
    /// @param aIntensities Laser intensity values.
    /// @param aTimestamps Timestaml values.
    ///
    void BuildStartRayStructureOp( MultiTensor &aOut, MultiTensor &aAzimuths, MultiTensor &aElevations, MultiTensor &aIntensities, MultiTensor &aTimestamps );

} // namespace LTSE::dSpaceCompatibility
