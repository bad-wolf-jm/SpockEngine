/// @file   ReturnDataDestructerKernels.h
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

namespace LTSE::SensorModel
{
    using MultiTensor = LTSE::Cuda::MultiTensor;
    using MemoryBuffer = LTSE::Cuda::MemoryBuffer;

    /// @brief Retrieve the radiance from the returns
    ///
    /// All parameters should be multitensors of the same shape
    ///
    /// @param aOut     Output tensor.
    /// @param aReturns OpNode holding the lidar returns from the simulated environment.
    ///
    void ExtractReflectivityOp( MultiTensor &aOut, MultiTensor &aReturns );

    /// @brief Retrieve the distance from the returns
    ///
    /// All parameters should be multitensors of the same shape
    ///
    /// @param aOut     Output tensor.
    /// @param aReturns OpNode holding the lidar returns from the simulated environment.
    ///
    void ExtractDistanceOp( MultiTensor &aOut, MultiTensor &aReturns );

} // namespace LTSE::SensorModel