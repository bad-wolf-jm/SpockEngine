/// @file   ReturnDataDestructer.cpp
///
/// @brief  Defines a new tensor node for converting Aurelion's LidarRayTracingPoint data into LTSE nodes.
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#pragma once

#include "Core/Math/Types.h"

#include "Core/EntityRegistry/Registry.h"

#include "Cuda/MemoryPool.h"
#include "Cuda/MultiTensor.h"

#include "TensorOps/NodeComponents.h"
#include "TensorOps/Scope.h"

namespace LTSE::SensorModel
{

    using sGraphOperationController = LTSE::TensorOps::sGraphOperationController;
    using OpNode                    = LTSE::TensorOps::OpNode;
    using Scope                     = LTSE::TensorOps::Scope;
    using MultiTensor               = LTSE::Cuda::MultiTensor;
    using MemoryBuffer              = LTSE::Cuda::MemoryBuffer;
    using sTensorShape              = LTSE::Cuda::sTensorShape;

    /// @struct sDestructHitRecordStructures
    ///
    /// Retrieve individual components from structures returned by the simulated environment
    ///
    struct sDestructHitRecordStructures
    {
        MultiTensor mReturnData{}; //!< Data from the simulated environment.

        sDestructHitRecordStructures()                                       = default;
        sDestructHitRecordStructures( const sDestructHitRecordStructures & ) = default;
    };

    /// @struct sExtractReflectivityController
    ///
    /// Controller structure for sDestructHitRecordStructures nodes. This controller retrieved the
    /// reflected intensity.
    ///
    struct sExtractReflectivityController : public sGraphOperationController
    {
        void Run();
    };

    /// @struct sExtractReflectivityController
    ///
    /// Controller structure for sDestructHitRecordStructures nodes. This controller retrieved the
    /// distance.
    ///
    struct sExtractDistanceController : public sGraphOperationController
    {
        void Run();
    };

    /// @brief Retrieve the distances from the detections returned by the simulated environment
    ///
    ///
    /// @param aScope Computation scope.
    /// @param aShape Shape of the original sampling array.
    /// @param aReturnData Data from the simulated environment.
    ///
    OpNode RetrieveDistance( Scope &aScope, MultiTensor &aReturnData );

    /// @brief Retrieve the distances from the detections returned by the simulated environment
    ///
    ///
    /// @param aScope Computation scope.
    /// @param aShape Shape of the original sampling array.
    /// @param aReturnData Data from the simulated environment.
    ///
    OpNode RetrieveIntensities( Scope &aScope, MultiTensor &aReturnData );

} // namespace LTSE::SensorModel
