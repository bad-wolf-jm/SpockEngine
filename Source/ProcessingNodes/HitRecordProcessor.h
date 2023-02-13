/// @file   ReturnDataDestructer.cpp
///
/// @brief  Defines a new tensor node for converting Aurelion's LidarRayTracingPoint data into SE nodes.
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#pragma once

#include "Core/Math/Types.h"

#include "Core/EntityCollection/Collection.h"

#include "Core/CUDA/Array/MemoryPool.h"
#include "Core/CUDA/Array/MultiTensor.h"

#include "TensorOps/NodeComponents.h"
#include "TensorOps/Scope.h"

namespace SE::SensorModel
{

    using sGraphOperationController = SE::TensorOps::sGraphOperationController;
    using OpNode                    = SE::TensorOps::OpNode;
    using Scope                     = SE::TensorOps::Scope;
    using MultiTensor               = SE::Cuda::MultiTensor;
    using MemoryBuffer              = SE::Cuda::MemoryBuffer;
    using sTensorShape              = SE::Cuda::sTensorShape;

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

} // namespace SE::SensorModel
