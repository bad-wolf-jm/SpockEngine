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

#include "Core/Cuda/MemoryPool.h"
#include "Core/Cuda/MultiTensor.h"

#include "TensorOps/NodeComponents.h"
#include "TensorOps/Scope.h"

namespace LTSE::dSpaceCompatibility
{

    using sGraphOperationController = LTSE::TensorOps::sGraphOperationController;
    using OpNode                    = LTSE::TensorOps::OpNode;
    using Scope                     = LTSE::TensorOps::Scope;
    using MultiTensor               = LTSE::Cuda::MultiTensor;
    using MemoryBuffer              = LTSE::Cuda::MemoryBuffer;
    using sTensorShape              = LTSE::Cuda::sTensorShape;

    /// @struct sDestructLidarReturnStructures
    ///
    /// Retrieve individual components from structures returned by the simulated environment
    ///
    struct sDestructLidarReturnStructures
    {
        MemoryBuffer mReturnData{}; //!< Data from the simulated environment.

        sDestructLidarReturnStructures()                                         = default;
        sDestructLidarReturnStructures( const sDestructLidarReturnStructures & ) = default;
    };

    /// @struct sExtractReflectivityController
    ///
    /// Controller structure for sDestructLidarReturnStructures nodes. This controller retrieved the
    /// reflected intensity.
    ///
    struct sExtractReflectivityController : public sGraphOperationController
    {
        void Run();
    };

    /// @struct sExtractReflectivityController
    ///
    /// Controller structure for sDestructLidarReturnStructures nodes. This controller retrieved the
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
    OpNode RetrieveDistance( Scope &aScope, sTensorShape &aShape, MemoryBuffer &aReturnData );

    /// @brief Retrieve the distances from the detections returned by the simulated environment
    ///
    ///
    /// @param aScope Computation scope.
    /// @param aShape Shape of the original sampling array.
    /// @param aReturnData Data from the simulated environment.
    ///
    OpNode RetrieveIntensities( Scope &aScope, sTensorShape &aShape, MemoryBuffer &aReturnData );

} // namespace LTSE::dSpaceCompatibility
