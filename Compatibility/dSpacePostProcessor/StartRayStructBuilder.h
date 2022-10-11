/// @file   StartRayStructBuilder.cpp
///
/// @brief  Defines a new tensor node for converting the EnvironmentSampler data into dSpace start ray structures.
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#pragma once

#include "Core/Math/Types.h"

#include "Core/EntityRegistry/Registry.h"

#include "TensorOps/NodeComponents.h"
#include "TensorOps/Scope.h"

namespace LTSE::dSpaceCompatibility
{

    using sGraphOperationController = LTSE::TensorOps::sGraphOperationController;
    using OpNode                    = LTSE::TensorOps::OpNode;
    using Scope                     = LTSE::TensorOps::Scope;

    /// @struct sBuildStartRayStructures
    ///
    /// Build start ray structures based on the values of the individual components.
    ///
    struct sBuildStartRayStructures
    {
        OpNode mAzimuths{};    //!< Azimuth values
        OpNode mElevations{};  //!< Elevation values
        OpNode mIntensities{}; //!< Laser intensity values
        OpNode mTimestamps{};  //!< Timestamp

        sBuildStartRayStructures()                                   = default;
        sBuildStartRayStructures( const sBuildStartRayStructures & ) = default;
    };

    /// @struct sBuildStartRayStructuresController
    ///
    /// Controller structure for sBuildStartRayStructures nodes.
    ///
    struct sBuildStartRayStructuresController : public sGraphOperationController
    {
        void Run();
    };

    /// @brief Build start ray structures based on the values of the indivisual components.
    ///
    /// All parameters should be multitensors of the same shape
    ///
    /// @param aScope Computation scope.
    /// @param aAzimuths Azimuth values.
    /// @param aElevations Elevation values.
    /// @param aIntensities Laser intensity values.
    /// @param aTimestamps Timestaml values.
    ///
    OpNode BuildStartRayStructures( Scope &aScope, OpNode aAzimuths, OpNode aElevations, OpNode aIntensities, OpNode aTimestamps );

} // namespace LTSE::dSpaceCompatibility
