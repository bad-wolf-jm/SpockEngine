/// @file   StartRayStructBuilder.cpp
///
/// @brief  Implementation file for converting LT data into dSpace start ray structures
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#include "StartRayStructBuilder.h"
#include "StartRayStructBuilderKernels.h"

#include "Cuda/MultiTensor.h"

#include "LidarSensorConfig.h"
#include "TensorOps/ScalarTypes.h"

namespace LTSE::dSpaceCompatibility
{
    using namespace LTSE::Core;
    using namespace LTSE::Cuda;
    using namespace LTSE::TensorOps;

    void sBuildStartRayStructuresController::Run()
    {
        auto &lOutputTensor = Get<sMultiTensorComponent>().mValue;
        auto &lOperandData  = Get<sBuildStartRayStructures>();

        auto &lAzimuths    = lOperandData.mAzimuths.Get<sMultiTensorComponent>().mValue;
        auto &lElevations  = lOperandData.mElevations.Get<sMultiTensorComponent>().mValue;
        auto &lIntensities = lOperandData.mIntensities.Get<sMultiTensorComponent>().mValue;
        auto &lTimestamps  = lOperandData.mTimestamps.Get<sMultiTensorComponent>().mValue;

        BuildStartRayStructureOp( lOutputTensor, lAzimuths, lElevations, lIntensities, lTimestamps );
    }

    OpNode BuildStartRayStructures( Scope &Scope, OpNode aAzimuths, OpNode aElevations, OpNode aIntensities, OpNode aTimestamps )
    {
        auto lNewEntity = Scope.CreateNode();
        auto &lType     = lNewEntity.Add<sTypeComponent>();
        lType.mValue    = eScalarType::FLOAT32;

        auto &lOperandData        = lNewEntity.Add<sBuildStartRayStructures>();
        lOperandData.mAzimuths    = aAzimuths;
        lOperandData.mElevations  = aElevations;
        lOperandData.mIntensities = aIntensities;
        lOperandData.mTimestamps  = aTimestamps;

        auto &lValue            = lNewEntity.Add<sMultiTensorComponent>();
        auto lOutputTensorShape = sTensorShape( aAzimuths.Get<sMultiTensorComponent>().mValue.Shape().mShape, sizeof( OptixSensorLidar::StartRay ) );
        lValue.mValue           = MultiTensor( Scope.mPool, lOutputTensorShape );

        lNewEntity.Add<sGraphOperationComponent>().Bind<sBuildStartRayStructuresController>();
        return lNewEntity;
    }

} // namespace LTSE::dSpaceCompatibility