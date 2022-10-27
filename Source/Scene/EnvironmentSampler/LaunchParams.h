// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "Core/Math/Types.h"
#include "Core/Optix/Optix7.h"
#include "Scene/VertexData.h"

// #include "SensorModelDev/Base/KernelComponents.h"

namespace LTSE::SensorModel::Dev
{
    using namespace LTSE::Core;

    struct sHitRecord
    {
        float mRayID     = 0.0f;
        float mAzimuth   = 0.0f;
        float mElevation = 0.0f;
        float mDistance  = 0.0f;
        float mIntensity = 0.0f;
    };

    struct TriangleMeshSBTData
    {
        math::vec3  mColor;
        math::vec3 *mVertex;
        uint32_t   *mIndex;
        uint32_t    mVertexOffset;
        uint32_t    mIndexOffset;
    };

    struct LaunchParams
    {
        OptixTraversableHandle mTraversable;
        math::vec3             mSensorPosition;
        math::mat3             mSensorRotation;
        float                 *mAzimuths;
        float                 *mElevations;
        float                 *mIntensities;
        VertexData            *mVertexBuffer;
        math::uvec3           *mIndexBuffer;
        sHitRecord            *mSamplePoints;
    };

} // namespace LTSE::SensorModel::Dev
