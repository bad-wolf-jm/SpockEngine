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
#include "Developer/Core/Optix/Optix7.h"
#include "Developer/Scene/VertexData.h"

// #include "SensorModelDev/Base/KernelComponents.h"

namespace LTSE::SensorModel::Dev
{
    using namespace LTSE::Core;

    struct HitRecord
    {
        float RayID     = 0.0f;
        float Azimuth   = 0.0f;
        float Elevation = 0.0f;
        float Distance  = 0.0f;
        float Intensity = 0.0f;
    };

    struct TriangleMeshSBTData
    {
        math::vec3 color;
        math::vec3 *vertex;
        uint32_t *index;
        uint32_t mVertexOffset;
        uint32_t mIndexOffset;
    };

    struct LaunchParams
    {
        OptixTraversableHandle traversable;
        math::vec3 SensorPosition;
        math::mat3 SensorRotation;
        float *Azimuths;
        float *Elevations;
        float *Intensities;
        VertexData *mVertexBuffer;
        math::uvec3 *mIndexBuffer;

        HitRecord *SamplePoints;
    };

} // namespace LTSE::SensorModel::Dev
