#pragma once
/** @file RayGenerator.h
 *
 * @brief FILE
 */

#include <vector>

#include "Core/Math/Types.h"

#include "Core/Cuda/CudaBuffer.h"
#include "Core/Cuda/ExternalMemory.h"
#include "Core/Cuda/MultiTensor.h"

#include "Scene/ParticleData.h"

namespace LTSE::SensorModel::Dev
{

    using namespace LTSE::Cuda;

    /**
     *  @struct LidarSamplePoint
     */
    struct LidarCartesianSamplePoint
    {
        uint32_t FlashID     = 0xffffffff;
        int Type             = -1;
        math::vec3 Direction = { 0.0f, 0.0f, 0.0f };
        float Distance       = 0.0f;
        float Intensity      = 0.0f;

        LidarCartesianSamplePoint()                                    = default;
        LidarCartesianSamplePoint( const LidarCartesianSamplePoint & ) = default;
    };

    struct FlashVisualizationData
    {
        math::vec2 FlashPosition = { 0.0f, 0.0f };
        math::vec2 FlashExtent   = { 0.0f, 0.0f };
        math::vec3 Color         = { 1.0f, 0.2f, 0.8f };
        float MinValue           = 0.0f;
        float MaxValue           = 1.0f;
    };

    class PointCloudVisualizer
    {
      public:
        bool HighlightFlashFOV   = false;
        bool HeatmapColoring     = true;
        float HeatmapColorOffset = 120.0f;
        bool LogScale            = true;
        bool InvertZAxis         = false;
        float PointDensity       = 0.5;
        float Resolution         = 0.1;

        PointCloudVisualizer()  = default;
        ~PointCloudVisualizer() = default;

        void Visualize( math::mat4 a_PointCloudTransform, MultiTensor &a_LidarReturnPoints, GPUExternalMemory &a_Particles );
    };

} // namespace LTSE::SensorModel::Dev
