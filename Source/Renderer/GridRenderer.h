#pragma once

#include "Core/Memory.h"

#include "Graphics/API.h"

#include "Scene/VertexData.h"

namespace SE::Core
{

    using namespace SE::Graphics;

    struct CameraViewUniforms
    {
        math::mat4 View;
        math::mat4 Projection;
    };

    class CoordinateGridRenderer
    {
      public:
        CoordinateGridRenderer( ref_t<IGraphicContext> aGraphicContext, ref_t<IRenderContext> aRenderContext );
        ~CoordinateGridRenderer() = default;

        void Render( math::mat4 aProjection, math::mat4 aView, ref_t<IRenderContext> aRenderContext );

      private:
        ref_t<IDescriptorSetLayout> mPipelineLayout     = nullptr;
        ref_t<IGraphicContext>      mGraphicContext    = nullptr;
        ref_t<IGraphicBuffer>       mCameraBuffer      = nullptr;
        ref_t<IDescriptorSet>       mCameraDescriptors = nullptr;
        ref_t<IGraphicsPipeline>    mPipeline          = nullptr;
    };

} // namespace SE::Core