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
        CoordinateGridRenderer( Ref<IGraphicContext> aGraphicContext, Ref<IRenderContext> aRenderContext );
        ~CoordinateGridRenderer() = default;

        void Render( math::mat4 aProjection, math::mat4 aView, Ref<IRenderContext> aRenderContext );

      private:
        Ref<IDescriptorSetLayout> mPipelineLayout     = nullptr;
        Ref<IGraphicContext>      mGraphicContext    = nullptr;
        Ref<IGraphicBuffer>       mCameraBuffer      = nullptr;
        Ref<IDescriptorSet>       mCameraDescriptors = nullptr;
        Ref<IGraphicsPipeline>    mPipeline          = nullptr;
    };

} // namespace SE::Core