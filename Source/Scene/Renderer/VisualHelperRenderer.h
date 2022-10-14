#pragma once

#include "Core/Math/Types.h"
#include "Core/Types.h"
#include "Core/Memory.h"

#include "Core/GraphicContext//Buffer.h"
#include "Core/GraphicContext//DescriptorSet.h"
#include "Core/GraphicContext//RenderContext.h"

#include "Core/GraphicContext//GraphicContext.h"
#include "Core/GraphicContext//GraphicsPipeline.h"
#include "Core/Vulkan/VkRenderPass.h"
#include "Core/GraphicContext//Texture2D.h"

#include "Scene/Components/VisualHelpers.h"
#include "Scene/VertexData.h"

#include "VisualHelperLineRenderer.h"
#include "VisualHelperMeshRenderer.h"

namespace LTSE::Graphics
{

    using namespace math;
    using namespace LTSE::Core::EntityComponentSystem::Components;

    class VisualHelperRenderer
    {
      public:
        math::mat4 View       = math::mat4( 1.0f );
        math::mat4 Projection = math::mat4( 1.0f );

        VisualHelperRenderer() = default;
        VisualHelperRenderer( GraphicContext &a_GraphicContext, Ref<LTSE::Graphics::Internal::sVkRenderPassObject>  a_RenderPass );

        ~VisualHelperRenderer() = default;

        void Render( math::mat4 a_Transform, ArrowMeshData &a_Arrow, math::vec3 a_Color, RenderContext &aRenderContext );
        void Render( math::mat4 a_Transform, ConeMeshData &a_Arrow, math::vec3 a_Color, RenderContext &aRenderContext );
        void Render( math::mat4 a_Transform, CircleMeshData &a_Arrow, math::vec3 a_Color, RenderContext &aRenderContext );
        void Render( math::mat4 a_Transform, CubeMeshData &a_Arrow, math::vec3 a_Color, RenderContext &aRenderContext );
        void Render( math::mat4 a_Transform, PyramidMeshData &a_Arrow, math::vec3 a_Color, RenderContext &aRenderContext );
        void Render( math::mat4 a_Transform, SurfaceMeshData &a_Arrow, math::vec3 a_Color, RenderContext &aRenderContext );

        void Render( math::mat4 a_Transform, AxesComponent &a_AxesComponent, RenderContext &aRenderContext );
        void Render( math::mat4 a_Transform, PointLightHelperComponent &a_AxesComponent, RenderContext &aRenderContext );
        void Render( math::mat4 a_Transform, DirectionalLightHelperComponent &a_AxesComponent, RenderContext &aRenderContext );
        void Render( math::mat4 a_Transform, SpotlightHelperComponent &a_AxesComponent, RenderContext &aRenderContext );
        void Render( math::mat4 a_Transform, FieldOfViewHelperComponent &a_AxesComponent, RenderContext &aRenderContext );
        void Render( math::mat4 a_Transform, CameraHelperComponent &a_AxesComponent, RenderContext &aRenderContext );

      private:
        GraphicContext mGraphicContext;
        Ref<GraphicsPipeline> m_RenderPipeline = nullptr;

        Ref<VisualHelperMeshRenderer> m_MeshRenderer             = nullptr;
        Ref<VisualHelperLineRenderer> m_VisualHelperLineRenderer = nullptr;
    };

} // namespace LTSE::Graphics
