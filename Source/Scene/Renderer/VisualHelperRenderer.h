#pragma once

#include "Core/Math/Types.h"
#include "Core/Memory.h"
#include "Core/Types.h"

// #include "Graphics/Vulkan/VkRenderContext.h"
// #include "Graphics/Vulkan/DescriptorSet.h"
// #include "Graphics/Vulkan/VkGraphicsPipeline.h"
// #include "Graphics/Vulkan/IGraphicContext.h"
// #include "Graphics/Vulkan/VkRenderPass.h"
#include "Graphics/API.h"


#include "Scene/Components/VisualHelpers.h"
#include "Scene/VertexData.h"

#include "VisualHelperLineRenderer.h"
#include "VisualHelperMeshRenderer.h"

namespace SE::Graphics
{

    using namespace math;
    using namespace SE::Core::EntityComponentSystem::Components;

    class VisualHelperRenderer
    {
      public:
        math::mat4 View       = math::mat4( 1.0f );
        math::mat4 Projection = math::mat4( 1.0f );

        VisualHelperRenderer() = default;
        VisualHelperRenderer( Ref<IGraphicContext> aGraphicContext, Ref<sVkAbstractRenderPassObject> aRenderPass );

        ~VisualHelperRenderer() = default;

        void Render( math::mat4 aTransform, ArrowMeshData &aArrow, math::vec3 aColor, VkRenderContext &aRenderContext );
        void Render( math::mat4 aTransform, ConeMeshData &aArrow, math::vec3 aColor, VkRenderContext &aRenderContext );
        void Render( math::mat4 aTransform, CircleMeshData &aArrow, math::vec3 aColor, VkRenderContext &aRenderContext );
        void Render( math::mat4 aTransform, CubeMeshData &aArrow, math::vec3 aColor, VkRenderContext &aRenderContext );
        void Render( math::mat4 aTransform, PyramidMeshData &aArrow, math::vec3 aColor, VkRenderContext &aRenderContext );
        void Render( math::mat4 aTransform, SurfaceMeshData &aArrow, math::vec3 aColor, VkRenderContext &aRenderContext );

        void Render( math::mat4 aTransform, AxesComponent &aAxesComponent, VkRenderContext &aRenderContext );
        void Render( math::mat4 aTransform, PointLightHelperComponent &aAxesComponent, VkRenderContext &aRenderContext );
        void Render( math::mat4 aTransform, DirectionalLightHelperComponent &aAxesComponent, VkRenderContext &aRenderContext );
        void Render( math::mat4 aTransform, SpotlightHelperComponent &aAxesComponent, VkRenderContext &aRenderContext );
        void Render( math::mat4 aTransform, FieldOfViewHelperComponent &aAxesComponent, VkRenderContext &aRenderContext );
        void Render( math::mat4 aTransform, CameraHelperComponent &aAxesComponent, VkRenderContext &aRenderContext );

      private:
        Ref<IGraphicContext> mGraphicContext;
        Ref<GraphicsPipeline> m_RenderPipeline = nullptr;

        Ref<VisualHelperMeshRenderer> m_MeshRenderer             = nullptr;
        Ref<VisualHelperLineRenderer> m_VisualHelperLineRenderer = nullptr;
    };

} // namespace SE::Graphics
