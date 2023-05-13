#pragma once

#include "Core/Math/Types.h"
#include "Core/Memory.h"
#include "Core/Types.h"

#include "Graphics/Vulkan/ARenderContext.h"
#include "Graphics/Vulkan/DescriptorSet.h"

#include "Graphics/Vulkan/VkGraphicsPipeline.h"
#include "Graphics/Vulkan/VkGraphicContext.h"

#include "Graphics/Vulkan/VkAbstractRenderPass.h"

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
        VisualHelperRenderer( Ref<VkGraphicContext> aGraphicContext, Ref<sVkAbstractRenderPassObject> aRenderPass );

        ~VisualHelperRenderer() = default;

        void Render( math::mat4 aTransform, ArrowMeshData &aArrow, math::vec3 aColor, ARenderContext &aRenderContext );
        void Render( math::mat4 aTransform, ConeMeshData &aArrow, math::vec3 aColor, ARenderContext &aRenderContext );
        void Render( math::mat4 aTransform, CircleMeshData &aArrow, math::vec3 aColor, ARenderContext &aRenderContext );
        void Render( math::mat4 aTransform, CubeMeshData &aArrow, math::vec3 aColor, ARenderContext &aRenderContext );
        void Render( math::mat4 aTransform, PyramidMeshData &aArrow, math::vec3 aColor, ARenderContext &aRenderContext );
        void Render( math::mat4 aTransform, SurfaceMeshData &aArrow, math::vec3 aColor, ARenderContext &aRenderContext );

        void Render( math::mat4 aTransform, AxesComponent &aAxesComponent, ARenderContext &aRenderContext );
        void Render( math::mat4 aTransform, PointLightHelperComponent &aAxesComponent, ARenderContext &aRenderContext );
        void Render( math::mat4 aTransform, DirectionalLightHelperComponent &aAxesComponent, ARenderContext &aRenderContext );
        void Render( math::mat4 aTransform, SpotlightHelperComponent &aAxesComponent, ARenderContext &aRenderContext );
        void Render( math::mat4 aTransform, FieldOfViewHelperComponent &aAxesComponent, ARenderContext &aRenderContext );
        void Render( math::mat4 aTransform, CameraHelperComponent &aAxesComponent, ARenderContext &aRenderContext );

      private:
        Ref<VkGraphicContext> mGraphicContext;
        Ref<GraphicsPipeline> m_RenderPipeline = nullptr;

        Ref<VisualHelperMeshRenderer> m_MeshRenderer             = nullptr;
        Ref<VisualHelperLineRenderer> m_VisualHelperLineRenderer = nullptr;
    };

} // namespace SE::Graphics
