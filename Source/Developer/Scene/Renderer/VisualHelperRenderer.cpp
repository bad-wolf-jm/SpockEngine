#include "VisualHelperRenderer.h"

#include "Developer/Scene/Primitives/Arrow.h"

namespace LTSE::Graphics
{

    using namespace math::literals;

    VisualHelperRenderer::VisualHelperRenderer( GraphicContext &a_GraphicContext, Ref<LTSE::Graphics::Internal::sVkRenderPassObject> a_RenderPass )
    {
        VisualHelperMeshRendererCreateInfo l_CreateInfo{};
        l_CreateInfo.RenderPass = a_RenderPass;
        m_MeshRenderer          = LTSE::Core::New<VisualHelperMeshRenderer>( a_GraphicContext, l_CreateInfo );

        VisualHelperLineRendererCreateInfo l_GizmoCreateInfo{};
        l_GizmoCreateInfo.LineWidth  = 2.0;
        l_GizmoCreateInfo.RenderPass = a_RenderPass;
        m_VisualHelperLineRenderer   = LTSE::Core::New<VisualHelperLineRenderer>( a_GraphicContext, l_GizmoCreateInfo );
    }

    void VisualHelperRenderer::Render( math::mat4 a_Transform, ArrowMeshData &a_Arrow, math::vec3 a_Color, RenderContext &aRenderContext )
    {
        m_MeshRenderer->Render( a_Transform, View, Projection, a_Color, a_Arrow.Mesh.Vertices, a_Arrow.Mesh.Indices, aRenderContext );
    }

    void VisualHelperRenderer::Render( math::mat4 a_Transform, ConeMeshData &a_Cone, math::vec3 a_Color, RenderContext &aRenderContext )
    {
        m_VisualHelperLineRenderer->Render( a_Transform, View, Projection, a_Color, a_Cone.Mesh.Vertices, a_Cone.Mesh.Indices, aRenderContext );
    }

    void VisualHelperRenderer::Render( math::mat4 a_Transform, CircleMeshData &a_Circle, math::vec3 a_Color, RenderContext &aRenderContext )
    {
        m_VisualHelperLineRenderer->Render( a_Transform, View, Projection, a_Color, a_Circle.Mesh.Vertices, a_Circle.Mesh.Indices, aRenderContext );
    }

    void VisualHelperRenderer::Render( math::mat4 a_Transform, CubeMeshData &a_Cube, math::vec3 a_Color, RenderContext &aRenderContext )
    {
        m_MeshRenderer->Render( a_Transform, View, Projection, a_Color, a_Cube.Mesh.Vertices, a_Cube.Mesh.Indices, aRenderContext );
    }

    void VisualHelperRenderer::Render( math::mat4 a_Transform, PyramidMeshData &a_Pyramid, math::vec3 a_Color, RenderContext &aRenderContext )
    {
        m_VisualHelperLineRenderer->Render( a_Transform, View, Projection, a_Color, a_Pyramid.Mesh.Vertices, a_Pyramid.Mesh.Indices, aRenderContext );
    }

    void VisualHelperRenderer::Render( math::mat4 a_Transform, SurfaceMeshData &a_Surface, math::vec3 a_Color, RenderContext &aRenderContext )
    {
        m_MeshRenderer->Render( a_Transform, View, Projection, a_Color, a_Surface.Mesh.Vertices, a_Surface.Mesh.Indices, aRenderContext );
    }

    void VisualHelperRenderer::Render( math::mat4 a_Transform, AxesComponent &a_AxesComponent, RenderContext &aRenderContext )
    {
        Render( a_Transform * math::Rotation( -90.0_degf, math::vec3{ 0.0f, 0.0f, 1.0f } ), a_AxesComponent.AxisArrow, a_AxesComponent.XAxisColor, aRenderContext );
        Render( a_Transform, a_AxesComponent.AxisArrow, a_AxesComponent.YAxisColor, aRenderContext );
        Render( a_Transform * math::Rotation( -90.0_degf, math::vec3{ 1.0f, 0.0f, 0.0f } ), a_AxesComponent.AxisArrow, a_AxesComponent.ZAxisColor, aRenderContext );
        Render( a_Transform * math::Scale( math::mat4( 1.0f ), math::vec3{ 0.05f, 0.05f, 0.05f } ), a_AxesComponent.Origin, a_AxesComponent.OriginColor, aRenderContext );
    }

    void VisualHelperRenderer::Render( math::mat4 a_Transform, PointLightHelperComponent &a_PointLightHelperComponent, RenderContext &aRenderContext )
    {
        math::mat4 l_Transform = a_Transform * math::Translation( a_PointLightHelperComponent.LightData.Position );

        Render( l_Transform, a_PointLightHelperComponent.AxisCircle, a_PointLightHelperComponent.LightData.Color, aRenderContext );
        Render( l_Transform * math::Rotation( 90.0_degf, math::vec3{ 1.0f, 0.0f, 0.0f } ), a_PointLightHelperComponent.AxisCircle, a_PointLightHelperComponent.LightData.Color,
                aRenderContext );
        Render( l_Transform * math::Rotation( 90.0_degf, math::vec3{ 0.0f, 1.0f, 0.0f } ), a_PointLightHelperComponent.AxisCircle, a_PointLightHelperComponent.LightData.Color,
                aRenderContext );

        Render( l_Transform * math::Scale( math::mat4( 1.0f ), math::vec3{ 0.05f, 0.05f, 0.05f } ), a_PointLightHelperComponent.Origin, a_PointLightHelperComponent.LightData.Color,
                aRenderContext );
    }

    void VisualHelperRenderer::Render( math::mat4 a_Transform, DirectionalLightHelperComponent &a_DirectionalLightHelperComponent, RenderContext &aRenderContext )
    {
        Render( a_Transform, a_DirectionalLightHelperComponent.Direction, math::vec3{ 0.8f, 0.1f, 0.15f }, aRenderContext );
        Render( a_Transform * math::Scale( math::mat4( 1.0f ), math::vec3{ 0.05f, 0.05f, 0.05f } ), a_DirectionalLightHelperComponent.Origin, math::vec3{ 0.1f, 0.25f, 0.8f },
                aRenderContext );
    }

    void VisualHelperRenderer::Render( math::mat4 a_Transform, SpotlightHelperComponent &a_SpotlightComponent, RenderContext &aRenderContext )
    {
        math::mat4 l_Transform = a_Transform * math::Translation( a_SpotlightComponent.LightData.Position ) *
                                 math::Rotation( math::radians( -90.0f + a_SpotlightComponent.LightData.Azimuth ), math::y_axis() ) *
                                 math::Rotation( math::radians( 90.0f - a_SpotlightComponent.LightData.Elevation ), math::x_axis() );

        Render( l_Transform, a_SpotlightComponent.Spot, math::vec3{ 0.1f, 0.25f, 0.8f }, aRenderContext );

        Render( l_Transform * math::Scale( math::mat4( 1.0f ), math::vec3{ 0.05f, 0.05f, 0.05f } ), a_SpotlightComponent.Origin, math::vec3{ 0.1f, 0.25f, 0.8f }, aRenderContext );
    }

    void VisualHelperRenderer::Render( math::mat4 a_Transform, FieldOfViewHelperComponent &a_FieldOfViewHelperComponent, RenderContext &aRenderContext )
    {
        Render( a_Transform, a_FieldOfViewHelperComponent.Outline, math::vec3{ 0.1f, 0.25f, 0.8f }, aRenderContext );
        Render( a_Transform, a_FieldOfViewHelperComponent.OuterLimit, math::vec3{ 0.1f, 0.25f, 0.8f }, aRenderContext );
    }

    void VisualHelperRenderer::Render( math::mat4 a_Transform, CameraHelperComponent &a_CameraHelperComponent, RenderContext &aRenderContext )
    {
        Render( a_Transform, a_CameraHelperComponent.FieldOfView, aRenderContext );
        Render( a_Transform * math::Scale( math::mat4( 1.0f ), math::vec3{ 0.05f, 0.05f, 0.05f } ), a_CameraHelperComponent.Origin, math::vec3{ 0.1f, 0.25f, 0.8f },
                aRenderContext );
    }

} // namespace LTSE::Graphics