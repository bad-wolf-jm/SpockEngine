#include "VisualHelperRenderer.h"

#include "Scene/Primitives/Arrow.h"

namespace SE::Graphics
{

    using namespace math::literals;

    VisualHelperRenderer::VisualHelperRenderer( Ref<IGraphicContext> aGraphicContext, Ref<sVkAbstractRenderPassObject> aRenderPass )
    {
        VisualHelperMeshRendererCreateInfo l_CreateInfo{};
        l_CreateInfo.RenderPass = aRenderPass;
        mMeshRenderer           = SE::Core::New<VisualHelperMeshRenderer>( aGraphicContext, l_CreateInfo );

        VisualHelperLineRendererCreateInfo l_GizmoCreateInfo{};
        l_GizmoCreateInfo.LineWidth  = 2.0;
        l_GizmoCreateInfo.RenderPass = aRenderPass;
        mVisualHelperLineRenderer    = SE::Core::New<VisualHelperLineRenderer>( aGraphicContext, l_GizmoCreateInfo );
    }

    void VisualHelperRenderer::Render( math::mat4 aTransform, ArrowMeshData &aArrow, math::vec3 aColor,
                                       Ref<iRenderContext> aRenderContext )
    {
        mMeshRenderer->Render( aTransform, View, Projection, aColor, aArrow.Mesh.Vertices, aArrow.Mesh.Indices, aRenderContext );
    }

    void VisualHelperRenderer::Render( math::mat4 aTransform, ConeMeshData &aCone, math::vec3 aColor,
                                       Ref<iRenderContext> aRenderContext )
    {
        mVisualHelperLineRenderer->Render( aTransform, View, Projection, aColor, aCone.Mesh.Vertices, aCone.Mesh.Indices,
                                           aRenderContext );
    }

    void VisualHelperRenderer::Render( math::mat4 aTransform, CircleMeshData &aCircle, math::vec3 aColor,
                                       Ref<iRenderContext> aRenderContext )
    {
        mVisualHelperLineRenderer->Render( aTransform, View, Projection, aColor, aCircle.Mesh.Vertices, aCircle.Mesh.Indices,
                                           aRenderContext );
    }

    void VisualHelperRenderer::Render( math::mat4 aTransform, CubeMeshData &aCube, math::vec3 aColor,
                                       Ref<iRenderContext> aRenderContext )
    {
        mMeshRenderer->Render( aTransform, View, Projection, aColor, aCube.Mesh.Vertices, aCube.Mesh.Indices, aRenderContext );
    }

    void VisualHelperRenderer::Render( math::mat4 aTransform, PyramidMeshData &aPyramid, math::vec3 aColor,
                                       Ref<iRenderContext> aRenderContext )
    {
        mVisualHelperLineRenderer->Render( aTransform, View, Projection, aColor, aPyramid.Mesh.Vertices, aPyramid.Mesh.Indices,
                                           aRenderContext );
    }

    void VisualHelperRenderer::Render( math::mat4 aTransform, SurfaceMeshData &aSurface, math::vec3 aColor,
                                       Ref<iRenderContext> aRenderContext )
    {
        mMeshRenderer->Render( aTransform, View, Projection, aColor, aSurface.Mesh.Vertices, aSurface.Mesh.Indices, aRenderContext );
    }

    void VisualHelperRenderer::Render( math::mat4 aTransform, AxesComponent &aAxesComponent, Ref<iRenderContext> aRenderContext )
    {
        Render( aTransform * math::Rotation( -90.0_degf, math::vec3{ 0.0f, 0.0f, 1.0f } ), aAxesComponent.AxisArrow,
                aAxesComponent.XAxisColor, aRenderContext );
        Render( aTransform, aAxesComponent.AxisArrow, aAxesComponent.YAxisColor, aRenderContext );
        Render( aTransform * math::Rotation( -90.0_degf, math::vec3{ 1.0f, 0.0f, 0.0f } ), aAxesComponent.AxisArrow,
                aAxesComponent.ZAxisColor, aRenderContext );
        Render( aTransform * math::Scale( math::mat4( 1.0f ), math::vec3{ 0.05f, 0.05f, 0.05f } ), aAxesComponent.Origin,
                aAxesComponent.OriginColor, aRenderContext );
    }

    void VisualHelperRenderer::Render( math::mat4 aTransform, PointLightHelperComponent &aPointLightHelperComponent,
                                       Ref<iRenderContext> aRenderContext )
    {
        math::mat4 lTransform{};

        Render( lTransform, aPointLightHelperComponent.AxisCircle, aPointLightHelperComponent.LightData.mColor, aRenderContext );
        Render( lTransform * math::Rotation( 90.0_degf, math::vec3{ 1.0f, 0.0f, 0.0f } ), aPointLightHelperComponent.AxisCircle,
                aPointLightHelperComponent.LightData.mColor, aRenderContext );
        Render( lTransform * math::Rotation( 90.0_degf, math::vec3{ 0.0f, 1.0f, 0.0f } ), aPointLightHelperComponent.AxisCircle,
                aPointLightHelperComponent.LightData.mColor, aRenderContext );

        Render( lTransform * math::Scale( math::mat4( 1.0f ), math::vec3{ 0.05f, 0.05f, 0.05f } ), aPointLightHelperComponent.Origin,
                aPointLightHelperComponent.LightData.mColor, aRenderContext );
    }

    void VisualHelperRenderer::Render( math::mat4 aTransform, DirectionalLightHelperComponent &aDirectionalLightHelperComponent,
                                       Ref<iRenderContext> aRenderContext )
    {
        Render( aTransform, aDirectionalLightHelperComponent.Direction, math::vec3{ 0.8f, 0.1f, 0.15f }, aRenderContext );
        Render( aTransform * math::Scale( math::mat4( 1.0f ), math::vec3{ 0.05f, 0.05f, 0.05f } ),
                aDirectionalLightHelperComponent.Origin, math::vec3{ 0.1f, 0.25f, 0.8f }, aRenderContext );
    }

    void VisualHelperRenderer::Render( math::mat4 aTransform, SpotlightHelperComponent &aSpotlightComponent,
                                       Ref<iRenderContext> aRenderContext )
    {
        math::mat4 lTransform{};

        Render( lTransform, aSpotlightComponent.Spot, math::vec3{ 0.1f, 0.25f, 0.8f }, aRenderContext );

        Render( lTransform * math::Scale( math::mat4( 1.0f ), math::vec3{ 0.05f, 0.05f, 0.05f } ), aSpotlightComponent.Origin,
                math::vec3{ 0.1f, 0.25f, 0.8f }, aRenderContext );
    }

    void VisualHelperRenderer::Render( math::mat4 aTransform, FieldOfViewHelperComponent &aFieldOfViewHelperComponent,
                                       Ref<iRenderContext> aRenderContext )
    {
        Render( aTransform, aFieldOfViewHelperComponent.Outline, math::vec3{ 0.1f, 0.25f, 0.8f }, aRenderContext );
        Render( aTransform, aFieldOfViewHelperComponent.OuterLimit, math::vec3{ 0.1f, 0.25f, 0.8f }, aRenderContext );
    }

    void VisualHelperRenderer::Render( math::mat4 aTransform, CameraHelperComponent &aCameraHelperComponent,
                                       Ref<iRenderContext> aRenderContext )
    {
        Render( aTransform, aCameraHelperComponent.FieldOfView, aRenderContext );
        Render( aTransform * math::Scale( math::mat4( 1.0f ), math::vec3{ 0.05f, 0.05f, 0.05f } ), aCameraHelperComponent.Origin,
                math::vec3{ 0.1f, 0.25f, 0.8f }, aRenderContext );
    }

} // namespace SE::Graphics