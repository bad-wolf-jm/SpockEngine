#pragma once
#include "Core/Memory.h"

#include "Graphics/API.h"
#include "Scene/Components.h"

#include "ASceneRenderer.h"
#include "ShadowRenderer.h"
#include "GridRenderer.h"

namespace SE::Core
{
    using namespace math;
    using namespace SE::Core::EntityComponentSystem::Components;

    class Scene;

    struct sMeshRenderData
    {
        // Shader data
        Material mMaterialID = 0;

        // Buffer data
        ref_t<IGraphicBuffer> mVertexBuffer = nullptr;
        ref_t<IGraphicBuffer> mIndexBuffer  = nullptr;
        uint32_t            mVertexOffset = 0;
        uint32_t            mVertexCount  = 0;
        uint32_t            mIndexOffset  = 0;
        uint32_t            mIndexCount   = 0;

        sMeshRenderData( sStaticMeshComponent const &aMesh, sMaterialComponent const &aMaterialID )
            : mMaterialID{ aMaterialID.mMaterialID }
            , mIndexBuffer{ aMesh.mIndexBuffer }
            , mVertexBuffer{ aMesh.mTransformedBuffer }
            , mVertexOffset{ aMesh.mVertexOffset }
            , mVertexCount{ aMesh.mVertexCount }
            , mIndexOffset{ aMesh.mIndexOffset }
            , mIndexCount{ aMesh.mIndexCount }
        {
        }
    };

    struct sRenderQueue
    {
        ref_t<IGraphicsPipeline>       mPipeline;
        std::vector<sMeshRenderData> mMeshes;
    };

    class SceneRenderer : public BaseSceneRenderer
    {
      public:
        bool mRenderCoordinateGrid = true;
        bool mRenderGizmos         = false;
        bool mGrayscaleRendering   = false;
        bool mUseFXAA              = false;

      public:
        SceneRenderer() = default;
        SceneRenderer( ref_t<IGraphicContext> aGraphicContext, eColorFormat aOutputFormat, uint32_t aOutputSampleCount );

        ~SceneRenderer() = default;

        ref_t<ITexture2D> GetOutputImage();

        void Update( ref_t<Scene> aWorld );
        void Render();

        void ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight );

      protected:
        ref_t<IRenderTarget>          mGeometryRenderTarget   = nullptr;
        ref_t<IRenderContext>         mGeometryContext        = nullptr;
        ref_t<ShadowSceneRenderer>    mShadowSceneRenderer    = nullptr;
        ref_t<CoordinateGridRenderer> mCoordinateGridRenderer = nullptr;
        ref_t<ISampler2D>             mFxaaSampler            = nullptr;
        ref_t<IRenderTarget>          mFxaaRenderTarget       = nullptr;
        ref_t<IRenderContext>         mFxaaContext            = nullptr;

        std::map<size_t, sRenderQueue> mPipelines;

        void CreateRenderTarget( uint32_t aOutputWidth, uint32_t aOutputHeight );
        void CreateMSAARenderTarget( uint32_t aOutputWidth, uint32_t aOutputHeight );
        void CreateFXAARenderTarget( uint32_t aOutputWidth, uint32_t aOutputHeight );
    };

} // namespace SE::Core