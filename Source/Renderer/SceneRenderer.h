#pragma once
#include "Core/Memory.h"

#include "Graphics/API.h"
#include "Scene/Components.h"

#include "ASceneRenderer.h"
#include "GridRenderer.h"
#include "ShadowRenderer.h"

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
        Ref<IGraphicBuffer> mVertexBuffer = nullptr;
        Ref<IGraphicBuffer> mIndexBuffer  = nullptr;
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
        Ref<IGraphicsPipeline>    mPipeline;
        vector_t<sMeshRenderData> mMeshes;
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
        SceneRenderer( Ref<IGraphicContext> aGraphicContext, eColorFormat aOutputFormat, uint32_t aOutputSampleCount );

        ~SceneRenderer() = default;

        Ref<ITexture2D> GetOutputImage();

        void Update( Ref<Scene> aWorld );
        void Render();

        void ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight );

      protected:
        Ref<IRenderTarget>          mGeometryRenderTarget   = nullptr;
        Ref<IRenderContext>         mGeometryContext        = nullptr;
        Ref<ShadowSceneRenderer>    mShadowSceneRenderer    = nullptr;
        Ref<CoordinateGridRenderer> mCoordinateGridRenderer = nullptr;
        Ref<ISampler2D>             mFxaaSampler            = nullptr;
        Ref<IRenderTarget>          mFxaaRenderTarget       = nullptr;
        Ref<IRenderContext>         mFxaaContext            = nullptr;

        std::map<size_t, sRenderQueue> mPipelines;

        void CreateRenderTarget( uint32_t aOutputWidth, uint32_t aOutputHeight );
        void CreateMSAARenderTarget( uint32_t aOutputWidth, uint32_t aOutputHeight );
        void CreateFXAARenderTarget( uint32_t aOutputWidth, uint32_t aOutputHeight );
    };

} // namespace SE::Core