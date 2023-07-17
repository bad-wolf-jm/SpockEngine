#pragma once
#include "Core/Memory.h"

#include "Scene/Components.h"
// #include "Scene/Scene.h"

// #include "Renderer/SceneRenderData.h"

namespace SE::Core
{
    using namespace math;

    class Scene;

    class BaseSceneRenderer
    {
      public:
        BaseSceneRenderer() = default;

        BaseSceneRenderer( Ref<IGraphicContext> aGraphicContext, eColorFormat aOutputFormat, uint32_t aOutputSampleCount );

        ~BaseSceneRenderer() = default;

        void SetProjection( mat4 aProjectionMatrix );

        void SetView( mat4 aViewMatrix );

        void SetGamma( float aGamma );

        void SetExposure( float aExposure );

        void SetAmbientLighting( vec4 aAmbientLight );

        virtual void Update( Ref<Scene> aWorld );
        virtual void Render();

        virtual void ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight );

        virtual Ref<ITexture2D> GetOutputImage() = 0;

      protected:
        Ref<IGraphicContext> mGraphicContext{};

        Ref<Scene> mScene = nullptr;

        uint32_t mOutputWidth  = 0;
        uint32_t mOutputHeight = 0;

        eColorFormat mOutputFormat      = eColorFormat::RGBA8_UNORM;
        uint32_t     mOutputSampleCount = 1;

        // std::vector<DirectionalLightData> mDirectionalLights = {};
        // std::vector<PointLightData>       mPointLights       = {};
        // std::vector<SpotlightData>        mSpotlights        = {};

        mat4  mProjectionMatrix{};
        mat4  mViewMatrix{};
        vec3  mCameraPosition{};
        float mExposure     = 4.5f;
        float mGamma        = 2.2f;
        vec4  mAmbientLight = { 1.0f, 1.0f, 1.0f, 0.0001f };

      protected:
        // Helpers for light position and intensity
        // std::vector<sLightGizmo> mLightGizmos{};

        // Mesh queues
        // std::vector<sMeshRenderData> mOpaqueMeshQueue{};
        // std::vector<sMeshRenderData> mTransparentMeshQueue{};

        // Point cloud visualizer queue
        // std::vector<sParticleRenderData> mParticleQueue{};

      private:
        // void AddLight( mat4 const &aTransform, sLightComponent &aLightComponent );
    };

} // namespace SE::Core