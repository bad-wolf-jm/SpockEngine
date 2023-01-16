#pragma once
#include "Core/Memory.h"

#include "Scene/Components.h"
#include "Scene/Scene.h"

#include "SceneRenderData.h"

namespace SE::Core
{
    class ASceneRenderer
    {
      public:
        ASceneRenderer() = default;

        ASceneRenderer( Ref<VkGraphicContext> aGraphicContext, eColorFormat aOutputFormat, uint32_t aOutputSampleCount );

        ~ASceneRenderer() = default;

        void SetProjection( math::mat4 aProjectionMatrix );

        void SetView( math::mat4 aViewMatrix );

        void SetGamma( float aGamma );

        void SetExposure( float aExposure );

        void SetAmbientLighting( math::vec4 aAmbientLight );

        virtual void Update( Ref<Scene> aWorld );
        virtual void Render();

        virtual void ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight );

        virtual Ref<VkTexture2D> GetOutputImage() = 0;

      protected:
        Ref<VkGraphicContext> mGraphicContext{};

        Ref<Scene> mScene = nullptr;

        uint32_t mOutputWidth  = 0;
        uint32_t mOutputHeight = 0;

        eColorFormat mOutputFormat      = eColorFormat::RGBA8_UNORM;
        uint32_t     mOutputSampleCount = 1;

        std::vector<DirectionalLightData> mDirectionalLights = {};
        std::vector<PointLightData>       mPointLights       = {};
        std::vector<SpotlightData>        mSpotlights        = {};

        math::mat4 mProjectionMatrix{};
        math::mat4 mViewMatrix{};
        math::vec3 mCameraPosition{};

        float      mExposure     = 4.5f;
        float      mGamma        = 2.2f;
        math::vec4 mAmbientLight = { 1.0f, 1.0f, 1.0f, 0.0001f };

      private:
        std::vector<sLightGizmo>     mLightGizmos{};
        std::vector<sMeshRenderData> mStaticMeshQueue{};
        std::vector<sParticleRenderData> mParticleQueue{};
    };

} // namespace SE::Core