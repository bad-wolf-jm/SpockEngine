#pragma once
#include "Core/Memory.h"

#include "Scene/Components.h"
#include "Scene/Scene.h"

#include "SceneRenderData.h"

namespace LTSE::Core
{

    class ASceneRenderer
    {
      public:
        ASceneRenderer()  = default;
        ~ASceneRenderer() = default;

        void SetProjection( math::mat4 aProjectionMatrix );
        void SetView( math::mat4 aViewMatrix );
        void SetGamma( float aGamma );
        void SetExposure( float aExposure );
        void SetAmbientLighting( math::vec4 aAmbientLight );

        virtual void Update( Ref<Scene> aWorld );
        virtual void Render( RenderContext &aRenderContext );

      protected:
        std::vector<DirectionalLightData> mDirectionalLights = {};
        std::vector<PointLightData>       mPointLights       = {};
        std::vector<SpotlightData>        mSpotlights        = {};

        math::mat4 mProjectionMatrix{};
        math::mat4 mViewMatrix{};

        float      mExposure     = 4.5f;
        float      mGamma        = 2.2f;
        math::vec4 mAmbientLight = { 1.0f, 1.0f, 1.0f, 0.0001f };
    };

} // namespace LTSE::Core