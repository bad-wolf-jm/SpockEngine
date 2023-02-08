#include "BaseEditorApplication.h"

#include "Core/File.h"
#include "yaml-cpp/yaml.h"
#include <fstream>

#include <direct.h>
#include <iostream>
#include <limits.h>
#include <string>

#include "UI/UI.h"
#include "UI/Widgets.h"

#include "Core/CUDA/Array/CudaBuffer.h"
#include "Core/CUDA/Array/MultiTensor.h"

#include "Core/CUDA/Texture/ColorFormat.h"
#include "Scene/Components.h"
#include "Scene/Importer/glTFImporter.h"

#include "Mono/MonoRuntime.h"

namespace SE::Editor
{

    using namespace SE::Core;
    using namespace SE::Cuda;
    using namespace SE::Core::EntityComponentSystem::Components;

    void BaseEditorApplication::RenderScene()
    {
        mDeferredRenderer->Render();
    }

    void BaseEditorApplication::Update( Timestep ts )
    {
        mEditorWindow.ActiveWorld->Update( ts );
        mEditorWindow.UpdateFramerate( ts );
        mDeferredRenderer->Update( mEditorWindow.ActiveWorld );
    }

    void BaseEditorApplication::RebuildOutputFramebuffer()
    {
        if( mViewportWidth == 0 || mViewportHeight == 0 ) return;

        mDeferredRenderer->ResizeOutput( mViewportWidth, mViewportHeight );

        sTextureSamplingInfo lSamplingInfo{};
        lSamplingInfo.mNormalizedCoordinates = true;
        lSamplingInfo.mNormalizedValues      = true;

        mOffscreenRenderTargetTexture = New<VkSampler2D>( SE::Core::Engine::GetInstance()->GetGraphicContext(),
                                                          mDeferredRenderer->GetOutputImage(), lSamplingInfo );

        if( !mOffscreenRenderTargetDisplayHandle.Handle )
        {
            mOffscreenRenderTargetDisplayHandle =
                SE::Core::Engine::GetInstance()->UIContext()->CreateTextureHandle( mOffscreenRenderTargetTexture );
            mEditorWindow.UpdateSceneViewport( mOffscreenRenderTargetDisplayHandle );
        }
        else
        {
            mOffscreenRenderTargetDisplayHandle.Handle->Write( mOffscreenRenderTargetTexture, 0 );
        }
    }

    bool BaseEditorApplication::RenderUI( ImGuiIO &io )
    {
        bool lRequestQuit = false;
        if( mShouldRebuildViewport )
        {
            RebuildOutputFramebuffer();
            mShouldRebuildViewport = false;
        }

        mEditorWindow.WorldRenderer = mDeferredRenderer;

        lRequestQuit = mEditorWindow.Display();

        auto lWorkspaceAreaSize = mEditorWindow.GetWorkspaceAreaSize();
        if( ( mViewportWidth != lWorkspaceAreaSize.x ) || ( mViewportHeight != lWorkspaceAreaSize.y ) )
        {
            mViewportWidth         = lWorkspaceAreaSize.x;
            mViewportHeight        = lWorkspaceAreaSize.y;
            mShouldRebuildViewport = true;
        }

        return lRequestQuit;
    }

    void BaseEditorApplication::Init()
    {
        mEditorWindow =
            EditorWindow( SE::Core::Engine::GetInstance()->GetGraphicContext(), SE::Core::Engine::GetInstance()->UIContext() );
        mEditorWindow.ApplicationIcon = ICON_FA_CODEPEN;

        mWorld = New<Scene>( SE::Core::Engine::GetInstance()->GetGraphicContext(), SE::Core::Engine::GetInstance()->UIContext() );
        mDeferredRenderer =
            New<DeferredRenderer>( SE::Core::Engine::GetInstance()->GetGraphicContext(), eColorFormat::RGBA8_UNORM, 1 );
        RebuildOutputFramebuffer();

        mDeferredRenderer->Update( mWorld );

        mEditorWindow.World       = mWorld;
        mEditorWindow.ActiveWorld = mWorld;

        {
            // Add sensor entity to the scene
            mEditorWindow.Sensor = mWorld->Create( "Sensor", mWorld->Root );
            mEditorWindow.Sensor.Add<sNodeTransformComponent>();
            mEditorWindow.Sensor.Add<sBehaviourComponent>();

            // Add a particle system to the sensor to display the point cloud
            auto &l_SensorPointCloud = mEditorWindow.Sensor.Add<sParticleSystemComponent>();

            // Create particle renderer for the point cloud
            auto &l_SensorPointCloudRenderer = mEditorWindow.Sensor.Add<sParticleShaderComponent>();

            mEditorWindow.ActiveSensor = mEditorWindow.Sensor;
        }

        mDeferredRenderer->mRenderCoordinateGrid = true;


        mDeferredRenderer->View.CameraPosition   = math::vec3( 0.0f, 1.0f, 7.5f );
        mDeferredRenderer->View.ModelFraming     = math::mat4( 0.5f );
        mDeferredRenderer->View.View = math::Inverse( math::Translate( math::mat4( 1.0f ), mDeferredRenderer->View.CameraPosition ) );
    }
} // namespace SE::Editor
