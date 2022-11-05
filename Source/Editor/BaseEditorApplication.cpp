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

#include "Core/Cuda/CudaBuffer.h"
#include "Core/Cuda/ExternalMemory.h"
#include "Core/Cuda/MultiTensor.h"

#include "Scene/Components.h"
#include "Scene/Importer/glTFImporter.h"

#include "Mono/Manager.h"

namespace LTSE::Editor
{

    using namespace LTSE::Core;
    using namespace LTSE::Cuda;
    using namespace LTSE::Core::EntityComponentSystem::Components;

    // BaseEditorApplication::BaseEditorApplication(  ) { mEngineLoop = aEngineLoop; }

    void BaseEditorApplication::RenderScene()
    {
        mViewportRenderContext.BeginRender();

        if( mViewportRenderContext ) mWorldRenderer->Render( mViewportRenderContext );

        mViewportRenderContext.EndRender();
    }

    void BaseEditorApplication::Update( Timestep ts )
    {
        mEditorWindow.ActiveWorld->Update( ts );
        mEditorWindow.UpdateFramerate( ts );
    }

    void BaseEditorApplication::RebuildOutputFramebuffer()
    {
        if( mViewportWidth == 0 || mViewportHeight == 0 ) return;

        if( !mOffscreenRenderTarget )
        {
            OffscreenRenderTargetDescription l_RenderTargetCI{};
            l_RenderTargetCI.OutputSize  = { mViewportWidth, mViewportHeight };
            l_RenderTargetCI.SampleCount = 4;
            l_RenderTargetCI.Sampled     = true;
            mOffscreenRenderTarget      = New<OffscreenRenderTarget>( LTSE::Core::Engine::GetInstance()->GetGraphicContext(), l_RenderTargetCI );
            mViewportRenderContext      = LTSE::Graphics::RenderContext( LTSE::Core::Engine::GetInstance()->GetGraphicContext(), mOffscreenRenderTarget );
        }
        else
        {
            mOffscreenRenderTarget->Resize( mViewportWidth, mViewportHeight );
        }

        mOffscreenRenderTargetTexture = New<Graphics::Texture2D>( LTSE::Core::Engine::GetInstance()->GetGraphicContext(), TextureDescription{},
                                                                   mOffscreenRenderTarget->GetOutputImage() );

        if( !mOffscreenRenderTargetDisplayHandle.Handle )
        {
            mOffscreenRenderTargetDisplayHandle = LTSE::Core::Engine::GetInstance()->UIContext()->CreateTextureHandle( mOffscreenRenderTargetTexture );
            mEditorWindow.UpdateSceneViewport( mOffscreenRenderTargetDisplayHandle );
        }
        else
        {
            mOffscreenRenderTargetDisplayHandle.Handle->Write( mOffscreenRenderTargetTexture, 0 );
        }

        if( mWorldRenderer )
        {
            mWorldRenderer->View.Projection = math::Perspective(
                90.0_degf, static_cast<float>( mViewportWidth ) / static_cast<float>( mViewportHeight ), 0.01f, 100000.0f );
            mWorldRenderer->View.Projection[1][1] *= -1.0f;
        }
    }

    bool BaseEditorApplication::RenderUI( ImGuiIO &io )
    {
        bool o_RequestQuit = false;
        if( mShouldRebuildViewport )
        {
            RebuildOutputFramebuffer();
            mShouldRebuildViewport = false;
        }

        mEditorWindow.mEngineLoop = mEngineLoop;
        // mEditorWindow.SensorModel    = m_SensorController;
        mEditorWindow.WorldRenderer  = mWorldRenderer;
        mEditorWindow.GraphicContext = LTSE::Core::Engine::GetInstance()->GetGraphicContext();

        o_RequestQuit = mEditorWindow.Display();
        // OnUI();

        auto l_WorkspaceAreaSize = mEditorWindow.GetWorkspaceAreaSize();
        if( ( mViewportWidth != l_WorkspaceAreaSize.x ) || ( mViewportHeight != l_WorkspaceAreaSize.y ) )
        {
            mViewportWidth         = l_WorkspaceAreaSize.x;
            mViewportHeight        = l_WorkspaceAreaSize.y;
            mShouldRebuildViewport = true;
        }

        if( o_RequestQuit ) return true;

        return false;
    }

    void BaseEditorApplication::Init()
    {
        mEditorWindow                 = EditorWindow( LTSE::Core::Engine::GetInstance()->GetGraphicContext(), LTSE::Core::Engine::GetInstance()->UIContext() );
        mEditorWindow.ApplicationIcon = ICON_FA_CODEPEN;

        RebuildOutputFramebuffer();
        mWorld         = New<Scene>( LTSE::Core::Engine::GetInstance()->GetGraphicContext(), LTSE::Core::Engine::GetInstance()->UIContext() );
        mWorldRenderer = New<SceneRenderer>( mWorld, mViewportRenderContext, mOffscreenRenderTarget->GetRenderPass() );

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

        mWorldRenderer->RenderCoordinateGrid = true;
        mWorldRenderer->View.CameraPosition  = math::vec3( 0.0f, 1.0f, 7.5f );
        mWorldRenderer->View.ModelFraming    = math::mat4( 0.5f );
        mWorldRenderer->View.View = math::Inverse( math::Translate( math::mat4( 1.0f ), mWorldRenderer->View.CameraPosition ) );
    }
} // namespace LTSE::Editor
