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

#include "Core/Textures/ColorFormat.h"
#include "Scene/Components.h"
#include "Scene/Importer/glTFImporter.h"

#include "Mono/Manager.h"

namespace LTSE::Editor
{

    using namespace LTSE::Core;
    using namespace LTSE::Cuda;
    using namespace LTSE::Core::EntityComponentSystem::Components;

    BaseEditorApplication::BaseEditorApplication( Ref<EngineLoop> aEngineLoop ) { mEngineLoop = aEngineLoop; }

    void BaseEditorApplication::RenderScene()
    {
        mDeferredRenderer->Render();
        mForwardRenderer->Render();
    }

    void BaseEditorApplication::Update( Timestep ts )
    {
        mEditorWindow.ActiveWorld->Update( ts );
        mEditorWindow.UpdateFramerate( ts );
        mDeferredRenderer->Update( mEditorWindow.ActiveWorld );
        mForwardRenderer->Update( mEditorWindow.ActiveWorld );
    }

    void BaseEditorApplication::RebuildOutputFramebuffer()
    {
        if( mViewportWidth == 0 || mViewportHeight == 0 ) return;

        mDeferredRenderer->ResizeOutput( mViewportWidth, mViewportHeight );
        mForwardRenderer->ResizeOutput( mViewportWidth, mViewportHeight );

        mOffscreenRenderTargetTexture =
            New<Graphics::Texture2D>( mEngineLoop->GetGraphicContext(), TextureDescription{}, mForwardRenderer->GetOutputImage() );

        if( !mOffscreenRenderTargetDisplayHandle.Handle )
        {
            mOffscreenRenderTargetDisplayHandle = mEngineLoop->UIContext()->CreateTextureHandle( mOffscreenRenderTargetTexture );
            mEditorWindow.UpdateSceneViewport( mOffscreenRenderTargetDisplayHandle );
        }
        else
        {
            mOffscreenRenderTargetDisplayHandle.Handle->Write( mOffscreenRenderTargetTexture, 0 );
        }

        mDeferredRenderTargetTexture =
            New<Graphics::Texture2D>( mEngineLoop->GetGraphicContext(), TextureDescription{}, mDeferredRenderer->GetOutputImage() );

        if( !mDeferredRenderTargetDisplayHandle.Handle )
        {
            mDeferredRenderTargetDisplayHandle = mEngineLoop->UIContext()->CreateTextureHandle( mDeferredRenderTargetTexture );
            mEditorWindow.UpdateSceneViewport_deferred( mDeferredRenderTargetDisplayHandle );
        }
        else
        {
            mDeferredRenderTargetDisplayHandle.Handle->Write( mDeferredRenderTargetTexture, 0 );
        }

        mDeferredRenderer->SetProjection( math::Perspective(
            90.0_degf, static_cast<float>( mViewportWidth ) / static_cast<float>( mViewportHeight ), 0.01f, 100000.0f ) );
        mForwardRenderer->SetProjection( math::Perspective(
            90.0_degf, static_cast<float>( mViewportWidth ) / static_cast<float>( mViewportHeight ), 0.01f, 100000.0f ) );
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
        mEditorWindow.WorldRenderer = mForwardRenderer;
        mEditorWindow.DefRenderer = mDeferredRenderer;
        mEditorWindow.GraphicContext = mEngineLoop->GetGraphicContext();

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
        mEditorWindow                 = EditorWindow( mEngineLoop->GetGraphicContext(), mEngineLoop->UIContext() );
        mEditorWindow.ApplicationIcon = ICON_FA_CODEPEN;

        mWorld            = New<Scene>( mEngineLoop->GetGraphicContext(), mEngineLoop->UIContext() );
        mDeferredRenderer = New<DeferredRenderer>( mEngineLoop->GetGraphicContext(), eColorFormat::RGBA8_UNORM, 1 );
        mForwardRenderer  = New<ForwardSceneRenderer>( mEngineLoop->GetGraphicContext(), eColorFormat::RGBA8_UNORM, 4 );
        RebuildOutputFramebuffer();

        mForwardRenderer->Update( mWorld );
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

        mForwardRenderer->RenderCoordinateGrid = true;
        mForwardRenderer->View.CameraPosition  = math::vec3( 0.0f, 1.0f, 7.5f );
        mForwardRenderer->View.ModelFraming    = math::mat4( 0.5f );
        mForwardRenderer->View.View = math::Inverse( math::Translate( math::mat4( 1.0f ), mForwardRenderer->View.CameraPosition ) );
    }
} // namespace LTSE::Editor
