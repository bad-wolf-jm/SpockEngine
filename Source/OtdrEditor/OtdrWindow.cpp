#include "OtdrWindow.h"

#include <fmt/core.h>
#include <fstream>
#include <iostream>
#include <locale>

#include "Core/Profiling/BlockTimer.h"

#include "UI/UI.h"
#include "UI/Widgets.h"

#include "Scene/Components.h"
#include "Scene/Importer/ObjImporter.h"
#include "Scene/Importer/glTFImporter.h"

#include "Core/CUDA/Texture/TextureData.h"
#include "Core/File.h"
#include "Core/Logging.h"

#include "Scene/Components.h"
#include "Scene/Serialize/AssetFile.h"
#include "Scene/Serialize/FileIO.h"

#include "Mono/MonoRuntime.h"

#include "mono/jit/jit.h"
#include "mono/metadata/assembly.h"
#include "mono/metadata/object.h"
#include "mono/metadata/tabledefs.h"

// #include "ShortWaveformDisplay.h"

namespace SE::Editor
{

    using namespace SE::Core;
    using namespace SE::Core::EntityComponentSystem::Components;

    static std::string UTF16ToAscii( const char *aPayloadData, size_t aSize )
    {
        size_t      lPayloadSize = static_cast<size_t>( aSize / 2 );
        std::string lItemPathStr( lPayloadSize - 1, '\0' );
        for( uint32_t i = 0; i < lPayloadSize - 1; i++ ) lItemPathStr[i] = aPayloadData[2 * i];

        return lItemPathStr;
    }

    static bool EditButton( Entity a_Node, math::vec2 a_Size )
    {
        char l_OnLabel[128];
        sprintf( l_OnLabel, "%s##%d", ICON_FA_PENCIL_SQUARE_O, (uint32_t)a_Node );

        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.0, 0.0, 0.0, 0.0 } );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 1.0, 1.0, 1.0, 0.10 } );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 1.0, 1.0, 1.0, 0.20 } );

        bool l_IsVisible;
        bool l_DoEdit = UI::Button( l_OnLabel, a_Size );

        ImGui::PopStyleColor();
        ImGui::PopStyleColor();
        ImGui::PopStyleColor();
        return l_DoEdit;
    }

    void OtdrWindow::ConfigureUI()
    {
        {
            SE::Core::sTextureCreateInfo lTextureCreateInfo{};
            TextureData2D        lTextureData( lTextureCreateInfo, "C:\\GitLab\\SpockEngine\\Saved\\Resources\\Icons\\Play.png" );
            sTextureSamplingInfo lSamplingInfo{};
            SE::Core::TextureSampler2D lTextureSampler = SE::Core::TextureSampler2D( lTextureData, lSamplingInfo );

            auto lTexture   = New<VkTexture2D>( mGraphicContext, lTextureData );
            mPlayIcon       = New<VkSampler2D>( mGraphicContext, lTexture, lSamplingInfo );
            mPlayIconHandle = mUIOverlay->CreateTextureHandle( mPlayIcon );
        }

        {
            SE::Core::sTextureCreateInfo lTextureCreateInfo{};
            TextureData2D        lTextureData( lTextureCreateInfo, "C:\\GitLab\\SpockEngine\\Saved\\Resources\\Icons\\Pause.png" );
            sTextureSamplingInfo lSamplingInfo{};
            SE::Core::TextureSampler2D lTextureSampler = SE::Core::TextureSampler2D( lTextureData, lSamplingInfo );

            auto lTexture    = New<VkTexture2D>( mGraphicContext, lTextureData );
            mPauseIcon       = New<VkSampler2D>( mGraphicContext, lTexture, lSamplingInfo );
            mPauseIconHandle = mUIOverlay->CreateTextureHandle( mPauseIcon );
        }
    }

    OtdrWindow::OtdrWindow( Ref<VkGraphicContext> aGraphicContext, Ref<UIContext> aUIOverlay )
        : mGraphicContext{ aGraphicContext }
        , mUIOverlay{ aUIOverlay }
    {
        ConfigureUI();
    }

    // OtdrWindow &OtdrWindow::AddMenuItem( std::string l_Icon, std::string l_Title, std::function<bool()> l_Action )
    // {
    //     m_MainMenuItems.push_back( { l_Icon, l_Title, l_Action } );
    //     return *this;
    // }

    bool OtdrWindow::Display()
    {
        ImGuizmo::SetOrthographic( false );
        static bool                  p_open          = true;
        constexpr ImGuiDockNodeFlags lDockSpaceFlags = ImGuiDockNodeFlags_PassthruCentralNode;
        constexpr ImGuiWindowFlags   lMainwindowFlags =
            ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

        ImGuiViewport *viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos( viewport->WorkPos );
        ImGui::SetNextWindowSize( viewport->WorkSize - ImVec2( 0.0f, StatusBarHeight ) );
        ImGui::SetNextWindowViewport( viewport->ID );

        ImGui::PushStyleVar( ImGuiStyleVar_WindowRounding, 0.0f );
        ImGui::PushStyleVar( ImGuiStyleVar_WindowBorderSize, 0.0f );
        ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2( 0.0f, 0.0f ) );
        ImGui::Begin( "DockSpace Demo", &p_open, lMainwindowFlags );
        ImGui::PopStyleVar( 3 );

        ImGuiID dockspace_id = ImGui::GetID( "MyDockSpace" );
        ImGui::DockSpace( dockspace_id, ImVec2( 0.0f, 0.0f ), lDockSpaceFlags );

        bool lRequestQuit = false;
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, ImVec2( 0, 8 ) );
        if( ImGui::BeginMainMenuBar() )
        {
            ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, ImVec2( 15, 14 ) );
            lRequestQuit = RenderMainMenu();
            ImGui::PopStyleVar();
            ImGui::EndMainMenuBar();
        }
        ImGui::PopStyleVar();
        ImGui::End();

        static bool p_open_3 = true;
        if( ImGui::Begin( "3D VIEW", &p_open_3, ImGuiWindowFlags_None ) )
        {
            auto lWorkspaceAreaSize = UI::GetAvailableContentSpace();
            Workspace( lWorkspaceAreaSize.x, lWorkspaceAreaSize.y );
        }
        ImGui::End();

        if( ImGui::Begin( "ASSEMBLIES", NULL, ImGuiWindowFlags_None ) )
        {
            MonoRuntime::DisplayAssemblies();
        }
        ImGui::End();

        if( ImGui::Begin( "CONNECTED MODULES", NULL, ImGuiWindowFlags_None ) )
        {
            static auto lFirstRun = true;
            static auto lLastTime =
                std::chrono::time_point_cast<std::chrono::microseconds>( std::chrono::high_resolution_clock::now() )
                    .time_since_epoch()
                    .count();
            auto lCurrentTime = std::chrono::time_point_cast<std::chrono::microseconds>( std::chrono::high_resolution_clock::now() )
                                    .time_since_epoch()
                                    .count();

            static auto &lMonoGlue          = MonoRuntime::GetClassType( "Metrino.Mono.Instruments" );
            static auto &lModuleDescription = MonoRuntime::GetClassType( "Metrino.Mono.ModuleDescription" );

            static uint32_t                 lNumConnectedModules  = 0;
            static std::vector<std::string> lConnectedModuleNames = {};

            if( lFirstRun || ( ( lCurrentTime - lLastTime ) > 1500000 ) )
            {
                lConnectedModuleNames.clear();
                lLastTime                  = lCurrentTime;
                auto lConnectedModulesList = lMonoGlue.CallMethod( "GetConnectedModules" );

                lNumConnectedModules = static_cast<uint32_t>( mono_array_length( (MonoArray *)lConnectedModulesList ) );

                std::vector<MonoObject *> lConnectedModules( lNumConnectedModules );
                for( uint32_t i = 0; i < lNumConnectedModules; i++ )
                {
                    auto lConnectedModule = *( mono_array_addr( (MonoArray *)lConnectedModulesList, MonoObject *, i ) );
                    auto lInstance        = MonoScriptInstance( lModuleDescription.Class(), lConnectedModule );
                    lConnectedModuleNames.push_back( MonoRuntime::NewString( lInstance.GetFieldValue<MonoString *>( "mName" ) ) );
                }

                lFirstRun = false;
            }

            if( lNumConnectedModules == 0 )
            {
                UI::Text( "No connected modules", lNumConnectedModules );
            }
            else
            {
                for( uint32_t i = 0; i < lNumConnectedModules; i++ )
                {
                    Text( "{} {}", ICON_FA_USB, lConnectedModuleNames[i] );
                }
            }
        }
        ImGui::End();

        if( ImGui::Begin( "LOGS", &p_open, ImGuiWindowFlags_None ) )
        {
            math::vec2 l_WindowConsoleSize = UI::GetAvailableContentSpace();
            Console( l_WindowConsoleSize.x, l_WindowConsoleSize.y );
        }
        ImGui::End();

        if( ImGui::Begin( "PROFILING", &p_open, ImGuiWindowFlags_None ) )
        {
            math::vec2  l_WindowConsoleSize     = UI::GetAvailableContentSpace();
            static bool lIsProfiling            = false;
            static bool lProfilingDataAvailable = false;

            static std::unordered_map<std::string, float>    lProfilingResults = {};
            static std::unordered_map<std::string, uint32_t> lProfilingCount   = {};

            std::string lButtonText = lIsProfiling ? "Stop capture" : "Start capture";
            if( UI::Button( lButtonText.c_str(), { 150.0f, 50.0f } ) )
            {
                if( lIsProfiling )
                {
                    auto lResults = SE::Core::Instrumentor::Get().EndSession();
                    SE::Logging::Info( "{}", lResults->mEvents.size() );
                    lIsProfiling            = false;
                    lProfilingDataAvailable = true;
                    lProfilingResults       = {};
                    for( auto &lEvent : lResults->mEvents )
                    {
                        if( lProfilingResults.find( lEvent.mName ) == lProfilingResults.end() )
                        {
                            lProfilingResults[lEvent.mName] = lEvent.mElapsedTime;
                            lProfilingCount[lEvent.mName]   = 1;
                        }
                        else
                        {
                            lProfilingResults[lEvent.mName] += lEvent.mElapsedTime;
                            lProfilingCount[lEvent.mName] += 1;
                        }
                    }

                    for( auto &lEntry : lProfilingResults )
                    {
                        lProfilingResults[lEntry.first] /= lProfilingCount[lEntry.first];
                    }
                }
                else
                {
                    SE::Core::Instrumentor::Get().BeginSession( "Session" );
                    lIsProfiling            = true;
                    lProfilingDataAvailable = false;
                    lProfilingResults       = {};
                }
            }

            if( lProfilingDataAvailable )
            {
                for( auto &lEntry : lProfilingResults )
                {
                    UI::Text( "{} ----> {} us", lEntry.first, lEntry.second );
                }
            }
            // Console( l_WindowConsoleSize.x, l_WindowConsoleSize.y );
        }
        ImGui::End();

        if( ( ImGui::Begin( "SCENE HIERARCHY", &p_open, ImGuiWindowFlags_None ) ) )
        {
            // auto l_WindowPropertiesSize = UI::GetAvailableContentSpace();
            // m_SceneHierarchyPanel.World = ActiveWorld;
            // m_SceneHierarchyPanel.Display( l_WindowPropertiesSize.x, l_WindowPropertiesSize.y );
        }
        ImGui::End();

        if( ImGui::Begin( "CONTENT BROWSER", &p_open, ImGuiWindowFlags_None ) )
        {
            // mContentBrowser.Display();
        }
        ImGui::End();

        if( ImGui::Begin( "PROPERTIES", &p_open, ImGuiWindowFlags_None ) )
        {
            // auto l_WindowPropertiesSize = UI::GetAvailableContentSpace();
            // m_SceneHierarchyPanel.ElementEditor.Display( l_WindowPropertiesSize.x, l_WindowPropertiesSize.y );
        }
        ImGui::End();

        ImGui::PushStyleColor( ImGuiCol_WindowBg, ImVec4{ 102.0f / 255.0f, 0.0f, 204.0f / 255.0f, 1.0f } );
        ImGui::PushStyleVar( ImGuiStyleVar_WindowRounding, 0.0f );
        ImGui::PushStyleVar( ImGuiStyleVar_WindowBorderSize, 0.0f );
        ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2( 5.0f, 7.0f ) );
        ImGui::SetNextWindowSize( ImVec2{ viewport->WorkSize.x, StatusBarHeight } );
        ImGui::SetNextWindowPos( ImVec2{ 0.0f, viewport->WorkSize.y } );
        constexpr ImGuiWindowFlags lStatusBarFlags =
            ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

        ImGui::Begin( "##STATUS_BAR", &p_open, lStatusBarFlags );
        ImGui::PopStyleVar( 3 );
        ImGui::PopStyleColor();
        auto l_WindowSize = UI::GetAvailableContentSpace();
        {
            if( mLastFPS > 0 )
                Text( "Render: {} fps ({:.2f} ms/frame)", mLastFPS, ( 1000.0f / mLastFPS ) );
            else
                Text( "Render: 0 fps (0 ms)" );
        }
        ImGui::End();

        return lRequestQuit;
    }

    bool OtdrWindow::RenderMainMenu()
    {
        ImVec2 l_MenuSize = ImGui::GetWindowSize();

        UI::Text( ApplicationIcon.c_str() );

        bool l_RequestQuit = false;

        if( ImGui::BeginMenu( "File" ) )
        {
            if( UI::MenuItem( fmt::format( "{} Load scene", ICON_FA_PLUS_CIRCLE ).c_str(), NULL ) )
            {
                auto lFilePath = FileDialogs::OpenFile( SE::Core::Engine::GetInstance()->GetMainApplicationWindow(),
                                                        "glTf Files (*.gltf)\0*.gltf\0All Files (*.*)\0*.*\0" );
                // if( lFilePath.has_value() ) LoadScenario( lFilePath.value() );
            }
            if( UI::MenuItem( fmt::format( "{} New material", ICON_FA_PLUS_CIRCLE ).c_str(), NULL ) )
            {
                // m_NewMaterial.Visible = true;
            }
            if( UI::MenuItem( fmt::format( "{} Import model...", ICON_FA_PLUS_CIRCLE ).c_str(), NULL ) )
            {
                auto lFilePath = FileDialogs::OpenFile( SE::Core::Engine::GetInstance()->GetMainApplicationWindow(),
                                                        "glTf Files (*.gltf)\0*.gltf\0All Files (*.*)\0*.*\0" );
                // if( lFilePath.has_value() ) ImportModel( lFilePath.value() );
            }
            if( UI::MenuItem( fmt::format( "{} Save Scene", ICON_FA_PLUS_CIRCLE ).c_str(), NULL ) )
            {
                // m_NewMaterial.Visible = true;
                // World->SaveAs( fs::path( "C:\\GitLab\\SpockEngine\\Saved" ) / "TEST" / "SCENE" );
            }

            if( UI::MenuItem( fmt::format( "{} Load Scenario Scene", ICON_FA_PLUS_CIRCLE ).c_str(), NULL ) )
            {
                // m_NewMaterial.Visible = true;
                // World->LoadScenario( fs::path( "C:\\GitLab\\SpockEngine\\Saved" ) / "TEST" / "SCENE" / "Scene.yaml" );
            }

            l_RequestQuit = UI::MenuItem( fmt::format( "{} Exit", ICON_FA_WINDOW_CLOSE_O ).c_str(), NULL );
            ImGui::EndMenu();
        }

        return l_RequestQuit;
    }

    math::ivec2 OtdrWindow::GetWorkspaceAreaSize() { return mWorkspaceAreaSize; }

    void OtdrWindow::Workspace( int32_t width, int32_t height )
    {
        auto &lIO = ImGui::GetIO();

        math::vec2 lWorkspacePosition = UI::GetCurrentCursorScreenPosition();
        math::vec2 lCursorPosition    = UI::GetCurrentCursorPosition();

        // UI::SameLine();
        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.0f, 0.0f, 0.0f, 0.0f } );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 1.0f, 1.0f, 1.0f, 0.01f } );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 1.0f, 1.0f, 1.0f, 0.02f } );

        if( true )
        {

            if( ImGui::ImageButton( (ImTextureID)mPlayIconHandle.Handle->GetVkDescriptorSet(), ImVec2{ 22.0f, 22.0f },
                                    ImVec2{ 0.0f, 0.0f }, ImVec2{ 1.0f, 1.0f }, 0, ImVec4{ 0.0f, 0.0f, 0.0f, 0.0f },
                                    ImVec4{ 0.0f, 1.0f, 0.0f, 0.8f } ) )
            {
                // if( OnBeginScenario ) OnBeginScenario();

                // ActiveWorld = New<Scene>( World );
                // ActiveWorld->BeginScenario();
            }
        }
        else
        {
            if( ImGui::ImageButton( (ImTextureID)mPauseIconHandle.Handle->GetVkDescriptorSet(), ImVec2{ 22.0f, 22.0f },
                                    ImVec2{ 0.0f, 0.0f }, ImVec2{ 1.0f, 1.0f }, 0, ImVec4{ 0.0f, 0.0f, 0.0f, 0.0f },
                                    ImVec4{ 1.0f, .2f, 0.0f, 0.8f } ) )
            {
                // if( OnEndScenario ) OnEndScenario();

                // ActiveWorld->EndScenario();
                // ActiveWorld  = World;
                // ActiveSensor = Sensor;
            }
        }

        ImGui::PopStyleColor();
        ImGui::PopStyleColor();
        ImGui::PopStyleColor();

        math::vec2  l3DViewPosition = UI::GetCurrentCursorScreenPosition();
        math::ivec2 l3DViewSize     = UI::GetAvailableContentSpace();
        mWorkspaceAreaSize          = l3DViewSize;
    }

    void OtdrWindow::Console( int32_t width, int32_t height )
    {
        const float           TEXT_BASE_HEIGHT = ImGui::GetTextLineHeightWithSpacing();
        const ImGuiTableFlags flags            = ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY |
                                      ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_Resizable |
                                      ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable;

        auto  &l_Logs     = SE::Logging::GetLogMessages();
        ImVec2 outer_size = ImVec2( 0.0f, height );
        if( ImGui::BeginTable( "table_scrolly", 3, flags, outer_size ) )
        {
            ImGui::TableSetupScrollFreeze( 2, 1 );
            ImGui::TableSetupColumn( "", ImGuiTableColumnFlags_WidthFixed, TEXT_BASE_HEIGHT );
            ImGui::TableSetupColumn( "Time", ImGuiTableColumnFlags_None, 150 );
            ImGui::TableSetupColumn( "Message", ImGuiTableColumnFlags_WidthStretch );
            ImGui::TableHeadersRow();

            ImGuiListClipper clipper;
            clipper.Begin( l_Logs.size() );
            while( clipper.Step() )
            {
                ImGui::TableNextRow();
                for( int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++ )
                {
                    ImGui::TableSetColumnIndex( 2 );
                    Text( l_Logs[row].Message );
                }
            }
            ImGui::EndTable();
        }
    }

    void OtdrWindow::UpdateFramerate( Timestep ts )
    {
        mFrameCounter++;
        mFpsTimer += (float)ts;
        if( mFpsTimer > 1000.0f )
        {
            mLastFPS      = static_cast<uint32_t>( (float)mFrameCounter * ( 1000.0f / mFpsTimer ) );
            mFpsTimer     = 0.0f;
            mFrameCounter = 0;
        }
    }

} // namespace SE::Editor