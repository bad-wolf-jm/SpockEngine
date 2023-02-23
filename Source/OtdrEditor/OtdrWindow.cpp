#include "OtdrWindow.h"

#include <fmt/core.h>
#include <fstream>
#include <iostream>
#include <locale>

#include "Core/Profiling/BlockTimer.h"

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

namespace SE::OtdrEditor
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
        mTestButton0 = UIButton( "Test button 0...", [&]() { SE::Logging::Info( "ClickedOnTest" ); } );
        mTestButton1 = UIButton( "Test button 1...", [&]() { SE::Logging::Info( "ClickedOnTest" ); } );
        mTestButton2 = UIButton( "Test button 2...", [&]() { SE::Logging::Info( "ClickedOnTest" ); } );

        mTestLabel0 = UILabel( "LABEL 1" );
        mTestLabel1 = UILabel( "LABEL 2" );
        mTestLabel2 = UILabel( "LABEL 3" );

        mTestLayout1 = BoxLayout( eBoxLayoutOrientation::HORIZONTAL );
        mTestLayout1.Add( &mTestLabel0, true, true );
        mTestLayout1.Add( &mTestLabel1, true, true );
        mTestLayout1.Add( &mTestLabel2, true, true );

        mImage0 = UIImage( "C:\\GitLab\\SpockEngine\\Saved\\Resources\\Icons\\Play.png", math::vec2{ 50, 50 } );
        mImage1 = UIImage( "C:\\GitLab\\SpockEngine\\Saved\\Resources\\Icons\\Pause.png", math::vec2{ 25, 25 } );

        mTestLayout2 = BoxLayout( eBoxLayoutOrientation::HORIZONTAL );
        mTestLayout2.Add( &mTestCheckBox0, true, true );
        mTestLayout2.Add( &mTestCheckBox1, true, true );
        mTestLayout2.Add( &mImage0, true, false );
        mTestLayout2.Add( &mImage1, true, false );

        mTestTextToggleButton0 = UITextToggleButton( "Test toggle 0",
                                                     [&]( bool v )
                                                     {
                                                         SE::Logging::Info( "ClickedOnTest" );
                                                         return true;
                                                     } );
        mTestTextToggleButton1 = UITextToggleButton( "Test toggle 1",
                                                     [&]( bool v )
                                                     {
                                                         SE::Logging::Info( "ClickedOnTest" );
                                                         return true;
                                                     } );
        mTestTextToggleButton2 = UITextToggleButton( "Test toggle 2",
                                                     [&]( bool v )
                                                     {
                                                         SE::Logging::Info( "ClickedOnTest" );
                                                         return true;
                                                     } );

        mTestImageToggleButton0.SetActiveImage( mImage0 );
        mTestImageToggleButton0.SetInactiveImage( mImage1 );
        mTestImageToggleButton0.OnChange( [&]( bool ) { return true; } );

        mTestLayout3 = BoxLayout( eBoxLayoutOrientation::HORIZONTAL );
        mTestLayout3.Add( &mTestTextToggleButton0, true, true );
        mTestLayout3.Add( &mTestImageToggleButton0, false, true, eHorizontalAlignment::CENTER, eVerticalAlignment::CENTER );
        mTestLayout3.Add( &mTestTextToggleButton1, true, true );
        mTestLayout3.Add( &mTestTextToggleButton2, true, true );

        mTestLayout0 = BoxLayout( eBoxLayoutOrientation::VERTICAL );
        mTestLayout0.SetItemSpacing( 5.0f );
        mTestLayout0.Add( &mTestButton0, true, true );
        mTestLayout0.Add( &mTestLayout1, true, true );
        mTestLayout0.Add( &mTestButton1, true, false, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        mTestLayout0.Add( &mTestLayout2, true, true );
        mTestLayout0.Add( &mTestButton2, true, true );
        mTestLayout0.Add( &mTestLayout3, true, true );

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

        mTestForm3.SetTitle("TEST_FORM");
        mTestForm3.SetContent(&mTestLayout0);
    }

    OtdrWindow::OtdrWindow( Ref<VkGraphicContext> aGraphicContext, Ref<UIContext> aUIOverlay )
        : mGraphicContext{ aGraphicContext }
        , mUIOverlay{ aUIOverlay }
    {
        // ConfigureUI();
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

        // if( ImGui::Begin( "WIDGET TEST", NULL, ImGuiWindowFlags_None ) )
        // {
        //     mTestLayout0.Update( ImGui::GetCursorPos(), ImGui::GetContentRegionAvail() );
        // }
        // ImGui::End();

        mTestForm3.Update();

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

        static MonoScriptInstance lInstance;
        if( ImGui::Begin( "SCRIPTS", NULL, ImGuiWindowFlags_None ) )
        {
            auto lScriptBaseClass = MonoRuntime::GetClassType( "SpockEngine.Script" );

            for( auto const &lScriptClass : lScriptBaseClass.DerivedClasses() )
            {
                if( UI::Button( lScriptClass->FullName().c_str(), math::vec2{ 100.0f, 30.0f } ) )
                {
                    mCurrentScript = lScriptClass->Instantiate();
                }
            }
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
            auto lWindowPropertiesSize = UI::GetAvailableContentSpace();

            mSceneHierarchyPanel.World = mActiveWorld;
            mSceneHierarchyPanel.Display( lWindowPropertiesSize.x, lWindowPropertiesSize.y );
        }
        ImGui::End();

        if( ImGui::Begin( "CONTENT BROWSER", &p_open, ImGuiWindowFlags_None ) )
        {
            // mContentBrowser.Display();
        }
        ImGui::End();

        ImGui::ShowDemoWindow();

        if( ImGui::Begin( "PROPERTIES", &p_open, ImGuiWindowFlags_None ) )
        {
            // auto lWindowPropertiesSize = UI::GetAvailableContentSpace();
            // m_SceneHierarchyPanel.ElementEditor.Display( lWindowPropertiesSize.x, lWindowPropertiesSize.y );
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

    void OtdrWindow::LoadIOlmData( fs::path aPath )
    {
        static auto &lFileLoader = MonoRuntime::GetClassType( "Metrino.Mono.FileLoader" );
        static auto &lFileClass = MonoRuntime::GetClassType( "Metrino.Mono.OlmFIle" );

        MonoString *lFilePath   = MonoRuntime::NewString( aPath.string() );
        MonoObject *lDataObject = lFileLoader.CallMethod( "LoadOlmData", lFilePath );

        Ref<MonoScriptInstance> lDataInstance = New<MonoScriptInstance>( lFileClass.Class(), lDataObject );
    }

    bool OtdrWindow::RenderMainMenu()
    {
        UI::Text( ApplicationIcon.c_str() );

        bool lRequestQuit = false;

        if( ImGui::BeginMenu( "File" ) )
        {
            if( UI::MenuItem( fmt::format( "{} Load", ICON_FA_PLUS_CIRCLE ).c_str(), NULL ) )
            {
                auto lFilePath = FileDialogs::OpenFile( SE::Core::Engine::GetInstance()->GetMainApplicationWindow(),
                                                        "YAML Files (*.yaml)\0*.yaml\0All Files (*.*)\0*.*\0" );
                mWorld->Load( fs::path( lFilePath.value() ) );
            }

            if( UI::MenuItem( fmt::format( "{} Load iOlm", ICON_FA_PLUS_CIRCLE ).c_str(), NULL ) )
            {
                auto lFilePath = FileDialogs::OpenFile( SE::Core::Engine::GetInstance()->GetMainApplicationWindow(),
                                                        "OLM Files (*.iolm)\0*.iolm\0All Files (*.*)\0*.*\0" );

                LoadIOlmData( fs::path( lFilePath.value() ) );
                // mWorld->Load( fs::path( lFilePath.value() ) );
            }

            if( UI::MenuItem( fmt::format( "{} Save", ICON_FA_ARCHIVE ).c_str(), NULL ) )
            {
                if( mCurrentPath.empty() )
                {
                    auto lFilePath = FileDialogs::SaveFile( SE::Core::Engine::GetInstance()->GetMainApplicationWindow(),
                                                            "YAML Files (*.yaml)\0*.yaml\0All Files (*.*)\0*.*\0" );

                    if( lFilePath.has_value() )
                    {
                        mCurrentPath = fs::path( lFilePath.value() );
                        mWorld->SaveAs( mCurrentPath );
                    }
                }
                else
                {
                    mWorld->SaveAs( mCurrentPath );
                }
            }

            if( UI::MenuItem( fmt::format( "{} Save as...", ICON_FA_ARCHIVE ).c_str(), NULL ) )
            {
                auto lFilePath = FileDialogs::SaveFile( SE::Core::Engine::GetInstance()->GetMainApplicationWindow(),
                                                        "YAML Files (*.yaml)\0*.yaml\0All Files (*.*)\0*.*\0" );
                mCurrentPath   = fs::path( lFilePath.value() );
                mWorld->SaveAs( mCurrentPath );
            }

            lRequestQuit = UI::MenuItem( fmt::format( "{} Exit", ICON_FA_WINDOW_CLOSE_O ).c_str(), NULL );
            ImGui::EndMenu();
        }

        return lRequestQuit;
    }

    math::ivec2 OtdrWindow::GetWorkspaceAreaSize() { return mWorkspaceAreaSize; }

    void OtdrWindow::Update( Timestep aTs )
    {
        mActiveWorld->Update( aTs );

        if( mCurrentScriptIsRunning )
        {
            mCurrentScript->CallMethod( "Tick", &aTs );
        }

        UpdateFramerate( aTs );
    }

    void OtdrWindow::Workspace( int32_t aWidth, int32_t height )
    {
        auto &lIO = ImGui::GetIO();

        math::vec2 lWorkspacePosition = UI::GetCurrentCursorScreenPosition();
        math::vec2 lCursorPosition    = UI::GetCurrentCursorPosition();

        // UI::SameLine();
        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.0f, 0.0f, 0.0f, 0.0f } );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 1.0f, 1.0f, 1.0f, 0.01f } );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 1.0f, 1.0f, 1.0f, 0.02f } );

        if( mCurrentScript )
        {
            if( !mCurrentScriptIsRunning )
            {
                if( ImGui::ImageButton( (ImTextureID)mPlayIconHandle.Handle->GetVkDescriptorSet(), ImVec2{ 22.0f, 22.0f },
                                        ImVec2{ 0.0f, 0.0f }, ImVec2{ 1.0f, 1.0f }, 0, ImVec4{ 0.0f, 0.0f, 0.0f, 0.0f },
                                        ImVec4{ 0.0f, 1.0f, 0.0f, 0.8f } ) )
                {
                    mCurrentScript->CallMethod( "BeginScenario" );
                    mCurrentScriptIsRunning = true;
                }
            }
            else
            {
                if( ImGui::ImageButton( (ImTextureID)mPauseIconHandle.Handle->GetVkDescriptorSet(), ImVec2{ 22.0f, 22.0f },
                                        ImVec2{ 0.0f, 0.0f }, ImVec2{ 1.0f, 1.0f }, 0, ImVec4{ 0.0f, 0.0f, 0.0f, 0.0f },
                                        ImVec4{ 1.0f, .2f, 0.0f, 0.8f } ) )
                {
                    mCurrentScript->CallMethod( "EndScenario" );
                    mCurrentScriptIsRunning = false;
                }
            }
        }
        else if( mActiveWorld->GetState() == OtdrScene::eSceneState::EDITING )
        {

            if( ImGui::ImageButton( (ImTextureID)mPlayIconHandle.Handle->GetVkDescriptorSet(), ImVec2{ 22.0f, 22.0f },
                                    ImVec2{ 0.0f, 0.0f }, ImVec2{ 1.0f, 1.0f }, 0, ImVec4{ 0.0f, 0.0f, 0.0f, 0.0f },
                                    ImVec4{ 0.0f, 1.0f, 0.0f, 0.8f } ) )
            {
                if( OnBeginScenario ) OnBeginScenario();

                mActiveWorld = New<OtdrScene>( mWorld );
                mActiveWorld->BeginScenario();
            }
        }
        else
        {
            if( ImGui::ImageButton( (ImTextureID)mPauseIconHandle.Handle->GetVkDescriptorSet(), ImVec2{ 22.0f, 22.0f },
                                    ImVec2{ 0.0f, 0.0f }, ImVec2{ 1.0f, 1.0f }, 0, ImVec4{ 0.0f, 0.0f, 0.0f, 0.0f },
                                    ImVec4{ 1.0f, .2f, 0.0f, 0.8f } ) )
            {
                if( OnEndScenario ) OnEndScenario();

                mActiveWorld->EndScenario();
                mActiveWorld = mWorld;
            }
        }

        ImGui::PopStyleColor();
        ImGui::PopStyleColor();
        ImGui::PopStyleColor();

        math::vec2  l3DViewPosition = UI::GetCurrentCursorScreenPosition();
        math::ivec2 l3DViewSize     = UI::GetAvailableContentSpace();
        mWorkspaceAreaSize          = l3DViewSize;
    }

    void OtdrWindow::Console( int32_t aWidth, int32_t height )
    {
        const float           lTextBaseHeight = ImGui::GetTextLineHeightWithSpacing();
        const ImGuiTableFlags flags           = ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY |
                                      ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_Resizable |
                                      ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable;

        auto  &l_Logs     = SE::Logging::GetLogMessages();
        ImVec2 outer_size = ImVec2( 0.0f, height );
        if( ImGui::BeginTable( "table_scrolly", 3, flags, outer_size ) )
        {
            ImGui::TableSetupScrollFreeze( 2, 1 );
            ImGui::TableSetupColumn( "", ImGuiTableColumnFlags_WidthFixed, lTextBaseHeight );
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

} // namespace SE::OtdrEditor