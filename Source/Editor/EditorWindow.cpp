#include "EditorWindow.h"

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

// #include "ShortWaveformDisplay.h"

namespace SE::Editor
{

    using namespace SE::Core;
    using namespace SE::Core::EntityComponentSystem::Components;

    template <typename _SliderType>
    class Slider
    {
      public:
        std::string ID = "";

        _SliderType MinValue{};
        _SliderType MaxValue{};
        _SliderType CurrentValue{};

        std::string Format = "";

        bool Changed = false;

        Slider() = default;
        Slider( std::string a_ID )
            : ID{ a_ID }
        {
        }
        ~Slider() = default;

        void Display( _SliderType *mValue ) { Changed = UI::Slider( ID, Format.c_str(), MinValue, MaxValue, mValue ); }
    };

    template <typename _ValueChooserType>
    class PropertyEditor
    {
      public:
        std::string       Label;
        float             LabelWidth;
        _ValueChooserType ValueChooser;

        PropertyEditor( std::string ID ) { ValueChooser = _ValueChooserType( ID ); }

        template <typename... _ArgTypes>
        void Display( _ArgTypes... a_ArgList )
        {
            float l_Width = UI::GetAvailableContentSpace().x;
            ImGui::AlignTextToFramePadding();
            Text( Label );
            UI::SameLine();
            ImGui::SetNextItemWidth( l_Width - LabelWidth );
            UI::SetCursorPosition( math::vec2( LabelWidth, UI::GetCurrentCursorPosition().y ) );
            ValueChooser.Display( std::forward<_ArgTypes>( a_ArgList )... );
        }
    };

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

    void EditorWindow::ConfigureUI()
    {
        m_SceneElementEditor                = SceneElementEditor( mGraphicContext );
        m_SceneHierarchyPanel.ElementEditor = m_SceneElementEditor;
        m_CurrentPanel                      = SidePanelID::SENSOR_CONFIGURATION;
        mContentBrowser                     = ContentBrowser( mGraphicContext, mUIOverlay, "C:\\GitLab\\SpockEngine\\Saved" );

        {
            SE::Core::sTextureCreateInfo lTextureCreateInfo{};
            TextureData2D        lTextureData( lTextureCreateInfo, "C:\\GitLab\\SpockEngine\\Saved\\Resources\\Icons\\Play.png" );
            sTextureSamplingInfo lSamplingInfo{};
            SE::Core::TextureSampler2D lTextureSampler = SE::Core::TextureSampler2D( lTextureData, lSamplingInfo );

            auto lTexture    = CreateTexture2D( mGraphicContext, lTextureData );
            m_PlayIcon       = CreateSampler2D( mGraphicContext, lTexture, lSamplingInfo );
            m_PlayIconHandle = mUIOverlay->CreateTextureHandle( m_PlayIcon );
        }

        {
            SE::Core::sTextureCreateInfo lTextureCreateInfo{};
            TextureData2D        lTextureData( lTextureCreateInfo, "C:\\GitLab\\SpockEngine\\Saved\\Resources\\Icons\\Pause.png" );
            sTextureSamplingInfo lSamplingInfo{};
            SE::Core::TextureSampler2D lTextureSampler = SE::Core::TextureSampler2D( lTextureData, lSamplingInfo );

            auto lTexture     = CreateTexture2D( mGraphicContext, lTextureData );
            m_PauseIcon       = CreateSampler2D( mGraphicContext, lTexture, lSamplingInfo );
            m_PauseIconHandle = mUIOverlay->CreateTextureHandle( m_PauseIcon );
        }

        {
            SE::Core::sTextureCreateInfo lTextureCreateInfo{};
            TextureData2D        lTextureData( lTextureCreateInfo, "C:\\GitLab\\SpockEngine\\Saved\\Resources\\Icons\\Play.png" );
            sTextureSamplingInfo lSamplingInfo{};
            SE::Core::TextureSampler2D lTextureSampler = SE::Core::TextureSampler2D( lTextureData, lSamplingInfo );

            auto lTexture               = CreateTexture2D( mGraphicContext, lTextureData );
            m_DefaultTextureImage       = CreateSampler2D( mGraphicContext, lTexture, lSamplingInfo );
            m_DefaultTextureImageHandle = mUIOverlay->CreateTextureHandle( m_PlayIcon );
        }

        {
            SE::Core::sTextureCreateInfo lTextureCreateInfo{};
            TextureData2D        lTextureData( lTextureCreateInfo, "C:\\GitLab\\SpockEngine\\Saved\\Resources\\Icons\\Camera.png" );
            sTextureSamplingInfo lSamplingInfo{};
            SE::Core::TextureSampler2D lTextureSampler = SE::Core::TextureSampler2D( lTextureData, lSamplingInfo );

            auto lTexture      = CreateTexture2D( mGraphicContext, lTextureData );
            m_CameraIcon       = CreateSampler2D( mGraphicContext, lTexture, lSamplingInfo );
            m_CameraIconHandle = mUIOverlay->CreateTextureHandle( m_CameraIcon );
        }
    }

    EditorWindow::EditorWindow( Ref<IGraphicContext> aGraphicContext, Ref<UIContext> aUIOverlay )
        : mGraphicContext{ aGraphicContext }
        , mUIOverlay{ aUIOverlay }
    {
        ConfigureUI();
    }

    void EditorWindow::UpdateSceneViewport( ImageHandle a_SceneViewport ) { m_SceneViewport = a_SceneViewport; }

    EditorWindow &EditorWindow::AddMenuItem( std::string l_Icon, std::string l_Title, std::function<bool()> l_Action )
    {
        m_MainMenuItems.push_back( { l_Icon, l_Title, l_Action } );
        return *this;
    }

    bool EditorWindow::Display()
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

        // if( m_NewMaterial.Visible )
        //     ImGui::OpenPopup( "NEW MATERIAL..." );
        // m_NewMaterial.Display();

        static bool p_open_3 = true;
        if( ImGui::Begin( "3D VIEW", &p_open_3, ImGuiWindowFlags_None ) )
        {
            auto lWorkspaceAreaSize = UI::GetAvailableContentSpace();
            Workspace( lWorkspaceAreaSize.x, lWorkspaceAreaSize.y );
        }
        ImGui::End();

        static bool p_open_30 = true;
        if( ImGui::Begin( "NEW 3D VIEW", &p_open_30, ImGuiWindowFlags_None ) )
        {
            auto lWorkspaceAreaSize = UI::GetAvailableContentSpace();
            WorkspaceNew( lWorkspaceAreaSize.x, lWorkspaceAreaSize.y );
        }
        ImGui::End();

        if( ImGui::Begin( "LIGHT PANEL", NULL, ImGuiWindowFlags_None ) )
        {
            auto     lWorkspaceAreaSize = UI::GetAvailableContentSpace();
            uint32_t i                  = 0;
            ActiveWorld->ForEach<sLightComponent>(
                [&]( auto aEntity, auto &aComponent )
                {
                    Text( aEntity.Get<sTag>().mValue );
                    UI::SameLine();
                    auto x = fmt::format( "##{}", i++ );
                    UI::Checkbox( x.c_str(), &aComponent.mIsOn );
                } );
        }
        ImGui::End();

        static bool p_open_4 = true;
        if( ImGui::Begin( "OBSERVER CAMERA", &p_open_4, ImGuiWindowFlags_None ) )
        {
            math::vec2 l_WorkspacePosition = UI::GetCurrentCursorScreenPosition();
            math::vec2 l_CursorPosition    = UI::GetCurrentCursorPosition();

            math::vec2 l_Size     = { 350.0f, 350.0f };
            math::vec2 l_Position = l_WorkspacePosition + l_CursorPosition + math::vec2( 40.0f, -20.0f );

            ImGui::PushStyleColor( ImGuiCol_WindowBg, ImVec4{ 12.0f / 255.0f, 12.0f / 255.0f, 12.0f / 255.0f, 1.0f } );
            ImGui::PushStyleVar( ImGuiStyleVar_WindowRounding, 0.0f );
            ImGui::PushStyleVar( ImGuiStyleVar_WindowBorderSize, 0.0f );
            ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2( 5.0f, 7.0f ) );
            ImGui::SetNextWindowSize( ImVec2{ l_Size.x, l_Size.y } );
            ImGui::SetNextWindowPos( ImVec2{ l_Position.x, l_Position.y } );
            constexpr ImGuiWindowFlags lStatusBarFlags = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar |
                                                         ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                                                         ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoNavFocus;

            math::ivec2        l_WindowSize = UI::GetAvailableContentSpace();
            ImGuiTreeNodeFlags l_Flags      = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_AllowItemOverlap;
            float              l_LabelSize  = 175.0f;

            Text( "Viewport: {} x {}", m_WorkspaceAreaSize.x, m_WorkspaceAreaSize.y );
            {
                const float  lAspect = static_cast<float>( m_WorkspaceAreaSize.x ) / static_cast<float>( m_WorkspaceAreaSize.y );
                static float lFoVY   = 48.7;
                static float lFoVX   = lFoVY * lAspect;
                Text( "Field of view:" );
                UI::SameLine();
                ImVec2 l_CursorPosition = ImGui::GetCursorPos();
                UI::SetCursorPosition( ImVec2{ l_LabelSize, l_CursorPosition.y } + ImVec2( 0.0f, -5.0f ) );
                ImGui::SetNextItemWidth( l_WindowSize.x - l_LabelSize );
                UI::Slider( "##camera_fov", "%.2f", 1.0f, 180.0f, &lFoVX );

                WorldRenderer->SetProjection( math::Perspective( math::radians( lFoVX ), lAspect, 0.01f, 100000.0f ) );
            }

            if( World->Environment.Has<sAmbientLightingComponent>() )
            {
                auto &l_AmbientLightComponent = World->Environment.Get<sAmbientLightingComponent>();
                UI::ColorChooser( "Ambient light color:", 175, l_AmbientLightComponent.Color );
                Text( "Ambient intensity:" );
                UI::SameLine();
                ImVec2 l_CursorPosition = ImGui::GetCursorPos();
                UI::SetCursorPosition( ImVec2{ l_LabelSize, l_CursorPosition.y } + ImVec2( 0.0f, -5.0f ) );
                ImGui::SetNextItemWidth( l_WindowSize.x - l_LabelSize );
                UI::Slider( "##ambient_intensity", "%.3f", 0.0f, 0.2f, &l_AmbientLightComponent.Intensity );

                WorldRenderer->SetAmbientLighting( math::vec4( l_AmbientLightComponent.Color, l_AmbientLightComponent.Intensity ) );
            }

            static float lExposure = 4.5f;
            static float lGamma    = 2.2f;
            ImGui::SliderFloat( "Exposure", &lExposure, 0.1f, 10.0f );
            ImGui::SliderFloat( "Gamma", &lGamma, 0.1f, 4.0f );

            WorldRenderer->SetExposure( lExposure );
            WorldRenderer->SetGamma( lGamma );
            ImGui::PopStyleVar( 3 );
            ImGui::PopStyleColor();
        }
        ImGui::End();

#if 0
        if( ImGui::Begin( "TRACKS", &p_open, ImGuiWindowFlags_None ) )
        {
            auto lWorkspaceAreaSize = UI::GetAvailableContentSpace();
        }
        ImGui::End();

        if( ImGui::Begin( "MATERIAL", &p_open, ImGuiWindowFlags_None ) )
        {
            auto  l_WindowSize       = UI::GetAvailableContentSpace();
            auto  lWorkspaceAreaSize = UI::GetAvailableContentSpace();
            float l_LabelSize        = 175.0f;
            auto  l_DrawList         = ImGui::GetWindowDrawList();

            {
            }

            UI::Text( "Material name:" );
            char buf[128] = { 0 };
            std::strncpy( buf, "NAME", std::min( std::size_t( 4 ), std::size_t( 128 ) ) );
            if( ImGui::InputText( "##material_name", buf, ARRAYSIZE( buf ), ImGuiInputTextFlags_EnterReturnsTrue ) )
            {
            }

            UI::Text( "Material preview:" );
            UI::Image( m_CameraIconHandle, math::vec2{ 200, 200 } );

            UI::Text( "Line width" );
            ImGui::SameLine();
            float l_LineWidth = 0.0f;
            UI::SetCursorPosition( ImVec2( static_cast<float>( l_WindowSize.x ) - 256.0f, ImGui::GetCursorPos().y - 5.0f ) );
            UI::Slider( "##line_width", "%.2f", 1.0f, 5.0f, &l_LineWidth );

            Text( "Use alpha mask" );
            ImGui::SameLine();
            bool l_AlphaMask = false;
            UI::SetCursorPosition( ImVec2( static_cast<float>( l_WindowSize.x ) - 32.0f, ImGui::GetCursorPos().y - 5.0f ) );
            UI::Checkbox( "##use_alpha_mask", &l_AlphaMask );

            UI::Text( "Alpha mask threshold" );
            ImGui::SameLine();
            float l_AlphaMaskThreshold = 0.0f;
            UI::SetCursorPosition( ImVec2( static_cast<float>( l_WindowSize.x ) - 256.0f, ImGui::GetCursorPos().y - 5.0f ) );
            UI::Slider( "##alpha_threshold", "%.2f", 0.0f, 1.0f, &l_AlphaMaskThreshold );

            {
                static Ref<Texture2D> lBaseColorTexture = nullptr;
                static ImageHandle                  lBaseColorTextureHandle{};
                UI::Text( "Base color:" );
                ImGui::Columns( 2, NULL, false );
                ImGui::SetColumnWidth( 0, 150 );
                auto l_TopLeft     = ImGui::GetCursorScreenPos();
                auto l_BottomRight = ImGui::GetCursorScreenPos() + ImVec2{ 128, 128 };
                if( lBaseColorTextureHandle.Handle )
                    UI::Image( lBaseColorTextureHandle, math::vec2{ 128, 128 } );
                else
                    UI::Image( m_DefaultTextureImageHandle, math::vec2{ 128, 128 } );

                if( ImGui::BeginDragDropTarget() )
                {
                    if( const ImGuiPayload *payload = ImGui::AcceptDragDropPayload( "CONTENT_BROWSER_ITEM" ) )
                    {
                        fs::path lItemPath( UTF16ToAscii( (const char *)payload->Data, payload->DataSize ) );
                        lBaseColorTexture = LoadImagePreview( mGraphicContext, lItemPath );
                        UpdateImageHandle( mUIOverlay, lBaseColorTextureHandle, lBaseColorTexture );
                    }
                    ImGui::EndDragDropTarget();
                }
                l_DrawList->AddRect( l_TopLeft, l_BottomRight, IM_COL32( 15, 15, 15, 255 ) );
                ImGui::NextColumn();
                UI::Text( "Base color factor:" );
                math::vec4 lBaseColorFactor = 0xffffffff_rgbaf;
                UI::VectorComponentEditor( "", lBaseColorFactor, 1.0, 0.0f );
                ImGui::Columns( 1 );
            }

            {
                static Ref<Texture2D> lEmissiveTexture = nullptr;
                static ImageHandle                  lEmissiveTextureHandle{};
                UI::Text( "Emissive:" );
                ImGui::Columns( 2, NULL, false );
                ImGui::SetColumnWidth( 0, 150 );
                auto l_TopLeft     = ImGui::GetCursorScreenPos();
                auto l_BottomRight = ImGui::GetCursorScreenPos() + ImVec2{ 128, 128 };
                if( lEmissiveTextureHandle.Handle )
                    UI::Image( lEmissiveTextureHandle, math::vec2{ 128, 128 } );
                else
                    UI::Image( m_DefaultTextureImageHandle, math::vec2{ 128, 128 } );
                if( ImGui::BeginDragDropTarget() )
                {
                    if( const ImGuiPayload *payload = ImGui::AcceptDragDropPayload( "CONTENT_BROWSER_ITEM" ) )
                    {
                        fs::path lItemPath( UTF16ToAscii( (const char *)payload->Data, payload->DataSize ) );
                        lEmissiveTexture = LoadImagePreview( mGraphicContext, lItemPath );
                        UpdateImageHandle( mUIOverlay, lEmissiveTextureHandle, lEmissiveTexture );
                    }
                    ImGui::EndDragDropTarget();
                }
                l_DrawList->AddRect( l_TopLeft, l_BottomRight, IM_COL32( 15, 15, 15, 255 ) );
                ImGui::NextColumn();
                UI::Text( "Emissive factor:" );
                math::vec4 lBaseColorFactor = 0xffffffff_rgbaf;
                UI::VectorComponentEditor( "", lBaseColorFactor, 1.0, 0.0f );
                ImGui::Columns( 1 );
            }

            {
                static Ref<Texture2D> lNormalsTexture = nullptr;
                static ImageHandle                  lNormalsTextureHandle{};
                UI::Text( "Normals:" );
                auto l_TopLeft     = ImGui::GetCursorScreenPos();
                auto l_BottomRight = ImGui::GetCursorScreenPos() + ImVec2{ 128, 128 };
                if( lNormalsTextureHandle.Handle )
                    UI::Image( lNormalsTextureHandle, math::vec2{ 128, 128 } );
                else
                    UI::Image( m_DefaultTextureImageHandle, math::vec2{ 128, 128 } );
                if( ImGui::BeginDragDropTarget() )
                {
                    if( const ImGuiPayload *payload = ImGui::AcceptDragDropPayload( "CONTENT_BROWSER_ITEM" ) )
                    {
                        fs::path lItemPath( UTF16ToAscii( (const char *)payload->Data, payload->DataSize ) );
                        lNormalsTexture = LoadImagePreview( mGraphicContext, lItemPath );
                        UpdateImageHandle( mUIOverlay, lNormalsTextureHandle, lNormalsTexture );
                    }
                    ImGui::EndDragDropTarget();
                }
                l_DrawList->AddRect( l_TopLeft, l_BottomRight, IM_COL32( 15, 15, 15, 255 ) );
            }

            {
                static Ref<Texture2D> lOcclusionTexture = nullptr;
                static ImageHandle                  lOcclusionTextureHandle{};
                UI::Text( "Occlusion:" );
                ImGui::Columns( 2, NULL, false );
                ImGui::SetColumnWidth( 0, 150 );
                auto l_TopLeft     = ImGui::GetCursorScreenPos();
                auto l_BottomRight = ImGui::GetCursorScreenPos() + ImVec2{ 128, 128 };
                if( lOcclusionTextureHandle.Handle )
                    UI::Image( lOcclusionTextureHandle, math::vec2{ 128, 128 } );
                else
                    UI::Image( m_DefaultTextureImageHandle, math::vec2{ 128, 128 } );
                if( ImGui::BeginDragDropTarget() )
                {
                    if( const ImGuiPayload *payload = ImGui::AcceptDragDropPayload( "CONTENT_BROWSER_ITEM" ) )
                    {
                        fs::path lItemPath( UTF16ToAscii( (const char *)payload->Data, payload->DataSize ) );
                        lOcclusionTexture = LoadImagePreview( mGraphicContext, lItemPath );
                        UpdateImageHandle( mUIOverlay, lOcclusionTextureHandle, lOcclusionTexture );
                    }
                    ImGui::EndDragDropTarget();
                }
                l_DrawList->AddRect( l_TopLeft, l_BottomRight, IM_COL32( 15, 15, 15, 255 ) );
                ImGui::NextColumn();
                UI::Text( "Occlusion strength:" );
                float l_OcclusionStrength = 0.0f;
                UI::Slider( "##occlusion", "%.2f", 0.0f, 1.0f, &l_OcclusionStrength );
                ImGui::Columns( 1 );
            }

            {
                static Ref<Texture2D> lPhysicalTexture = nullptr;
                static ImageHandle                  lPhysicalTextureHandle{};
                UI::Text( "Physical properties:" );
                ImGui::Columns( 2, NULL, false );
                ImGui::SetColumnWidth( 0, 150 );
                auto l_TopLeft     = ImGui::GetCursorScreenPos();
                auto l_BottomRight = ImGui::GetCursorScreenPos() + ImVec2{ 128, 128 };
                if( lPhysicalTextureHandle.Handle )
                    UI::Image( lPhysicalTextureHandle, math::vec2{ 128, 128 } );
                else
                    UI::Image( m_DefaultTextureImageHandle, math::vec2{ 128, 128 } );
                if( ImGui::BeginDragDropTarget() )
                {
                    if( const ImGuiPayload *payload = ImGui::AcceptDragDropPayload( "CONTENT_BROWSER_ITEM" ) )
                    {
                        fs::path lItemPath( UTF16ToAscii( (const char *)payload->Data, payload->DataSize ) );
                        lPhysicalTexture = LoadImagePreview( mGraphicContext, lItemPath );
                        UpdateImageHandle( mUIOverlay, lPhysicalTextureHandle, lPhysicalTexture );
                    }
                    ImGui::EndDragDropTarget();
                }
                l_DrawList->AddRect( l_TopLeft, l_BottomRight, IM_COL32( 15, 15, 15, 255 ) );
                ImGui::NextColumn();
                float l_Roughness = 0.0f;
                UI::Text( "Roughness factor" );
                UI::Slider( "##roughness", "%.2f", 0.0f, 1.0f, &l_Roughness );
                float l_Metalness = 0.0f;
                UI::Text( "Metalness factor" );
                UI::Slider( "##metalness", "%.2f", 0.0f, 1.0f, &l_Metalness );
                ImGui::Columns( 1 );
            }
        }
        ImGui::End();
#endif

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
            auto l_WindowPropertiesSize = UI::GetAvailableContentSpace();
            m_SceneHierarchyPanel.World = ActiveWorld;
            m_SceneHierarchyPanel.Display( l_WindowPropertiesSize.x, l_WindowPropertiesSize.y );
        }
        ImGui::End();

        if( ImGui::Begin( "CONTENT BROWSER", &p_open, ImGuiWindowFlags_None ) )
        {
            mContentBrowser.Display();
        }
        ImGui::End();

        if( ImGui::Begin( "PROPERTIES", &p_open, ImGuiWindowFlags_None ) )
        {
            auto l_WindowPropertiesSize = UI::GetAvailableContentSpace();
            m_SceneHierarchyPanel.ElementEditor.Display( l_WindowPropertiesSize.x, l_WindowPropertiesSize.y );
        }
        ImGui::End();

        ImGui::ShowDemoWindow();

        ImGui::PushStyleColor( ImGuiCol_WindowBg, ImVec4{ 102.0f / 255.0f, 0.0f, 204.0f / 255.0f, 1.0f } );
        ImGui::PushStyleVar( ImGuiStyleVar_WindowRounding, 0.0f );
        ImGui::PushStyleVar( ImGuiStyleVar_WindowBorderSize, 0.0f );
        ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2( 5.0f, 7.0f ) );
        ImGui::SetNextWindowSize( ImVec2{ viewport->WorkSize.x, StatusBarHeight } );
        ImGui::SetNextWindowPos( ImVec2{ viewport->Pos.x, viewport->Pos.y + viewport->WorkSize.y + 1 } );
        ImGui::SetNextWindowViewport( viewport->ID );
        constexpr ImGuiWindowFlags lStatusBarFlags =
            ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

        ImGui::Begin( "##STATUS_BAR", &p_open, lStatusBarFlags );
        ImGui::PopStyleVar( 3 );
        ImGui::PopStyleColor();
        auto l_WindowSize = UI::GetAvailableContentSpace();
        {
            if( m_LastFPS > 0 )
                Text( "Render: {} fps ({:.2f} ms/frame)", m_LastFPS, ( 1000.0f / m_LastFPS ) );
            else
                Text( "Render: 0 fps (0 ms)" );
        }
        ImGui::End();

        return lRequestQuit;
    }

    bool EditorWindow::RenderMainMenu()
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
                if( lFilePath.has_value() ) LoadScenario( lFilePath.value() );
            }
            if( UI::MenuItem( fmt::format( "{} New material", ICON_FA_PLUS_CIRCLE ).c_str(), NULL ) )
            {
                // m_NewMaterial.Visible = true;
            }
            if( UI::MenuItem( fmt::format( "{} Import model...", ICON_FA_PLUS_CIRCLE ).c_str(), NULL ) )
            {
                auto lFilePath = FileDialogs::OpenFile( SE::Core::Engine::GetInstance()->GetMainApplicationWindow(),
                                                        "glTf Files (*.gltf)\0*.gltf\0All Files (*.*)\0*.*\0" );
                if( lFilePath.has_value() ) ImportModel( lFilePath.value() );
            }
            if( UI::MenuItem( fmt::format( "{} Save Scene", ICON_FA_PLUS_CIRCLE ).c_str(), NULL ) )
            {
                // m_NewMaterial.Visible = true;
                World->SaveAs( fs::path( "C:\\GitLab\\SpockEngine\\Saved" ) / "TEST" / "SCENE" );
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

    math::ivec2 EditorWindow::GetWorkspaceAreaSize() { return m_WorkspaceAreaSize; }

    class ManipulationTypeChooser
    {
      public:
        std::string                    ID = "";
        UI::ComboBox<ManipulationType> Dropdown;

      public:
        ManipulationTypeChooser() = default;
        ManipulationTypeChooser( std::string a_ID )
            : ID{ a_ID }
            , Dropdown{ UI::ComboBox<ManipulationType>( a_ID ) } {};

        ~ManipulationTypeChooser() = default;

        void Display( ManipulationType &Current )
        {
            auto it = std::find( Dropdown.Values.begin(), Dropdown.Values.end(), Current );
            if( it != Dropdown.Values.end() ) Dropdown.CurrentItem = std::distance( Dropdown.Values.begin(), it );

            Dropdown.Display();
        }
    };

    void EditorWindow::ClearScene() {}

    void EditorWindow::LoadScenario( fs::path aPath )
    {
        auto lName = aPath.filename().string();
        auto lExt  = aPath.extension().string();

        Ref<sImportedModel> lImporter = nullptr;
        if( lExt == ".gltf" )
            lImporter = New<GlTFImporter>( aPath );
        else if( lExt == ".obj" )
            lImporter = New<ObjImporter>( aPath );

        if( lImporter != nullptr )
        {
            auto lNewModel = World->LoadModel( lImporter, math::mat4( 1.0f ), lName );
            // lNewModel.Add<LockComponent>();
            World->ForEach<sStaticMeshComponent>( [&]( auto aEntity, auto &aComponent )
                                                  { World->MarkAsRayTracingTarget( aEntity ); } );
        }
    }

    void EditorWindow::ImportModel( fs::path aPath )
    {
        // auto lName = aPath.filename().string();
        // auto lExt  = aPath.extension().string();

        // Ref<sImportedModel> lModelData = nullptr;
        // if( lExt == ".gltf" )
        //     lModelData = New<GlTFImporter>( aPath );
        // else if( lExt == ".obj" )
        //     lModelData = New<ObjImporter>( aPath );

        // if( !lModelData ) return;

        // auto lModelName = aPath.stem().string();
        // if( !fs::exists( mModelsPath / lModelName ) ) fs::create_directories( mModelsPath / lModelName );
        // if( !fs::exists( mModelsPath / lModelName / "Materials" ) ) fs::create_directories( mModelsPath / lModelName / "Materials"
        // ); if( !fs::exists( mModelsPath / lModelName / "Meshes" ) ) fs::create_directories( mModelsPath / lModelName / "Meshes" );
        // if( !fs::exists( mModelsPath / lModelName / "Animations" ) ) fs::create_directories( mModelsPath / lModelName / "Animations"
        // );

        // for( auto &lMaterial : lModelData->mMaterials )
        // {
        //     std::string lSerializedMeshName = fmt::format( "{}.material", lMaterial.mName );
        //     auto        lPath               = mModelsPath / lModelName / "Materials" / lSerializedMeshName;

        //     BinaryAsset                    lBinaryDataFile;
        //     std::vector<sAssetIndex>       lAssetIndex{};
        //     std::vector<std::vector<char>> lPackets{};

        //     sMaterial lNewMaterial{};
        //     lNewMaterial.mName                         = lMaterial.mName;
        //     lNewMaterial.mType                         = eMaterialType::Opaque;
        //     lNewMaterial.mLineWidth                    = 1.0f;
        //     lNewMaterial.mIsTwoSided                   = lMaterial.mConstants.mIsTwoSided;
        //     lNewMaterial.mUseAlphaMask                 = false;
        //     lNewMaterial.mAlphaThreshold               = 0.5;
        //     lNewMaterial.mBaseColorFactor              = lMaterial.mConstants.mBaseColorFactor;
        //     lNewMaterial.mBaseColorTexture.mTextureID  = 0;
        //     lNewMaterial.mBaseColorTexture.mUVChannel  = 0;
        //     lNewMaterial.mNormalsTexture.mTextureID    = 1;
        //     lNewMaterial.mNormalsTexture.mUVChannel    = 0;
        //     lNewMaterial.mRoughnessFactor              = lMaterial.mConstants.mRoughnessFactor;
        //     lNewMaterial.mMetallicFactor               = lMaterial.mConstants.mMetallicFactor;
        //     lNewMaterial.mMetalRoughTexture.mTextureID = 2;
        //     lNewMaterial.mMetalRoughTexture.mUVChannel = 0;
        //     lNewMaterial.mOcclusionStrength            = 0.0f;
        //     lNewMaterial.mOcclusionTexture.mTextureID  = 3;
        //     lNewMaterial.mOcclusionTexture.mUVChannel  = 0;
        //     lNewMaterial.mEmissiveFactor               = lMaterial.mConstants.mEmissiveFactor;
        //     lNewMaterial.mEmissiveTexture.mTextureID   = 4;
        //     lNewMaterial.mEmissiveTexture.mUVChannel   = 0;

        //     lPackets.push_back( lBinaryDataFile.Package( lNewMaterial ) );
        //     lAssetIndex.emplace_back();
        //     lAssetIndex.back().mType      = eAssetType::MATERIAL_DATA;
        //     lAssetIndex.back().mByteStart = 0;
        //     lAssetIndex.back().mByteEnd   = sizeof( sMaterial );

        //     sImportedTexture lBaseColorTexture{};
        //     if( lMaterial.mTextures.mBaseColorTexture.TextureID != std::numeric_limits<uint32_t>::max() )
        //         lBaseColorTexture = lModelData->mTextures[lMaterial.mTextures.mBaseColorTexture.TextureID];

        //     lPackets.push_back( lBinaryDataFile.Package( lBaseColorTexture ) );

        //     lAssetIndex.emplace_back();
        //     lAssetIndex.back().mType      = eAssetType::KTX_TEXTURE_2D;
        //     lAssetIndex.back().mByteStart = 0;
        //     lAssetIndex.back().mByteEnd   = lPackets.back().size();

        //     sImportedTexture lNormalTexture{};
        //     if( lMaterial.mTextures.mNormalTexture.TextureID != std::numeric_limits<uint32_t>::max() )
        //         lNormalTexture = lModelData->mTextures[lMaterial.mTextures.mNormalTexture.TextureID];
        //     lPackets.push_back( lBinaryDataFile.Package( lNormalTexture ) );

        //     lAssetIndex.emplace_back();
        //     lAssetIndex.back().mType      = eAssetType::KTX_TEXTURE_2D;
        //     lAssetIndex.back().mByteStart = 0;
        //     lAssetIndex.back().mByteEnd   = lPackets.back().size();

        //     sImportedTexture lMetallicRoughnessTexture{};
        //     if( lMaterial.mTextures.mMetallicRoughnessTexture.TextureID != std::numeric_limits<uint32_t>::max() )
        //         lMetallicRoughnessTexture = lModelData->mTextures[lMaterial.mTextures.mMetallicRoughnessTexture.TextureID];
        //     lPackets.push_back( lBinaryDataFile.Package( lMetallicRoughnessTexture ) );

        //     lAssetIndex.emplace_back();
        //     lAssetIndex.back().mType      = eAssetType::KTX_TEXTURE_2D;
        //     lAssetIndex.back().mByteStart = 0;
        //     lAssetIndex.back().mByteEnd   = lPackets.back().size();

        //     sImportedTexture lOcclusionTexture{};
        //     if( lMaterial.mTextures.mOcclusionTexture.TextureID != std::numeric_limits<uint32_t>::max() )
        //         lOcclusionTexture = lModelData->mTextures[lMaterial.mTextures.mOcclusionTexture.TextureID];
        //     lPackets.push_back( lBinaryDataFile.Package( lOcclusionTexture ) );

        //     lAssetIndex.emplace_back();
        //     lAssetIndex.back().mType      = eAssetType::KTX_TEXTURE_2D;
        //     lAssetIndex.back().mByteStart = 0;
        //     lAssetIndex.back().mByteEnd   = lPackets.back().size();

        //     sImportedTexture lEmissiveTexture{};
        //     if( lMaterial.mTextures.mEmissiveTexture.TextureID != std::numeric_limits<uint32_t>::max() )
        //         lEmissiveTexture = lModelData->mTextures[lMaterial.mTextures.mEmissiveTexture.TextureID];

        //     lPackets.push_back( lBinaryDataFile.Package( lEmissiveTexture ) );
        //     lAssetIndex.emplace_back();
        //     lAssetIndex.back().mType      = eAssetType::KTX_TEXTURE_2D;
        //     lAssetIndex.back().mByteStart = 0;
        //     lAssetIndex.back().mByteEnd   = lPackets.back().size();

        //     uint32_t lAssetCount = static_cast<uint32_t>( lAssetIndex.size() );

        //     uint32_t lCurrentByte = BinaryAsset::GetMagicLength() + sizeof( uint32_t ) + lAssetIndex.size() * sizeof( sAssetIndex );
        //     for( uint32_t i = 0; i < lAssetCount; i++ )
        //     {
        //         lAssetIndex[i].mByteStart = lCurrentByte;
        //         lCurrentByte = lAssetIndex[i].mByteEnd = lAssetIndex[i].mByteStart + static_cast<uint32_t>( lPackets[i].size() );
        //     }

        //     auto *lMagic       = BinaryAsset::GetMagic();
        //     auto  lMagicLength = BinaryAsset::GetMagicLength();
        //     auto  lOutFile     = std::ofstream( lPath.string(), std::ofstream::binary );
        //     lOutFile.write( (const char *)lMagic, lMagicLength );
        //     lOutFile.write( (const char *)&lAssetCount, sizeof( uint32_t ) );
        //     lOutFile.write( (const char *)lAssetIndex.data(), lAssetIndex.size() * sizeof( sAssetIndex ) );

        //     for( auto &lPacket : lPackets ) lOutFile.write( (const char *)lPacket.data(), lPacket.size() );
        // }

        // for( auto &lMesh : lModelData->mMeshes )
        // {
        //     BinaryAsset lBinaryDataFile;

        //     std::string lSerializedMeshName = fmt::format( "{}.mesh", lMesh.mName );
        //     auto        lPath               = mModelsPath / lModelName / "Meshes" / lSerializedMeshName;

        //     std::vector<VertexData> lVertices( lMesh.mPositions.size() );
        //     for( uint32_t i = 0; i < lMesh.mPositions.size(); i++ )
        //     {
        //         lVertices[i].Position    = lMesh.mPositions[i];
        //         lVertices[i].Normal      = lMesh.mNormals[i];
        //         lVertices[i].TexCoords_0 = lMesh.mUV0[i];
        //         lVertices[i].TexCoords_1 = lMesh.mUV1[i];
        //         lVertices[i].Bones       = lMesh.mJoints[i];
        //         lVertices[i].Weights     = lMesh.mWeights[i];
        //     }

        //     auto lMeshData = lBinaryDataFile.Package( lVertices, lMesh.mIndices );

        //     sAssetIndex lMeshAssetIndexEntry{};
        //     lMeshAssetIndexEntry.mType      = eAssetType::MESH_DATA;
        //     lMeshAssetIndexEntry.mByteStart = 0;
        //     lMeshAssetIndexEntry.mByteEnd   = 0;

        //     auto    *lMagic       = BinaryAsset::GetMagic();
        //     auto     lMagicLength = BinaryAsset::GetMagicLength();
        //     auto     lOutFile     = std::ofstream( lPath.string(), std::ofstream::binary );
        //     uint32_t lAssetCount  = static_cast<uint32_t>( 1 );
        //     lOutFile.write( (const char *)lMagic, lMagicLength );
        //     lOutFile.write( (const char *)&lAssetCount, sizeof( uint32_t ) );
        //     lOutFile.write( (const char *)&lMeshAssetIndexEntry, sizeof( sAssetIndex ) );
        //     lOutFile.write( (const char *)lMeshData.data(), lMeshData.size() );
        // }

        // auto lOut = ConfigurationWriter( mModelsPath / lModelName / "Model.yaml" );
        // lOut.BeginMap();
        // lOut.WriteKey( "model" );
        // {
        //     lOut.BeginMap();
        //     {
        //         lOut.WriteKey( "meshes" );
        //         {
        //             lOut.WriteNull();
        //         }
        //         lOut.WriteKey( "materials" );
        //         {
        //             lOut.WriteNull();
        //         }
        //         lOut.WriteKey( "nodes" );
        //         {
        //             lOut.BeginMap();
        //             {
        //                 uint32_t lNodeID = 0;
        //                 for( auto &lNode : lModelData->mNodes )
        //                 {
        //                     auto lNodeKey = fmt::format( "NODE_{}", lNodeID );

        //                     lOut.WriteKey( lNodeKey );
        //                     lOut.BeginMap();
        //                     {
        //                         lOut.WriteKey( "parent" );
        //                         if( lNode.mParentID == std::numeric_limits<uint32_t>::max() )
        //                             lOut.WriteNull();
        //                         else
        //                             lOut.Write( fmt::format( "NODE_{}", lNode.mParentID ) );

        //                         lOut.WriteKey( "transform" );
        //                         lOut.WriteNull();
        //                         lOut.WriteKey( "skin" );
        //                         lOut.WriteNull();
        //                     }
        //                     lOut.EndMap();

        //                     lNodeID++;
        //                 }

        //                 // Meshes
        //                 uint32_t lMeshNodeID = 0;
        //                 for( uint32_t lNodeID = 0; lNodeID < lModelData->mNodes.size(); lNodeID++ )
        //                 {
        //                     auto &lNode = lModelData->mNodes[lNodeID];
        //                     if( lNode.mMeshes.size() == 0 ) continue;

        //                     for( uint32_t lNodeMeshID = 0; lNodeMeshID < lNode.mMeshes.size(); lNodeMeshID++ )
        //                     {
        //                         auto &lMesh    = lModelData->mMeshes[lNode.mMeshes[lNodeMeshID]];
        //                         auto  lNodeKey = fmt::format( "MESH_NODE_{}", lMeshNodeID );
        //                         lOut.WriteKey( lNodeKey );
        //                         lOut.BeginMap();
        //                         {
        //                             lOut.WriteKey( "parent", fmt::format( "NODE_{}", lNodeID ) );
        //                             lOut.WriteKey( "mesh", fmt::format( "MESH_{}", lNode.mMeshes[lNodeMeshID] ) );
        //                             lOut.WriteKey( "material", fmt::format( "MATERIAL_{}", lMesh.mMaterialID ) );
        //                         }
        //                         lOut.EndMap();

        //                         lMeshNodeID++;
        //                     }
        //                 }
        //             }
        //             lOut.EndMap();
        //         }
        //     }
        //     lOut.EndMap();
        // }
        // lOut.EndMap();
    }

    void EditorWindow::WorkspaceNew( int32_t width, int32_t height )
    {
        // auto &lIO = ImGui::GetIO();

        // math::vec2 l_WorkspacePosition = UI::GetCurrentCursorScreenPosition();
        // math::vec2 l_CursorPosition    = UI::GetCurrentCursorPosition();

        // ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.0f, 0.0f, 0.0f, 0.0f } );
        // ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 1.0f, 1.0f, 1.0f, 0.01f } );
        // ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 1.0f, 1.0f, 1.0f, 0.02f } );

        // float                   l_SliderSize0            = 150.0f;
        // static ManipulationType lCurrentManipulationType = ManipulationType::ROTATION;
        // ManipulationTypeChooser lManipulationTypeChooser( "##FOO_2" );
        // ImGui::SetNextItemWidth( l_SliderSize0 );
        // lManipulationTypeChooser.Dropdown.Labels = { "Rotation", "Translation", "Scale" };
        // lManipulationTypeChooser.Dropdown.Values = { ManipulationType::ROTATION, ManipulationType::TRANSLATION,
        //                                              ManipulationType::SCALE };
        // lManipulationTypeChooser.Display( lCurrentManipulationType );
        // if( lManipulationTypeChooser.Dropdown.Changed ) lCurrentManipulationType = lManipulationTypeChooser.Dropdown.GetValue();

        // math::vec2  l3DViewPosition = UI::GetCurrentCursorScreenPosition();
        // math::ivec2 l3DViewSize     = UI::GetAvailableContentSpace();
        // m_WorkspaceAreaSize         = l3DViewSize;

        // UI::SameLine();

        // if( ActiveWorld->GetState() == Scene::eSceneState::EDITING )
        // {

        //     if( ImGui::ImageButton( (ImTextureID)m_PlayIconHandle.Handle->GetID(), ImVec2{ 22.0f, 22.0f }, ImVec2{ 0.0f, 0.0f },
        //                             ImVec2{ 1.0f, 1.0f }, 0, ImVec4{ 0.0f, 0.0f, 0.0f, 0.0f }, ImVec4{ 0.0f, 1.0f, 0.0f, 0.8f } ) )
        //     {
        //         if( OnBeginScenario ) OnBeginScenario();

        //         ActiveWorld = New<Scene>( World );
        //         ActiveWorld->BeginScenario();
        //     }
        // }
        // else
        // {
        //     if( ImGui::ImageButton( (ImTextureID)m_PauseIconHandle.Handle->GetID(), ImVec2{ 22.0f, 22.0f }, ImVec2{ 0.0f, 0.0f },
        //                             ImVec2{ 1.0f, 1.0f }, 0, ImVec4{ 0.0f, 0.0f, 0.0f, 0.0f }, ImVec4{ 1.0f, .2f, 0.0f, 0.8f } ) )
        //     {
        //         if( OnEndScenario ) OnEndScenario();

        //         ActiveWorld->EndScenario();
        //         ActiveWorld  = World;
        //         ActiveSensor = Sensor;
        //     }
        // }
        // UI::SameLine();

        // ImGui::PushStyleColor( ImGuiCol_Text,
        //                        WorldRenderer->mRenderGizmos ? ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f } : ImVec4{ 1.0f, 1.0f, 1.0f, .2f } );
        // if( UI::Button( "Gizmos", math::vec2{ 65.0f, 24.0f } ) )
        // {
        //     // WorldRenderer->mRenderGizmos = !WorldRenderer->mRenderGizmos;
        // }
        // ImGui::PopStyleColor();
        // UI::SameLine();
        // ImGui::PushStyleColor( ImGuiCol_Text, WorldRenderer->mRenderCoordinateGrid ? ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f }
        //                                                                            : ImVec4{ 1.0f, 1.0f, 1.0f, .2f } );
        // if( UI::Button( "Grid", math::vec2{ 65.0f, 24.0f } ) )
        // {
        //     // WorldRenderer->mRenderCoordinateGrid = !WorldRenderer->mRenderCoordinateGrid;
        // }
        // ImGui::PopStyleColor();
        // UI::SameLine();

        // ImGui::PushStyleColor( ImGuiCol_Text, WorldRenderer->mGrayscaleRendering ? ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f }
        //                                                                          : ImVec4{ 1.0f, 1.0f, 1.0f, .2f } );
        // if( UI::Button( "Grayscale", math::vec2{ 65.0f, 24.0f } ) )
        // {
        //     // WorldRenderer->mGrayscaleRendering = !WorldRenderer->mGrayscaleRendering;
        // }
        // UI::SameLine();

        // ImGui::PushStyleColor( ImGuiCol_Text,
        //                        WorldRenderer->mUseFXAA ? ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f } : ImVec4{ 1.0f, 1.0f, 1.0f, .2f } );
        // if( UI::Button( "FXAA", math::vec2{ 65.0f, 24.0f } ) )
        // {
        //     // WorldRenderer->mUseFXAA = !WorldRenderer->mUseFXAA;
        // }
        // ImGui::PopStyleColor();
        // ImGui::PopStyleColor();
        // ImGui::PopStyleColor();
        // ImGui::PopStyleColor();
        // ImGui::PopStyleColor();

        // ActiveWorld->SetViewport( l3DViewPosition,
        //                           math::vec2{ static_cast<float>( l3DViewSize.x ), static_cast<float>( l3DViewSize.y ) } );

        // if( m_SceneViewport.Handle )
        // {
        //     UI::Image( m_SceneViewport, l3DViewSize );
        //     if( ImGui::BeginDragDropTarget() )
        //     {
        //         if( const ImGuiPayload *payload = ImGui::AcceptDragDropPayload( "CONTENT_BROWSER_ITEM" ) )
        //         {
        //             const char *lPayloadData = (const char *)payload->Data;
        //             size_t      lPayloadSize = static_cast<size_t>( payload->DataSize / 2 );
        //             std::string lItemPath( lPayloadSize - 1, '\0' );
        //             for( uint32_t i = 0; i < lPayloadSize - 1; i++ )
        //             {
        //                 lItemPath[i] = lPayloadData[2 * i];
        //             }

        //             LoadScenario( fs::path( lItemPath ) );
        //         }
        //         ImGui::EndDragDropTarget();
        //     }

        //     math::mat4 lCameraView     = math::Inverse( World->mEditorView );
        //     math::vec3 lCameraPosition = math::Translation( lCameraView );
        //     math::mat3 lRotationMatrix = math::Rotation( lCameraView );

        //     float lRotationSpeed = 2.5_degf;
        //     float lPanningSpeed  = 0.25f;

        //     bool lViewLanipulate =
        //         ViewManipulate( lCameraPosition, World->mEditorView, l3DViewPosition + math::vec2{ l3DViewSize.x - 125.0f, 35.0f }
        //         );

        //     WorldRenderer->SetView( World->mEditorView );

        //     ManipulationConfig lManipulator{};
        //     lManipulator.Type             = lCurrentManipulationType;
        //     lManipulator.Projection       = WorldRenderer->mView.Projection;
        //     lManipulator.WorldTransform   = World->mEditorView;
        //     lManipulator.ViewportPosition = l3DViewPosition;
        //     lManipulator.ViewportSize     = l3DViewSize;

        //     if( m_SceneHierarchyPanel.SelectedElement && m_SceneHierarchyPanel.SelectedElement.Has<sNodeTransformComponent>() )
        //     {
        //         auto &lSensorTransform = m_SceneHierarchyPanel.SelectedElement.Get<sNodeTransformComponent>();
        //         Manipulate( lManipulator, lSensorTransform.mMatrix );
        //         m_SceneHierarchyPanel.SelectedElement.Replace<sNodeTransformComponent>( lSensorTransform );
        //     }

        //     if( !ImGui::GetDragDropPayload() && ImGui::IsItemHovered() && !ImGuizmo::IsUsing() && !lViewLanipulate )
        //     {
        //         if( ImGui::IsMouseDragging( ImGuiMouseButton_Left ) )
        //         {
        //             ImVec2 lDragDelta =
        //                 ImGui::GetMouseDragDelta() / ImVec2{ static_cast<float>( width ), static_cast<float>( height ) };
        //             if( lIO.KeyShift )
        //             {
        //                 math::mat3 lNewRotation =
        //                     math::mat3( math::Rotation( -lDragDelta.x, lRotationMatrix * math::vec3{ 0.0f, 1.0f, 0.0f } ) ) *
        //                     lRotationMatrix;
        //                 World->mEditorView = math::Inverse( math::FromComponents( lNewRotation, lCameraPosition ) );

        //                 WorldRenderer->SetView( World->mEditorView );
        //             }
        //             else if( lIO.KeyAlt )
        //             {
        //                 math::mat3 lNewRotation =
        //                     math::mat3( math::Rotation( -lDragDelta.y, lRotationMatrix * math::vec3{ 1.0f, 0.0f, 0.0f } ) ) *
        //                     lRotationMatrix;
        //                 World->mEditorView = math::Inverse( math::FromComponents( lNewRotation, lCameraPosition ) );

        //                 WorldRenderer->SetView( World->mEditorView );
        //             }
        //             else if( lIO.KeyCtrl )
        //             {
        //                 math::mat3 lNewRotation =
        //                     math::mat3( math::Rotation( -lDragDelta.x, lRotationMatrix * math::vec3{ 0.0f, 0.0f, 1.0f } ) ) *
        //                     lRotationMatrix;
        //                 World->mEditorView = math::Inverse( math::FromComponents( lNewRotation, lCameraPosition ) );

        //                 WorldRenderer->SetView( World->mEditorView );
        //             }
        //             else
        //             {
        //                 math::mat3 lNewRotationX =
        //                     math::mat3( math::Rotation( -lDragDelta.x, lRotationMatrix * math::vec3{ 0.0f, 1.0f, 0.0f } ) );
        //                 math::mat3 lNewRotationY =
        //                     math::mat3( math::Rotation( -lDragDelta.y, lRotationMatrix * math::vec3{ 1.0f, 0.0f, 0.0f } ) );
        //                 World->mEditorView =
        //                     math::Inverse( math::FromComponents( lNewRotationX * lNewRotationY * lRotationMatrix, lCameraPosition )
        //                     );

        //                 WorldRenderer->SetView( World->mEditorView );
        //             }
        //             ImGui::ResetMouseDragDelta( ImGuiMouseButton_Left );
        //         }

        //         if( ImGui::IsMouseDragging( ImGuiMouseButton_Right ) )
        //         {

        //             ImVec2     lDragDelta0 = ImGui::GetMouseDragDelta( ImGuiMouseButton_Right );
        //             math::vec4 lDragDelta{ lDragDelta0.x, lDragDelta0.y, 0.0f, 1.0f };
        //             lDragDelta = math::Inverse( WorldRenderer->mView.Projection ) * lDragDelta;
        //             lDragDelta = lDragDelta / lDragDelta.w;

        //             math::vec3 lPanAmount   = lRotationMatrix * math::vec3{ -lDragDelta.x, -lDragDelta.y, 0.0f };
        //             math::vec3 lNewPosition = lCameraPosition + lPanAmount;
        //             World->mEditorView      = math::Inverse( math::FromComponents( lRotationMatrix, lNewPosition ) );

        //             WorldRenderer->SetView( World->mEditorView );

        //             ImGui::ResetMouseDragDelta( ImGuiMouseButton_Right );
        //         }
        //     }

        //     if( ImGui::IsItemHovered() )
        //     {
        //         if( ImGui::IsKeyPressed( ImGuiKey_UpArrow ) )
        //         {
        //             if( lIO.KeyShift )
        //             {
        //                 math::vec3 lPanAmount   = lRotationMatrix * math::vec3{ 0.0f, lPanningSpeed, 0.0f };
        //                 math::vec3 lNewPosition = lCameraPosition + lPanAmount;
        //                 World->mEditorView      = math::Inverse( math::FromComponents( lRotationMatrix, lNewPosition ) );

        //                 WorldRenderer->SetView( World->mEditorView );
        //             }
        //             else
        //             {
        //                 math::vec3 lPanAmount   = lRotationMatrix * math::vec3{ 0.0f, 0.0f, -lPanningSpeed };
        //                 math::vec3 lNewPosition = lCameraPosition + lPanAmount;
        //                 World->mEditorView      = math::Inverse( math::FromComponents( lRotationMatrix, lNewPosition ) );

        //                 WorldRenderer->SetView( World->mEditorView );
        //             }
        //         }
        //         else if( ImGui::IsKeyPressed( ImGuiKey_DownArrow ) )
        //         {
        //             if( lIO.KeyShift )
        //             {
        //                 math::vec3 lPanAmount   = lRotationMatrix * math::vec3{ 0.0f, -lPanningSpeed, 0.0f };
        //                 math::vec3 lNewPosition = lCameraPosition + lPanAmount;
        //                 World->mEditorView      = math::Inverse( math::FromComponents( lRotationMatrix, lNewPosition ) );

        //                 WorldRenderer->SetView( World->mEditorView );
        //             }
        //             else
        //             {
        //                 math::vec3 lPanAmount   = lRotationMatrix * math::vec3{ 0.0f, 0.0f, lPanningSpeed };
        //                 math::vec3 lNewPosition = lCameraPosition + lPanAmount;
        //                 World->mEditorView      = math::Inverse( math::FromComponents( lRotationMatrix, lNewPosition ) );

        //                 WorldRenderer->SetView( World->mEditorView );
        //             }
        //         }
        //         else if( ImGui::IsKeyPressed( ImGuiKey_LeftArrow ) )
        //         {
        //             math::vec3 lNewPosition = lCameraPosition + lRotationMatrix * math::vec3{ -lPanningSpeed, 0.0f, 0.0f };
        //             World->mEditorView      = math::Inverse( math::FromComponents( lRotationMatrix, lNewPosition ) );

        //             WorldRenderer->SetView( World->mEditorView );
        //         }
        //         else if( ImGui::IsKeyPressed( ImGuiKey_RightArrow ) )
        //         {
        //             math::vec3 lNewPosition = lCameraPosition + lRotationMatrix * math::vec3{ lPanningSpeed, 0.0f, 0.0f };
        //             World->mEditorView      = math::Inverse( math::FromComponents( lRotationMatrix, lNewPosition ) );

        //             WorldRenderer->SetView( World->mEditorView );
        //         }

        //         if( ImGui::IsKeyPressed( ImGuiKey_W ) )
        //         {
        //             math::mat3 lNewRotation =
        //                 math::mat3( math::Rotation( -lRotationSpeed, lRotationMatrix * math::vec3{ 1.0f, 0.0f, 0.0f } ) ) *
        //                 lRotationMatrix;
        //             World->mEditorView = math::Inverse( math::FromComponents( lNewRotation, lCameraPosition ) );

        //             WorldRenderer->SetView( World->mEditorView );
        //         }
        //         else if( ImGui::IsKeyPressed( ImGuiKey_S ) )
        //         {
        //             math::mat3 lNewRotation =
        //                 math::mat3( math::Rotation( lRotationSpeed, lRotationMatrix * math::vec3{ 1.0f, 0.0f, 0.0f } ) ) *
        //                 lRotationMatrix;
        //             World->mEditorView = math::Inverse( math::FromComponents( lNewRotation, lCameraPosition ) );

        //             WorldRenderer->SetView( World->mEditorView );
        //         }
        //         else if( ImGui::IsKeyPressed( ImGuiKey_A ) )
        //         {
        //             math::mat3 lNewRotation =
        //                 math::mat3( math::Rotation( lRotationSpeed, lRotationMatrix * math::vec3{ 0.0f, 1.0f, 0.0f } ) ) *
        //                 lRotationMatrix;
        //             World->mEditorView = math::Inverse( math::FromComponents( lNewRotation, lCameraPosition ) );

        //             WorldRenderer->SetView( World->mEditorView );
        //         }
        //         else if( ImGui::IsKeyPressed( ImGuiKey_D ) )
        //         {
        //             math::mat3 lNewRotation =
        //                 math::mat3( math::Rotation( -lRotationSpeed, lRotationMatrix * math::vec3{ 0.0f, 1.0f, 0.0f } ) ) *
        //                 lRotationMatrix;
        //             World->mEditorView = math::Inverse( math::FromComponents( lNewRotation, lCameraPosition ) );

        //             WorldRenderer->SetView( World->mEditorView );
        //         }
        //     }
        // }
    }

    void EditorWindow::Workspace( int32_t width, int32_t height )
    {
        auto &lIO = ImGui::GetIO();

        math::vec2 l_WorkspacePosition = UI::GetCurrentCursorScreenPosition();
        math::vec2 l_CursorPosition    = UI::GetCurrentCursorPosition();

        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.0f, 0.0f, 0.0f, 0.0f } );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 1.0f, 1.0f, 1.0f, 0.01f } );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 1.0f, 1.0f, 1.0f, 0.02f } );

        float                   l_SliderSize0            = 150.0f;
        static ManipulationType lCurrentManipulationType = ManipulationType::ROTATION;
        ManipulationTypeChooser lManipulationTypeChooser( "##FOO_2" );
        ImGui::SetNextItemWidth( l_SliderSize0 );
        lManipulationTypeChooser.Dropdown.Labels = { "Rotation", "Translation", "Scale" };
        lManipulationTypeChooser.Dropdown.Values = { ManipulationType::ROTATION, ManipulationType::TRANSLATION,
                                                     ManipulationType::SCALE };
        lManipulationTypeChooser.Display( lCurrentManipulationType );
        if( lManipulationTypeChooser.Dropdown.Changed ) lCurrentManipulationType = lManipulationTypeChooser.Dropdown.GetValue();

        math::vec2  l3DViewPosition = UI::GetCurrentCursorScreenPosition();
        math::ivec2 l3DViewSize     = UI::GetAvailableContentSpace();
        m_WorkspaceAreaSize         = l3DViewSize;

        UI::SameLine();

        if( ActiveWorld->GetState() == Scene::eSceneState::EDITING )
        {

            if( ImGui::ImageButton( (ImTextureID)m_PlayIconHandle.Handle->GetID(), ImVec2{ 22.0f, 22.0f }, ImVec2{ 0.0f, 0.0f },
                                    ImVec2{ 1.0f, 1.0f }, 0, ImVec4{ 0.0f, 0.0f, 0.0f, 0.0f }, ImVec4{ 0.0f, 1.0f, 0.0f, 0.8f } ) )
            {
                if( OnBeginScenario ) OnBeginScenario();

                ActiveWorld = New<Scene>( World );
                ActiveWorld->BeginScenario();
            }
        }
        else
        {
            if( ImGui::ImageButton( (ImTextureID)m_PauseIconHandle.Handle->GetID(), ImVec2{ 22.0f, 22.0f }, ImVec2{ 0.0f, 0.0f },
                                    ImVec2{ 1.0f, 1.0f }, 0, ImVec4{ 0.0f, 0.0f, 0.0f, 0.0f }, ImVec4{ 1.0f, .2f, 0.0f, 0.8f } ) )
            {
                if( OnEndScenario ) OnEndScenario();

                ActiveWorld->EndScenario();
                ActiveWorld  = World;
                ActiveSensor = Sensor;
            }
        }
        UI::SameLine();

        ImGui::PushStyleColor( ImGuiCol_Text,
                               WorldRenderer->mRenderGizmos ? ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f } : ImVec4{ 1.0f, 1.0f, 1.0f, .2f } );
        if( UI::Button( "Gizmos", math::vec2{ 65.0f, 24.0f } ) )
        {
            WorldRenderer->mRenderGizmos = !WorldRenderer->mRenderGizmos;
        }
        ImGui::PopStyleColor();
        UI::SameLine();
        ImGui::PushStyleColor( ImGuiCol_Text, WorldRenderer->mRenderCoordinateGrid ? ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f }
                                                                                   : ImVec4{ 1.0f, 1.0f, 1.0f, .2f } );
        if( UI::Button( "Grid", math::vec2{ 65.0f, 24.0f } ) )
        {
            WorldRenderer->mRenderCoordinateGrid = !WorldRenderer->mRenderCoordinateGrid;
        }
        ImGui::PopStyleColor();
        UI::SameLine();

        ImGui::PushStyleColor( ImGuiCol_Text, WorldRenderer->mGrayscaleRendering ? ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f }
                                                                                 : ImVec4{ 1.0f, 1.0f, 1.0f, .2f } );
        if( UI::Button( "Grayscale", math::vec2{ 65.0f, 24.0f } ) )
        {
            WorldRenderer->mGrayscaleRendering = !WorldRenderer->mGrayscaleRendering;
        }
        UI::SameLine();

        ImGui::PushStyleColor( ImGuiCol_Text,
                               WorldRenderer->mUseFXAA ? ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f } : ImVec4{ 1.0f, 1.0f, 1.0f, .2f } );
        if( UI::Button( "FXAA", math::vec2{ 65.0f, 24.0f } ) )
        {
            WorldRenderer->mUseFXAA = !WorldRenderer->mUseFXAA;
        }
        ImGui::PopStyleColor();
        ImGui::PopStyleColor();
        ImGui::PopStyleColor();
        ImGui::PopStyleColor();
        ImGui::PopStyleColor();

        ActiveWorld->SetViewport( l3DViewPosition,
                                  math::vec2{ static_cast<float>( l3DViewSize.x ), static_cast<float>( l3DViewSize.y ) } );

        if( m_SceneViewport.Handle )
        {
            UI::Image( m_SceneViewport, l3DViewSize );
            if( ImGui::BeginDragDropTarget() )
            {
                if( const ImGuiPayload *payload = ImGui::AcceptDragDropPayload( "CONTENT_BROWSER_ITEM" ) )
                {
                    const char *lPayloadData = (const char *)payload->Data;
                    size_t      lPayloadSize = static_cast<size_t>( payload->DataSize / 2 );
                    std::string lItemPath( lPayloadSize - 1, '\0' );
                    for( uint32_t i = 0; i < lPayloadSize - 1; i++ )
                    {
                        lItemPath[i] = lPayloadData[2 * i];
                    }

                    LoadScenario( fs::path( lItemPath ) );
                }
                ImGui::EndDragDropTarget();
            }

            math::mat4 lCameraView     = math::Inverse( World->mEditorView );
            math::vec3 lCameraPosition = math::Translation( lCameraView );
            math::mat3 lRotationMatrix = math::Rotation( lCameraView );

            float lRotationSpeed = 2.5_degf;
            float lPanningSpeed  = 0.25f;

            bool lViewLanipulate =
                ViewManipulate( lCameraPosition, World->mEditorView, l3DViewPosition + math::vec2{ l3DViewSize.x - 125.0f, 35.0f } );

            WorldRenderer->SetView( World->mEditorView );

            ManipulationConfig lManipulator{};
            lManipulator.Type             = lCurrentManipulationType;
            lManipulator.Projection       = WorldRenderer->mView.Projection;
            lManipulator.WorldTransform   = World->mEditorView;
            lManipulator.ViewportPosition = l3DViewPosition;
            lManipulator.ViewportSize     = l3DViewSize;

            if( m_SceneHierarchyPanel.SelectedElement && m_SceneHierarchyPanel.SelectedElement.Has<sNodeTransformComponent>() )
            {
                auto &lSensorTransform = m_SceneHierarchyPanel.SelectedElement.Get<sNodeTransformComponent>();
                Manipulate( lManipulator, lSensorTransform.mMatrix );
                m_SceneHierarchyPanel.SelectedElement.Replace<sNodeTransformComponent>( lSensorTransform );
            }

            if( !ImGui::GetDragDropPayload() && ImGui::IsItemHovered() && !ImGuizmo::IsUsing() && !lViewLanipulate )
            {
                if( ImGui::IsMouseDragging( ImGuiMouseButton_Left ) )
                {
                    ImVec2 lDragDelta =
                        ImGui::GetMouseDragDelta() / ImVec2{ static_cast<float>( width ), static_cast<float>( height ) };
                    if( lIO.KeyShift )
                    {
                        math::mat3 lNewRotation =
                            math::mat3( math::Rotation( -lDragDelta.x, lRotationMatrix * math::vec3{ 0.0f, 1.0f, 0.0f } ) ) *
                            lRotationMatrix;
                        World->mEditorView = math::Inverse( math::FromComponents( lNewRotation, lCameraPosition ) );

                        WorldRenderer->SetView( World->mEditorView );
                    }
                    else if( lIO.KeyAlt )
                    {
                        math::mat3 lNewRotation =
                            math::mat3( math::Rotation( -lDragDelta.y, lRotationMatrix * math::vec3{ 1.0f, 0.0f, 0.0f } ) ) *
                            lRotationMatrix;
                        World->mEditorView = math::Inverse( math::FromComponents( lNewRotation, lCameraPosition ) );

                        WorldRenderer->SetView( World->mEditorView );
                    }
                    else if( lIO.KeyCtrl )
                    {
                        math::mat3 lNewRotation =
                            math::mat3( math::Rotation( -lDragDelta.x, lRotationMatrix * math::vec3{ 0.0f, 0.0f, 1.0f } ) ) *
                            lRotationMatrix;
                        World->mEditorView = math::Inverse( math::FromComponents( lNewRotation, lCameraPosition ) );

                        WorldRenderer->SetView( World->mEditorView );
                    }
                    else
                    {
                        math::mat3 lNewRotationX =
                            math::mat3( math::Rotation( -lDragDelta.x, lRotationMatrix * math::vec3{ 0.0f, 1.0f, 0.0f } ) );
                        math::mat3 lNewRotationY =
                            math::mat3( math::Rotation( -lDragDelta.y, lRotationMatrix * math::vec3{ 1.0f, 0.0f, 0.0f } ) );
                        World->mEditorView =
                            math::Inverse( math::FromComponents( lNewRotationX * lNewRotationY * lRotationMatrix, lCameraPosition ) );

                        WorldRenderer->SetView( World->mEditorView );
                    }
                    ImGui::ResetMouseDragDelta( ImGuiMouseButton_Left );
                }

                if( ImGui::IsMouseDragging( ImGuiMouseButton_Right ) )
                {

                    ImVec2     lDragDelta0 = ImGui::GetMouseDragDelta( ImGuiMouseButton_Right );
                    math::vec4 lDragDelta{ lDragDelta0.x, lDragDelta0.y, 0.0f, 1.0f };
                    lDragDelta = math::Inverse( WorldRenderer->mView.Projection ) * lDragDelta;
                    lDragDelta = lDragDelta / lDragDelta.w;

                    math::vec3 lPanAmount   = lRotationMatrix * math::vec3{ -lDragDelta.x, -lDragDelta.y, 0.0f };
                    math::vec3 lNewPosition = lCameraPosition + lPanAmount;
                    World->mEditorView      = math::Inverse( math::FromComponents( lRotationMatrix, lNewPosition ) );

                    WorldRenderer->SetView( World->mEditorView );

                    ImGui::ResetMouseDragDelta( ImGuiMouseButton_Right );
                }
            }

            if( ImGui::IsItemHovered() )
            {
                if( ImGui::IsKeyPressed( ImGuiKey_UpArrow ) )
                {
                    if( lIO.KeyShift )
                    {
                        math::vec3 lPanAmount   = lRotationMatrix * math::vec3{ 0.0f, lPanningSpeed, 0.0f };
                        math::vec3 lNewPosition = lCameraPosition + lPanAmount;
                        World->mEditorView      = math::Inverse( math::FromComponents( lRotationMatrix, lNewPosition ) );

                        WorldRenderer->SetView( World->mEditorView );
                    }
                    else
                    {
                        math::vec3 lPanAmount   = lRotationMatrix * math::vec3{ 0.0f, 0.0f, -lPanningSpeed };
                        math::vec3 lNewPosition = lCameraPosition + lPanAmount;
                        World->mEditorView      = math::Inverse( math::FromComponents( lRotationMatrix, lNewPosition ) );

                        WorldRenderer->SetView( World->mEditorView );
                    }
                }
                else if( ImGui::IsKeyPressed( ImGuiKey_DownArrow ) )
                {
                    if( lIO.KeyShift )
                    {
                        math::vec3 lPanAmount   = lRotationMatrix * math::vec3{ 0.0f, -lPanningSpeed, 0.0f };
                        math::vec3 lNewPosition = lCameraPosition + lPanAmount;
                        World->mEditorView      = math::Inverse( math::FromComponents( lRotationMatrix, lNewPosition ) );

                        WorldRenderer->SetView( World->mEditorView );
                    }
                    else
                    {
                        math::vec3 lPanAmount   = lRotationMatrix * math::vec3{ 0.0f, 0.0f, lPanningSpeed };
                        math::vec3 lNewPosition = lCameraPosition + lPanAmount;
                        World->mEditorView      = math::Inverse( math::FromComponents( lRotationMatrix, lNewPosition ) );

                        WorldRenderer->SetView( World->mEditorView );
                    }
                }
                else if( ImGui::IsKeyPressed( ImGuiKey_LeftArrow ) )
                {
                    math::vec3 lNewPosition = lCameraPosition + lRotationMatrix * math::vec3{ -lPanningSpeed, 0.0f, 0.0f };
                    World->mEditorView      = math::Inverse( math::FromComponents( lRotationMatrix, lNewPosition ) );

                    WorldRenderer->SetView( World->mEditorView );
                }
                else if( ImGui::IsKeyPressed( ImGuiKey_RightArrow ) )
                {
                    math::vec3 lNewPosition = lCameraPosition + lRotationMatrix * math::vec3{ lPanningSpeed, 0.0f, 0.0f };
                    World->mEditorView      = math::Inverse( math::FromComponents( lRotationMatrix, lNewPosition ) );

                    WorldRenderer->SetView( World->mEditorView );
                }

                if( ImGui::IsKeyPressed( ImGuiKey_W ) )
                {
                    math::mat3 lNewRotation =
                        math::mat3( math::Rotation( -lRotationSpeed, lRotationMatrix * math::vec3{ 1.0f, 0.0f, 0.0f } ) ) *
                        lRotationMatrix;
                    World->mEditorView = math::Inverse( math::FromComponents( lNewRotation, lCameraPosition ) );

                    WorldRenderer->SetView( World->mEditorView );
                }
                else if( ImGui::IsKeyPressed( ImGuiKey_S ) )
                {
                    math::mat3 lNewRotation =
                        math::mat3( math::Rotation( lRotationSpeed, lRotationMatrix * math::vec3{ 1.0f, 0.0f, 0.0f } ) ) *
                        lRotationMatrix;
                    World->mEditorView = math::Inverse( math::FromComponents( lNewRotation, lCameraPosition ) );

                    WorldRenderer->SetView( World->mEditorView );
                }
                else if( ImGui::IsKeyPressed( ImGuiKey_A ) )
                {
                    math::mat3 lNewRotation =
                        math::mat3( math::Rotation( lRotationSpeed, lRotationMatrix * math::vec3{ 0.0f, 1.0f, 0.0f } ) ) *
                        lRotationMatrix;
                    World->mEditorView = math::Inverse( math::FromComponents( lNewRotation, lCameraPosition ) );

                    WorldRenderer->SetView( World->mEditorView );
                }
                else if( ImGui::IsKeyPressed( ImGuiKey_D ) )
                {
                    math::mat3 lNewRotation =
                        math::mat3( math::Rotation( -lRotationSpeed, lRotationMatrix * math::vec3{ 0.0f, 1.0f, 0.0f } ) ) *
                        lRotationMatrix;
                    World->mEditorView = math::Inverse( math::FromComponents( lNewRotation, lCameraPosition ) );

                    WorldRenderer->SetView( World->mEditorView );
                }
            }
        }
    }

    void EditorWindow::Console( int32_t width, int32_t height )
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

    void EditorWindow::UpdateFramerate( Timestep ts )
    {
        m_FrameCounter++;
        m_FpsTimer += (float)ts;
        if( m_FpsTimer > 1000.0f )
        {
            m_LastFPS      = static_cast<uint32_t>( (float)m_FrameCounter * ( 1000.0f / m_FpsTimer ) );
            m_FpsTimer     = 0.0f;
            m_FrameCounter = 0;
        }
    }

} // namespace SE::Editor