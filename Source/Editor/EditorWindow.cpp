#include "EditorWindow.h"

#include <fmt/core.h>
#include <fstream>
#include <locale>

#include "Core/Profiling/BlockTimer.h"

#include "UI/UI.h"
#include "UI/Widgets.h"

#include "Scene/Components.h"
#include "Scene/Importer/glTFImporter.h"

#include "Core/File.h"
#include "Core/Logging.h"
#include "Core/TextureData.h"

#include "Scene/Components.h"

// #include "ShortWaveformDisplay.h"

namespace LTSE::Editor
{

    using namespace LTSE::Core;
    using namespace LTSE::Core::EntityComponentSystem::Components;

    // class SamplerChooser
    // {
    //   public:
    //     // Ref<SensorDeviceBase> SensorModel = nullptr;
    //     std::string ID                    = "";
    //     UI::ComboBox<Entity> Dropdown;

    //   public:
    //     SamplerChooser() = default;
    //     SamplerChooser( std::string a_ID )
    //         : ID{ a_ID }
    //         , Dropdown{ UI::ComboBox<Entity>( a_ID ) } {};

    //     ~SamplerChooser() = default;

    //     Entity GetValue()
    //     {
    //         if( Dropdown.Values.size() > 0 )
    //             return Dropdown.Values[Dropdown.CurrentItem];
    //         return Entity{};
    //     }

    //     void Display( Entity &a_TargetEntity )
    //     {
    //         Dropdown.Labels = { "None" };
    //         Dropdown.Values = { Entity{} };

    //         uint32_t n = 1;

    //         SensorModel->mSensorDefinition->ForEach<sSampler>(
    //             [&]( auto aEntity, auto &aComponent )
    //             {
    //                 Dropdown.Labels.push_back( aEntity.Get<sTag>().mValue );
    //                 Dropdown.Values.push_back( aEntity );

    //                 if( (uint32_t)aEntity == (uint32_t)a_TargetEntity )
    //                     Dropdown.CurrentItem = n;
    //                 n++;
    //             } );

    //         Dropdown.Display();

    //         if( Dropdown.Changed )
    //         {
    //             a_TargetEntity = Dropdown.GetValue();
    //         }
    //     }
    // };

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

    static Ref<LTSE::Graphics::Texture2D> LoadImagePreview( LTSE::Graphics::GraphicContext &aGraphicContext, fs::path aImagePath )
    {
        if( aImagePath.extension() == ".png" )
        {
            LTSE::Core::TextureData::sCreateInfo lTextureCreateInfo{};
            TextureData2D                        lTextureData( lTextureCreateInfo, aImagePath );
            sTextureSamplingInfo                 lSamplingInfo{};
            LTSE::Core::TextureSampler2D         lTextureSampler = LTSE::Core::TextureSampler2D( lTextureData, lSamplingInfo );

            return New<LTSE::Graphics::Texture2D>( aGraphicContext, lTextureData, lTextureSampler );
        }
        return nullptr;
    }

    static void UpdateImageHandle(
        Ref<LTSE::Core::UIContext> aUIOverlay, ImageHandle &aImageHandle, Ref<LTSE::Graphics::Texture2D> aTexture )
    {
        if( !aTexture ) return;

        if( !aImageHandle.Handle )
            aImageHandle = aUIOverlay->CreateTextureHandle( aTexture );
        else
            aImageHandle.Handle->Write( aTexture, 0 );
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
            LTSE::Core::TextureData::sCreateInfo lTextureCreateInfo{};
            TextureData2D lTextureData( lTextureCreateInfo, "C:\\GitLab\\SpockEngine\\Saved\\Resources\\Icons\\Play.png" );
            sTextureSamplingInfo         lSamplingInfo{};
            LTSE::Core::TextureSampler2D lTextureSampler = LTSE::Core::TextureSampler2D( lTextureData, lSamplingInfo );

            m_PlayIcon       = New<LTSE::Graphics::Texture2D>( mGraphicContext, lTextureData, lTextureSampler );
            m_PlayIconHandle = mUIOverlay->CreateTextureHandle( m_PlayIcon );
        }

        {
            LTSE::Core::TextureData::sCreateInfo lTextureCreateInfo{};
            TextureData2D lTextureData( lTextureCreateInfo, "C:\\GitLab\\SpockEngine\\Saved\\Resources\\Icons\\Pause.png" );
            sTextureSamplingInfo         lSamplingInfo{};
            LTSE::Core::TextureSampler2D lTextureSampler = LTSE::Core::TextureSampler2D( lTextureData, lSamplingInfo );

            m_PauseIcon       = New<LTSE::Graphics::Texture2D>( mGraphicContext, lTextureData, lTextureSampler );
            m_PauseIconHandle = mUIOverlay->CreateTextureHandle( m_PauseIcon );
        }

        {
            LTSE::Core::TextureData::sCreateInfo lTextureCreateInfo{};
            TextureData2D lTextureData( lTextureCreateInfo, "C:\\GitLab\\SpockEngine\\Saved\\Resources\\Icons\\Play.png" );
            sTextureSamplingInfo         lSamplingInfo{};
            LTSE::Core::TextureSampler2D lTextureSampler = LTSE::Core::TextureSampler2D( lTextureData, lSamplingInfo );

            m_DefaultTextureImage       = New<LTSE::Graphics::Texture2D>( mGraphicContext, lTextureData, lTextureSampler );
            m_DefaultTextureImageHandle = mUIOverlay->CreateTextureHandle( m_PlayIcon );
        }

        {
            LTSE::Core::TextureData::sCreateInfo lTextureCreateInfo{};
            TextureData2D lTextureData( lTextureCreateInfo, "C:\\GitLab\\SpockEngine\\Saved\\Resources\\Icons\\Camera.png" );
            sTextureSamplingInfo         lSamplingInfo{};
            LTSE::Core::TextureSampler2D lTextureSampler = LTSE::Core::TextureSampler2D( lTextureData, lSamplingInfo );

            m_CameraIcon       = New<LTSE::Graphics::Texture2D>( mGraphicContext, lTextureData, lTextureSampler );
            m_CameraIconHandle = mUIOverlay->CreateTextureHandle( m_CameraIcon );
        }
    }

    EditorWindow::EditorWindow( LTSE::Graphics::GraphicContext &aGraphicContext, Ref<UIContext> aUIOverlay )
        : mGraphicContext{ aGraphicContext }
        , mUIOverlay{ aUIOverlay }
    {
        ConfigureUI();

        // LayoutEditor        = TileLayoutEditor( "EDIT TILE LAYOUT...", { 1800.0f, 1000.0f } );
        // FlashEditor         = FlashAttenuationBindingPopup( "EDIT FLASHES...", { 1600.0f, 1200.0f } );
        // PhotodetectorEditor = PhotodetectorCellEditor( "EDIT PHOTODETECTOR...", { 1600.0f, 1200.0f } );

        std::ifstream mFileStream{};
        mFileStream.open( fs::path( "C:\\GitLab\\SpockEngine\\Saved\\tile.bin" ), std::ios::binary );
        if( mFileStream.good() )
        {
            mFileStream.seekg( 0, mFileStream.end );
            uint32_t mFileSize = mFileStream.tellg();
            mFileStream.seekg( 0, mFileStream.beg );
            mTestTile = std::vector<uint8_t>( mFileSize );
            mFileStream.read( (char*)mTestTile.data(), mFileSize );
            LTSE::Logging::Info( "FOO - {}", mFileSize );
        }
        // std::ifstream mFileStream{};
        // mFileStream.open( fs::path( "C:\\GitLab\\EchoDS_FLM\\Saved\\tile.bin" ), std::ios::binary );
        // if( mFileStream.good() )
        // {
        //     mFileStream.seekg( 0, mFileStream.end );
        //     uint32_t mFileSize = mFileStream.tellg();
        //     mFileStream.seekg( 0, mFileStream.beg );
        //     mTestTile = std::vector<uint8_t>( mFileSize );
        //     mFileStream.read( (char*)mTestTile.data(), mFileSize );
        //     LTSE::Logging::Info( "FOO - {}", mFileSize );
        // }
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

        bool o_RequestQuit = false;
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, ImVec2( 0, 8 ) );
        if( ImGui::BeginMainMenuBar() )
        {
            ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, ImVec2( 15, 14 ) );
            o_RequestQuit = RenderMainMenu();
            ImGui::PopStyleVar();
            ImGui::EndMainMenuBar();
        }
        ImGui::PopStyleVar();
        ImGui::End();

        // if( m_NewMaterial.Visible )
        //     ImGui::OpenPopup( "NEW MATERIAL..." );
        // m_NewMaterial.Display();

        if( ImGui::Begin( "3D VIEW", &p_open, ImGuiWindowFlags_None ) )
        {
            auto lWorkspaceAreaSize = UI::GetAvailableContentSpace();
            Workspace( lWorkspaceAreaSize.x, lWorkspaceAreaSize.y );
        }
        ImGui::End();

        if( ImGui::Begin( "TRACKS", &p_open, ImGuiWindowFlags_None ) )
        {
            auto lWorkspaceAreaSize = UI::GetAvailableContentSpace();
            // Workspace( lWorkspaceAreaSize.x, lWorkspaceAreaSize.y );
        }
        ImGui::End();

        // if( ImGui::Begin( "TILE EXPLORER", &p_open, ImGuiWindowFlags_None ) )
        // {
        //     DisplayShortWaveforms( *mUIOverlay, mTestTile );
        // }
        // ImGui::End();

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
                static Ref<LTSE::Graphics::Texture2D> lBaseColorTexture = nullptr;
                static ImageHandle                    lBaseColorTextureHandle{};
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
                static Ref<LTSE::Graphics::Texture2D> lEmissiveTexture = nullptr;
                static ImageHandle                    lEmissiveTextureHandle{};
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
                static Ref<LTSE::Graphics::Texture2D> lNormalsTexture = nullptr;
                static ImageHandle                    lNormalsTextureHandle{};
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
                static Ref<LTSE::Graphics::Texture2D> lOcclusionTexture = nullptr;
                static ImageHandle                    lOcclusionTextureHandle{};
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
                static Ref<LTSE::Graphics::Texture2D> lPhysicalTexture = nullptr;
                static ImageHandle                    lPhysicalTextureHandle{};
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
                    auto lResults = LTSE::Core::Instrumentor::Get().EndSession();
                    LTSE::Logging::Info( "{}", lResults->mEvents.size() );
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
                    LTSE::Core::Instrumentor::Get().BeginSession( "Session" );
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

        // if( ImGui::Begin( "SENSOR CONFIGURATION EDITOR", &p_open, ImGuiWindowFlags_None ) )
        // {
        //     static Entity ElementInFocus{};

        //     auto l_DrawList   = ImGui::GetWindowDrawList();
        //     auto l_WindowSize = UI::GetAvailableContentSpace();

        //     auto l_TopLeft     = ImGui::GetCursorScreenPos() + ImVec2{ -10.0f, -10.0f };
        //     auto l_BottomRight = ImGui::GetCursorScreenPos() + ImVec2{ static_cast<float>( l_WindowSize.x ), 25.0f };
        //     l_DrawList->AddRectFilled( l_TopLeft, l_BottomRight, IM_COL32( 5, 5, 5, 255 ) );
        //     // if( !( SensorModel->mSensorDefinition->mName.empty() ) )
        //     //     UI::Text( "{}", SensorModel->mSensorDefinition->mName );
        //     // else
        //     UI::Text( "SENSOR_001" );

        //     UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( 0.0f, 10.0f ) );
        //     ImGuiTreeNodeFlags l_Flags = ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_FramePadding |
        //     ImGuiTreeNodeFlags_DefaultOpen;
        //     // ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, ImVec2( 3, 2 ) );

        //     // if( ImGui::TreeNodeEx( "ASSETS", l_Flags ) )
        //     // {
        //     //     for( auto &l_Tile : SensorModel->mSensorDefinition->mRootAsset.Get<sRelationshipComponent>().mChildren
        //     )
        //     //     {
        //     //         ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 0.5f, 0.2f, 0.7f, 1.0f } );
        //     //         UI::Text( "{}", ICON_FA_CIRCLE_O );
        //     //         UI::SameLine();
        //     //         ImGui::PopStyleColor();

        //     //         UI::Text( l_Tile.Get<sTag>().mValue );
        //     //         UI::SameLine();
        //     //         UI::SetCursorPosition( math::vec2( l_WindowSize.x - 30.0f, UI::GetCurrentCursorPosition().y - 3.0f
        //     ) );
        //     //         if( EditButton( l_Tile, math::vec2{ 20.0, 22.0 } ) )
        //     //         {
        //     //             ElementInFocus = l_Tile;
        //     //         }
        //     //     }
        //     //     ImGui::TreePop();
        //     // }
        //     // ImGui::PopStyleVar();

        //     UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( 0.0f, 15.0f ) );
        //     // ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, ImVec2( 3, 2 ) );
        //     // if( ImGui::TreeNodeEx( "COMPONENTS", l_Flags ) )
        //     // {
        //     //     for( auto &l_Tile :
        //     SensorModel->mSensorDefinition->mRootComponent.Get<sRelationshipComponent>().mChildren )
        //     //     {
        //     //         ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 0.5f, 0.2f, 0.7f, 1.0f } );
        //     //         UI::Text( "{}", ICON_FA_CIRCLE_O );
        //     //         UI::SameLine();
        //     //         ImGui::PopStyleColor();

        //     //         UI::Text( l_Tile.Get<sTag>().mValue );
        //     //         UI::SameLine();
        //     //         UI::SetCursorPosition( math::vec2( l_WindowSize.x - 30.0f, UI::GetCurrentCursorPosition().y - 3.0f
        //     ) );
        //     //         if( EditButton( l_Tile, math::vec2{ 20.0, 22.0 } ) )
        //     //         {
        //     //             ElementInFocus = l_Tile;
        //     //         }
        //     //     }
        //     //     ImGui::TreePop();
        //     // }
        //     // ImGui::PopStyleVar();

        //     // UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( 0.0f, 15.0f ) );
        //     // ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, ImVec2( 3, 2 ) );
        //     // if( ImGui::TreeNodeEx( "TILE CONFIGURATIONS", l_Flags ) )
        //     // {
        //     //     for( auto &l_Tile : SensorModel->mSensorDefinition->mRootTile.Get<sRelationshipComponent>().mChildren )
        //     //     {
        //     //         ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 0.5f, 0.2f, 0.7f, 1.0f } );
        //     //         UI::Text( "{}", ICON_FA_CIRCLE_O );
        //     //         UI::SameLine();
        //     //         ImGui::PopStyleColor();
        //     //         UI::Text( "Tile ID: {} ({} flashes)", l_Tile.Get<sTileSpecificationComponent>().mID,
        //     l_Tile.Get<sRelationshipComponent>().mChildren.size() );
        //     //         UI::SameLine();
        //     //         UI::SetCursorPosition( math::vec2( l_WindowSize.x - 30.0f, UI::GetCurrentCursorPosition().y - 3.0f
        //     ) );
        //     //         if( EditButton( l_Tile, math::vec2{ 20.0, 22.0 } ) )
        //     //         {
        //     //             ElementInFocus = l_Tile;
        //     //         }
        //     //     }
        //     //     ImGui::TreePop();
        //     // }
        //     // ImGui::PopStyleVar();

        //     // UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( 0.0f, 15.0f ) );
        //     // ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, ImVec2( 3, 2 ) );
        //     // if( ImGui::TreeNodeEx( "TILE LAYOUTS", l_Flags ) )
        //     // {
        //     //     for( auto &l_TileLayout :
        //     SensorModel->mSensorDefinition->mRootLayout.Get<sRelationshipComponent>().mChildren )
        //     //     {
        //     //         ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 0.5f, 0.2f, 0.7f, 1.0f } );
        //     //         UI::Text( "{}", ICON_FA_CIRCLE_O );
        //     //         UI::SameLine();
        //     //         ImGui::PopStyleColor();
        //     //         UI::Text( "{}", l_TileLayout.Get<sTag>().mValue );
        //     //         UI::SameLine();
        //     //         UI::SetCursorPosition( math::vec2( l_WindowSize.x - 30.0f, UI::GetCurrentCursorPosition().y - 3.0f
        //     ) );
        //     //         if( EditButton( l_TileLayout, math::vec2{ 20.0, 22.0 } ) )
        //     //         {
        //     //             ElementInFocus = l_TileLayout;
        //     //         }
        //     //     }
        //     //     ImGui::TreePop();
        //     // }
        //     // ImGui::PopStyleVar();

        //     if( ElementInFocus )
        //     {
        //         if( ElementInFocus.Has<sPhotoDetector>() )
        //         {
        //             auto l_WindowSize = UI::GetAvailableContentSpace();
        //             auto l_TextSize0  = ImGui::CalcTextSize( "Cell positions:" );

        //             auto l_TextSize1 = ImGui::CalcTextSize( "ID:" );
        //             UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //             UI::Text( "ID:" );
        //             UI::SameLine();
        //             UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize1.x ) + 10.0f, 0.0f
        //             ) ); UI::Text( "XXXX" );

        //             auto l_TextSize2 = ImGui::CalcTextSize( "Name:" );
        //             UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //             UI::Text( "Name:" );
        //             UI::SameLine();
        //             UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize2.x ) + 10.0f, 0.0f
        //             ) ); char buf[128] = { 0 }; std::strncpy( buf, ElementInFocus.Get<sTag>().mValue.c_str(), std::min(
        //             ElementInFocus.Get<sTag>().mValue.size(), std::size_t( 128 ) ) ); if( ImGui::InputText( "##TAG_INPUT",
        //             buf, ARRAYSIZE( buf ), ImGuiInputTextFlags_EnterReturnsTrue ) )
        //             {
        //                 ElementInFocus.Get<sTag>().mValue = std::string( buf );
        //             }

        //             auto l_TextSize3 = ImGui::CalcTextSize( "Cell positions:" );
        //             UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //             UI::Text( "Cell positions:" );
        //             UI::SameLine();
        //             UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize3.x ) + 10.0f, 0.0f
        //             ) ); UI::Text( "XXXX" );

        //             auto l_TextSize4 = ImGui::CalcTextSize( "Baseline:" );
        //             UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //             UI::Text( "Baseline:" );
        //             UI::SameLine();
        //             UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize4.x ) + 10.0f, 0.0f
        //             ) ); UI::Text( "XXXX" );

        //             auto l_TextSize5 = ImGui::CalcTextSize( "Static Noise:" );
        //             UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //             UI::Text( "Static Noise:" );
        //             UI::SameLine();
        //             UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize5.x ) + 10.0f, 0.0f
        //             ) ); UI::Text( "XXXX" );

        //             if( UI::Button( "Edit layout", { 100.0f, 35.0f } ) )
        //             {
        //                 PhotodetectorEditor.PhotodetectorToEdit = ElementInFocus;
        //                 PhotodetectorEditor.SensorModel         = SensorModel;
        //                 PhotodetectorEditor.Visible             = true;
        //             }

        //             if( PhotodetectorEditor.Visible )
        //                 ImGui::OpenPopup( "EDIT PHOTODETECTOR..." );
        //             PhotodetectorEditor.Display();
        //         }
        //         else if( ElementInFocus.Has<sLaserAssembly>() )
        //         {
        //             auto l_WindowSize = UI::GetAvailableContentSpace();
        //             auto l_TextSize0  = ImGui::CalcTextSize( "Timebase delay:" );

        //             auto l_TextSize1 = ImGui::CalcTextSize( "ID:" );
        //             UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //             UI::Text( "ID:" );
        //             UI::SameLine();
        //             UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize1.x ) + 10.0f, 0.0f
        //             ) ); UI::Text( "XXXX" );

        //             auto l_TextSize2 = ImGui::CalcTextSize( "Name:" );
        //             UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //             UI::Text( "Name:" );
        //             UI::SameLine();
        //             UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize2.x ) + 10.0f, 0.0f
        //             ) ); char buf[128] = { 0 }; std::strncpy( buf, ElementInFocus.Get<sTag>().mValue.c_str(), std::min(
        //             ElementInFocus.Get<sTag>().mValue.size(), std::size_t( 128 ) ) ); if( ImGui::InputText( "##TAG_INPUT",
        //             buf, ARRAYSIZE( buf ), ImGuiInputTextFlags_EnterReturnsTrue ) )
        //             {
        //                 ElementInFocus.Get<sTag>().mValue = std::string( buf );
        //             }

        //             auto l_TextSize3 = ImGui::CalcTextSize( "Pulse template:" );
        //             UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //             UI::Text( "Pulse template:" );
        //             UI::SameLine();
        //             UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize3.x ) + 10.0f, 0.0f
        //             ) ); UI::Text( "XXXX" );

        //             auto l_TextSize3a = ImGui::CalcTextSize( "Diffusion:" );
        //             UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //             UI::Text( "Diffusion:" );
        //             UI::SameLine();
        //             UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize3a.x ) + 10.0f,
        //             0.0f ) ); UI::Text( "XXXX" );

        //             auto l_TextSize4 = ImGui::CalcTextSize( "Timebase delay:" );
        //             UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //             UI::VectorComponentEditor( "Timebase delay:", ElementInFocus.Get<sLaserAssembly>().mTimebaseDelay,
        //             0.0, l_TextSize0.x );

        //             auto l_TextSize5 = ImGui::CalcTextSize( "Flash time:" );
        //             UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //             UI::VectorComponentEditor( "Flash time:", ElementInFocus.Get<sLaserAssembly>().mFlashTime, 0.0,
        //             l_TextSize0.x );
        //         }
        //         else if( ElementInFocus.Has<sSampler>() )
        //         {
        //             auto l_WindowSize = UI::GetAvailableContentSpace();
        //             auto l_TextSize0  = ImGui::CalcTextSize( "Timebase delay:" );

        //             auto l_TextSize1 = ImGui::CalcTextSize( "ID:" );
        //             UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //             UI::Text( "ID:" );
        //             UI::SameLine();
        //             UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize1.x ) + 10.0f, 0.0f
        //             ) ); UI::Text( "XXXX" );

        //             auto l_TextSize2 = ImGui::CalcTextSize( "Name:" );
        //             UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //             UI::Text( "Name:" );
        //             UI::SameLine();
        //             UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize2.x ) + 10.0f, 0.0f
        //             ) ); char buf[128] = { 0 }; std::strncpy( buf, ElementInFocus.Get<sTag>().mValue.c_str(), std::min(
        //             ElementInFocus.Get<sTag>().mValue.size(), std::size_t( 128 ) ) ); if( ImGui::InputText( "##TAG_INPUT",
        //             buf, ARRAYSIZE( buf ), ImGuiInputTextFlags_EnterReturnsTrue ) )
        //             {
        //                 ElementInFocus.Get<sTag>().mValue = std::string( buf );
        //             }

        //             auto l_TextSize3 = ImGui::CalcTextSize( "Basepoints:" );
        //             UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //             UI::Text( "Basepoints:" );
        //             UI::SameLine();
        //             UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize3.x ) + 10.0f, 0.0f
        //             ) ); UI::Text( "{}", ElementInFocus.Get<sSampler>().mLength );

        //             auto l_TextSize4 = ImGui::CalcTextSize( "Frequency:" );
        //             UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //             UI::Text( "Frequency:" );
        //             UI::SameLine();
        //             UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize4.x ) + 10.0f, 0.0f
        //             ) ); UI::Text( "{}", ElementInFocus.Get<sSampler>().mFrequency );
        //         }
        //         else if( ElementInFocus.Has<sDiffusionAssetTag>() )
        //         {
        //             auto l_TextSize0 = ImGui::CalcTextSize( "Cell positions:" );
        //             auto l_TextSize1 = ImGui::CalcTextSize( "ID:" );
        //             UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //             UI::Text( "ID:" );
        //             UI::SameLine();
        //             UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize1.x ) + 10.0f, 0.0f
        //             ) ); UI::Text( ElementInFocus.Get<sAssetMetadata>().mID );

        //             auto l_TextSize2 = ImGui::CalcTextSize( "Name:" );
        //             UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //             UI::Text( "Name:" );
        //             UI::SameLine();
        //             UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize2.x ) + 10.0f, 0.0f
        //             ) ); char buf[128] = { 0 }; std::strncpy( buf, ElementInFocus.Get<sTag>().mValue.c_str(), std::min(
        //             ElementInFocus.Get<sTag>().mValue.size(), std::size_t( 128 ) ) ); if( ImGui::InputText( "##TAG_INPUT",
        //             buf, ARRAYSIZE( buf ), ImGuiInputTextFlags_EnterReturnsTrue ) )
        //             {
        //                 ElementInFocus.Get<sTag>().mValue = std::string( buf );
        //             }

        //             auto l_TextSize3 = ImGui::CalcTextSize( "Type:" );
        //             UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //             UI::Text( "Type:" );
        //             UI::SameLine();
        //             UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize3.x ) + 10.0f, 0.0f
        //             ) ); UI::Text( "DIFFUSION" );
        //         }
        //         else if( ElementInFocus.Has<sPulseTemplateAssetTag>() )
        //         {
        //             auto l_TextSize0 = ImGui::CalcTextSize( "Cell positions:" );
        //             auto l_TextSize1 = ImGui::CalcTextSize( "ID:" );

        //             UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //             UI::Text( "ID:" );
        //             UI::SameLine();
        //             UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize1.x ) + 10.0f, 0.0f
        //             ) ); UI::Text( ElementInFocus.Get<sAssetMetadata>().mID );

        //             auto l_TextSize2 = ImGui::CalcTextSize( "Name:" );
        //             UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //             UI::Text( "Name:" );
        //             UI::SameLine();
        //             UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize2.x ) + 10.0f, 0.0f
        //             ) ); char buf[128] = { 0 }; std::strncpy( buf, ElementInFocus.Get<sTag>().mValue.c_str(), std::min(
        //             ElementInFocus.Get<sTag>().mValue.size(), std::size_t( 128 ) ) ); if( ImGui::InputText( "##TAG_INPUT",
        //             buf, ARRAYSIZE( buf ), ImGuiInputTextFlags_EnterReturnsTrue ) )
        //             {
        //                 ElementInFocus.Get<sTag>().mValue = std::string( buf );
        //             }

        //             auto l_TextSize3 = ImGui::CalcTextSize( "Type:" );
        //             UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //             UI::Text( "Type:" );
        //             UI::SameLine();
        //             UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize3.x ) + 10.0f, 0.0f
        //             ) ); UI::Text( "PULSE TEMPLATE" );
        //         }
        //         else if( ElementInFocus.Has<sStaticNoiseAssetTag>() )
        //         {
        //             auto l_TextSize0 = ImGui::CalcTextSize( "Cell positions:" );
        //             auto l_TextSize1 = ImGui::CalcTextSize( "ID:" );

        //             UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //             UI::Text( "ID:" );
        //             UI::SameLine();
        //             UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize1.x ) + 10.0f, 0.0f
        //             ) ); UI::Text( ElementInFocus.Get<sAssetMetadata>().mID );

        //             auto l_TextSize2 = ImGui::CalcTextSize( "Name:" );
        //             UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //             UI::Text( "Name:" );
        //             UI::SameLine();
        //             UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize2.x ) + 10.0f, 0.0f
        //             ) ); char buf[128] = { 0 }; std::strncpy( buf, ElementInFocus.Get<sTag>().mValue.c_str(), std::min(
        //             ElementInFocus.Get<sTag>().mValue.size(), std::size_t( 128 ) ) ); if( ImGui::InputText( "##TAG_INPUT",
        //             buf, ARRAYSIZE( buf ), ImGuiInputTextFlags_EnterReturnsTrue ) )
        //             {
        //                 ElementInFocus.Get<sTag>().mValue = std::string( buf );
        //             }

        //             auto l_TextSize3 = ImGui::CalcTextSize( "Type:" );
        //             UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //             UI::Text( "Type:" );
        //             UI::SameLine();
        //             UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize3.x ) + 10.0f, 0.0f
        //             ) ); UI::Text( "STATIC NOISE" );
        //         }
        //         else if( ElementInFocus.Has<sTileSpecificationComponent>() )
        //         {
        //             auto &l_TileSpecificationComponent = ElementInFocus.Get<sTileSpecificationComponent>();

        //             auto l_WindowSize  = UI::GetAvailableContentSpace();
        //             auto l_TextSize0   = ImGui::CalcTextSize( "Field of view hint:" );
        //             auto l_SliderSize0 = static_cast<float>( l_WindowSize.x ) - l_TextSize0.x - 35.0f;

        //             UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 5.0f ) );
        //             UI::Text( "ID: {}", ElementInFocus.Get<sTileSpecificationComponent>().mID );

        //             UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 5.0f ) );
        //             UI::VectorComponentEditor( "Field of view hint:", 0,
        //             ElementInFocus.Get<sTileSpecificationComponent>().mPosition, -10.0f, 10.0f, 0.01f, -10.0f, 10.0f,
        //             0.01f,
        //                                        0.0f, l_TextSize0.x );

        //             UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //             auto l_TextSize5 = ImGui::CalcTextSize( "Sampler:" );
        //             UI::Text( "Sampler:" );
        //             UI::SameLine();
        //             SamplerChooser lSamplerChooser( "##FOO" );
        //             lSamplerChooser.SensorModel = SensorModel;
        //             ImGui::SetNextItemWidth( l_SliderSize0 );
        //             UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize5.x ) + 10.0f,
        //             -5.0f ) ); lSamplerChooser.Display( Entity{} );

        //             UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 25.0f ) );
        //             auto l_TopLeft     = ImGui::GetCursorScreenPos() + ImVec2{ -10.0f, -10.0f };
        //             auto l_BottomRight = ImGui::GetCursorScreenPos() + ImVec2{ static_cast<float>( l_WindowSize.x ), 25.0f
        //             }; l_DrawList->AddRectFilled( l_TopLeft, l_BottomRight, IM_COL32( 5, 5, 5, 255 ) ); UI::Text(
        //             "FLASHES" ); UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 0.0f, 10.0f ) );

        //             auto &lFlashes                = ElementInFocus.Get<sRelationshipComponent>().mChildren;
        //             constexpr float lFlashTagSize = 40.0f;
        //             constexpr float lFlashPadding = 3.0f;

        //             uint32_t lFlasherPerRow = static_cast<uint32_t>( std::floor( ( l_WindowSize.x - 20.0f - (
        //             lFlashes.size() - 1 ) * lFlashPadding ) / lFlashTagSize ) );

        //             math::vec2 lFlashGroupCursorPosition = UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f );
        //             float lLeftMargin                    = lFlashGroupCursorPosition.x;
        //             uint32_t lCurrentHPosition           = 0;
        //             math::vec2 lCursorPosition           = lFlashGroupCursorPosition;

        //             if( UI::Button( "View flashes", { 100.0f, 35.0f } ) )
        //             {
        //                 FlashEditor.TileToEdit  = ElementInFocus;
        //                 FlashEditor.SensorModel = SensorModel;
        //                 FlashEditor.Visible     = true;
        //             }

        //             if( FlashEditor.Visible )
        //                 ImGui::OpenPopup( "EDIT FLASHES..." );
        //             FlashEditor.Display();
        //         }
        //         else if( ElementInFocus.Has<sTileLayoutComponent>() )
        //         {
        //             auto l_WindowSize = UI::GetAvailableContentSpace();
        //             auto l_TextSize0  = ImGui::CalcTextSize( "Field of view:" );

        //             auto l_TextSize1 = ImGui::CalcTextSize( "ID:" );
        //             UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //             UI::Text( "ID:" );
        //             UI::SameLine();
        //             UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize1.x ) + 10.0f, 0.0f
        //             ) ); UI::Text( "layout_id_goes_here" );

        //             auto l_TextSize2 = ImGui::CalcTextSize( "Name:" );
        //             UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //             UI::Text( "Name:" );
        //             UI::SameLine();
        //             UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize2.x ) + 10.0f, 0.0f
        //             ) ); char buf[128] = { 0 }; std::strncpy( buf, ElementInFocus.Get<sTag>().mValue.c_str(), std::min(
        //             ElementInFocus.Get<sTag>().mValue.size(), std::size_t( 128 ) ) ); if( ImGui::InputText( "##TAG_INPUT",
        //             buf, ARRAYSIZE( buf ), ImGuiInputTextFlags_EnterReturnsTrue ) )
        //             {
        //                 ElementInFocus.Get<sTag>().mValue = std::string( buf );
        //             }

        //             UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //             math::vec2 x{ 0.0f, 0.0f };
        //             UI::VectorComponentEditor( "Field of view:", 0, x, 0.0, 2.0, 0.01, 0.1, 2.0, 0.01, 0.0, l_TextSize0.x
        //             );

        //             if( UI::Button( "Edit layout", { 100.0f, 35.0f } ) )
        //             {
        //                 LayoutEditor.LayoutToEdit = ElementInFocus;
        //                 LayoutEditor.SensorModel  = SensorModel;
        //                 LayoutEditor.Visible      = true;
        //             }

        //             if( LayoutEditor.Visible )
        //                 ImGui::OpenPopup( "EDIT TILE LAYOUT..." );
        //             LayoutEditor.Display();
        //         }
        //         else
        //         {
        //         }
        //     }
        // }
        // ImGui::End();

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

        // if( ImGui::Begin( "ENVIRONMENT", &p_open, ImGuiWindowFlags_None ) )
        // {
        //     auto l_DrawList   = ImGui::GetWindowDrawList();
        //     auto l_WindowSize = UI::GetAvailableContentSpace();

        //     auto l_TextSize0   = ImGui::CalcTextSize( "Sampling resolution:" );
        //     auto l_SliderSize0 = static_cast<float>( l_WindowSize.x ) - l_TextSize0.x - 35.0f;

        //     if( Sensor )
        //     {
        //         auto &lSensorTransform = Sensor.Get<sNodeTransformComponent>();
        //         math::vec3 lRotation   = lSensorTransform.GetEulerRotation();
        //         math::vec3 lPosition   = lSensorTransform.GetTranslation();
        //         bool lPositionChanged  = false;
        //         bool lRotationChanged  = false;

        //         UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //         auto l_TopLeft0     = ImGui::GetCursorScreenPos() + ImVec2{ -10.0f, -10.0f };
        //         auto l_BottomRight0 = ImGui::GetCursorScreenPos() + ImVec2{ static_cast<float>( l_WindowSize.x ), 25.0f };
        //         l_DrawList->AddRectFilled( l_TopLeft0, l_BottomRight0, IM_COL32( 5, 5, 5, 255 ) );
        //         Text( "Sensor position:" );
        //         UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 0.0f, 15.0f ) );

        //         UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 5.0f ) );
        //         lPositionChanged = UI::VectorComponentEditor( "Position:", lPosition, 0.0, l_TextSize0.x );
        //         UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 0.0f, 5.0f ) );

        //         UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 5.0f ) );
        //         lRotationChanged = UI::VectorComponentEditor( "Orientation:", lRotation, 0.0, l_TextSize0.x );
        //         UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 0.0f, 5.0f ) );

        //         if( lPositionChanged || lRotationChanged )

        //             Sensor.Replace<sNodeTransformComponent>( lPosition, lRotation, math::vec3{ 1.0f } );

        //             Sensor.Replace<sLocalTransformComponent>( lPosition, lRotation, math::vec3{ 1.0f } );
        //         l_DrawList->AddRectFilled( l_TopLeft, l_BottomRight, IM_COL32( 5, 5, 5, 255 ) );
        //         Text( "Environment sampling:" );
        //         UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 15.0f ) );

        //         UI::VectorComponentEditor( "Sampling resolution:", 0,
        //         Sensor.Get<EnvironmentSampler::sCreateInfo>().mSamplingResolution, 0.1, 2.0, 0.01, 0.1, 2.0, 0.01, 0.2,
        //                                    l_TextSize0.x );

        //         auto l_TextSize1 = ImGui::CalcTextSize( "Multisampling:" );
        //         UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //         Text( "Multisampling:" );
        //         UI::SameLine();
        //         UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize1.x ) + 10.0f, -5.0f )
        //         ); ImGui::SetNextItemWidth( l_SliderSize0 ); UI::Slider( "##multisampling", "%d", 1, 128,
        //         &Sensor.Get<EnvironmentSampler::sCreateInfo>().mMultiSamplingFactor );

        //         UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        //         Text( "Use regular multisampling" );
        //         ImGui::SameLine();
        //         UI::SetCursorPosition( ImVec2( static_cast<float>( l_WindowSize.x ) - 32.0f, ImGui::GetCursorPos().y
        //         - 5.0f ) ); UI::Checkbox( "##use_regular_multisampling",
        //         &Sensor.Get<EnvironmentSampler::sCreateInfo>().mUseRegularMultiSampling );
        //     }
        // }
        // ImGui::End();

        if( ImGui::Begin( "PROPERTIES", &p_open, ImGuiWindowFlags_None ) )
        {
            auto l_WindowPropertiesSize = UI::GetAvailableContentSpace();
            m_SceneHierarchyPanel.ElementEditor.Display( l_WindowPropertiesSize.x, l_WindowPropertiesSize.y );
        }
        ImGui::End();

        if( o_RequestQuit ) return true;

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
            if( m_LastFPS > 0 )
                Text( "Render: {} fps ({:.2f} ms/frame)", m_LastFPS, ( 1000.0f / m_LastFPS ) );
            else
                Text( "Render: 0 fps (0 ms)" );
        }
        ImGui::End();

        return false;
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
                auto lFilePath = FileDialogs::OpenFile(
                    mEngineLoop->GetMainApplicationWindow(), "glTf Files (*.gltf)\0*.gltf\0All Files (*.*)\0*.*\0" );
                if( lFilePath.has_value() ) LoadScenario( lFilePath.value() );
            }
            if( UI::MenuItem( fmt::format( "{} New material", ICON_FA_PLUS_CIRCLE ).c_str(), NULL ) )
            {
                // m_NewMaterial.Visible = true;
            }
            if( UI::MenuItem( fmt::format( "{} Save Scene", ICON_FA_PLUS_CIRCLE ).c_str(), NULL ) )
            {
                // m_NewMaterial.Visible = true;
                World->SaveAs( fs::path( "C:\\GitLab\\SpockEngine\\Saved" ) / "TEST" / "SCENE" );
            }

            if( UI::MenuItem( fmt::format( "{} Load Scenario Scene", ICON_FA_PLUS_CIRCLE ).c_str(), NULL ) )
            {
                // m_NewMaterial.Visible = true;
                World->LoadScenario( fs::path( "C:\\GitLab\\SpockEngine\\Saved" ) / "TEST" / "SCENE" / "Scene.yaml" );
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

    void EditorWindow::ClearScene()
    {
        // auto lSensorTransform    = Sensor.TryGet<sNodeTransformComponent>( sNodeTransformComponent{ math::mat4( 1.0f ) }
        // ); auto lEnvironmentSampler = Sensor.TryGet<EnvironmentSampler::sCreateInfo>( EnvironmentSampler::sCreateInfo{} );
        // AcquisitionSpecification lAcqCreateInfo{};
        // lAcqCreateInfo.mBasePoints     = 100;
        // lAcqCreateInfo.mOversampling   = 1;
        // auto lAcquisitionSpecification = Sensor.TryGet<AcquisitionSpecification>( lAcqCreateInfo );

        // World->ClearScene();
        // m_SceneHierarchyPanel.SelectedElement             = Entity{};
        // m_SceneHierarchyPanel.ElementEditor.ElementToEdit = Entity{};

        // // Add sensor entity to the scene
        // Sensor = World->Create( "Sensor", World->Root );
        // Sensor.Add<sNodeTransformComponent>( lSensorTransform );
        // Sensor.Add<sLocalTransformComponent>( lSensorTransform );
        // Sensor.Add<EnvironmentSampler::sCreateInfo>( lEnvironmentSampler );
        // // Add a particle system to the sensor to display the point cloud
        // auto &l_SensorPointCloud = Sensor.Add<sParticleSystemComponent>();

        // // Create particle renderer for the point cloud
        // auto &l_SensorPointCloudRenderer    = Sensor.Add<RendererComponent>();
        // l_SensorPointCloudRenderer.Material = World->CreateEntity( "ParticleSystemMaterial" );
        // l_SensorPointCloudRenderer.Material.Add<sParticleShaderComponent>();
    }

    void EditorWindow::LoadScenario( fs::path aPath )
    {
        auto lImporter = New<GlTFImporter>( aPath );
        auto lName     = aPath.filename().string();

        auto lNewModel = World->LoadModel( lImporter, math::mat4( 1.0f ), lName );
        // lNewModel.Add<LockComponent>();
        World->ForEach<sStaticMeshComponent>( [&]( auto aEntity, auto &aComponent ) { World->MarkAsRayTracingTarget( aEntity ); } );
    }

    void EditorWindow::Workspace( int32_t width, int32_t height )
    {
        static bool s_DisplayCameraSettings = false;

        auto &lIO = ImGui::GetIO();

        math::vec2 l_WorkspacePosition = UI::GetCurrentCursorScreenPosition();
        math::vec2 l_CursorPosition    = UI::GetCurrentCursorPosition();

        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.0f, 0.0f, 0.0f, 0.0f } );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 1.0f, 1.0f, 1.0f, 0.01f } );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 1.0f, 1.0f, 1.0f, 0.02f } );

        UI::SetCursorPosition( l_CursorPosition );
        if( ImGui::ImageButton( (ImTextureID)m_CameraIconHandle.Handle->GetVkDescriptorSet(), ImVec2{ 22.0f, 22.0f },
                ImVec2{ 0.0f, 0.0f }, ImVec2{ 1.0f, 1.0f }, 0, ImVec4{ 0.0f, 0.0f, 0.0f, 0.0f }, ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f } ) )
        {
            s_DisplayCameraSettings = !s_DisplayCameraSettings;
        }
        UI::SameLine();
        float                   l_SliderSize0            = 150.0f;
        static ManipulationType lCurrentManipulationType = ManipulationType::ROTATION;
        ManipulationTypeChooser lManipulationTypeChooser( "##FOO_2" );
        ImGui::SetNextItemWidth( l_SliderSize0 );
        lManipulationTypeChooser.Dropdown.Labels = { "Rotation", "Translation", "Scale" };
        lManipulationTypeChooser.Dropdown.Values = {
            ManipulationType::ROTATION, ManipulationType::TRANSLATION, ManipulationType::SCALE };
        lManipulationTypeChooser.Display( lCurrentManipulationType );
        if( lManipulationTypeChooser.Dropdown.Changed ) lCurrentManipulationType = lManipulationTypeChooser.Dropdown.GetValue();

        math::vec2  l3DViewPosition = UI::GetCurrentCursorScreenPosition();
        math::ivec2 l3DViewSize     = UI::GetAvailableContentSpace();
        m_WorkspaceAreaSize         = l3DViewSize;

        UI::SameLine();

        if( ActiveWorld->GetState() == Scene::eSceneState::EDITING )
        {

            if( ImGui::ImageButton( (ImTextureID)m_PlayIconHandle.Handle->GetVkDescriptorSet(), ImVec2{ 22.0f, 22.0f },
                    ImVec2{ 0.0f, 0.0f }, ImVec2{ 1.0f, 1.0f }, 0, ImVec4{ 0.0f, 0.0f, 0.0f, 0.0f },
                    ImVec4{ 0.0f, 1.0f, 0.0f, 0.8f } ) )
            {
                if( OnBeginScenario ) OnBeginScenario();

                ActiveWorld = New<Scene>( World );
                ActiveWorld->BeginScenario();
                WorldRenderer->m_World = ActiveWorld;
            }
        }
        else
        {
            if( ImGui::ImageButton( (ImTextureID)m_PauseIconHandle.Handle->GetVkDescriptorSet(), ImVec2{ 22.0f, 22.0f },
                    ImVec2{ 0.0f, 0.0f }, ImVec2{ 1.0f, 1.0f }, 0, ImVec4{ 0.0f, 0.0f, 0.0f, 0.0f },
                    ImVec4{ 1.0f, .2f, 0.0f, 0.8f } ) )
            {
                if( OnEndScenario ) OnEndScenario();

                ActiveWorld->EndScenario();
                ActiveWorld = World;
                // ActiveSensor = Sensor;

                WorldRenderer->m_World = ActiveWorld;
            }
        }
        UI::SameLine();

        ImGui::PushStyleColor(
            ImGuiCol_Text, WorldRenderer->RenderGizmos ? ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f } : ImVec4{ 1.0f, 1.0f, 1.0f, .2f } );
        if( UI::Button( "Gizmos", math::vec2{ 65.0f, 24.0f } ) )
        {
            WorldRenderer->RenderGizmos = !WorldRenderer->RenderGizmos;
        }
        ImGui::PopStyleColor();
        UI::SameLine();
        ImGui::PushStyleColor(
            ImGuiCol_Text, WorldRenderer->RenderCoordinateGrid ? ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f } : ImVec4{ 1.0f, 1.0f, 1.0f, .2f } );
        if( UI::Button( "Grid", math::vec2{ 65.0f, 24.0f } ) )
        {
            WorldRenderer->RenderCoordinateGrid = !WorldRenderer->RenderCoordinateGrid;
        }
        ImGui::PopStyleColor();
        UI::SameLine();

        ImGui::PushStyleColor(
            ImGuiCol_Text, WorldRenderer->GrayscaleRendering ? ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f } : ImVec4{ 1.0f, 1.0f, 1.0f, .2f } );
        if( UI::Button( "Grayscale", math::vec2{ 65.0f, 24.0f } ) )
        {
            WorldRenderer->GrayscaleRendering = !WorldRenderer->GrayscaleRendering;
        }
        ImGui::PopStyleColor();
        ImGui::PopStyleColor();
        ImGui::PopStyleColor();
        ImGui::PopStyleColor();

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

            math::mat4 lCameraView     = math::Inverse( WorldRenderer->View.View );
            math::vec3 lCameraPosition = math::Translation( lCameraView );
            math::mat3 lRotationMatrix = math::Rotation( lCameraView );

            float lRotationSpeed = 2.5_degf;
            float lPanningSpeed  = 0.25f;

            bool lViewLanipulate               = ViewManipulate( WorldRenderer->View.CameraPosition, WorldRenderer->View.View,
                              l3DViewPosition + math::vec2{ l3DViewSize.x - 125.0f, 35.0f } );
            WorldRenderer->View.CameraPosition = math::vec3( math::Inverse( WorldRenderer->View.View )[3] );

            ManipulationConfig l_Manipulator{};
            l_Manipulator.Type             = lCurrentManipulationType;
            l_Manipulator.Projection       = WorldRenderer->View.Projection;
            l_Manipulator.WorldTransform   = WorldRenderer->View.View;
            l_Manipulator.ViewportPosition = l3DViewPosition;
            l_Manipulator.ViewportSize     = l3DViewSize;

            if( m_SceneHierarchyPanel.SelectedElement &&
                m_SceneHierarchyPanel.SelectedElement.Has<sNodeTransformComponent>() )
            {
                auto& lSensorTransform = m_SceneHierarchyPanel.SelectedElement.Get<sNodeTransformComponent>();
                Manipulate( l_Manipulator, lSensorTransform.mMatrix );
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
                        WorldRenderer->View.View = math::Inverse( math::FromComponents( lNewRotation, lCameraPosition ) );
                    }
                    else if( lIO.KeyAlt )
                    {
                        math::mat3 lNewRotation =
                            math::mat3( math::Rotation( -lDragDelta.y, lRotationMatrix * math::vec3{ 1.0f, 0.0f, 0.0f } ) ) *
                            lRotationMatrix;
                        WorldRenderer->View.View = math::Inverse( math::FromComponents( lNewRotation, lCameraPosition ) );
                    }
                    else if( lIO.KeyCtrl )
                    {
                        math::mat3 lNewRotation =
                            math::mat3( math::Rotation( -lDragDelta.x, lRotationMatrix * math::vec3{ 0.0f, 0.0f, 1.0f } ) ) *
                            lRotationMatrix;
                        WorldRenderer->View.View = math::Inverse( math::FromComponents( lNewRotation, lCameraPosition ) );
                    }
                    else
                    {
                        math::mat3 lNewRotationX =
                            math::mat3( math::Rotation( -lDragDelta.x, lRotationMatrix * math::vec3{ 0.0f, 1.0f, 0.0f } ) );
                        math::mat3 lNewRotationY =
                            math::mat3( math::Rotation( -lDragDelta.y, lRotationMatrix * math::vec3{ 1.0f, 0.0f, 0.0f } ) );
                        WorldRenderer->View.View =
                            math::Inverse( math::FromComponents( lNewRotationX * lNewRotationY * lRotationMatrix, lCameraPosition ) );
                    }
                    ImGui::ResetMouseDragDelta( ImGuiMouseButton_Left );
                }

                if( ImGui::IsMouseDragging( ImGuiMouseButton_Right ) )
                {

                    ImVec2     lDragDelta0 = ImGui::GetMouseDragDelta( ImGuiMouseButton_Right );
                    math::vec4 lDragDelta{ lDragDelta0.x, lDragDelta0.y, 0.0f, 1.0f };
                    lDragDelta = math::Inverse( WorldRenderer->View.Projection ) * lDragDelta;
                    lDragDelta = lDragDelta / lDragDelta.w;

                    math::vec3 lPanAmount    = lRotationMatrix * math::vec3{ -lDragDelta.x, -lDragDelta.y, 0.0f };
                    math::vec3 lNewPosition  = lCameraPosition + lPanAmount;
                    WorldRenderer->View.View = math::Inverse( math::FromComponents( lRotationMatrix, lNewPosition ) );

                    ImGui::ResetMouseDragDelta( ImGuiMouseButton_Right );
                }
            }

            if( ImGui::IsItemHovered() )
            {
                if( ImGui::IsKeyPressed( ImGuiKey_UpArrow ) )
                {
                    if( lIO.KeyShift )
                    {
                        math::vec3 lPanAmount    = lRotationMatrix * math::vec3{ 0.0f, lPanningSpeed, 0.0f };
                        math::vec3 lNewPosition  = lCameraPosition + lPanAmount;
                        WorldRenderer->View.View = math::Inverse( math::FromComponents( lRotationMatrix, lNewPosition ) );
                    }
                    else
                    {
                        math::vec3 lPanAmount    = lRotationMatrix * math::vec3{ 0.0f, 0.0f, -lPanningSpeed };
                        math::vec3 lNewPosition  = lCameraPosition + lPanAmount;
                        WorldRenderer->View.View = math::Inverse( math::FromComponents( lRotationMatrix, lNewPosition ) );
                    }
                }
                else if( ImGui::IsKeyPressed( ImGuiKey_DownArrow ) )
                {
                    if( lIO.KeyShift )
                    {
                        math::vec3 lPanAmount    = lRotationMatrix * math::vec3{ 0.0f, -lPanningSpeed, 0.0f };
                        math::vec3 lNewPosition  = lCameraPosition + lPanAmount;
                        WorldRenderer->View.View = math::Inverse( math::FromComponents( lRotationMatrix, lNewPosition ) );
                    }
                    else
                    {
                        math::vec3 lPanAmount    = lRotationMatrix * math::vec3{ 0.0f, 0.0f, lPanningSpeed };
                        math::vec3 lNewPosition  = lCameraPosition + lPanAmount;
                        WorldRenderer->View.View = math::Inverse( math::FromComponents( lRotationMatrix, lNewPosition ) );
                    }
                }
                else if( ImGui::IsKeyPressed( ImGuiKey_LeftArrow ) )
                {
                    math::vec3 lNewPosition  = lCameraPosition + lRotationMatrix * math::vec3{ -lPanningSpeed, 0.0f, 0.0f };
                    WorldRenderer->View.View = math::Inverse( math::FromComponents( lRotationMatrix, lNewPosition ) );
                }
                else if( ImGui::IsKeyPressed( ImGuiKey_RightArrow ) )
                {
                    math::vec3 lNewPosition  = lCameraPosition + lRotationMatrix * math::vec3{ lPanningSpeed, 0.0f, 0.0f };
                    WorldRenderer->View.View = math::Inverse( math::FromComponents( lRotationMatrix, lNewPosition ) );
                }

                if( ImGui::IsKeyPressed( ImGuiKey_W ) )
                {
                    math::mat3 lNewRotation =
                        math::mat3( math::Rotation( -lRotationSpeed, lRotationMatrix * math::vec3{ 1.0f, 0.0f, 0.0f } ) ) *
                        lRotationMatrix;
                    WorldRenderer->View.View = math::Inverse( math::FromComponents( lNewRotation, lCameraPosition ) );
                }
                else if( ImGui::IsKeyPressed( ImGuiKey_S ) )
                {
                    math::mat3 lNewRotation =
                        math::mat3( math::Rotation( lRotationSpeed, lRotationMatrix * math::vec3{ 1.0f, 0.0f, 0.0f } ) ) *
                        lRotationMatrix;
                    WorldRenderer->View.View = math::Inverse( math::FromComponents( lNewRotation, lCameraPosition ) );
                }
                else if( ImGui::IsKeyPressed( ImGuiKey_A ) )
                {
                    math::mat3 lNewRotation =
                        math::mat3( math::Rotation( lRotationSpeed, lRotationMatrix * math::vec3{ 0.0f, 1.0f, 0.0f } ) ) *
                        lRotationMatrix;
                    WorldRenderer->View.View = math::Inverse( math::FromComponents( lNewRotation, lCameraPosition ) );
                }
                else if( ImGui::IsKeyPressed( ImGuiKey_D ) )
                {
                    math::mat3 lNewRotation =
                        math::mat3( math::Rotation( -lRotationSpeed, lRotationMatrix * math::vec3{ 0.0f, 1.0f, 0.0f } ) ) *
                        lRotationMatrix;
                    WorldRenderer->View.View = math::Inverse( math::FromComponents( lNewRotation, lCameraPosition ) );
                }
            }

            if( s_DisplayCameraSettings )
            {
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

                if( ImGui::Begin( "##CAMERA_SETTINGS", &s_DisplayCameraSettings, lStatusBarFlags ) )
                {
                    math::ivec2        l_WindowSize = UI::GetAvailableContentSpace();
                    ImGuiTreeNodeFlags l_Flags      = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_AllowItemOverlap;
                    float              l_LabelSize  = 175.0f;
                    if( ImGui::CollapsingHeader( "Environment", l_Flags ) )
                    {
                        if( World->Environment.Has<sAmbientLightingComponent>() )
                        {
                            auto &l_AmbientLightComponent = World->Environment.Get<sAmbientLightingComponent>();
                            UI::ColorChooser( "Ambient light color:", 175, l_AmbientLightComponent.Color );
                            Text( "Ambient intensity:" );
                            UI::SameLine();
                            ImVec2 l_CursorPosition = ImGui::GetCursorPos();
                            UI::SetCursorPosition( ImVec2{ l_LabelSize, l_CursorPosition.y } + ImVec2( 0.0f, -5.0f ) );
                            ImGui::SetNextItemWidth( l_WindowSize.x - l_LabelSize );
                            UI::Slider( "##ambient_intensity", "%.3f", 0.0f, 0.1f, &l_AmbientLightComponent.Intensity );
                        }

                        if( World->Environment.Has<sBackgroundComponent>() )
                        {
                            auto &l_BackgroundComponent = World->Environment.Get<sBackgroundComponent>();
                            UI::ColorChooser( "Background color:", 175, l_BackgroundComponent.Color );
                        }
                    }

                    if( ImGui::CollapsingHeader( "Camera", l_Flags ) )
                    {
                        ImGui::SliderFloat( "Exposure", &WorldRenderer->Settings.Exposure, 0.1f, 10.0f );
                        ImGui::SliderFloat( "Gamma", &WorldRenderer->Settings.Gamma, 0.1f, 4.0f );
                    }
                    ImGui::End();
                }
                ImGui::PopStyleVar( 3 );
                ImGui::PopStyleColor();
            }
        }
    }

    void EditorWindow::Console( int32_t width, int32_t height )
    {
        const float           TEXT_BASE_HEIGHT = ImGui::GetTextLineHeightWithSpacing();
        const ImGuiTableFlags flags            = ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY |
                                      ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_Resizable |
                                      ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable;

        auto  &l_Logs     = LTSE::Logging::GetLogMessages();
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

} // namespace LTSE::Editor