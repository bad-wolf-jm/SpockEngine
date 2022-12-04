#include <filesystem>

#include "ContentBrowser.h"

#include "Core/GPUResource/Texture/TextureData.h"

#include "UI/UI.h"
#include "UI/Widgets.h"

namespace SE::Editor
{
    ContentBrowser::ContentBrowser( GraphicContext &aGraphicContext, Ref<UIContext> aUIOverlay, fs::path aRoot )
        : mGraphicContext{ aGraphicContext }
        , m_CurrentDirectory( aRoot )
        , Root{ aRoot }
    {
        {
            SE::Core::TextureData::sCreateInfo lTextureCreateInfo{};
            TextureData2D lTextureData( lTextureCreateInfo, "C:\\GitLab\\SpockEngine\\Saved\\Resources\\Icons\\Folder.png" );
            sTextureSamplingInfo lSamplingInfo{};
            TextureSampler2D lTextureSampler = TextureSampler2D( lTextureData, lSamplingInfo );

            m_DirectoryIcon       = New<Texture2D>( mGraphicContext, lTextureData, lTextureSampler );
            m_DirectoryIconHandle = aUIOverlay->CreateTextureHandle( m_DirectoryIcon );
        }

        {
            SE::Core::TextureData::sCreateInfo lTextureCreateInfo{};
            TextureData2D lTextureData( lTextureCreateInfo, "C:\\GitLab\\SpockEngine\\Saved\\Resources\\Icons\\File.png" );
            sTextureSamplingInfo lSamplingInfo{};
            TextureSampler2D lTextureSampler = TextureSampler2D( lTextureData, lSamplingInfo );

            m_FileIcon       = New<Texture2D>( mGraphicContext, lTextureData, lTextureSampler );
            m_FileIconHandle = aUIOverlay->CreateTextureHandle( m_FileIcon );
        }

        {
            SE::Core::TextureData::sCreateInfo lTextureCreateInfo{};
            TextureData2D lTextureData( lTextureCreateInfo, "C:\\GitLab\\SpockEngine\\Saved\\Resources\\Icons\\Back.png" );
            sTextureSamplingInfo lSamplingInfo{};
            TextureSampler2D lTextureSampler = TextureSampler2D( lTextureData, lSamplingInfo );

            m_BackIcon       = New<Texture2D>( mGraphicContext, lTextureData, lTextureSampler );
            m_BackIconHandle = aUIOverlay->CreateTextureHandle( m_BackIcon );
        }
    }

    void ContentBrowser::Display()
    {
        bool lBackButtonDisabled = m_CurrentDirectory == std::filesystem::path( Root );

        if( lBackButtonDisabled )
            ImGui::BeginDisabled();

        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 0, 0, 0, 0 ) );
        if( ImGui::ImageButton( (ImTextureID)m_BackIconHandle.Handle->GetVkDescriptorSet(), ImVec2{ 15, 15 } ) )
        {
            m_CurrentDirectory = m_CurrentDirectory.parent_path();
        }
        ImGui::PopStyleColor();
        if( lBackButtonDisabled )
            ImGui::EndDisabled();

        float cellSize = thumbnailSize + padding + textSize;

        float panelWidth = ImGui::GetContentRegionAvail().x;
        int columnCount  = (int)( panelWidth / cellSize );
        if( columnCount < 1 )
            columnCount = 1;

        ImGui::Columns( columnCount, 0, false );
        ImDrawList *draw_list = ImGui::GetWindowDrawList();

        std::vector<fs::path> lFolderContent;
        std::vector<fs::path> lFiles;

        for( auto &directoryEntry : std::filesystem::directory_iterator( m_CurrentDirectory ) )
        {
            if( directoryEntry.is_directory() )
                lFolderContent.push_back( directoryEntry );
            else
                lFiles.push_back( directoryEntry );
        }

        lFolderContent.insert( lFolderContent.end(), lFiles.begin(), lFiles.end() );

        for( auto &directoryEntry : lFolderContent )
        {
            const auto &path           = directoryEntry;
            auto relativePath          = std::filesystem::relative( path, Root );
            std::string filenameString = relativePath.filename().string();

            ImGui::PushID( filenameString.c_str() );
            auto icon = std::filesystem::is_directory( directoryEntry ) ? m_DirectoryIconHandle : m_FileIconHandle;

            auto lPos0 = UI::GetCurrentCursorPosition();
            ImGui::Dummy( ImVec2{ cellSize, thumbnailSize } );

            if( ImGui::BeginDragDropSource( ImGuiDragDropFlags_SourceAllowNullID ) )
            {
                const wchar_t *itemPath = path.c_str();
                ImGui::SetDragDropPayload( "CONTENT_BROWSER_ITEM", itemPath, ( wcslen( itemPath ) + 1 ) * sizeof( wchar_t ) );
                ImGui::EndDragDropSource();
            }

            if( ImGui::IsItemHovered() )
            {
                draw_list->AddRectFilled( ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), IM_COL32( 255, 0, 255, 5 ) );
                draw_list->AddRect( ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), IM_COL32( 255, 0, 255, 25 ) );
            }

            if( ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked( ImGuiMouseButton_Left ) )
            {
                if( std::filesystem::is_directory( directoryEntry ) )
                    m_CurrentDirectory /= path.filename();
            }

            UI::SetCursorPosition( lPos0 );
            UI::Image( icon, math::vec2{ thumbnailSize, thumbnailSize } );
            UI::SameLine();
            auto lTextSize = ImGui::CalcTextSize( filenameString.c_str() );
            UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2{ 0.0f, ( thumbnailSize - lTextSize.y ) / 2.0f } );
            ImGui::Text( filenameString.c_str() );
            ImGui::NextColumn();

            ImGui::PopID();
        }

        ImGui::Columns( 1 );
    }

} // namespace SE::Editor