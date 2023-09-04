#include <filesystem>

#include "ContentBrowser.h"

#include "Core/CUDA/Texture/TextureData.h"

#include "UI/UI.h"
#include "UI/Widgets.h"

namespace SE::Editor
{
    ContentBrowser::ContentBrowser( ref_t<IGraphicContext> aGraphicContext, ref_t<UIContext> aUIOverlay, fs::path aRoot )
        : mGraphicContext{ aGraphicContext }
        , mCurrentDirectory( aRoot )
        , Root{ aRoot }
    {
        {
            SE::Core::sTextureCreateInfo lTextureCreateInfo{};
            TextureData2D        lTextureData( lTextureCreateInfo, "C:\\GitLab\\SpockEngine\\Saved\\Resources\\Icons\\Folder.png" );
            sTextureSamplingInfo lSamplingInfo{};
            TextureSampler2D     lTextureSampler = TextureSampler2D( lTextureData, lSamplingInfo );

            auto lTexture         = CreateTexture2D( mGraphicContext, lTextureData );
            mDirectoryIcon       = CreateSampler2D( mGraphicContext, lTexture, lSamplingInfo );
            mDirectoryIconHandle = aUIOverlay->CreateTextureHandle( mDirectoryIcon );
        }

        {
            SE::Core::sTextureCreateInfo lTextureCreateInfo{};
            TextureData2D        lTextureData( lTextureCreateInfo, "C:\\GitLab\\SpockEngine\\Saved\\Resources\\Icons\\File.png" );
            sTextureSamplingInfo lSamplingInfo{};
            TextureSampler2D     lTextureSampler = TextureSampler2D( lTextureData, lSamplingInfo );

            auto lTexture    = CreateTexture2D( mGraphicContext, lTextureData );
            mFileIcon       = CreateSampler2D( mGraphicContext, lTexture, lSamplingInfo );
            mFileIconHandle = aUIOverlay->CreateTextureHandle( mFileIcon );
        }

        {
            SE::Core::sTextureCreateInfo lTextureCreateInfo{};
            TextureData2D        lTextureData( lTextureCreateInfo, "C:\\GitLab\\SpockEngine\\Saved\\Resources\\Icons\\Back.png" );
            sTextureSamplingInfo lSamplingInfo{};
            TextureSampler2D     lTextureSampler = TextureSampler2D( lTextureData, lSamplingInfo );

            auto lTexture    = CreateTexture2D( mGraphicContext, lTextureData );
            mBackIcon       = CreateSampler2D( mGraphicContext, lTexture, lSamplingInfo );
            mBackIconHandle = aUIOverlay->CreateTextureHandle( mBackIcon );
        }
    }

    void ContentBrowser::Display()
    {
        bool lBackButtonDisabled = mCurrentDirectory == std::filesystem::path( Root );

        if( lBackButtonDisabled ) ImGui::BeginDisabled();

        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 0, 0, 0, 0 ) );
        if( ImGui::ImageButton( (ImTextureID)mBackIconHandle.Handle->GetID(), ImVec2{ 15, 15 } ) )
        {
            mCurrentDirectory = mCurrentDirectory.parent_path();
        }
        ImGui::PopStyleColor();
        if( lBackButtonDisabled ) ImGui::EndDisabled();

        float cellSize = mThumbnailSize + mPadding + mTextSize;

        float panelWidth  = ImGui::GetContentRegionAvail().x;
        int   columnCount = (int)( panelWidth / cellSize );
        if( columnCount < 1 ) columnCount = 1;

        ImGui::Columns( columnCount, 0, false );
        ImDrawList *draw_list = ImGui::GetWindowDrawList();

        vector_t<fs::path> lFolderContent;
        vector_t<fs::path> lFiles;

        for( auto &directoryEntry : std::filesystem::directory_iterator( mCurrentDirectory ) )
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
            auto        relativePath   = std::filesystem::relative( path, Root );
            string_t filenameString = relativePath.filename().string();

            ImGui::PushID( filenameString.c_str() );
            auto icon = std::filesystem::is_directory( directoryEntry ) ? mDirectoryIconHandle : mFileIconHandle;

            auto lPos0 = UI::GetCurrentCursorPosition();
            ImGui::Dummy( ImVec2{ cellSize, mThumbnailSize } );

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
                if( std::filesystem::is_directory( directoryEntry ) ) mCurrentDirectory /= path.filename();
            }

            UI::SetCursorPosition( lPos0 );
            UI::Image( icon, math::vec2{ mThumbnailSize, mThumbnailSize } );
            UI::SameLine();
            auto lTextSize = ImGui::CalcTextSize( filenameString.c_str() );
            UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2{ 0.0f, ( mThumbnailSize - lTextSize.y ) / 2.0f } );
            ImGui::Text( filenameString.c_str() );
            ImGui::NextColumn();

            ImGui::PopID();
        }

        ImGui::Columns( 1 );
    }

} // namespace SE::Editor