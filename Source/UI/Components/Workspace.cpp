#include "Workspace.h"

#include <algorithm>
#include <iterator>

namespace SE::Core
{

    void   UIWorkspaceDocument::PushStyles() {}
    void   UIWorkspaceDocument::PopStyles() {}
    ImVec2 UIWorkspaceDocument::RequiredSize() { return ImVec2{}; }
    void   UIWorkspaceDocument::DrawContent( ImVec2 aPosition, ImVec2 aSize ) {}
    void   UIWorkspaceDocument::SetContent( UIComponent *aContent ) { mContent = aContent; }

    void UIWorkspaceDocument::Update()
    {
        ImVec2 lContentSize     = ImGui::GetContentRegionAvail();
        ImVec2 lContentPosition = ImGui::GetCursorPos();

        if( mContent != nullptr ) mContent->Update( lContentPosition, lContentSize );
    }

    void UIWorkspace::PushStyles() {}
    void UIWorkspace::PopStyles() {}

    ImVec2 UIWorkspace::RequiredSize()
    {
        auto lTextSize = ImVec2{ 0.0f, 0.0f };

        return lTextSize;
    }

    void UIWorkspace::Add( Ref<UIWorkspaceDocument> aDocument ) { mDocuments.push_back( aDocument ); }

    void UIWorkspace::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        ImGuiTabBarFlags lTabBarFlags = ImGuiTabBarFlags_FittingPolicyDefault_ | ImGuiTabBarFlags_Reorderable;
        if( ImGui::BeginTabBar( "##tabs", lTabBarFlags ) )
        {
            for( int i = 0; i < mDocuments.size(); i++ )
            {
                auto &lDocument = mDocuments[i];
                if( !lDocument->mOpen && lDocument->mOpenPrev ) ImGui::SetTabItemClosed( lDocument->mName.c_str() );
                lDocument->mOpenPrev = lDocument->mOpen;
            }

            // Submit Tabs
            for( int i = 0; i < mDocuments.size(); i++ )
            {
                auto lDocument = mDocuments[i];
                if( !lDocument->mOpen ) continue;

                ImGuiTabItemFlags lCurrentTabFlags = ( lDocument->mDirty ? ImGuiTabItemFlags_UnsavedDocument : 0 );

                bool lVisible = ImGui::BeginTabItem( lDocument->mName.c_str(), &lDocument->mOpen, lCurrentTabFlags );

                // Cancel attempt to close when unsaved add to save queue so we can display a popup.
                if( !lDocument->mOpen && lDocument->mDirty )
                {
                    lDocument->mOpen = true;
                    lDocument->DoQueueClose();
                }

                // MyDocument::DisplayContextMenu( lDocument );
                if( lVisible )
                {
                    lDocument->Update();
                    ImGui::EndTabItem();
                }
            }

            ImGui::EndTabBar();
        }

        if( mCloseQueue.empty() )
        {
            std::copy_if( mDocuments.begin(), mDocuments.end(), std::back_inserter( mCloseQueue ),
                          []( auto x ) { return x->mWantClose; } );
            std::for_each( mDocuments.begin(), mDocuments.end(), []( auto x ) { x->mWantClose = false; } );
        }

        // Display closing confirmation UI
        if( !mCloseQueue.empty() )
        {
            int lUnsavedDocumentCount = std::count_if( mCloseQueue.begin(), mCloseQueue.end(), []( auto x ) { return x->mDirty; } );
            // for( int n = 0; n < mCloseQueue.size(); n++ )
            //     if( mCloseQueue[n]->mDirty ) lUnsavedDocumentCount++;

            if( lUnsavedDocumentCount == 0 )
            {
                // Close documents when all are unsaved
                for( int n = 0; n < mCloseQueue.size(); n++ ) mCloseQueue[n]->DoForceClose();
                mCloseQueue.clear();
            }
            else
            {
                if( !ImGui::IsPopupOpen( "Save?" ) ) ImGui::OpenPopup( "Save?" );
                if( ImGui::BeginPopupModal( "Save?", NULL, ImGuiWindowFlags_AlwaysAutoResize ) )
                {
                    ImGui::Text( "Save change to the following items?" );
                    float item_height = ImGui::GetTextLineHeightWithSpacing();
                    if( ImGui::BeginChildFrame( ImGui::GetID( "frame" ), ImVec2( -FLT_MIN, 6.25f * item_height ) ) )
                    {
                        for( int n = 0; n < mCloseQueue.size(); n++ )
                            if( mCloseQueue[n]->mDirty ) ImGui::Text( "%s", mCloseQueue[n]->mName );
                        ImGui::EndChildFrame();
                    }

                    ImVec2 button_size( ImGui::GetFontSize() * 7.0f, 0.0f );
                    if( ImGui::Button( "Yes", button_size ) )
                    {
                        for( int n = 0; n < mCloseQueue.size(); n++ )
                        {
                            if( mCloseQueue[n]->mDirty ) mCloseQueue[n]->DoSave();
                            mCloseQueue[n]->DoForceClose();
                        }
                        mCloseQueue.clear();
                        ImGui::CloseCurrentPopup();
                    }
                    ImGui::SameLine();
                    if( ImGui::Button( "No", button_size ) )
                    {
                        for( int n = 0; n < mCloseQueue.size(); n++ ) mCloseQueue[n]->DoForceClose();
                        mCloseQueue.clear();
                        ImGui::CloseCurrentPopup();
                    }
                    ImGui::SameLine();
                    if( ImGui::Button( "Cancel", button_size ) )
                    {
                        mCloseQueue.clear();
                        ImGui::CloseCurrentPopup();
                    }
                    ImGui::EndPopup();
                }
            }
        }

        std::vector<Ref<UIWorkspaceDocument>> lOpenedDocuments;
        std::copy_if( mDocuments.begin(), mDocuments.end(), std::back_inserter( lOpenedDocuments ),
                      []( Ref<UIWorkspaceDocument> x ) { return x->mOpen; } );
        mDocuments = std::move( lOpenedDocuments );
    }
} // namespace SE::Core