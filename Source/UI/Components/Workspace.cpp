#include "Workspace.h"

namespace SE::Core
{

    UIWorkspace::UIWorkspace( std::string const &aText )
        : mText{ aText }
    {
    }

    void UIWorkspace::PushStyles() {}
    void UIWorkspace::PopStyles() {}

    ImVec2 UIWorkspace::RequiredSize()
    {
        auto lTextSize = ImVec2{ 0.0f, 0.0f };

        return lTextSize;
    }

    void UIWorkspace::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        ImGuiTabBarFlags lTabBarFlags = ImGuiTabBarFlags_FittingPolicyDefault_ | ImGuiTabBarFlags_Reorderable;
        if( ImGui::BeginTabBar( "##tabs", lTabBarFlags ) )
        {
            // if( opt_reorderable ) NotifyOfDocumentsClosedElsewhere( app );
            for( int doc_n = 0; doc_n < mDocuments.Size; doc_n++ )
            {
                auto &doc = mDocuments[doc_n];
                if( !doc->mOpen && doc->mOpenPrev ) ImGui::SetTabItemClosed( doc->mName );
                doc->mOpenPrev = doc->mOpen;
            }

            // Submit Tabs
            for( int doc_n = 0; doc_n < mDocuments.size(); doc_n++ )
            {
                auto doc = mDocuments[doc_n];
                if( !doc->mOpen ) continue;

                ImGuiTabItemFlags lCurrentTabFlags = ( doc->mDirty ? ImGuiTabItemFlags_UnsavedDocument : 0 );

                bool lVisible = ImGui::BeginTabItem( doc->Name, &doc->Open, lCurrentTabFlags );

                // Cancel attempt to close when unsaved add to save queue so we can display a popup.
                if( !doc->mOpen && doc->mDirty )
                {
                    doc->mOpen = true;
                    doc->DoQueueClose();
                }

                // MyDocument::DisplayContextMenu( doc );
                if( lVisible )
                {
                    doc->Update( aPosition, aSize );
                    MyDocument::DisplayContents( doc );
                    ImGui::EndTabItem();
                }
            }

            ImGui::EndTabBar();
        }

        
        if( mCloseQueue.empty() )
        {
            // Close queue is locked once we started a popup
            for( int doc_n = 0; doc_n < mDocuments.Size; doc_n++ )
            {
                MyDocument *doc = &mDocuments[doc_n];
                if( doc->WantClose )
                {
                    doc->WantClose = false;
                    mCloseQueue.push_back( doc );
                }
            }
        }

        // Display closing confirmation UI
        if( !mCloseQueue.empty() )
        {
            int close_queue_unsaved_documents = 0;
            for( int n = 0; n < mCloseQueue.size(); n++ )
                if( mCloseQueue[n]->Dirty ) close_queue_unsaved_documents++;

            if( close_queue_unsaved_documents == 0 )
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
                            if( mCloseQueue[n]->Dirty ) ImGui::Text( "%s", mCloseQueue[n]->Name );
                        ImGui::EndChildFrame();
                    }

                    ImVec2 button_size( ImGui::GetFontSize() * 7.0f, 0.0f );
                    if( ImGui::Button( "Yes", button_size ) )
                    {
                        for( int n = 0; n < mCloseQueue.size(); n++ )
                        {
                            if( mCloseQueue[n]->Dirty ) mCloseQueue[n]->DoSave();
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

    } // namespace SE::Core