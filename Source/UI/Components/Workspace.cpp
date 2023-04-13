#include "Workspace.h"
#include "DotNet/Runtime.h"

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

    void *UIWorkspaceDocument::UIWorkspaceDocument_Create()
    {
        auto lNewDocument = new UIWorkspaceDocument();

        return static_cast<void *>( lNewDocument );
    }

    void UIWorkspaceDocument::UIWorkspaceDocument_Destroy( void *aInstance )
    {
        delete static_cast<UIWorkspaceDocument *>( aInstance );
    }

    void UIWorkspaceDocument::UIWorkspaceDocument_SetContent( void *aInstance, void *aContent )
    {
        auto lInstance = static_cast<UIWorkspaceDocument *>( aInstance );
        auto lContent  = static_cast<UIComponent *>( aContent );

        lInstance->SetContent( lContent );
    }

    void UIWorkspaceDocument::UIWorkspaceDocument_Update( void *aInstance )
    {
        auto lInstance = static_cast<UIWorkspaceDocument *>( aInstance );

        lInstance->Update();
    }

    void UIWorkspaceDocument::UIWorkspaceDocument_SetName( void *aInstance, void *aName )
    {
        auto lInstance = static_cast<UIWorkspaceDocument *>( aInstance );
        auto lName     = DotNetRuntime::NewString( static_cast<MonoString *>( aName ) );

        lInstance->mName = lName;
    }

    bool UIWorkspaceDocument::UIWorkspaceDocument_IsDirty( void *aInstance )
    {
        auto lInstance = static_cast<UIWorkspaceDocument *>( aInstance );

        return lInstance->mDirty;
    }

    void UIWorkspaceDocument::UIWorkspaceDocument_MarkAsDirty( void *aInstance, bool aDirty )
    {
        auto lInstance = static_cast<UIWorkspaceDocument *>( aInstance );

        lInstance->mDirty = aDirty;
    }

    void UIWorkspaceDocument::UIWorkspaceDocument_Open( void *aInstance )
    {
        auto lInstance = static_cast<UIWorkspaceDocument *>( aInstance );

        lInstance->DoOpen();
    }

    void UIWorkspaceDocument::UIWorkspaceDocument_RequestClose( void *aInstance )
    {
        auto lInstance = static_cast<UIWorkspaceDocument *>( aInstance );

        lInstance->DoQueueClose();
    }

    void UIWorkspaceDocument::UIWorkspaceDocument_ForceClose( void *aInstance )
    {
        auto lInstance = static_cast<UIWorkspaceDocument *>( aInstance );

        lInstance->DoForceClose();
    }

    void UIWorkspaceDocument::UIWorkspaceDocument_RegisterSaveDelegate( void *aInstance, void *aDelegate )
    {
        auto lInstance = static_cast<UIWorkspaceDocument *>( aInstance );
        auto lDelegate = static_cast<MonoObject *>( aDelegate );

        if( lInstance->mSaveDelegate != nullptr ) mono_gchandle_free( lInstance->mSaveDelegateHandle );

        lInstance->mSaveDelegate       = aDelegate;
        lInstance->mSaveDelegateHandle = mono_gchandle_new( static_cast<MonoObject *>( aDelegate ), true );

        lInstance->mDoSave = [lInstance, lDelegate]()
        {
            auto lDelegateClass = mono_object_get_class( lDelegate );
            auto lInvokeMethod  = mono_get_delegate_invoke( lDelegateClass );

            auto lValue = mono_runtime_invoke( lInvokeMethod, lDelegate, nullptr, nullptr );

            return *( (bool *)mono_object_unbox( lValue ) );
        };
    }

    void UIWorkspace::PushStyles() {}
    void UIWorkspace::PopStyles() {}

    ImVec2 UIWorkspace::RequiredSize()
    {
        auto lTextSize = ImVec2{ 0.0f, 0.0f };

        return lTextSize;
    }

    void UIWorkspace::Add( UIWorkspaceDocument *aDocument ) { mDocuments.push_back( aDocument ); }
    void UIWorkspace::Add( Ref<UIWorkspaceDocument> aDocument )
    {
        Add( aDocument.get() );

        mDocumentRefs.push_back( aDocument );
    }

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
                    lDocument->DoOpen();
                    lDocument->DoQueueClose();
                }

                if( lVisible )
                {
                    lDocument->Update();
                    ImGui::EndTabItem();
                }
            }

            ImGui::EndTabBar();
        }

        std::copy_if( mDocuments.begin(), mDocuments.end(), std::back_inserter( mCloseQueue ),
                      []( auto x ) { return x->mWantClose; } );
        std::for_each( mDocuments.begin(), mDocuments.end(), []( auto x ) { x->mWantClose = false; } );

        // Display closing confirmation UI
        if( !mCloseQueue.empty() )
        {
            int lUnsavedDocumentCount = std::count_if( mCloseQueue.begin(), mCloseQueue.end(), []( auto x ) { return x->mDirty; } );

            if( lUnsavedDocumentCount == 0 )
            {
                // Close documents when all are unsaved
                for( int n = 0; n < mCloseQueue.size(); n++ ) mCloseQueue[n]->DoForceClose();
                UpdateDocumentList();
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
                            mCloseQueue[n]->DoSave();
                            mCloseQueue[n]->DoForceClose();
                        }

                        UpdateDocumentList();
                        // mCloseQueue.clear();
                        ImGui::CloseCurrentPopup();
                    }
                    ImGui::SameLine();
                    if( ImGui::Button( "No", button_size ) )
                    {
                        for( int n = 0; n < mCloseQueue.size(); n++ ) mCloseQueue[n]->DoForceClose();
                        UpdateDocumentList();
                        // mCloseQueue.clear();
                        ImGui::CloseCurrentPopup();
                    }
                    ImGui::SameLine();
                    if( ImGui::Button( "Cancel", button_size ) )
                    {
                        UpdateDocumentList();
                        // mCloseQueue.clear();
                        ImGui::CloseCurrentPopup();
                    }
                    ImGui::EndPopup();
                }
            }
        }
    }

    void UIWorkspace::UpdateDocumentList()
    {
        std::vector<UIWorkspaceDocument *> lOpenedDocuments;
        std::copy_if( mDocuments.begin(), mDocuments.end(), std::back_inserter( lOpenedDocuments ),
                      []( UIWorkspaceDocument *x ) { return x->mOpen; } );

        mDocuments = std::move( lOpenedDocuments );

        if( mOnCloseDocuments ) mOnCloseDocuments( mCloseQueue );
        mCloseQueue.clear();
    }

    void *UIWorkspace::UIWorkspace_Create()
    {
        auto lNewWorkspace = new UIWorkspace();

        return static_cast<void *>( lNewWorkspace );
    }

    void UIWorkspace::UIWorkspace_Destroy( void *aSelf ) { delete static_cast<UIWorkspace *>( aSelf ); }

    void UIWorkspace::UIWorkspace_Add( void *aSelf, void *aDocument )
    {
        auto lSelf     = static_cast<UIWorkspace *>( aSelf );
        auto lDocument = static_cast<UIWorkspaceDocument *>( aDocument );

        lSelf->Add( lDocument );
    }

} // namespace SE::Core