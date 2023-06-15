#pragma once

#include "Component.h"
#include <functional>

namespace SE::Core
{
    struct UIWorkspaceDocument : public UIComponent
    {
        string_t mName;

        bool mOpen      = true;
        bool mOpenPrev  = true;
        bool mDirty     = false;
        bool mWantClose = false;

        UIWorkspaceDocument()                              = default;
        UIWorkspaceDocument( UIWorkspaceDocument const & ) = default;

        void DoOpen() { mOpen = true; }

        void DoQueueClose() { mWantClose = true; }

        void DoForceClose()
        {
            mOpen  = false;
            mDirty = false;
        }

        std::function<bool()> mDoSave;
        void                  DoSave() { mDirty = mDoSave ? mDoSave() : false; }

        void   PushStyles();
        void   PopStyles();
        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

        UIComponent *mContent = nullptr;

        void SetContent( UIComponent *aContent );

        void Update();

        void *mSaveDelegate       = nullptr;
        int   mSaveDelegateHandle = -1;

        // static void *UIWorkspaceDocument_Create();
        // static void  UIWorkspaceDocument_Destroy( void *aInstance );
        // static void  UIWorkspaceDocument_SetName( void *aInstance, void *aName );
        // static void  UIWorkspaceDocument_SetContent( void *aInstance, void *aContent );
        // static void  UIWorkspaceDocument_Update( void *aInstance );
        // static bool  UIWorkspaceDocument_IsDirty( void *aInstance );
        // static void  UIWorkspaceDocument_MarkAsDirty( void *aInstance, bool aDirty );
        // static void  UIWorkspaceDocument_Open( void *aInstance );
        // static void  UIWorkspaceDocument_RequestClose( void *aInstance );
        // static void  UIWorkspaceDocument_ForceClose( void *aInstance );
        // static void  UIWorkspaceDocument_RegisterSaveDelegate( void *aInstance, void *aDelegate );
    };

    class UIWorkspace : public UIComponent
    {
      public:
        UIWorkspace()  = default;
        ~UIWorkspace() = default;

        void Add( UIWorkspaceDocument *aDocument );
        void Add( Ref<UIWorkspaceDocument> aDocument );

        std::function<void( std::vector<UIWorkspaceDocument *> )> mOnCloseDocuments;

      protected:
        std::vector<UIWorkspaceDocument *> mDocuments;
        std::vector<UIWorkspaceDocument *> mCloseQueue;

        std::vector<Ref<UIWorkspaceDocument>> mDocumentRefs;
        std::vector<Ref<UIWorkspaceDocument>> mCloseQueueRefs;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

      public:
        void UpdateDocumentList();

        void *mCloseDocumentDelegate       = nullptr;
        int   mCloseDocumentDelegateHandle = -1;

    //   public:
    //     static void *UIWorkspace_Create();
    //     static void  UIWorkspace_Destroy( void *aSelf );
    //     static void  UIWorkspace_Add( void *aSelf, void *aDocument );
    //     static void  UIWorkspace_RegisterCloseDocumentDelegate( void *aSelf, void *aDelegate );
    };
} // namespace SE::Core