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
    };
} // namespace SE::Core