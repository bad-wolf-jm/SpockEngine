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

        void DoOpen()
        {
            mOpen = true;
        }

        void DoQueueClose()
        {
            mWantClose = true;
        }

        void DoForceClose()
        {
            mOpen  = false;
            mDirty = false;
        }

        std::function<bool()> mDoSave;
        void                  DoSave()
        {
            mDirty = mDoSave ? mDoSave() : false;
        }

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

        std::function<void( vector_t<UIWorkspaceDocument *> )> mOnCloseDocuments;

      protected:
        vector_t<UIWorkspaceDocument *> mDocuments;
        vector_t<UIWorkspaceDocument *> mCloseQueue;

        vector_t<Ref<UIWorkspaceDocument>> mDocumentRefs;
        vector_t<Ref<UIWorkspaceDocument>> mCloseQueueRefs;

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