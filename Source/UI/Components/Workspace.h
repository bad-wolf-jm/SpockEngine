#pragma once

#include "Component.h"

namespace SE::Core
{
    struct UIWorkspaceDocument : public UIComponent
    {
        std::string mName;
        bool        mOpen      = true;
        bool        mOpenPrev  = true;
        bool        mDirty     = false;
        bool        mWantClose = false;

        UIWorkspaceDocument()                              = default;
        UIWorkspaceDocument( UIWorkspaceDocument const & ) = default;

        void DoOpen() { mOpen = true; }

        void DoQueueClose() { mWantClose = true; }

        void DoForceClose()
        {
            mOpen  = false;
            mDirty = false;
        }

        void DoSave() { mDirty = false; }

        void   PushStyles();
        void   PopStyles();
        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

        UIComponent *mContent = nullptr;

        void SetContent( UIComponent *aContent );

        void Update();
    };

    class UIWorkspace : public UIComponent
    {
      public:
        UIWorkspace()  = default;
        ~UIWorkspace() = default;

        void Add( Ref<UIWorkspaceDocument> aDocument );

      protected:
        std::vector<Ref<UIWorkspaceDocument>> mDocuments;
        std::vector<Ref<UIWorkspaceDocument>> mCloseQueue;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core