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

        static void *UIWorkspaceDocument_Create();
        static void  UIWorkspaceDocument_Destroy( void *aInstance );
        static void  UIWorkspaceDocument_SetContent( void *aInstance, void *aContent );
        static void  UIWorkspaceDocument_Update( void *aInstance );
    };

    class UIWorkspace : public UIComponent
    {
      public:
        UIWorkspace()  = default;
        ~UIWorkspace() = default;

        void Add( UIWorkspaceDocument* aDocument );
        void Add( Ref<UIWorkspaceDocument> aDocument );

      protected:
        std::vector<UIWorkspaceDocument*> mDocuments;
        std::vector<UIWorkspaceDocument*> mCloseQueue;

        std::vector<Ref<UIWorkspaceDocument>> mDocumentRefs;
        std::vector<Ref<UIWorkspaceDocument>> mCloseQueueRefs;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

      public:
        static void *UIWorkspace_Create();
        static void  UIWorkspace_Destroy( void *aSelf );
        static void  UIWorkspace_Add( void *aSelf, void *aDocument );
    };
} // namespace SE::Core