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

        static void *UIWorkspaceDocument_Create();
        static void  UIWorkspaceDocument_Destroy( void *aInstance );
        static void  UIWorkspaceDocument_SetName( void *aInstance, void *aName );
        static void  UIWorkspaceDocument_SetContent( void *aInstance, void *aContent );
        static void  UIWorkspaceDocument_Update( void *aInstance );
        static bool  UIWorkspaceDocument_IsDirty( void *aInstance );
        static void  UIWorkspaceDocument_MarkAsDirty( void *aInstance, bool aDirty );
        static void  UIWorkspaceDocument_Open( void *aInstance );
        static void  UIWorkspaceDocument_RequestClose( void *aInstance );
        static void  UIWorkspaceDocument_ForceClose( void *aInstance );
        static void  UIWorkspaceDocument_RegisterSaveDelegate( void *aInstance, void *aDelegate );
    };

    class UIWorkspace : public UIComponent
    {
      public:
        UIWorkspace()  = default;
        ~UIWorkspace() = default;

        void Add( UIWorkspaceDocument *aDocument );
        void Add( ref_t<UIWorkspaceDocument> aDocument );

        std::function<void( vector_t<UIWorkspaceDocument *> )> mOnCloseDocuments;

      protected:
        vector_t<UIWorkspaceDocument *> mDocuments;
        vector_t<UIWorkspaceDocument *> mCloseQueue;

        vector_t<ref_t<UIWorkspaceDocument>> mDocumentRefs;
        vector_t<ref_t<UIWorkspaceDocument>> mCloseQueueRefs;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

      private:
        void UpdateDocumentList();

        void *mCloseDocumentDelegate       = nullptr;
        int   mCloseDocumentDelegateHandle = -1;

      public:
        static void *UIWorkspace_Create();
        static void  UIWorkspace_Destroy( void *aSelf );
        static void  UIWorkspace_Add( void *aSelf, void *aDocument );
        static void  UIWorkspace_RegisterCloseDocumentDelegate( void *aSelf, void *aDelegate );
    };
} // namespace SE::Core