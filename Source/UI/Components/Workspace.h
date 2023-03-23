#pragma once

#include "Component.h"

namespace SE::Core
{
    struct UIWorkspaceDocument : public UIComponent
    {
        std::string mName;      // Document title
        bool        mOpen;      // Set when open (we keep an array of all available documents to simplify demo code!)
        bool        mOpenPrev;  // Copy of Open from last update.
        bool        mDirty;     // Set when the document has been modified
        bool        mWantClose; // Set when the document

        UIWorkspaceDocument() = default;

        void DoOpen() { mOpen = true; }

        void DoQueueClose() { mWantClose = true; }

        void DoForceClose()
        {
            mOpen  = false;
            mDirty = false;
        }

        void DoSave() { mDirty = false; }
    };

    class UIWorkspace : public UIComponent
    {
      public:
        UIWorkspace() = default;
        ~UIWorkspace() = default;


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