#pragma once

#include <mutex>

#include "UI/Components/Component.h"
#include "UI/Components/Label.h"
#include "UI/Components/Image.h"

#include "UI/Layouts/BoxLayout.h"
#include "UI/Layouts/StackLayout.h"

namespace SE::Core
{
    class UITreeView;

    class UITreeViewNode : public UIComponent
    {
      public:
        UITreeViewNode() = default;
        UITreeViewNode(UITreeView* aTreeView, UITreeViewNode *aParent);

        void SetIcon(UIImage* aImage);
        void SetIndicator(UIComponent* aImage);
        void SetText( string_t const &aText );
        void SetTextColor( math::vec4 aColor );

        virtual vector_t<UITreeViewNode*> const& Children();

        UITreeViewNode* Add();

        void OnSelected( std::function<void(UITreeViewNode*)> aOnSelected );

      protected:
        std::function<void(UITreeViewNode*)> mOnSelected;

      protected:
        ImGuiTreeNodeFlags mFlags;

        UIImage* mIcon = nullptr;

        ref_t<UIStackLayout> mImage  = nullptr;
        ref_t<UIStackLayout> mIndicator = nullptr;
        ref_t<UILabel>       mText   = nullptr;
        ref_t<UIBoxLayout>   mLayout = nullptr;

        UITreeView* mTreeView;
        UITreeViewNode *mParent;
        vector_t<UITreeViewNode*> mChildren;

        uint32_t mLevel = 0;
        bool mIsOpen= false;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

      public:
        bool IsOpen();
        void RenderNode();
        void RenderArrow( ImDrawList *aDrawList, ImVec2 aPosition, ImU32 aColor, ImGuiDir aDirection, float aScale );
        void RenderIcon( ImDrawList *aDrawList, ImVec2 aPosition );
        
        virtual bool IsLeaf();
    };

    class UITreeView : public UIComponent
    {
        public:
        UITreeView();

        void SetIndent(float aIndent);
        void SetIconSpacing(float aSpacing);
        UITreeViewNode* Add();
        void UpdateRows();

        protected:
            float mIndent = 5.0f;
            float mIconSpacing = 12.0f;
            uint32_t mCurrentID = 0;
            UITreeViewNode* mRoot; 
            std::vector<UITreeViewNode*> mRows;
            std::mutex mRowsLock;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
        friend class UITreeViewNode;
    };
} // namespace SE::Core