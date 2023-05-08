#pragma once

#include "UI/Components/Component.h"
#include "UI/Components/Label.h"
#include "UI/Components/Image.h"

#include "UI/Layouts/BoxLayout.h"

namespace SE::Core
{
    class UITreeView;

    class UITreeViewNode : public UIComponent
    {
      public:
        UITreeViewNode() = default;
        UITreeViewNode(UITreeView* aTreeView, UITreeViewNode *aParent);

        void SetIcon(UIImage* aImage);
        void SetText( std::string const &aText );
        void SetTextColor( math::vec4 aColor );

        UITreeViewNode* Add();

      protected:
        ImGuiTreeNodeFlags mFlags;

        float mIndent = 0.0f;
        UIImage* mIcon;
        UILabel* mNode;
        UIBoxLayout* mActions;
        UIBoxLayout* mNodeLayout;

        UITreeView* mTreeView;
        UITreeViewNode *mParent;
        std::vector<UITreeViewNode*> mChildren;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

      private:
        void TreePushOverrideID( );
        void TreePop();
        bool IsOpen();
        bool IsLeaf();
        bool RenderNode();
        void RenderArrow( ImDrawList *aDrawList, ImVec2 aPosition, ImU32 aColor, ImGuiDir aDirection, float aScale );

      public:
        static void *UITreeViewNode_Create();
        static void  UITreeViewNode_Destroy( void *aInstance );        
        static void  UITreeViewNode_SetIcon( void *aInstance, void *aIcon );
        static void  UITreeViewNode_SetText( void *aInstance, void *aText );
        static void  UITreeViewNode_SetTextColor( void *aInstance, math::vec4 aTextColor );
        static void *UITreeViewNode_Add( void *aInstance );
    };

    class UITreeView : public UIComponent
    {
        public:
        UITreeView();

        void SetIndent(float aIndent);
        UITreeViewNode* Add();

        protected:
            float mIndent = 15.0f;
            UITreeViewNode* mRoot; 

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

      public:
        static void *UITreeView_Create();
        static void  UITreeView_Destroy( void *aInstance );
        static void  UITreeView_SetIndent( void *aInstance, float aIndent );
        static void  *UITreeView_Add( void *aInstance );

        friend class UITreeViewNode;
    };
} // namespace SE::Core