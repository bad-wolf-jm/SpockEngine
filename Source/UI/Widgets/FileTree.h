#pragma once

#include "UI/Components/Component.h"
#include "UI/Components/Image.h"
#include "UI/Components/Label.h"
#include "UI/Components/TreeView.h"

#include "UI/Layouts/BoxLayout.h"
#include "UI/Layouts/StackLayout.h"

namespace SE::Core
{
    class UIFileTree;

    class UIFileTreeNode : public UITreeViewNode
    {
      public:
        UIFileTreeNode() = default;
        UIFileTreeNode( UIFileTree *aTreeView, UIFileTreeNode *aParent, fs::path const &aPath, std::string const &aName );

        // void SetIcon(UIImage* aImage);
        // void SetIndicator(UIComponent* aImage);
        // void SetText( std::string const &aText );
        // void SetTextColor( math::vec4 aColor );

        UIFileTreeNode *Add();

        std::vector<UITreeViewNode *> const &Children();

      protected:
        fs::path    mPath;
        std::string mName;
        // ImGuiTreeNodeFlags mFlags;

        // Ref<UIStackLayout> mImage  = nullptr;
        // Ref<UIStackLayout> mIndicator = nullptr;
        // Ref<UILabel>       mText   = nullptr;
        // Ref<UIBoxLayout>   mLayout = nullptr;

        // UIFileTree* mTreeView;
        // UIFileTreeNode *mParent;
        // std::vector<UIFileTreeNode*> mChildren;

      protected:
        // void PushStyles();
        // void PopStyles();

        // ImVec2 RequiredSize();
        // void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

      private:
        // void TreePushOverrideID( );
        // void TreePop();
        // bool IsOpen();
        // bool IsLeaf();
        // bool RenderNode();
        // void RenderArrow( ImDrawList *aDrawList, ImVec2 aPosition, ImU32 aColor, ImGuiDir aDirection, float aScale );

      public:
        static void *UIFileTreeNode_Create();
        static void  UIFileTreeNode_Destroy( void *aInstance );
        // static void  UIFileTreeNode_SetIcon( void *aInstance, void *aIcon );
        // static void  UIFileTreeNode_SetIndicator( void *aInstance, void *aIndicator );
        // static void  UIFileTreeNode_SetText( void *aInstance, void *aText );
        // static void  UIFileTreeNode_SetTextColor( void *aInstance, math::vec4 aTextColor );
        // static void *UIFileTreeNode_Add( void *aInstance );
    };

    class UIFileTree : public UITreeView
    {
      public:
        UIFileTree();

        // void SetIndent(float aIndent);
        UIFileTreeNode *Add();

      protected:
        // float mIndent = 5.0f;
        // UIFileTreeNode* mRoot;

      protected:
        // void PushStyles();
        // void PopStyles();

        // ImVec2 RequiredSize();
        // void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

      public:
        static void *UIFileTree_Create();
        static void  UIFileTree_Destroy( void *aInstance );
        // static void  UIFileTree_SetIndent( void *aInstance, float aIndent );
        static void *UIFileTree_Add( void *aInstance );

        friend class UIFileTreeNode;
    };
} // namespace SE::Core