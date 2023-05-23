#pragma once

#include "UI/Components/Component.h"
#include "UI/Components/Label.h"
#include "UI/Components/Image.h"

#include "UI/Layouts/BoxLayout.h"
#include "UI/Layouts/StackLayout.h"

namespace SE::Core
{
    class UIFileTree;

    class UIFileNode : public UIFileTreeNode
    {
      public:
        UIFileNode() = default;
        UIFileNode(UIFileTree* aTreeView, UIFileNode *aParent);

        void SetIcon(UIImage* aImage);
        void SetIndicator(UIComponent* aImage);
        void SetText( std::string const &aText );
        void SetTextColor( math::vec4 aColor );

        UIFileNode* Add();

      protected:
        ImGuiTreeNodeFlags mFlags;

        Ref<UIStackLayout> mImage  = nullptr;
        Ref<UIStackLayout> mIndicator = nullptr;
        Ref<UILabel>       mText   = nullptr;
        Ref<UIBoxLayout>   mLayout = nullptr;

        UIFileTree* mTreeView;
        UIFileNode *mParent;
        std::vector<UIFileNode*> mChildren;

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
        static void *UIFileNode_Create();
        static void  UIFileNode_Destroy( void *aInstance );        
        static void  UIFileNode_SetIcon( void *aInstance, void *aIcon );
        static void  UIFileNode_SetIndicator( void *aInstance, void *aIndicator );
        static void  UIFileNode_SetText( void *aInstance, void *aText );
        static void  UIFileNode_SetTextColor( void *aInstance, math::vec4 aTextColor );
        static void *UIFileNode_Add( void *aInstance );
    };

    class UIFileTree : public UIComponent
    {
        public:
        UIFileTree();

        void SetIndent(float aIndent);
        UIFileNode* Add();

        protected:
            float mIndent = 5.0f;
            UIFileNode* mRoot; 

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

      public:
        static void *UIFileTree_Create();
        static void  UIFileTree_Destroy( void *aInstance );
        static void  UIFileTree_SetIndent( void *aInstance, float aIndent );
        static void  *UIFileTree_Add( void *aInstance );

        friend class UIFileNode;
    };
} // namespace SE::Core