#pragma once

#include "UI/Components/LComponent.h"
#include "UI/Components/Label.h"
#include "UI/Components/Image.h"

#include "UI/Layouts/BoxLayout.h"

namespace SE::Core
{
    class UITreeViewNode : public UIComponent
    {
      public:
        UITreeViewNode() = default;
        UITreeViewNode(std::string const& aText) = default;

        void SetIcon(UIImage* aImage);

        void SetIndent(float aIndent);
        float GetIndent();

        void SetText( std::string const &aText );
        void SetTextColor( math::vec4 aColor );

        void SetActions(std::vector<UIComponent*> aActions);
        void ClearActions();

      protected:
        ImGuiID mID;
        ImGuiTreeNodeFlags mFlags;

        float mIndent = 0.0f;
        UIImage* mIcon;
        UILabel* mNode;
        UIBoxLayout* mActions;
        UIBoxLayout* mNodeLayout;

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
        void RenderArrow( ImDrawList *aDrawList, ImVec2 aPosition, ImU32 aColor, ImGuiDir aDirection, float aScale )

      public:
        static void *UITreeViewNode_Create();
        static void *UITreeViewNode_CreateWithText( void *aText );
        static void  UITreeViewNode_Destroy( void *aInstance );        
        static void  UITreeViewNode_SetText( void *aInstance, void *aText );
        static void  UITreeViewNode_SetTextColor( void *aInstance, math::vec4 aTextColor );
        static void  UITreeViewNode_SetIcon( void *aInstance, void *aIcon );
        static void  UITreeViewNode_SetActions( void *aInstance, void *aActions );
        static void  UITreeViewNode_ClearActions( void *aInstance, void *aActions );
        static void  UITreeViewNode_SetIndent(void *aInstance, float aIndent);
        static float  UITreeViewNode_GetIndent(void *aInstance );




    };
} // namespace SE::Core