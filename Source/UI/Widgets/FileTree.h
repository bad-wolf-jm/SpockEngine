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
        UIFileTreeNode( UIFileTree *aTreeView, UIFileTreeNode *aParent, path_t const &aPath, string_t const &aName );

        UIFileTreeNode *Add(path_t const& aPath);

        std::vector<UITreeViewNode *> const &Children();

      protected:
        path_t    mPath;
        string_t mName;

        bool IsLeaf();
    };

    class UIFileTree : public UITreeView
    {
      public:
        UIFileTree();

        UIFileTreeNode *Add(path_t const& aPath);

    //   public:
    //     static void *UIFileTree_Create();
    //     static void  UIFileTree_Destroy( void *aInstance );
    //     static void *UIFileTree_Add( void *aInstance, void* aPath );

        friend class UIFileTreeNode;
    };
} // namespace SE::Core