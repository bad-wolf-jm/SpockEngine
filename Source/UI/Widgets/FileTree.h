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

        UIFileTreeNode *Add(fs::path const& aPath);

        std::vector<UITreeViewNode *> const &Children();

      protected:
        fs::path    mPath;
        std::string mName;

        bool IsLeaf();
    };

    class UIFileTree : public UITreeView
    {
      public:
        UIFileTree();

        UIFileTreeNode *Add(fs::path const& aPath);

      public:
        static void *UIFileTree_Create();
        static void  UIFileTree_Destroy( void *aInstance );
        static void *UIFileTree_Add( void *aInstance, void* aPath );

        friend class UIFileTreeNode;
    };
} // namespace SE::Core