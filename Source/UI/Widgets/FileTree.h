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

        UIFileTreeNode *Add( path_t const &aPath );
        void            Remove( path_t const &aPath );

        std::vector<UITreeViewNode *> const &Children();
        path_t const                        &GetPath()
        {
            return mFullPath;
        }

      protected:
        path_t   mPath;
        string_t mName;
        path_t   mFullPath;

        bool IsLeaf();
    };

    class UIFileTree : public UITreeView
    {
      public:
        UIFileTree();

        UIFileTreeNode *Add( path_t const &aPath );
        void            Remove( path_t const &aPath );

        void OnSelected( std::function<void( path_t const &aPath )> aOnSelected );

      protected:
        std::function<void( path_t const &aPath )> mOnSelected;

        void HandleOnSelected( UIFileTreeNode *a );

        friend class UIFileTreeNode;
    };
} // namespace SE::Core