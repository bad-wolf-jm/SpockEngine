#include "FileTree.h"

#include <filesystem>

#include "imgui.h"
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui_internal.h"

namespace SE::Core
{
    ref_t<UIImage> mOpenFolder  = nullptr;
    ref_t<UIImage> mCloseFolder = nullptr;
    ref_t<UIImage> mDefaultFile = nullptr;

    UIFileTreeNode::UIFileTreeNode( UIFileTree *aTreeView, UIFileTreeNode *aParent, path_t const &aPath, string_t const &aName )
        : UITreeViewNode( aTreeView, aParent )
        , mPath{ aPath }
        , mName{ aName }
    {
        SetText( aName );

        mFullPath = mPath / mName;

        if( !aName.substr( 0, 1 ).compare( "." ) || !aName.substr( 0, 2 ).compare( "__" ) )
            SetTextColor( math::vec4{ 0.15, 0.15, 0.15, 1.0 } );

        if( IsLeaf() )
            SetIcon( mDefaultFile.get() );
    }

    bool UIFileTreeNode::IsLeaf()
    {
        return fs::is_regular_file( mPath / mName );
    }

    vector_t<UITreeViewNode *> const &UIFileTreeNode::Children()
    {
        auto const &lFullPath = mPath / mName;
        if( ( mChildren.size() == 0 ) && fs::is_directory( lFullPath ) )
        {
            fs::directory_iterator lIterator;

            lIterator = fs::directory_iterator( lFullPath );
            vector_t<path_t> lFolders;
            std::copy_if( fs::begin( lIterator ), fs::end( lIterator ), std::back_inserter( lFolders ),
                          [&]( auto const &aPath ) { return fs::is_directory( aPath ); } );
            std::sort( lFolders.begin(), lFolders.end(),
                       [&]( auto const &s1, auto const &s2 )
                       { return s1.filename().string().compare( s2.filename().string() ) <= 0; } );

            lIterator = fs::directory_iterator( lFullPath );
            vector_t<path_t> lFiles;
            std::copy_if( fs::begin( lIterator ), fs::end( lIterator ), std::back_inserter( lFiles ),
                          [&]( auto const &aPath ) { return fs::is_regular_file( aPath ); } );
            std::sort( lFiles.begin(), lFiles.end(),
                       [&]( auto const &s1, auto const &s2 )
                       { return s1.filename().string().compare( s2.filename().string() ) <= 0; } );

            for( auto const &F : lFolders )
                Add( F );

            for( auto const &F : lFiles )
                Add( F );
        }

        return UITreeViewNode::Children();
    }

    UIFileTreeNode *UIFileTreeNode::Add( path_t const &aPath )
    {
        auto lNewChild = new UIFileTreeNode( (UIFileTree *)mTreeView, this, aPath.parent_path(), aPath.filename().string() );
        lNewChild->OnSelected( [&]( UITreeViewNode *a ) { ( (UIFileTree *)mTreeView )->HandleOnSelected( (UIFileTreeNode *)a ); } );
        mChildren.push_back( lNewChild );

        return lNewChild;
    }

    void UIFileTreeNode::Remove( path_t const &aPath )
    {
        UIFileTreeNode *lChild = nullptr;
        for( auto const &lX : mChildren )
        {
            if( ( (UIFileTreeNode *)lX )->mPath == aPath )
            {
                lChild = (UIFileTreeNode *)lX;
                break;
            }
        }

        vector_t<UITreeViewNode *> lRemainingChildren;
        std::copy_if( mChildren.begin(), mChildren.end(), std::back_inserter( lRemainingChildren ),
                      [&]( auto *x ) { return x == lChild; } );

        mChildren = std::move( lRemainingChildren );
    }

    UIFileTree::UIFileTree()
    {
        SetIndent( 9.0f );

        mRoot = new UIFileTreeNode( this, nullptr, "", "" );

        if( mDefaultFile == nullptr )
            mDefaultFile = New<UIImage>( "D:\\Work\\Git\\SpockEngine\\Saved\\Resources\\Icons\\File.png", math::vec2{ 20, 20 } );
    }

    UIFileTreeNode *UIFileTree::Add( path_t const &aPath )
    {
        UIFileTreeNode *lNewNode = ( (UIFileTreeNode *)mRoot )->Add( aPath );

        return lNewNode;
    }
    
    void UIFileTree::Remove( path_t const &aPath )
    {
        ( (UIFileTreeNode *)mRoot )->Remove( aPath );
    }

    void UIFileTree::OnSelected( std::function<void( path_t const &aPath )> aOnSelected )
    {
        mOnSelected = aOnSelected;
    }

    void UIFileTree::HandleOnSelected( UIFileTreeNode *a )
    {
        if( mOnSelected )
            mOnSelected( a->GetPath() );
    }

} // namespace SE::Core