#include "FileTree.h"

#include <filesystem>

#include "DotNet/Runtime.h"

#include "imgui.h"
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui_internal.h"

namespace SE::Core
{
    Ref<UIImage> mOpenFolder  = nullptr;
    Ref<UIImage> mCloseFolder = nullptr;
    Ref<UIImage> mDefaultFile = nullptr;

    UIFileTreeNode::UIFileTreeNode( UIFileTree *aTreeView, UIFileTreeNode *aParent, fs::path const &aPath, std::string const &aName )
        : UITreeViewNode( aTreeView, aParent )
        , mPath{ aPath }
        , mName{ aName }
    {
        SetText( aName );

        if( !aName.substr( 0, 1 ).compare( "." ) || !aName.substr( 0, 2 ).compare( "__" ) )
            SetTextColor( math::vec4{ 0.15, 0.15, 0.15, 1.0 } );

        if( IsLeaf() ) SetIcon( mDefaultFile.get() );
    }

    bool UIFileTreeNode::IsLeaf()
    {
        if( mParent == nullptr ) return false;

        return fs::is_regular_file( mPath / mName );
    }

    std::vector<UITreeViewNode *> const &UIFileTreeNode::Children()
    {
        auto const &lFullPath = mPath / mName;
        if( ( mChildren.size() == 0 ) && fs::is_directory( lFullPath ) )
        {
            fs::directory_iterator lIterator;

            lIterator = fs::directory_iterator( lFullPath );
            std::vector<fs::path> lFolders;
            std::copy_if( fs::begin( lIterator ), fs::end( lIterator ), std::back_inserter( lFolders ),
                          [&]( auto const &aPath ) { return fs::is_directory( aPath ); } );
            std::sort( lFolders.begin(), lFolders.end(),
                       [&]( auto const &s1, auto const &s2 )
                       { return s1.filename().string().compare( s2.filename().string() ) <= 0; } );

            lIterator = fs::directory_iterator( lFullPath );
            std::vector<fs::path> lFiles;
            std::copy_if( fs::begin( lIterator ), fs::end( lIterator ), std::back_inserter( lFiles ),
                          [&]( auto const &aPath ) { return fs::is_regular_file( aPath ); } );
            std::sort( lFiles.begin(), lFiles.end(),
                       [&]( auto const &s1, auto const &s2 )
                       { return s1.filename().string().compare( s2.filename().string() ) <= 0; } );

            for( auto const &F : lFolders ) Add( F );

            for( auto const &F : lFiles ) Add( F );
        }

        return UITreeViewNode::Children();
    }

    UIFileTreeNode *UIFileTreeNode::Add( fs::path const &aPath )
    {
        auto lNewChild = new UIFileTreeNode( (UIFileTree *)mTreeView, this, aPath.parent_path(), aPath.filename().string() );
        mChildren.push_back( lNewChild );

        return lNewChild;
    }

    UIFileTree::UIFileTree()
    {
        SetIndent( 9.0f );

        mRoot = new UIFileTreeNode( this, nullptr, "", "" );

        if( mDefaultFile == nullptr )
            mDefaultFile = New<UIImage>( "C:\\GitLab\\SpockEngine\\Saved\\Resources\\Icons\\File.png", math::vec2{ 20, 20 } );
    }

    UIFileTreeNode *UIFileTree::Add( fs::path const &aPath ) { return ( (UIFileTreeNode *)mRoot )->Add( aPath ); }

    // void *UIFileTree::UIFileTree_Create()
    // {
    //     auto lNewLabel = new UIFileTree();

    //     return static_cast<void *>( lNewLabel );
    // }

    // void UIFileTree::UIFileTree_Destroy( void *aInstance ) { delete static_cast<UIFileTree *>( aInstance ); }

    // void *UIFileTree::UIFileTree_Add( void *aInstance, void *aPath )
    // {
    //     auto lInstance = static_cast<UIFileTree *>( aInstance );
    //     auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aPath ) );

    //     return static_cast<void *>( lInstance->Add( lString ) );
    // }
} // namespace SE::Core