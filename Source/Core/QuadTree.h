#pragma once

#include "Box.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <memory>
#include <type_traits>
#include <vector>

#include "Vector2.h"

namespace LTSE::Core
{

    class Box
    {
      public:
        float left;
        float top;
        float width;  // Must be positive
        float height; // Must be positive

        constexpr Box( T Left = 0, T Top = 0, T Width = 0, T Height = 0 ) noexcept
            : left( Left )
            , top( Top )
            , width( Width )
            , height( Height )
        {
        }

        constexpr Box( const math::vec2 &position, const math::vec2 &size ) noexcept
            : left( position.x )
            , top( position.y )
            , width( size.x )
            , height( size.y )
        {
        }

        constexpr T getRight() const noexcept { return left + width; }

        constexpr T getBottom() const noexcept { return top + height; }

        constexpr math::vec2 getTopLeft() const noexcept { return math::vec2( left, top ); }

        constexpr math::vec2 getCenter() const noexcept { return math::vec2( left + width / 2, top + height / 2 ); }

        constexpr math::vec2 getSize() const noexcept { return math::vec2( width, height ); }

        constexpr bool contains( const Box<T> &box ) const noexcept { return left <= box.left && box.getRight() <= getRight() && top <= box.top && box.getBottom() <= getBottom(); }

        constexpr bool intersects( const Box<T> &box ) const noexcept
        {
            return !( left >= box.getRight() || getRight() <= box.left || top >= box.getBottom() || getBottom() <= box.top );
        }
    };

    enum class Quadrant : uint8_t
    {
        NORTH_WEST = 0,
        NORTH_EAST = 1,
        SOUTH_WEST = 2,
        SOUTH_EAST = 3,
        NONE       = 5
    };

    template <typename T, typename GetBox, typename Equal = std::equal_to<T>> class Quadtree
    {
        static_assert( std::is_convertible_v<std::invoke_result_t<GetBox, const T &>, Box>, "GetBox must be a callable of signature Box(const T&)" );
        static_assert( std::is_convertible_v<std::invoke_result_t<Equal, const T &, const T &>, bool>, "Equal must be a callable of signature bool(const T&, const T&)" );
        static_assert( std::is_arithmetic_v<float> );

      public:
        Quadtree( const Box &box, const GetBox &getBox = GetBox(), const Equal &equal = Equal() )
            : mBox( box )
            , mRoot( std::make_unique<Node>() )
            , mGetBox( getBox )
            , mEqual( equal )
        {
        }

        void add( const T &value ) { add( mRoot.get(), 0, mBox, value ); }

        void remove( const T &value ) { remove( mRoot.get(), mBox, value ); }

        std::vector<T> query( const Box &box ) const
        {
            auto values = std::vector<T>();
            query( mRoot.get(), mBox, box, values );
            return values;
        }

        std::vector<std::pair<T, T>> findAllIntersections() const
        {
            auto intersections = std::vector<std::pair<T, T>>();
            findAllIntersections( mRoot.get(), intersections );
            return intersections;
        }

      private:
        static constexpr auto Threshold = std::size_t( 16 );
        static constexpr auto MaxDepth  = std::size_t( 8 );

        struct Node
        {
            std::array<std::unique_ptr<Node>, 4> children;
            std::vector<T> values;
        };

        Box mBox;
        std::unique_ptr<Node> mRoot;
        GetBox mGetBox;
        Equal mEqual;

        bool isLeaf( const Node *node ) const { return !static_cast<bool>( node->children[0] ); }

        Box computeBox( const Box &box, Quadrant i ) const
        {
            auto origin    = box.getTopLeft();
            auto childSize = box.getSize() / static_cast<float>( 2 );
            switch( i )
            {
            case Quadrant::NORTH_WEST:
                return Box( origin, childSize );
            case Quadrant::NORTH_EAST:
                return Box( math::vec2( origin.x + childSize.x, origin.y ), childSize );
            case Quadrant::SOUTH_WEST:
                return Box( math::vec2( origin.x, origin.y + childSize.y ), childSize );
            case Quadrant::SOUTH_EAST:
                return Box( origin + childSize, childSize );
            default:
                assert( false && "Invalid child index" );
                return Box();
            }
        }

        int getQuadrant( const Box &nodeBox, const Box &valueBox ) const
        {
            auto center = nodeBox.getCenter();
            // West
            if( valueBox.getRight() < center.x )
            {
                // North West
                if( valueBox.getBottom() < center.y )
                    return Quadrant::NORTH_WEST;
                // South West
                else if( valueBox.top >= center.y )
                    return Quadrant::SOUTH_WEST;
                // Not contained in any quadrant
                else
                    return -1;
            }
            // East
            else if( valueBox.left >= center.x )
            {
                // North East
                if( valueBox.getBottom() < center.y )
                    return Quadrant::NORTH_EAST;
                // South East
                else if( valueBox.top >= center.y )
                    return Quadrant::SOUTH_EAST;
                // Not contained in any quadrant
                else
                    return Quadrant::NONE;
            }
            // Not contained in any quadrant
            else
                return -1;
        }

        void add( Node *node, std::size_t depth, const Box &box, const T &value )
        {
            assert( node != nullptr );
            assert( box.contains( mGetBox( value ) ) );
            if( isLeaf( node ) )
            {
                // Insert the value in this node if possible
                if( depth >= MaxDepth || node->values.size() < Threshold )
                    node->values.push_back( value );
                // Otherwise, we split and we try again
                else
                {
                    split( node, box );
                    add( node, depth, box, value );
                }
            }
            else
            {
                auto i = getQuadrant( box, mGetBox( value ) );
                // Add the value in a child if the value is entirely contained in it
                if( i != -1 )
                    add( node->children[static_cast<std::size_t>( i )].get(), depth + 1, computeBox( box, i ), value );
                // Otherwise, we add the value in the current node
                else
                    node->values.push_back( value );
            }
        }

        void split( Node *node, const Box &box )
        {
            assert( node != nullptr );
            assert( isLeaf( node ) && "Only leaves can be split" );
            // Create children
            for( auto &child : node->children )
                child = std::make_unique<Node>();
            // Assign values to children
            auto newValues = std::vector<T>(); // New values for this node
            for( const auto &value : node->values )
            {
                auto i = getQuadrant( box, mGetBox( value ) );
                if( i != -1 )
                    node->children[static_cast<std::size_t>( i )]->values.push_back( value );
                else
                    newValues.push_back( value );
            }
            node->values = std::move( newValues );
        }

        bool remove( Node *node, const Box &box, const T &value )
        {
            assert( node != nullptr );
            assert( box.contains( mGetBox( value ) ) );
            if( isLeaf( node ) )
            {
                // Remove the value from node
                removeValue( node, value );
                return true;
            }
            else
            {
                // Remove the value in a child if the value is entirely contained in it
                auto i = getQuadrant( box, mGetBox( value ) );
                if( i != -1 )
                {
                    if( remove( node->children[static_cast<std::size_t>( i )].get(), computeBox( box, i ), value ) )
                        return tryMerge( node );
                }
                // Otherwise, we remove the value from the current node
                else
                    removeValue( node, value );
                return false;
            }
        }

        void removeValue( Node *node, const T &value )
        {
            // Find the value in node->values
            auto it = std::find_if( std::begin( node->values ), std::end( node->values ), [this, &value]( const auto &rhs ) { return mEqual( value, rhs ); } );
            assert( it != std::end( node->values ) && "Trying to remove a value that is not present in the node" );
            // Swap with the last element and pop back
            *it = std::move( node->values.back() );
            node->values.pop_back();
        }

        bool tryMerge( Node *node )
        {
            assert( node != nullptr );
            assert( !isLeaf( node ) && "Only interior nodes can be merged" );
            auto nbValues = node->values.size();
            for( const auto &child : node->children )
            {
                if( !isLeaf( child.get() ) )
                    return false;
                nbValues += child->values.size();
            }
            if( nbValues <= Threshold )
            {
                node->values.reserve( nbValues );
                // Merge the values of all the children
                for( const auto &child : node->children )
                {
                    for( const auto &value : child->values )
                        node->values.push_back( value );
                }
                // Remove the children
                for( auto &child : node->children )
                    child.reset();
                return true;
            }
            else
                return false;
        }

        void query( Node *node, const Box &box, const Box &queryBox, std::vector<T> &values ) const
        {
            assert( node != nullptr );
            assert( queryBox.intersects( box ) );
            for( const auto &value : node->values )
            {
                if( queryBox.intersects( mGetBox( value ) ) )
                    values.push_back( value );
            }
            if( !isLeaf( node ) )
            {
                for( auto i = std::size_t( 0 ); i < node->children.size(); ++i )
                {
                    auto childBox = computeBox( box, static_cast<int>( i ) );
                    if( queryBox.intersects( childBox ) )
                        query( node->children[i].get(), childBox, queryBox, values );
                }
            }
        }

        void findAllIntersections( Node *node, std::vector<std::pair<T, T>> &intersections ) const
        {
            // Find intersections between values stored in this node
            // Make sure to not report the same intersection twice
            for( auto i = std::size_t( 0 ); i < node->values.size(); ++i )
            {
                for( auto j = std::size_t( 0 ); j < i; ++j )
                {
                    if( mGetBox( node->values[i] ).intersects( mGetBox( node->values[j] ) ) )
                        intersections.emplace_back( node->values[i], node->values[j] );
                }
            }
            if( !isLeaf( node ) )
            {
                // Values in this node can intersect values in descendants
                for( const auto &child : node->children )
                {
                    for( const auto &value : node->values )
                        findIntersectionsInDescendants( child.get(), value, intersections );
                }
                // Find intersections in children
                for( const auto &child : node->children )
                    findAllIntersections( child.get(), intersections );
            }
        }

        void findIntersectionsInDescendants( Node *node, const T &value, std::vector<std::pair<T, T>> &intersections ) const
        {
            // Test against the values stored in this node
            for( const auto &other : node->values )
            {
                if( mGetBox( value ).intersects( mGetBox( other ) ) )
                    intersections.emplace_back( value, other );
            }
            // Test against values stored into descendants of this node
            if( !isLeaf( node ) )
            {
                for( const auto &child : node->children )
                    findIntersectionsInDescendants( child.get(), value, intersections );
            }
        }
    };

} // namespace quadtree