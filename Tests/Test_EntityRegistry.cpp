#include <catch2/catch_test_macros.hpp>

#define ENTT_DISABLE_ASSERT
#include "Core/EntityRegistry/Registry.h"

using namespace SE::Core;

struct ComponentA
{
    float a = 0.0f;

    ComponentA()                     = default;
    ComponentA( const ComponentA & ) = default;
};

struct ComponentB
{
    float a = 0.0f;

    ComponentB()                     = default;
    ComponentB( const ComponentB & ) = default;
};

struct ComponentC
{
    float a = 0.0f;

    ComponentC()                     = default;
    ComponentC( const ComponentC & ) = default;
};

struct ComponentD
{
    float a = 0.0f;

    ComponentD()                     = default;
    ComponentD( const ComponentD & ) = default;
};

TEST_CASE( "Create registry", "[CORE_ENTITIES]" )
{
    EntityRegistry lRegistry{};

    REQUIRE( true );
}

TEST_CASE( "Create entities without a name", "[CORE_ENTITIES]" )
{
    EntityRegistry lRegistry{};

    auto lEntity0 = lRegistry.CreateEntity();
    auto lEntity1 = lRegistry.CreateEntity();
    auto lEntity2 = lRegistry.CreateEntity();

    REQUIRE( lEntity0.IsValid() );
    REQUIRE( lEntity1.IsValid() );
    REQUIRE( lEntity2.IsValid() );

    REQUIRE( lEntity0.GetRegistry() == &lRegistry );
    REQUIRE( lEntity1.GetRegistry() == &lRegistry );
    REQUIRE( lEntity2.GetRegistry() == &lRegistry );

    REQUIRE( static_cast<uint32_t>( lEntity0 ) != static_cast<uint32_t>( lEntity1 ) );
    REQUIRE( static_cast<uint32_t>( lEntity0 ) != static_cast<uint32_t>( lEntity2 ) );
    REQUIRE( static_cast<uint32_t>( lEntity1 ) != static_cast<uint32_t>( lEntity2 ) );
}

TEST_CASE( "Create entities with a name", "[CORE_ENTITIES]" )
{
    EntityRegistry lRegistry{};

    auto lEntity0 = lRegistry.CreateEntity( "NAME" );

    REQUIRE( lEntity0.Has<sTag>() );
    REQUIRE( lEntity0.Get<sTag>().mValue == "NAME" );
}

TEST_CASE( "Add components 1", "[CORE_ENTITIES]" )
{
    EntityRegistry lRegistry{};

    auto lEntity0 = lRegistry.CreateEntity();
    REQUIRE( !lEntity0.Has<ComponentA>() );
    REQUIRE( !lEntity0.Has<ComponentB>() );

    auto &lComponentA = lEntity0.Add<ComponentA>();
    REQUIRE( lEntity0.Has<ComponentA>() );

    lEntity0.Remove<ComponentA>();
    REQUIRE( !lEntity0.Has<ComponentA>() );

    auto &lComponentB = lEntity0.TryGet<ComponentB>( ComponentB{ 11.0f } );
    REQUIRE( lComponentB.a == 11.0f );
}

TEST_CASE( "Create entities with relationship", "[CORE_ENTITIES]" )
{
    EntityRegistry lRegistry{};

    auto lEntity0 = lRegistry.CreateEntityWithRelationship();
    auto lEntity1 = lRegistry.CreateEntityWithRelationship( "NAME" );

    REQUIRE( lEntity0.Has<sRelationshipComponent>() );

    REQUIRE( lEntity1.Has<sRelationshipComponent>() );
    REQUIRE( lEntity1.Has<sTag>() );
    REQUIRE( lEntity1.Get<sTag>().mValue == "NAME" );
}

TEST_CASE( "Create entities with a parent", "[CORE_ENTITIES]" )
{
    EntityRegistry lRegistry{};

    auto lEntity0 = lRegistry.CreateEntity();
    auto lEntity1 = lRegistry.CreateEntity( lEntity0, "NAME" );

    REQUIRE( lEntity0.Has<sRelationshipComponent>() );
    REQUIRE( lEntity1.Has<sRelationshipComponent>() );
    REQUIRE( lEntity1.Has<sTag>() );
    REQUIRE( lEntity1.Get<sTag>().mValue == "NAME" );

    REQUIRE( lEntity1.Get<sRelationshipComponent>().mParent == lEntity0 );
    REQUIRE( lEntity0.Get<sRelationshipComponent>().mChildren.size() == 1 );
    REQUIRE( lEntity0.Get<sRelationshipComponent>().mChildren[0] == lEntity1 );
}

TEST_CASE( "Destroy entities", "[CORE_ENTITIES]" )
{
    EntityRegistry lRegistry{};

    auto lEntity0 = lRegistry.CreateEntity();
    lRegistry.DestroyEntity( lEntity0 );

    REQUIRE( !lEntity0.IsValid() );
}

TEST_CASE( "Adding components twice throws exception", "[CORE_ENTITIES]" )
{
    EntityRegistry lRegistry{};

    auto  lEntity0    = lRegistry.CreateEntity();
    auto &lComponentA = lEntity0.Add<ComponentA>();
    REQUIRE_THROWS( ( lEntity0.Add<ComponentA>() ) );
}

TEST_CASE( "Removing component that is not there throws exception", "[CORE_ENTITIES]" )
{
    EntityRegistry lRegistry{};

    auto lEntity0 = lRegistry.CreateEntity();
    REQUIRE_THROWS( ( lEntity0.Remove<ComponentA>() ) );
}

TEST_CASE( "TryRemove", "[CORE_ENTITIES]" )
{
    EntityRegistry lRegistry{};

    auto lEntity0 = lRegistry.CreateEntity();

    auto &lComponentA = lEntity0.TryAdd<ComponentA>();
    REQUIRE( lEntity0.Has<ComponentA>() );

    lEntity0.TryRemove<ComponentA>();
    REQUIRE( !lEntity0.Has<ComponentA>() );
    REQUIRE_NOTHROW( ( lEntity0.TryRemove<ComponentA>() ) );
}

TEST_CASE( "OnComponentAdded event", "[CORE_ENTITIES]" )
{
    EntityRegistry lRegistry{};

    bool lComponentAddedCalled = false;
    lRegistry.OnComponentAdded<ComponentA>( [&]( auto lEntity, auto &lComponent ) { lComponentAddedCalled = true; } );

    auto lEntity0 = lRegistry.CreateEntity();

    REQUIRE( !lComponentAddedCalled );

    lEntity0.Add<ComponentA>();

    REQUIRE( lComponentAddedCalled );
}

TEST_CASE( "OnComponentUpdated event", "[CORE_ENTITIES]" )
{
    EntityRegistry lRegistry{};

    bool lComponentUpdatedCalled = false;
    lRegistry.OnComponentUpdated<ComponentA>( [&]( auto lEntity, auto &lComponent ) { lComponentUpdatedCalled = true; } );

    auto lEntity0 = lRegistry.CreateEntity();
    lEntity0.Add<ComponentA>();

    REQUIRE( !lComponentUpdatedCalled );

    lEntity0.Replace<ComponentA>( ComponentA{ 3.0 } );

    REQUIRE( lComponentUpdatedCalled );
}

TEST_CASE( "OnComponentDestroyed event", "[CORE_ENTITIES]" )
{
    EntityRegistry lRegistry{};

    bool lComponentDestroyedCalled = false;
    lRegistry.OnComponentDestroyed<ComponentA>( [&]( auto lEntity, auto &lComponent ) { lComponentDestroyedCalled = true; } );

    auto lEntity0 = lRegistry.CreateEntity();
    lEntity0.Add<ComponentA>();

    REQUIRE( !lComponentDestroyedCalled );

    lEntity0.Remove<ComponentA>();

    REQUIRE( lComponentDestroyedCalled );
}

struct sTagComponent
{
};

// Tag
TEST_CASE( "Tag Entity", "[CORE_ENTITIES]" )
{
    EntityRegistry lRegistry{};

    auto lEntity0 = lRegistry.CreateEntity();
    lEntity0.Add<sTagComponent>();

    REQUIRE( lEntity0.Has<sTagComponent>() );

    lEntity0.Remove<sTagComponent>();

    REQUIRE( !( lEntity0.Has<sTagComponent>() ) );
}

TEST_CASE( "Update components 1", "[CORE_ENTITIES]" )
{
    EntityRegistry lRegistry{};

    auto lEntity0 = lRegistry.CreateEntity();

    lEntity0.Add<ComponentA>();
    REQUIRE( lEntity0.Get<ComponentA>().a == 0.0f );

    auto &lComponentA = lEntity0.Get<ComponentA>();
    lComponentA.a     = 3.0f;
    REQUIRE( lEntity0.Get<ComponentA>().a == 3.0f );

    lEntity0.Replace<ComponentA>( ComponentA{ 5.0 } );
    REQUIRE( lEntity0.Get<ComponentA>().a == 5.0f );
}

TEST_CASE( "Update components 2", "[CORE_ENTITIES]" )
{
    EntityRegistry lRegistry{};

    auto lEntity0 = lRegistry.CreateEntity();

    lEntity0.AddOrReplace<ComponentA>();
    REQUIRE( lEntity0.Has<ComponentA>() );
    REQUIRE( lEntity0.Get<ComponentA>().a == 0.0f );

    lEntity0.AddOrReplace<ComponentA>( ComponentA{ 3.0 } );
    REQUIRE( lEntity0.Get<ComponentA>().a == 3.0f );
}

TEST_CASE( "Iterate Entities 1", "[CORE_ENTITIES]" )
{
    EntityRegistry lRegistry0{};

    auto lEntity0 = lRegistry0.CreateEntity();
    lEntity0.Add<ComponentA>();

    auto lEntity1 = lRegistry0.CreateEntity();
    lEntity1.Add<ComponentB>();

    auto lEntity2 = lRegistry0.CreateEntity();
    lEntity2.Add<ComponentA>();

    auto lEntity3 = lRegistry0.CreateEntity();
    lEntity3.Add<ComponentB>();

    auto lEntity4 = lRegistry0.CreateEntity();
    lEntity4.Add<ComponentA>();

    std::vector<Entity> lEntitiesA = {};
    lRegistry0.ForEach<ComponentA>( [&]( auto a_Entity, auto &a_Component ) { lEntitiesA.push_back( a_Entity ); } );
    REQUIRE( lEntitiesA.size() == 3 );

    std::vector<Entity> lEntitiesB = {};
    lRegistry0.ForEach<ComponentB>( [&]( auto a_Entity, auto &a_Component ) { lEntitiesB.push_back( a_Entity ); } );
    REQUIRE( lEntitiesB.size() == 2 );
}

TEST_CASE( "Sort Entities 1", "[CORE_ENTITIES]" )
{
    EntityRegistry lRegistry0{};

    auto lEntity0 = lRegistry0.CreateEntity();
    lEntity0.Add<ComponentA>( 1.0f );

    auto lEntity1 = lRegistry0.CreateEntity();
    lEntity1.Add<ComponentA>( 4.0f );

    auto lEntity2 = lRegistry0.CreateEntity();
    lEntity2.Add<ComponentA>( 2.0f );

    auto lEntity3 = lRegistry0.CreateEntity();
    lEntity3.Add<ComponentA>( 6.0f );

    auto lEntity4 = lRegistry0.CreateEntity();
    lEntity4.Add<ComponentA>( 0.0f );

    std::vector<float> lEntitiesA    = {};
    std::vector<float> lSortedValues = { 0.0f, 1.0f, 2.0f, 4.0f, 6.0f };
    lRegistry0.ForEach<ComponentA>( [&]( auto a_Entity, auto &a_Component ) { lEntitiesA.push_back( a_Component.a ); } );
    REQUIRE( lEntitiesA != lSortedValues );

    lEntitiesA.clear();
    lRegistry0.Sort<ComponentA>( [&]( const ComponentA &l, const ComponentA &r ) { return l.a < r.a; } );
    lRegistry0.ForEach<ComponentA>( [&]( auto a_Entity, auto &a_Component ) { lEntitiesA.push_back( a_Component.a ); } );
    REQUIRE( lEntitiesA == lSortedValues );
}

TEST_CASE( "Sort Entities 2", "[CORE_ENTITIES]" )
{
    EntityRegistry lRegistry0{};

    auto lEntity0 = lRegistry0.CreateEntity();
    lEntity0.Add<ComponentA>( 1.0f );

    auto lEntity1 = lRegistry0.CreateEntity();
    lEntity1.Add<ComponentA>( 4.0f );

    auto lEntity2 = lRegistry0.CreateEntity();
    lEntity2.Add<ComponentA>( 2.0f );

    auto lEntity3 = lRegistry0.CreateEntity();
    lEntity3.Add<ComponentA>( 6.0f );

    auto lEntity4 = lRegistry0.CreateEntity();
    lEntity4.Add<ComponentA>( 0.0f );

    std::vector<float> lEntitiesA    = {};
    std::vector<float> lSortedValues = { 0.0f, 1.0f, 2.0f, 4.0f, 6.0f };
    lRegistry0.ForEach<ComponentA>( [&]( auto a_Entity, auto &a_Component ) { lEntitiesA.push_back( a_Component.a ); } );
    REQUIRE( lEntitiesA != lSortedValues );

    lEntitiesA.clear();
    lRegistry0.Sort<ComponentA>( [&]( Entity l, Entity r ) { return ( l.Get<ComponentA>().a ) < ( r.Get<ComponentA>().a ); } );
    lRegistry0.ForEach<ComponentA>( [&]( auto a_Entity, auto &a_Component ) { lEntitiesA.push_back( a_Component.a ); } );
    REQUIRE( lEntitiesA == lSortedValues );
}

TEST_CASE( "Test components", "[CORE_ENTITIES]" )
{
    EntityRegistry lRegistry0{};

    auto lEntity0 = lRegistry0.CreateEntity();
    lEntity0.Add<ComponentA>( 1.0f );
    lEntity0.Add<ComponentB>( 1.0f );

    REQUIRE( lEntity0.Has<ComponentA>() );
    REQUIRE( lEntity0.Has<ComponentB>() );
    REQUIRE( !( lEntity0.Has<ComponentC>() ) );

    REQUIRE( lEntity0.HasAll<ComponentA, ComponentB>() );
    REQUIRE( lEntity0.HasAny<ComponentA, ComponentB>() );
    REQUIRE( !( lEntity0.HasAll<ComponentA, ComponentB, ComponentC>() ) );
    REQUIRE( !( lEntity0.HasAny<ComponentC, ComponentD>() ) );
}

TEST_CASE( "Adjoin components", "[CORE_ENTITIES]" )
{
    EntityRegistry lRegistry0{};

    auto lEntity0 = lRegistry0.CreateEntity();
    auto lEntity1 = lRegistry0.CreateEntity();

    lEntity0.Add<ComponentA>( 1.0f );
    lEntity1.Adjoin<ComponentA>( lEntity0 );

    REQUIRE( lEntity1.Has<sJoinComponent<ComponentA>>() );
    REQUIRE( lEntity1.Get<sJoinComponent<ComponentA>>().JoinedComponent().a == 1.0f );
    REQUIRE( lEntity1.Get<sJoinComponent<ComponentA>>().mJoinEntity == lEntity0 );
}

TEST_CASE( "Relationships", "[CORE_ENTITIES]" )
{
    EntityRegistry lRegistry0{};

    auto lEntity0 = lRegistry0.CreateEntity();
    auto lEntity1 = lRegistry0.CreateEntity();
    auto lEntity2 = lRegistry0.CreateEntity();
    auto lEntity3 = lRegistry0.CreateEntity();

    lRegistry0.SetParent( lEntity1, lEntity0 );
    lRegistry0.SetParent( lEntity2, lEntity0 );
    lRegistry0.SetParent( lEntity3, lEntity0 );

    REQUIRE( lEntity0.Get<sRelationshipComponent>().mChildren.size() == 3 );

    REQUIRE( lEntity1.Get<sRelationshipComponent>().mParent == lEntity0 );
    REQUIRE( lEntity2.Get<sRelationshipComponent>().mParent == lEntity0 );
    REQUIRE( lEntity3.Get<sRelationshipComponent>().mParent == lEntity0 );
}

TEST_CASE( "Removing parent removes from siblings", "[CORE_ENTITIES]" )
{
    EntityRegistry lRegistry0{};

    auto lEntity0 = lRegistry0.CreateEntity();
    auto lEntity1 = lRegistry0.CreateEntity();
    auto lEntity2 = lRegistry0.CreateEntity();
    auto lEntity3 = lRegistry0.CreateEntity();

    lRegistry0.SetParent( lEntity1, lEntity0 );
    lRegistry0.SetParent( lEntity2, lEntity0 );
    lRegistry0.SetParent( lEntity3, lEntity0 );

    lRegistry0.SetParent( lEntity3, lEntity2 );
    REQUIRE( lEntity0.Get<sRelationshipComponent>().mChildren.size() == 2 );

    REQUIRE( lEntity1.Get<sRelationshipComponent>().mParent == lEntity0 );
    REQUIRE( lEntity2.Get<sRelationshipComponent>().mParent == lEntity0 );
    REQUIRE( lEntity3.Get<sRelationshipComponent>().mParent == lEntity2 );
}

TEST_CASE( "Ability to set parent to NULL", "[CORE_ENTITIES]" )
{
    EntityRegistry lRegistry0{};

    auto lEntity0 = lRegistry0.CreateEntity();
    auto lEntity1 = lRegistry0.CreateEntity();
    auto lEntity2 = lRegistry0.CreateEntity();
    auto lEntity3 = lRegistry0.CreateEntity();

    lRegistry0.SetParent( lEntity1, lEntity0 );
    lRegistry0.SetParent( lEntity2, lEntity0 );
    lRegistry0.SetParent( lEntity3, lEntity0 );

    lRegistry0.SetParent( lEntity3, Entity{} );
    REQUIRE( lEntity0.Get<sRelationshipComponent>().mChildren.size() == 2 );

    REQUIRE( lEntity1.Get<sRelationshipComponent>().mParent == lEntity0 );
    REQUIRE( lEntity2.Get<sRelationshipComponent>().mParent == lEntity0 );
    REQUIRE( !( lEntity3.Get<sRelationshipComponent>().mParent ) );
}
