#include "InternalCalls.h"
#include "EntityRegistry.h"
#include "TypeReflection.h"
#include <iostream>
#include <string>

#include "Core/Logging.h"

#include "MonoScriptEngine.h"

namespace SE::MonoInternalCalls
{
    uint32_t Entity_Create( EntityRegistry *aRegistry, MonoString *aName, uint32_t aEntityID )
    {
        auto lName      = std::string( mono_string_to_utf8( aName ) );
        auto lNewEntity = aRegistry->CreateEntity( aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ), lName );

        return static_cast<uint32_t>( lNewEntity );
    }

    bool Entity_IsValid( uint32_t aEntityID, EntityRegistry *aRegistry )
    {
        return aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ).IsValid();
    }

    bool Entity_Has( uint32_t aEntityID, EntityRegistry *aRegistry, MonoReflectionType *aComponentType )
    {
        MonoType *lMonoType = mono_reflection_type_get_type( aComponentType );

        const entt::meta_type lMetaType = Core::GetMetaType( lMonoType );
        const entt::meta_any  lMaybeAny =
            Core::InvokeMetaFunction( lMetaType, "Has"_hs, aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ) );

        return lMaybeAny.cast<bool>();
    }

    MonoObject *Entity_Get( uint32_t aEntityID, EntityRegistry *aRegistry, MonoReflectionType *aComponentType )
    {
        MonoType *lMonoType = mono_reflection_type_get_type( aComponentType );

        const entt::meta_type lMetaType = Core::GetMetaType( lMonoType );
        const entt::meta_any  lMaybeAny = Core::InvokeMetaFunction(
            lMetaType, "Get"_hs, aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ), MonoScriptClass( lMonoType ) );

        return lMaybeAny.cast<MonoScriptInstance>().GetInstance();
    }

    void Entity_Replace( uint32_t aEntityID, EntityRegistry *aRegistry, MonoReflectionType *aComponentType, MonoObject *aNewComponent )
    {
        MonoType  *lMonoType  = mono_reflection_type_get_type( aComponentType );
        MonoClass *lMonoClass = mono_class_from_mono_type( lMonoType );

        const entt::meta_type lMetaType = Core::GetMetaType( lMonoType );
        Core::InvokeMetaFunction( lMetaType, "Replace"_hs, aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ),
                                  MonoScriptInstance( lMonoClass, aNewComponent ) );
    }

    void Entity_Add( uint32_t aEntityID, EntityRegistry *aRegistry, MonoReflectionType *aComponentType, MonoObject *aNewComponent )
    {
        MonoType  *lMonoType  = mono_reflection_type_get_type( aComponentType );
        MonoClass *lMonoClass = mono_class_from_mono_type( lMonoType );

        const entt::meta_type lMetaType = Core::GetMetaType( lMonoType );
        Core::InvokeMetaFunction( lMetaType, "Add"_hs, aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ),
                                  MonoScriptInstance( lMonoClass, aNewComponent ) );
    }

    void Entity_Remove( uint32_t aEntityID, EntityRegistry *aRegistry, MonoReflectionType *aComponentType )
    {
        MonoType *lMonoType = mono_reflection_type_get_type( aComponentType );

        const entt::meta_type lMetaType = Core::GetMetaType( lMonoType );
        const entt::meta_any  lMaybeAny =
            Core::InvokeMetaFunction( lMetaType, "Remove"_hs, aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ) );
    }

    size_t OpNode_NewTensorShape( MonoArray *aShape, uint32_t aRank, uint32_t aLayers, uint32_t aElementSize )
    {
        std::vector<std::vector<uint32_t>> lShape{};

        for( uint32_t i = 0; i < aLayers; i++ )
        {
            lShape.push_back( {} );
            for( uint32_t j = 0; j < aRank; j++ )
            {
                uint32_t lDim = *( mono_array_addr( aShape, uint32_t, ( i * aRank + j ) ) );

                lShape.back().push_back( lDim );
            }
        }

        return (size_t)( new Cuda::sTensorShape( lShape, aElementSize ) );
    }

    void OpNode_DestroyTensorShape( Cuda::sTensorShape *aTensorShape ) { delete aTensorShape; }

    uint32_t OpNode_CountLayers( Cuda::sTensorShape *aTensorShape ) { return aTensorShape->CountLayers(); }

    MonoArray *OpNode_GetDimension( Cuda::sTensorShape *aTensorShape, int i )
    {
        auto lDim = aTensorShape->GetDimension( i );

        MonoArray *lNewArray = mono_array_new( mono_domain_get(), mono_get_uint32_class(), lDim.size() );

        for( uint32_t i = 0; i < lDim.size(); i++ ) mono_array_set( lNewArray, uint32_t, i, lDim[i] );

        return lNewArray;
    }

    void OpNode_Trim( Cuda::sTensorShape *aTensorShape, int aToDimension ) { aTensorShape->Trim( aToDimension ); }

    void OpNode_Flatten( Cuda::sTensorShape *aTensorShape, int aToDimension ) { aTensorShape->Flatten( aToDimension ); }

    void OpNode_InsertDimension( Cuda::sTensorShape *aTensorShape, int aPosition, MonoArray *aDimension )
    {
        uint32_t lArrayLength = static_cast<uint32_t>( mono_array_length( aDimension ) );

        std::vector<uint32_t> lNewDimension{};

        for( uint32_t i = 0; i < lArrayLength; i++ )
        {
            lNewDimension.push_back( *( mono_array_addr( aDimension, uint32_t, i ) ) );
        }

        aTensorShape->InsertDimension( aPosition, lNewDimension );
    }

    size_t OpNode_NewScope( uint32_t aMemorySize ) { return (size_t)( new Scope( aMemorySize ) ); }

    void OpNode_DestroyScope( Scope *aScope ) { delete aScope; }

} // namespace SE::MonoInternalCalls