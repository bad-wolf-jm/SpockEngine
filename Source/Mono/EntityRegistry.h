#pragma once
// #define SOL_ALL_SAFETIES_ON 1
// #include <sol/sol.hpp>

#include "mono/jit/jit.h"
#include "mono/metadata/assembly.h"
#include "mono/metadata/object.h"
#include "mono/metadata/tabledefs.h"

#include "Core/EntityRegistry/Registry.h"
#include "Manager.h"
#include "Scene/Components.h"

namespace LTSE::Core
{
    using namespace entt::literals;
    using namespace LTSE::Core::EntityComponentSystem::Components;
    namespace
    {
        template <typename _Ty>
        bool CheckMetaType( MonoType *aObject )
        {
            return entt::resolve<_Ty>() == GetMetaType( aObject );
        }

        template <typename _Ty>
        std::string_view GetTypeName()
        {
            std::string_view lTypeName          = typeid( _Ty ).name();
            size_t           lLastColonPosition = lTypeName.find_last_of( ':' );

            return lTypeName.substr( lLastColonPosition + 1 );
        }

        template <typename _Ty>
        MonoType *RetrieveMonoTypeFromNamespace( std::string const &aNamespace )
        {
            std::string_view lStructName   = GetTypeName<_Ty>();
            std::string      lMonoTypeName = fmt::format( "{}.{}", aNamespace, lStructName );

            MonoType *lMonoType = mono_reflection_type_from_name( lMonoTypeName.data(), ScriptManager::GetCoreAssemblyImage() );
            if( !lMonoType )
            {
                LTSE::Logging::Info( "Could not find type '{}'", lMonoTypeName );
                return nullptr;
            }

            return lMonoType;
        }

        template <typename _Ty>
        auto IsValid( Entity &aEntity )
        {
            return aEntity.IsValid();
        }

        template <typename _Ty>
        auto Has( Entity &aEntity )
        {
            return aEntity.Has<_Ty>();
        }

        template <typename _Ty>
        ScriptClassInstance Get( Entity &aEntity, ScriptClass &aMonoType )
        {
            auto &aComponent = aEntity.Get<_Ty>();

            return MarshallComponent( aMonoType, aComponent );
        }

        // template <typename _Ty>
        // auto Add( Entity &aEntity, const sol::table &aInstance )
        // {
        //     auto &lNewComponent = aEntity.Add<_Ty>( aInstance.valid() ? aInstance.as<_Ty>() : _Ty{} );

        //     return sol::make_reference( aScriptState, std::ref( lNewComponent ) );
        // }

        // template <typename _Ty>
        // auto AddOrReplace( Entity &aEntity, const sol::table &aInstance )
        // {
        //     auto &lNewComponent = aEntity.AddOrReplace<_Ty>( aInstance.valid() ? aInstance.as<_Ty>() : _Ty{} );

        //     return sol::make_reference( aScriptState, std::ref( lNewComponent ) );
        // }

        // template <typename _Ty>
        // auto Replace( Entity &aEntity, const sol::table &aInstance )
        // {
        //     auto &lNewComponent = aEntity.Replace<_Ty>( aInstance.valid() ? aInstance.as<_Ty>() : _Ty{} );

        //     return sol::make_reference( aScriptState, std::ref( lNewComponent ) );
        // }

        // template <typename _Ty>
        // auto Remove( Entity &aEntity )
        // {
        //     aEntity.Remove<_Ty>();
        // }
    } // namespace

    ScriptClassInstance MarshallComponent( ScriptClass &aMonoType, sNodeTransformComponent &aComponent );
    entt::meta_type     GetMetaType( MonoType *aObject );
    // {
    //     auto lHashValue = std::hash<uint64_t>()( (uint64_t)aObject );

    //     return entt::resolve( (uint32_t)( lHashValue & 0xFFFFFFFF ) );
    // }

    // void IsValid( uint32_t aEntityID, EntityRegistry *aRegistry )
    // {
    //     const auto lMaybeAny = InvokeMetaFunction( GetMetaType( aTagType ), "IsValid"_hs, aRegistry->WrapEntity( aEntityID ) );

    //     return static_cast<bool>( lMaybeAny );
    // }

    // void Has( uint32_t aEntityID, EntityRegistry *aRegistry, MonoType *aTagType )
    // {
    //     const auto lMaybeAny = InvokeMetaFunction( GetMetaType( aTagType ), "Has"_hs, aRegistry->WrapEntity( aEntityID ) );

    //     return static_cast<bool>( lMaybeAny );
    // }

    // template <typename _ComponentType>
    // void Get( uint32_t aEntityID, EntityRegistry *aRegistry, MonoType *aTagType, _ComponentType *aOutput )
    // {
    //     if( !CheckMetaType<_ComponentType>( aTagType ) ) return;

    //     const auto lMaybeAny = InvokeMetaFunction( GetMetaType( aTagType ), "Get"_hs, aRegistry->WrapEntity( aEntityID ) );

    //     if( lMaybeAny ) *aOutput = lMaybeAny.cast<_ComponentType>();
    // }

    // template <typename _ComponentType>
    // void Add( uint32_t aEntityID, EntityRegistry *aRegistry, MonoType *aTagType, _ComponentType *aValue )
    // {
    //     //
    // }

    // template <typename _ComponentType>
    // void Replace( uint32_t aEntityID, EntityRegistry *aRegistry, MonoType *aTagType, _ComponentType *aValue )
    // {
    //     //
    // }

    // template <typename _ComponentType>
    // void AddOrReplace( uint32_t aEntityID, EntityRegistry *aRegistry, MonoType *aTagType, _ComponentType *aValue )
    // {
    //     //
    // }

    // void Remove( uint32_t aEntityID, EntityRegistry *aRegistry, MonoType *aTagType )
    // {
    //     //
    // }

    template <typename _Ty>
    void RegisterComponentType()
    {
        using namespace entt::literals;

        auto lMonoType = RetrieveMonoTypeFromNamespace<_Ty>( "SpockEngine" );

        auto lHashValue = std::hash<uint64_t>{}( (uint64_t)lMonoType );
        auto lNewType   = entt::meta<_Ty>().type( lHashValue & 0xFFFFFFFF );

        if constexpr( std::is_class<_Ty>::value )
        {
            lNewType.template func<&Has<_Ty>>( "Has"_hs );
            lNewType.template func<&Get<_Ty>>( "Get"_hs );
        }
    }

}; // namespace LTSE::Core