#pragma once
// #define SOL_ALL_SAFETIES_ON 1
// #include <sol/sol.hpp>

#include "Core/EntityRegistry/Registry.h"

namespace LTSE::Core
{
    using namespace entt::literals;
    namespace
    {

        auto GetMetaType( MonoType *aObject ) { return entt::resolve( static_cast<entt::hash_type>( aObject ) ); }

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
            auto        lStructName   = GetTypeName();
            std::string lMonoTypeName = fmt::format( "{}.{}", aNamespace, lStructName );

            return mono_reflection_type_from_name( lMonoTypeName.data(), ScriptEngine::GetCoreAssemblyImage() );
        }

        template <typename _Ty>
        auto IsValid( Entity &aEntity )
        {
            return aEntity.IsValid();
        }

        template <typename _Ty>
        auto Add( Entity &aEntity, const sol::table &aInstance )
        {
            auto &lNewComponent = aEntity.Add<_Ty>( aInstance.valid() ? aInstance.as<_Ty>() : _Ty{} );

            return sol::make_reference( aScriptState, std::ref( lNewComponent ) );
        }

        template <typename _Ty>
        auto AddOrReplace( Entity &aEntity, const sol::table &aInstance )
        {
            auto &lNewComponent = aEntity.AddOrReplace<_Ty>( aInstance.valid() ? aInstance.as<_Ty>() : _Ty{} );

            return sol::make_reference( aScriptState, std::ref( lNewComponent ) );
        }

        template <typename _Ty>
        auto Replace( Entity &aEntity, const sol::table &aInstance )
        {
            auto &lNewComponent = aEntity.Replace<_Ty>( aInstance.valid() ? aInstance.as<_Ty>() : _Ty{} );

            return sol::make_reference( aScriptState, std::ref( lNewComponent ) );
        }

        template <typename _Ty>
        auto Get( Entity &aEntity )
        {
            auto &lNewComponent = aEntity.Get<_Ty>();

            return sol::make_reference( aScriptState, std::ref( lNewComponent ) );
        }

        template <typename _Ty>
        auto Has( Entity &aEntity )
        {
            return aEntity.Has<_Ty>();
        }

        template <typename _Ty>
        auto Remove( Entity &aEntity )
        {
            aEntity.Remove<_Ty>();
        }
    } // namespace

    void Has( uint32_t aEntityID, EntityRegistry *aRegistry, MonoType *aTagType )
    {
        const auto lMaybeAny = InvokeMetaFunction( GetMetaType( aTagType ), "Has"_hs, aRegistry->WrapEntity( aEntityID ) );

        return static_cast<bool>( lMaybeAny );
    }

    template <typename _ComponentType>
    void Get( uint32_t aEntityID, EntityRegistry *aRegistry, MonoType *aTagType, _ComponentType *aOutput )
    {
        if( !CheckMetaType<_ComponentType>( aTagType ) ) return;

        const auto lMaybeAny = InvokeMetaFunction( GetMetaType( aTagType ), "Get"_hs, aRegistry->WrapEntity( aEntityID ) );

        if( lMaybeAny ) *aOutput = lMaybeAny.cast<_ComponentType>();
    }

    template <typename _ComponentType>
    void Add( uint32_t aEntityID, EntityRegistry *aRegistry, MonoType *aTagType, _ComponentType *aValue )
    {
        //
    }

    template <typename _ComponentType>
    void Replace( uint32_t aEntityID, EntityRegistry *aRegistry, MonoType *aTagType, _ComponentType *aValue )
    {
        //
    }

    template <typename _ComponentType>
    void AddOrReplace( uint32_t aEntityID, EntityRegistry *aRegistry, MonoType *aTagType, _ComponentType *aValue )
    {
        //
    }

    void Remove( uint32_t aEntityID, EntityRegistry *aRegistry, MonoType *aTagType )
    {
        //
    }
}; // namespace LTSE::Core