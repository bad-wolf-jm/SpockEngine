#pragma once

#include "mono/jit/jit.h"
#include "mono/metadata/assembly.h"
#include "mono/metadata/object.h"
#include "mono/metadata/tabledefs.h"

#include "Core/Logging.h"
#include "Core/EntityCollection/Collection.h"
#include "Scene/Components.h"

#include "MonoRuntime.h"

namespace SE::Core
{
    using namespace entt::literals;
    using namespace SE::Core::EntityComponentSystem::Components;
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

            return MonoRuntime::GetCoreTypeFromName( lMonoTypeName );
        }

        template <typename _Ty>
        bool IsValid( Entity &aEntity )
        {
            return aEntity.IsValid();
        }

        template <typename _Ty>
        bool Has( Entity &aEntity )
        {
            return aEntity.Has<_Ty>();
        }

        template <typename _Ty>
        MonoScriptInstance Get( Entity &aEntity, MonoScriptClass &aMonoType )
        {
            auto &aComponent = aEntity.Get<_Ty>();
            auto  lInstance  = MarshallComponent( aMonoType, aComponent );

            return lInstance;
        }

        template <typename _Ty>
        void Replace( Entity &aEntity, MonoScriptInstance &aNewComponent )
        {
            _Ty lInstance;
            UnmarshallComponent( aNewComponent, lInstance );

            aEntity.Replace<_Ty>( lInstance );
        }

        template <typename _Ty>
        void Add( Entity &aEntity, MonoScriptInstance &aNewComponent )
        {
            _Ty lInstance;
            UnmarshallComponent( aNewComponent, lInstance );

            aEntity.Add<_Ty>( lInstance );
        }

        template <typename _Ty>
        auto Remove( Entity &aEntity )
        {
            aEntity.Remove<_Ty>();
        }
    } // namespace

    MonoScriptInstance MarshallComponent( MonoScriptClass &aMonoType, sTag &aComponent );
    void               UnmarshallComponent( MonoScriptInstance &aMonoType, sTag &aComponent );

    MonoScriptInstance MarshallComponent( MonoScriptClass &aMonoType, sNodeTransformComponent &aComponent );
    void               UnmarshallComponent( MonoScriptInstance &aMonoType, sNodeTransformComponent &aComponent );

    MonoScriptInstance MarshallComponent( MonoScriptClass &aMonoType, sLightComponent &aComponent );
    void               UnmarshallComponent( MonoScriptInstance &aMonoType, sLightComponent &aComponent );

    entt::meta_type GetMetaType( MonoType *aObject );

    template <typename _Ty>
    void RegisterComponentType()
    {
        static_assert( std::is_class<_Ty>::value );

        using namespace entt::literals;

        auto lMonoType  = RetrieveMonoTypeFromNamespace<_Ty>( "SpockEngine" );
        auto lHashValue = std::hash<uint64_t>{}( (uint64_t)lMonoType );
        auto lNewType   = entt::meta<_Ty>().type( lHashValue & 0xFFFFFFFF );

        lNewType.template func<&Has<_Ty>>( "Has"_hs );
        lNewType.template func<&Get<_Ty>>( "Get"_hs );
        lNewType.template func<&Replace<_Ty>>( "Replace"_hs );
        lNewType.template func<&Add<_Ty>>( "Add"_hs );
        lNewType.template func<&Remove<_Ty>>( "Remove"_hs );
    }

}; // namespace SE::Core