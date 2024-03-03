#pragma once
#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

#include <type_traits>

#include "Core/Entity/Collection.h"
#include "Core/CUDA/Texture/TextureData.h"

#include "Core/CUDA/Array/MultiTensor.h"

namespace SE::Core
{
    namespace
    {

        template <typename _Ty> auto RandomVector( size_t aSize, double aMin, double aMax, sol::this_state aScriptState )
        {
            std::random_device dev;
            std::mt19937 rng( dev() );

            _Ty lMin = static_cast<_Ty>( aMin );
            _Ty lMax = static_cast<_Ty>( aMax );

            if constexpr( std::is_floating_point<_Ty>::value )
            {
                std::uniform_real_distribution<_Ty> dist6( lMin, lMax );
                auto gen = [&dist6, &rng]() { return dist6( rng ); };
                vector_t<_Ty> lNewVector( aSize );
                std::generate( lNewVector.begin(), lNewVector.end(), gen );

                return sol::make_reference( aScriptState, std::move( lNewVector ) );
            }
            else
            {
                std::uniform_int_distribution<_Ty> dist6( lMin, lMax );
                auto gen = [&dist6, &rng]() { return dist6( rng ); };
                vector_t<_Ty> lNewVector( aSize );
                std::generate( lNewVector.begin(), lNewVector.end(), gen );

                return sol::make_reference( aScriptState, std::move( lNewVector ) );
            }
        }

        template <typename _Ty> auto CreateVector0( uint32_t aSize, sol::this_state aScriptState )
        {
            auto lNewVector = vector_t<_Ty>( aSize );
            return sol::make_reference( aScriptState, std::move( lNewVector ) );
        }

        template <typename _Ty> auto CreateVector1( uint32_t aSize, _Ty aFill, sol::this_state aScriptState )
        {
            auto lNewVector = vector_t<_Ty>( aSize, aFill );
            return sol::make_reference( aScriptState, std::move( lNewVector ) );
        }

        template <typename _Ty> auto FetchFlattened( Cuda::multi_tensor_t &aMT, sol::this_state aScriptState )
        {
            auto x = aMT.FetchFlattened<_Ty>();
            return sol::make_reference( aScriptState, std::move( x ) );
        }

        template <typename _Ty> auto FetchBufferAt( Cuda::multi_tensor_t &aMT, uint32_t aLayer, sol::this_state aScriptState )
        {
            auto x = aMT.FetchBufferAt<_Ty>( aLayer );
            return sol::make_reference( aScriptState, std::move( x ) );
        }

        template <typename _Ty> size_t SizeAs( Cuda::multi_tensor_t &aMT ) { return aMT.SizeAs<_Ty>(); }

        template <typename _Ty> void Upload0( Cuda::multi_tensor_t &aM, sol::table &aArray )
        {
            auto &lArray = aArray.as<vector_t<_Ty>>();
            aM.Upload<_Ty>( lArray );
        }

        template <typename _Ty> void Upload1( Cuda::multi_tensor_t &aM, sol::table &aArray, uint32_t aLayer )
        {
            auto lArray = aArray.as<vector_t<_Ty>>();
            aM.Upload( lArray, aLayer, 0 );
        }

        template <typename _Ty> void Upload2( Cuda::multi_tensor_t &aM, sol::table &aArray, uint32_t aLayer, uint32_t aOffset )
        {
            auto lArray = aArray.as<vector_t<_Ty>>();
            aM.Upload<_Ty>( lArray, aOffset );
        }

        template <typename _Ty> auto Valid( entity_t &aEntity ) { return aEntity.IsValid(); }

        template <typename _Ty> auto Add( entity_t &aEntity, const sol::table &aInstance, sol::this_state aScriptState )
        {
            auto &lNewComponent = aEntity.Add<_Ty>( aInstance.valid() ? aInstance.as<_Ty>() : _Ty{} );
            return sol::make_reference( aScriptState, std::ref( lNewComponent ) );
        }

        template <typename _Ty> auto AddOrReplace( entity_t &aEntity, const sol::table &aInstance, sol::this_state aScriptState )
        {
            auto &lNewComponent = aEntity.AddOrReplace<_Ty>( aInstance.valid() ? aInstance.as<_Ty>() : _Ty{} );
            return sol::make_reference( aScriptState, std::ref( lNewComponent ) );
        }

        template <typename _Ty> auto Replace( entity_t &aEntity, const sol::table &aInstance, sol::this_state aScriptState )
        {
            auto &lNewComponent = aEntity.Replace<_Ty>( aInstance.valid() ? aInstance.as<_Ty>() : _Ty{} );
            return sol::make_reference( aScriptState, std::ref( lNewComponent ) );
        }

        template <typename _Ty> auto TryAdd( entity_t &aEntity, const sol::table &aInstance, sol::this_state aScriptState )
        {
            auto &lNewComponent = aEntity.TryAdd<_Ty>( aInstance.valid() ? aInstance.as<_Ty>() : _Ty{} );
            return sol::make_reference( aScriptState, std::ref( lNewComponent ) );
        }

        template <typename _Ty> auto Tag( entity_t &aEntity ) { aEntity.Tag<_Ty>(); }
        template <typename _Ty> auto Untag( entity_t &aEntity ) { aEntity.Untag<_Ty>(); }

        template <typename _Ty> auto Get( entity_t &aEntity, sol::this_state aScriptState )
        {
            auto &lNewComponent = aEntity.Get<_Ty>();
            return sol::make_reference( aScriptState, std::ref( lNewComponent ) );
        }

        template <typename _Ty> auto TryGet( entity_t &aEntity, sol::this_state aScriptState )
        {
            auto &lNewComponent = aEntity.TryGet<_Ty>( _Ty{} );
            return sol::make_reference( aScriptState, std::ref( lNewComponent ) );
        }

        template <typename _Ty> auto Has( entity_t &aEntity ) { return aEntity.Has<_Ty>(); }
        template <typename _Ty> auto Remove( entity_t &aEntity ) { aEntity.Remove<_Ty>(); }
        template <typename _Ty> auto TryRemove( entity_t &aEntity ) { aEntity.TryRemove<_Ty>(); }
    } // namespace

    [[nodiscard]] entt::id_type GetTypeID( const sol::table &aObject );

    template <typename T> [[nodiscard]] entt::id_type DeduceType( T &&aObject )
    {
        switch( aObject.get_type() )
        {
        case sol::type::number:
            return aObject.as<entt::id_type>();
        case sol::type::table:
            return GetTypeID( aObject );
        }
        assert( false );
        return -1;
    }

    template <typename... Args> inline auto InvokeMetaFunction( entt::meta_type meta_type, entt::id_type function_id, Args &&...args )
    {
        if( !meta_type )
        {
            assert( false );
        }
        else
        {
            auto meta_function = meta_type.func( function_id );
            if( meta_function )
                return meta_function.invoke( {}, std::forward<Args>( args )... );
        }
        return entt::meta_any{};
    }

    template <typename... Args> inline auto InvokeMetaFunction( entt::id_type type_id, entt::id_type function_id, Args &&...args )
    {
        return InvokeMetaFunction( entt::resolve( type_id ), function_id, std::forward<Args>( args )... );
    }

    template <typename _Ty> auto DeclarePrimitiveType( sol::table &aScriptingState, std::string const &aLuaName )
    {
        using namespace entt::literals;

        auto lNewLuaType                   = aScriptingState.new_usertype<_Ty>( aLuaName );
        lNewLuaType["type_id"]             = &entt::type_hash<_Ty>::value;
        lNewLuaType[sol::call_constructor] = []( _Ty value ) { return _Ty{ value }; };

        auto lNewType = entt::meta<_Ty>().type();

        lNewType.template func<&CreateVector0<_Ty>>( "CreateVector0"_hs );
        lNewType.template func<&CreateVector1<_Ty>>( "CreateVector1"_hs );

        if constexpr( std::is_arithmetic<_Ty>::value && !std::is_same<_Ty, uint8_t>::value && !std::is_same<_Ty, int8_t>::value )
        {
            lNewType.template func<&RandomVector<_Ty>>( "RandomVector"_hs );
        }

        lNewType.template func<&FetchFlattened<_Ty>>( "FetchFlattened"_hs );
        lNewType.template func<&FetchBufferAt<_Ty>>( "FetchBufferAt"_hs );
        lNewType.template func<&SizeAs<_Ty>>( "SizeAs"_hs );
        lNewType.template func<&Upload0<_Ty>>( "Upload0"_hs );
        lNewType.template func<&Upload1<_Ty>>( "Upload1"_hs );
        lNewType.template func<&Upload2<_Ty>>( "Upload2"_hs );

        if constexpr( std::is_class<_Ty>::value && std::is_empty<_Ty>::value )
        {
            lNewType.template func<&Valid<_Ty>>( "Valid"_hs );
            lNewType.template func<&Tag<_Ty>>( "Tag"_hs );
            lNewType.template func<&Untag<_Ty>>( "Untag"_hs );
            lNewType.template func<&Has<_Ty>>( "Has"_hs );
        }
        else if constexpr( std::is_class<_Ty>::value )
        {
            lNewType.template func<&Valid<_Ty>>( "Valid"_hs );
            lNewType.template func<&Add<_Ty>>( "Add"_hs );
            lNewType.template func<&AddOrReplace<_Ty>>( "AddOrReplace"_hs );
            lNewType.template func<&Replace<_Ty>>( "Replace"_hs );
            lNewType.template func<&TryAdd<_Ty>>( "TryAdd"_hs );
            lNewType.template func<&Get<_Ty>>( "Get"_hs );
            lNewType.template func<&TryGet<_Ty>>( "TryGet"_hs );
            lNewType.template func<&Has<_Ty>>( "Has"_hs );
            lNewType.template func<&Remove<_Ty>>( "Remove"_hs );
            lNewType.template func<&TryRemove<_Ty>>( "TryRemove"_hs );
        }

        return lNewLuaType;
    }
}; // namespace SE::Core