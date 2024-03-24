#pragma once
#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

#include <type_traits>

#include "Core/TypeReflection.h"
#include "Core/Entity/Collection.h"
// #include "Core/CUDA/Texture/TextureData.h"

#include "Core/CUDA/Array/MultiTensor.h"

namespace SE::Core
{
    namespace
    {
        template <typename _Ty>
        auto RandomVector( size_t size, double min, double max, sol::this_state scriptState )
        {
            std::random_device dev;
            std::mt19937       rng( dev() );

            _Ty lMin = static_cast<_Ty>( min );
            _Ty lMax = static_cast<_Ty>( max );

            if constexpr( std::is_floating_point<_Ty>::value )
            {
                std::uniform_real_distribution<_Ty> dist6( lMin, lMax );
                auto                                gen = [&dist6, &rng]() { return dist6( rng ); };
                vector_t<_Ty>                       newVector( size );
                std::generate( newVector.begin(), newVector.end(), gen );

                return sol::make_reference( scriptState, std::move( newVector ) );
            }
            else
            {
                std::uniform_int_distribution<_Ty> dist6( lMin, lMax );
                auto                               gen = [&dist6, &rng]() { return dist6( rng ); };
                vector_t<_Ty>                      newVector( size );
                std::generate( newVector.begin(), newVector.end(), gen );

                return sol::make_reference( scriptState, std::move( newVector ) );
            }
        }

        template <typename _Ty>
        auto CreateVector0( uint32_t size, sol::this_state scriptState )
        {
            auto newVector = vector_t<_Ty>( size );
            return sol::make_reference( scriptState, std::move( newVector ) );
        }

        template <typename _Ty>
        auto CreateVector1( uint32_t size, _Ty fill, sol::this_state scriptState )
        {
            auto newVector = vector_t<_Ty>( size, fill );
            return sol::make_reference( scriptState, std::move( newVector ) );
        }

        template <typename _Ty>
        auto FetchFlattened( Cuda::multi_tensor_t &self, sol::this_state scriptState )
        {
            auto x = self.FetchFlattened<_Ty>();
            return sol::make_reference( scriptState, std::move( x ) );
        }

        template <typename _Ty>
        auto FetchBufferAt( Cuda::multi_tensor_t &self, uint32_t layer, sol::this_state scriptState )
        {
            auto x = self.FetchBufferAt<_Ty>( layer );
            return sol::make_reference( scriptState, std::move( x ) );
        }

        template <typename _Ty>
        size_t SizeAs( Cuda::multi_tensor_t &self )
        {
            return self.SizeAs<_Ty>();
        }

        template <typename _Ty>
        void Upload0( Cuda::multi_tensor_t &self, sol::table &array )
        {
            auto &array0 = array.as<vector_t<_Ty>>();
            self.Upload<_Ty>( array0 );
        }

        template <typename _Ty>
        void Upload1( Cuda::multi_tensor_t &self, sol::table &array, uint32_t layer )
        {
            auto array0 = array.as<vector_t<_Ty>>();
            self.Upload( array0, layer, 0 );
        }

        template <typename _Ty>
        void Upload2( Cuda::multi_tensor_t &self, sol::table &array, uint32_t layer, uint32_t offset )
        {
            auto array0 = array.as<vector_t<_Ty>>();
            self.Upload<_Ty>( array0, offset );
        }

        template <typename _Ty>
        auto Valid( entity_t &self )
        {
            return self.IsValid();
        }

        template <typename _Ty>
        auto Add( entity_t &self, const sol::table &instance, sol::this_state scriptState )
        {
            auto &newComponent = self.Add<_Ty>( instance.valid() ? instance.as<_Ty>() : _Ty{} );
            return sol::make_reference( scriptState, std::ref( newComponent ) );
        }

        template <typename _Ty>
        auto AddOrReplace( entity_t &self, const sol::table &instance, sol::this_state scriptState )
        {
            auto &newComponent = self.AddOrReplace<_Ty>( instance.valid() ? instance.as<_Ty>() : _Ty{} );
            return sol::make_reference( scriptState, std::ref( newComponent ) );
        }

        template <typename _Ty>
        auto Replace( entity_t &self, const sol::table &instance, sol::this_state scriptState )
        {
            auto &newComponent = self.Replace<_Ty>( instance.valid() ? instance.as<_Ty>() : _Ty{} );
            return sol::make_reference( scriptState, std::ref( newComponent ) );
        }

        template <typename _Ty>
        auto TryAdd( entity_t &self, const sol::table &instance, sol::this_state scriptState )
        {
            auto &newComponent = self.TryAdd<_Ty>( instance.valid() ? instance.as<_Ty>() : _Ty{} );
            return sol::make_reference( scriptState, std::ref( newComponent ) );
        }

        template <typename _Ty>
        auto Tag( entity_t &self )
        {
            self.Tag<_Ty>();
        }

        template <typename _Ty>
        auto Untag( entity_t &self )
        {
            self.Untag<_Ty>();
        }

        template <typename _Ty>
        auto Get( entity_t &self, sol::this_state scriptState )
        {
            auto &newComponent = self.Get<_Ty>();
            return sol::make_reference( scriptState, std::ref( newComponent ) );
        }

        template <typename _Ty>
        auto TryGet( entity_t &self, sol::this_state scriptState )
        {
            auto &newComponent = self.TryGet<_Ty>( _Ty{} );
            return sol::make_reference( scriptState, std::ref( newComponent ) );
        }

        template <typename _Ty>
        auto Has( entity_t &self )
        {
            return self.Has<_Ty>();
        }

        template <typename _Ty>
        auto Remove( entity_t &self )
        {
            self.Remove<_Ty>();
        }

        template <typename _Ty>
        auto TryRemove( entity_t &self )
        {
            self.TryRemove<_Ty>();
        }
    } // namespace

    // [[nodiscard]] entt::id_type get_type_id( const sol::table &aObject );

    // template <typename T>
    // [[nodiscard]] entt::id_type deduce_type( T &&aObject )
    // {
    //     switch( aObject.get_type() )
    //     {
    //     case sol::type::number:
    //         return aObject.as<entt::id_type>();
    //     case sol::type::table:
    //         return get_type_id( aObject );
    //     }
    //     assert( false );
    //     return -1;
    // }

    // template <typename... Args>
    // inline auto invoke_meta_function( entt::meta_type meta_type, entt::id_type function_id, Args &&...args )
    // {
    //     if( !meta_type )
    //     {
    //         assert( false );
    //     }
    //     else
    //     {
    //         auto meta_function = meta_type.func( function_id );
    //         if( meta_function )
    //             return meta_function.invoke( {}, std::forward<Args>( args )... );
    //     }
    //     return entt::meta_any{};
    // }

    // template <typename... Args>
    // inline auto invoke_meta_function( entt::id_type type_id, entt::id_type function_id, Args &&...args )
    // {
    //     return invoke_meta_function( entt::resolve( type_id ), function_id, std::forward<Args>( args )... );
    // }

    template <typename _Ty>
    auto declare_primitive_type( sol::table &scriptingState, std::string const &name )
    {
        using namespace entt::literals;

        auto newLuaType                   = scriptingState.new_usertype<_Ty>( name );
        newLuaType["type_id"]             = &entt::type_hash<_Ty>::value;
        newLuaType[sol::call_constructor] = []( _Ty value ) { return _Ty{ value }; };

        auto newType = entt::meta<_Ty>().type();

        newType.template func<&CreateVector0<_Ty>>( "CreateVector0"_hs );
        newType.template func<&CreateVector1<_Ty>>( "CreateVector1"_hs );

        if constexpr( std::is_arithmetic<_Ty>::value && !std::is_same<_Ty, uint8_t>::value && !std::is_same<_Ty, int8_t>::value )
        {
            newType.template func<&RandomVector<_Ty>>( "RandomVector"_hs );
        }

        newType.template func<&FetchFlattened<_Ty>>( "FetchFlattened"_hs );
        newType.template func<&FetchBufferAt<_Ty>>( "FetchBufferAt"_hs );
        newType.template func<&SizeAs<_Ty>>( "SizeAs"_hs );
        newType.template func<&Upload0<_Ty>>( "Upload0"_hs );
        newType.template func<&Upload1<_Ty>>( "Upload1"_hs );
        newType.template func<&Upload2<_Ty>>( "Upload2"_hs );

        if constexpr( std::is_class<_Ty>::value && std::is_empty<_Ty>::value )
        {
            newType.template func<&Valid<_Ty>>( "Valid"_hs );
            newType.template func<&Tag<_Ty>>( "Tag"_hs );
            newType.template func<&Untag<_Ty>>( "Untag"_hs );
            newType.template func<&Has<_Ty>>( "Has"_hs );
        }
        else if constexpr( std::is_class<_Ty>::value )
        {
            newType.template func<&Valid<_Ty>>( "Valid"_hs );
            newType.template func<&Add<_Ty>>( "Add"_hs );
            newType.template func<&AddOrReplace<_Ty>>( "AddOrReplace"_hs );
            newType.template func<&Replace<_Ty>>( "Replace"_hs );
            newType.template func<&TryAdd<_Ty>>( "TryAdd"_hs );
            newType.template func<&Get<_Ty>>( "Get"_hs );
            newType.template func<&TryGet<_Ty>>( "TryGet"_hs );
            newType.template func<&Has<_Ty>>( "Has"_hs );
            newType.template func<&Remove<_Ty>>( "Remove"_hs );
            newType.template func<&TryRemove<_Ty>>( "TryRemove"_hs );
        }

        return newLuaType;
    }
}; // namespace SE::Core