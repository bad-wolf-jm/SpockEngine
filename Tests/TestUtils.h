/// @file   TestUtils.h
///
/// @brief  Utility functions for unit tests
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert All rights reserved.

#pragma once

#include "Core/Math/Types.h"

#include <array>
#include <cstdint>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

namespace TestUtils
{
    constexpr float EPSILON = 0.000001;

    template <typename T>
    T Prod( std::vector<T> x )
    {
        return std::accumulate( x.begin(), x.end(), 1, std::multiplies<T>() );
    }

    template <typename T>
    T Sum( std::vector<T> x )
    {
        return std::accumulate( x.begin(), x.end(), static_cast<T>( 0 ), std::plus<T>() );
    }

    template <typename T>
    std::vector<T> Absol( std::vector<T> x )
    {
        std::vector<T> z( x.size() );
        for( uint32_t i = 0; i < z.size(); i++ )
        {
            z[i] = x[i] < static_cast<T>( 0 ) ? ( -x[i] ) : x[i];
        }
        return z;
    }

    template <typename T>
    T ArrayMax( std::vector<T> x )
    {
        auto m = std::max_element( x.begin(), x.end() );
        return *m;
    }

    template <typename T>
    std::vector<T> operator+( T y, std::vector<T> x )
    {
        std::vector<T> z( x.size() );
        for( uint32_t i = 0; i < z.size(); i++ )
        {
            z[i] = x[i] + y;
        }
        return z;
    }

    template <typename T>
    std::vector<T> operator+( std::vector<T> x, T y )
    {
        std::vector<T> z( x.size() );
        for( uint32_t i = 0; i < z.size(); i++ )
        {
            z[i] = x[i] + y;
        }
        return z;
    }

    template <typename T>
    std::vector<T> operator+( std::vector<T> x, std::vector<T> y )
    {
        std::vector<T> z( x.size() );
        for( uint32_t i = 0; i < z.size(); i++ )
        {
            z[i] = x[i] + y[i];
        }
        return z;
    }

    template <typename T>
    std::vector<T> operator*( std::vector<T> x, std::vector<T> y )
    {
        std::vector<T> z( x.size() );
        for( uint32_t i = 0; i < z.size(); i++ )
        {
            z[i] = x[i] * y[i];
        }
        return z;
    }

    template <typename T>
    std::vector<T> operator*( std::vector<T> x, T y )
    {
        std::vector<T> z( x.size() );
        for( uint32_t i = 0; i < z.size(); i++ )
        {
            z[i] = x[i] * y;
        }
        return z;
    }

    template <typename T>
    std::vector<T> operator-( std::vector<T> x, std::vector<T> y )
    {
        std::vector<T> z( x.size() );
        for( uint32_t i = 0; i < z.size(); i++ )
        {
            z[i] = x[i] - y[i];
        }
        return z;
    }

    template <typename T, size_t N>
    std::vector<T> operator-( std::array<T, N> x, std::array<T, N> y )
    {
        std::vector<T> z( N );
        for( uint32_t i = 0; i < z.size(); i++ )
        {
            z[i] = x[i] - y[i];
        }
        return z;
    }

    template <typename T>
    float SquareNorm( std::vector<T> x )
    {
        return Sum( x * x );
    }

    template <typename T>
    bool VectorEqual( std::vector<T> x, std::vector<T> y )
    {
        return ArrayMax( Absol( y - x ) ) < EPSILON;
    }
    template <typename T>
    bool VectorEqual( std::vector<T> x, std::vector<T> y, T e )
    {
        return ArrayMax( Absol( y - x ) ) < e;
    }
    // template <typename T> bool VectorEqual( std::vector<T> x, std::vector<T> y ) { return SquareNorm( y - x ) < EPSILON; }

    template <typename T, size_t N>
    bool VectorEqual( std::array<T, N> x, std::array<T, N> y )
    {
        return SquareNorm( y - x ) < EPSILON;
    }

    bool VectorEqual( math::vec2 x, math::vec2 y );
    bool VectorEqual( math::vec2 x, math::vec2 y, float e );

    std::vector<uint8_t> RandomBool( size_t aSize );

    uint8_t RandomBool();

    template <typename _IntType>
    std::vector<_IntType> RandomNumber( size_t aSize )
    {
        std::random_device dev;
        std::mt19937       rng( dev() );

        if constexpr( std::is_floating_point<_IntType>::value )
        {
            std::uniform_real_distribution<_IntType> dist6(
                std::numeric_limits<_IntType>::min(), std::numeric_limits<_IntType>::max() );
            auto                  gen = [&dist6, &rng]() { return dist6( rng ); };
            std::vector<_IntType> x( aSize );
            std::generate( x.begin(), x.end(), gen );

            return x;
        }
        else
        {
            std::uniform_int_distribution<_IntType> dist6(
                std::numeric_limits<_IntType>::min(), std::numeric_limits<_IntType>::max() );
            auto                  gen = [&dist6, &rng]() { return dist6( rng ); };
            std::vector<_IntType> x( aSize );
            std::generate( x.begin(), x.end(), gen );

            return x;
        }
    }

    template <typename _IntType>
    _IntType RandomNumber( _IntType aMin, _IntType aMax )
    {
        std::random_device dev;
        std::mt19937       rng( dev() );
        if constexpr( std::is_floating_point<_IntType>::value )
        {
            std::uniform_real_distribution<_IntType> dist6( aMin, aMax );
            return dist6( rng );
        }
        else
        {
            std::uniform_int_distribution<_IntType> dist6( aMin, aMax );
            return dist6( rng );
        }
    }

    template <typename _IntType>
    std::vector<_IntType> RandomNumber( size_t aSize, _IntType aMin, _IntType aMax )
    {
        std::random_device dev;
        std::mt19937       rng( dev() );

        if constexpr( std::is_floating_point<_IntType>::value )
        {
            std::uniform_real_distribution<_IntType> dist6( aMin, aMax );
            auto                                     gen = [&dist6, &rng]() { return dist6( rng ); };
            std::vector<_IntType>                    x( aSize );
            std::generate( x.begin(), x.end(), gen );

            return x;
        }
        else
        {
            std::uniform_int_distribution<_IntType> dist6( aMin, aMax );
            auto                                    gen = [&dist6, &rng]() { return dist6( rng ); };
            std::vector<_IntType>                   x( aSize );
            std::generate( x.begin(), x.end(), gen );

            return x;
        }
    }

    template <typename _Ty>
    std::vector<_Ty> ConcatenateVectors( std::vector<std::vector<_Ty>> aList )
    {
        std::vector<_Ty> lResult;
        for( auto &lV : aList ) lResult.insert( lResult.end(), lV.begin(), lV.end() );

        return lResult;
    }

} // namespace TestUtils
