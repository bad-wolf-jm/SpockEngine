#pragma once

#include <cmath>
#include <cstdint>

#include "Core/Logging.h"

namespace TestUtils
{
    constexpr float EPSILON = 0.000001;

    template <typename T> uint32_t prod( std::vector<T> x ) { return std::accumulate( x.begin(), x.end(), 1, std::multiplies<T>() ); }

    template <typename T> T sum( std::vector<T> x ) { return std::accumulate( x.begin(), x.end(), static_cast<T>( 0 ), std::plus<T>() ); }

    template <typename T> std::vector<T> absol( std::vector<T> x )
    {
        std::vector<T> z( x.size() );
        for( uint32_t i = 0; i < z.size(); i++ )
        {
            z[i] = x[i] < static_cast<T>( 0 ) ? ( -x[i] ) : x[i];
        }
        return z;
    }

    template <typename T> T amax( std::vector<T> x )
    {
        auto m = std::max_element( x.begin(), x.end() );
        // WB::Logging::Info("{}", *m);
        return *m;
    }

    template <typename T> std::vector<T> operator*( std::vector<T> x, std::vector<T> y )
    {
        std::vector<T> z( x.size() );
        for( uint32_t i = 0; i < z.size(); i++ )
        {
            z[i] = x[i] * y[i];
        }
        return z;
    }

    template <typename T> std::vector<T> operator-( std::vector<T> x, std::vector<T> y )
    {
        std::vector<T> z( x.size() );
        for( uint32_t i = 0; i < z.size(); i++ )
        {
            z[i] = x[i] - y[i];
        }
        return z;
    }

    template <typename T> float SNorm( std::vector<T> x ) { return sum( x * x ); }

    template <typename T> bool VectorEqual( std::vector<T> x, std::vector<T> y ) { return amax( absol( y - x ) ) < EPSILON; }

} // namespace TestUtils