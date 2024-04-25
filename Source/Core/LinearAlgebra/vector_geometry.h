#include "core.h"

#include <cmath>

namespace numlua::linalg
{
    namespace detail
    {
        template <length_t L, typename T, bool Aligned>
        struct compute_length
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF static T call( vect<L, T> const &v )
            {
                return sqrt( dot( v, v ) );
            }
        };

        template <length_t L, typename T, bool Aligned>
        struct compute_distance
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF static T call( vect<L, T> const &p0, vect<L, T> const &p1 )
            {
                return length( p1 - p0 );
            }
        };

        template <typename V, typename T, bool Aligned>
        struct compute_dot
        {
        };

        template <typename T, bool Aligned>
        struct compute_dot<vect<2, T>, T, Aligned>
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF GLM_CONSTEXPR static T call( vect<2, T> const &a, vect<2, T> const &b )
            {
                vect<2, T> tmp( a * b );

                return tmp.x + tmp.y;
            }
        };

        template <typename T, bool Aligned>
        struct compute_dot<vect<3, T>, T, Aligned>
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF GLM_CONSTEXPR static T call( vect<3, T> const &a, vect<3, T> const &b )
            {
                vect<3, T> tmp( a * b );

                return tmp.x + tmp.y + tmp.z;
            }
        };

        template <typename T, bool Aligned>
        struct compute_dot<vect<4, T>, T, Aligned>
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF GLM_CONSTEXPR static T call( vect<4, T> const &a, vect<4, T> const &b )
            {
                vect<4, T> tmp( a * b );

                return ( tmp.x + tmp.y ) + ( tmp.z + tmp.w );
            }
        };

        template <typename T, bool Aligned>
        struct compute_cross
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF GLM_CONSTEXPR static vect<3, T> call( vect<3, T> const &x, vect<3, T> const &y )
            {
                return vect<3, T>( x.y * y.z - y.y * x.z, x.z * y.x - y.z * x.x, x.x * y.y - y.x * x.y );
            }

            SE_CUDA_HOST_DEVICE_FUNCTION_DEF GLM_CONSTEXPR static vect<4, T> call( vect<4, T> const &x, vect<4, T> const &y )
            {
                return vect<4, T>( x.y * y.z - y.y * x.z, x.z * y.x - y.z * x.x, x.x * y.y - y.x * x.y, 0.0f );
            }
        };

        template <length_t L, typename T, bool Aligned>
        struct compute_normalize
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF static vect<L, T> call( vect<L, T> const &v )
            {
                return v * inversesqrt( dot( v, v ) );
            }
        };

        template <length_t L, typename T, bool Aligned>
        struct compute_faceforward
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF static vect<L, T> call( vect<L, T> const &N, vect<L, T> const &I,
                                                                       vect<L, T> const &Nref )
            {
                return dot( Nref, I ) < static_cast<T>( 0 ) ? N : -N;
            }
        };

        template <length_t L, typename T, bool Aligned>
        struct compute_reflect
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF static vect<L, T> call( vect<L, T> const &I, vect<L, T> const &N )
            {
                return I - N * dot( N, I ) * static_cast<T>( 2 );
            }
        };

        template <length_t L, typename T, bool Aligned>
        struct compute_refract
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF static vect<L, T> call( vect<L, T> const &I, vect<L, T> const &N, T eta )
            {
                T const            dotValue( dot( N, I ) );
                T const            k( static_cast<T>( 1 ) - eta * eta * ( static_cast<T>( 1 ) - dotValue * dotValue ) );
                vect<L, T> const Result =
                    ( k >= static_cast<T>( 0 ) ) ? ( eta * I - ( eta * dotValue + std::sqrt( k ) ) * N ) : vect<L, T>( 0 );
                return Result;
            }
        };

        using std::log2;
        template <length_t L, typename T, bool isFloat, bool Aligned>
        struct compute_log2
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF static vect<L, T> call( vect<L, T> const &v )
            {
                return detail::functor1<vec, L, T, T, Q>::call( log2, v );
            }
        };

        template <length_t L, typename T, bool Aligned>
        struct compute_sqrt
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF static vect<L, T> call( vect<L, T> const &x )
            {
                return detail::functor1<vec, L, T, T, Q>::call( std::sqrt, x );
            }
        };

        template <length_t L, typename T, bool Aligned>
        struct compute_inversesqrt
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF static vect<L, T> call( vect<L, T> const &x )
            {
                return static_cast<T>( 1 ) / sqrt( x );
            }
        };

        template <length_t L, bool Aligned>
        struct compute_inversesqrt<L, float, lowp, Aligned>
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF static vec<L, float, lowp> call( vec<L, float, lowp> const &x )
            {
                vec<L, float, lowp>  tmp( x );
                vec<L, float, lowp>  xhalf( tmp * 0.5f );
                vec<L, uint, lowp>  *p    = reinterpret_cast<vec<L, uint, lowp> *>( const_cast<vec<L, float, lowp> *>( &x ) );
                vec<L, uint, lowp>   i    = vec<L, uint, lowp>( 0x5f375a86 ) - ( *p >> vec<L, uint, lowp>( 1 ) );
                vec<L, float, lowp> *ptmp = reinterpret_cast<vec<L, float, lowp> *>( &i );
                tmp                       = *ptmp;
                tmp                       = tmp * ( 1.5f - xhalf * tmp * tmp );

                return tmp;
            }
        };
    } // namespace detail

    // length
    template <typename genType>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF genType length( genType x )
    {
        return abs( x );
    }

    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF T length( vect<L, T> const &v )
    {
        return detail::compute_length<L, T, Q, detail::is_aligned<Q>::value>::call( v );
    }

    // distance
    template <typename genType>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF genType distance( genType const &p0, genType const &p1 )
    {
        return length( p1 - p0 );
    }

    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF T distance( vect<L, T> const &p0, vect<L, T> const &p1 )
    {
        return detail::compute_distance<L, T, Q, detail::is_aligned<Q>::value>::call( p0, p1 );
    }

    // dot
    template <typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF GLM_CONSTEXPR T dot( T x, T y )
    {
        return x * y;
    }

    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF GLM_CONSTEXPR T dot( vect<L, T> const &x, vect<L, T> const &y )
    {
        return detail::compute_dot<vect<L, T>, T, detail::is_aligned<Q>::value>::call( x, y );
    }

    // cross
    template <typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF GLM_CONSTEXPR vect<3, T> cross( vect<3, T> const &x, vect<3, T> const &y )
    {
        return detail::compute_cross<T, Q, detail::is_aligned<Q>::value>::call( x, y );
    }
    /*
        // normalize
        template<typename genType>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF genType normalize(genType const& x)
        {
            GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'normalize' accepts only floating-point inputs");

            return x < genType(0) ? genType(-1) : genType(1);
        }
    */
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> normalize( vect<L, T> const &x )
    {
        return detail::compute_normalize<L, T, Q, detail::is_aligned<Q>::value>::call( x );
    }

    // faceforward
    template <typename genType>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF genType faceforward( genType const &N, genType const &I, genType const &Nref )
    {
        return dot( Nref, I ) < static_cast<genType>( 0 ) ? N : -N;
    }

    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> faceforward( vect<L, T> const &N, vect<L, T> const &I, vect<L, T> const &Nref )
    {
        return detail::compute_faceforward<L, T, Q, detail::is_aligned<Q>::value>::call( N, I, Nref );
    }

    // reflect
    template <typename genType>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF genType reflect( genType const &I, genType const &N )
    {
        return I - N * dot( N, I ) * genType( 2 );
    }

    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> reflect( vect<L, T> const &I, vect<L, T> const &N )
    {
        return detail::compute_reflect<L, T, Q, detail::is_aligned<Q>::value>::call( I, N );
    }

    // refract
    template <typename genType>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF genType refract( genType const &I, genType const &N, genType eta )
    {
        genType const dotValue( dot( N, I ) );
        genType const k( static_cast<genType>( 1 ) - eta * eta * ( static_cast<genType>( 1 ) - dotValue * dotValue ) );

        return ( eta * I - ( eta * dotValue + sqrt( k ) ) * N ) * static_cast<genType>( k >= static_cast<genType>( 0 ) );
    }

    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> refract( vect<L, T> const &I, vect<L, T> const &N, T eta )
    {
        return detail::compute_refract<L, T, Q, detail::is_aligned<Q>::value>::call( I, N, eta );
    }

    // pow
    using std::pow;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> pow( vect<L, T> const &base, vect<L, T> const &exponent )
    {
        return detail::functor2<vec, L, T, Q>::call( pow, base, exponent );
    }

    // exp
    using std::exp;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> exp( vect<L, T> const &x )
    {
        return detail::functor1<vec, L, T, T, Q>::call( exp, x );
    }

    // log
    using std::log;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> log( vect<L, T> const &x )
    {
        return detail::functor1<vec, L, T, T, Q>::call( log, x );
    }

    using std::exp2;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> exp2( vect<L, T> const &x )
    {
        return detail::functor1<vec, L, T, T, Q>::call( exp2, x );
    }

    using std::log2;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> log2( vect<L, T> const &x )
    {
        return detail::compute_log2<L, T, Q, std::numeric_limits<T>::is_iec559, detail::is_aligned<Q>::value>::call( x );
    }

    // sqrt
    using std::sqrt;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> sqrt( vect<L, T> const &x )
    {
        return detail::compute_sqrt<L, T, Q, detail::is_aligned<Q>::value>::call( x );
    }

    // inversesqrt
    template <typename genType>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF genType inversesqrt( genType x )
    {
        return static_cast<genType>( 1 ) / sqrt( x );
    }

    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> inversesqrt( vect<L, T> const &x )
    {
        return detail::compute_inversesqrt<L, T, Q, detail::is_aligned<Q>::value>::call( x );
    }

    // radians
    template <typename genType>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF GLM_CONSTEXPR genType radians( genType degrees )
    {
        return degrees * static_cast<genType>( 0.01745329251994329576923690768489 );
    }

    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF GLM_CONSTEXPR vect<L, T> radians( vect<L, T> const &v )
    {
        return detail::functor1<vec, L, T, T, Q>::call( radians, v );
    }

    // degrees
    template <typename genType>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF GLM_CONSTEXPR genType degrees( genType radians )
    {
        return radians * static_cast<genType>( 57.295779513082320876798154814105 );
    }

    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF GLM_CONSTEXPR vect<L, T> degrees( vect<L, T> const &v )
    {
        return detail::functor1<vec, L, T, T, Q>::call( degrees, v );
    }

    // sin
    using ::std::sin;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> sin( vect<L, T> const &v )
    {
        return detail::functor1<vec, L, T, T, Q>::call( sin, v );
    }

    // cos
    using std::cos;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> cos( vect<L, T> const &v )
    {
        return detail::functor1<vec, L, T, T, Q>::call( cos, v );
    }

    // tan
    using std::tan;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> tan( vect<L, T> const &v )
    {
        return detail::functor1<vec, L, T, T, Q>::call( tan, v );
    }

    // asin
    using std::asin;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> asin( vect<L, T> const &v )
    {
        return detail::functor1<vec, L, T, T, Q>::call( asin, v );
    }

    // acos
    using std::acos;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> acos( vect<L, T> const &v )
    {
        return detail::functor1<vec, L, T, T, Q>::call( acos, v );
    }

    // atan
    template <typename genType>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF genType atan( genType y, genType x )
    {
        return ::std::atan2( y, x );
    }

    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> atan( vect<L, T> const &y, vect<L, T> const &x )
    {
        return detail::functor2<vec, L, T, Q>::call( ::std::atan2, y, x );
    }

    using std::atan;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> atan( vect<L, T> const &v )
    {
        return detail::functor1<vec, L, T, T, Q>::call( atan, v );
    }

    // sinh
    using std::sinh;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> sinh( vect<L, T> const &v )
    {
        return detail::functor1<vec, L, T, T, Q>::call( sinh, v );
    }

    // cosh
    using std::cosh;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> cosh( vect<L, T> const &v )
    {
        return detail::functor1<vec, L, T, T, Q>::call( cosh, v );
    }

    // tanh
    using std::tanh;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> tanh( vect<L, T> const &v )
    {
        return detail::functor1<vec, L, T, T, Q>::call( tanh, v );
    }

    // asinh
    using std::asinh;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> asinh( vect<L, T> const &v )
    {
        return detail::functor1<vec, L, T, T, Q>::call( asinh, v );
    }

    // acosh
    using std::acosh;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> acosh( vect<L, T> const &v )
    {
        return detail::functor1<vec, L, T, T, Q>::call( acosh, v );
    }

    // atanh
    using std::atanh;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> atanh( vect<L, T> const &v )
    {
        return detail::functor1<vec, L, T, T, Q>::call( atanh, v );
    }
} // namespace numlua::linalg
