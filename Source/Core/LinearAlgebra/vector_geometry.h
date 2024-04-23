#include "../common.hpp"
#include "../exponential.hpp"

namespace glm
{
    namespace detail
    {
        template <length_t L, typename T, qualifier Q, bool Aligned>
        struct compute_length
        {
            GLM_FUNC_QUALIFIER static T call( vec<L, T, Q> const &v )
            {
                return sqrt( dot( v, v ) );
            }
        };

        template <length_t L, typename T, qualifier Q, bool Aligned>
        struct compute_distance
        {
            GLM_FUNC_QUALIFIER static T call( vec<L, T, Q> const &p0, vec<L, T, Q> const &p1 )
            {
                return length( p1 - p0 );
            }
        };

        template <typename V, typename T, bool Aligned>
        struct compute_dot
        {
        };

        template <typename T, qualifier Q, bool Aligned>
        struct compute_dot<vec<2, T, Q>, T, Aligned>
        {
            GLM_FUNC_QUALIFIER GLM_CONSTEXPR static T call( vec<2, T, Q> const &a, vec<2, T, Q> const &b )
            {
                vec<2, T, Q> tmp( a * b );

                return tmp.x + tmp.y;
            }
        };

        template <typename T, qualifier Q, bool Aligned>
        struct compute_dot<vec<3, T, Q>, T, Aligned>
        {
            GLM_FUNC_QUALIFIER GLM_CONSTEXPR static T call( vec<3, T, Q> const &a, vec<3, T, Q> const &b )
            {
                vec<3, T, Q> tmp( a * b );

                return tmp.x + tmp.y + tmp.z;
            }
        };

        template <typename T, qualifier Q, bool Aligned>
        struct compute_dot<vec<4, T, Q>, T, Aligned>
        {
            GLM_FUNC_QUALIFIER GLM_CONSTEXPR static T call( vec<4, T, Q> const &a, vec<4, T, Q> const &b )
            {
// VS 17.7.4 generates longer assembly (~20 instructions vs 11 instructions)
#if defined( _MSC_VER )
                return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
#else
                vec<4, T, Q> tmp( a * b );
                return ( tmp.x + tmp.y ) + ( tmp.z + tmp.w );
#endif
            }
        };

        template <typename T, qualifier Q, bool Aligned>
        struct compute_cross
        {
            GLM_FUNC_QUALIFIER GLM_CONSTEXPR static vec<3, T, Q> call( vec<3, T, Q> const &x, vec<3, T, Q> const &y )
            {
                GLM_STATIC_ASSERT( std::numeric_limits<T>::is_iec559, "'cross' accepts only floating-point inputs" );

                return vec<3, T, Q>( x.y * y.z - y.y * x.z, x.z * y.x - y.z * x.x, x.x * y.y - y.x * x.y );
            }

            GLM_FUNC_QUALIFIER GLM_CONSTEXPR static vec<4, T, Q> call( vec<4, T, Q> const &x, vec<4, T, Q> const &y )
            {
                GLM_STATIC_ASSERT( std::numeric_limits<T>::is_iec559, "'cross' accepts only floating-point inputs" );

                return vec<4, T, Q>( x.y * y.z - y.y * x.z, x.z * y.x - y.z * x.x, x.x * y.y - y.x * x.y, 0.0f );
            }
        };

        template <length_t L, typename T, qualifier Q, bool Aligned>
        struct compute_normalize
        {
            GLM_FUNC_QUALIFIER static vec<L, T, Q> call( vec<L, T, Q> const &v )
            {
                GLM_STATIC_ASSERT( std::numeric_limits<T>::is_iec559, "'normalize' accepts only floating-point inputs" );

                return v * inversesqrt( dot( v, v ) );
            }
        };

        template <length_t L, typename T, qualifier Q, bool Aligned>
        struct compute_faceforward
        {
            GLM_FUNC_QUALIFIER static vec<L, T, Q> call( vec<L, T, Q> const &N, vec<L, T, Q> const &I, vec<L, T, Q> const &Nref )
            {
                GLM_STATIC_ASSERT( std::numeric_limits<T>::is_iec559, "'normalize' accepts only floating-point inputs" );

                return dot( Nref, I ) < static_cast<T>( 0 ) ? N : -N;
            }
        };

        template <length_t L, typename T, qualifier Q, bool Aligned>
        struct compute_reflect
        {
            GLM_FUNC_QUALIFIER static vec<L, T, Q> call( vec<L, T, Q> const &I, vec<L, T, Q> const &N )
            {
                return I - N * dot( N, I ) * static_cast<T>( 2 );
            }
        };

        template <length_t L, typename T, qualifier Q, bool Aligned>
        struct compute_refract
        {
            GLM_FUNC_QUALIFIER static vec<L, T, Q> call( vec<L, T, Q> const &I, vec<L, T, Q> const &N, T eta )
            {
                T const            dotValue( dot( N, I ) );
                T const            k( static_cast<T>( 1 ) - eta * eta * ( static_cast<T>( 1 ) - dotValue * dotValue ) );
                vec<L, T, Q> const Result =
                    ( k >= static_cast<T>( 0 ) ) ? ( eta * I - ( eta * dotValue + std::sqrt( k ) ) * N ) : vec<L, T, Q>( 0 );
                return Result;
            }
        };
    } // namespace detail

    // length
    template <typename genType>
    GLM_FUNC_QUALIFIER genType length( genType x )
    {
        GLM_STATIC_ASSERT( std::numeric_limits<genType>::is_iec559, "'length' accepts only floating-point inputs" );

        return abs( x );
    }

    template <length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER T length( vec<L, T, Q> const &v )
    {
        GLM_STATIC_ASSERT( std::numeric_limits<T>::is_iec559, "'length' accepts only floating-point inputs" );

        return detail::compute_length<L, T, Q, detail::is_aligned<Q>::value>::call( v );
    }

    // distance
    template <typename genType>
    GLM_FUNC_QUALIFIER genType distance( genType const &p0, genType const &p1 )
    {
        GLM_STATIC_ASSERT( std::numeric_limits<genType>::is_iec559, "'distance' accepts only floating-point inputs" );

        return length( p1 - p0 );
    }

    template <length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER T distance( vec<L, T, Q> const &p0, vec<L, T, Q> const &p1 )
    {
        return detail::compute_distance<L, T, Q, detail::is_aligned<Q>::value>::call( p0, p1 );
    }

    // dot
    template <typename T>
    GLM_FUNC_QUALIFIER GLM_CONSTEXPR T dot( T x, T y )
    {
        GLM_STATIC_ASSERT( std::numeric_limits<T>::is_iec559, "'dot' accepts only floating-point inputs" );

        return x * y;
    }

    template <length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER GLM_CONSTEXPR T dot( vec<L, T, Q> const &x, vec<L, T, Q> const &y )
    {
        GLM_STATIC_ASSERT( std::numeric_limits<T>::is_iec559, "'dot' accepts only floating-point inputs" );

        return detail::compute_dot<vec<L, T, Q>, T, detail::is_aligned<Q>::value>::call( x, y );
    }

    // cross
    template <typename T, qualifier Q>
    GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> cross( vec<3, T, Q> const &x, vec<3, T, Q> const &y )
    {
        return detail::compute_cross<T, Q, detail::is_aligned<Q>::value>::call( x, y );
    }
    /*
        // normalize
        template<typename genType>
        GLM_FUNC_QUALIFIER genType normalize(genType const& x)
        {
            GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'normalize' accepts only floating-point inputs");

            return x < genType(0) ? genType(-1) : genType(1);
        }
    */
    template <length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER vec<L, T, Q> normalize( vec<L, T, Q> const &x )
    {
        GLM_STATIC_ASSERT( std::numeric_limits<T>::is_iec559, "'normalize' accepts only floating-point inputs" );

        return detail::compute_normalize<L, T, Q, detail::is_aligned<Q>::value>::call( x );
    }

    // faceforward
    template <typename genType>
    GLM_FUNC_QUALIFIER genType faceforward( genType const &N, genType const &I, genType const &Nref )
    {
        return dot( Nref, I ) < static_cast<genType>( 0 ) ? N : -N;
    }

    template <length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER vec<L, T, Q> faceforward( vec<L, T, Q> const &N, vec<L, T, Q> const &I, vec<L, T, Q> const &Nref )
    {
        return detail::compute_faceforward<L, T, Q, detail::is_aligned<Q>::value>::call( N, I, Nref );
    }

    // reflect
    template <typename genType>
    GLM_FUNC_QUALIFIER genType reflect( genType const &I, genType const &N )
    {
        return I - N * dot( N, I ) * genType( 2 );
    }

    template <length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER vec<L, T, Q> reflect( vec<L, T, Q> const &I, vec<L, T, Q> const &N )
    {
        return detail::compute_reflect<L, T, Q, detail::is_aligned<Q>::value>::call( I, N );
    }

    // refract
    template <typename genType>
    GLM_FUNC_QUALIFIER genType refract( genType const &I, genType const &N, genType eta )
    {
        GLM_STATIC_ASSERT( std::numeric_limits<genType>::is_iec559, "'refract' accepts only floating-point inputs" );
        genType const dotValue( dot( N, I ) );
        genType const k( static_cast<genType>( 1 ) - eta * eta * ( static_cast<genType>( 1 ) - dotValue * dotValue ) );

        return ( eta * I - ( eta * dotValue + sqrt( k ) ) * N ) * static_cast<genType>( k >= static_cast<genType>( 0 ) );
    }

    template <length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER vec<L, T, Q> refract( vec<L, T, Q> const &I, vec<L, T, Q> const &N, T eta )
    {
        GLM_STATIC_ASSERT( std::numeric_limits<T>::is_iec559, "'refract' accepts only floating-point inputs" );

        return detail::compute_refract<L, T, Q, detail::is_aligned<Q>::value>::call( I, N, eta );
    }
} // namespace glm

#if GLM_CONFIG_SIMD == GLM_ENABLE
#    include "func_geometric_simd.inl"
#endif

/// @ref core
/// @file glm/detail/func_exponential.inl

#include "../vector_relational.hpp"
#include "_vectorize.hpp"
#include <cassert>
#include <cmath>
#include <limits>

namespace glm
{
    namespace detail
    {
#if GLM_HAS_CXX11_STL
        using std::log2;
#else
        template <typename genType>
        GLM_FUNC_QUALIFIER genType log2( genType Value )
        {
            return std::log( Value ) * static_cast<genType>( 1.4426950408889634073599246810019 );
        }
#endif

        template <length_t L, typename T, qualifier Q, bool isFloat, bool Aligned>
        struct compute_log2
        {
            GLM_FUNC_QUALIFIER static vec<L, T, Q> call( vec<L, T, Q> const &v )
            {
                GLM_STATIC_ASSERT( std::numeric_limits<T>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT,
                                   "'log2' only accept floating-point inputs. Include <glm/gtc/integer.hpp> for integer inputs." );

                return detail::functor1<vec, L, T, T, Q>::call( log2, v );
            }
        };

        template <length_t L, typename T, qualifier Q, bool Aligned>
        struct compute_sqrt
        {
            GLM_FUNC_QUALIFIER static vec<L, T, Q> call( vec<L, T, Q> const &x )
            {
                return detail::functor1<vec, L, T, T, Q>::call( std::sqrt, x );
            }
        };

        template <length_t L, typename T, qualifier Q, bool Aligned>
        struct compute_inversesqrt
        {
            GLM_FUNC_QUALIFIER static vec<L, T, Q> call( vec<L, T, Q> const &x )
            {
                return static_cast<T>( 1 ) / sqrt( x );
            }
        };

        template <length_t L, bool Aligned>
        struct compute_inversesqrt<L, float, lowp, Aligned>
        {
            GLM_FUNC_QUALIFIER static vec<L, float, lowp> call( vec<L, float, lowp> const &x )
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

    // pow
    using std::pow;
    template <length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER vec<L, T, Q> pow( vec<L, T, Q> const &base, vec<L, T, Q> const &exponent )
    {
        return detail::functor2<vec, L, T, Q>::call( pow, base, exponent );
    }

    // exp
    using std::exp;
    template <length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER vec<L, T, Q> exp( vec<L, T, Q> const &x )
    {
        return detail::functor1<vec, L, T, T, Q>::call( exp, x );
    }

    // log
    using std::log;
    template <length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER vec<L, T, Q> log( vec<L, T, Q> const &x )
    {
        return detail::functor1<vec, L, T, T, Q>::call( log, x );
    }

#if GLM_HAS_CXX11_STL
    using std::exp2;
#else
    // exp2, ln2 = 0.69314718055994530941723212145818f
    template <typename genType>
    GLM_FUNC_QUALIFIER genType exp2( genType x )
    {
        GLM_STATIC_ASSERT( std::numeric_limits<genType>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT,
                           "'exp2' only accept floating-point inputs" );

        return std::exp( static_cast<genType>( 0.69314718055994530941723212145818 ) * x );
    }
#endif

    template <length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER vec<L, T, Q> exp2( vec<L, T, Q> const &x )
    {
        return detail::functor1<vec, L, T, T, Q>::call( exp2, x );
    }

    // log2, ln2 = 0.69314718055994530941723212145818f
    template <typename genType>
    GLM_FUNC_QUALIFIER genType log2( genType x )
    {
        return log2( vec<1, genType>( x ) ).x;
    }

    template <length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER vec<L, T, Q> log2( vec<L, T, Q> const &x )
    {
        return detail::compute_log2<L, T, Q, std::numeric_limits<T>::is_iec559, detail::is_aligned<Q>::value>::call( x );
    }

    // sqrt
    using std::sqrt;
    template <length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER vec<L, T, Q> sqrt( vec<L, T, Q> const &x )
    {
        GLM_STATIC_ASSERT( std::numeric_limits<T>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT,
                           "'sqrt' only accept floating-point inputs" );

        return detail::compute_sqrt<L, T, Q, detail::is_aligned<Q>::value>::call( x );
    }

    // inversesqrt
    template <typename genType>
    GLM_FUNC_QUALIFIER genType inversesqrt( genType x )
    {
        return static_cast<genType>( 1 ) / sqrt( x );
    }

    template <length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER vec<L, T, Q> inversesqrt( vec<L, T, Q> const &x )
    {
        GLM_STATIC_ASSERT( std::numeric_limits<T>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT,
                           "'inversesqrt' only accept floating-point inputs" );

        return detail::compute_inversesqrt<L, T, Q, detail::is_aligned<Q>::value>::call( x );
    }
} // namespace glm

#if GLM_CONFIG_SIMD == GLM_ENABLE
#    include "func_exponential_simd.inl"
#endif
#include "_vectorize.hpp"
#include <cmath>
#include <limits>

namespace glm
{
    // radians
    template <typename genType>
    GLM_FUNC_QUALIFIER GLM_CONSTEXPR genType radians( genType degrees )
    {
        GLM_STATIC_ASSERT( std::numeric_limits<genType>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT,
                           "'radians' only accept floating-point input" );

        return degrees * static_cast<genType>( 0.01745329251994329576923690768489 );
    }

    template <length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, T, Q> radians( vec<L, T, Q> const &v )
    {
        return detail::functor1<vec, L, T, T, Q>::call( radians, v );
    }

    // degrees
    template <typename genType>
    GLM_FUNC_QUALIFIER GLM_CONSTEXPR genType degrees( genType radians )
    {
        GLM_STATIC_ASSERT( std::numeric_limits<genType>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT,
                           "'degrees' only accept floating-point input" );

        return radians * static_cast<genType>( 57.295779513082320876798154814105 );
    }

    template <length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, T, Q> degrees( vec<L, T, Q> const &v )
    {
        return detail::functor1<vec, L, T, T, Q>::call( degrees, v );
    }

    // sin
    using ::std::sin;

    template <length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER vec<L, T, Q> sin( vec<L, T, Q> const &v )
    {
        return detail::functor1<vec, L, T, T, Q>::call( sin, v );
    }

    // cos
    using std::cos;

    template <length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER vec<L, T, Q> cos( vec<L, T, Q> const &v )
    {
        return detail::functor1<vec, L, T, T, Q>::call( cos, v );
    }

    // tan
    using std::tan;

    template <length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER vec<L, T, Q> tan( vec<L, T, Q> const &v )
    {
        return detail::functor1<vec, L, T, T, Q>::call( tan, v );
    }

    // asin
    using std::asin;

    template <length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER vec<L, T, Q> asin( vec<L, T, Q> const &v )
    {
        return detail::functor1<vec, L, T, T, Q>::call( asin, v );
    }

    // acos
    using std::acos;

    template <length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER vec<L, T, Q> acos( vec<L, T, Q> const &v )
    {
        return detail::functor1<vec, L, T, T, Q>::call( acos, v );
    }

    // atan
    template <typename genType>
    GLM_FUNC_QUALIFIER genType atan( genType y, genType x )
    {
        GLM_STATIC_ASSERT( std::numeric_limits<genType>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT,
                           "'atan' only accept floating-point input" );

        return ::std::atan2( y, x );
    }

    template <length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER vec<L, T, Q> atan( vec<L, T, Q> const &y, vec<L, T, Q> const &x )
    {
        return detail::functor2<vec, L, T, Q>::call( ::std::atan2, y, x );
    }

    using std::atan;

    template <length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER vec<L, T, Q> atan( vec<L, T, Q> const &v )
    {
        return detail::functor1<vec, L, T, T, Q>::call( atan, v );
    }

    // sinh
    using std::sinh;

    template <length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER vec<L, T, Q> sinh( vec<L, T, Q> const &v )
    {
        return detail::functor1<vec, L, T, T, Q>::call( sinh, v );
    }

    // cosh
    using std::cosh;

    template <length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER vec<L, T, Q> cosh( vec<L, T, Q> const &v )
    {
        return detail::functor1<vec, L, T, T, Q>::call( cosh, v );
    }

    // tanh
    using std::tanh;

    template <length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER vec<L, T, Q> tanh( vec<L, T, Q> const &v )
    {
        return detail::functor1<vec, L, T, T, Q>::call( tanh, v );
    }

    // asinh
#if GLM_HAS_CXX11_STL
    using std::asinh;
#else
    template <typename genType>
    GLM_FUNC_QUALIFIER genType asinh( genType x )
    {
        GLM_STATIC_ASSERT( std::numeric_limits<genType>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT,
                           "'asinh' only accept floating-point input" );

        return ( x < static_cast<genType>( 0 )
                     ? static_cast<genType>( -1 )
                     : ( x > static_cast<genType>( 0 ) ? static_cast<genType>( 1 ) : static_cast<genType>( 0 ) ) ) *
               log( std::abs( x ) + sqrt( static_cast<genType>( 1 ) + x * x ) );
    }
#endif

    template <length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER vec<L, T, Q> asinh( vec<L, T, Q> const &v )
    {
        return detail::functor1<vec, L, T, T, Q>::call( asinh, v );
    }

    // acosh
#if GLM_HAS_CXX11_STL
    using std::acosh;
#else
    template <typename genType>
    GLM_FUNC_QUALIFIER genType acosh( genType x )
    {
        GLM_STATIC_ASSERT( std::numeric_limits<genType>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT,
                           "'acosh' only accept floating-point input" );

        if( x < static_cast<genType>( 1 ) )
            return static_cast<genType>( 0 );

        return log( x + sqrt( x * x - static_cast<genType>( 1 ) ) );
    }
#endif

    template <length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER vec<L, T, Q> acosh( vec<L, T, Q> const &v )
    {
        return detail::functor1<vec, L, T, T, Q>::call( acosh, v );
    }

    // atanh
#if GLM_HAS_CXX11_STL
    using std::atanh;
#else
    template <typename genType>
    GLM_FUNC_QUALIFIER genType atanh( genType x )
    {
        GLM_STATIC_ASSERT( std::numeric_limits<genType>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT,
                           "'atanh' only accept floating-point input" );

        if( std::abs( x ) >= static_cast<genType>( 1 ) )
            return 0;

        return static_cast<genType>( 0.5 ) * log( ( static_cast<genType>( 1 ) + x ) / ( static_cast<genType>( 1 ) - x ) );
    }
#endif

    template <length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER vec<L, T, Q> atanh( vec<L, T, Q> const &v )
    {
        return detail::functor1<vec, L, T, T, Q>::call( atanh, v );
    }
} // namespace glm

#if GLM_CONFIG_SIMD == GLM_ENABLE
#    include "func_trigonometric_simd.inl"
#endif
