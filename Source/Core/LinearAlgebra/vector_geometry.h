#include "core.h"

#include <cmath>

namespace numlua::linalg
{
    namespace detail
    {
        template <length_t L, typename T>
        struct compute_length
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF static T call( vect<L, T> const &v )
            {
                return sqrt( dot( v, v ) );
            }
        };

        template <length_t L, typename T>
        struct compute_distance
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF static T call( vect<L, T> const &p0, vect<L, T> const &p1 )
            {
                return length( p1 - p0 );
            }
        };

        template <typename V, typename T>
        struct compute_dot
        {
        };

        template <typename T>
        struct compute_dot<vect<2, T>, T>
        {
            LINALG_FUNCTION static T call( vect<2, T> const &a, vect<2, T> const &b )
            {
                vect<2, T> tmp( a * b );

                return tmp.x + tmp.y;
            }
        };

        template <typename T>
        struct compute_dot<vect<3, T>, T>
        {
            LINALG_FUNCTION static T call( vect<3, T> const &a, vect<3, T> const &b )
            {
                vect<3, T> tmp( a * b );

                return tmp.x + tmp.y + tmp.z;
            }
        };

        template <typename T>
        struct compute_dot<vect<4, T>, T>
        {
            LINALG_FUNCTION static T call( vect<4, T> const &a, vect<4, T> const &b )
            {
                vect<4, T> tmp( a * b );

                return ( tmp.x + tmp.y ) + ( tmp.z + tmp.w );
            }
        };

        template <typename T>
        struct compute_cross
        {
            LINALG_FUNCTION static vect<3, T> call( vect<3, T> const &x, vect<3, T> const &y )
            {
                return vect<3, T>( x.y * y.z - y.y * x.z, x.z * y.x - y.z * x.x, x.x * y.y - y.x * x.y );
            }

            LINALG_FUNCTION static vect<4, T> call( vect<4, T> const &x, vect<4, T> const &y )
            {
                return vect<4, T>( x.y * y.z - y.y * x.z, x.z * y.x - y.z * x.x, x.x * y.y - y.x * x.y, 0.0f );
            }
        };

        template <length_t L, typename T>
        struct compute_normalize
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF static vect<L, T> call( vect<L, T> const &v )
            {
                return v * inversesqrt( dot( v, v ) );
            }
        };

        template <length_t L, typename T>
        struct compute_faceforward
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF static vect<L, T> call( vect<L, T> const &N, vect<L, T> const &I, vect<L, T> const &Nref )
            {
                return dot( Nref, I ) < static_cast<T>( 0 ) ? N : -N;
            }
        };

        template <length_t L, typename T>
        struct compute_reflect
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF static vect<L, T> call( vect<L, T> const &I, vect<L, T> const &N )
            {
                return I - N * dot( N, I ) * static_cast<T>( 2 );
            }
        };

        template <length_t L, typename T>
        struct compute_refract
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF static vect<L, T> call( vect<L, T> const &I, vect<L, T> const &N, T eta )
            {
                T const          dotValue( dot( N, I ) );
                T const          k( static_cast<T>( 1 ) - eta * eta * ( static_cast<T>( 1 ) - dotValue * dotValue ) );
                vect<L, T> const Result =
                    ( k >= static_cast<T>( 0 ) ) ? ( eta * I - ( eta * dotValue + std::sqrt( k ) ) * N ) : vect<L, T>( 0 );
                return Result;
            }
        };

        using std::log2;
        template <length_t L, typename T, bool isFloat>
        struct compute_log2
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF static vect<L, T> call( vect<L, T> const &v )
            {
                return detail::functor1<vect, L, T, T>::call( log2, v );
            }
        };

        template <length_t L, typename T>
        struct compute_sqrt
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF static vect<L, T> call( vect<L, T> const &x )
            {
                return detail::functor1<vect, L, T, T>::call( std::sqrt, x );
            }
        };

        template <length_t L, typename T>
        struct compute_inversesqrt
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF static vect<L, T> call( vect<L, T> const &x )
            {
                return static_cast<T>( 1 ) / sqrt( x );
            }
        };
    } // namespace detail

    // length
    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF _Ty length( _Ty x )
    {
        return abs( x );
    }

    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF T length( vect<L, T> const &v )
    {
        return detail::compute_length<L, T>::call( v );
    }

    // distance
    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF _Ty distance( _Ty const &p0, _Ty const &p1 )
    {
        return length( p1 - p0 );
    }

    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF T distance( vect<L, T> const &p0, vect<L, T> const &p1 )
    {
        return detail::compute_distance<L, T>::call( p0, p1 );
    }

    // dot
    template <typename T>
    LINALG_FUNCTION T dot( T x, T y )
    {
        return x * y;
    }

    template <length_t L, typename T>
    LINALG_FUNCTION T dot( vect<L, T> const &x, vect<L, T> const &y )
    {
        return detail::compute_dot<vect<L, T>, T>::call( x, y );
    }

    // cross
    template <typename T>
    LINALG_FUNCTION vect<3, T> cross( vect<3, T> const &x, vect<3, T> const &y )
    {
        return detail::compute_cross<T>::call( x, y );
    }

    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> normalize( vect<L, T> const &x )
    {
        return detail::compute_normalize<L, T>::call( x );
    }

    // faceforward
    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF _Ty faceforward( _Ty const &N, _Ty const &I, _Ty const &Nref )
    {
        return dot( Nref, I ) < static_cast<_Ty>( 0 ) ? N : -N;
    }

    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> faceforward( vect<L, T> const &N, vect<L, T> const &I, vect<L, T> const &Nref )
    {
        return detail::compute_faceforward<L, T>::call( N, I, Nref );
    }

    // reflect
    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF _Ty reflect( _Ty const &I, _Ty const &N )
    {
        return I - N * dot( N, I ) * _Ty( 2 );
    }

    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> reflect( vect<L, T> const &I, vect<L, T> const &N )
    {
        return detail::compute_reflect<L, T>::call( I, N );
    }

    // refract
    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF _Ty refract( _Ty const &I, _Ty const &N, _Ty eta )
    {
        _Ty const dotValue( dot( N, I ) );
        _Ty const k( static_cast<_Ty>( 1 ) - eta * eta * ( static_cast<_Ty>( 1 ) - dotValue * dotValue ) );

        return ( eta * I - ( eta * dotValue + sqrt( k ) ) * N ) * static_cast<_Ty>( k >= static_cast<_Ty>( 0 ) );
    }

    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> refract( vect<L, T> const &I, vect<L, T> const &N, T eta )
    {
        return detail::compute_refract<L, T>::call( I, N, eta );
    }

    // pow
    using std::pow;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> pow( vect<L, T> const &base, vect<L, T> const &exponent )
    {
        return detail::functor2<vect, L, T>::call( pow, base, exponent );
    }

    // exp
    using std::exp;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> exp( vect<L, T> const &x )
    {
        return detail::functor1<vect, L, T, T>::call( exp, x );
    }

    // log
    using std::log;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> log( vect<L, T> const &x )
    {
        return detail::functor1<vect, L, T, T>::call( log, x );
    }

    using std::exp2;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> exp2( vect<L, T> const &x )
    {
        return detail::functor1<vect, L, T, T>::call( exp2, x );
    }

    using std::log2;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> log2( vect<L, T> const &x )
    {
        return detail::compute_log2<L, T, std::numeric_limits<T>::is_iec559>::call( x );
    }

    // sqrt
    using std::sqrt;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> sqrt( vect<L, T> const &x )
    {
        return detail::compute_sqrt<L, T>::call( x );
    }

    // inversesqrt
    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF _Ty inversesqrt( _Ty x )
    {
        return static_cast<_Ty>( 1 ) / sqrt( x );
    }

    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> inversesqrt( vect<L, T> const &x )
    {
        return detail::compute_inversesqrt<L, T>::call( x );
    }

    // radians
    template <typename _Ty>
    LINALG_FUNCTION _Ty radians( _Ty degrees )
    {
        return degrees * static_cast<_Ty>( 0.01745329251994329576923690768489 );
    }

    template <length_t L, typename T>
    LINALG_FUNCTION vect<L, T> radians( vect<L, T> const &v )
    {
        return detail::functor1<vect, L, T, T>::call( radians, v );
    }

    // degrees
    template <typename _Ty>
    LINALG_FUNCTION _Ty degrees( _Ty radians )
    {
        return radians * static_cast<_Ty>( 57.295779513082320876798154814105 );
    }

    template <length_t L, typename T>
    LINALG_FUNCTION vect<L, T> degrees( vect<L, T> const &v )
    {
        return detail::functor1<vect, L, T, T>::call( degrees, v );
    }

    // sin
    using ::std::sin;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> sin( vect<L, T> const &v )
    {
        return detail::functor1<vect, L, T, T>::call( sin, v );
    }

    // cos
    using std::cos;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> cos( vect<L, T> const &v )
    {
        return detail::functor1<vect, L, T, T>::call( cos, v );
    }

    // tan
    using std::tan;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> tan( vect<L, T> const &v )
    {
        return detail::functor1<vect, L, T, T>::call( tan, v );
    }

    // asin
    using std::asin;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> asin( vect<L, T> const &v )
    {
        return detail::functor1<vect, L, T, T>::call( asin, v );
    }

    // acos
    using std::acos;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> acos( vect<L, T> const &v )
    {
        return detail::functor1<vect, L, T, T>::call( acos, v );
    }

    // atan
    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF _Ty atan( _Ty y, _Ty x )
    {
        return ::std::atan2( y, x );
    }

    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> atan( vect<L, T> const &y, vect<L, T> const &x )
    {
        return detail::functor2<vect, L, T>::call( ::std::atan2, y, x );
    }

    using std::atan;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> atan( vect<L, T> const &v )
    {
        return detail::functor1<vect, L, T, T>::call( atan, v );
    }

    // sinh
    using std::sinh;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> sinh( vect<L, T> const &v )
    {
        return detail::functor1<vect, L, T, T>::call( sinh, v );
    }

    // cosh
    using std::cosh;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> cosh( vect<L, T> const &v )
    {
        return detail::functor1<vect, L, T, T>::call( cosh, v );
    }

    // tanh
    using std::tanh;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> tanh( vect<L, T> const &v )
    {
        return detail::functor1<vect, L, T, T>::call( tanh, v );
    }

    // asinh
    using std::asinh;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> asinh( vect<L, T> const &v )
    {
        return detail::functor1<vect, L, T, T>::call( asinh, v );
    }

    // acosh
    using std::acosh;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> acosh( vect<L, T> const &v )
    {
        return detail::functor1<vect, L, T, T>::call( acosh, v );
    }

    // atanh
    using std::atanh;
    template <length_t L, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF vect<L, T> atanh( vect<L, T> const &v )
    {
        return detail::functor1<vect, L, T, T>::call( atanh, v );
    }
} // namespace numlua::linalg
