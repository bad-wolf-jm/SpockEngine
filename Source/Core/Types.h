#pragma once

#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>


// #include <glm/glm.hpp>
// #include <glm/gtx/dual_quaternion.hpp>
// #include <glm/gtc/type_ptr.hpp>

template <class T, typename std::underlying_type<T>::type fullValue> class EnumSet
{
    static_assert( std::is_enum<T>::value, "EnumSet type must be strongly typed enum" );

  public:
    typedef T Type; /**< @brief Enum type */
    typedef typename std::underlying_type<T>::type UnderlyingType;

    enum : UnderlyingType
    {
        FullValue = fullValue /**< All enum values together */
    };

    constexpr EnumSet()
        : value()
    {
    }
    constexpr EnumSet( const std::initializer_list<T> a_Init )
    {
        if( a_Init.size() == 0 )
        {
            value = 0;
            return;
        }
        UnderlyingType l_Value{};
        for( auto i : a_Init )
            l_Value |= ( (UnderlyingType)i );
        value = l_Value;
    }
    constexpr EnumSet( T value )
        : value( static_cast<UnderlyingType>( value ) )
    {
    }
    constexpr EnumSet( UnderlyingType value )
        : value( static_cast<UnderlyingType>( value ) & fullValue )
    {
    }

    constexpr bool operator==( EnumSet<T, fullValue> other ) const { return value == other.value; }

    constexpr bool operator!=( EnumSet<T, fullValue> other ) const { return !operator==( other ); }

    constexpr bool operator>=( EnumSet<T, fullValue> other ) const { return ( *this & other ) == other; }

    constexpr bool operator<=( EnumSet<T, fullValue> other ) const { return ( *this & other ) == *this; }

    constexpr EnumSet<T, fullValue> operator|( EnumSet<T, fullValue> other ) const { return EnumSet<T, fullValue>( value | other.value ); }

    constexpr EnumSet<T, fullValue> operator|( T other ) const { return EnumSet<T, fullValue>( value | other ); }

    EnumSet<T, fullValue> &operator|=( EnumSet<T, fullValue> other )
    {
        value |= other.value;
        return *this;
    }

    constexpr EnumSet<T, fullValue> operator&( EnumSet<T, fullValue> other ) const { return EnumSet<T, fullValue>( value & other.value ); }

    EnumSet<T, fullValue> &operator&=( EnumSet<T, fullValue> other )
    {
        value &= other.value;
        return *this;
    }

    constexpr EnumSet<T, fullValue> operator^( EnumSet<T, fullValue> other ) const { return EnumSet<T, fullValue>( value ^ other.value ); }

    EnumSet<T, fullValue> &operator^=( EnumSet<T, fullValue> other )
    {
        value ^= other.value;
        return *this;
    }

    constexpr EnumSet<T, fullValue> operator~() const { return EnumSet<T, fullValue>( fullValue & ~value ); }

    constexpr explicit operator bool() const { return value != 0; }

    constexpr explicit operator UnderlyingType() const { return value; }

    UnderlyingType value;
};
