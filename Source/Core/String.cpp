#include "Engine/Engine.h"

#include "File.h"
#include "String.h"

#include <codecvt>
#include <locale>

namespace SE::Core
{
    string_t ConvertStringForCoreclr( wchar_t *aCharacters )
    {
        std::wstring u16str( aCharacters );

        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>, wchar_t> convert;
        string_t                                                        utf8 = convert.to_bytes( u16str );

        return utf8;
    }

    std::wstring ConvertStringForCoreclr( const string_t &utf8 )
    {
        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>, wchar_t> convert;

        std::wstring utf16 = convert.from_bytes( utf8 );

        return utf16;
    }

    wchar_t *CopyCharactersForCoreClr( string_t const &aString )
    {
        return CopyCharactersForCoreClr( ConvertStringForCoreclr( aString ) );
    }

    wchar_t *CopyCharactersForCoreClr( std::wstring const &aString )
    {
        wchar_t *pszReturn = (wchar_t *)::CoTaskMemAlloc( ( aString.size() + 1 ) * sizeof( wchar_t ) );
        wcscpy( pszReturn, aString.c_str() );

        return pszReturn;
    }

} // namespace SE::Core