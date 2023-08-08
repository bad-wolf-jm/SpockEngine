#include "Engine/Engine.h"

#include "File.h"
#include "Profiling/BlockTimer.h"
#include "String.h"

#include <codecvt>
#include <locale>

static std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>, wchar_t> gConverter;

namespace SE::Core
{
    string_t ConvertStringForCoreclr( wchar_t *aCharacters )
    {
        SE_PROFILE_FUNCTION();

        std::wstring u16str( aCharacters );
        string_t     utf8 = gConverter.to_bytes( u16str );

        return utf8;
    }

    std::wstring ConvertStringForCoreclr( const string_t &utf8 )
    {
        std::wstring utf16 = gConverter.from_bytes( utf8 );

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