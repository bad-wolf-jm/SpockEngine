#pragma once

#include <filesystem>
#include <string>

namespace SE::Core
{
    using char_t   = char;
    using string_t = std::string;
    using path_t   = std::filesystem::path;

    // string_t ConvertStringForCoreclr( wchar_t *aCharacters );
    // std::wstring ConvertStringForCoreclr( const string_t &utf8 );

    // wchar_t* CopyCharactersForCoreClr(string_t const& aString);
    // wchar_t* CopyCharactersForCoreClr(std::wstring const& aString);
} // namespace SE::Core