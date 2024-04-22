#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

namespace numlua::core::details
{
    int  do_isabsolute( const char *path );
    void do_translate( char *value, const char sep );
    void do_normalize( lua_State *L, char *buffer, const char *path );
    int  do_getcwd( char *buffer, size_t size );
    int  do_absolutetype( const char *path );
} // namespace numlua::core::details
