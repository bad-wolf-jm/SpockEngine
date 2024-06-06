#include "lua/lua_api.h"

namespace numlua::core::lib::utilities
{
    /*
    ** translate a relative initial string position
    ** (negative means back from end): clip result to [1, inf).
    ** The length of any string in Lua must fit in a lua_Integer,
    ** so there are no overflows in the casts.
    ** The inverted comparison avoids a possible overflow
    ** computing '-pos'.
    */
    size_t translate_relative_position( lua_Integer pos, size_t len );

    /*
    ** Gets an optional ending string position from argument 'arg',
    ** with default value 'def'.
    ** Negative means back from end: clip result to [0, len]
    */
    size_t get_end_position( lua::state_t *L, int arg, lua_Integer def, size_t len );
    
    int format( lua::state_t *L );
} // namespace numlua::core::lib
