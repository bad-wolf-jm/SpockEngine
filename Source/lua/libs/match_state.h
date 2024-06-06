
#pragma once

#include "lua/lua_api.h"

/*
** maximum number of captures that a pattern can do during
** pattern-matching. This limit is arbitrary, but must fit in
** an unsigned char.
*/
#if !defined( LUA_MAXCAPTURES )
#    define LUA_MAXCAPTURES 32
#endif
#define L_ESC    '%'
#define SPECIALS "^$*+?.([%-"
/* maximum recursion depth for 'match' */
#if !defined( MAXCCALLS )
#    define MAXCCALLS 200
#endif
namespace numlua::core::lib
{
    struct match_state_t
    {
        const char   *src_init; /* init of source string */
        const char   *src_end;  /* end ('\0') of source string */
        const char   *p_end;    /* end ('\0') of pattern */
        lua::state_t *L;
        int           matchdepth; /* control for recursive depth (to avoid C stack overflow) */
        unsigned char level;      /* total number of captures (finished or unfinished) */
        struct
        {
            const char *init;
            ptrdiff_t   len;
        } capture[LUA_MAXCAPTURES];

        int         check_capture( int l );
        int         capture_to_close();
        const char *class_end( const char *p );
        int         single_match( const char *s, const char *p, const char *ep );
        const char *match_balance( const char *s, const char *p );
        const char *max_expand( const char *s, const char *p, const char *ep );
        const char *min_expand( const char *s, const char *p, const char *ep );
        const char *start_capture( const char *s, const char *p, int what );
        const char *end_capture( const char *s, const char *p );
        const char *match_capture( const char *s, int l );
        const char *match( const char *s, const char *p );
        size_t      get_one_capture( int i, const char *s, const char *e, const char **cap );
        void        push_one_capture( int i, const char *s, const char *e );
        int         push_captures( const char *s, const char *e );
        void        add_s( luaL_Buffer *b, const char *s, const char *e );
        int         add_value( luaL_Buffer *b, const char *s, const char *e, int tr );
        void        prepstate( lua::state_t *L, const char *s, size_t ls, const char *p, size_t lp );
        void        reprepstate();
    };
    int str_format( lua_State *L );
    int str_find( lua_State *L );
    int gmatch( lua_State *L );
    int str_gsub( lua_State *L );
    int str_match( lua_State *L );
} // namespace numlua::core::lib
