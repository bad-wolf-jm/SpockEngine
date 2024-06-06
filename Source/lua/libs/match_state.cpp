
/*
** $Id: lstrlib.c $
** Standard library for string operations and pattern-matching
** See Copyright Notice in lua.h
*/

// #define lstrlib_c
// #define LUA_LIB
#include "lua/libs/match_state.h"
#include "lua/libs/utilities.h"
#include "lua/lua_api.h"

#include <ctype.h>
#include <float.h>
#include <limits.h>
#include <locale.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* macro to 'unsign' a character */
#define uchar( c ) ( (unsigned char)( c ) )

/*
** Some sizes are better limited to fit in 'int', but must also fit in
** 'size_t'. (We assume that 'lua_Integer' cannot be smaller than 'int'.)
*/
#define MAX_SIZET ( (size_t)( ~(size_t)0 ) )

#define MAXSIZE ( sizeof( size_t ) < sizeof( int ) ? MAX_SIZET : (size_t)( INT_MAX ) )

namespace numlua::core::lib
{
#define CAP_UNFINISHED ( -1 )
#define CAP_POSITION   ( -2 )

    static int match_class( int c, int cl )
    {
        int res;
        switch( tolower( cl ) )
        {
            // clang-format off
        case 'a': res = isalpha( c ); break;
        case 'c': res = iscntrl( c ); break;
        case 'd': res = isdigit( c ); break;
        case 'g': res = isgraph( c ); break;
        case 'l': res = islower( c ); break;
        case 'p': res = ispunct( c ); break;
        case 's': res = isspace( c ); break;
        case 'u': res = isupper( c ); break;
        case 'w': res = isalnum( c ); break;
        case 'x': res = isxdigit( c ); break;
        case 'z': res = ( c == 0 ); break; /* deprecated option */
        default: return ( cl == c );
            // clang-format on
        }
        return ( islower( cl ) ? res : !res );
    }

    static int match_bracket_class( int c, const char *p, const char *ec )
    {
        int sig = 1;
        if( *( p + 1 ) == '^' )
        {
            sig = 0;
            p++; /* skip the '^' */
        }

        while( ++p < ec )
        {
            if( *p == L_ESC )
            {
                p++;
                if( match_class( c, uchar( *p ) ) )
                    return sig;
            }
            else if( ( *( p + 1 ) == '-' ) && ( p + 2 < ec ) )
            {
                p += 2;
                if( uchar( *( p - 2 ) ) <= c && c <= uchar( *p ) )
                    return sig;
            }
            else if( uchar( *p ) == c )
            {
                return sig;
            }
        }

        return !sig;
    }

    int match_state_t::check_capture( int l )
    {
        l -= '1';
        if( l < 0 || l >= level || capture[l].len == CAP_UNFINISHED )
            return lua::error( L, "invalid capture index %%%d", l + 1 );
        return l;
    }

    int match_state_t::capture_to_close()
    {
        int level = level;
        for( level--; level >= 0; level-- )
            if( capture[level].len == CAP_UNFINISHED )
                return level;

        return lua::error( L, "invalid pattern capture" );
    }

    const char *match_state_t::class_end( const char *p )
    {
        switch( *p++ )
        {
        case L_ESC:
        {
            if( p == p_end )
                lua::error( L, "malformed pattern (ends with '%%')" );
            return p + 1;
        }
        case '[':
        {
            if( *p == '^' )
                p++;
            do
            {
                /* look for a ']' */
                if( p == p_end )
                    lua::error( L, "malformed pattern (missing ']')" );

                if( *( p++ ) == L_ESC && p < p_end )
                    p++; /* skip escapes (e.g. '%]') */
            } while( *p != ']' );
            return p + 1;
        }
        default:
        {
            return p;
        }
        }
    }

    int match_state_t::single_match( const char *s, const char *p, const char *ep )
    {
        if( s >= src_end )
            return 0;
        else
        {
            int c = uchar( *s );
            switch( *p )
            {
            case '.':
                return 1; /* matches any char */
            case L_ESC:
                return match_class( c, uchar( *( p + 1 ) ) );
            case '[':
                return match_bracket_class( c, p, ep - 1 );
            default:
                return ( uchar( *p ) == c );
            }
        }
    }

    const char *match_state_t::match_balance( const char *s, const char *p )
    {
        if( p >= p_end - 1 )
            lua::error( L, "malformed pattern (missing arguments to '%%b')" );

        if( *s != *p )
        {
            return NULL;
        }
        else
        {
            int b    = *p;
            int e    = *( p + 1 );
            int cont = 1;
            while( ++s < src_end )
            {
                if( *s == e )
                {
                    if( --cont == 0 )
                        return s + 1;
                }
                else if( *s == b )
                    cont++;
            }
        }

        return NULL; /* string ends out of balance */
    }

    const char *match_state_t::max_expand( const char *s, const char *p, const char *ep )
    {
        ptrdiff_t i = 0; /* counts maximum expand for item */
        while( single_match( s + i, p, ep ) )
            i++;
        /* keeps trying to match with the maximum repetitions */
        while( i >= 0 )
        {
            const char *res = match( ( s + i ), ep + 1 );
            if( res )
                return res;
            i--; /* else didn't match; reduce 1 repetition to try again */
        }
        return NULL;
    }

    const char *match_state_t::min_expand( const char *s, const char *p, const char *ep )
    {
        for( ;; )
        {
            const char *res = match( s, ep + 1 );
            if( res != NULL )
                return res;
            else if( single_match( s, p, ep ) )
                s++; /* try with one more repetition */
            else
                return NULL;
        }
    }

    const char *match_state_t::start_capture( const char *s, const char *p, int what )
    {
        const char *res;
        int         level = level;
        if( level >= LUA_MAXCAPTURES )
            lua::error( L, "too many captures" );
        capture[level].init = s;
        capture[level].len  = what;
        level               = level + 1;
        if( ( res = match( s, p ) ) == NULL ) /* match failed? */
            level--;                          /* undo capture */
        return res;
    }

    const char *match_state_t::end_capture( const char *s, const char *p )
    {
        int         l = capture_to_close();
        const char *res;
        capture[l].len = s - capture[l].init; /* close capture */
        if( ( res = match( s, p ) ) == NULL ) /* match failed? */
            capture[l].len = CAP_UNFINISHED;  /* undo capture */
        return res;
    }

    const char *match_state_t::match_capture( const char *s, int l )
    {
        size_t len;
        l   = check_capture( l );
        len = capture[l].len;
        if( (size_t)( src_end - s ) >= len && memcmp( capture[l].init, s, len ) == 0 )
            return s + len;
        else
            return NULL;
    }

    const char *match_state_t::match( const char *s, const char *p )
    {
        if( matchdepth-- == 0 )
            lua::error( L, "pattern too complex" );

    init: /* using goto's to optimize tail recursion */
        if( p != p_end )
        {
            /* end of pattern? */
            switch( *p )
            {
            case '(':
            {                           /* start capture */
                if( *( p + 1 ) == ')' ) /* position capture? */
                    s = start_capture( s, p + 2, CAP_POSITION );
                else
                    s = start_capture( s, p + 1, CAP_UNFINISHED );
                break;
            }
            case ')':
            { /* end capture */
                s = end_capture( s, p + 1 );
                break;
            }
            case '$':
            {
                if( ( p + 1 ) != p_end )         /* is the '$' the last char in pattern? */
                    goto dflt;                   /* no; go to default */
                s = ( s == src_end ) ? s : NULL; /* check end of string */
                break;
            }
            case L_ESC:
            { /* escaped sequences not in the format class[*+?-]? */
                switch( *( p + 1 ) )
                {
                case 'b':
                { /* balanced string? */
                    s = match_balance( s, p + 2 );
                    if( s != NULL )
                    {
                        p += 4;
                        goto init; /* return match(ms, s, p + 4); */
                    } /* else fail (s == NULL) */
                    break;
                }
                case 'f':
                { /* frontier? */
                    const char *ep;
                    char        previous;
                    p += 2;
                    if( *p != '[' )
                        lua::error( L, "missing '[' after '%%f' in pattern" );
                    ep       = class_end( p ); /* points to what is next */
                    previous = ( s == src_init ) ? '\0' : *( s - 1 );
                    if( !match_bracket_class( uchar( previous ), p, ep - 1 ) && match_bracket_class( uchar( *s ), p, ep - 1 ) )
                    {
                        p = ep;
                        goto init; /* return match(ms, s, ep); */
                    }
                    s = NULL; /* match failed */
                    break;
                }
                case '0':
                case '1':
                case '2':
                case '3':
                case '4':
                case '5':
                case '6':
                case '7':
                case '8':
                case '9':
                { /* capture results (%0-%9)? */
                    s = match_capture( s, uchar( *( p + 1 ) ) );
                    if( s != NULL )
                    {
                        p += 2;
                        goto init; /* return match(ms, s, p + 2) */
                    }
                    break;
                }
                default:
                    goto dflt;
                }
                break;
            }
            default:
            dflt:
            {                                    /* pattern class plus optional suffix */
                const char *ep = class_end( p ); /* points to optional suffix */
                /* does not match at least once? */
                if( !single_match( s, p, ep ) )
                {
                    if( *ep == '*' || *ep == '?' || *ep == '-' )
                    { /* accept empty? */
                        p = ep + 1;
                        goto init; /* return match(ms, s, ep + 1); */
                    }
                    else          /* '+' or no suffix */
                        s = NULL; /* fail */
                }
                else
                { /* matched once */
                    switch( *ep )
                    { /* handle optional suffix */
                    case '?':
                    { /* optional */
                        const char *res;
                        if( ( res = match( s + 1, ep + 1 ) ) != NULL )
                            s = res;
                        else
                        {
                            p = ep + 1;
                            goto init; /* else return match(ms, s, ep + 1); */
                        }
                        break;
                    }
                    case '+': /* 1 or more repetitions */
                        s++;  /* 1 match already done */
                              /* FALLTHROUGH */
                    case '*': /* 0 or more repetitions */
                        s = max_expand( s, p, ep );
                        break;
                    case '-': /* 0 or more repetitions (minimum) */
                        s = min_expand( s, p, ep );
                        break;
                    default: /* no suffix */
                        s++;
                        p = ep;
                        goto init; /* return match(ms, s + 1, ep); */
                    }
                }
                break;
            }
            }
        }

        matchdepth++;
        return s;
    }
    /*
    ** get information about the i-th capture. If there are no captures
    ** and 'i==0', return information about the whole match, which
    ** is the range 's'..'e'. If the capture is a string, return
    ** its length and put its address in '*cap'. If it is an integer
    ** (a position), push it on the stack and return CAP_POSITION.
    */
    size_t match_state_t::get_one_capture( int i, const char *s, const char *e, const char **cap )
    {
        if( i >= level )
        {
            if( i != 0 )
                lua::error( L, "invalid capture index %%%d", i + 1 );
            *cap = s;
            return e - s;
        }
        else
        {
            ptrdiff_t capl = capture[i].len;
            *cap           = capture[i].init;
            if( capl == CAP_UNFINISHED )
                lua::error( L, "unfinished capture" );
            else if( capl == CAP_POSITION )
                lua_pushinteger( L, ( capture[i].init - src_init ) + 1 );
            return capl;
        }
    }

    /*
    ** Push the i-th capture on the stack.
    */
    void match_state_t::push_one_capture( int i, const char *s, const char *e )
    {
        const char *cap;
        ptrdiff_t   l = get_one_capture( i, s, e, &cap );
        if( l != CAP_POSITION )
            lua_pushlstring( L, cap, l );
        /* else position was already pushed */
    }

    int match_state_t::push_captures( const char *s, const char *e )
    {
        int i;
        int nlevels = ( level == 0 && s ) ? 1 : level;
        luaL_checkstack( L, nlevels, "too many captures" );
        for( i = 0; i < nlevels; i++ )
            push_one_capture( i, s, e );
        return nlevels; /* number of strings pushed */
    }

    void match_state_t::add_s( luaL_Buffer *b, const char *s, const char *e )
    {
        size_t      l;
        const char *news = lua_tolstring( L, 3, &l );
        const char *p;
        while( ( p = (char *)memchr( news, L_ESC, l ) ) != NULL )
        {
            luaL_addlstring( b, news, p - news );
            p++;              /* skip ESC */
            if( *p == L_ESC ) /* '%%' */
                luaL_addchar( b, *p );
            else if( *p == '0' ) /* '%0' */
                luaL_addlstring( b, s, e - s );
            else if( isdigit( uchar( *p ) ) )
            {
                /* '%n' */
                const char *cap;
                ptrdiff_t   resl = get_one_capture( *p - '1', s, e, &cap );
                if( resl == CAP_POSITION )
                    luaL_addvalue( b ); /* add position to accumulated result */
                else
                    luaL_addlstring( b, cap, resl );
            }
            else
            {
                lua::error( L, "invalid use of '%c' in replacement string", L_ESC );
            }
            l -= p + 1 - news;
            news = p + 1;
        }
        luaL_addlstring( b, news, l );
    }

    /*
    ** Add the replacement value to the string buffer 'b'.
    ** Return true if the original string was changed. (Function calls and
    ** table indexing resulting in nil or false do not change the subject.)
    */
    int match_state_t::add_value( luaL_Buffer *b, const char *s, const char *e, int tr )
    {
        switch( tr )
        {
        case LUA_TFUNCTION:
        { /* call the function */
            int n;
            lua::push( L, lua::stack_index_t{ 3 } ); /* push the function */
            n = push_captures( s, e );               /* all captures as arguments */
            lua_call( L, n, 1 );                     /* call it */
            break;
        }
        case LUA_TTABLE:
        {                                /* index the table */
            push_one_capture( 0, s, e ); /* first capture is the index */
            lua_gettable( L, 3 );
            break;
        }
        default:
        {                     /* LUA_TNUMBER or LUA_TSTRING */
            add_s( b, s, e ); /* add value to the buffer */
            return 1;         /* something changed */
        }
        }
        if( !lua_toboolean( L, -1 ) )
        {                                   /* nil or false? */
            lua_pop( L, 1 );                /* remove value */
            luaL_addlstring( b, s, e - s ); /* keep original text */
            return 0;                       /* no changes */
        }
        else if( !lua_isstring( L, -1 ) )
            return lua::error( L, "invalid replacement value (a %s)", luaL_typename( L, -1 ) );
        else
        {
            luaL_addvalue( b ); /* add result to accumulator */
            return 1;           /* something changed */
        }
    }
    //

    void match_state_t::prepstate( lua::state_t *L_, const char *s, size_t ls, const char *p, size_t lp )
    {
        L          = L_;
        matchdepth = MAXCCALLS;
        src_init   = s;
        src_end    = s + ls;
        p_end      = p + lp;
    }

    void match_state_t::reprepstate()
    {
        level = 0;
        lua_assert( matchdepth == MAXCCALLS );
    }
} // namespace numlua::core::lib
