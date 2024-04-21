namespace numlua::core::details
{
    int  do_isabsolute( const char *path );
    void do_translate( char *value, const char sep );
    void do_normalize( lua_State *L, char *buffer, const char *path );
    int  do_getcwd( char *buffer, size_t size );
} // namespace numlua::core::details
