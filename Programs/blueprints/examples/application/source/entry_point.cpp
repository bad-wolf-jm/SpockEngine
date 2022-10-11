# include "application.h"
# include "platform.h"

# if PLATFORM(WINDOWS)
#     define NOMINMAX
#     define WIN32_LEAN_AND_MEAN
#     include <windows.h>
#     include <stdlib.h> // __argc, argv
# endif

int main(int argc, char** argv)
{
    return Main(argc, argv);
}
