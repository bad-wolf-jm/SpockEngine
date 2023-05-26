#include "Resource.h"

namespace SE::Core
{

    static fs::path a_ResourceRoot = "C:\\GitLab\\SpockEngine\\Source\\Scene\\Renderer";
    // static fs::path a_ResourceRoot = "C:\\GitLab\\SpockEngine\\Resources";

    fs::path GetResourcePath( fs::path a_RelativePath ) { return a_ResourceRoot / a_RelativePath; }

} // namespace SE::Core