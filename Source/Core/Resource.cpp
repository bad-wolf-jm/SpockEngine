#include "Resource.h"

namespace SE::Core
{

    // static path_t a_ResourceRoot = "Deps\\LTSimulationEngine\\Resources";
    static path_t a_ResourceRoot = "C:\\GitLab\\SpockEngine\\Resources";

    path_t GetResourcePath( path_t a_RelativePath ) { return a_ResourceRoot / a_RelativePath; }

} // namespace SE::Core