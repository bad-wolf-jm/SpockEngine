#include "Resource.h"

namespace LTSE::Core
{

    // static fs::path a_ResourceRoot = "Deps\\LTSimulationEngine\\Resources";
    static fs::path a_ResourceRoot = "C:\\GitLab\\SpockEngine\\Resources";

    fs::path GetResourcePath( fs::path a_RelativePath ) { return a_ResourceRoot / a_RelativePath; }

} // namespace LTSE::Core