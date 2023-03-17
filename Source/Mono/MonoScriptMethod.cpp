#include "MonoScriptMethod.h"

namespace SE::Core
{
    DotNetMehod::DotNetMehod( MonoMethod *aMonoMethod, DotNetInstance *aInstance )
        : mMonoMethod{ aMonoMethod }
        , mInstance{ aInstance }
    {
    }

} // namespace SE::Core