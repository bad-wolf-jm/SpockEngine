#include "MonoScriptMethod.h"

namespace SE::Core
{
    MonoScriptMehod::MonoScriptMehod( MonoMethod *aMonoMethod, MonoScriptInstance *aInstance )
        : mMonoMethod{ aMonoMethod }
        , mInstance{ aInstance }
    {
    }

} // namespace SE::Core