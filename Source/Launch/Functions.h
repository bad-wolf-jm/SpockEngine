#pragma once

namespace SE::OtdrEditor
{
    typedef bool(*RenderUIFn)(float aTs);
    typedef bool(*RenderMenuFn)();
    typedef void(*UpdateFn)(float aTs);
    typedef void(*RenderSceneFn)();
} // namespace SE::OtdrEditor