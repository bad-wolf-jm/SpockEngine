#pragma once
#include <string>

// #include "Core/EntityCollection/Collection.h"
#include "Core/Memory.h"

// #include "OtdrScene/OtdrScene.h"

namespace SE::OtdrEditor
{
    using namespace SE::Core;

    struct LockComponent
    {
        bool mLocked = true;
    };

    class MonoClassHierarchy
    {
      public:
        MonoClassHierarchy() = default;
        ~MonoClassHierarchy() = default;

        void Display( int32_t width, int32_t height );

      private:
        void DisplayNode( DotNetClass &aClass, float a_Width );

      private:
        std::vector<std::string> mClassNames;
    };

} // namespace SE::Editor