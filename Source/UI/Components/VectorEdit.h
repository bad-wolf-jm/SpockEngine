#pragma once

#include "Component.h"

namespace SE::Core
{
    class UIVectorInputBase : public UIComponent
    {
      public:
        UIVectorInputBase( int aDimension );

        UIVectorInputBase( std::string const &aText );

        ImVec2 RequiredSize();

      protected:
        int         mDimension{};
        math::vec4  mValues{};
        math::vec4  mResetValues{};
        std::string mFormat = ".2f";

      protected:
        void PushStyles();
        void PopStyles();

        void DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core