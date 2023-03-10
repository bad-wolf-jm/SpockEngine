#pragma once

#include <functional>
#include <string>
#include <vector>

#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "UI/Components/Table.h"
#include "UI/Layouts/BoxLayout.h"
#include "UI/UI.h"

#include "Enums.h"

namespace SE::OtdrEditor
{
    using namespace SE::Core;
    struct sLinkElement
    {
        int              mRowIndex;
        int              mLinkIndex;
        int              mEventIndex;
        eLinkElementType mType;
        int32_t          mStatus;
        int              mDiagnosicCount;
        double           mPosition;
        double           mWavelength;
        bool             mIsTwoByNSplitter;
        int              mSplitterRatio;
        double           mRatio;
        bool             mChanged;
        double           mLoss;
        double           mReflectance;
        double           mCurveLevel;
        eEventType       mEventType;
        int32_t          mEventStatus;
        eReflectanceType mReflectanceType;
        ePassFail        mLossPassFail;
        ePassFail        mReflectancePassFail;
        void            *mPhysicalEvent;
        void            *mAttributes;
    };

    class UILinkElementTable : public UITable
    {
      public:
        UILinkElementTable();
        UILinkElementTable( UILinkElementTable const & ) = default;

        ~UILinkElementTable() = default;

        void SetData( std::vector<sLinkElement> const &aData );

      protected:
        Ref<sStringColumn>  mType;
        Ref<sStringColumn>  mStatus;
        Ref<sFloat64Column> mDiagnosicCount;
        Ref<sFloat64Column> mWavelength;
        Ref<sFloat64Column> mPositionColumn;
        Ref<sFloat64Column> mLoss;
        Ref<sFloat64Column> mLossPassFail;
        Ref<sFloat64Column> mReflectance;
        Ref<sFloat64Column> mReflectancePassFail;
        Ref<sFloat64Column> mCurveLevelColumn;
        Ref<sStringColumn>  mEventType;
        Ref<sStringColumn>  mEventStatus;
        Ref<sStringColumn>  mReflectanceType;

        std::vector<sLinkElement> mEventDataVector;
    };
} // namespace SE::OtdrEditor