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
        int                mRowIndex;
        int                mLinkIndex;
        int                mEventIndex;
        eLinkElementType   mType;
        eLinkElementStatus mStatus;
        int                mDiagnosicCount;
        double             mPosition;
        double             mWavelength;
        bool               mIsTwoByNSplitter;
        int                mSplitterRatio;
        double             mRatio;
        bool               mChanged;
        double             mPreviousFiberSectionA;
        double             mPreviousFiberSectionB;
        double             mPreviousFiberSectionLength;
        double             mPreviousFiberSectionLoss;
        double             mPreviousFiberSectionAttenuation;
        double             mLoss;
        double             mLossError;
        double             mReflectance;
        double             mCurveLevel;
        double             mPositionTolerance;
        eEventType         mEventType;
        eEventStatus       mEventStatus;
        eReflectanceType   mReflectanceType;
        ePassFail          mLossPassFail;
        ePassFail          mReflectancePassFail;
    };

    class UILinkElementTable : public UITable
    {
      public:
        UILinkElementTable();
        UILinkElementTable( UILinkElementTable const & ) = default;

        ~UILinkElementTable() = default;

        void SetData(std::vector<sLinkElement> const& aData);

      protected:
        Ref<sStringColumn> mType;
        Ref<sFloat64Column> mStatus;
        Ref<sFloat64Column> mDiagnosicCount;
        Ref<sFloat64Column> mWavelength;
        Ref<sFloat64Column> mPositionColumn;
        Ref<sFloat64Column> mPreviousFiberSectionA;
        Ref<sFloat64Column> mPreviousFiberSectionB;
        Ref<sFloat64Column> mPreviousFiberSectionLength;
        Ref<sFloat64Column> mPreviousFiberSectionLoss;
        Ref<sFloat64Column> mPreviousFiberSectionAttenuation;
        Ref<sFloat64Column> mLoss;
        Ref<sFloat64Column> mLossPassFail;
        Ref<sFloat64Column> mLossError;
        Ref<sFloat64Column> mReflectance;
        Ref<sFloat64Column> mReflectancePassFail;
        Ref<sFloat64Column> mCurveLevelColumn;
        Ref<sFloat64Column> mPositionTolerance;
        Ref<sStringColumn> mEventType;
        Ref<sFloat64Column> mEventStatus;
        Ref<sStringColumn> mReflectanceType;

        std::vector<sLinkElement> mEventDataVector;
    };
} // namespace SE::OtdrEditor