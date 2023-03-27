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
        int       mRowIndex;
        int       mLinkIndex;
        int       mSubLinkIndex;
        int       mEventIndex;
        int       mSubEventIndex;
        bool      mIsSubElement;
        int       mDiagnosicCount;
        ePassFail mLossPassFail;
        ePassFail mReflectancePassFail;
        void     *mLinkElement;
        void     *mPhysicalEvent;
        void     *mPeakTrace;
        void     *mDetectionTrace;
        void     *mAttributes;
        void     *mAcquisitionData;
        void     *mFiberInfo;
    };

    class UILinkElementTable : public UITable
    {
      public:
        UILinkElementTable();
        UILinkElementTable( UILinkElementTable const & ) = default;

        ~UILinkElementTable() = default;

        void Clear();
        void SetData( std::vector<sLinkElement> const &aData );
        void OnElementClicked( std::function<void( sLinkElement const & )> const &aOnRowClicked );

        std::vector<sLinkElement> GetElementsByIndex( uint32_t aElementIndex );

      protected:
        Ref<sStringColumn>  mIndex;
        Ref<sStringColumn>  mType;
        Ref<sStringColumn>  mChanged;
        Ref<sStringColumn>  mComment;
        Ref<sStringColumn>  mHidden;
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
        Ref<sStringColumn>  mEventSpan;
        Ref<sFloat64Column> mPositionTolerance;
        Ref<sFloat64Column> mLossError;
        Ref<sFloat64Column> mPeakPower;

        std::vector<sLinkElement> mEventDataVector;

      private:
        std::function<void( sLinkElement const & )> mOnElementClicked;
        void                                        UILinkElementTable::DoSelectElement( uint32_t aRow );
    };
} // namespace SE::OtdrEditor