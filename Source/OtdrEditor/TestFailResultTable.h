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
    struct sTestFailElement
    {
        std::string mTestName;
        std::string mTestDate;
        std::string mFilename;
        std::string mLinkElementIndex;
        std::string mSubLinkElementIndex;
        std::string mPhysicalEventIndex;
        double mLinkElementPosition;
        std::string mIsSubElement;
        double mWavelength;
        double mPhysicalEventPosition;
        std::string mSinglePulseTraceIndex;
        std::string mMessage;
    };

    class UITestFailResultTable : public UITable
    {
      public:
        UITestFailResultTable();
        UITestFailResultTable( UITestFailResultTable const & ) = default;

        ~UITestFailResultTable() = default;

        void Clear();
        void SetData( std::vector<sTestFailElement> const &aData );
        void OnElementClicked( std::function<void( sTestFailElement const & )> const &aOnRowClicked );

        // std::vector<sLinkElement> GetElementsByIndex( uint32_t aElementIndex );

      protected:
        Ref<sStringColumn> mTestName;
        Ref<sStringColumn> mTestDate;
        Ref<sStringColumn> mFilename;
        // Ref<sStringColumn> mLinkElementIndex;
        // Ref<sStringColumn> mSubLinkElementIndex;
        // Ref<sStringColumn> mPhysicalEventIndex;
        Ref<sFloat64Column> mLinkElementPosition;
        // Ref<sStringColumn> mIsSubElement;
        Ref<sFloat64Column> mWavelength;
        Ref<sFloat64Column> mPhysicalEventPosition;
        // Ref<sStringColumn> mSinglePulseTraceIndex;
        Ref<sStringColumn> mMessage;

        std::vector<sTestFailElement> mEventDataVector;

      private:
        std::function<void( sTestFailElement const & )> mOnElementClicked;

        void UITestFailResultTable::DoSelectElement( uint32_t aRow );
    };
} // namespace SE::OtdrEditor